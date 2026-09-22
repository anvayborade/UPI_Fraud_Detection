from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.canonical_schema import ensure_canonical_schema


def _find_banksim_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.rglob("*.csv"), key=lambda item: item.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No BankSim CSV found under {path}")
    required = {"step", "customer", "merchant", "amount", "fraud"}
    for candidate in candidates:
        try:
            header = set(pd.read_csv(candidate, nrows=0).columns)
        except Exception:
            continue
        if required.issubset(header):
            return candidate
    raise ValueError(
        f"CSV files were found under {path}, but none contained BankSim columns {sorted(required)}"
    )


def load_banksim_canonical(path: Path, n_rows: int | None = None) -> pd.DataFrame:
    csv_path = _find_banksim_csv(path)
    raw = pd.read_csv(csv_path, nrows=n_rows)
    raw.columns = [str(column).strip() for column in raw.columns]
    required = {"step", "customer", "merchant", "amount", "fraud"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"BankSim file {csv_path} is missing columns: {sorted(missing)}")

    clean = lambda series: series.astype("string").str.strip().str.strip("'").str.strip('"')
    customer = clean(raw["customer"])
    merchant = clean(raw["merchant"])
    category = clean(raw.get("category", pd.Series("UNKNOWN", index=raw.index))).str.upper()
    start = pd.Timestamp("2026-02-01T00:00:00Z")
    offsets = pd.to_timedelta(pd.to_numeric(raw["step"], errors="coerce").fillna(0), unit="h")
    offsets += pd.to_timedelta(np.arange(len(raw)) % 3600, unit="s")

    out = pd.DataFrame(
        {
            "transaction_id": [f"BS_TXN_{idx:010d}" for idx in range(len(raw))],
            "timestamp": start + offsets,
            "payer_id": "BS_CUSTOMER_" + customer,
            "payee_id": "BS_MERCHANT_" + merchant,
            "amount": pd.to_numeric(raw["amount"], errors="coerce"),
            "currency": "EUR",
            "payment_channel": "P2M",
            "merchant_category": category,
            "source_dataset": "banksim",
            "source_partition": "default",
            "transaction_fraud_label": pd.to_numeric(raw["fraud"], errors="coerce"),
            "laundering_label": pd.Series(pd.NA, index=raw.index, dtype="Int64"),
        }
    )
    return ensure_canonical_schema(out, source_name="banksim")
