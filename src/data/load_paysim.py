from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.canonical_schema import canonical_to_model_base, ensure_canonical_schema
from src.simulation.generate_base import generate_base_transactions


def load_paysim_canonical(path: Path, n_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, nrows=n_rows)
    required = {"step", "type", "amount", "nameOrig", "nameDest", "isFraud"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"PaySim file is missing required columns: {sorted(missing)}")

    start = pd.Timestamp("2026-01-01T00:00:00Z")
    offsets = pd.to_timedelta(df["step"].astype(int), unit="h") + pd.to_timedelta(
        np.arange(len(df)) % 3600, unit="s"
    )
    out = pd.DataFrame(
        {
            "transaction_id": [f"PS_TXN_{idx:010d}" for idx in range(len(df))],
            "timestamp": start + offsets,
            "payer_id": "PS_" + df["nameOrig"].astype(str),
            "payee_id": "PS_" + df["nameDest"].astype(str),
            "amount": df["amount"].astype(float),
            "currency": "SYNTHETIC",
            "payment_channel": df["type"].astype(str),
            "merchant_category": df["type"].astype(str),
            "source_dataset": "paysim",
            "source_partition": "default",
            "transaction_fraud_label": df["isFraud"].astype(int),
            "laundering_label": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        }
    )
    return ensure_canonical_schema(out, source_name="paysim")


def load_paysim(path: Path, n_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Compatibility loader used by the original one-source pipeline."""
    if not path.exists():
        return generate_base_transactions(n_rows=n_rows or 30_000, seed=seed)
    canonical = load_paysim_canonical(path, n_rows=n_rows)
    out = canonical_to_model_base(canonical)
    raw = pd.read_csv(path, nrows=n_rows)
    out["oldbalanceOrg"] = pd.to_numeric(raw.get("oldbalanceOrg", 0.0), errors="coerce").fillna(0).to_numpy()
    out["newbalanceOrig"] = pd.to_numeric(raw.get("newbalanceOrig", 0.0), errors="coerce").fillna(0).to_numpy()
    out["oldbalanceDest"] = pd.to_numeric(raw.get("oldbalanceDest", 0.0), errors="coerce").fillna(0).to_numpy()
    out["newbalanceDest"] = pd.to_numeric(raw.get("newbalanceDest", 0.0), errors="coerce").fillna(0).to_numpy()
    return out.sort_values("timestamp").reset_index(drop=True)
