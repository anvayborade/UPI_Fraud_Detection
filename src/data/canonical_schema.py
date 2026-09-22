from __future__ import annotations

"""Canonical schemas shared by PaySim, BankSim and IBM AML.

The key design rule is that source labels remain separate:

* ``transaction_fraud_label`` means the source dataset labelled the individual
  payment as fraud.
* ``laundering_label`` means the source dataset labelled the payment as part of
  money laundering.

They are never silently treated as the same task. Downstream data products decide
which label is appropriate for each specialist model.
"""

from collections.abc import Iterable

import pandas as pd

CANONICAL_TRANSACTION_COLUMNS = [
    "transaction_id",
    "timestamp",
    "payer_id",
    "payee_id",
    "amount",
    "currency",
    "payment_channel",
    "merchant_category",
    "source_dataset",
    "source_partition",
    "transaction_fraud_label",
    "laundering_label",
]

MODEL_BASE_COLUMNS = [
    "transaction_id",
    "timestamp",
    "payer_id",
    "payee_id",
    "base_type",
    "amount",
    "base_is_fraud",
    "source_dataset",
    "source_partition",
    "source_transaction_fraud_label",
    "source_laundering_label",
    "merchant_category_source",
]


def _clean_text(series: pd.Series, default: str = "UNKNOWN") -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.strip("'")
        .str.strip('"')
        .fillna(default)
        .replace("", default)
    )


def ensure_canonical_schema(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    """Validate and normalise a source-specific canonical transaction table."""
    out = frame.copy()
    missing = [column for column in CANONICAL_TRANSACTION_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"{source_name} canonical table is missing columns: {missing}")

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        bad = int(out["timestamp"].isna().sum())
        raise ValueError(f"{source_name} has {bad} unparseable timestamps")

    out["transaction_id"] = _clean_text(out["transaction_id"])
    out["payer_id"] = _clean_text(out["payer_id"])
    out["payee_id"] = _clean_text(out["payee_id"])
    out["currency"] = _clean_text(out["currency"], "UNKNOWN").str.upper()
    out["payment_channel"] = _clean_text(out["payment_channel"], "UNKNOWN").str.upper()
    out["merchant_category"] = _clean_text(out["merchant_category"], "UNKNOWN").str.upper()
    out["source_dataset"] = _clean_text(out["source_dataset"], source_name).str.lower()
    out["source_partition"] = _clean_text(out["source_partition"], "default").str.lower()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[out["amount"].notna() & (out["amount"] >= 0)].copy()

    for label in ["transaction_fraud_label", "laundering_label"]:
        values = pd.to_numeric(out[label], errors="coerce")
        out[label] = values.round().clip(0, 1).astype("Int64")

    if out["transaction_id"].duplicated().any():
        out["transaction_id"] = [
            f"{value}_{index:08d}" for index, value in enumerate(out["transaction_id"].astype(str))
        ]

    return out[CANONICAL_TRANSACTION_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def canonical_to_model_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate the common source schema into the table expected by UPI augmentation."""
    out = pd.DataFrame(
        {
            "transaction_id": frame["transaction_id"].astype(str),
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
            "payer_id": frame["payer_id"].astype(str),
            "payee_id": frame["payee_id"].astype(str),
            "base_type": frame["payment_channel"].astype(str),
            "amount": frame["amount"].astype(float),
            # Only the transaction-fraud label is used for the fast-path base target.
            # Missing labels remain zero here, while their source label stays available.
            "base_is_fraud": frame["transaction_fraud_label"].fillna(0).astype(int),
            "source_dataset": frame["source_dataset"].astype(str),
            "source_partition": frame["source_partition"].astype(str),
            "source_transaction_fraud_label": frame["transaction_fraud_label"],
            "source_laundering_label": frame["laundering_label"],
            "merchant_category_source": frame["merchant_category"].astype(str),
        }
    )
    return out[MODEL_BASE_COLUMNS]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
