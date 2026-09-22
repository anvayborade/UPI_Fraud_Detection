from __future__ import annotations

import pandas as pd

from src.data.canonical_schema import ensure_canonical_schema
from src.data.split_data import source_stratified_temporal_split


def test_canonical_schema_keeps_labels_separate() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": ["x1"],
            "timestamp": ["2026-01-01T00:00:00Z"],
            "payer_id": ["p1"],
            "payee_id": ["q1"],
            "amount": [100.0],
            "currency": ["INR"],
            "payment_channel": ["P2P"],
            "merchant_category": ["PERSONAL"],
            "source_dataset": ["demo"],
            "source_partition": ["default"],
            "transaction_fraud_label": [0],
            "laundering_label": [1],
        }
    )
    out = ensure_canonical_schema(frame, source_name="demo")
    assert int(out.loc[0, "transaction_fraud_label"]) == 0
    assert int(out.loc[0, "laundering_label"]) == 1


def test_source_stratified_temporal_split_keeps_both_sources() -> None:
    rows = []
    for source in ["paysim", "banksim"]:
        for index in range(20):
            rows.append(
                {
                    "source_dataset": source,
                    "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=index),
                    "value": index,
                }
            )
    frame = pd.DataFrame(rows)
    train, val, test = source_stratified_temporal_split(frame)
    for part in [train, val, test]:
        assert set(part["source_dataset"]) == {"paysim", "banksim"}
