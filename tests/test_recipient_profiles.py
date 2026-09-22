from __future__ import annotations

import pandas as pd

from src.simulation.recipient_profiles import assign_recipient_aware_scenarios


def test_recipient_profile_is_stable_and_compromise_is_chronological() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "transaction_id": f"T{index}",
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=index),
                "payer_id": f"P{index}",
                "payee_id": "RECIPIENT_A",
                "base_type": "TRANSFER",
                "amount": 100.0 + index,
                "base_is_fraud": int(index == 18),
                "source_dataset": "paysim",
                "source_partition": "default",
                "source_transaction_fraud_label": int(index == 18),
                "source_laundering_label": pd.NA,
                "merchant_category_source": "UNKNOWN",
            }
        )
    frame = pd.DataFrame(rows)
    result = assign_recipient_aware_scenarios(
        frame,
        seed=42,
        synthetic_fraud_rate=0.04,
    )
    assert result["recipient_profile"].nunique() == 1
    assert result["recipient_archetype"].nunique() == 1
    compromised = result.sort_values("timestamp")["recipient_compromised"].to_list()
    assert compromised == sorted(compromised)
