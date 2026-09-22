from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_base_transactions(n_rows: int = 30_000, seed: int = 42) -> pd.DataFrame:
    """Generate a PaySim-like transaction table when the real CSV is unavailable.

    This is intentionally generic. UPI-specific scenarios are injected later by
    ``generate_scenarios.py``.
    """
    rng = np.random.default_rng(seed)
    n_users = max(500, n_rows // 12)
    n_payees = max(600, n_rows // 10)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    payer_ids = np.array([f"U{idx:06d}" for idx in range(n_users)])
    payee_ids = np.array([f"A{idx:06d}" for idx in range(n_payees)])
    payer = rng.choice(payer_ids, n_rows)
    payee = rng.choice(payee_ids, n_rows)

    step_seconds = rng.integers(5, 180, size=n_rows)
    timestamps = [start + timedelta(seconds=int(v)) for v in np.cumsum(step_seconds)]
    txn_type = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], n_rows, p=[0.68, 0.25, 0.07])
    amount = np.exp(rng.normal(np.log(650), 1.05, n_rows)).clip(10, 200_000).round(2)

    old_org = rng.uniform(500, 250_000, n_rows)
    new_org = np.maximum(old_org - amount, 0)
    old_dest = rng.uniform(0, 300_000, n_rows)
    new_dest = old_dest + amount

    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{idx:09d}" for idx in range(n_rows)],
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "payer_id": payer,
            "payee_id": payee,
            "base_type": txn_type,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "base_is_fraud": np.zeros(n_rows, dtype=np.int8),
        }
    )
