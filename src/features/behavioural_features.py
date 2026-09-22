from __future__ import annotations

from collections import Counter, deque
from datetime import timedelta

import numpy as np
import pandas as pd


def _robust_z(values: pd.Series) -> pd.Series:
    """Leakage-safe robust deviation based on prior observations only."""
    median = values.expanding(min_periods=3).median().shift(1)
    abs_dev = (values - median).abs()
    mad = abs_dev.expanding(min_periods=3).median().shift(1)
    return ((values - median) / (1.4826 * mad + 1e-6)).replace([np.inf, -np.inf], 0).fillna(0)


def _window_features(group: pd.DataFrame) -> pd.DataFrame:
    """Calculate prior-only rolling counts, sums and unique-payee counts.

    A deque implementation is used instead of ``Series.rolling`` on strings so
    that unique counterparties are handled reliably across pandas versions.
    """
    g = group.sort_values("timestamp").copy()
    window_5m: deque[tuple[pd.Timestamp, float]] = deque()
    window_1h: deque[tuple[pd.Timestamp, float, str]] = deque()
    payee_counts: Counter[str] = Counter()

    counts_5m: list[int] = []
    counts_1h: list[int] = []
    amount_sums_1h: list[float] = []
    unique_payees_1h: list[int] = []

    amount_sum_1h = 0.0
    for row in g.itertuples(index=False):
        now = pd.Timestamp(row.timestamp)

        while window_5m and now - window_5m[0][0] > timedelta(minutes=5):
            window_5m.popleft()

        while window_1h and now - window_1h[0][0] > timedelta(hours=1):
            _, old_amount, old_payee = window_1h.popleft()
            amount_sum_1h -= old_amount
            payee_counts[old_payee] -= 1
            if payee_counts[old_payee] <= 0:
                del payee_counts[old_payee]

        # Record features before inserting the current transaction to prevent
        # target leakage from the current event into its own rolling context.
        counts_5m.append(len(window_5m))
        counts_1h.append(len(window_1h))
        amount_sums_1h.append(max(0.0, amount_sum_1h))
        unique_payees_1h.append(len(payee_counts))

        amount = float(row.amount)
        payee = str(row.payee_id)
        window_5m.append((now, amount))
        window_1h.append((now, amount, payee))
        amount_sum_1h += amount
        payee_counts[payee] += 1

    g["txn_count_5m"] = counts_5m
    g["txn_count_1h"] = counts_1h
    g["amount_sum_1h"] = amount_sums_1h
    g["unique_payees_1h"] = unique_payees_1h
    return g


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add payer-relative and leakage-safe behavioural features."""
    out = df.sort_values("timestamp").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    pieces: list[pd.DataFrame] = []
    for _, group in out.groupby("payer_id", sort=False):
        g = _window_features(group)
        g["time_since_previous_sec"] = (
            g["timestamp"].diff().dt.total_seconds().fillna(86_400).clip(lower=0)
        )
        g["amount_robust_z"] = _robust_z(g["amount"]).clip(-20, 20)
        prior_median = g["amount"].expanding(min_periods=3).median().shift(1)
        fallback_median = float(g["amount"].median()) if len(g) else 0.0
        g["payer_median_amount_prior"] = prior_median.fillna(fallback_median)
        g["amount_to_user_median"] = g["amount"] / (g["payer_median_amount_prior"] + 1e-6)
        pieces.append(g)

    result = pd.concat(pieces, axis=0).sort_values("timestamp").reset_index(drop=True)
    for column in ["txn_count_5m", "txn_count_1h", "amount_sum_1h", "unique_payees_1h"]:
        result[column] = result[column].fillna(0)
    return result
