from __future__ import annotations

import numpy as np
import pandas as pd


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a legitimate-context score without using complaint or graph outcomes.

    Complaint intelligence and money-flow behaviour remain independent experts in
    the final fusion model instead of being copied into the context classifier.
    """
    out = df.copy()
    proximity = out.get("payer_payee_proximity_km", pd.Series(20.0, index=out.index)).astype(float)
    continuity = out.get("location_continuity", pd.Series(0.5, index=out.index)).astype(float)
    verified = out.get("qr_verified", pd.Series(0, index=out.index)).astype(float)
    stable_device = out.get("device_known", pd.Series(1, index=out.index)).astype(float)
    no_remote = 1 - out.get("remote_access_indicator", pd.Series(0, index=out.index)).astype(float)
    category_consistency = out.get(
        "merchant_category_consistency", pd.Series(0.5, index=out.index)
    ).astype(float)
    account_age = out.get(
        "recipient_account_age_days", pd.Series(180.0, index=out.index)
    ).astype(float)
    integrity = out.get("play_integrity_ok", pd.Series(1, index=out.index)).astype(float)

    out["proximity_consistency"] = np.exp(-proximity / 15.0)
    age_support = np.clip(np.log1p(account_age) / np.log1p(365.0), 0, 1)
    out["context_support_score"] = np.clip(
        0.22 * continuity
        + 0.15 * out["proximity_consistency"]
        + 0.17 * stable_device
        + 0.12 * no_remote
        + 0.10 * verified
        + 0.12 * category_consistency
        + 0.08 * age_support
        + 0.04 * integrity,
        0,
        1,
    )
    out["novel_transaction_flag"] = (
        (out.get("is_new_payee", 0).astype(int) == 1)
        | (out.get("amount_robust_z", 0).abs() >= 2.5)
        | (out.get("distance_from_usual_km", 0).astype(float) >= 10)
    ).astype(int)
    return out
