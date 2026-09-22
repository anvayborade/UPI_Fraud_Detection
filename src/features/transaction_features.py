from __future__ import annotations

import numpy as np
import pandas as pd


ARCHETYPE_CODES = {
    "personal": 0,
    "merchant": 1,
    "transport_provider": 2,
    "gig_worker": 3,
    "aggregator": 4,
    "new_business": 5,
    "money_mule": 6,
    "unknown": 7,
}


def add_basic_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["hour"] = ts.dt.hour.astype(int)
    out["day_of_week"] = ts.dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    out["recipient_archetype_code"] = (
        out.get("recipient_archetype", "unknown").map(ARCHETYPE_CODES).fillna(ARCHETYPE_CODES["unknown"]).astype(int)
    )
    out["device_session_risk"] = np.clip(
        0.25 * (1 - out.get("device_known", 1).astype(float))
        + 0.20 * out.get("recent_pin_reset", 0).astype(float)
        + 0.20 * out.get("app_reregistered", 0).astype(float)
        + 0.15 * out.get("remote_access_indicator", 0).astype(float)
        + 0.10 * out.get("overlay_indicator", 0).astype(float)
        + 0.10 * (1 - out.get("play_integrity_ok", 1).astype(float)),
        0,
        1,
    )
    return out
