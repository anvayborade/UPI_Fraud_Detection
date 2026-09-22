from __future__ import annotations

import pandas as pd


def aggregate_recipient_intelligence(complaints: pd.DataFrame) -> pd.DataFrame:
    """Aggregate structured complaints into a recipient-level risk table."""
    if complaints.empty:
        return pd.DataFrame(columns=["payee_id", "complaint_count", "complaint_intelligence_score"])
    required = {"payee_id", "confidence", "urgency_language", "remote_access_mentioned"}
    missing = required.difference(complaints.columns)
    if missing:
        raise ValueError(f"Complaint table is missing columns: {sorted(missing)}")
    grouped = complaints.groupby("payee_id", observed=True).agg(
        complaint_count=("payee_id", "size"),
        mean_confidence=("confidence", "mean"),
        urgency_rate=("urgency_language", "mean"),
        remote_rate=("remote_access_mentioned", "mean"),
    )
    grouped["complaint_intelligence_score"] = (
        0.45 * grouped["mean_confidence"]
        + 0.25 * grouped["urgency_rate"]
        + 0.20 * grouped["remote_rate"]
        + 0.10 * (grouped["complaint_count"] / (grouped["complaint_count"] + 3))
    ).clip(0, 1)
    return grouped.reset_index()
