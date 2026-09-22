from __future__ import annotations

import json

from src.features.redis_features import OnlineFeatureStore


def aggregate_session_features(store: OnlineFeatureStore, session_id: str) -> dict[str, float]:
    rows = store.client.lrange(f"telemetry:session:{session_id}", 0, -1)
    if not rows:
        return {
            "event_count": 0.0,
            "payment_retry_count": 0.0,
            "external_message_opened": 0.0,
            "remote_guidance_signal": 0.0,
            "recipient_review_seconds": 0.0,
        }
    events = [json.loads(row) for row in rows]
    event_types = [event["event_type"] for event in events]
    return {
        "event_count": float(len(events)),
        "payment_retry_count": float(sum(v in {"PAYMENT_FAILED", "PAYMENT_RETRIED"} for v in event_types)),
        "external_message_opened": float("EXTERNAL_MESSAGE_OPEN" in event_types),
        "remote_guidance_signal": float("REMOTE_GUIDANCE_SIGNAL" in event_types),
        "recipient_review_seconds": float(
            next((event.get("metadata", {}).get("review_seconds", 0) for event in reversed(events) if event["event_type"] == "RECIPIENT_NAME_VIEWED"), 0)
        ),
    }
