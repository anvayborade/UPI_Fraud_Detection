from __future__ import annotations

from dataclasses import dataclass

from src.features.redis_features import OnlineFeatureStore
from src.schemas.transaction import TelemetryEvent


@dataclass
class TelemetryCollector:
    store: OnlineFeatureStore

    def record(self, event: TelemetryEvent) -> None:
        key = f"telemetry:session:{event.session_id}"
        self.store.client.rpush(key, event.model_dump_json())
        self.store.client.expire(key, 86_400)
        self.store.set_hash(
            f"telemetry:latest:{event.user_id}",
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "device_id": event.device_id,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "recipient_id": event.recipient_id,
            },
            ttl_seconds=86_400,
        )
