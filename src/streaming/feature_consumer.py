from __future__ import annotations

import time

from src.features.redis_features import OnlineFeatureStore
from src.settings import get_settings
from src.streaming.kafka_utils import build_consumer, json_loads


def run() -> None:
    settings = get_settings()
    store = OnlineFeatureStore(settings.redis_url)
    consumer = build_consumer(settings.kafka_bootstrap_servers, f"{settings.kafka_group_id}-features", ["upi.transactions.raw"])
    print("Feature consumer is running. Press Ctrl+C to stop.")
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                print(message.error())
                continue
            event = json_loads(message.value())
            timestamp = time.time()
            store.add_transaction_window(
                entity_id=str(event["payer_id"]),
                timestamp=timestamp,
                transaction_id=str(event["transaction_id"]),
                amount=float(event["amount"]),
                payee_id=str(event["payee_id"]),
            )
            window = store.compute_window_features(str(event["payer_id"]), timestamp)
            store.set_hash(f"features:payer:{event['payer_id']}", window, ttl_seconds=172_800)
            recipient = {
                "recipient_complaint_score": float(event.get("recipient_complaint_score", 0)),
                "rapid_outflow_ratio": float(event.get("rapid_outflow_ratio", 0)),
                "recipient_mule_score_cached": float(event.get("recipient_mule_score_cached", 0)),
                "recipient_archetype": str(event.get("recipient_archetype", "unknown")),
            }
            store.set_hash(f"features:recipient:{event['payee_id']}", recipient, ttl_seconds=604_800)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    run()
