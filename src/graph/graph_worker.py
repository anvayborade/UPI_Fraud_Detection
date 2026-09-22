from __future__ import annotations

import collections
import time

from src.features.redis_features import OnlineFeatureStore
from src.settings import get_settings
from src.streaming.kafka_utils import build_consumer, json_loads


def run(refresh_seconds: int = 60) -> None:
    """Near-real-time recipient graph worker.

    The trained HGT/TGN scores are loaded into Redis by the training pipeline. This
    worker keeps the short-window graph signals fresh between model refreshes.
    """
    settings = get_settings()
    store = OnlineFeatureStore(settings.redis_url)
    consumer = build_consumer(settings.kafka_bootstrap_servers, f"{settings.kafka_group_id}-graph", ["upi.transactions.raw"])
    incoming: dict[str, collections.deque] = collections.defaultdict(collections.deque)
    outgoing: dict[str, collections.deque] = collections.defaultdict(collections.deque)
    last_refresh = time.time()
    print("Graph worker is running. Press Ctrl+C to stop.")
    try:
        while True:
            message = consumer.poll(1.0)
            now = time.time()
            if message is not None and not message.error():
                event = json_loads(message.value())
                payer, payee = str(event["payer_id"]), str(event["payee_id"])
                amount = float(event["amount"])
                incoming[payee].append((now, amount, payer))
                outgoing[payer].append((now, amount, payee))
            if now - last_refresh >= refresh_seconds:
                cutoff = now - 3600
                entities = set(incoming) | set(outgoing)
                for entity in entities:
                    while incoming[entity] and incoming[entity][0][0] < cutoff:
                        incoming[entity].popleft()
                    while outgoing[entity] and outgoing[entity][0][0] < cutoff:
                        outgoing[entity].popleft()
                    in_amount = sum(value for _, value, _ in incoming[entity])
                    out_amount = sum(value for _, value, _ in outgoing[entity])
                    fan_in = len({counterparty for _, _, counterparty in incoming[entity]})
                    fan_out = len({counterparty for _, _, counterparty in outgoing[entity]})
                    flow_ratio = out_amount / (in_amount + 1e-6)
                    previous = store.get_hash(f"risk:{entity}")
                    learned = float(previous.get("learned_mule_score", 0.0))
                    short_window = min(1.0, 0.25 * fan_in / 10 + 0.30 * fan_out / 8 + 0.45 * min(flow_ratio, 1.5) / 1.5)
                    store.set_hash(
                        f"risk:{entity}",
                        {
                            **previous,
                            "fan_in_1h_live": fan_in,
                            "fan_out_1h_live": fan_out,
                            "flow_through_ratio_live": flow_ratio,
                            "short_window_graph_risk": short_window,
                            "recipient_mule_score": max(learned, short_window),
                            "updated_at": now,
                        },
                        ttl_seconds=604_800,
                    )
                last_refresh = now
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Refresh short-window graph risk in Redis")
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    run(refresh_seconds=max(1, args.interval_seconds))


if __name__ == "__main__":
    main()
