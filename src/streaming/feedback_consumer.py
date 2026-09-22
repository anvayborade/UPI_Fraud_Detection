from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import get_settings
from src.streaming.kafka_utils import build_consumer, json_loads


def run(output_path: Path = Path("data/processed/feedback.parquet")) -> None:
    settings = get_settings()
    consumer = build_consumer(settings.kafka_bootstrap_servers, f"{settings.kafka_group_id}-feedback", ["upi.feedback"])
    records: list[dict] = []
    print("Feedback consumer is running. Press Ctrl+C to stop.")
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                print(message.error())
                continue
            records.append(json_loads(message.value()))
            if len(records) >= 50:
                current = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
                pd.concat([current, pd.DataFrame(records)], ignore_index=True).to_parquet(output_path, index=False)
                records.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if records:
            current = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
            pd.concat([current, pd.DataFrame(records)], ignore_index=True).to_parquet(output_path, index=False)
        consumer.close()


if __name__ == "__main__":
    run()
