from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.settings import get_settings
from src.streaming.kafka_utils import build_producer, json_dumps


def publish_transactions(input_path: Path, limit: int | None = None, sleep_seconds: float = 0.0) -> None:
    settings = get_settings()
    producer = build_producer(settings.kafka_bootstrap_servers)
    frame = pd.read_parquet(input_path)
    if limit:
        frame = frame.head(limit)
    for row in frame.to_dict(orient="records"):
        producer.produce("upi.transactions.raw", key=str(row["transaction_id"]), value=json_dumps(row))
        producer.poll(0)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    producer.flush()
    print(f"Published {len(frame):,} transactions to upi.transactions.raw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()
    publish_transactions(args.input, args.limit, args.sleep)


if __name__ == "__main__":
    main()
