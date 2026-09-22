from __future__ import annotations

import json
from typing import Any

from confluent_kafka import Consumer, Producer


def json_dumps(value: Any) -> bytes:
    return json.dumps(value, default=str).encode("utf-8")


def json_loads(value: bytes | None) -> dict[str, Any]:
    if value is None:
        return {}
    return json.loads(value.decode("utf-8"))


def build_producer(bootstrap_servers: str) -> Producer:
    return Producer({"bootstrap.servers": bootstrap_servers, "client.id": "upi-fraud-producer"})


def build_consumer(bootstrap_servers: str, group_id: str, topics: list[str]) -> Consumer:
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe(topics)
    return consumer
