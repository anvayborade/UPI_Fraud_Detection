from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import redis

from datetime import date, datetime
import math

import numpy as np
import pandas as pd

def _to_json_safe(value):
    """Convert pandas, NumPy and datetime objects into JSON-safe values."""

    # Recursively handle dictionaries.
    if isinstance(value, dict):
        return {
            str(key): _to_json_safe(item)
            for key, item in value.items()
        }

    # Recursively handle sequences.
    if isinstance(value, (list, tuple, set)):
        return [
            _to_json_safe(item)
            for item in value
        ]

    if value is None:
        return None

    # Pandas missing values.
    if value is pd.NA or value is pd.NaT:
        return None

    # Pandas Timestamp.
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    # NumPy datetime.
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return pd.Timestamp(value).isoformat()

    # Native Python dates and datetimes.
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # Convert NumPy numbers and booleans to normal Python values.
    if isinstance(value, np.generic):
        value = value.item()

    # JSON should not contain NaN or infinity.
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


@dataclass
class OnlineFeatureStore:
    url: str

    def __post_init__(self) -> None:
        self.client = redis.Redis.from_url(self.url, decode_responses=True)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def get_hash(self, key: str) -> dict[str, Any]:
        raw = self.client.hgetall(key)
        result: dict[str, Any] = {}
        for field, value in raw.items():
            try:
                result[field] = json.loads(value)
            except json.JSONDecodeError:
                result[field] = value
        return result

    def set_hash(self, key: str, values: dict[str, Any], ttl_seconds: int | None = None) -> None:
        payload = {
            str(key): json.dumps(
                _to_json_safe(value),
                allow_nan=False,
            )
            for key, value in values.items()
        }
        if payload:
            self.client.hset(key, mapping=payload)
        if ttl_seconds:
            self.client.expire(key, ttl_seconds)

    def add_transaction_window(self, entity_id: str, timestamp: float, transaction_id: str, amount: float, payee_id: str) -> None:
        key = f"window:payer:{entity_id}"
        member = json.dumps({"id": transaction_id, "amount": amount, "payee": payee_id})
        pipe = self.client.pipeline()
        pipe.zadd(key, {member: timestamp})
        pipe.zremrangebyscore(key, 0, timestamp - 86_400)
        pipe.expire(key, 172_800)
        pipe.execute()

    def compute_window_features(self, entity_id: str, now: float | None = None) -> dict[str, float]:
        now = now or time.time()
        key = f"window:payer:{entity_id}"
        rows_5m = self.client.zrangebyscore(key, now - 300, now)
        rows_1h = self.client.zrangebyscore(key, now - 3600, now)
        parsed_1h = [json.loads(v) for v in rows_1h]
        return {
            "txn_count_5m": float(len(rows_5m)),
            "txn_count_1h": float(len(rows_1h)),
            "amount_sum_1h": float(sum(float(v["amount"]) for v in parsed_1h)),
            "unique_payees_1h": float(len({v["payee"] for v in parsed_1h})),
        }
