from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.features.redis_features import OnlineFeatureStore
from src.settings import get_settings

TTL_SECONDS = 2_592_000


def _float(row: pd.Series | dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _text(row: pd.Series | dict[str, Any], name: str, default: str = "unknown") -> str:
    value = row.get(name, default)
    if value is None or pd.isna(value):
        return default
    return str(value)


def _transaction_snapshot(row: pd.Series) -> dict[str, Any]:
    return {
        "snapshot_transaction_id": _text(row, "transaction_id", ""),
        "snapshot_timestamp": _text(row, "timestamp", ""),
        "snapshot_payer_id": _text(row, "payer_id", ""),
        "snapshot_payee_id": _text(row, "payee_id", ""),
        "txn_count_5m": _float(row, "txn_count_5m"),
        "txn_count_1h": _float(row, "txn_count_1h"),
        "amount_sum_1h": _float(row, "amount_sum_1h"),
        "unique_payees_1h": _float(row, "unique_payees_1h"),
        "time_since_previous_sec": _float(row, "time_since_previous_sec", 86_400.0),
        "payer_median_amount_prior": _float(row, "payer_median_amount_prior", _float(row, "amount", 1.0)),
        "location_continuity": _float(row, "location_continuity", 0.70),
        "journey_plausibility": _float(row, "journey_plausibility", 0.70),
        "sequence_account_takeover_score": _float(row, "sequence_account_takeover_score"),
        "sequence_social_engineering_score": _float(row, "sequence_social_engineering_score"),
        "sequence_history_size": _float(row, "sequence_history_size"),
        "graph_mule_score": _float(row, "graph_mule_score"),
        "graph_merchant_score": _float(row, "graph_merchant_score"),
        "graph_anomaly_score": _float(row, "graph_anomaly_score"),
        "graph_score_available": _float(row, "graph_score_available"),
        "complaint_intelligence_score": _float(row, "complaint_intelligence_score"),
        "complaint_history_count": _float(row, "complaint_history_count"),
        "recipient_archetype": _text(row, "recipient_archetype", "unknown"),
        "recipient_profile": _text(row, "recipient_profile", "unknown"),
        "recipient_account_age_days": _float(row, "recipient_account_age_days", 365.0),
        "fan_in_1h": _float(row, "fan_in_1h"),
        "fan_out_1h": _float(row, "fan_out_1h"),
        "rapid_outflow_ratio": _float(row, "rapid_outflow_ratio"),
        "median_holding_minutes": _float(row, "median_holding_minutes", 60.0),
        "merchant_category_consistency": _float(row, "merchant_category_consistency", 0.55),
        "recipient_history_available": _float(row, "recipient_history_available"),
        "payer_history_available": _float(row, "payer_history_available"),
    }


def load_cache() -> None:
    settings = get_settings()
    store = OnlineFeatureStore(settings.redis_url)

    recipient_path = Path("models/graph/recipient_scores.parquet")
    if recipient_path.exists():
        recipients = pd.read_parquet(recipient_path)
        for row in recipients.to_dict(orient="records"):
            payee_id = str(row.pop("payee_id"))
            store.set_hash(f"risk:{payee_id}", row, ttl_seconds=TTL_SECONDS)
            store.set_hash(f"features:recipient:{payee_id}", row, ttl_seconds=TTL_SECONDS)
        print(f"Loaded {len(recipients):,} train-cutoff recipient risk records into Redis")

    scored_path = Path("data/processed/scored_transactions.pkl")
    if not scored_path.exists():
        return

    scored = pd.read_pickle(scored_path).copy()
    scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True, errors="coerce")
    scored = scored.sort_values("timestamp", kind="stable")

    for _, row in scored.iterrows():
        transaction_id = str(row["transaction_id"])
        snapshot = _transaction_snapshot(row)
        store.set_hash(f"snapshot:txn:{transaction_id}", snapshot, ttl_seconds=TTL_SECONDS)
        sequence_payload = {
            "account_takeover_score": snapshot["sequence_account_takeover_score"],
            "social_engineering_score": snapshot["sequence_social_engineering_score"],
            "sequence_history_size": snapshot["sequence_history_size"],
        }
        store.set_hash(f"sequence:txn:{transaction_id}", sequence_payload, ttl_seconds=TTL_SECONDS)
        store.set_hash(f"sequence:session:S-{transaction_id}", sequence_payload, ttl_seconds=TTL_SECONDS)
    print(f"Loaded {len(scored):,} transaction/session replay snapshots into Redis")

    # Live fallback state is limited to the training period.  Validation/test rows
    # are retained only in exact replay snapshots and cannot contaminate live cache.
    train_path = Path("data/processed/train.parquet")
    if train_path.exists():
        train_ids = set(pd.read_parquet(train_path)["transaction_id"].astype(str))
        live_source = scored[scored["transaction_id"].astype(str).isin(train_ids)].copy()
    else:
        live_source = scored.copy()

    latest_payer = live_source.groupby("payer_id", observed=True, sort=False).tail(1)
    for _, row in latest_payer.iterrows():
        payer_id = str(row["payer_id"])
        context = {
            "account_takeover_score": _float(row, "sequence_account_takeover_score"),
            "social_engineering_score": _float(row, "sequence_social_engineering_score"),
            "sequence_history_size": _float(row, "sequence_history_size"),
            "journey_plausibility": _float(row, "journey_plausibility", 0.70),
            "location_continuity": _float(row, "location_continuity", 0.70),
            "payer_median_amount_prior": _float(row, "payer_median_amount_prior", _float(row, "amount", 1.0)),
            "txn_count_5m": _float(row, "txn_count_5m"),
            "txn_count_1h": _float(row, "txn_count_1h"),
            "amount_sum_1h": _float(row, "amount_sum_1h"),
            "unique_payees_1h": _float(row, "unique_payees_1h"),
            "time_since_previous_sec": _float(row, "time_since_previous_sec", 86_400.0),
            "payer_history_available": _float(row, "payer_history_available"),
        }
        store.set_hash(f"sequence:{payer_id}", context, ttl_seconds=TTL_SECONDS)
        store.set_hash(f"features:payer:{payer_id}", context, ttl_seconds=TTL_SECONDS)
    print(f"Loaded {len(latest_payer):,} training-period payer context records into Redis")


if __name__ == "__main__":
    load_cache()
