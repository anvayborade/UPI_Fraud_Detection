from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    processed = Path("data/processed/upi_transaction_training.parquet")
    scored_path = Path("data/processed/scored_transactions.pkl")
    if not processed.exists():
        raise FileNotFoundError("Run data preparation first.")

    data = pd.read_parquet(processed)
    profile_counts = data.groupby("payee_id", observed=True)["recipient_profile"].nunique()
    archetype_counts = data.groupby("payee_id", observed=True)["recipient_archetype"].nunique()
    print(f"Stable recipient profiles: {(profile_counts <= 1).all()}")
    print(f"Stable recipient archetypes: {(archetype_counts <= 1).all()}")
    print(f"Payees with multiple profiles: {(profile_counts > 1).sum()}")

    chronological_failures = 0
    for _, group in data.sort_values("timestamp").groupby("payee_id", observed=True):
        values = group["recipient_compromised"].astype(int).to_list()
        if values != sorted(values):
            chronological_failures += 1
    print(f"Non-chronological compromise transitions: {chronological_failures}")

    if scored_path.exists():
        scored = pd.read_pickle(scored_path)
        required = {
            "sequence_history_size",
            "tgn_history_size",
            "graph_score_available",
            "sequence_score_available",
            "cache_coverage",
            "expert_conflict_score",
        }
        print(f"Transaction-time/availability columns present: {required.issubset(scored.columns)}")

    metadata_path = Path("models/fusion/metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        thresholds = metadata.get("fraud_type_thresholds", {})
        print(f"Calibrated fraud-type thresholds: {len(thresholds)}")

    manifest_path = Path("data/processed/demo_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payees = [item["payee_id"] for item in manifest.values()]
        print(f"Demo recipients are unique: {len(payees) == len(set(payees))}")


if __name__ == "__main__":
    main()
