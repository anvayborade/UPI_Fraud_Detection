from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.load_paysim import load_paysim
from src.data.split_data import save_splits
from src.features.behavioural_features import add_behavioural_features
from src.features.context_features import add_context_features
from src.features.graph_features import add_peer_normalised_graph_features
from src.features.transaction_features import add_basic_transaction_features
from src.settings import get_settings
from src.simulation.generate_scenarios import inject_upi_scenarios
from src.simulation.generate_sequences import generate_event_sequences


def prepare_dataset(rows: int | None = None, seed: int | None = None) -> pd.DataFrame:
    settings = get_settings()
    seed = seed if seed is not None else settings.random_seed
    rows = rows if rows is not None else settings.demo_rows

    base = load_paysim(settings.paysim_csv, n_rows=rows, seed=seed)
    data = inject_upi_scenarios(base, seed=seed)
    data = add_basic_transaction_features(data)
    data = add_behavioural_features(data)
    data = add_context_features(data)
    data = add_peer_normalised_graph_features(data)

    settings.processed_data.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(settings.processed_data, index=False)
    save_splits(data, settings.processed_data.parent)
    generate_event_sequences(data, Path("data/sequences/events.parquet"), seed=seed)

    edge_cases = data[data["scenario"].isin(
        [
            "auto_long_distance_legitimate",
            "hospital_emergency_legitimate",
            "travel_hotel_legitimate",
            "gig_worker_payment_legitimate",
            "new_merchant_legitimate",
            "legitimate_split_payment",
            "fake_refund_qr",
            "collect_request_scam",
            "account_takeover",
            "mule_directed",
            "remote_access_scam",
        ]
    )]
    edge_cases.groupby("scenario", group_keys=False).head(100).to_parquet(
        settings.processed_data.parent / "edge_cases.parquet", index=False
    )
    print(f"Prepared {len(data):,} rows at {settings.processed_data}")
    print(data["scenario"].value_counts().to_string())
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the UPI-enhanced fraud dataset")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    prepare_dataset(args.rows, args.seed)


if __name__ == "__main__":
    main()
