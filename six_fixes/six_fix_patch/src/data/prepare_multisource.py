from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.canonical_schema import canonical_to_model_base
from src.data.load_banksim import load_banksim_canonical
from src.data.load_ibm_aml import load_ibm_aml_canonical
from src.data.load_paysim import load_paysim_canonical
from src.data.split_data import save_source_stratified_splits
from src.features.behavioural_features import add_behavioural_features
from src.features.context_features import add_context_features
from src.features.graph_features import add_peer_normalised_graph_features, add_prior_graph_flow_features
from src.features.transaction_features import add_basic_transaction_features
from src.settings import get_settings
from src.simulation.generate_scenarios import inject_upi_scenarios
from src.simulation.recipient_profiles import assign_recipient_aware_scenarios, recipient_consistency_report
from src.simulation.generate_sequences import generate_event_sequences

LEGITIMATE_SCENARIOS = [
    "familiar_legitimate",
    "auto_long_distance_legitimate",
    "hospital_emergency_legitimate",
    "travel_hotel_legitimate",
    "gig_worker_payment_legitimate",
    "new_merchant_legitimate",
    "legitimate_split_payment",
]
FRAUD_SCENARIOS = [
    "fake_refund_qr",
    "collect_request_scam",
    "account_takeover",
    "mule_directed",
    "remote_access_scam",
]


def _prepare_main_upi_product(
    paysim: pd.DataFrame,
    banksim: pd.DataFrame,
    *,
    seed: int,
    synthetic_fraud_rate: float,
) -> pd.DataFrame:
    base = pd.concat(
        [canonical_to_model_base(paysim), canonical_to_model_base(banksim)],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    base = assign_recipient_aware_scenarios(
        base, seed=seed, synthetic_fraud_rate=synthetic_fraud_rate
    )
    data = inject_upi_scenarios(base, seed=seed)

    data = add_basic_transaction_features(data)
    data = add_behavioural_features(data)
    data = add_prior_graph_flow_features(data)
    data = add_context_features(data)
    data = add_peer_normalised_graph_features(data)
    return data.sort_values("timestamp").reset_index(drop=True)


def _prepare_ibm_graph_product(hi: pd.DataFrame, li: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create leakage-safe IBM money-flow edges for HGT/TGN training."""
    rng = np.random.default_rng(seed)
    canonical = pd.concat([hi, li], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    data = pd.DataFrame(
        {
            "transaction_id": canonical["transaction_id"],
            "timestamp": canonical["timestamp"],
            "payer_id": canonical["payer_id"],
            "payee_id": canonical["payee_id"],
            "amount": canonical["amount"].astype(float),
            "payment_mode": "P2P",
            "collect_request": 0,
            "qr_verified": 0,
            "source_dataset": "ibm_aml",
            "source_partition": canonical["source_partition"],
            "source_laundering_label": canonical["laundering_label"].fillna(0).astype(int),
            "is_fraud": canonical["laundering_label"].fillna(0).astype(int),
            "is_mule_directed": canonical["laundering_label"].fillna(0).astype(int),
            "scenario": np.where(
                canonical["laundering_label"].fillna(0).astype(int) == 1,
                "ibm_laundering",
                "ibm_legitimate_flow",
            ),
        }
    )

    # These features are computed chronologically from prior events only.
    data["recipient_archetype"] = "personal"
    data["recipient_complaint_score"] = rng.beta(0.5, 25.0, len(data))
    data = add_prior_graph_flow_features(data)

    data["merchant_id"] = None
    data["merchant_category"] = canonical["merchant_category"].astype(str)
    data["device_id"] = "IBM_DEVICE_" + pd.factorize(data["payer_id"])[0].astype(str)
    data["vpa_id"] = "ibm_vpa_" + pd.factorize(data["payee_id"])[0].astype(str) + "@upi"
    data["ip_asn"] = "IBM_AS" + (pd.factorize(data["payer_id"])[0] % 500).astype(str)
    data["qr_id"] = "IBM_QR_" + pd.factorize(data["payee_id"])[0].astype(str)
    data["latitude"] = 19.076
    data["longitude"] = 72.878
    data["location_continuity"] = 0.50
    data["is_rapid_pass_through"] = (data["rapid_outflow_ratio"] >= 0.70).astype(int)
    return data.sort_values("timestamp").reset_index(drop=True)

def prepare_multisource(
    *,
    paysim_rows: int | None = None,
    banksim_rows: int | None = None,
    ibm_negative_ratio: int | None = None,
    ibm_max_negatives_per_file: int | None = None,
    synthetic_fraud_rate: float = 0.04,
    seed: int | None = None,
) -> dict[str, Path]:
    settings = get_settings()
    seed = settings.random_seed if seed is None else seed
    paysim_rows = settings.paysim_rows if paysim_rows is None else paysim_rows
    banksim_rows = settings.banksim_rows if banksim_rows is None else banksim_rows
    ibm_negative_ratio = settings.ibm_negative_ratio if ibm_negative_ratio is None else ibm_negative_ratio
    ibm_max_negatives_per_file = (
        settings.ibm_max_negatives_per_file
        if ibm_max_negatives_per_file is None
        else ibm_max_negatives_per_file
    )

    canonical_dir = Path("data/processed/canonical")
    canonical_dir.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    paysim = load_paysim_canonical(settings.paysim_csv, n_rows=paysim_rows)
    banksim = load_banksim_canonical(settings.banksim_csv, n_rows=banksim_rows)
    ibm_hi = load_ibm_aml_canonical(
        settings.ibm_hi_csv,
        partition="hi_small",
        negative_ratio=ibm_negative_ratio,
        max_negative_rows=ibm_max_negatives_per_file,
        seed=seed,
    )
    ibm_li = load_ibm_aml_canonical(
        settings.ibm_li_csv,
        partition="li_small",
        negative_ratio=ibm_negative_ratio,
        max_negative_rows=ibm_max_negatives_per_file,
        seed=seed + 1,
    )

    canonical_paths = {
        "paysim_canonical": canonical_dir / "paysim_transactions.parquet",
        "banksim_canonical": canonical_dir / "banksim_transactions.parquet",
        "ibm_hi_canonical": canonical_dir / "ibm_hi_small_transactions.parquet",
        "ibm_li_canonical": canonical_dir / "ibm_li_small_transactions.parquet",
    }
    for frame, path in [
        (paysim, canonical_paths["paysim_canonical"]),
        (banksim, canonical_paths["banksim_canonical"]),
        (ibm_hi, canonical_paths["ibm_hi_canonical"]),
        (ibm_li, canonical_paths["ibm_li_canonical"]),
    ]:
        frame.to_parquet(path, index=False)

    main = _prepare_main_upi_product(
        paysim,
        banksim,
        seed=seed,
        synthetic_fraud_rate=synthetic_fraud_rate,
    )
    consistency = recipient_consistency_report(main)
    consistency_path = settings.processed_data.parent / "recipient_consistency.parquet"
    consistency.to_parquet(consistency_path, index=False)
    inconsistent = consistency[
        (consistency["recipient_profile"] > 1)
        | (consistency["recipient_archetype"] > 1)
    ]
    if not inconsistent.empty:
        raise RuntimeError(
            f"Stable-recipient validation failed for {len(inconsistent)} payees. "
            f"Inspect {consistency_path}."
        )
    main.to_parquet(settings.processed_data, index=False)
    save_source_stratified_splits(main, settings.processed_data.parent)
    generate_event_sequences(main, Path("data/sequences/events.parquet"), seed=seed)

    edge_scenarios = [
        "auto_long_distance_legitimate", "hospital_emergency_legitimate",
        "travel_hotel_legitimate", "gig_worker_payment_legitimate",
        "new_merchant_legitimate", "legitimate_split_payment",
        "fake_refund_qr", "collect_request_scam", "account_takeover",
        "mule_directed", "remote_access_scam",
    ]
    main[main["scenario"].isin(edge_scenarios)].groupby("scenario", group_keys=False).head(150).to_parquet(
        settings.processed_data.parent / "edge_cases.parquet", index=False
    )

    archetype_product = main[
        main["source_dataset"].eq("banksim")
        | main["recipient_archetype"].isin(["merchant", "transport_provider", "money_mule"])
    ].copy()
    archetype_product.to_parquet(settings.recipient_archetype_data, index=False)

    aml_graph = _prepare_ibm_graph_product(ibm_hi, ibm_li, seed=seed)
    aml_graph.to_parquet(settings.aml_graph_data, index=False)

    manifest = {
        "paysim_rows": int(len(paysim)),
        "banksim_rows": int(len(banksim)),
        "ibm_hi_sample_rows": int(len(ibm_hi)),
        "ibm_li_sample_rows": int(len(ibm_li)),
        "main_upi_training_rows": int(len(main)),
        "aml_graph_rows": int(len(aml_graph)),
        "source_counts_main": {str(k): int(v) for k, v in main["source_dataset"].value_counts().items()},
        "scenario_counts": {str(k): int(v) for k, v in main["scenario"].value_counts().items()},
        "products": {
            **{key: str(value) for key, value in canonical_paths.items()},
            "upi_transaction_training": str(settings.processed_data),
            "aml_graph_edges": str(settings.aml_graph_data),
            "recipient_archetype_training": str(settings.recipient_archetype_data),
            "events": "data/sequences/events.parquet",
            "train": "data/processed/train.parquet",
            "validation": "data/processed/validation.parquet",
            "test": "data/processed/test.parquet",
            "edge_cases": "data/processed/edge_cases.parquet",
            "recipient_consistency": str(consistency_path),
        },
    }
    manifest_path = Path("data/processed/source_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return {**canonical_paths, "manifest": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PaySim + BankSim + IBM HI/LI data products")
    parser.add_argument("--paysim-rows", type=int, default=None)
    parser.add_argument("--banksim-rows", type=int, default=None)
    parser.add_argument("--ibm-negative-ratio", type=int, default=None)
    parser.add_argument("--ibm-max-negatives-per-file", type=int, default=None)
    parser.add_argument("--synthetic-fraud-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    prepare_multisource(
        paysim_rows=args.paysim_rows,
        banksim_rows=args.banksim_rows,
        ibm_negative_ratio=args.ibm_negative_ratio,
        ibm_max_negatives_per_file=args.ibm_max_negatives_per_file,
        synthetic_fraud_rate=args.synthetic_fraud_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
