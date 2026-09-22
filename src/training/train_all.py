from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from src.data.prepare_multisource import prepare_multisource
from src.graph.build_views import build_and_save_all_views
from src.graph.event_encoder import encode_temporal_events
from src.intelligence.complaint_extractor import extract_complaint
from src.models.anomaly_model import GraphAnomalyModel
from src.models.context_model import LegitimateNoveltyModel
from src.models.fusion_model import (
    EXPERT_COLUMNS,
    add_fusion_runtime_columns,
    calibrate_fraud_type_thresholds,
    predict_fusion,
    train_fusion_model,
)
from src.models.hetero_tgn import (
    score_temporal_nodes,
    score_temporal_transactions,
    train_hetero_tgn,
)
from src.models.hgt_model import train_hgt_view
from src.models.lightgbm_model import TransactionRiskModel
from src.models.multiview_attention import score_multiview_attention, train_multiview_attention
from src.models.time_transformer import score_sequences, train_time_transformer
from src.models.uncertainty import calculate_uncertainty
from src.policy.decision_engine import PolicyEngine
from src.rules.engine import RulesEngine
from src.settings import get_settings
from src.simulation.generate_complaints import generate_complaints


def _embedding_frame(scores: pd.DataFrame, embedding_column: str, prefix: str) -> pd.DataFrame:
    matrix = np.stack(scores[embedding_column].to_numpy())
    result = pd.DataFrame(matrix, columns=[f"{prefix}_{idx}" for idx in range(matrix.shape[1])])
    result.insert(0, "account_id", scores["account_id"].to_numpy())
    return result


def _align_embeddings(
    transactions: pd.DataFrame,
    score_frames: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    merged = transactions[["payee_id"]].rename(columns={"payee_id": "account_id"}).copy()
    outputs = []
    availability = np.ones(len(transactions), dtype=np.float32)
    for view_name in ["money", "identity", "context"]:
        frame = _embedding_frame(score_frames[view_name], f"{view_name}_embedding", view_name)
        aligned_raw = merged.merge(frame, on="account_id", how="left").drop(columns="account_id")
        availability *= (~aligned_raw.isna().all(axis=1)).to_numpy(np.float32)
        outputs.append(aligned_raw.fillna(0.0).to_numpy(np.float32))
    return outputs[0], outputs[1], outputs[2], availability


def _query_features(frame: pd.DataFrame) -> np.ndarray:
    complaint = frame.get("complaint_intelligence_score", pd.Series(0.0, index=frame.index))
    return np.column_stack(
        [
            frame["log_amount"],
            frame["is_new_payee"],
            (frame["payment_mode"] == "QR").astype(float),
            frame["collect_request"],
            frame["journey_plausibility"],
            frame["device_session_risk"],
            complaint,
            np.log1p(frame["recipient_account_age_days"]) / np.log1p(3650.0),
        ]
    ).astype(np.float32)


def _build_rule_scores(frame: pd.DataFrame) -> tuple[np.ndarray, list[str | None], list[list[str]]]:
    engine = RulesEngine()
    risk: list[float] = []
    actions: list[str | None] = []
    reason_codes: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        result = engine.evaluate(row)
        risk.append(max(0.0, result.risk_delta))
        actions.append(result.hard_action)
        reason_codes.append(result.reason_codes + result.protective_codes)
    return np.asarray(risk, dtype=np.float32), actions, reason_codes


def _process_all_delayed_complaints(all_data: pd.DataFrame, use_llm: bool) -> pd.DataFrame:
    """Create reports and retain their real delayed availability timestamp."""
    complaint_path = Path("data/complaints/raw_complaints.parquet")
    max_rows = min(max(1000, len(all_data) // 5), 10_000)
    raw = generate_complaints(all_data, complaint_path, max_rows=max_rows)
    rows: list[dict[str, object]] = []
    for record in raw.itertuples(index=False):
        intelligence = extract_complaint(record.text, use_llm=use_llm)
        rows.append(
            {
                "complaint_id": record.complaint_id,
                "transaction_id": record.transaction_id,
                "payee_id": record.payee_id,
                "transaction_timestamp": pd.Timestamp(record.transaction_timestamp),
                "complaint_timestamp": pd.Timestamp(record.complaint_timestamp),
                "reported_is_fraud": int(record.reported_is_fraud),
                **intelligence.model_dump(),
            }
        )
    structured = pd.DataFrame(rows)
    if structured.empty:
        structured = pd.DataFrame(
            columns=[
                "complaint_id",
                "transaction_id",
                "payee_id",
                "transaction_timestamp",
                "complaint_timestamp",
                "reported_is_fraud",
                "confidence",
            ]
        )
    Path("data/complaints").mkdir(parents=True, exist_ok=True)
    structured.to_parquet("data/complaints/structured_complaints.parquet", index=False)
    return structured


def _add_time_aware_complaint_scores(
    transactions: pd.DataFrame,
    complaints: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only complaints that existed before each transaction timestamp."""
    out = transactions.copy().sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    if complaints.empty:
        out["complaint_intelligence_score"] = 0.0
        out["complaint_history_count"] = 0
        return out

    complaint_groups = {
        str(payee): group.sort_values("complaint_timestamp")
        for payee, group in complaints.groupby("payee_id", observed=True)
    }
    scores = np.zeros(len(out), dtype=float)
    counts = np.zeros(len(out), dtype=int)

    for payee_id, indices in out.groupby("payee_id", observed=True).groups.items():
        payee = str(payee_id)
        reports = complaint_groups.get(payee)
        if reports is None or reports.empty:
            continue
        report_times = pd.to_datetime(reports["complaint_timestamp"], utc=True).to_numpy()
        confidence = pd.to_numeric(reports["confidence"], errors="coerce").fillna(0).to_numpy(float)
        cumulative_sum = np.cumsum(confidence)
        tx_indices = np.asarray(list(indices), dtype=int)
        tx_times = pd.to_datetime(out.loc[tx_indices, "timestamp"], utc=True).to_numpy()
        positions = np.searchsorted(report_times, tx_times, side="right") - 1
        valid = positions >= 0
        if valid.any():
            pos = positions[valid]
            scores[tx_indices[valid]] = cumulative_sum[pos] / (pos + 1)
            counts[tx_indices[valid]] = pos + 1

    out["complaint_intelligence_score"] = np.clip(scores, 0, 1)
    out["complaint_history_count"] = counts
    return out


def _historical_flow_risk(frame: pd.DataFrame) -> np.ndarray:
    fan_in = pd.to_numeric(frame["fan_in_1h"], errors="coerce").fillna(0).to_numpy(float)
    fan_out = pd.to_numeric(frame["fan_out_1h"], errors="coerce").fillna(0).to_numpy(float)
    rapid = pd.to_numeric(frame["rapid_outflow_ratio"], errors="coerce").fillna(0).to_numpy(float)
    holding = pd.to_numeric(frame["median_holding_minutes"], errors="coerce").fillna(180).to_numpy(float)
    return np.clip(
        0.42 * rapid
        + 0.22 * (fan_out / (fan_out + 4.0))
        + 0.18 * (fan_in / (fan_in + 8.0))
        + 0.18 * np.exp(-holding / 75.0),
        0,
        1,
    )


def _fraud_type_metrics(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for column in columns:
        target = frame[column].to_numpy(int)
        score = frame[f"pred_{column}"].to_numpy(float)
        if target.sum() == 0:
            continue
        output[column] = {
            "pr_auc": float(average_precision_score(target, score)),
            "prevalence": float(target.mean()),
        }
    return output


def train_all(use_llm: bool = False, skip_prepare: bool = False) -> dict[str, float]:
    settings = get_settings()
    required_products = [settings.processed_data, settings.aml_graph_data, settings.recipient_archetype_data]
    if not skip_prepare or not all(path.exists() for path in required_products):
        prepare_multisource()

    all_data = pd.read_parquet(settings.processed_data)
    aml_graph_data = pd.read_parquet(settings.aml_graph_data)
    train = pd.read_parquet("data/processed/train.parquet")
    validation = pd.read_parquet("data/processed/validation.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    train_ids = set(train["transaction_id"].astype(str))
    validation_ids = set(validation["transaction_id"].astype(str))
    test_ids = set(test["transaction_id"].astype(str))
    Path("models").mkdir(exist_ok=True)

    # Fast transaction expert.
    transaction_model = TransactionRiskModel.create(seed=settings.random_seed)
    transaction_metrics = transaction_model.fit(train, validation)
    transaction_model.save("models/lightgbm/transaction_model.joblib")
    all_data["transaction_fraud_score"] = transaction_model.predict_proba(all_data)

    # Legitimate-novelty expert.
    context_model = LegitimateNoveltyModel.create(seed=settings.random_seed)
    context_metrics = context_model.fit(train, validation)
    context_model.save("models/context/context_model.joblib")
    all_data["legitimate_novelty_score"] = context_model.predict_proba(all_data)
    all_data["legitimate_novelty_inverse"] = 1 - all_data["legitimate_novelty_score"]

    # Transaction/session-specific sequence expert.
    events = pd.read_parquet("data/sequences/events.parquet")
    sequence_bundle = train_time_transformer(
        events[events["transaction_id"].astype(str).isin(train_ids)],
        "models/sequence",
        epochs=6,
    )
    sequence_scores = score_sequences(sequence_bundle, events).drop(columns="sequence_embedding")
    all_data = all_data.merge(sequence_scores, on="transaction_id", how="left")
    for column in ["sequence_account_takeover_score", "sequence_social_engineering_score"]:
        all_data[column] = all_data[column].fillna(0.0)
    all_data["sequence_history_size"] = all_data["sequence_history_size"].fillna(0).astype(int)

    # Delayed complaint intelligence is calculated as-of every transaction time.
    structured_complaints = _process_all_delayed_complaints(all_data, use_llm=use_llm)
    all_data = _add_time_aware_complaint_scores(all_data, structured_complaints)

    # Graph experts are fitted only on the training snapshot. IBM LI remains held out.
    graph_upi_train = all_data[all_data["transaction_id"].astype(str).isin(train_ids)].copy()
    graph_upi_train["recipient_complaint_score"] = graph_upi_train["complaint_intelligence_score"]
    ibm_hi_train = aml_graph_data[aml_graph_data["source_partition"] == "hi_small"].copy()
    money_flow_train = pd.concat([graph_upi_train, ibm_hi_train], ignore_index=True, sort=False)

    view_paths = build_and_save_all_views(
        graph_upi_train,
        "data/graph_views",
        money_flow_df=money_flow_train,
    )
    hgt_scores: dict[str, pd.DataFrame] = {}
    for view_name, path in view_paths.items():
        result = train_hgt_view(path, "models/graph", view_name=view_name, epochs=12)
        hgt_scores[view_name] = result.scores

    money_emb, identity_emb, context_emb, hgt_available = _align_embeddings(all_data, hgt_scores)
    query = _query_features(all_data)
    train_positions = all_data["transaction_id"].astype(str).isin(train_ids).to_numpy()
    multiview = train_multiview_attention(
        money_emb[train_positions],
        identity_emb[train_positions],
        context_emb[train_positions],
        query[train_positions],
        all_data.loc[train_positions, "is_mule_directed"].to_numpy(np.float32),
        all_data.loc[train_positions, "recipient_archetype"].isin(
            ["merchant", "transport_provider", "gig_worker", "new_business"]
        ).to_numpy(np.float32),
        "models/graph/multiview_attention.pt",
        epochs=18,
    )
    multiview_mule, graph_merchant, fused_graph_embedding, view_weights = score_multiview_attention(
        multiview, money_emb, identity_emb, context_emb, query
    )
    all_data["multiview_mule_score"] = multiview_mule
    all_data["graph_merchant_score"] = graph_merchant
    all_data["money_view_weight"] = view_weights[:, 0]
    all_data["identity_view_weight"] = view_weights[:, 1]
    all_data["context_view_weight"] = view_weights[:, 2]

    # Train TGN on train-period events, then score each UPI transaction from prior
    # recipient history before the current event is inserted.
    temporal_events_train, indexer = encode_temporal_events(money_flow_train)
    temporal_bundle = train_hetero_tgn(
        temporal_events_train,
        "models/graph/hetero_tgn",
        epochs=10,
    )
    node_scores = score_temporal_nodes(temporal_bundle, temporal_events_train)
    node_scores = node_scores.merge(indexer.to_frame(), on="global_node_id", how="left")
    node_scores.to_pickle("models/graph/tgn_node_scores.pkl")

    upi_temporal_events, _ = encode_temporal_events(all_data, indexer=indexer)
    transaction_tgn = score_temporal_transactions(temporal_bundle, upi_temporal_events)
    all_data = all_data.merge(transaction_tgn.drop(columns="tgn_embedding_prior"), on="transaction_id", how="left")
    all_data["tgn_mule_score_prior"] = all_data["tgn_mule_score_prior"].fillna(0.0)
    all_data["tgn_rapid_outflow_score_prior"] = all_data["tgn_rapid_outflow_score_prior"].fillna(0.0)
    all_data["tgn_history_size"] = all_data["tgn_history_size"].fillna(0).astype(int)
    all_data["historical_flow_risk"] = _historical_flow_risk(all_data)
    all_data["graph_mule_score"] = np.clip(
        0.45 * all_data["multiview_mule_score"]
        + 0.35 * all_data["tgn_mule_score_prior"]
        + 0.20 * all_data["historical_flow_risk"],
        0,
        1,
    )

    anomaly_model = GraphAnomalyModel.fit(fused_graph_embedding[train_positions])
    anomaly_model.save("models/graph/anomaly_model.joblib")
    all_data["graph_anomaly_score"] = anomaly_model.score(fused_graph_embedding)

    rule_risk, hard_actions, reasons = _build_rule_scores(all_data)
    all_data["rule_risk"] = rule_risk
    all_data["hard_action"] = hard_actions
    all_data["rule_reason_codes"] = [json.dumps(value) for value in reasons]
    all_data["statistical_anomaly"] = np.clip(
        0.60 * (1 - np.exp(-np.abs(all_data["amount_robust_z"].to_numpy(float)) / 3))
        + 0.25
        * (
            all_data["txn_count_5m"].to_numpy(float)
            / (all_data["txn_count_5m"].to_numpy(float) + 4)
        )
        + 0.15 * (1 - all_data["location_continuity"].to_numpy(float)),
        0,
        1,
    )
    all_data["journey_risk"] = 1 - all_data["journey_plausibility"]
    all_data["is_qr"] = (all_data["payment_mode"] == "QR").astype(int)
    all_data["device_changed"] = 1 - all_data["device_known"].astype(int)
    all_data["graph_neighbourhood_size"] = all_data["fan_in_1h"] + all_data["fan_out_1h"]
    all_data["missing_feature_ratio"] = all_data[
        ["latitude", "longitude", "merchant_id", "ip_asn", "qr_id"]
    ].isna().mean(axis=1)
    all_data["graph_score_available"] = (
        (hgt_available > 0) | (all_data["tgn_history_size"].to_numpy() > 0)
    ).astype(float)
    all_data["sequence_score_available"] = (all_data["sequence_history_size"] > 0).astype(float)
    all_data["complaint_score_available"] = (all_data["complaint_history_count"] > 0).astype(float)
    all_data["recipient_history_available"] = (
        (all_data["graph_neighbourhood_size"] > 0) | (all_data["tgn_history_size"] > 0)
    ).astype(float)
    all_data["payer_history_available"] = (
        (all_data["txn_count_1h"] > 0) | (all_data["time_since_previous_sec"] < 86_400)
    ).astype(float)
    all_data = add_fusion_runtime_columns(all_data)

    # Train fusion on held-out validation predictions, augmented with missing and
    # contradictory expert combinations. Test remains untouched.
    fusion_meta = all_data[all_data["transaction_id"].astype(str).isin(validation_ids)].copy()
    fusion_meta.to_parquet("data/processed/fusion_training.parquet", index=False)
    fusion_bundle = train_fusion_model(
        fusion_meta,
        "models/fusion",
        epochs=30,
        seed=settings.random_seed,
    )
    fraud_type_thresholds = calibrate_fraud_type_thresholds(fusion_bundle, fusion_meta)
    fusion_bundle.save("models/fusion")

    final_probability, label_probabilities, mixture_weights = predict_fusion(fusion_bundle, all_data)
    all_data["final_fraud_probability"] = final_probability
    for idx, column in enumerate(fusion_bundle.fraud_type_columns):
        all_data[f"pred_{column}"] = label_probabilities[:, idx]
        threshold = fraud_type_thresholds[column]
        all_data[f"flag_{column}"] = (label_probabilities[:, idx] >= threshold).astype(int)
    for idx, column in enumerate(fusion_bundle.expert_columns):
        all_data[f"mixture_weight_{column}"] = mixture_weights[:, idx]

    expert_matrix = all_data[EXPERT_COLUMNS].to_numpy(np.float32)
    all_data["uncertainty"] = calculate_uncertainty(
        expert_matrix,
        final_probability,
        all_data["missing_feature_ratio"].to_numpy(float),
        1 / (1 + all_data["graph_neighbourhood_size"].to_numpy(float)),
        all_data["graph_anomaly_score"].to_numpy(float),
    )

    policy = PolicyEngine()
    all_data["recommended_action"] = [
        policy.select_action(
            fraud_risk=float(row.final_fraud_probability),
            uncertainty=float(row.uncertainty),
            context_legitimacy=float(row.legitimate_novelty_score),
            mule_risk=float(row.graph_mule_score),
            account_takeover_risk=float(row.sequence_account_takeover_score),
            hard_action=row.hard_action if isinstance(row.hard_action, str) else None,
            transaction_risk=float(row.transaction_fraud_score),
            graph_anomaly=float(row.graph_anomaly_score),
            social_engineering_risk=float(row.sequence_social_engineering_score),
            qr_unverified=bool(row.payment_mode == "QR" and int(row.qr_verified) == 0),
        )
        for row in all_data.itertuples(index=False)
    ]

    all_data.to_pickle("data/processed/scored_transactions.pkl")
    all_data.to_parquet("data/processed/scored_transactions.parquet", index=False)

    # Online recipient cache is a train-cutoff snapshot, never a max over future
    # validation/test activity. Historical demos use transaction-specific snapshots.
    train_scored = all_data[all_data["transaction_id"].astype(str).isin(train_ids)].sort_values("timestamp")
    recipient_scores = (
        train_scored.groupby("payee_id", observed=True, sort=False)
        .tail(1)[
            [
                "payee_id",
                "graph_mule_score",
                "graph_merchant_score",
                "complaint_intelligence_score",
                "graph_anomaly_score",
                "recipient_archetype",
                "recipient_account_age_days",
                "rapid_outflow_ratio",
                "median_holding_minutes",
                "fan_in_1h",
                "fan_out_1h",
                "timestamp",
            ]
        ]
        .rename(
            columns={
                "graph_mule_score": "learned_mule_score",
                "graph_merchant_score": "merchant_legitimacy",
                "complaint_intelligence_score": "complaint_intelligence",
                "graph_anomaly_score": "graph_anomaly",
                "timestamp": "as_of_timestamp",
            }
        )
        .reset_index(drop=True)
    )
    recipient_scores.to_parquet("models/graph/recipient_scores.parquet", index=False)

    test_scored = all_data[all_data["transaction_id"].astype(str).isin(test_ids)].copy()
    metrics: dict[str, object] = {
        "transaction_pr_auc": transaction_metrics["pr_auc"],
        "transaction_roc_auc": transaction_metrics["roc_auc"],
        "context_pr_auc": context_metrics["pr_auc"],
        "final_pr_auc": float(
            average_precision_score(test_scored["is_fraud"], test_scored["final_fraud_probability"])
        ),
        "final_roc_auc": float(
            roc_auc_score(test_scored["is_fraud"], test_scored["final_fraud_probability"])
        ),
        "legitimate_novelty_false_positive_rate_at_05": float(
            (
                test_scored.loc[
                    test_scored["is_legitimate_novelty"] == 1,
                    "final_fraud_probability",
                ]
                >= 0.5
            ).mean()
        ),
        "fraud_type_thresholds": fraud_type_thresholds,
        "fraud_type_test_metrics": _fraud_type_metrics(
            test_scored, fusion_bundle.fraud_type_columns
        ),
    }
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(
        classification_report(
            test_scored["is_fraud"],
            test_scored["final_fraud_probability"] >= 0.5,
            digits=4,
        )
    )

    # Build fixed unique-recipient demos after every successful retraining run.
    try:
        from scripts.build_demo_manifest import build_demo_manifest

        build_demo_manifest()
    except Exception as exc:
        print(f"Warning: demo manifest was not created automatically: {exc}")
    return {key: value for key, value in metrics.items() if isinstance(value, float)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all fraud-intelligence models")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the configured OpenAI API for complaint extraction",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    train_all(use_llm=args.use_llm, skip_prepare=args.skip_prepare)


if __name__ == "__main__":
    main()
