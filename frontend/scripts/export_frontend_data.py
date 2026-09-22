from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FRONTEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRONTEND_ROOT.parent
PUBLIC = FRONTEND_ROOT / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "auto": {
        "scenario": "auto_long_distance_legitimate",
        "label": "Long-distance auto payment",
        "short_label": "Auto",
        "kind": "legitimate",
        "category": "Legitimate novelty",
        "description": "A distant or unfamiliar transport payment that should remain frictionless when journey and recipient context are plausible.",
    },
    "hospital": {
        "scenario": "hospital_emergency_legitimate",
        "label": "Emergency hospital payment",
        "short_label": "Hospital",
        "kind": "legitimate",
        "category": "Legitimate novelty",
        "description": "A large, unusual emergency payment where contextual evidence should override a suspicious-looking transaction amount.",
    },
    "qr_scam": {
        "scenario": "fake_refund_qr",
        "label": "Fake-refund QR scam",
        "short_label": "QR scam",
        "kind": "fraud",
        "category": "Social engineering",
        "description": "A refund lure using an unverified QR flow, social-engineering sequence evidence and recipient intelligence.",
    },
    "account_takeover": {
        "scenario": "account_takeover",
        "label": "Account takeover",
        "short_label": "ATO",
        "kind": "fraud",
        "category": "Identity & device",
        "description": "A compromised customer session where temporal behaviour, device signals and journey inconsistencies matter more than one row alone.",
    },
    "mule": {
        "scenario": "mule_directed",
        "label": "Mule-directed payment",
        "short_label": "Mule",
        "kind": "fraud",
        "category": "Graph & recipient",
        "description": "A transfer to a suspicious recipient where transaction, graph anomaly, complaint and merchant-trust evidence are fused.",
    },
    "collect_scam": {
        "scenario": "collect_request_scam",
        "label": "Fraudulent collect request",
        "short_label": "Collect",
        "kind": "fraud",
        "category": "Social engineering",
        "description": "A deceptive collect request designed to make the victim approve a payment they believe is an incoming credit.",
    },
    "remote_access": {
        "scenario": "remote_access_scam",
        "label": "Remote-access scam",
        "short_label": "Remote",
        "kind": "fraud",
        "category": "Device & session",
        "description": "A payment occurring in a remote-control compromise pattern with sequence, device and social-engineering evidence.",
    },
    "legitimate_split": {
        "scenario": "legitimate_split_payment",
        "label": "Legitimate split payment",
        "short_label": "Split legit",
        "kind": "legitimate",
        "category": "Hard negative",
        "description": "Several related payments that resemble structuring but have legitimate context and should avoid unnecessary intervention.",
    },
    "new_merchant": {
        "scenario": "new_merchant_legitimate",
        "label": "First payment to a new merchant",
        "short_label": "New merchant",
        "kind": "legitimate",
        "category": "Legitimate novelty",
        "description": "A new-payee transaction where merchant trust and normal device behaviour can explain novelty.",
    },
    "gig_worker": {
        "scenario": "gig_worker_payment_legitimate",
        "label": "Gig-worker payment",
        "short_label": "Gig worker",
        "kind": "legitimate",
        "category": "High fan-in hard negative",
        "description": "A recipient who legitimately receives many payments, testing whether the system confuses fan-in with mule behaviour.",
    },
    "travel_hotel": {
        "scenario": "travel_hotel_legitimate",
        "label": "Travel / hotel payment",
        "short_label": "Travel",
        "kind": "legitimate",
        "category": "Location hard negative",
        "description": "An unusual geographic payment that should be explained by plausible travel rather than treated as impossible movement.",
    },
    "familiar": {
        "scenario": "familiar_legitimate",
        "label": "Familiar recurring payment",
        "short_label": "Familiar",
        "kind": "legitimate",
        "category": "Baseline legitimate",
        "description": "A known-pattern transaction used as a sanity check for normal behaviour and low-friction approval.",
    },
}


def safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def bool_value(value: Any, default: bool = False) -> bool:
    value = safe(value)
    return default if value is None else bool(value)


def int_value(value: Any, default: int) -> int:
    value = safe(value)
    return default if value is None else int(value)


def float_value(value: Any, default: float | None = None) -> float | None:
    value = safe(value)
    return default if value is None else float(value)


def row_to_event(row: pd.Series, short_name: str) -> dict[str, Any]:
    original = str(row["transaction_id"])
    latitude = float_value(row.get("latitude"))
    longitude = float_value(row.get("longitude"))
    payee_latitude = float_value(row.get("payee_latitude"), latitude)
    payee_longitude = float_value(row.get("payee_longitude"), longitude)
    return {
        "transaction_id": f"WEB-{short_name.upper()}-{original}",
        "reference_transaction_id": original,
        "simulation_mode": True,
        "session_id": f"S-{original}",
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "payer_id": str(row["payer_id"]),
        "payee_id": str(row["payee_id"]),
        "amount": float(row["amount"]),
        "payment_mode": str(row.get("payment_mode", "P2P")),
        "device_id": str(safe(row.get("device_id")) or "D_DEMO"),
        "vpa_id": safe(row.get("vpa_id")),
        "merchant_id": safe(row.get("merchant_id")),
        "qr_id": safe(row.get("qr_id")),
        "ip_asn": safe(row.get("ip_asn")),
        "latitude": latitude,
        "longitude": longitude,
        "payee_latitude": payee_latitude,
        "payee_longitude": payee_longitude,
        "is_new_payee": bool_value(row.get("is_new_payee")),
        "qr_verified": bool_value(row.get("qr_verified")),
        "collect_request": bool_value(row.get("collect_request")),
        "merchant_category": safe(row.get("merchant_category")),
        "device_age_days": int_value(row.get("device_age_days"), 30),
        "sim_age_days": int_value(row.get("sim_age_days"), 60),
        "app_registration_age_days": int_value(row.get("app_registration_age_days"), 30),
        "device_known": bool_value(row.get("device_known"), True),
        "play_integrity_ok": bool_value(row.get("play_integrity_ok"), True),
        "recent_pin_reset": bool_value(row.get("recent_pin_reset")),
        "app_reregistered": bool_value(row.get("app_reregistered")),
        "remote_access_indicator": bool_value(row.get("remote_access_indicator")),
        "overlay_indicator": bool_value(row.get("overlay_indicator")),
        "recipient_complaint_score": float_value(row.get("recipient_complaint_score"), 0.0) or 0.0,
        "known_mule": False,
        "common_owner_verified": bool_value(row.get("common_owner_verified")),
    }


def choose_row(candidates: pd.DataFrame, kind: str) -> pd.Series:
    work = candidates.copy()
    final = pd.to_numeric(work.get("final_fraud_probability", 0.0), errors="coerce").fillna(0.0)
    legit = pd.to_numeric(work.get("legitimate_novelty_score", 0.0), errors="coerce").fillna(0.0)
    merchant = pd.to_numeric(work.get("graph_merchant_score", 0.0), errors="coerce").fillna(0.0)
    journey = pd.to_numeric(work.get("journey_plausibility", 0.5), errors="coerce").fillna(0.5)
    if kind == "fraud":
        work["_ui_rank"] = final
        return work.sort_values(["_ui_rank", "timestamp"], ascending=[False, True], kind="stable").iloc[0]
    work["_ui_rank"] = (1 - final) + 0.45 * legit + 0.25 * merchant + 0.20 * journey
    return work.sort_values(["_ui_rank", "timestamp"], ascending=[False, True], kind="stable").iloc[0]


def model_scores(row: pd.Series) -> dict[str, float]:
    mapping = {
        "rule_risk": "rule_risk",
        "statistical_anomaly": "statistical_anomaly_score",
        "transaction_fraud": "transaction_fraud_score",
        "legitimate_novelty": "legitimate_novelty_score",
        "device_session_risk": "device_session_risk",
        "sequence_account_takeover": "sequence_account_takeover_score",
        "sequence_social_engineering": "sequence_social_engineering_score",
        "graph_mule": "graph_mule_score",
        "graph_merchant_legitimacy": "graph_merchant_score",
        "graph_anomaly": "graph_anomaly_score",
        "complaint_intelligence": "complaint_intelligence_score",
        "journey_plausibility": "journey_plausibility",
    }
    result: dict[str, float] = {}
    for output, column in mapping.items():
        result[output] = float_value(row.get(column), 0.0) or 0.0
    return result


def offline_prediction(row: pd.Series) -> dict[str, Any]:
    return {
        "transaction_id": str(row["transaction_id"]),
        "fraud_probability": float_value(row.get("final_fraud_probability"), 0.0) or 0.0,
        "legitimate_novelty_probability": float_value(row.get("legitimate_novelty_score"), 0.0) or 0.0,
        "uncertainty": float_value(row.get("uncertainty"), 0.0) or 0.0,
        "recommended_action": str(safe(row.get("recommended_action")) or "UNKNOWN"),
        "fraud_types": [],
        "reason_codes": [],
        "model_scores": model_scores(row),
        "latency_ms": 0.0,
    }


def graph_for(data: pd.DataFrame, row: pd.Series) -> dict[str, list[dict[str, Any]]]:
    target = str(row["payee_id"])
    at = pd.Timestamp(row["timestamp"])
    hist = data[pd.to_datetime(data["timestamp"], errors="coerce").le(at)].copy()
    links = hist[(hist["payee_id"].astype(str).eq(target)) | (hist["payer_id"].astype(str).eq(target))]
    links = links.sort_values("timestamp").tail(80)
    if links.empty:
        return {"nodes": [{"id": target, "label": target, "type": "target", "risk": 0.0}], "edges": []}

    edge_table = (
        links.assign(src=links["payer_id"].astype(str), dst=links["payee_id"].astype(str))
        .groupby(["src", "dst"], observed=True)
        .agg(amount=("amount", "sum"), count=("transaction_id", "count"))
        .reset_index()
        .sort_values(["count", "amount"], ascending=False)
        .head(36)
    )
    ids = {target}
    edges: list[dict[str, Any]] = []
    for index, item in enumerate(edge_table.itertuples(index=False)):
        source, dest = str(item.src), str(item.dst)
        ids.update([source, dest])
        edges.append({"id": f"e{index}", "source": source, "target": dest, "amount": float(item.amount), "count": int(item.count)})
    nodes = []
    for node_id in sorted(ids):
        if node_id == target:
            node_type = "target"
        elif node_id.startswith("BS_MERCHANT") or "MERCHANT" in node_id:
            node_type = "merchant"
        elif node_id in set(edge_table["src"].astype(str)):
            node_type = "payer"
        else:
            node_type = "recipient"
        nodes.append({"id": node_id, "label": node_id, "type": node_type})
    return {"nodes": nodes, "edges": edges}


def timeline_for(data: pd.DataFrame, row: pd.Series) -> list[dict[str, Any]]:
    payer = str(row["payer_id"])
    selected_id = str(row["transaction_id"])
    at = pd.Timestamp(row["timestamp"])
    hist = data[(data["payer_id"].astype(str).eq(payer)) & pd.to_datetime(data["timestamp"], errors="coerce").le(at)].copy()
    hist = hist.sort_values("timestamp").tail(10)
    timeline = []
    for item in hist.itertuples(index=False):
        tags: list[str] = []
        for column, label in [
            ("is_new_payee", "New payee"),
            ("recent_pin_reset", "PIN reset"),
            ("app_reregistered", "App re-registered"),
            ("remote_access_indicator", "Remote access"),
            ("overlay_indicator", "Overlay detected"),
            ("collect_request", "Collect request"),
        ]:
            if bool_value(getattr(item, column, False)):
                tags.append(label)
        if hasattr(item, "qr_verified") and not bool_value(getattr(item, "qr_verified"), True) and str(getattr(item, "payment_mode", "")) == "QR":
            tags.append("Unverified QR")
        timeline.append({
            "transaction_id": str(item.transaction_id),
            "timestamp": pd.Timestamp(item.timestamp).isoformat(),
            "amount": float(item.amount),
            "scenario": str(getattr(item, "scenario", "unknown")),
            "action": str(getattr(item, "recommended_action", "")),
            "fraud_probability": float_value(getattr(item, "final_fraud_probability", None)),
            "tags": tags,
            "selected": str(item.transaction_id) == selected_id,
        })
    return timeline


def main() -> None:
    scored_path = REPO_ROOT / "data" / "processed" / "scored_transactions.pkl"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing {scored_path}. Run training first.")
    data = pd.read_pickle(scored_path).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    test_path = REPO_ROOT / "data" / "processed" / "test.parquet"
    if test_path.exists():
        test_ids = set(pd.read_parquet(test_path)["transaction_id"].astype(str))
        test = data[data["transaction_id"].astype(str).isin(test_ids)].copy()
    else:
        test = data.copy()

    manifest_path = REPO_ROOT / "data" / "processed" / "demo_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    result: list[dict[str, Any]] = []
    used_payees: set[str] = set()
    for key, meta in SCENARIOS.items():
        scenario = meta["scenario"]
        candidates = test[test["scenario"].astype(str).eq(scenario)].copy()
        if candidates.empty:
            continue
        if key in existing_manifest:
            exact_id = str(existing_manifest[key].get("transaction_id", ""))
            exact = candidates[candidates["transaction_id"].astype(str).eq(exact_id)]
            selected = exact.iloc[0] if not exact.empty else choose_row(candidates, meta["kind"])
        else:
            unused = candidates[~candidates["payee_id"].astype(str).isin(used_payees)]
            selected = choose_row(unused if not unused.empty else candidates, meta["kind"])
        used_payees.add(str(selected["payee_id"]))
        entry = {
            "key": key,
            **meta,
            "source_dataset": str(safe(selected.get("source_dataset")) or "unknown"),
            "recipient_profile": str(safe(selected.get("recipient_profile")) or "unknown"),
            "reference_transaction_id": str(selected["transaction_id"]),
            "payer_id": str(selected["payer_id"]),
            "payee_id": str(selected["payee_id"]),
            "amount": float(selected["amount"]),
            "payload": row_to_event(selected, key),
            "offline": offline_prediction(selected),
            "graph": graph_for(data, selected),
            "timeline": timeline_for(data, selected),
        }
        result.append(entry)

    (PUBLIC / "demo_scenarios.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    recent_cols = [
        "transaction_id", "timestamp", "source_dataset", "scenario", "payer_id", "payee_id", "amount",
        "final_fraud_probability", "legitimate_novelty_score", "uncertainty", "recommended_action"
    ]
    recent_cols = [c for c in recent_cols if c in test.columns]
    recent = test.sort_values("timestamp").tail(180)[recent_cols].copy()
    records = []
    for record in recent.to_dict(orient="records"):
        records.append({key: safe(value) for key, value in record.items()})
    (PUBLIC / "recent_decisions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    for filename in ["metrics.json", "evaluation_report.json"]:
        source = REPO_ROOT / "models" / filename
        if source.exists():
            shutil.copy2(source, PUBLIC / filename)

    print(f"Exported {len(result)} demo scenarios to {PUBLIC / 'demo_scenarios.json'}")
    print(f"Exported {len(records)} recent held-out decisions")
    print("Metrics/evaluation copied when available.")


if __name__ == "__main__":
    main()
