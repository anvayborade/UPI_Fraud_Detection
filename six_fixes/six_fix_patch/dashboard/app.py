from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("UPI_API_URL", "http://localhost:8000")

st.set_page_config(page_title="UPI Fraud Intent Firewall", layout="wide")
st.title("UPI Fraud Intent Firewall")
st.caption("Context-aware dual-speed fraud intelligence research prototype")


def api_post(path: str, payload: dict[str, Any]):
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def nullable(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def row_to_event(row: pd.Series) -> dict[str, Any]:
    original_transaction_id = str(row["transaction_id"])
    latitude = nullable(row.get("latitude"))
    longitude = nullable(row.get("longitude"))
    payee_latitude = nullable(row.get("payee_latitude"))
    payee_longitude = nullable(row.get("payee_longitude"))
    if payee_latitude is None:
        payee_latitude = latitude
    if payee_longitude is None:
        payee_longitude = longitude
    return {
        "transaction_id": f"DASH-{original_transaction_id}",
        "reference_transaction_id": original_transaction_id,
        "simulation_mode": True,
        "session_id": f"S-{original_transaction_id}",
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "payer_id": str(row["payer_id"]),
        "payee_id": str(row["payee_id"]),
        "amount": float(row["amount"]),
        "payment_mode": str(row["payment_mode"]),
        "device_id": str(row.get("device_id", "D_DEMO")),
        "vpa_id": nullable(row.get("vpa_id")),
        "merchant_id": nullable(row.get("merchant_id")),
        "qr_id": nullable(row.get("qr_id")),
        "ip_asn": nullable(row.get("ip_asn")),
        "latitude": latitude,
        "longitude": longitude,
        "payee_latitude": payee_latitude,
        "payee_longitude": payee_longitude,
        "is_new_payee": bool(row.get("is_new_payee", False)),
        "qr_verified": bool(row.get("qr_verified", False)),
        "collect_request": bool(row.get("collect_request", False)),
        "merchant_category": nullable(row.get("merchant_category")),
        "device_age_days": int(row.get("device_age_days", 30)),
        "sim_age_days": int(row.get("sim_age_days", 60)),
        "app_registration_age_days": int(row.get("app_registration_age_days", 30)),
        "device_known": bool(row.get("device_known", True)),
        "play_integrity_ok": bool(row.get("play_integrity_ok", True)),
        "recent_pin_reset": bool(row.get("recent_pin_reset", False)),
        "app_reregistered": bool(row.get("app_reregistered", False)),
        "remote_access_indicator": bool(row.get("remote_access_indicator", False)),
        "overlay_indicator": bool(row.get("overlay_indicator", False)),
        "recipient_complaint_score": float(row.get("recipient_complaint_score", 0.0)),
        "known_mule": False,
        "common_owner_verified": bool(row.get("common_owner_verified", False)),
    }


@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path("data/processed/scored_transactions.pkl")
    return pd.read_pickle(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_manifest() -> dict[str, dict[str, str]]:
    path = Path("data/processed/demo_manifest.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


data = load_data()
manifest = load_manifest()

tab_live, tab_simulator, tab_explain, tab_graph, tab_metrics, tab_complaint = st.tabs(
    [
        "Offline Decisions",
        "Edge-Case Simulator",
        "Expert Explanation",
        "Recipient Graph",
        "Model Performance",
        "Complaint Intelligence",
    ]
)

with tab_live:
    st.subheader("Recent offline-scored transactions")
    if data.empty:
        st.info("Run data preparation and training first.")
    else:
        columns = [
            column
            for column in [
                "transaction_id",
                "source_dataset",
                "scenario",
                "recipient_profile",
                "amount",
                "final_fraud_probability",
                "uncertainty",
                "recommended_action",
            ]
            if column in data.columns
        ]
        st.dataframe(
            data[columns].tail(200).sort_values(
                "final_fraud_probability", ascending=False
            ),
            use_container_width=True,
        )

with tab_simulator:
    st.subheader("Replay a held-out transaction using its time-specific expert snapshot")
    if data.empty:
        st.info("No scored transactions are available.")
    elif not manifest:
        st.info("Run `python -m scripts.build_demo_manifest` first.")
    else:
        short_name = st.selectbox("Demo", sorted(manifest))
        item = manifest[short_name]
        matches = data[data["transaction_id"].astype(str).eq(item["transaction_id"])]
        if matches.empty:
            st.error("The manifest transaction is not present in scored_transactions.pkl.")
        else:
            selected = matches.iloc[0]
            st.write(
                {
                    "scenario": item["scenario"],
                    "recipient_profile": item.get("recipient_profile", "unknown"),
                    "source_dataset": item.get("source_dataset", "unknown"),
                    "payer_id": item["payer_id"],
                    "payee_id": item["payee_id"],
                }
            )
            event = row_to_event(selected)
            with st.expander("Request payload"):
                st.json(event)
            first, second = st.columns(2)
            if first.button("Run recipient precheck", use_container_width=True):
                try:
                    st.session_state["precheck"] = api_post("/precheck", event)
                except Exception as exc:
                    st.error(str(exc))
            if second.button(
                "Score before authorisation", type="primary", use_container_width=True
            ):
                try:
                    st.session_state["decision"] = api_post("/score", event)
                except Exception as exc:
                    st.error(str(exc))
            if "precheck" in st.session_state:
                st.markdown("**Precheck result**")
                st.json(st.session_state["precheck"])
            if "decision" in st.session_state:
                decision = st.session_state["decision"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fraud probability", f"{decision['fraud_probability']:.1%}")
                c2.metric(
                    "Legitimate novelty",
                    f"{decision['legitimate_novelty_probability']:.1%}",
                )
                c3.metric("Uncertainty", f"{decision['uncertainty']:.1%}")
                c4.metric("Action", decision["recommended_action"])
                st.json(decision)

with tab_explain:
    st.subheader("Specialist-model contributions")
    decision = st.session_state.get("decision")
    if not decision:
        st.info("Score a demo transaction first.")
    else:
        scores = pd.DataFrame(
            {
                "Expert": list(decision["model_scores"].keys()),
                "Score": list(decision["model_scores"].values()),
            }
        )
        st.plotly_chart(
            px.bar(scores, x="Expert", y="Score", range_y=[0, 1]),
            use_container_width=True,
        )
        st.markdown("**Reason codes**")
        st.write(" • ".join(decision["reason_codes"]) or "No reason code was triggered.")

with tab_graph:
    st.subheader("Local payer-payee network")
    if data.empty:
        st.info("No graph data available.")
    else:
        target = st.selectbox(
            "Recipient", data["payee_id"].value_counts().head(100).index.tolist()
        )
        neighbourhood = data[
            (data["payee_id"] == target) | (data["payer_id"] == target)
        ].tail(100)
        graph = nx.DiGraph()
        for row in neighbourhood.itertuples(index=False):
            graph.add_edge(str(row.payer_id), str(row.payee_id), amount=float(row.amount))
        positions = nx.spring_layout(graph, seed=42)
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for source, destination in graph.edges():
            x0, y0 = positions[source]
            x1, y1 = positions[destination]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_x = [positions[node][0] for node in graph.nodes]
        node_y = [positions[node][1] for node in graph.nodes]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none"))
        figure.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=list(graph.nodes),
                textposition="top center",
                marker={"size": 14},
            )
        )
        figure.update_layout(
            showlegend=False,
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=650,
        )
        st.plotly_chart(figure, use_container_width=True)

with tab_metrics:
    st.subheader("Held-out test metrics")
    metrics_path = Path("models/metrics.json")
    if not metrics_path.exists():
        st.info("Run training first.")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        scalar_metrics = {
            key: value for key, value in metrics.items() if isinstance(value, (int, float))
        }
        columns = st.columns(min(4, max(1, len(scalar_metrics))))
        for index, (name, value) in enumerate(scalar_metrics.items()):
            columns[index % len(columns)].metric(
                name.replace("_", " ").title(), f"{float(value):.4f}"
            )
        if metrics.get("fraud_type_thresholds"):
            st.markdown("**Calibrated fraud-type thresholds**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"fraud_type": key, "threshold": value}
                        for key, value in metrics["fraud_type_thresholds"].items()
                    ]
                ),
                use_container_width=True,
            )
        report_path = Path("models/evaluation_report.json")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            st.json(report.get("summary", {}))
            if report.get("by_source"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"source_dataset": source, **values}
                            for source, values in report["by_source"].items()
                        ]
                    ),
                    use_container_width=True,
                )

with tab_complaint:
    st.subheader("Convert a complaint into structured scam intelligence")
    payee = st.text_input("Recipient ID", value="A000001")
    text = st.text_area(
        "Complaint text",
        value=(
            "A caller said I would receive a refund and asked me to scan a QR code "
            "urgently. Money was debited after I entered my UPI PIN."
        ),
    )
    use_llm = st.checkbox("Use configured external LLM", value=False)
    if st.button("Extract intelligence"):
        try:
            st.json(
                api_post(
                    "/complaint",
                    {"payee_id": payee, "text": text, "use_llm": use_llm},
                )
            )
        except Exception as exc:
            st.error(str(exc))
