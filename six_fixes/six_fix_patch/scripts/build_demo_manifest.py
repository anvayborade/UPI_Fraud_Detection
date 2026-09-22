from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCENARIOS = {
    "auto": "auto_long_distance_legitimate",
    "hospital": "hospital_emergency_legitimate",
    "qr_scam": "fake_refund_qr",
    "account_takeover": "account_takeover",
    "mule": "mule_directed",
}


def numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def suitability(frame: pd.DataFrame, short_name: str) -> pd.Series:
    final = numeric(frame, "final_fraud_probability")
    txn = numeric(frame, "transaction_fraud_score")
    mule = numeric(frame, "graph_mule_score")
    merchant = numeric(frame, "graph_merchant_score")
    anomaly = numeric(frame, "graph_anomaly_score")
    complaint = numeric(frame, "complaint_intelligence_score")
    journey = numeric(frame, "journey_plausibility", 0.5)
    social = numeric(frame, "sequence_social_engineering_score")
    ato = numeric(frame, "sequence_account_takeover_score")
    qr_verified = numeric(frame, "qr_verified", 1.0)

    if short_name in {"auto", "hospital"}:
        return (
            3.0 * (1 - final)
            + 2.0 * (1 - txn)
            + 2.0 * merchant
            + 1.5 * journey
            + 1.5 * (1 - mule)
            + 1.0 * (1 - social)
            + 1.0 * (1 - complaint)
        )
    if short_name == "qr_scam":
        return 3.0 * final + 2.0 * txn + 2.0 * social + complaint + anomaly + (1 - qr_verified)
    if short_name == "account_takeover":
        return 3.0 * final + 2.5 * txn + 3.0 * ato + anomaly
    if short_name == "mule":
        return 2.0 * final + 2.0 * txn + 3.0 * mule + 1.5 * anomaly + complaint - 0.75 * merchant
    return final


def build_demo_manifest() -> Path:
    scored_path = Path("data/processed/scored_transactions.pkl")
    if not scored_path.exists():
        raise FileNotFoundError("Run training first; scored_transactions.pkl is missing.")

    data = pd.read_pickle(scored_path).copy()
    test_path = Path("data/processed/test.parquet")
    if test_path.exists():
        test_ids = set(pd.read_parquet(test_path)["transaction_id"].astype(str))
        data = data[data["transaction_id"].astype(str).isin(test_ids)].copy()

    manifest: dict[str, dict[str, str]] = {}
    used_payees: set[str] = set()
    selection_order = ["mule", "qr_scam", "account_takeover", "auto", "hospital"]

    for short_name in selection_order:
        scenario = SCENARIOS[short_name]
        candidates = data[data["scenario"].astype(str).eq(scenario)].copy()
        if candidates.empty:
            raise ValueError(f"No held-out rows found for scenario: {scenario}")
        candidates["_demo_score"] = suitability(candidates, short_name)
        candidates = candidates.sort_values(
            ["_demo_score", "timestamp"], ascending=[False, True], kind="stable"
        )
        unique = candidates[~candidates["payee_id"].astype(str).isin(used_payees)]
        if unique.empty:
            raise ValueError(
                f"Could not find a distinct recipient for {scenario}. "
                "Stable recipient generation should be checked before demonstrating."
            )
        selected = unique.iloc[0]
        payee_id = str(selected["payee_id"])
        used_payees.add(payee_id)
        manifest[short_name] = {
            "scenario": scenario,
            "transaction_id": str(selected["transaction_id"]),
            "payer_id": str(selected["payer_id"]),
            "payee_id": payee_id,
            "recipient_profile": str(selected.get("recipient_profile", "unknown")),
            "source_dataset": str(selected.get("source_dataset", "unknown")),
        }

    output = Path("data/processed/demo_manifest.json")
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nSaved {output}")
    return output


def main() -> None:
    build_demo_manifest()


if __name__ == "__main__":
    main()
