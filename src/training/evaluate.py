from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _safe_auc(y_true: pd.Series, y_score: pd.Series, kind: str) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    if kind == "pr":
        return float(average_precision_score(y_true, y_score))
    return float(roc_auc_score(y_true, y_score))


def evaluate(
    scored_path: str | Path = "data/processed/scored_transactions.pkl",
    output_path: str | Path = "models/evaluation_report.json",
) -> dict:
    path = Path(scored_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run 'python -m src.training.train_all --skip-prepare' first."
        )
    frame = pd.read_pickle(path)
    test_path = Path("data/processed/test.parquet")
    if test_path.exists():
        test_ids = set(pd.read_parquet(test_path)["transaction_id"].astype(str))
        frame = frame[frame["transaction_id"].astype(str).isin(test_ids)].copy()
    required = {"is_fraud", "final_fraud_probability", "scenario", "recommended_action"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Scored table is missing columns: {sorted(missing)}")

    metrics = {
        "rows": int(len(frame)),
        "fraud_prevalence": float(frame["is_fraud"].mean()),
        "pr_auc": _safe_auc(frame["is_fraud"], frame["final_fraud_probability"], "pr"),
        "roc_auc": _safe_auc(frame["is_fraud"], frame["final_fraud_probability"], "roc"),
        "brier_score": float(brier_score_loss(frame["is_fraud"], frame["final_fraud_probability"])),
    }

    legitimate_novelty = frame[frame.get("is_legitimate_novelty", 0) == 1]
    metrics["legitimate_novelty_count"] = int(len(legitimate_novelty))
    metrics["legitimate_novelty_fpr_at_050"] = float(
        (legitimate_novelty["final_fraud_probability"] >= 0.50).mean()
    ) if len(legitimate_novelty) else float("nan")
    metrics["legitimate_novelty_block_or_hold_rate"] = float(
        legitimate_novelty["recommended_action"].isin(["BLOCK", "HOLD"]).mean()
    ) if len(legitimate_novelty) else float("nan")

    by_scenario: dict[str, dict] = {}
    for scenario, group in frame.groupby("scenario", observed=True):
        by_scenario[str(scenario)] = {
            "rows": int(len(group)),
            "fraud_rate": float(group["is_fraud"].mean()),
            "mean_risk": float(group["final_fraud_probability"].mean()),
            "mean_uncertainty": float(group.get("uncertainty", pd.Series(0.0, index=group.index)).mean()),
            "actions": {str(key): int(value) for key, value in group["recommended_action"].value_counts().items()},
        }

    by_source: dict[str, dict] = {}
    if "source_dataset" in frame.columns:
        for source, group in frame.groupby("source_dataset", observed=True):
            by_source[str(source)] = {
                "rows": int(len(group)),
                "fraud_prevalence": float(group["is_fraud"].mean()),
                "pr_auc": _safe_auc(group["is_fraud"], group["final_fraud_probability"], "pr"),
                "roc_auc": _safe_auc(group["is_fraud"], group["final_fraud_probability"], "roc"),
                "mean_risk": float(group["final_fraud_probability"].mean()),
            }

    metrics["scope"] = "held_out_test"
    report = {
        "summary": metrics,
        "by_source": by_source,
        "by_scenario": by_scenario,
        "action_distribution": {
            str(key): int(value) for key, value in frame["recommended_action"].value_counts().items()
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, allow_nan=True))
    print(f"Full report written to {output}")
    return report


if __name__ == "__main__":
    evaluate()
