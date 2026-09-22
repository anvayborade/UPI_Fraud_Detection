from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


TRANSACTION_ID = "BS_TXN_0000025517"


def load_scored_data() -> pd.DataFrame:
    candidates = [
        Path("data/processed/scored_transactions.pkl"),
        Path("data/processed/scored_transactions.parquet"),
    ]

    for path in candidates:
        if not path.exists():
            continue

        if path.suffix == ".pkl":
            return pd.read_pickle(path)

        return pd.read_parquet(path)

    raise FileNotFoundError(
        "Could not find scored_transactions.pkl or "
        "scored_transactions.parquet."
    )


def first_existing(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    return next(
        (column for column in candidates if column in frame.columns),
        None,
    )


df = load_scored_data()

score_column = first_existing(
    df,
    ["graph_mule", "graph_mule_score"],
)

label_column = first_existing(
    df,
    ["is_mule_directed", "mule_label"],
)

time_column = first_existing(
    df,
    ["timestamp", "transaction_timestamp", "event_time"],
)

if score_column is None:
    raise KeyError("No graph-mule score column was found.")

if label_column is None:
    raise KeyError("No mule label column was found.")

target = df[
    df["transaction_id"].astype(str).eq(TRANSACTION_ID)
].copy()

if target.empty:
    raise ValueError(
        f"Transaction {TRANSACTION_ID} was not found."
    )

payee_id = str(target.iloc[0]["payee_id"])

print("\nTARGET TRANSACTION")
print("-" * 60)

relevant_terms = [
    "graph",
    "mule",
    "tgn",
    "flow",
    "fan_",
    "complaint",
    "merchant",
    "scenario",
    "profile",
    "archetype",
    "timestamp",
    "payee",
]

target_columns = [
    column
    for column in df.columns
    if any(term in column.lower() for term in relevant_terms)
]

print(target[target_columns].T.to_string())

print("\nRECIPIENT HISTORY")
print("-" * 60)

history = df[
    df["payee_id"].astype(str).eq(payee_id)
].copy()

if time_column is not None:
    history[time_column] = pd.to_datetime(
        history[time_column],
        errors="coerce",
    )
    history = history.sort_values(time_column)

history_columns = [
    column
    for column in [
        time_column,
        "transaction_id",
        "scenario",
        "is_fraud",
        label_column,
        score_column,
        "graph_anomaly",
        "graph_merchant_legitimacy",
        "complaint_intelligence",
        "fan_in_1h",
        "fan_out_1h",
        "rapid_outflow_ratio",
        "sequence_history_size",
    ]
    if column is not None and column in history.columns
]

print(history[history_columns].tail(30).to_string(index=False))

# Prefer validation data for threshold diagnosis.
evaluation = df.copy()
scope_name = "all scored rows"

if "split" in df.columns:
    validation_mask = (
        df["split"].astype(str).str.lower() == "validation"
    )

    if validation_mask.any():
        evaluation = df.loc[validation_mask].copy()
        scope_name = "validation split"

evaluation = evaluation[
    [label_column, score_column]
].dropna()

evaluation[label_column] = (
    evaluation[label_column].astype(int)
)

evaluation[score_column] = (
    evaluation[score_column].astype(float)
)

print(f"\nGRAPH-MULE PERFORMANCE ON {scope_name.upper()}")
print("-" * 60)

if evaluation[label_column].nunique() < 2:
    print("Both positive and negative labels are required.")
    raise SystemExit(0)

y_true = evaluation[label_column].to_numpy()
scores = evaluation[score_column].to_numpy()

pr_auc = average_precision_score(y_true, scores)
roc_auc = roc_auc_score(y_true, scores)

print(f"Rows: {len(evaluation)}")
print(f"Mule prevalence: {y_true.mean():.6f}")
print(f"Graph-mule PR-AUC: {pr_auc:.6f}")
print(f"Graph-mule ROC-AUC: {roc_auc:.6f}")

for label, name in [(0, "Non-mule"), (1, "Mule")]:
    values = scores[y_true == label]

    print(
        f"\n{name} scores:",
        {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        },
    )

precision, recall, thresholds = precision_recall_curve(
    y_true,
    scores,
)

f1 = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
)

best_index = int(np.nanargmax(f1))

print("\nBest validation F1 threshold")
print(f"Threshold: {thresholds[best_index]:.6f}")
print(f"Precision: {precision[best_index]:.6f}")
print(f"Recall: {recall[best_index]:.6f}")
print(f"F1: {f1[best_index]:.6f}")

high_precision_indices = np.where(
    precision[:-1] >= 0.90
)[0]

if len(high_precision_indices):
    chosen_index = high_precision_indices[
        np.argmax(recall[high_precision_indices])
    ]

    print("\nThreshold providing at least 90% precision")
    print(f"Threshold: {thresholds[chosen_index]:.6f}")
    print(f"Precision: {precision[chosen_index]:.6f}")
    print(f"Recall: {recall[chosen_index]:.6f}")
else:
    print(
        "\nNo threshold achieved at least 90% precision."
    )

for threshold in [0.20, 0.25, 0.30, 0.50, 0.70]:
    predicted = scores >= threshold

    true_positive = int(
        np.sum(predicted & (y_true == 1))
    )
    false_positive = int(
        np.sum(predicted & (y_true == 0))
    )
    false_negative = int(
        np.sum((~predicted) & (y_true == 1))
    )

    precision_at_threshold = (
        true_positive
        / max(true_positive + false_positive, 1)
    )

    recall_at_threshold = (
        true_positive
        / max(true_positive + false_negative, 1)
    )

    print(
        f"\nThreshold {threshold:.2f}: "
        f"precision={precision_at_threshold:.4f}, "
        f"recall={recall_at_threshold:.4f}, "
        f"TP={true_positive}, "
        f"FP={false_positive}, "
        f"FN={false_negative}"
    )