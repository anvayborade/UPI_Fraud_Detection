from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPERT_COLUMNS = [
    "rule_risk",
    "statistical_anomaly",
    "transaction_fraud_score",
    "legitimate_novelty_inverse",
    "device_session_risk",
    "sequence_account_takeover_score",
    "sequence_social_engineering_score",
    "graph_mule_score",
    "merchant_legitimacy_inverse",
    "graph_anomaly_score",
    "complaint_intelligence_score",
    "journey_risk",
]

GATING_COLUMNS = [
    "is_new_payee",
    "collect_request",
    "is_qr",
    "device_changed",
    "graph_neighbourhood_size",
    "recipient_archetype_code",
    "missing_feature_ratio",
    "graph_score_available",
    "sequence_score_available",
    "complaint_score_available",
    "recipient_history_available",
    "payer_history_available",
    "cache_coverage",
    "expert_conflict_score",
]

FRAUD_TYPE_COLUMNS = [
    "is_account_takeover",
    "is_social_engineering",
    "is_qr_deception",
    "is_collect_scam",
    "is_mule_directed",
    "is_transaction_splitting",
    "is_remote_access_fraud",
]


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts: int, gate_dim: int, num_labels: int, hidden_dim: int = 40) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, num_experts),
        )
        self.context = nn.Sequential(
            nn.Linear(num_experts + gate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.12),
        )
        self.overall_head = nn.Linear(hidden_dim + 1, 1)
        self.label_head = nn.Linear(hidden_dim + 1, num_labels)

    def forward(self, experts: torch.Tensor, gating: torch.Tensor):
        weights = torch.softmax(self.gate(gating), dim=-1)
        weighted_score = (weights * experts).sum(dim=-1, keepdim=True)
        context = self.context(torch.cat([experts, gating], dim=-1))
        combined = torch.cat([weighted_score, context], dim=-1)
        return self.overall_head(combined).squeeze(-1), self.label_head(combined), weights


@dataclass
class FusionBundle:
    model: MixtureOfExperts
    expert_columns: list[str]
    gating_columns: list[str]
    fraud_type_columns: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    fraud_type_thresholds: dict[str, float]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        metadata = {
            "expert_columns": self.expert_columns,
            "gating_columns": self.gating_columns,
            "fraud_type_columns": self.fraud_type_columns,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "fraud_type_thresholds": self.fraud_type_thresholds,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "FusionBundle":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model = MixtureOfExperts(
            len(metadata["expert_columns"]),
            len(metadata["gating_columns"]),
            len(metadata["fraud_type_columns"]),
        )
        model.load_state_dict(torch.load(directory / "model.pt", map_location=device))
        model.to(device).eval()
        return cls(
            model=model,
            expert_columns=metadata["expert_columns"],
            gating_columns=metadata["gating_columns"],
            fraud_type_columns=metadata["fraud_type_columns"],
            feature_mean=np.asarray(metadata["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(metadata["feature_std"], dtype=np.float32),
            fraud_type_thresholds={
                str(key): float(value)
                for key, value in metadata.get("fraud_type_thresholds", {}).items()
            }
            or {column: 0.5 for column in metadata["fraud_type_columns"]},
        )


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = np.column_stack(
        [
            pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy(np.float32)
            if col in frame.columns
            else np.zeros(len(frame), dtype=np.float32)
            for col in columns
        ]
    )
    return values


def add_fusion_runtime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add availability and disagreement fields used by the gating network."""
    out = frame.copy()

    def numeric_series(name: str, default: float = 0.0) -> pd.Series:
        if name not in out.columns:
            return pd.Series(default, index=out.index, dtype=float)
        return pd.to_numeric(out[name], errors="coerce").fillna(default)

    if "merchant_legitimacy_inverse" not in out:
        out["merchant_legitimacy_inverse"] = 1 - numeric_series("graph_merchant_score")

    defaults = {
        "graph_score_available": (numeric_series("graph_neighbourhood_size") > 0).astype(float),
        "sequence_score_available": (numeric_series("sequence_history_size", 1) > 0).astype(float),
        "complaint_score_available": (numeric_series("complaint_intelligence_score") > 0).astype(float),
        "recipient_history_available": (numeric_series("graph_neighbourhood_size") > 0).astype(float),
        "payer_history_available": (numeric_series("txn_count_1h") > 0).astype(float),
    }
    for column, values in defaults.items():
        if column not in out:
            out[column] = values

    availability = out[
        [
            "graph_score_available",
            "sequence_score_available",
            "complaint_score_available",
            "recipient_history_available",
            "payer_history_available",
        ]
    ].astype(float)
    out["cache_coverage"] = availability.mean(axis=1)
    expert_values = _matrix(out, EXPERT_COLUMNS)
    out["expert_conflict_score"] = np.std(expert_values, axis=1)
    return out


def augment_fusion_training(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Teach fusion to handle missing and contradictory specialist evidence."""
    rng = np.random.default_rng(seed)
    base = add_fusion_runtime_columns(frame)
    augmented = [base]

    # Cold-start and partial-cache examples.
    masked = base.copy()
    graph_mask = rng.random(len(masked)) < 0.28
    sequence_mask = rng.random(len(masked)) < 0.24
    complaint_mask = rng.random(len(masked)) < 0.22
    recipient_mask = rng.random(len(masked)) < 0.15
    payer_mask = rng.random(len(masked)) < 0.12

    masked.loc[graph_mask, ["graph_mule_score", "merchant_legitimacy_inverse", "graph_anomaly_score"]] = 0.0
    masked.loc[graph_mask, "graph_score_available"] = 0.0
    masked.loc[sequence_mask, ["sequence_account_takeover_score", "sequence_social_engineering_score"]] = 0.0
    masked.loc[sequence_mask, "sequence_score_available"] = 0.0
    masked.loc[complaint_mask, "complaint_intelligence_score"] = 0.0
    masked.loc[complaint_mask, "complaint_score_available"] = 0.0
    masked.loc[recipient_mask, "recipient_history_available"] = 0.0
    masked.loc[payer_mask, "payer_history_available"] = 0.0
    masked = add_fusion_runtime_columns(masked)
    augmented.append(masked)

    # Contradictory evidence examples.  These retain the true target and force the
    # fusion model to use combinations instead of one dominant specialist.
    contradiction = base.copy()
    legitimate = contradiction["is_fraud"].eq(0).to_numpy()
    fraud = contradiction["is_fraud"].eq(1).to_numpy()
    merchant_like = contradiction.get("graph_merchant_score", pd.Series(0, index=contradiction.index)).to_numpy(float) > 0.65
    mule_label = contradiction.get("is_mule_directed", pd.Series(0, index=contradiction.index)).to_numpy(float) > 0
    social_label = contradiction.get("is_social_engineering", pd.Series(0, index=contradiction.index)).to_numpy(float) > 0

    noisy_legitimate = legitimate & merchant_like & (rng.random(len(contradiction)) < 0.38)
    contradiction.loc[noisy_legitimate, "graph_mule_score"] = rng.uniform(0.55, 0.88, noisy_legitimate.sum())
    contradiction.loc[noisy_legitimate, "graph_anomaly_score"] = rng.uniform(0.45, 0.95, noisy_legitimate.sum())
    contradiction.loc[noisy_legitimate, "complaint_intelligence_score"] = rng.uniform(0.10, 0.65, noisy_legitimate.sum())

    disguised_mules = fraud & mule_label & (rng.random(len(contradiction)) < 0.65)
    contradiction.loc[disguised_mules, "merchant_legitimacy_inverse"] = rng.uniform(0.02, 0.30, disguised_mules.sum())
    contradiction.loc[disguised_mules, "journey_risk"] = rng.uniform(0.01, 0.25, disguised_mules.sum())
    contradiction.loc[disguised_mules, "sequence_social_engineering_score"] = rng.uniform(0.0, 0.30, disguised_mules.sum())

    unreported_social = fraud & social_label & (rng.random(len(contradiction)) < 0.48)
    contradiction.loc[unreported_social, "complaint_intelligence_score"] = rng.uniform(0.0, 0.18, unreported_social.sum())
    contradiction.loc[unreported_social, "graph_mule_score"] = rng.uniform(0.0, 0.35, unreported_social.sum())

    contradiction = add_fusion_runtime_columns(contradiction)
    augmented.append(contradiction)
    return pd.concat(augmented, ignore_index=True)


def _positive_weight(target: np.ndarray, maximum: float = 25.0) -> float:
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    if positives <= 0:
        return 1.0
    return float(np.clip(negatives / positives, 1.0, maximum))


def train_fusion_model(
    frame: pd.DataFrame,
    output_dir: str | Path,
    epochs: int = 25,
    seed: int = 42,
) -> FusionBundle:
    torch.manual_seed(seed)
    training_frame = augment_fusion_training(frame, seed=seed)
    experts = _matrix(training_frame, EXPERT_COLUMNS)
    gating = _matrix(training_frame, GATING_COLUMNS)
    combined = np.concatenate([experts, gating], axis=1)
    mean, std = combined.mean(axis=0), combined.std(axis=0) + 1e-6
    standardised = (combined - mean) / std
    experts_std = standardised[:, : len(EXPERT_COLUMNS)]
    gating_std = standardised[:, len(EXPERT_COLUMNS) :]
    overall = training_frame["is_fraud"].to_numpy(np.float32)
    labels = _matrix(training_frame, FRAUD_TYPE_COLUMNS)

    dataset = TensorDataset(
        torch.tensor(experts_std),
        torch.tensor(gating_std),
        torch.tensor(overall),
        torch.tensor(labels),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixtureOfExperts(len(EXPERT_COLUMNS), len(GATING_COLUMNS), len(FRAUD_TYPE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    overall_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(_positive_weight(overall), device=device)
    )
    label_weights = torch.tensor(
        [_positive_weight(labels[:, idx]) for idx in range(labels.shape[1])],
        dtype=torch.float32,
        device=device,
    )
    label_loss = nn.BCEWithLogitsLoss(pos_weight=label_weights)
    model.train()
    for _ in range(epochs):
        for expert_batch, gate_batch, y_batch, labels_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            overall_logits, label_logits, _ = model(expert_batch.to(device), gate_batch.to(device))
            loss = overall_loss(overall_logits, y_batch.to(device)) + 0.70 * label_loss(
                label_logits, labels_batch.to(device)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    bundle = FusionBundle(
        model=model.eval(),
        expert_columns=EXPERT_COLUMNS,
        gating_columns=GATING_COLUMNS,
        fraud_type_columns=FRAUD_TYPE_COLUMNS,
        feature_mean=mean,
        feature_std=std,
        fraud_type_thresholds={column: 0.5 for column in FRAUD_TYPE_COLUMNS},
    )
    bundle.save(output_dir)
    return bundle


@torch.no_grad()
def predict_fusion(bundle: FusionBundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared = add_fusion_runtime_columns(frame)
    experts = _matrix(prepared, bundle.expert_columns)
    gating = _matrix(prepared, bundle.gating_columns)
    combined = np.concatenate([experts, gating], axis=1)
    standardised = (combined - bundle.feature_mean) / bundle.feature_std
    experts_tensor = torch.tensor(standardised[:, : len(bundle.expert_columns)], dtype=torch.float32)
    gating_tensor = torch.tensor(standardised[:, len(bundle.expert_columns) :], dtype=torch.float32)
    device = next(bundle.model.parameters()).device
    overall, labels, weights = bundle.model(experts_tensor.to(device), gating_tensor.to(device))
    return (
        torch.sigmoid(overall).cpu().numpy(),
        torch.sigmoid(labels).cpu().numpy(),
        weights.cpu().numpy(),
    )


def calibrate_fraud_type_thresholds(
    bundle: FusionBundle,
    validation_frame: pd.DataFrame,
    *,
    minimum_threshold: float = 0.15,
    maximum_threshold: float = 0.85,
) -> dict[str, float]:
    """Choose one validation F1 threshold for every fraud type."""
    _, probabilities, _ = predict_fusion(bundle, validation_frame)
    thresholds: dict[str, float] = {}
    grid = np.linspace(minimum_threshold, maximum_threshold, 71)
    for idx, column in enumerate(bundle.fraud_type_columns):
        target = pd.to_numeric(validation_frame[column], errors="coerce").fillna(0).to_numpy(int)
        if target.sum() == 0:
            thresholds[column] = 0.50
            continue
        scores = probabilities[:, idx]
        best_threshold = 0.50
        best_f1 = -1.0
        for threshold in grid:
            value = f1_score(target, scores >= threshold, zero_division=0)
            if value > best_f1:
                best_f1 = value
                best_threshold = float(threshold)
        thresholds[column] = best_threshold
    bundle.fraud_type_thresholds = thresholds
    return thresholds
