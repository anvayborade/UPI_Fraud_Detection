from __future__ import annotations

import numpy as np


def expert_disagreement(scores: np.ndarray) -> np.ndarray:
    """Return normalised per-row dispersion across expert probability scores."""
    return np.clip(np.std(scores, axis=1) / 0.5, 0, 1)


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def calculate_uncertainty(
    expert_scores: np.ndarray,
    final_probability: np.ndarray,
    missing_feature_ratio: np.ndarray | float = 0.0,
    graph_sparsity: np.ndarray | float = 0.0,
    anomaly_score: np.ndarray | float = 0.0,
) -> np.ndarray:
    disagreement = expert_disagreement(expert_scores)
    entropy = binary_entropy(final_probability)
    return np.clip(
        0.35 * disagreement
        + 0.25 * entropy
        + 0.15 * np.asarray(missing_feature_ratio)
        + 0.10 * np.asarray(graph_sparsity)
        + 0.15 * np.asarray(anomaly_score),
        0,
        1,
    )
