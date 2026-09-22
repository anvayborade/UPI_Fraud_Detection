from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class GraphAnomalyModel:
    scaler: StandardScaler
    model: IsolationForest

    @classmethod
    def fit(cls, embeddings: np.ndarray, seed: int = 42) -> "GraphAnomalyModel":
        scaler = StandardScaler().fit(embeddings)
        transformed = scaler.transform(embeddings)
        model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1).fit(transformed)
        return cls(scaler=scaler, model=model)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        raw = -self.model.decision_function(self.scaler.transform(embeddings))
        low, high = np.percentile(raw, [1, 99])
        return np.clip((raw - low) / (high - low + 1e-9), 0, 1)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "GraphAnomalyModel":
        return joblib.load(path)
