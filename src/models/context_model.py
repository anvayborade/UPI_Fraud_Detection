from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.common import load_feature_list, numeric_matrix


@dataclass
class LegitimateNoveltyModel:
    model: LGBMClassifier
    features: list[str]

    @classmethod
    def create(cls, features: list[str] | None = None, seed: int = 42) -> "LegitimateNoveltyModel":
        model = LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=24,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        return cls(model=model, features=features or load_feature_list("context_features"))

    @staticmethod
    def _novel_subset(df: pd.DataFrame) -> pd.DataFrame:
        mask = df.get("novel_transaction_flag", 0).astype(int) == 1
        subset = df.loc[mask].copy()
        # The fallback keeps training possible on tiny custom datasets.
        return subset if len(subset) >= 50 else df.copy()

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float]:
        train_novel = self._novel_subset(train)
        val_novel = self._novel_subset(validation)
        x_train = numeric_matrix(train_novel, self.features)
        x_val = numeric_matrix(val_novel, self.features)
        y_train = train_novel["is_legitimate_novelty"].astype(int)
        y_val = val_novel["is_legitimate_novelty"].astype(int)
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="average_precision",
            callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
        )
        pred = self.model.predict_proba(x_val)[:, 1]
        metrics = {"pr_auc": float(average_precision_score(y_val, pred))}
        if y_val.nunique() > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_val, pred))
        return metrics

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(numeric_matrix(rows, self.features))[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LegitimateNoveltyModel":
        payload = joblib.load(path)
        return cls(model=payload["model"], features=payload["features"])
