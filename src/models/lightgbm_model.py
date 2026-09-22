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
class TransactionRiskModel:
    model: LGBMClassifier
    features: list[str]

    @classmethod
    def create(cls, features: list[str] | None = None, seed: int = 42) -> "TransactionRiskModel":
        model = LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        return cls(model=model, features=features or load_feature_list("transaction_features"))

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float]:
        x_train = numeric_matrix(train, self.features)
        x_val = numeric_matrix(validation, self.features)
        y_train = train["is_fraud"].astype(int)
        y_val = validation["is_fraud"].astype(int)
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="average_precision",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        pred = self.model.predict_proba(x_val)[:, 1]
        return {
            "pr_auc": float(average_precision_score(y_val, pred)),
            "roc_auc": float(roc_auc_score(y_val, pred)),
        }

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(numeric_matrix(rows, self.features))[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransactionRiskModel":
        payload = joblib.load(path)
        return cls(model=payload["model"], features=payload["features"])
