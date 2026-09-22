from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score


def main() -> None:
    train = pd.read_parquet("data/processed/train.parquet")
    test = pd.read_parquet("data/processed/test.parquet")

    print("\nTransaction overlap:")
    overlap = set(train["transaction_id"]) & set(test["transaction_id"])
    print(f"Shared transaction IDs: {len(overlap)}")

    print("\nScenario and label relationship:")
    print(pd.crosstab(test["scenario"], test["is_fraud"]))

    with open("configs/features.yaml", encoding="utf-8") as file:
        feature_config = yaml.safe_load(file)

    results = []

    for feature in feature_config["transaction_features"]:
        if feature not in test.columns:
            continue

        values = pd.to_numeric(test[feature], errors="coerce")
        if values.nunique(dropna=True) < 2:
            continue

        values = values.fillna(values.median())
        auc = roc_auc_score(test["is_fraud"], values)

        # A negatively correlated feature can also be a perfect predictor.
        separation_auc = max(auc, 1 - auc)

        results.append((feature, separation_auc))

    results.sort(key=lambda item: item[1], reverse=True)

    print("\nStrongest single-feature fraud predictors:")
    for feature, auc in results[:20]:
        print(f"{feature:45s} {auc:.4f}")


if __name__ == "__main__":
    main()