from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split_data import temporal_split
from src.features.behavioural_features import add_behavioural_features
from src.features.context_features import add_context_features
from src.features.graph_features import add_peer_normalised_graph_features
from src.features.transaction_features import add_basic_transaction_features
from src.models.context_model import LegitimateNoveltyModel
from src.models.lightgbm_model import TransactionRiskModel
from src.simulation.generate_base import generate_base_transactions
from src.simulation.generate_scenarios import inject_upi_scenarios


def main() -> None:
    frame = generate_base_transactions(5000, seed=42)
    frame = inject_upi_scenarios(frame, seed=42)
    frame = add_basic_transaction_features(frame)
    frame = add_behavioural_features(frame)
    frame = add_context_features(frame)
    frame = add_peer_normalised_graph_features(frame)
    train, validation, test = temporal_split(frame)

    transaction = TransactionRiskModel.create(seed=42)
    transaction_metrics = transaction.fit(train, validation)
    context = LegitimateNoveltyModel.create(seed=42)
    context_metrics = context.fit(train, validation)

    auto = test[test["scenario"] == "auto_long_distance_legitimate"].head(1)
    scam = test[test["scenario"] == "fake_refund_qr"].head(1)
    examples = auto if scam.empty else (scam if auto.empty else __import__("pandas").concat([auto, scam]))
    examples = examples.copy()
    examples["transaction_score"] = transaction.predict_proba(examples)
    examples["legitimate_context_score"] = context.predict_proba(examples)

    print("Transaction model:", transaction_metrics)
    print("Context model:", context_metrics)
    print(examples[["scenario", "amount", "transaction_score", "legitimate_context_score"]].to_string(index=False))
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
