from __future__ import annotations

import pandas as pd

from src.models.fusion_model import add_fusion_runtime_columns


def test_fusion_runtime_columns_distinguish_missing_from_zero_risk() -> None:
    frame = pd.DataFrame(
        {
            "graph_mule_score": [0.0],
            "graph_merchant_score": [0.0],
            "graph_anomaly_score": [0.0],
            "sequence_account_takeover_score": [0.0],
            "sequence_social_engineering_score": [0.0],
            "complaint_intelligence_score": [0.0],
            "graph_neighbourhood_size": [0.0],
            "sequence_history_size": [0],
            "txn_count_1h": [0],
        }
    )
    result = add_fusion_runtime_columns(frame)
    assert result.loc[0, "graph_score_available"] == 0
    assert result.loc[0, "sequence_score_available"] == 0
    assert result.loc[0, "cache_coverage"] < 1
