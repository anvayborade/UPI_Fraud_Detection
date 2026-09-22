from __future__ import annotations

import pandas as pd

from src.features.behavioural_features import add_behavioural_features
from src.geospatial.h3_encoder import haversine_km, latlon_to_cell


def test_behavioural_features_use_prior_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T10:00:00Z", "2026-01-01T10:02:00Z", "2026-01-01T10:04:00Z"]
            ),
            "payer_id": ["U1", "U1", "U1"],
            "payee_id": ["A1", "A2", "A1"],
            "amount": [100.0, 200.0, 300.0],
        }
    )
    result = add_behavioural_features(frame)
    assert result.loc[0, "txn_count_5m"] == 0
    assert result.loc[1, "txn_count_5m"] == 1
    assert result.loc[2, "txn_count_5m"] == 2
    assert result.loc[2, "unique_payees_1h"] == 2
    assert result.loc[2, "amount_sum_1h"] == 300.0


def test_geospatial_helpers() -> None:
    assert haversine_km(19.076, 72.878, 19.076, 72.878) == 0.0
    assert isinstance(latlon_to_cell(19.076, 72.878), str)
