from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.graph.build_views import build_and_save_all_views


def build_time_window_snapshot(
    transactions: pd.DataFrame,
    output_dir: str | Path,
    hours: int = 24,
) -> dict[str, Path]:
    ordered = transactions.sort_values("timestamp").copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    cutoff = ordered["timestamp"].max() - pd.Timedelta(hours=hours)
    snapshot = ordered.loc[ordered["timestamp"] >= cutoff].copy()
    return build_and_save_all_views(snapshot, output_dir)
