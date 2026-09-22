from __future__ import annotations

from pathlib import Path

import pandas as pd


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    i = int(n * train_ratio)
    j = int(n * (train_ratio + val_ratio))
    return ordered.iloc[:i].copy(), ordered.iloc[i:j].copy(), ordered.iloc[j:].copy()


def save_splits(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = temporal_split(df)
    paths = (
        output_dir / "train.parquet",
        output_dir / "validation.parquet",
        output_dir / "test.parquet",
    )
    for part, path in zip((train, val, test), paths, strict=True):
        part.to_parquet(path, index=False)
    return paths


def source_stratified_temporal_split(
    df: pd.DataFrame,
    source_column: str = "source_dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """Preserve time order inside every source while keeping all sources in each split."""
    if source_column not in df.columns:
        return temporal_split(df, train_ratio=train_ratio, val_ratio=val_ratio)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in df.groupby(source_column, observed=True, sort=False):
        train, val, test = temporal_split(group, train_ratio=train_ratio, val_ratio=val_ratio)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    return (
        pd.concat(train_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True),
        pd.concat(val_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True),
        pd.concat(test_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True),
    )


def save_source_stratified_splits(
    df: pd.DataFrame,
    output_dir: Path,
    source_column: str = "source_dataset",
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = source_stratified_temporal_split(df, source_column=source_column)
    paths = (
        output_dir / "train.parquet",
        output_dir / "validation.parquet",
        output_dir / "test.parquet",
    )
    for part, path in zip((train, val, test), paths, strict=True):
        part.to_parquet(path, index=False)
    return paths
