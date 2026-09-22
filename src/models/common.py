from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def load_feature_list(section: str, path: str | Path = "configs/features.yaml") -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return list(config[section])


def numeric_matrix(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col not in df.columns:
            out[col] = 0.0
            continue
        series = df[col]
        if series.dtype == bool:
            out[col] = series.astype(float)
        elif pd.api.types.is_numeric_dtype(series):
            out[col] = pd.to_numeric(series, errors="coerce").astype(float)
        else:
            out[col] = pd.factorize(series.astype(str))[0].astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
