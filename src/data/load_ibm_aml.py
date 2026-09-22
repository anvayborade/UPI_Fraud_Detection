from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.canonical_schema import ensure_canonical_schema


def _resolve_column(columns: list[str], *candidates: str) -> str:
    lookup = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"None of the expected columns {candidates!r} were found. Available: {columns}")


def _read_sampled_ibm(
    path: Path,
    *,
    negative_ratio: int = 10,
    max_negative_rows: int = 120_000,
    chunksize: int = 250_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Keep every laundering row and a reproducible reservoir of negatives.

    IBM Small files still contain millions of rows. Keeping all positive rows and
    a controlled negative sample is practical for a laptop prototype and preserves
    the rare-class examples required by the graph models.
    """
    positives: list[pd.DataFrame] = []
    negative_parts: list[pd.DataFrame] = []
    total_positive = 0
    rng = np.random.default_rng(seed)

    for chunk_index, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        label_col = _resolve_column(list(chunk.columns), "Is Laundering", "is_laundering")
        labels = pd.to_numeric(chunk[label_col], errors="coerce").fillna(0).astype(int)
        fraud = chunk.loc[labels == 1].copy()
        legitimate = chunk.loc[labels == 0].copy()
        if len(fraud):
            positives.append(fraud)
            total_positive += len(fraud)

        # Sample enough negatives from every chunk so time coverage is retained.
        per_chunk = min(len(legitimate), max(2_000, int(max_negative_rows / 30)))
        if per_chunk:
            negative_parts.append(
                legitimate.sample(n=per_chunk, random_state=int(rng.integers(0, 2**31 - 1)))
            )

    positive_frame = pd.concat(positives, ignore_index=True) if positives else pd.DataFrame()
    negative_frame = pd.concat(negative_parts, ignore_index=True) if negative_parts else pd.DataFrame()
    desired_negatives = min(max_negative_rows, max(total_positive * negative_ratio, 10_000))
    if len(negative_frame) > desired_negatives:
        negative_frame = negative_frame.sample(n=desired_negatives, random_state=seed)
    return pd.concat([positive_frame, negative_frame], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_ibm_aml_canonical(
    path: Path,
    *,
    partition: str,
    negative_ratio: int = 10,
    max_negative_rows: int = 120_000,
    seed: int = 42,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _read_sampled_ibm(
        path,
        negative_ratio=negative_ratio,
        max_negative_rows=max_negative_rows,
        seed=seed,
    )
    columns = list(raw.columns)
    timestamp_col = _resolve_column(columns, "Timestamp")
    from_bank_col = _resolve_column(columns, "From Bank")
    from_account_col = _resolve_column(columns, "Account")
    to_bank_col = _resolve_column(columns, "To Bank")
    to_account_col = _resolve_column(columns, "Account.1", "To Account")
    amount_col = _resolve_column(columns, "Amount Paid", "Amount Received")
    currency_col = _resolve_column(columns, "Payment Currency", "Receiving Currency")
    format_col = _resolve_column(columns, "Payment Format")
    label_col = _resolve_column(columns, "Is Laundering", "is_laundering")

    timestamp = pd.to_datetime(raw[timestamp_col], utc=True, errors="coerce")
    # Some releases use a format that pandas cannot infer in a mixed column.
    if timestamp.isna().mean() > 0.01:
        timestamp = pd.to_datetime(raw[timestamp_col], format="mixed", utc=True, errors="coerce")

    prefix = f"IBM_{partition.upper()}"
    payer = prefix + "_B" + raw[from_bank_col].astype(str) + "_A" + raw[from_account_col].astype(str)
    payee = prefix + "_B" + raw[to_bank_col].astype(str) + "_A" + raw[to_account_col].astype(str)
    out = pd.DataFrame(
        {
            "transaction_id": [f"{prefix}_TXN_{idx:012d}" for idx in range(len(raw))],
            "timestamp": timestamp,
            "payer_id": payer,
            "payee_id": payee,
            "amount": pd.to_numeric(raw[amount_col], errors="coerce"),
            "currency": raw[currency_col].astype(str),
            "payment_channel": raw[format_col].astype(str),
            "merchant_category": raw[format_col].astype(str),
            "source_dataset": "ibm_aml",
            "source_partition": partition.lower(),
            "transaction_fraud_label": pd.Series(pd.NA, index=raw.index, dtype="Int64"),
            "laundering_label": pd.to_numeric(raw[label_col], errors="coerce"),
        }
    )
    return ensure_canonical_schema(out, source_name=f"ibm_aml_{partition}")
