from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


NUMERIC_EVENT_FEATURES = [
    "delta_seconds", "amount", "new_device", "new_payee", "collect_request",
    "qr_unverified", "remote_access", "location_inconsistent",
]


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[dict[str, torch.Tensor]]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.sequences[index]


class TimeAwareTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        numeric_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 40,
    ) -> None:
        super().__init__()
        self.event_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.numeric_projection = nn.Linear(numeric_dim, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.account_takeover_head = nn.Linear(d_model, 1)
        self.social_engineering_head = nn.Linear(d_model, 1)

    def forward(self, event_ids: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length = event_ids.shape
        positions = torch.arange(length, device=event_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.event_embedding(event_ids) + self.numeric_projection(numeric) + self.position_embedding(positions)
        encoded = self.encoder(x, src_key_padding_mask=~mask.bool())
        weights = mask.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = self.norm(pooled)
        return self.account_takeover_head(pooled).squeeze(-1), self.social_engineering_head(pooled).squeeze(-1), pooled


@dataclass
class SequenceModelBundle:
    model: TimeAwareTransformer
    vocabulary: dict[str, int]
    max_len: int
    numeric_mean: np.ndarray
    numeric_std: np.ndarray

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        metadata = {
            "vocabulary": self.vocabulary,
            "max_len": self.max_len,
            "numeric_mean": self.numeric_mean.tolist(),
            "numeric_std": self.numeric_std.tolist(),
            "d_model": self.model.event_embedding.embedding_dim,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "SequenceModelBundle":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model = TimeAwareTransformer(
            vocab_size=len(metadata["vocabulary"]),
            numeric_dim=len(NUMERIC_EVENT_FEATURES),
            d_model=int(metadata["d_model"]),
            max_len=int(metadata["max_len"]),
        )
        model.load_state_dict(torch.load(directory / "model.pt", map_location=device))
        model.to(device).eval()
        return cls(
            model=model,
            vocabulary=metadata["vocabulary"],
            max_len=int(metadata["max_len"]),
            numeric_mean=np.asarray(metadata["numeric_mean"], dtype=np.float32),
            numeric_std=np.asarray(metadata["numeric_std"], dtype=np.float32),
        )


def _build_vocabulary(events: pd.Series) -> dict[str, int]:
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for value in sorted(events.astype(str).unique()):
        vocabulary[value] = len(vocabulary)
    return vocabulary


def _normalise_numeric(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / np.where(std < 1e-6, 1.0, std)


def build_sequence_tensors(
    events: pd.DataFrame,
    vocabulary: dict[str, int],
    max_len: int,
    numeric_mean: np.ndarray,
    numeric_std: np.ndarray,
) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
    samples: list[dict[str, torch.Tensor]] = []
    transaction_ids: list[str] = []
    for transaction_id, group in events.groupby("transaction_id", sort=False):
        g = group.sort_values("event_index").tail(max_len)
        event_ids = np.zeros(max_len, dtype=np.int64)
        numeric = np.zeros((max_len, len(NUMERIC_EVENT_FEATURES)), dtype=np.float32)
        mask = np.zeros(max_len, dtype=np.bool_)
        length = len(g)
        event_ids[:length] = [vocabulary.get(str(v), 1) for v in g["event_type"]]
        raw_numeric = g[NUMERIC_EVENT_FEATURES].to_numpy(np.float32)
        raw_numeric[:, 1] = np.log1p(raw_numeric[:, 1])
        numeric[:length] = _normalise_numeric(raw_numeric, numeric_mean, numeric_std)
        mask[:length] = True
        samples.append(
            {
                "event_ids": torch.tensor(event_ids),
                "numeric": torch.tensor(numeric),
                "mask": torch.tensor(mask),
                "ato_label": torch.tensor(float(g["label_account_takeover"].max())),
                "social_label": torch.tensor(float(g["label_social_engineering"].max())),
            }
        )
        transaction_ids.append(str(transaction_id))
    return samples, transaction_ids


def train_time_transformer(
    events: pd.DataFrame,
    output_dir: str | Path,
    max_len: int = 40,
    epochs: int = 6,
    batch_size: int = 128,
    seed: int = 42,
) -> SequenceModelBundle:
    torch.manual_seed(seed)
    vocabulary = _build_vocabulary(events["event_type"])
    numeric_values = events[NUMERIC_EVENT_FEATURES].to_numpy(np.float32)
    numeric_values[:, 1] = np.log1p(numeric_values[:, 1])
    mean = numeric_values.mean(axis=0)
    std = numeric_values.std(axis=0) + 1e-6
    samples, _ = build_sequence_tensors(events, vocabulary, max_len, mean, std)
    loader = DataLoader(SequenceDataset(samples), batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimeAwareTransformer(len(vocabulary), len(NUMERIC_EVENT_FEATURES), max_len=max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            ato, social, _ = model(
                batch["event_ids"].to(device),
                batch["numeric"].to(device),
                batch["mask"].to(device),
            )
            loss = loss_fn(ato, batch["ato_label"].to(device)) + loss_fn(social, batch["social_label"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    bundle = SequenceModelBundle(model=model.eval(), vocabulary=vocabulary, max_len=max_len, numeric_mean=mean, numeric_std=std)
    bundle.save(output_dir)
    return bundle


@torch.no_grad()
def score_sequences(bundle: SequenceModelBundle, events: pd.DataFrame) -> pd.DataFrame:
    device = next(bundle.model.parameters()).device
    samples, transaction_ids = build_sequence_tensors(
        events, bundle.vocabulary, bundle.max_len, bundle.numeric_mean, bundle.numeric_std
    )
    sequence_lengths = (
        events.groupby("transaction_id", sort=False).size().reindex(transaction_ids).fillna(0).astype(int).tolist()
    )
    loader = DataLoader(SequenceDataset(samples), batch_size=256, shuffle=False)
    rows: list[dict] = []
    offset = 0
    for batch in loader:
        ato, social, embeddings = bundle.model(
            batch["event_ids"].to(device), batch["numeric"].to(device), batch["mask"].to(device)
        )
        size = len(ato)
        for idx in range(size):
            rows.append(
                {
                    "transaction_id": transaction_ids[offset + idx],
                    "sequence_account_takeover_score": float(torch.sigmoid(ato[idx]).cpu()),
                    "sequence_social_engineering_score": float(torch.sigmoid(social[idx]).cpu()),
                    "sequence_embedding": embeddings[idx].cpu().numpy().astype(np.float32),
                    "sequence_history_size": int(sequence_lengths[offset + idx]),
                }
            )
        offset += size
    return pd.DataFrame(rows)
