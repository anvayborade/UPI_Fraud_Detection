from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

RAW_COLUMNS = [
    "log_amount", "is_qr", "is_collect", "recipient_complaint_score",
    "rapid_outflow_ratio", "location_continuity",
]


class TemporalNodeDataset(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class HeterogeneousTemporalGraphEncoder(nn.Module):
    """A compact TGN-style typed temporal memory encoder.

    Each destination account receives a chronological sequence of typed edge
    messages. Relation embeddings, numeric message features and time-gap encoding
    are fused and passed through a GRU memory. This is deliberately smaller than a
    distributed production TGN, but it preserves the core event-memory principle.
    """

    def __init__(self, raw_dim: int, num_relations: int, hidden_dim: int = 64, relation_dim: int = 16) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, relation_dim)
        self.raw_projection = nn.Sequential(nn.Linear(raw_dim + 2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.message_projection = nn.Linear(hidden_dim + relation_dim, hidden_dim)
        self.memory = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mule_head = nn.Linear(hidden_dim, 1)
        self.rapid_outflow_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        raw_messages: torch.Tensor,
        relation_ids: torch.Tensor,
        delta_seconds: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_delta = torch.log1p(delta_seconds.clamp_min(0)).unsqueeze(-1)
        time_features = torch.cat([torch.sin(log_delta), torch.cos(log_delta)], dim=-1)
        raw = self.raw_projection(torch.cat([raw_messages, time_features], dim=-1))
        relation = self.relation_embedding(relation_ids)
        messages = torch.relu(self.message_projection(torch.cat([raw, relation], dim=-1)))
        lengths = mask.sum(dim=1).long().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(messages, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.memory(packed)
        embedding = hidden[-1]
        return self.mule_head(embedding).squeeze(-1), self.rapid_outflow_head(embedding).squeeze(-1), embedding


@dataclass
class TemporalGraphBundle:
    model: HeterogeneousTemporalGraphEncoder
    max_events: int
    raw_mean: np.ndarray
    raw_std: np.ndarray

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        metadata = {
            "max_events": self.max_events,
            "raw_mean": self.raw_mean.tolist(),
            "raw_std": self.raw_std.tolist(),
            "hidden_dim": self.model.memory.hidden_size,
            "num_relations": self.model.relation_embedding.num_embeddings,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "TemporalGraphBundle":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model = HeterogeneousTemporalGraphEncoder(
            raw_dim=len(RAW_COLUMNS),
            num_relations=int(metadata["num_relations"]),
            hidden_dim=int(metadata["hidden_dim"]),
        )
        model.load_state_dict(torch.load(directory / "model.pt", map_location=device))
        model.to(device).eval()
        return cls(
            model=model,
            max_events=int(metadata["max_events"]),
            raw_mean=np.asarray(metadata["raw_mean"], dtype=np.float32),
            raw_std=np.asarray(metadata["raw_std"], dtype=np.float32),
        )


def _build_samples(events: pd.DataFrame, max_events: int, mean: np.ndarray, std: np.ndarray) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
    samples: list[dict[str, torch.Tensor]] = []
    node_ids: list[int] = []
    for node_id, group in events.groupby("dst", sort=False):
        g = group.sort_values("timestamp").tail(max_events)
        length = len(g)
        raw = np.zeros((max_events, len(RAW_COLUMNS)), dtype=np.float32)
        relations = np.zeros(max_events, dtype=np.int64)
        deltas = np.zeros(max_events, dtype=np.float32)
        mask = np.zeros(max_events, dtype=np.bool_)
        raw_values = g[RAW_COLUMNS].to_numpy(np.float32)
        raw[:length] = (raw_values - mean) / np.where(std < 1e-6, 1.0, std)
        relations[:length] = g["relation_id"].to_numpy(np.int64)
        timestamp = g["timestamp"].to_numpy(np.float64)
        deltas[:length] = np.diff(timestamp, prepend=timestamp[0]).astype(np.float32)
        mask[:length] = True
        samples.append(
            {
                "raw": torch.tensor(raw),
                "relation": torch.tensor(relations),
                "delta": torch.tensor(deltas),
                "mask": torch.tensor(mask),
                "mule_label": torch.tensor(float(g["label_mule"].max())),
                "rapid_label": torch.tensor(float(g["rapid_outflow_ratio"].max() >= 0.65)),
            }
        )
        node_ids.append(int(node_id))
    return samples, node_ids


def train_hetero_tgn(
    events: pd.DataFrame,
    output_dir: str | Path,
    max_events: int = 40,
    epochs: int = 10,
    seed: int = 42,
) -> TemporalGraphBundle:
    torch.manual_seed(seed)
    raw_values = events[RAW_COLUMNS].to_numpy(np.float32)
    mean, std = raw_values.mean(axis=0), raw_values.std(axis=0) + 1e-6
    samples, _ = _build_samples(events, max_events, mean, std)
    loader = DataLoader(TemporalNodeDataset(samples), batch_size=128, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeterogeneousTemporalGraphEncoder(len(RAW_COLUMNS), int(events["relation_id"].max()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            mule, rapid, _ = model(
                batch["raw"].to(device),
                batch["relation"].to(device),
                batch["delta"].to(device),
                batch["mask"].to(device),
            )
            loss = loss_fn(mule, batch["mule_label"].to(device)) + 0.5 * loss_fn(rapid, batch["rapid_label"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    bundle = TemporalGraphBundle(model=model.eval(), max_events=max_events, raw_mean=mean, raw_std=std)
    bundle.save(output_dir)
    return bundle


@torch.no_grad()
def score_temporal_nodes(bundle: TemporalGraphBundle, events: pd.DataFrame) -> pd.DataFrame:
    samples, node_ids = _build_samples(events, bundle.max_events, bundle.raw_mean, bundle.raw_std)
    loader = DataLoader(TemporalNodeDataset(samples), batch_size=256, shuffle=False)
    device = next(bundle.model.parameters()).device
    rows: list[dict] = []
    offset = 0
    for batch in loader:
        mule, rapid, embeddings = bundle.model(
            batch["raw"].to(device), batch["relation"].to(device), batch["delta"].to(device), batch["mask"].to(device)
        )
        for idx in range(len(mule)):
            rows.append(
                {
                    "global_node_id": node_ids[offset + idx],
                    "tgn_mule_score": float(torch.sigmoid(mule[idx]).cpu()),
                    "tgn_rapid_outflow_score": float(torch.sigmoid(rapid[idx]).cpu()),
                    "tgn_embedding": embeddings[idx].cpu().numpy().astype(np.float32),
                }
            )
        offset += len(mule)
    return pd.DataFrame(rows)

@torch.no_grad()
def score_temporal_transactions(
    bundle: TemporalGraphBundle,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Score every transaction from the destination's strictly prior history.

    The current event is appended only after its score sample has been created.  This
    mirrors pre-authorisation inference and prevents a payment from describing its
    own future graph state.
    """
    from collections import defaultdict, deque

    ordered = events.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    histories: dict[int, deque[dict[str, float | int]]] = defaultdict(
        lambda: deque(maxlen=bundle.max_events)
    )
    samples: list[dict[str, torch.Tensor]] = []
    transaction_ids: list[str] = []
    history_sizes: list[int] = []

    for record in ordered.itertuples(index=False):
        dst = int(record.dst)
        history = list(histories[dst])
        raw = np.zeros((bundle.max_events, len(RAW_COLUMNS)), dtype=np.float32)
        relations = np.zeros(bundle.max_events, dtype=np.int64)
        deltas = np.zeros(bundle.max_events, dtype=np.float32)
        mask = np.zeros(bundle.max_events, dtype=np.bool_)

        if history:
            selected = history[-bundle.max_events :]
            length = len(selected)
            raw_values = np.asarray(
                [[float(item[column]) for column in RAW_COLUMNS] for item in selected],
                dtype=np.float32,
            )
            raw[:length] = (raw_values - bundle.raw_mean) / np.where(
                bundle.raw_std < 1e-6, 1.0, bundle.raw_std
            )
            relations[:length] = np.asarray(
                [int(item["relation_id"]) for item in selected], dtype=np.int64
            )
            timestamps = np.asarray(
                [float(item["timestamp"]) for item in selected], dtype=np.float64
            )
            deltas[:length] = np.diff(timestamps, prepend=timestamps[0]).astype(np.float32)
            mask[:length] = True
        else:
            # The model requires a length of at least one.  A masked zero message
            # represents a true cold-start recipient.
            length = 1
            mask[0] = True

        samples.append(
            {
                "raw": torch.tensor(raw),
                "relation": torch.tensor(relations),
                "delta": torch.tensor(deltas),
                "mask": torch.tensor(mask),
                "mule_label": torch.tensor(0.0),
                "rapid_label": torch.tensor(0.0),
            }
        )
        transaction_ids.append(str(record.transaction_id))
        history_sizes.append(len(history))

        current = {column: float(getattr(record, column)) for column in RAW_COLUMNS}
        current["relation_id"] = int(record.relation_id)
        current["timestamp"] = float(record.timestamp)
        histories[dst].append(current)

    loader = DataLoader(TemporalNodeDataset(samples), batch_size=256, shuffle=False)
    device = next(bundle.model.parameters()).device
    rows: list[dict[str, object]] = []
    offset = 0
    for batch in loader:
        mule, rapid, embeddings = bundle.model(
            batch["raw"].to(device),
            batch["relation"].to(device),
            batch["delta"].to(device),
            batch["mask"].to(device),
        )
        for idx in range(len(mule)):
            absolute = offset + idx
            rows.append(
                {
                    "transaction_id": transaction_ids[absolute],
                    "tgn_mule_score_prior": float(torch.sigmoid(mule[idx]).cpu()),
                    "tgn_rapid_outflow_score_prior": float(torch.sigmoid(rapid[idx]).cpu()),
                    "tgn_history_size": int(history_sizes[absolute]),
                    "tgn_embedding_prior": embeddings[idx].cpu().numpy().astype(np.float32),
                }
            )
        offset += len(mule)
    return pd.DataFrame(rows)
