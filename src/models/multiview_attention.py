from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MultiViewGraphAttention(nn.Module):
    def __init__(self, embedding_dim: int = 32, query_dim: int = 8, hidden_dim: int = 32, num_views: int = 3) -> None:
        super().__init__()
        self.num_views = num_views
        self.view_projection = nn.ModuleList([nn.Linear(embedding_dim, hidden_dim) for _ in range(num_views)])
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.mule_head = nn.Linear(hidden_dim, 1)
        self.merchant_head = nn.Linear(hidden_dim, 1)

    def forward(self, views: list[torch.Tensor], query: torch.Tensor):
        q = self.query_projection(query).unsqueeze(1)
        projected = torch.stack([torch.tanh(layer(view)) for layer, view in zip(self.view_projection, views, strict=True)], dim=1)
        logits = self.attention(torch.tanh(projected + q)).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        fused = (projected * weights.unsqueeze(-1)).sum(dim=1)
        return self.mule_head(fused).squeeze(-1), self.merchant_head(fused).squeeze(-1), fused, weights


@dataclass
class MultiViewBundle:
    model: MultiViewGraphAttention

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


def train_multiview_attention(
    money_embeddings: np.ndarray,
    identity_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    query_features: np.ndarray,
    mule_labels: np.ndarray,
    merchant_labels: np.ndarray,
    output_path: str | Path,
    epochs: int = 20,
    seed: int = 42,
) -> MultiViewBundle:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewGraphAttention(
        embedding_dim=money_embeddings.shape[1],
        query_dim=query_features.shape[1],
    ).to(device)
    tensors = [
        torch.tensor(money_embeddings, dtype=torch.float32),
        torch.tensor(identity_embeddings, dtype=torch.float32),
        torch.tensor(context_embeddings, dtype=torch.float32),
        torch.tensor(query_features, dtype=torch.float32),
        torch.tensor(mule_labels, dtype=torch.float32),
        torch.tensor(merchant_labels, dtype=torch.float32),
    ]
    dataset = torch.utils.data.TensorDataset(*tensors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for money, identity, context, query, mule, merchant in loader:
            optimizer.zero_grad(set_to_none=True)
            mule_logit, merchant_logit, _, _ = model(
                [money.to(device), identity.to(device), context.to(device)], query.to(device)
            )
            loss = loss_fn(mule_logit, mule.to(device)) + 0.6 * loss_fn(merchant_logit, merchant.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    bundle = MultiViewBundle(model=model.eval())
    bundle.save(output_path)
    return bundle


@torch.no_grad()
def score_multiview_attention(
    bundle: MultiViewBundle,
    money_embeddings: np.ndarray,
    identity_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    query_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(bundle.model.parameters()).device
    mule, merchant, fused, weights = bundle.model(
        [
            torch.tensor(money_embeddings, dtype=torch.float32, device=device),
            torch.tensor(identity_embeddings, dtype=torch.float32, device=device),
            torch.tensor(context_embeddings, dtype=torch.float32, device=device),
        ],
        torch.tensor(query_features, dtype=torch.float32, device=device),
    )
    return (
        torch.sigmoid(mule).cpu().numpy(),
        torch.sigmoid(merchant).cpu().numpy(),
        fused.cpu().numpy(),
        weights.cpu().numpy(),
    )
