from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.transforms import ToUndirected

def _normalise_heterodata_shapes(data: HeteroData) -> HeteroData:
    """Ensure all HGT node and edge tensors have valid dimensions."""

    for node_type in data.node_types:
        x = data[node_type].x

        if x is None:
            raise ValueError(
                f"Node type '{node_type}' does not have an x feature tensor."
            )

        x = torch.as_tensor(
            x,
            dtype=torch.float32,
        )

        num_nodes = int(data[node_type].num_nodes or 0)

        if x.ndim == 0:
            # One node with one scalar feature.
            x = x.reshape(1, 1)

        elif x.ndim == 1:
            if x.numel() == 0:
                # Empty node type: [0] becomes [0, 1].
                x = torch.empty(
                    (0, 1),
                    dtype=torch.float32,
                )

            elif num_nodes == 1:
                # One node with several features: [F] becomes [1, F].
                x = x.reshape(1, -1)

            elif x.numel() == num_nodes:
                # N nodes with one feature each: [N] becomes [N, 1].
                x = x.reshape(num_nodes, 1)

            else:
                raise ValueError(
                    f"Cannot infer a valid feature matrix for node type "
                    f"'{node_type}'. Shape={tuple(x.shape)}, "
                    f"num_nodes={num_nodes}."
                )

        elif x.ndim > 2:
            # Keep the node dimension and flatten all feature dimensions.
            x = x.reshape(x.shape[0], -1)

        if x.ndim != 2:
            raise ValueError(
                f"Node type '{node_type}' must have a 2D feature tensor. "
                f"Received shape {tuple(x.shape)}."
            )

        data[node_type].x = x.contiguous()

    for edge_type in data.edge_types:
        edge_index = torch.as_tensor(
            data[edge_type].edge_index,
            dtype=torch.long,
        )

        if edge_index.numel() == 0:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        elif edge_index.ndim == 1:
            if edge_index.numel() % 2 != 0:
                raise ValueError(
                    f"Edge type {edge_type} contains an invalid flat "
                    f"edge tensor of length {edge_index.numel()}."
                )

            edge_index = edge_index.reshape(2, -1)

        elif edge_index.ndim == 2:
            if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.transpose(0, 1)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Edge type {edge_type} must have shape [2, num_edges]. "
                f"Received {tuple(edge_index.shape)}."
            )

        data[edge_type].edge_index = edge_index.contiguous()

    return data


class HGTAccountModel(nn.Module):
    def __init__(self, metadata, hidden_channels: int = 64, out_channels: int = 32, heads: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        self.node_types = list(metadata[0])
        self.projections = nn.ModuleDict({node_type: Linear(-1, hidden_channels) for node_type in self.node_types})
        self.convs = nn.ModuleList()
        for index in range(num_layers):
            output = out_channels if index == num_layers - 1 else hidden_channels
            self.convs.append(HGTConv(hidden_channels if index == 0 else hidden_channels, output, metadata, heads=heads))
            if index == num_layers - 1 and output != hidden_channels:
                self.final_to_hidden = nn.Linear(output, out_channels)
        self.mule_head = nn.Linear(out_channels, 1)
        self.merchant_head = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        projected = {
            node_type: torch.relu(
                self.projections[node_type](x)
            )
            for node_type, x in x_dict.items()
        }

        x_dict = projected

        for layer_index, conv in enumerate(self.convs):
            output_dict = conv(
                x_dict,
                edge_index_dict,
            )

            is_last_layer = (
                layer_index == len(self.convs) - 1
            )

            if not is_last_layer:
                next_dict = {}

                for node_type, previous_value in x_dict.items():
                    updated_value = output_dict.get(node_type)

                    # Preserve the previous embedding when a node type receives
                    # no messages in this layer.
                    if updated_value is None:
                        updated_value = previous_value

                    next_dict[node_type] = torch.relu(
                        updated_value
                    )

                x_dict = next_dict

            else:
                x_dict = output_dict

        account_embedding = x_dict.get("account")

        if account_embedding is None:
            raise RuntimeError(
                "The account node type did not receive any HGT messages. "
                "Check that the graph contains account-related edges."
            )

        return (
            self.mule_head(account_embedding).squeeze(-1),
            self.merchant_head(account_embedding).squeeze(-1),
            account_embedding,
        )


@dataclass
class HGTResult:
    scores: pd.DataFrame
    model_path: Path


def train_hgt_view(
    view_path: str | Path,
    output_dir: str | Path,
    view_name: str,
    epochs: int = 15,
    seed: int = 42,
) -> HGTResult:
    torch.manual_seed(seed)
    payload = torch.load(
    view_path,
    weights_only=False,
    )

    data: HeteroData = ToUndirected()(
        payload["data"]
    )

    data = _normalise_heterodata_shapes(data)

    id_maps = payload["id_maps"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = HGTAccountModel(data.metadata()).to(device)
    # Initialise lazy layers.
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mule, merchant, _ = model(data.x_dict, data.edge_index_dict)
        mask = data["account"].train_mask
        loss = loss_fn(mule[mask], data["account"].y_mule[mask]) + 0.6 * loss_fn(merchant[mask], data["account"].y_merchant[mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        mule, merchant, embedding = model(data.x_dict, data.edge_index_dict)
    reverse_map = {idx: account for account, idx in id_maps["account"].items()}
    rows = [
        {
            "account_id": reverse_map[idx],
            f"{view_name}_hgt_mule_score": float(torch.sigmoid(mule[idx]).cpu()),
            f"{view_name}_hgt_merchant_score": float(torch.sigmoid(merchant[idx]).cpu()),
            f"{view_name}_embedding": embedding[idx].cpu().numpy().astype(np.float32),
        }
        for idx in range(len(reverse_map))
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{view_name}_hgt.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": data.metadata(),
            "hidden_channels": 64,
            "out_channels": 32,
            "id_maps": id_maps,
        },
        model_path,
    )
    scores = pd.DataFrame(rows)
    scores.to_pickle(output / f"{view_name}_hgt_scores.pkl")
    return HGTResult(scores=scores, model_path=model_path)
