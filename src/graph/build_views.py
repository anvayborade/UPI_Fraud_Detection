from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.geospatial.h3_encoder import latlon_to_cell
from src.graph.schema import ARCHETYPES


@dataclass
class BuiltView:
    data: HeteroData
    id_maps: dict[str, dict[str, int]]


def _index(values: Iterable[object]) -> tuple[dict[str, int], list[str]]:
    unique = sorted({str(v) for v in values if v is not None and str(v) != "nan"})
    return {value: idx for idx, value in enumerate(unique)}, unique


def _edge_tensor(sources: list[int], destinations: list[int]) -> torch.Tensor:
    if not sources:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([sources, destinations], dtype=torch.long)

def _column_tensor(values: Iterable[float]) -> torch.Tensor:
    """Create a float feature matrix with shape [num_nodes, 1].

    torch.tensor([]) normally produces shape [0], which HGTConv cannot use.
    This function guarantees [0, 1] for an empty node type and [N, 1]
    for a non-empty node type.
    """
    values_list = list(values)

    if not values_list:
        return torch.empty((0, 1), dtype=torch.float32)

    return torch.tensor(
        values_list,
        dtype=torch.float32,
    ).reshape(-1, 1)


def _account_features(df: pd.DataFrame, account_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outgoing = df.groupby("payer_id").agg(
        out_count=("transaction_id", "count"),
        out_amount=("amount", "sum"),
        out_unique=("payee_id", "nunique"),
    )
    incoming = df.groupby("payee_id").agg(
        in_count=("transaction_id", "count"),
        in_amount=("amount", "sum"),
        in_unique=("payer_id", "nunique"),
        complaint=("recipient_complaint_score", "max"),
        rapid_outflow=("rapid_outflow_ratio", "mean"),
        holding=("median_holding_minutes", "median"),
        account_age=("recipient_account_age_days", "median"),
        mule_label=("is_mule_directed", "max"),
        merchant_label=("recipient_archetype", lambda s: int(any(v in {"merchant", "transport_provider"} for v in s))),
    )
    rows: list[list[float]] = []
    mule: list[int] = []
    merchant: list[int] = []
    for account in account_ids:
        out = outgoing.loc[account] if account in outgoing.index else None
        inc = incoming.loc[account] if account in incoming.index else None
        rows.append(
            [
                np.log1p(float(out["out_count"]) if out is not None else 0),
                np.log1p(float(out["out_amount"]) if out is not None else 0),
                np.log1p(float(out["out_unique"]) if out is not None else 0),
                np.log1p(float(inc["in_count"]) if inc is not None else 0),
                np.log1p(float(inc["in_amount"]) if inc is not None else 0),
                np.log1p(float(inc["in_unique"]) if inc is not None else 0),
                float(inc["complaint"]) if inc is not None else 0,
                float(inc["rapid_outflow"]) if inc is not None else 0,
                np.log1p(float(inc["holding"]) if inc is not None else 60),
                np.log1p(float(inc["account_age"]) if inc is not None else 365),
            ]
        )
        mule.append(int(inc["mule_label"]) if inc is not None else 0)
        merchant.append(int(inc["merchant_label"]) if inc is not None else 0)
    if rows:
        x_tensor = torch.tensor(
            rows,
            dtype=torch.float32,
        )
    else:
        # There are ten account features in each row above.
        x_tensor = torch.empty(
            (0, 10),
            dtype=torch.float32,
        )

    return (
        x_tensor,
        torch.tensor(mule, dtype=torch.float32),
        torch.tensor(merchant, dtype=torch.float32),
    )

def _masks(num_nodes: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_nodes)
    train_end = int(num_nodes * 0.7)
    val_end = int(num_nodes * 0.85)
    masks = []
    for subset in [order[:train_end], order[train_end:val_end], order[val_end:]]:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.tensor(subset, dtype=torch.long)] = True
        masks.append(mask)
    return tuple(masks)  # type: ignore[return-value]




def _money_source_masks(account_ids: list[str], seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use IBM HI for training and reserve IBM LI as a low-prevalence holdout.

    PaySim/BankSim/synthetic UPI accounts are split 70/15/15. IBM HI and LI
    account prefixes are disjoint, so LI nodes can be held out without sharing
    identities with HI nodes.
    """
    rng = np.random.default_rng(seed)
    train = torch.zeros(len(account_ids), dtype=torch.bool)
    val = torch.zeros(len(account_ids), dtype=torch.bool)
    test = torch.zeros(len(account_ids), dtype=torch.bool)
    other: list[int] = []
    for idx, account in enumerate(account_ids):
        if account.startswith("IBM_HI_SMALL_"):
            train[idx] = True
        elif account.startswith("IBM_LI_SMALL_"):
            test[idx] = True
        else:
            other.append(idx)
    order = rng.permutation(other) if other else np.asarray([], dtype=int)
    i = int(len(order) * 0.70)
    j = int(len(order) * 0.85)
    if i:
        train[torch.tensor(order[:i], dtype=torch.long)] = True
    if j > i:
        val[torch.tensor(order[i:j], dtype=torch.long)] = True
    if len(order) > j:
        test[torch.tensor(order[j:], dtype=torch.long)] = True
    # Tiny test fixtures may not contain every source; fall back to random masks.
    if not train.any() or not val.any() or not test.any():
        return _masks(len(account_ids), seed=seed)
    return train, val, test

def build_money_flow_view(df: pd.DataFrame) -> BuiltView:
    account_map, accounts = _index(pd.concat([df["payer_id"], df["payee_id"]]))
    data = HeteroData()
    x, y_mule, y_merchant = _account_features(df, accounts)
    data["account"].x = x
    data["account"].y_mule = y_mule
    data["account"].y_merchant = y_merchant
    train_mask, val_mask, test_mask = _money_source_masks(accounts)
    data["account"].train_mask = train_mask
    data["account"].val_mask = val_mask
    data["account"].test_mask = test_mask

    src = [account_map[str(v)] for v in df["payer_id"]]
    dst = [account_map[str(v)] for v in df["payee_id"]]
    data[("account", "pays", "account")].edge_index = _edge_tensor(src, dst)
    data[("account", "pays", "account")].edge_attr = torch.tensor(
        np.column_stack(
            [
                np.log1p(df["amount"].to_numpy(float)),
                (df["payment_mode"] == "QR").astype(float),
                df["collect_request"].astype(float),
                df["recipient_complaint_score"].astype(float),
                df["rapid_outflow_ratio"].astype(float),
            ]
        ),
        dtype=torch.float32,
    )
    data[("account", "pays", "account")].edge_time = torch.tensor(
        pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy() / 1e9,
        dtype=torch.float64,
    )
    return BuiltView(data=data, id_maps={"account": account_map})


def build_identity_view(df: pd.DataFrame) -> BuiltView:
    account_map, accounts = _index(pd.concat([df["payer_id"], df["payee_id"]]))
    user_map, users = _index(df["payer_id"])
    device_map, devices = _index(df["device_id"])
    vpa_map, vpas = _index(df["vpa_id"])
    ip_map, ips = _index(df["ip_asn"])
    data = HeteroData()
    account_x, y_mule, y_merchant = _account_features(df, accounts)
    data["account"].x = account_x
    data["account"].y_mule = y_mule
    data["account"].y_merchant = y_merchant
    train_mask, val_mask, test_mask = _masks(len(accounts), seed=43)
    data["account"].train_mask, data["account"].val_mask, data["account"].test_mask = train_mask, val_mask, test_mask
    data["user"].x = _column_tensor(
    np.log1p((df["payer_id"] == user).sum())
    for user in users
    )

    data["device"].x = _column_tensor(
        np.log1p((df["device_id"] == device).sum())
        for device in devices
    )

    data["vpa"].x = _column_tensor(
        np.log1p((df["vpa_id"] == vpa).sum())
        for vpa in vpas
    )

    data["ip_asn"].x = _column_tensor(
        np.log1p((df["ip_asn"] == ip).sum())
        for ip in ips
    )

    first_by_user = df.sort_values("timestamp").drop_duplicates("payer_id")
    data[("user", "owns", "account")].edge_index = _edge_tensor(
        [user_map[str(v)] for v in first_by_user["payer_id"]],
        [account_map[str(v)] for v in first_by_user["payer_id"]],
    )
    data[("user", "uses", "device")].edge_index = _edge_tensor(
        [user_map[str(v)] for v in first_by_user["payer_id"]],
        [device_map[str(v)] for v in first_by_user["device_id"]],
    )
    first_by_device = df.sort_values("timestamp").drop_duplicates("device_id")
    data[("device", "connects_from", "ip_asn")].edge_index = _edge_tensor(
        [device_map[str(v)] for v in first_by_device["device_id"]],
        [ip_map[str(v)] for v in first_by_device["ip_asn"]],
    )
    first_by_payee = df.sort_values("timestamp").drop_duplicates("payee_id")
    data[("account", "owns", "vpa")].edge_index = _edge_tensor(
        [account_map[str(v)] for v in first_by_payee["payee_id"]],
        [vpa_map[str(v)] for v in first_by_payee["vpa_id"]],
    )
    return BuiltView(data=data, id_maps={"account": account_map, "user": user_map, "device": device_map, "vpa": vpa_map, "ip_asn": ip_map})


def build_context_view(df: pd.DataFrame) -> BuiltView:
    account_map, accounts = _index(pd.concat([df["payer_id"], df["payee_id"]]))
    vpa_map, vpas = _index(df["vpa_id"])
    merchant_map, merchants = _index(df["merchant_id"].dropna())
    qr_map, qrs = _index(df["qr_id"])
    h3_values = [latlon_to_cell(float(lat), float(lon), 7) for lat, lon in zip(df["latitude"], df["longitude"], strict=True)]
    h3_map, h3_cells = _index(h3_values)
    complaint_count = int((df["recipient_complaint_score"] >= 0.65).sum())
    complaint_ids = [f"C{idx:08d}" for idx in range(complaint_count)]
    complaint_map, complaints = _index(complaint_ids)

    data = HeteroData()
    account_x, y_mule, y_merchant = _account_features(df, accounts)
    data["account"].x = account_x
    data["account"].y_mule = y_mule
    data["account"].y_merchant = y_merchant
    train_mask, val_mask, test_mask = _masks(len(accounts), seed=44)
    data["account"].train_mask, data["account"].val_mask, data["account"].test_mask = train_mask, val_mask, test_mask

    data["vpa"].x = _column_tensor(
    np.log1p((df["vpa_id"] == vpa).sum())
    for vpa in vpas
    )

    data["merchant"].x = _column_tensor(
        1.0 for _ in merchants
    )

    data["qr"].x = _column_tensor(
        float(df.loc[df["qr_id"] == qr_id, "qr_verified"].mean())
        for qr_id in qrs
    )

    data["h3_cell"].x = _column_tensor(
        np.log1p(h3_values.count(cell))
        for cell in h3_cells
    )

    data["complaint"].x = _column_tensor(
        1.0 for _ in complaints
    )

    first_by_payee = df.sort_values("timestamp").drop_duplicates("payee_id")
    data[("account", "owns", "vpa")].edge_index = _edge_tensor(
        [account_map[str(v)] for v in first_by_payee["payee_id"]],
        [vpa_map[str(v)] for v in first_by_payee["vpa_id"]],
    )
    merchant_rows = first_by_payee[first_by_payee["merchant_id"].notna()]
    data[("vpa", "belongs_to", "merchant")].edge_index = _edge_tensor(
        [vpa_map[str(v)] for v in merchant_rows["vpa_id"]],
        [merchant_map[str(v)] for v in merchant_rows["merchant_id"]],
    )
    data[("qr", "resolves_to", "vpa")].edge_index = _edge_tensor(
        [qr_map[str(v)] for v in first_by_payee["qr_id"]],
        [vpa_map[str(v)] for v in first_by_payee["vpa_id"]],
    )
    payer_rows = df.drop_duplicates("payer_id")
    payer_h3 = [latlon_to_cell(float(lat), float(lon), 7) for lat, lon in zip(payer_rows["latitude"], payer_rows["longitude"], strict=True)]
    data[("account", "occurred_in", "h3_cell")].edge_index = _edge_tensor(
        [account_map[str(v)] for v in payer_rows["payer_id"]],
        [h3_map[cell] for cell in payer_h3],
    )
    complaint_src: list[int] = []
    complaint_dst: list[int] = []
    counter = 0
    for row in df.itertuples(index=False):
        if float(row.recipient_complaint_score) >= 0.65:
            complaint_src.append(complaint_map[f"C{counter:08d}"])
            complaint_dst.append(account_map[str(row.payee_id)])
            counter += 1
    data[("complaint", "flags", "account")].edge_index = _edge_tensor(complaint_src, complaint_dst)
    return BuiltView(data=data, id_maps={"account": account_map, "vpa": vpa_map, "merchant": merchant_map, "qr": qr_map, "h3_cell": h3_map, "complaint": complaint_map})


def build_all_views(df: pd.DataFrame) -> dict[str, HeteroData]:
    """Build all three graph views in memory.

    This helper is convenient for notebooks and tests. The training pipeline uses
    :func:`build_and_save_all_views` to persist the same views to disk.
    """
    builders = {
        "money": build_money_flow_view,
        "identity": build_identity_view,
        "context": build_context_view,
    }
    return {name: builder(df).data for name, builder in builders.items()}


def build_and_save_all_views(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    money_flow_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist the three graph views.

    ``money_flow_df`` may contain IBM AML edges in addition to the UPI-like
    transaction product. Identity/device and context/trust views deliberately use
    only ``df`` because IBM AML has no genuine UPI device, QR or location data.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "money": money_flow_df if money_flow_df is not None else df,
        "identity": df,
        "context": df,
    }
    builders = {
        "money": build_money_flow_view,
        "identity": build_identity_view,
        "context": build_context_view,
    }
    paths: dict[str, Path] = {}
    for name, builder in builders.items():
        view = builder(inputs[name])
        path = output / f"{name}_view.pt"
        torch.save({"data": view.data, "id_maps": view.id_maps}, path)
        paths[name] = path
    return paths
