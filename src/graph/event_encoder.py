from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GlobalNodeIndexer:
    mapping: dict[tuple[str, str], int] = field(default_factory=dict)
    node_types: list[str] = field(default_factory=list)

    def get_or_add(self, node_type: str, original_id: str) -> int:
        key = (node_type, str(original_id))
        if key not in self.mapping:
            self.mapping[key] = len(self.mapping)
            self.node_types.append(node_type)
        return self.mapping[key]

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"node_type": node_type, "original_id": original_id, "global_node_id": global_id}
            for (node_type, original_id), global_id in self.mapping.items()
        ]
        return pd.DataFrame(rows)


RELATION_IDS = {
    "ACCOUNT_PAYS_ACCOUNT": 0,
    "USER_USES_DEVICE": 1,
    "DEVICE_CONNECTS_IP": 2,
    "ACCOUNT_OWNS_VPA": 3,
    "VPA_BELONGS_MERCHANT": 4,
}


def encode_temporal_events(df: pd.DataFrame, indexer: GlobalNodeIndexer | None = None) -> tuple[pd.DataFrame, GlobalNodeIndexer]:
    indexer = indexer or GlobalNodeIndexer()
    rows: list[dict] = []
    for record in df.sort_values("timestamp").itertuples(index=False):
        src = indexer.get_or_add("account", str(record.payer_id))
        dst = indexer.get_or_add("account", str(record.payee_id))
        timestamp = pd.Timestamp(record.timestamp).timestamp()
        rows.append(
            {
                "transaction_id": str(record.transaction_id),
                "src": src,
                "dst": dst,
                "timestamp": timestamp,
                "relation_id": RELATION_IDS["ACCOUNT_PAYS_ACCOUNT"],
                "source_type": "account",
                "destination_type": "account",
                "log_amount": float(np.log1p(record.amount)),
                "is_qr": float(record.payment_mode == "QR"),
                "is_collect": float(record.payment_mode == "COLLECT"),
                "recipient_complaint_score": float(record.recipient_complaint_score),
                "rapid_outflow_ratio": float(record.rapid_outflow_ratio),
                "location_continuity": float(record.location_continuity),
                "label_mule": int(record.is_mule_directed),
                "source_dataset": str(getattr(record, "source_dataset", "unknown")),
                "source_partition": str(getattr(record, "source_partition", "default")),
            }
        )
    return pd.DataFrame(rows), indexer
