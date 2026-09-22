from __future__ import annotations

NODE_TYPES = (
    "account", "user", "device", "vpa", "merchant", "ip_asn", "h3_cell", "qr", "complaint", "transaction"
)

EDGE_TYPES = (
    ("user", "owns", "account"),
    ("user", "uses", "device"),
    ("device", "connects_from", "ip_asn"),
    ("account", "owns", "vpa"),
    ("account", "pays", "account"),
    ("transaction", "sent_by", "account"),
    ("transaction", "received_by", "account"),
    ("transaction", "initiated_from", "device"),
    ("transaction", "occurred_in", "h3_cell"),
    ("user", "scanned", "qr"),
    ("qr", "resolves_to", "vpa"),
    ("vpa", "belongs_to", "merchant"),
    ("complaint", "flags", "account"),
)

ARCHETYPES = (
    "personal", "merchant", "transport_provider", "gig_worker", "aggregator", "new_business", "money_mule", "unknown"
)

GRAPH_VIEW_DESCRIPTIONS = {
    "money": {
        "purpose": "Money-flow, rapid pass-through, splitting and circular-transfer evidence",
        "node_types": ["account"],
        "edge_types": [("account", "pays", "account")],
    },
    "identity": {
        "purpose": "Shared device, VPA and IP relationships linked to account farms or takeover",
        "node_types": ["user", "account", "device", "vpa", "ip_asn"],
        "edge_types": [
            ("user", "owns", "account"),
            ("user", "uses", "device"),
            ("device", "connects_from", "ip_asn"),
            ("account", "owns", "vpa"),
        ],
    },
    "context": {
        "purpose": "Merchant, QR, coarse location and complaint evidence for context legitimacy",
        "node_types": ["account", "vpa", "merchant", "qr", "h3_cell", "complaint"],
        "edge_types": [
            ("account", "owns", "vpa"),
            ("vpa", "belongs_to", "merchant"),
            ("qr", "resolves_to", "vpa"),
            ("account", "occurred_in", "h3_cell"),
            ("complaint", "flags", "account"),
        ],
    },
}
