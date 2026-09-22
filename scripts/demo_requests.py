from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

SCENARIO_MAP = {
    "auto": "auto_long_distance_legitimate",
    "hospital": "hospital_emergency_legitimate",
    "qr_scam": "fake_refund_qr",
    "account_takeover": "account_takeover",
    "mule": "mule_directed",
}


def nullable(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def optional_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def optional_int(value: Any, default: int) -> int:
    return default if value is None or pd.isna(value) else int(value)


def optional_bool(value: Any, default: bool = False) -> bool:
    return default if value is None or pd.isna(value) else bool(value)


def load_data() -> pd.DataFrame:
    path = Path("data/processed/scored_transactions.pkl")
    if not path.exists():
        raise FileNotFoundError("Run training first; scored_transactions.pkl is missing.")
    return pd.read_pickle(path)


def select_row(data: pd.DataFrame, short_name: str) -> pd.Series:
    manifest_path = Path("data/processed/demo_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            "demo_manifest.json is missing. Run: py -3.11 -m scripts.build_demo_manifest"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transaction_id = manifest[short_name]["transaction_id"]
    rows = data[data["transaction_id"].astype(str).eq(str(transaction_id))]
    if rows.empty:
        raise ValueError(f"Manifest transaction {transaction_id} was not found in scored data.")
    return rows.iloc[0]


def row_to_event(row: pd.Series, short_name: str) -> dict[str, Any]:
    latitude = optional_float(row.get("latitude"))
    longitude = optional_float(row.get("longitude"))
    payee_latitude = optional_float(row.get("payee_latitude"))
    payee_longitude = optional_float(row.get("payee_longitude"))
    if payee_latitude is None:
        payee_latitude = latitude
    if payee_longitude is None:
        payee_longitude = longitude
    original_transaction_id = str(row["transaction_id"])

    return {
        "transaction_id": f"DEMO-{short_name.upper()}-{original_transaction_id}",
        "reference_transaction_id": original_transaction_id,
        "simulation_mode": True,
        "session_id": f"S-{original_transaction_id}",
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "payer_id": str(row["payer_id"]),
        "payee_id": str(row["payee_id"]),
        "amount": float(row["amount"]),
        "payment_mode": str(row["payment_mode"]),
        "device_id": str(row.get("device_id", "D_DEMO")),
        "vpa_id": nullable(row.get("vpa_id")),
        "merchant_id": nullable(row.get("merchant_id")),
        "qr_id": nullable(row.get("qr_id")),
        "ip_asn": nullable(row.get("ip_asn")),
        "latitude": latitude,
        "longitude": longitude,
        "payee_latitude": payee_latitude,
        "payee_longitude": payee_longitude,
        "is_new_payee": optional_bool(row.get("is_new_payee")),
        "qr_verified": optional_bool(row.get("qr_verified")),
        "collect_request": optional_bool(row.get("collect_request")),
        "merchant_category": nullable(row.get("merchant_category")),
        "device_age_days": optional_int(row.get("device_age_days"), 30),
        "sim_age_days": optional_int(row.get("sim_age_days"), 60),
        "app_registration_age_days": optional_int(row.get("app_registration_age_days"), 30),
        "device_known": optional_bool(row.get("device_known"), True),
        "play_integrity_ok": optional_bool(row.get("play_integrity_ok"), True),
        "recent_pin_reset": optional_bool(row.get("recent_pin_reset")),
        "app_reregistered": optional_bool(row.get("app_reregistered")),
        "remote_access_indicator": optional_bool(row.get("remote_access_indicator")),
        "overlay_indicator": optional_bool(row.get("overlay_indicator")),
        # Use the transaction's own value only; the API retrieves its historical
        # complaint snapshot through reference_transaction_id.
        "recipient_complaint_score": float(row.get("recipient_complaint_score", 0.0)),
        "known_mule": False,
        "common_owner_verified": optional_bool(row.get("common_owner_verified")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a held-out dataset-backed demo transaction")
    parser.add_argument("scenario", choices=sorted(SCENARIO_MAP))
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    data = load_data()
    selected = select_row(data, args.scenario)
    payload = row_to_event(selected, args.scenario)

    print(f"Scenario: {SCENARIO_MAP[args.scenario]}")
    print(f"Source dataset: {selected.get('source_dataset', 'unknown')}")
    print(f"Reference transaction: {payload['reference_transaction_id']}")
    print(f"Payer: {payload['payer_id']}")
    print(f"Payee: {payload['payee_id']}")
    print(f"Amount: {payload['amount']}\n")

    response = httpx.post(f"{args.api.rstrip('/')}/score", json=payload, timeout=30)
    print(f"HTTP {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    main()
