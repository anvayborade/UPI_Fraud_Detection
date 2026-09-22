from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _build_noisy_flow(record: object, rng: np.random.Generator) -> list[str]:
    """Generate an overlapping app journey instead of a label-coded template."""
    events: list[str] = ["APP_OPEN", "LOGIN"]
    is_fraud = int(getattr(record, "is_fraud")) == 1
    is_social = int(getattr(record, "is_social_engineering")) == 1
    is_ato = int(getattr(record, "is_account_takeover")) == 1

    # Device recovery events can also occur for genuine users.
    if int(getattr(record, "device_known")) == 0 and rng.random() < (0.82 if is_ato else 0.55):
        events.append("NEW_DEVICE_LOGIN")
    if int(getattr(record, "app_reregistered")) == 1 and rng.random() < 0.78:
        events.append("APP_REREGISTRATION")
    if int(getattr(record, "recent_pin_reset")) == 1 and rng.random() < 0.82:
        events.append("PIN_RESET")
    if is_ato and rng.random() < 0.48:
        events.append("BALANCE_ENQUIRY")
    if int(getattr(record, "is_new_payee")) == 1 and rng.random() < 0.70:
        events.append("NEW_PAYEE_ADDED")

    # Social-engineering signals are incomplete and occasionally appear in normal
    # activity, so the sequence model must combine multiple weak indicators.
    if is_social and rng.random() < 0.62:
        events.append("EXTERNAL_MESSAGE_OPEN")
    elif not is_fraud and rng.random() < 0.025:
        events.append("EXTERNAL_MESSAGE_OPEN")

    if int(getattr(record, "remote_access_indicator")) == 1 and rng.random() < 0.80:
        events.append("REMOTE_GUIDANCE_SIGNAL")
    elif is_social and rng.random() < 0.08:
        events.append("REMOTE_GUIDANCE_SIGNAL")

    mode = str(getattr(record, "payment_mode"))
    if mode == "COLLECT":
        events.append("COLLECT_REQUEST_OPENED")
    elif mode == "QR":
        gallery_probability = 0.42 if is_social else 0.035
        events.append("QR_SCAN_FROM_GALLERY" if rng.random() < gallery_probability else "PHYSICAL_QR_SCAN")
    else:
        events.append("RECIPIENT_SELECTED")

    if rng.random() < 0.88:
        events.append("RECIPIENT_NAME_VIEWED")
    events.append("AMOUNT_ENTERED")

    if rng.random() < (0.20 if is_fraud else 0.06):
        events.append("PAYMENT_REVIEW_DELAY")
    if rng.random() < (0.15 if is_fraud else 0.04):
        events.append("PAYMENT_RETRY")

    events.extend(["PIN_SCREEN_OPENED", "PAYMENT_CONFIRMED"])
    return events


def generate_event_sequences(transactions: pd.DataFrame, output_path: Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for record in transactions.itertuples(index=False):
        events = _build_noisy_flow(record, rng)
        current = pd.Timestamp(getattr(record, "timestamp")) - timedelta(seconds=len(events) * 12)
        previous = current
        for idx, event_type in enumerate(events):
            current = current + timedelta(seconds=int(rng.integers(2, 24)))
            delta = 0 if idx == 0 else max(1, int((current - previous).total_seconds()))
            previous = current
            rows.append(
                {
                    "transaction_id": record.transaction_id,
                    "payer_id": record.payer_id,
                    "session_id": f"S-{record.transaction_id}",
                    "event_index": idx,
                    "event_type": event_type,
                    "timestamp": current,
                    "delta_seconds": delta,
                    "amount": float(record.amount) if event_type in {"AMOUNT_ENTERED", "PAYMENT_CONFIRMED"} else 0.0,
                    "new_device": int(getattr(record, "device_known") == 0),
                    "new_payee": int(getattr(record, "is_new_payee")),
                    "collect_request": int(getattr(record, "collect_request")),
                    "qr_unverified": int(getattr(record, "payment_mode") == "QR" and getattr(record, "qr_verified") == 0),
                    "remote_access": int(getattr(record, "remote_access_indicator")),
                    "location_inconsistent": int(getattr(record, "location_continuity") < 0.25),
                    "label_account_takeover": int(getattr(record, "is_account_takeover")),
                    "label_social_engineering": int(getattr(record, "is_social_engineering")),
                }
            )
    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out
