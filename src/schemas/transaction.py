from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PaymentMode = Literal["P2P", "P2M", "QR", "COLLECT", "MANDATE", "SELF_TRANSFER"]


class UPIEvent(BaseModel):
    transaction_id: str
    # Optional ID of an existing offline-scored transaction whose historical
    # expert snapshot should be used for an honest replay/demo.
    reference_transaction_id: str | None = None
    # When True, score the event without mutating rolling windows or Kafka.
    simulation_mode: bool = False
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payer_id: str
    payee_id: str
    amount: float = Field(gt=0)
    payment_mode: PaymentMode = "P2P"

    device_id: str
    vpa_id: str | None = None
    merchant_id: str | None = None
    qr_id: str | None = None
    ip_asn: str | None = None

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    payee_latitude: float | None = Field(default=None, ge=-90, le=90)
    payee_longitude: float | None = Field(default=None, ge=-180, le=180)
    h3_cell: str | None = None
    payee_h3_cell: str | None = None

    is_new_payee: bool = True
    qr_verified: bool = False
    collect_request: bool = False
    merchant_category: str | None = None

    device_age_days: int = Field(default=30, ge=0)
    sim_age_days: int = Field(default=60, ge=0)
    app_registration_age_days: int = Field(default=30, ge=0)
    device_known: bool = True
    play_integrity_ok: bool | None = None
    recent_pin_reset: bool = False
    app_reregistered: bool = False
    remote_access_indicator: bool | None = None
    overlay_indicator: bool | None = None
    recipient_complaint_score: float = Field(default=0.0, ge=0, le=1)
    known_mule: bool = False
    common_owner_verified: bool = False

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class TelemetryEvent(BaseModel):
    event_id: str
    user_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str
    device_id: str
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    amount: float = Field(default=0.0, ge=0)
    recipient_id: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
