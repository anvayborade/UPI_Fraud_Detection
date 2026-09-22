from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScamIntelligence(BaseModel):
    scam_type: Literal[
        "FAKE_REFUND", "IMPERSONATION", "QR_DECEPTION", "COLLECT_REQUEST",
        "REMOTE_ACCESS", "ACCOUNT_TAKEOVER", "MULE_TRANSFER", "UNKNOWN",
    ] = "UNKNOWN"
    impersonated_entity: str | None = None
    communication_channel: str | None = None
    payment_mechanism: str | None = None
    urgency_language: bool = False
    refund_claim: bool = False
    remote_access_mentioned: bool = False
    collect_request_mentioned: bool = False
    qr_code_mentioned: bool = False
    referenced_vpas: list[str] = Field(default_factory=list)
    referenced_mobile_hashes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
