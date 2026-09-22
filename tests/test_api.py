from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.schemas.transaction import UPIEvent


def test_upi_event_accepts_valid_transaction() -> None:
    event = UPIEvent(
        transaction_id="T1",
        timestamp=datetime.now(timezone.utc),
        payer_id="U1",
        payee_id="A1",
        amount=500,
        payment_mode="QR",
        device_id="D1",
    )
    assert event.amount == 500
    assert event.timestamp.tzinfo is not None


def test_upi_event_rejects_negative_amount() -> None:
    with pytest.raises(ValidationError):
        UPIEvent(
            transaction_id="T2",
            payer_id="U1",
            payee_id="A1",
            amount=-1,
            device_id="D1",
        )
