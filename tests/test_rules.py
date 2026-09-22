from __future__ import annotations

from src.rules.engine import RulesEngine


def test_known_mule_is_hard_block() -> None:
    result = RulesEngine().evaluate({"known_mule": True})
    assert result.hard_action == "BLOCK"
    assert "KNOWN_MULE_RECIPIENT" in result.reason_codes


def test_legitimate_transport_context_is_protective() -> None:
    result = RulesEngine().evaluate(
        {
            "known_mule": False,
            "qr_verified": True,
            "device_known": True,
            "recipient_archetype": "transport_provider",
            "journey_plausibility": 0.95,
            "payer_payee_proximity_km": 0.2,
        }
    )
    assert result.risk_delta < 0
    assert "VERIFIED_MERCHANT_AND_STABLE_DEVICE" in result.protective_codes
