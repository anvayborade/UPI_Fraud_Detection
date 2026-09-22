from __future__ import annotations

from src.policy.decision_engine import PolicyEngine


def test_high_legitimate_context_can_allow_novel_payment() -> None:
    action = PolicyEngine().select_action(
        fraud_risk=0.40,
        uncertainty=0.10,
        context_legitimacy=0.90,
        mule_risk=0.05,
        account_takeover_risk=0.05,
    )
    assert action == "ALLOW"


def test_known_hard_block_wins() -> None:
    action = PolicyEngine().select_action(
        fraud_risk=0.10,
        uncertainty=0.10,
        context_legitimacy=0.95,
        mule_risk=0.0,
        account_takeover_risk=0.0,
        hard_action="BLOCK",
    )
    assert action == "BLOCK"
