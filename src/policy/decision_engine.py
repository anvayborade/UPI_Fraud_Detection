from __future__ import annotations

from pathlib import Path

import yaml


_ACTION_ORDER = {
    "ALLOW": 0,
    "WARN": 1,
    "CONFIRM": 2,
    "STEP_UP": 3,
    "HOLD": 4,
    "BLOCK": 5,
}


class PolicyEngine:
    def __init__(self, config_path: str | Path = "configs/thresholds.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.thresholds = config["policy"]
        self.guardrails = config.get("runtime_guardrails", {})

    @staticmethod
    def _at_least(current: str, minimum: str) -> str:
        return minimum if _ACTION_ORDER[minimum] > _ACTION_ORDER[current] else current

    def select_action(
        self,
        fraud_risk: float,
        uncertainty: float,
        context_legitimacy: float,
        mule_risk: float,
        account_takeover_risk: float,
        hard_action: str | None = None,
        *,
        transaction_risk: float = 0.0,
        graph_anomaly: float = 0.0,
        social_engineering_risk: float = 0.0,
        qr_unverified: bool = False,
    ) -> str:
        if hard_action == "BLOCK":
            return "BLOCK"
        if hard_action == "HOLD":
            return "HOLD"

        # Strong specialist consensus should never be weakened to CONFIRM merely
        # because experts disagree elsewhere.
        if (
            transaction_risk >= float(self.guardrails.get("mule_transaction_risk", 0.80))
            and mule_risk >= float(self.guardrails.get("mule_graph_risk", 0.70))
            and graph_anomaly >= float(self.guardrails.get("mule_graph_anomaly", 0.70))
        ):
            return "HOLD"

        if (
            qr_unverified
            and transaction_risk >= float(self.guardrails.get("qr_transaction_risk", 0.72))
            and social_engineering_risk >= float(self.guardrails.get("qr_social_risk", 0.72))
        ):
            return "BLOCK"

        if account_takeover_risk >= self.thresholds["high_account_takeover_risk"]:
            return "HOLD" if fraud_risk >= self.thresholds["hold_risk"] else "STEP_UP"

        # High final risk takes priority over uncertainty.  Uncertainty is used to
        # decide friction in the grey zone, not to downgrade a severe risk signal.
        if fraud_risk >= self.thresholds["block_risk"]:
            return "BLOCK"
        if fraud_risk >= self.thresholds["hold_risk"]:
            return "HOLD"
        if mule_risk >= self.thresholds["high_mule_risk"] and fraud_risk >= self.thresholds["step_up_risk"]:
            return "HOLD"
        if hard_action == "STEP_UP" and fraud_risk >= self.thresholds["warn_risk"]:
            return "STEP_UP"
        if fraud_risk >= self.thresholds["step_up_risk"]:
            return "STEP_UP"

        if context_legitimacy >= self.thresholds["high_context_legitimacy"] and fraud_risk < self.thresholds["warn_risk"]:
            return "ALLOW"
        if uncertainty >= self.thresholds["high_uncertainty"]:
            return "CONFIRM"
        if fraud_risk >= self.thresholds["warn_risk"]:
            return "WARN"
        return "ALLOW"
