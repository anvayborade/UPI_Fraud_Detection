from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.schemas.prediction import RuleResult


class RulesEngine:
    def __init__(self, config_path: str | Path = "configs/rules.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.rules = config.get("rules", {})
        self.protective_rules = config.get("protective_rules", {})

    @staticmethod
    def _truthy(features: dict[str, Any], key: str) -> bool:
        return bool(features.get(key, False))

    def evaluate(self, features: dict[str, Any]) -> RuleResult:
        risk_delta = 0.0
        hard_action: str | None = None
        reasons: list[str] = []
        protective: list[str] = []

        if self._truthy(features, "known_mule"):
            cfg = self.rules["known_mule"]
            risk_delta += float(cfg["risk_delta"])
            hard_action = str(cfg["hard_action"])
            reasons.append("KNOWN_MULE_RECIPIENT")

        if self._truthy(features, "remote_access_indicator") and self._truthy(features, "collect_request"):
            cfg = self.rules["remote_access_collect"]
            risk_delta += float(cfg["risk_delta"])
            hard_action = hard_action or str(cfg["hard_action"])
            reasons.append("REMOTE_ACCESS_DURING_COLLECT_REQUEST")

        if (
            float(features.get("device_age_days", 999)) <= 3
            and self._truthy(features, "recent_pin_reset")
            and self._truthy(features, "is_new_payee")
        ):
            cfg = self.rules["new_device_pin_reset_new_payee"]
            risk_delta += float(cfg["risk_delta"])
            hard_action = hard_action or str(cfg["hard_action"])
            reasons.append("NEW_DEVICE_PIN_RESET_NEW_PAYEE")

        if (
            not self._truthy(features, "qr_verified")
            and float(features.get("complaint_intelligence_score", 0)) >= 0.65
            and str(features.get("payment_mode", "")) == "QR"
        ):
            cfg = self.rules["unverified_qr_high_complaint"]
            risk_delta += float(cfg["risk_delta"])
            hard_action = hard_action or str(cfg["hard_action"])
            reasons.append("UNVERIFIED_QR_HIGH_COMPLAINT_RISK")

        if self._truthy(features, "impossible_travel"):
            cfg = self.rules["impossible_travel"]
            risk_delta += float(cfg["risk_delta"])
            hard_action = hard_action or str(cfg["hard_action"])
            reasons.append("IMPOSSIBLE_TRAVEL")

        if (
            self._truthy(features, "qr_verified")
            and self._truthy(features, "device_known")
            and str(features.get("recipient_archetype", "")) in {"merchant", "transport_provider"}
        ):
            cfg = self.protective_rules["verified_merchant_normal_device"]
            risk_delta += float(cfg["risk_delta"])
            protective.append("VERIFIED_MERCHANT_AND_STABLE_DEVICE")

        if (
            float(features.get("journey_plausibility", 0.5)) >= 0.75
            and float(features.get("payer_payee_proximity_km", 999)) <= 3
        ):
            cfg = self.protective_rules["plausible_travel_near_payee"]
            risk_delta += float(cfg["risk_delta"])
            protective.append("PLAUSIBLE_JOURNEY_AND_PROXIMITY")

        if self._truthy(features, "common_owner_verified"):
            cfg = self.protective_rules["trusted_self_transfer"]
            risk_delta += float(cfg["risk_delta"])
            protective.append("VERIFIED_SELF_TRANSFER")

        return RuleResult(
            risk_delta=max(-1.0, min(1.0, risk_delta)),
            hard_action=hard_action,
            reason_codes=reasons,
            protective_codes=protective,
        )
