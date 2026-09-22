from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.redis_features import OnlineFeatureStore
from src.geospatial.h3_encoder import haversine_km, latlon_to_cell
from src.intelligence.complaint_extractor import extract_complaint
from src.models.context_model import LegitimateNoveltyModel
from src.models.fusion_model import (
    EXPERT_COLUMNS,
    FusionBundle,
    add_fusion_runtime_columns,
    predict_fusion,
)
from src.models.lightgbm_model import TransactionRiskModel
from src.models.uncertainty import calculate_uncertainty
from src.policy.decision_engine import PolicyEngine
from src.rules.engine import RulesEngine
from src.schemas.prediction import FraudPrediction, ModelScores
from src.schemas.transaction import UPIEvent
from src.settings import Settings
from src.streaming.kafka_utils import build_producer, json_dumps
from src.telemetry.integrity_service import integrity_risk

ARCHETYPE_CODES = {
    "personal": 0,
    "merchant": 1,
    "transport_provider": 2,
    "gig_worker": 3,
    "aggregator": 4,
    "new_business": 5,
    "money_mule": 6,
    "unknown": 7,
}

FRAUD_TYPE_NAMES = {
    "is_account_takeover": "ACCOUNT_TAKEOVER",
    "is_social_engineering": "SOCIAL_ENGINEERING",
    "is_qr_deception": "QR_DECEPTION",
    "is_collect_scam": "COLLECT_REQUEST_SCAM",
    "is_mule_directed": "MULE_DIRECTED_PAYMENT",
    "is_transaction_splitting": "TRANSACTION_SPLITTING",
    "is_remote_access_fraud": "REMOTE_ACCESS_FRAUD",
}


class ScoringService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = OnlineFeatureStore(settings.redis_url)
        self.rules = RulesEngine()
        self.policy = PolicyEngine()
        self.transaction_model = self._try_load(
            TransactionRiskModel, Path("models/lightgbm/transaction_model.joblib")
        )
        self.context_model = self._try_load(
            LegitimateNoveltyModel, Path("models/context/context_model.joblib")
        )
        self.fusion_model = self._try_load(FusionBundle, Path("models/fusion"))
        try:
            self.producer = build_producer(settings.kafka_bootstrap_servers)
        except Exception:
            self.producer = None

    @staticmethod
    def _try_load(cls, path: Path):
        try:
            return cls.load(path)
        except Exception:
            return None

    def _recipient_cache(self, payee_id: str) -> dict[str, Any]:
        data = self.store.get_hash(f"features:recipient:{payee_id}")
        data.update(self.store.get_hash(f"risk:{payee_id}"))
        return data

    def _payer_cache(self, payer_id: str) -> dict[str, Any]:
        return self.store.get_hash(f"features:payer:{payer_id}")

    def _sequence_cache(self, event: UPIEvent) -> dict[str, Any]:
        if event.reference_transaction_id:
            exact = self.store.get_hash(f"sequence:txn:{event.reference_transaction_id}")
            if exact:
                return exact
        if event.session_id:
            session = self.store.get_hash(f"sequence:session:{event.session_id}")
            if session:
                return session
        return self.store.get_hash(f"sequence:{event.payer_id}")

    def _transaction_snapshot(self, event: UPIEvent) -> dict[str, Any]:
        if not event.reference_transaction_id:
            return {}
        return self.store.get_hash(f"snapshot:txn:{event.reference_transaction_id}")

    def build_features(self, event: UPIEvent) -> dict[str, Any]:
        payer = self._payer_cache(event.payer_id)
        recipient = self._recipient_cache(event.payee_id)
        snapshot = self._transaction_snapshot(event)
        sequence = self._sequence_cache(event)
        timestamp = pd.Timestamp(event.timestamp)

        def snap(name: str, fallback: Any) -> Any:
            value = snapshot.get(name) if snapshot else None
            return fallback if value is None else value

        proximity = 20.0
        if None not in (
            event.latitude,
            event.longitude,
            event.payee_latitude,
            event.payee_longitude,
        ):
            proximity = haversine_km(
                float(event.latitude),
                float(event.longitude),
                float(event.payee_latitude),
                float(event.payee_longitude),
            )

        continuity = float(snap("location_continuity", payer.get("location_continuity", 0.70)))
        journey_default = max(
            0.0,
            min(1.0, 0.7 * continuity + 0.3 * math.exp(-proximity / 15)),
        )
        journey_plausibility = float(
            snap("journey_plausibility", payer.get("journey_plausibility", journey_default))
        )
        impossible_travel = int(float(payer.get("max_speed_kmh", 0)) > 900)

        archetype = str(
            snap("recipient_archetype", recipient.get("recipient_archetype", "unknown"))
        )
        if snapshot:
            complaint = max(
                float(event.recipient_complaint_score),
                float(snapshot.get("complaint_intelligence_score", 0)),
            )
            graph_mule = float(snapshot.get("graph_mule_score", 0))
            graph_anomaly = float(snapshot.get("graph_anomaly_score", 0))
            graph_merchant = float(snapshot.get("graph_merchant_score", 0))
            graph_available = float(snapshot.get("graph_score_available", 0))
            recipient_history_available = float(
                snapshot.get("recipient_history_available", graph_available)
            )
            complaint_available = float(
                float(snapshot.get("complaint_history_count", 0)) > 0
            )
        else:
            complaint = max(
                float(event.recipient_complaint_score),
                float(recipient.get("recipient_complaint_score", 0)),
                float(recipient.get("complaint_intelligence", 0)),
            )
            graph_mule = float(
                recipient.get("recipient_mule_score", recipient.get("learned_mule_score", 0))
            )
            graph_anomaly = float(recipient.get("graph_anomaly", 0))
            graph_merchant = float(recipient.get("merchant_legitimacy", 0))
            graph_available = float(bool(recipient))
            recipient_history_available = float(bool(recipient))
            complaint_available = float(
                "complaint_intelligence" in recipient or event.recipient_complaint_score > 0
            )

        rapid_outflow = float(
            snap(
                "rapid_outflow_ratio",
                recipient.get(
                    "rapid_outflow_ratio", recipient.get("flow_through_ratio_live", 0)
                ),
            )
        )
        user_median = float(
            snap("payer_median_amount_prior", payer.get("payer_median_amount_prior", event.amount))
        )
        amount_robust_z = float(
            (event.amount - user_median) / max(0.5 * user_median, 1.0)
        )
        device_risk = integrity_risk(
            event.play_integrity_ok,
            event.device_known,
            event.app_registration_age_days,
            event.overlay_indicator,
            event.remote_access_indicator,
        )
        sequence_available = float(bool(sequence))
        payer_history_available = float(
            snap("payer_history_available", payer.get("payer_history_available", bool(payer)))
        )

        return {
            **event.model_dump(),
            "timestamp": event.timestamp,
            "hour": timestamp.hour,
            "is_weekend": int(timestamp.dayofweek >= 5),
            "log_amount": math.log1p(event.amount),
            "h3_cell": event.h3_cell
            or (
                latlon_to_cell(event.latitude, event.longitude, 8)
                if event.latitude is not None and event.longitude is not None
                else None
            ),
            "txn_count_5m": float(snap("txn_count_5m", payer.get("txn_count_5m", 0))),
            "txn_count_1h": float(snap("txn_count_1h", payer.get("txn_count_1h", 0))),
            "amount_sum_1h": float(snap("amount_sum_1h", payer.get("amount_sum_1h", 0))),
            "unique_payees_1h": float(
                snap("unique_payees_1h", payer.get("unique_payees_1h", 0))
            ),
            "amount_robust_z": amount_robust_z,
            "time_since_previous_sec": float(
                snap(
                    "time_since_previous_sec",
                    payer.get("time_since_previous_sec", 86_400),
                )
            ),
            "fan_in_1h": float(snap("fan_in_1h", recipient.get("fan_in_1h_live", 0))),
            "fan_out_1h": float(snap("fan_out_1h", recipient.get("fan_out_1h_live", 0))),
            "rapid_outflow_ratio": rapid_outflow,
            "median_holding_minutes": float(
                snap("median_holding_minutes", recipient.get("median_holding_minutes", 60))
            ),
            "recipient_mule_score_cached": graph_mule,
            "graph_merchant_legitimacy_cached": graph_merchant,
            "graph_anomaly_cached": graph_anomaly,
            "sequence_social_engineering_score_cached": float(
                sequence.get("social_engineering_score", 0)
            ),
            "sequence_account_takeover_score_cached": float(
                sequence.get("account_takeover_score", 0)
            ),
            "device_session_risk": device_risk,
            "recipient_complaint_score": complaint,
            "recipient_archetype": archetype,
            "recipient_archetype_code": ARCHETYPE_CODES.get(archetype, 7),
            "recipient_account_age_days": float(
                snap(
                    "recipient_account_age_days",
                    recipient.get("recipient_account_age_days", 365),
                )
            ),
            "payer_payee_proximity_km": proximity,
            "location_continuity": continuity,
            "journey_plausibility": journey_plausibility,
            "impossible_travel": impossible_travel,
            "merchant_category_consistency": float(
                snap(
                    "merchant_category_consistency",
                    0.85 if archetype in {"merchant", "transport_provider", "gig_worker", "new_business"} else 0.45,
                )
            ),
            "qr_verified": int(event.qr_verified),
            "collect_request": int(event.collect_request),
            "is_new_payee": int(event.is_new_payee),
            "device_known": int(event.device_known),
            "recent_pin_reset": int(event.recent_pin_reset),
            "app_reregistered": int(event.app_reregistered),
            "remote_access_indicator": int(bool(event.remote_access_indicator)),
            "overlay_indicator": int(bool(event.overlay_indicator)),
            "play_integrity_ok": int(event.play_integrity_ok is not False),
            "common_owner_verified": int(event.common_owner_verified),
            "known_mule": int(event.known_mule),
            "graph_score_available": graph_available,
            "sequence_score_available": sequence_available,
            "complaint_score_available": complaint_available,
            "recipient_history_available": recipient_history_available,
            "payer_history_available": payer_history_available,
            "snapshot_cache_used": int(bool(snapshot)),
        }

    def precheck(self, event: UPIEvent) -> dict[str, Any]:
        features = self.build_features(event)
        rule = self.rules.evaluate(features)
        mule_risk = float(features["recipient_mule_score_cached"])
        complaint = float(features["recipient_complaint_score"])
        warning = None
        if mule_risk >= 0.78 or event.known_mule:
            warning = (
                "The recipient has elevated network risk. Do not proceed unless independently verified."
            )
        elif complaint >= 0.60:
            warning = (
                "This recipient has elevated complaint intelligence. Verify the recipient before paying."
            )
        elif event.is_new_payee:
            warning = (
                "This is a first-time recipient. Confirm the displayed name and payment direction."
            )
        return {
            "transaction_id": event.transaction_id,
            "recipient_risk": mule_risk,
            "complaint_risk": complaint,
            "qr_verified": event.qr_verified,
            "hard_action": rule.hard_action,
            "warning": warning,
        }

    def score(self, event: UPIEvent) -> FraudPrediction:
        started = time.perf_counter()
        features = self.build_features(event)
        row = pd.DataFrame([features])
        rule = self.rules.evaluate(features)

        statistical = float(
            np.clip(
                0.60 * (1 - math.exp(-abs(float(features["amount_robust_z"])) / 3))
                + 0.25
                * float(features["txn_count_5m"])
                / (float(features["txn_count_5m"]) + 4)
                + 0.15 * (1 - float(features["location_continuity"])),
                0,
                1,
            )
        )
        transaction_score = (
            float(self.transaction_model.predict_proba(row)[0])
            if self.transaction_model
            else statistical
        )
        legitimate_context = (
            float(self.context_model.predict_proba(row)[0])
            if self.context_model
            else float(features["journey_plausibility"])
        )
        graph_mule = float(features["recipient_mule_score_cached"])
        sequence_ato = float(features["sequence_account_takeover_score_cached"])
        sequence_social = float(features["sequence_social_engineering_score_cached"])
        graph_anomaly = float(features["graph_anomaly_cached"])
        graph_merchant = float(features["graph_merchant_legitimacy_cached"])
        complaint = float(features["recipient_complaint_score"])

        fusion_row = pd.DataFrame(
            [
                {
                    "rule_risk": max(0.0, rule.risk_delta),
                    "statistical_anomaly": statistical,
                    "transaction_fraud_score": transaction_score,
                    "legitimate_novelty_inverse": 1 - legitimate_context,
                    "device_session_risk": float(features["device_session_risk"]),
                    "sequence_account_takeover_score": sequence_ato,
                    "sequence_social_engineering_score": sequence_social,
                    "graph_mule_score": graph_mule,
                    "merchant_legitimacy_inverse": 1 - graph_merchant,
                    "graph_anomaly_score": graph_anomaly,
                    "complaint_intelligence_score": complaint,
                    "journey_risk": 1 - float(features["journey_plausibility"]),
                    "is_new_payee": int(event.is_new_payee),
                    "collect_request": int(event.collect_request),
                    "is_qr": int(event.payment_mode == "QR"),
                    "device_changed": int(not event.device_known),
                    "graph_neighbourhood_size": float(features["fan_in_1h"])
                    + float(features["fan_out_1h"]),
                    "recipient_archetype_code": features["recipient_archetype_code"],
                    "missing_feature_ratio": sum(
                        value is None
                        for value in [
                            event.latitude,
                            event.longitude,
                            event.merchant_id,
                            event.ip_asn,
                            event.qr_id,
                        ]
                    )
                    / 5,
                    "graph_score_available": features["graph_score_available"],
                    "sequence_score_available": features["sequence_score_available"],
                    "complaint_score_available": features["complaint_score_available"],
                    "recipient_history_available": features["recipient_history_available"],
                    "payer_history_available": features["payer_history_available"],
                }
            ]
        )
        fusion_row = add_fusion_runtime_columns(fusion_row)

        if self.fusion_model:
            probability, label_probs, _ = predict_fusion(self.fusion_model, fusion_row)
            final_risk = float(probability[0])
            fraud_types = []
            for column, probability_value in zip(
                self.fusion_model.fraud_type_columns,
                label_probs[0],
                strict=True,
            ):
                threshold = self.fusion_model.fraud_type_thresholds.get(column, 0.50)
                if probability_value >= threshold:
                    fraud_types.append(FRAUD_TYPE_NAMES[column])
            expert_values = fusion_row[self.fusion_model.expert_columns].to_numpy(np.float32)
        else:
            expert_values = fusion_row[EXPERT_COLUMNS].to_numpy(np.float32)
            final_risk = float(np.clip(np.mean(expert_values), 0, 1))
            fraud_types = []

        uncertainty = float(
            calculate_uncertainty(
                expert_values,
                np.asarray([final_risk]),
                fusion_row["missing_feature_ratio"].to_numpy(float),
                np.asarray(
                    [1 / (1 + fusion_row["graph_neighbourhood_size"].iloc[0])]
                ),
                np.asarray([graph_anomaly]),
            )[0]
        )
        action = self.policy.select_action(
            fraud_risk=final_risk,
            uncertainty=uncertainty,
            context_legitimacy=legitimate_context,
            mule_risk=graph_mule,
            account_takeover_risk=sequence_ato,
            hard_action=rule.hard_action,
            transaction_risk=transaction_score,
            graph_anomaly=graph_anomaly,
            social_engineering_risk=sequence_social,
            qr_unverified=event.payment_mode == "QR" and not event.qr_verified,
        )

        reasons = rule.reason_codes + rule.protective_codes
        if transaction_score >= 0.70:
            reasons.append("TRANSACTION_MODEL_HIGH_RISK")
        if graph_mule >= 0.70:
            reasons.append("RECIPIENT_NETWORK_RISK")
        if sequence_ato >= 0.70:
            reasons.append("ACCOUNT_TAKEOVER_SEQUENCE")
        if sequence_social >= 0.70:
            reasons.append("SOCIAL_ENGINEERING_SEQUENCE")
        if legitimate_context >= 0.75:
            reasons.append("LEGITIMATE_CONTEXT_SUPPORT")
        if event.payment_mode == "QR" and not event.qr_verified:
            reasons.append("UNVERIFIED_QR")
        if features["snapshot_cache_used"]:
            reasons.append("TRANSACTION_TIME_SNAPSHOT_USED")
        if not fraud_types and final_risk >= 0.55:
            fraud_types = ["UNSPECIFIED_HIGH_RISK_PAYMENT"]

        prediction = FraudPrediction(
            transaction_id=event.transaction_id,
            fraud_probability=final_risk,
            legitimate_novelty_probability=legitimate_context,
            uncertainty=uncertainty,
            fraud_types=sorted(set(fraud_types)),
            recommended_action=action,
            reason_codes=sorted(set(reasons)),
            model_scores=ModelScores(
                rule_risk=max(0.0, rule.risk_delta),
                statistical_anomaly=statistical,
                transaction_fraud=transaction_score,
                legitimate_novelty=legitimate_context,
                device_session_risk=float(features["device_session_risk"]),
                sequence_account_takeover=sequence_ato,
                sequence_social_engineering=sequence_social,
                graph_mule=graph_mule,
                graph_merchant_legitimacy=graph_merchant,
                graph_anomaly=graph_anomaly,
                complaint_intelligence=complaint,
                journey_plausibility=float(features["journey_plausibility"]),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self.store.set_hash(
            f"decision:{event.transaction_id}",
            prediction.model_dump(mode="json"),
            ttl_seconds=604_800,
        )
        if not event.simulation_mode:
            self.store.add_transaction_window(
                event.payer_id,
                event.timestamp.timestamp(),
                event.transaction_id,
                event.amount,
                event.payee_id,
            )
            if self.producer is not None:
                try:
                    self.producer.produce(
                        "upi.transactions.scored",
                        key=event.transaction_id,
                        value=json_dumps(prediction.model_dump(mode="json")),
                    )
                    self.producer.poll(0)
                except Exception:
                    pass
        return prediction

    def process_complaint(
        self,
        payee_id: str,
        text: str,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        intelligence = extract_complaint(text, use_llm=use_llm)
        existing = self._recipient_cache(payee_id)
        previous = float(existing.get("complaint_intelligence", 0))
        updated = max(previous, intelligence.confidence)
        self.store.set_hash(
            f"risk:{payee_id}",
            {
                **existing,
                "complaint_intelligence": updated,
                "latest_scam_type": intelligence.scam_type,
            },
            ttl_seconds=2_592_000,
        )
        return intelligence.model_dump()
