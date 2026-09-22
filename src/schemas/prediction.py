from __future__ import annotations

from pydantic import BaseModel, Field


class RuleResult(BaseModel):
    risk_delta: float = 0.0
    hard_action: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    protective_codes: list[str] = Field(default_factory=list)


class ModelScores(BaseModel):
    rule_risk: float = 0.0
    statistical_anomaly: float = 0.0
    transaction_fraud: float = 0.0
    legitimate_novelty: float = 0.0
    device_session_risk: float = 0.0
    sequence_account_takeover: float = 0.0
    sequence_social_engineering: float = 0.0
    graph_mule: float = 0.0
    graph_merchant_legitimacy: float = 0.0
    graph_anomaly: float = 0.0
    complaint_intelligence: float = 0.0
    journey_plausibility: float = 0.5


class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(ge=0, le=1)
    legitimate_novelty_probability: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    fraud_types: list[str]
    recommended_action: str
    reason_codes: list[str]
    model_scores: ModelScores
    latency_ms: float = Field(ge=0)
