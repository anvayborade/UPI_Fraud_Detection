from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.fusion.service import ScoringService
from src.monitoring.metrics import COMPLAINTS_PROCESSED, SCORED_TRANSACTIONS, SCORING_LATENCY
from src.schemas.prediction import FraudPrediction
from src.schemas.transaction import TelemetryEvent, UPIEvent
from src.settings import get_settings
from src.streaming.kafka_utils import json_dumps
from src.telemetry.collector import TelemetryCollector

settings = get_settings()
service = ScoringService(settings)
telemetry_collector = TelemetryCollector(service.store)

app = FastAPI(
    title="UPI Fraud Intent Firewall",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    description="Research prototype for context-aware pre-authorisation UPI fraud detection.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ComplaintRequest(BaseModel):
    payee_id: str
    text: str = Field(min_length=5)
    use_llm: bool = True


class FeedbackRequest(BaseModel):
    transaction_id: str
    confirmed_fraud: bool
    fraud_types: list[str] = Field(default_factory=list)
    analyst_notes: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        redis_ok = service.store.ping()
    except Exception:
        redis_ok = False
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "transaction_model_loaded": service.transaction_model is not None,
        "context_model_loaded": service.context_model is not None,
        "fusion_model_loaded": service.fusion_model is not None,
    }


@app.post("/telemetry")
def ingest_telemetry(event: TelemetryEvent) -> dict[str, str]:
    telemetry_collector.record(event)
    return {"status": "accepted", "event_id": event.event_id}


@app.post("/precheck")
def precheck(event: UPIEvent) -> dict[str, Any]:
    try:
        return service.precheck(event)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/score", response_model=FraudPrediction)
def score(event: UPIEvent) -> FraudPrediction:
    started = time.perf_counter()
    try:
        result = service.score(event)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    SCORED_TRANSACTIONS.labels(action=result.recommended_action).inc()
    SCORING_LATENCY.observe(time.perf_counter() - started)
    return result


@app.post("/complaint")
def complaint(request: ComplaintRequest) -> dict[str, Any]:
    try:
        result = service.process_complaint(request.payee_id, request.text, request.use_llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    COMPLAINTS_PROCESSED.labels(scam_type=result["scam_type"]).inc()
    return result


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    payload = request.model_dump(mode="json")
    service.store.set_hash(f"feedback:{request.transaction_id}", payload, ttl_seconds=2_592_000)
    if service.producer is not None:
        try:
            service.producer.produce(
                "upi.feedback",
                key=request.transaction_id,
                value=json_dumps(payload),
            )
            service.producer.poll(0)
        except Exception:
            pass
    return {"status": "accepted", "transaction_id": request.transaction_id}


@app.get("/transaction/{transaction_id}")
def get_transaction(transaction_id: str) -> dict[str, Any]:
    result = service.store.get_hash(f"decision:{transaction_id}")
    if not result:
        raise HTTPException(status_code=404, detail="Transaction decision not found")
    return result


@app.get("/entity/{entity_type}/{entity_id}")
def get_entity(entity_type: str, entity_id: str) -> dict[str, Any]:
    if entity_type not in {"payer", "recipient", "risk", "sequence"}:
        raise HTTPException(status_code=400, detail="Unsupported entity type")
    key_map = {
        "payer": f"features:payer:{entity_id}",
        "recipient": f"features:recipient:{entity_id}",
        "risk": f"risk:{entity_id}",
        "sequence": f"sequence:{entity_id}",
    }
    result = service.store.get_hash(key_map[entity_type])
    if not result:
        raise HTTPException(status_code=404, detail="Entity data not found")
    return result


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
