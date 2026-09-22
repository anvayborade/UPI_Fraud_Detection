# File-by-file implementation guide

This document explains what each executable/configuration file does, what depends on it, and how to run it. Empty `__init__.py` files only mark directories as Python packages and do not need to be run.

## Root files

### `README.md`

Primary installation and quick-start guide. Read this first.

### `requirements.txt`

Python dependencies. Install PyTorch using the official CPU/CUDA selector first, then run:

```bash
pip install -r requirements.txt
```

### `.env.example`

Template for environment variables. Copy it to `.env`. Important settings include the PaySim path, Redis/Kafka endpoints, model directory, optional OpenAI credentials and demo row count.

### `docker-compose.yml`

Starts Redis and a single-node Kafka development broker:

```bash
docker compose up -d
```

It is intentionally a local prototype configuration, not a secure production cluster.

## Configuration files

### `configs/features.yaml`

Lists the online/tabular features consumed by the transaction LightGBM and context model. Change feature lists here rather than editing model code.

### `configs/rules.yaml`

Defines risk rules and protective rules. Protective rules are important for the auto-driver, verified merchant and self-transfer edge cases.

### `configs/thresholds.yaml`

Policy thresholds for `ALLOW`, `CONFIRM`, `WARN`, `STEP_UP`, `HOLD` and `BLOCK`.

### `configs/graph_schema.yaml`

Human-readable graph node types, relation types and view membership.

### `configs/model_config.yaml`

Suggested hyperparameters for the Transformer, graph and fusion models. The current trainer uses matching defaults in code; use this file as the single place to record experimental configurations when extending the pipeline.

## Settings and schemas

### `src/settings.py`

Loads `.env` using `pydantic-settings`, creates model/data paths and exposes cached application settings.

Used by almost every CLI command.

### `src/schemas/transaction.py`

Pydantic request models:

- `UPIEvent`: canonical transaction request;
- `TelemetryEvent`: app/session event from Android or a simulator.

### `src/schemas/prediction.py`

Pydantic response and internal result models:

- per-expert model scores;
- rule results;
- final fraud prediction.

## Data loading and simulation

### `src/data/load_paysim.py`

Looks for the configured PaySim CSV. If found, maps its columns to the canonical base schema. If absent and fallback is enabled, calls `generate_base_transactions()`.

Run indirectly through data preparation, or inspect from Python:

```python
from src.data.load_paysim import load_paysim
```

### `src/simulation/generate_base.py`

Generates a deterministic PaySim-like transaction table for users who want to start without downloading data. This is the base only; it does not contain full UPI scenarios.

### `src/simulation/generate_scenarios.py`

Adds:

- QR, collect, P2P/P2M modes;
- devices, VPAs, merchants, IPs and location;
- account takeover and social-engineering indicators;
- recipient archetypes;
- fan-in/fan-out and flow-through patterns;
- fraud multi-labels;
- hard-negative legitimate novelty scenarios.

This is the most important synthetic-data file for the research question.

### `src/simulation/generate_sequences.py`

Expands every transaction into an ordered app/session event journey. The sequence Transformer trains on this table.

### `src/simulation/generate_complaints.py`

Creates synthetic complaint narratives linked to recipients. The complaint extractor can process these locally or with an optional LLM.

### `src/data/split_data.py`

Performs a chronological 70/15/15 split. It deliberately avoids a purely random split, reducing future-to-past leakage.

### `src/data/prepare_dataset.py`

Main dataset orchestration command:

```bash
python -m src.data.prepare_dataset --rows 30000
```

Runs loading/generation, scenario injection, feature engineering, splitting and sequence generation.

## Feature engineering

### `src/features/transaction_features.py`

Adds time, amount, archetype and combined device/session-risk features.

### `src/features/behavioural_features.py`

Computes payer-relative prior-only rolling features:

- counts over five minutes/one hour;
- one-hour amount sum;
- unique payees;
- time since previous payment;
- robust amount deviation;
- amount relative to the user's prior median.

The current transaction is inserted after its features are computed, preventing self-leakage.

### `src/features/context_features.py`

Builds the context-support score and the novel-transaction flag from journey, proximity, device, QR, complaint, outflow and merchant-category evidence.

### `src/features/graph_features.py`

Adds peer-normalised fan-in/fan-out and flow-through scores. It compares recipients to their archetype peers instead of treating all high-degree recipients as fraud.

### `src/features/redis_features.py`

Redis-backed online feature store. Handles hashes, JSON conversion, sliding-window sorted sets and rolling live updates.

## Rules and policy

### `src/rules/engine.py`

Evaluates `configs/rules.yaml`. Returns:

- risk adjustment;
- optional hard action;
- risk reason codes;
- protective reason codes.

Study directly:

```python
from src.rules.engine import RulesEngine
print(RulesEngine().evaluate({"known_mule": True}))
```

### `src/policy/decision_engine.py`

Converts model risk, context legitimacy, uncertainty, mule risk, account-takeover risk and hard-rule results into a proportionate action.

The classifier itself does not decide the action.

## Geospatial context

### `src/geospatial/h3_encoder.py`

Converts coordinates into coarse H3 cells and provides Haversine distance. It has a deterministic grid fallback when `h3` is unavailable.

### `src/geospatial/journey_builder.py`

Maintains/reconstructs recent location transitions and calculates distance, speed and continuity.

### `src/geospatial/journey_features.py`

Creates journey plausibility, proximity and impossible-travel features used by context scoring and rules.

## Telemetry

### `src/telemetry/collector.py`

Receives validated telemetry events, stores session history in Redis and updates latest telemetry-derived features.

### `src/telemetry/integrity_service.py`

Converts device-known, Play Integrity, registration age, overlay and remote-access indicators into a bounded device/session risk score.

### `src/telemetry/telemetry_features.py`

Aggregates session-event timing, retry and journey signals into features suitable for online scoring.

## Sequence model

### `src/models/time_transformer.py`

Contains:

- event vocabulary encoding;
- numeric event-feature projection;
- relative-time representation;
- compact two-layer Transformer encoder;
- multi-task heads for account takeover and social engineering;
- training, saving and batch-scoring helpers.

It is called by `train_all.py`; study independently in notebook 04.

## Heterogeneous graph construction

### `src/graph/schema.py`

Python representation of graph-view descriptions and typed relations.

### `src/graph/build_views.py`

Builds three PyG `HeteroData` graphs:

1. money flow;
2. identity/device;
3. context/trust.

It creates typed node feature matrices, relation edges and account labels.

### `src/graph/event_encoder.py`

Converts transaction rows into ordered typed temporal messages and maintains global IDs across entity types. These messages feed the temporal model.

### `src/graph/snapshot_builder.py`

Creates graph snapshots for selected time windows, supporting periodic HGT refreshes.

### `src/graph/graph_worker.py`

Near-real-time worker. It reads newly scored events/cached artifacts and refreshes recipient/account graph scores in Redis at a configurable interval:

```bash
python -m src.graph.graph_worker --interval-seconds 60
```

## Graph models

### `src/models/hetero_tgn.py`

Compact typed temporal event-memory model. It embeds source/destination node types, relation type, message features and time deltas, then updates entity states in temporal order. It produces mule and rapid-outflow scores.

This is a bounded research implementation that demonstrates the architecture without claiming the scalability of a distributed production TGN.

### `src/models/hgt_model.py`

Two-layer Heterogeneous Graph Transformer. It applies type- and relation-aware attention to graph snapshots and produces account embeddings with mule and merchant-legitimacy heads.

### `src/models/multiview_attention.py`

Learns transaction-conditioned weights for money-flow, identity/device and context/trust graph embeddings. It outputs a fused graph embedding, mule score, merchant score and interpretable view weights.

### `src/models/anomaly_model.py`

Fits Isolation Forest on fused graph embeddings and converts anomaly scores to a bounded risk value. This supports previously unseen/sparse fraud patterns but should not independently hard-block a payment.

## Tabular models

### `src/models/common.py`

Feature-list loading and safe numeric matrix conversion shared by tabular models.

### `src/models/lightgbm_model.py`

LightGBM transaction-risk model with early stopping, PR-AUC/ROC-AUC evaluation, saving and loading.

### `src/models/context_model.py`

Separate LightGBM model trained to estimate legitimate novelty among unusual transactions. Keeping it separate makes false-positive reduction measurable.

## Fusion and uncertainty

### `src/models/fusion_model.py`

Defines the mixture-of-experts gating network and multi-label fraud heads. It combines rule, statistical, tabular, sequence, graph, complaint and journey scores.

### `src/models/uncertainty.py`

Calculates uncertainty using expert disagreement, missing information, graph sparsity, ensemble confidence and anomaly evidence.

### `src/fusion/service.py`

Online orchestration layer used by FastAPI. It:

1. builds request features;
2. retrieves Redis scores;
3. runs rules/statistics/LightGBM/context experts;
4. runs or loads fusion output;
5. calculates uncertainty;
6. applies policy;
7. stores the decision and publishes events.

## Complaint/LLM intelligence

### `src/intelligence/complaint_schema.py`

Pydantic schema for structured scam intelligence.

### `src/intelligence/llm_client.py`

Optional OpenAI Responses API client using schema-constrained parsing. It requires `OPENAI_API_KEY`.

### `src/intelligence/complaint_extractor.py`

Redacts obvious identifiers, calls the optional LLM, validates the schema and falls back to a deterministic local extractor on failure.

### `src/intelligence/recipient_intelligence.py`

Aggregates structured complaints into recipient-level complaint/scam scores and updates the online intelligence record.

## Streaming

### `src/streaming/kafka_utils.py`

Common Kafka producer/consumer configuration and JSON serialization.

### `src/streaming/producer.py`

Replays prepared transactions to `upi.transactions.raw`:

```bash
python -m src.streaming.producer --limit 200 --sleep 0.05
```

### `src/streaming/feature_consumer.py`

Consumes raw transactions and updates rolling payer/recipient features in Redis.

### `src/streaming/feedback_consumer.py`

Consumes confirmed outcomes/analyst feedback and stores them for future labelling/retraining.

## API and monitoring

### `src/api/main.py`

FastAPI application. Run:

```bash
uvicorn src.api.main:app --reload
```

### `src/monitoring/metrics.py`

Prometheus counters/histograms for scoring volume, actions, complaints and latency.

## Training and evaluation

### `src/training/train_all.py`

Complete training orchestrator. It is intentionally explicit so the order and intermediate outputs are visible.

```bash
python -m src.training.train_all --skip-prepare
```

### `src/training/load_online_cache.py`

Loads trained sequence/recipient/entity scores into Redis for low-latency API lookup.

### `src/training/evaluate.py`

Builds a final JSON evaluation report with PR-AUC, ROC-AUC, calibration, legitimate-novelty false positives, action rates and per-scenario results.

```bash
python -m src.training.evaluate
```

## Dashboard

### `dashboard/app.py`

Streamlit investigator interface. It reads the scored table and can call the API for interactive scenarios. Run:

```bash
streamlit run dashboard/app.py
```

## Scripts

### `scripts/download_data.py`

Downloads the public PaySim Kaggle dataset through `kagglehub` and copies its largest CSV into `data/raw/`.

### `scripts/demo_requests.py`

Sends one of five complete sample requests to a running API.

### `scripts/run_demo.sh` and `scripts/run_demo.ps1`

Convenience scripts for dataset preparation, training and cache loading. Infrastructure and UI processes are started separately so their logs remain visible.

### `scripts/smoke_test.py`

Runs an in-memory 5,000-row feature and LightGBM/context-model check without Kafka, Redis, Parquet output or the graph stack. Use this first to verify Python and core ML dependencies.

## Android files

### `android-client/settings.gradle.kts`

Declares the Android project and repositories.

### `android-client/build.gradle.kts`

Top-level Android/Kotlin plugin versions.

### `android-client/app/build.gradle.kts`

Android application settings, SDK levels, JDK 17 and dependencies.

### `android-client/app/src/main/AndroidManifest.xml`

Internet and coarse/fine location permissions plus main activity registration.

### `android-client/app/src/main/res/layout/activity_main.xml`

Simple form for payer, payee, amount, risk toggles, API URL and request buttons.

### `android-client/app/src/main/java/com/example/upifraud/MainActivity.kt`

Collects app-scoped telemetry/current location, builds JSON requests and calls `/telemetry`, `/precheck` and `/score`.

## Tests

### `tests/test_features.py`

Verifies prior-only rolling features and geospatial helper behaviour.

### `tests/test_rules.py`

Verifies known-mule blocking and protective transport/merchant context.

### `tests/test_policy.py`

Verifies context-aware allow behaviour and hard-action priority.

### `tests/test_api.py`

Verifies canonical request validation.

Run all tests:

```bash
python -m pytest -q
```

## Notebooks

The nine notebooks are guided interfaces over the same modules; they do not contain separate hidden implementations. Use them in numeric order to understand and experiment with each stage.
