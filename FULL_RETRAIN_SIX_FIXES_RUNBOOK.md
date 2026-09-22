# Full retraining runbook: all six fixes

This patch fixes the six issues discovered during the online demonstrations:

1. Stable recipient identities and chronological compromise transitions.
2. Different recipients for the auto, hospital, QR-scam, account-takeover and mule demos.
3. Sequence scores cached by transaction/session rather than the payer's maximum-ever score.
4. Transaction-time recipient risk, delayed complaints and prior-only TGN scoring.
5. Fusion training with missing-cache and contradictory-expert examples.
6. Separately calibrated thresholds for each fraud type plus safer policy ordering.

## Files replaced or added

- `src/simulation/recipient_profiles.py`
- `src/simulation/generate_scenarios.py`
- `src/data/prepare_multisource.py`
- `src/graph/event_encoder.py`
- `src/models/hetero_tgn.py`
- `src/models/time_transformer.py`
- `src/models/fusion_model.py`
- `src/training/train_all.py`
- `src/training/evaluate.py`
- `src/training/load_online_cache.py`
- `src/fusion/service.py`
- `src/policy/decision_engine.py`
- `src/schemas/transaction.py`
- `configs/thresholds.yaml`
- `scripts/build_demo_manifest.py`
- `scripts/demo_requests.py`
- `scripts/validate_six_fixes.py`
- `dashboard/app.py`
- two tests under `tests/`

## 1. Back up and apply the patch

From the parent folder:

```powershell
Copy-Item `
  "UPI_Fraud_Multisource_Data_Project" `
  "UPI_Fraud_Multisource_Data_Project_Before_Six_Fixes" `
  -Recurse
```

Extract the patch and copy its contents over the project root, selecting Replace when prompted.

Do not copy or delete any raw dataset files.

## 2. Stop current services

Press `Ctrl+C` in FastAPI, Streamlit, feature-consumer, graph-worker and feedback-consumer terminals.

Docker may remain running, although it is also safe to stop it:

```powershell
docker compose down
```

## 3. Activate the Python environment and validate the code

```powershell
cd "C:\Users\Anvay Borade\OneDrive\Desktop\UPI_Fraud_Multisource_Data_Project"
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m compileall src scripts dashboard tests
python -m pytest -q
```

Expected result for the supplied patch: `12 passed`.

## 4. Delete all stale processed data and models

Do not delete `data/raw`.

```powershell
Remove-Item -Recurse -Force data\processed -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data\sequences -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data\graph_views -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data\complaints -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force models -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force mlruns -ErrorAction SilentlyContinue
```

## 5. Verify the downloaded source datasets

```powershell
python -m scripts.verify_multisource_data
```

The following transaction files must exist:

- PaySim CSV
- BankSim CSV
- IBM `HI-Small_Trans.csv`
- IBM `LI-Small_Trans.csv`

## 6. Prepare fresh stable-recipient data

First laptop-friendly run:

```powershell
python -m src.data.prepare_multisource `
  --paysim-rows 30000 `
  --banksim-rows 30000 `
  --ibm-negative-ratio 10 `
  --ibm-max-negatives-per-file 20000 `
  --synthetic-fraud-rate 0.04
```

This now assigns one permanent profile to every payee and only allows a fraud profile to change state chronologically at its activation/compromise point.

## 7. Check recipient consistency before training

```powershell
python -m scripts.validate_six_fixes
```

At this stage the important lines should be:

```text
Stable recipient profiles: True
Stable recipient archetypes: True
Payees with multiple profiles: 0
Non-chronological compromise transitions: 0
```

You can inspect the generated table:

```powershell
python -c "import pandas as pd; d=pd.read_parquet('data/processed/recipient_consistency.parquet'); print(d.head()); print(d[['recipient_profile','recipient_archetype']].max())"
```

## 8. Run the leakage checker

```powershell
python -m scripts.check_leakage
```

There should be no synthetic cached score in the LightGBM feature list and no single generated field should nearly reproduce the label.

## 9. Train every model again

```powershell
python -m src.training.train_all --skip-prepare
```

The command trains:

1. Transaction LightGBM
2. Legitimate-novelty LightGBM
3. Transaction/session Transformer
4. Money-flow HGT
5. Identity/device HGT
6. Context/trust HGT
7. Multi-view attention
8. Prior-only transaction-time TGN
9. Graph anomaly model
10. Delayed complaint intelligence
11. Robust mixture-of-experts fusion
12. Per-fraud-type threshold calibration
13. Uncertainty and policy actions
14. A fixed unique-recipient demo manifest

## 10. Evaluate only the held-out test split

```powershell
python -m src.training.evaluate
```

The report is written to:

```text
models/evaluation_report.json
```

The updated evaluator filters to the held-out test transaction IDs by default.

## 11. Validate all six fixes after training

```powershell
python -m scripts.validate_six_fixes
```

You should additionally see:

```text
Transaction-time/availability columns present: True
Calibrated fraud-type thresholds: 7
Demo recipients are unique: True
```

Inspect the demo recipients:

```powershell
Get-Content data\processed\demo_manifest.json
```

Every demo should have a different `payee_id`.

## 12. Start Docker infrastructure

```powershell
docker compose up -d
docker compose ps
```

Wait until Kafka and Redis are running/healthy.

Create topics if they do not already exist:

```powershell
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic upi.transactions.raw --partitions 1 --replication-factor 1

docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic upi.transactions.scored --partitions 1 --replication-factor 1

docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic upi.feedback --partitions 1 --replication-factor 1
```

List the topics:

```powershell
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

## 13. Clear old Redis state and load the new cache

```powershell
docker compose exec redis redis-cli FLUSHDB
python -m src.training.load_online_cache
```

The loader now stores:

- train-cutoff recipient state for genuine live requests;
- exact `snapshot:txn:<transaction_id>` historical snapshots;
- exact `sequence:txn:<transaction_id>` and `sequence:session:<session_id>` scores;
- training-period payer fallback state only.

## 14. Start the seven-terminal system

### Terminal 1 — Docker

Already running after `docker compose up -d`.

### Terminal 2 — cache loader

Runs once and exits:

```powershell
python -m src.training.load_online_cache
```

### Terminal 3 — FastAPI

```powershell
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Check:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

### Terminal 4 — rolling feature consumer

```powershell
python -m src.streaming.feature_consumer
```

### Terminal 5 — graph worker

```powershell
python -m src.graph.graph_worker --interval-seconds 60
```

### Terminal 6 — feedback consumer

```powershell
python -m src.streaming.feedback_consumer
```

### Terminal 7 — Streamlit

```powershell
python -m streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

## 15. Run the fixed demonstrations

```powershell
python scripts/demo_requests.py auto
python scripts/demo_requests.py hospital
python scripts/demo_requests.py qr_scam
python scripts/demo_requests.py account_takeover
python scripts/demo_requests.py mule
```

The terminal and Streamlit dashboard use the same held-out manifest rows and exact transaction-time expert snapshots.

Expected broad behaviour:

- auto: `ALLOW` or `CONFIRM`
- hospital: `ALLOW` or `CONFIRM`
- QR scam: `HOLD` or `BLOCK`
- account takeover: `STEP_UP`, `HOLD` or `BLOCK`
- mule: `HOLD` or `BLOCK`

## 16. Replay raw transactions through Kafka

```powershell
python -m src.streaming.producer `
  --input data/processed/test.parquet `
  --limit 200 `
  --sleep 0.10
```

This updates Redis rolling features and the near-real-time graph worker. The demo requests separately call the scoring API.

## What each fix changes

### Fix 1 — stable recipient identity

A payee now has one persistent profile, such as transport provider, hospital, merchant, mule or scam recipient. A compromised merchant changes only after a chronological compromise point.

### Fix 2 — separate demo recipients

The manifest requires distinct held-out payees for every demo scenario.

### Fix 3 — transaction/session sequence cache

Sequence risk is looked up by exact transaction or session. A payer's most suspicious historical transaction can no longer overwrite every other transaction.

### Fix 4 — transaction-time recipient risk

TGN scores each transaction using only the payee's earlier events. Complaints are available only after their report timestamp. Online live cache is frozen at the training cutoff, while demos use exact historical snapshots.

### Fix 5 — robust fusion training

Fusion is trained with deliberately missing graph/sequence/complaint inputs and contradictory combinations such as a merchant-looking mule or a legitimate merchant with noisy graph risk. Availability flags tell the model the difference between “zero risk” and “no information.”

### Fix 6 — calibrated fraud types and safe policy ordering

Every fraud type receives its own threshold selected on validation data. High final risk and strong mule/QR specialist consensus are handled before uncertainty, so uncertainty cannot downgrade a severe case to a simple confirmation.
