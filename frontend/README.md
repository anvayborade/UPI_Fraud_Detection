# UPI Fraud Intent Firewall — React Intelligence Console

This folder replaces the **presentation layer only**. It does not modify the Python models, FastAPI service, Redis, Kafka, training pipeline, thresholds, or policy engine.

## What is implemented

- Executive / Technical explanation modes
- Decision Command Center with live `/score` and `/precheck`
- 12-scenario Edge-Case Gallery without retraining
- Model Debate view: suspicious vs protective experts
- Counterfactual What-If simulator
- Recipient graph explorer from transaction-time prior history
- Payer transaction timeline
- Performance dashboard from `models/metrics.json` and `models/evaluation_report.json`
- Leakage / temporal integrity guardrail page
- Full architecture visualization
- Complaint Intelligence lab using the existing `/complaint` endpoint
- Live API / Redis / model health indicators
- Raw request and response views in Technical mode

## Folder placement

Place this folder directly inside your existing repository:

```text
UPI_Fraud_Multisource_Data_Project/
├── src/
├── scripts/
├── data/
├── models/
├── dashboard/              # old Streamlit UI can remain untouched
├── docker-compose.yml
└── frontend/               # this React UI
```

## One-time setup

Install Node.js 22 or newer, then from the project root:

```powershell
cd frontend
npm install
```

## Export the current trained/test data into browser-readable JSON

This does **not retrain anything**:

```powershell
cd frontend
py -3.11 scripts/export_frontend_data.py
```

The exporter:

1. Reads `../data/processed/scored_transactions.pkl`.
2. Restricts demo selection to `../data/processed/test.parquet` when available.
3. Preserves the five existing `demo_manifest.json` choices.
4. Adds additional existing held-out scenarios without retraining.
5. Creates prior-only graph and timeline data for visualization.
6. Copies `../models/metrics.json` and `../models/evaluation_report.json` into `frontend/public/`.

Re-run this export after a future retraining so the React UI reflects the new model artefacts.

## Normal startup order

Start the existing backend exactly as before. From the project root:

```powershell
docker compose up -d
```

Then your existing Python service terminals:

```powershell
py -3.11 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

```powershell
py -3.11 -m src.streaming.feature_consumer
```

```powershell
py -3.11 -m src.graph.graph_worker --interval-seconds 60
```

```powershell
py -3.11 -m src.streaming.feedback_consumer
```

Then replace the Streamlit terminal with:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

Alternatively, inside `frontend` run:

```powershell
.\start_ui.ps1
```

That exports current data, installs packages when needed, and launches the UI.

## Redis cache

The React frontend does not change cache behavior. If your Redis transaction snapshots still exist after Docker restart, do not reload them. If they are missing/expired, use your existing command:

```powershell
py -3.11 -m src.training.load_online_cache
```

## Environment

Default API URL:

```text
http://localhost:8000
```

To override it, copy `.env.example` to `.env` and change:

```text
VITE_API_URL=http://localhost:8000
```

## Important interpretation of the What-If Lab

The What-If Lab retains `reference_transaction_id` and `simulation_mode=true`. Fast-path request fields can be changed, while cached graph/sequence state remains anchored to the historical reference transaction. The UI labels this explicitly as a **counterfactual stress test**, not a new held-out evaluation sample.

## Build a production bundle

```powershell
npm run build
```

The static bundle is created in:

```text
frontend/dist/
```
