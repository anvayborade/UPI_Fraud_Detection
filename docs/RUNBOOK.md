# Step-by-step runbook

This runbook explains the exact implementation order and which command to run after each stage.

## Stage 0 — Prerequisites

Install:

1. Python 3.11/3.12 64-bit
2. Git
3. Docker Desktop
4. JDK 17 and Android Studio only for the Android client
5. Optional CUDA-compatible driver/GPU

Create and activate a virtual environment, install the correct PyTorch build from the official selector, then install `requirements.txt`.

## Stage 1 — Verify the codebase

```bash
python -m compileall src scripts dashboard
python -m pytest -q
```

At this point, tests that do not require Kafka/Redis should pass. Infrastructure tests are intentionally isolated.

## Stage 2 — Acquire/generate data

Optional PaySim download:

```bash
python scripts/download_data.py paysim
```

Generate the complete modelling tables:

```bash
python -m src.data.prepare_dataset --rows 30000
```

Inspect:

```bash
jupyter lab notebooks/01_paysim_analysis.ipynb
```

## Stage 3 — Inspect scenario generation

Open/run:

```bash
jupyter lab notebooks/02_upi_scenario_generation.ipynb
```

Confirm that the following appear in `scenario` counts:

- legitimate auto/travel/hospital/gig-worker/new-merchant cases;
- fake-refund QR and collect scams;
- account takeover;
- mule-directed transactions;
- remote-access-guided scams.

## Stage 4 — Train fast-path experts

The full trainer invokes them automatically, but they can also be studied through:

```bash
jupyter lab notebooks/03_lightgbm_and_context.ipynb
```

Run the full trainer when ready:

```bash
python -m src.training.train_all --skip-prepare
```

## Stage 5 — Sequence Transformer

The trainer reads `data/sequences/events.parquet`, groups events by transaction/session and trains a small Transformer. Study it through:

```bash
jupyter lab notebooks/04_time_transformer.ipynb
```

## Stage 6 — Heterogeneous graphs

Graph views are written under `data/graph_views/`. Inspect schema and node/edge counts:

```bash
jupyter lab notebooks/05_heterogeneous_graph.ipynb
```

## Stage 7 — Temporal graph and HGT

Study temporal messages:

```bash
jupyter lab notebooks/06_heterogeneous_tgn.ipynb
```

Study typed snapshot attention:

```bash
jupyter lab notebooks/07_hgt.ipynb
```

## Stage 8 — Multi-view fusion

```bash
jupyter lab notebooks/08_multiview_attention.ipynb
```

This combines money-flow, identity/device and context/trust embeddings using transaction-conditioned attention.

## Stage 9 — Final fusion/evaluation

```bash
jupyter lab notebooks/09_fusion_and_evaluation.ipynb
python -m src.training.evaluate
```

## Stage 10 — Infrastructure and online serving

```bash
docker compose up -d
python -m src.training.load_online_cache
```

Terminal A:

```bash
uvicorn src.api.main:app --reload
```

Terminal B:

```bash
python -m src.streaming.feature_consumer
```

Terminal C:

```bash
python -m src.graph.graph_worker --interval-seconds 60
```

Terminal D:

```bash
streamlit run dashboard/app.py
```

Terminal E:

```bash
python -m src.streaming.producer --limit 200 --sleep 0.05
```

## Stage 11 — Validate edge cases

```bash
python scripts/demo_requests.py auto
python scripts/demo_requests.py hospital
python scripts/demo_requests.py qr_scam
python scripts/demo_requests.py account_takeover
python scripts/demo_requests.py mule
```

Interpretation:

- auto/hospital: high novelty can coexist with high context legitimacy and low recipient risk;
- QR scam: unverified QR + complaint/remote-guidance signals should raise social-engineering risk;
- account takeover: device/registration/PIN sequence should raise step-up/block risk;
- mule: known/cached graph risk should trigger hold/block.

## Stage 12 — Optional LLM complaint intelligence

Add API credentials to `.env` and run:

```bash
python -m src.training.train_all --skip-prepare --use-llm
```

Or call the API:

```bash
curl -X POST http://localhost:8000/complaint \
  -H "Content-Type: application/json" \
  -d '{"payee_id":"A_SCAM_001","text":"A caller claimed I needed to scan a QR to receive a refund and asked me to enter my PIN.","use_llm":true}'
```

## Stage 13 — Android telemetry demo

1. Start the API.
2. Open `android-client` in Android Studio.
3. Run an emulator.
4. Set API URL to `http://10.0.2.2:8000`.
5. Send telemetry, precheck and score requests.

## Common problems

### `ModuleNotFoundError: torch_geometric`

Install PyTorch first with the correct CPU/CUDA command, then run `pip install torch-geometric`.

### Parquet engine error

```bash
pip install pyarrow
```

### Kafka does not become healthy

```bash
docker compose logs kafka
docker compose down -v
docker compose up -d
```

### Redis connection refused

Start Docker services or change `REDIS_URL` in `.env`.

### Model files missing in API health

Run data preparation and training before the API. The API can return a rule-only fallback for some requests, but the complete demo requires trained artifacts.

### Training is too slow

- use `--rows 5000` for the first run;
- reduce epochs in `src/training/train_all.py`;
- run on a CUDA machine;
- train each notebook component separately;
- increase to 30,000–100,000 rows for final experiments.
