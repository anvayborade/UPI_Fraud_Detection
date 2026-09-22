# Multi-source runbook: PaySim + IBM HI/LI Small + BankSim

This runbook replaces the original one-source PaySim workflow. It keeps each dataset in a separate canonical table, creates stage-specific data products, and only combines model scores at the final fusion stage.

## 1. Which dataset is used by each architecture component?

| Architecture component | Main data source | Why |
|---|---|---|
| Rules and statistical anomaly layer | PaySim + BankSim + synthetic UPI fields | These models need transaction amount, velocity, new-payee and contextual signals. |
| LightGBM transaction-risk expert | PaySim + BankSim + synthetic UPI scenarios | PaySim adds transfer fraud; BankSim adds merchant-payment behaviour; synthetic fields add UPI context. |
| Legitimate-novelty/context expert | BankSim legitimate rows + synthetic auto/hospital/travel/rent edge cases + legitimate PaySim rows | This expert learns that a new recipient, large amount or unusual location can still be legitimate. |
| Time-aware Transformer | Synthetic app/session journeys generated from the PaySim/BankSim UPI-like product | None of the public datasets contains real UPI app event sequences, so the sequence layer uses documented synthetic journeys. |
| Money-flow HGT/TGN | IBM HI-Small for training + IBM LI-Small as lower-prevalence holdout + UPI-like PaySim/BankSim graph | IBM provides laundering paths and mule-like money movement. The UPI-like graph adapts the expert to the final project domain. |
| Identity/device graph | Synthetic device, VPA and IP relations added to PaySim/BankSim | IBM and BankSim do not contain genuine UPI device-binding data. |
| Context/trust graph | BankSim merchant categories + synthetic QR, coarse location, complaint and merchant signals | This view distinguishes legitimate merchants, transport providers and gig workers from mules. |
| Recipient-archetype classifier | BankSim + synthetic transport/gig/merchant cases + IBM mule labels in the money-flow view | This prevents high fan-in merchants or drivers from being treated as mules. |
| Graph anomaly detector | HGT/TGN embeddings from all three graph views | Detects unusual structures not fully represented in labels. |
| Complaint/LLM intelligence | Synthetic complaint texts by default; optional de-identified real complaints | Public transaction datasets do not contain UPI complaint narratives. |
| Mixture-of-experts fusion and policy engine | PaySim/BankSim/synthetic UPI transaction product plus cached expert scores | The final decision is evaluated on UPI-like rows, not directly on raw IBM AML rows. |

## 2. Standardised data products

The preparation command creates these files:

```text
data/processed/
├── canonical/
│   ├── paysim_transactions.parquet
│   ├── banksim_transactions.parquet
│   ├── ibm_hi_small_transactions.parquet
│   └── ibm_li_small_transactions.parquet
│
├── upi_transaction_training.parquet
├── aml_graph_edges.parquet
├── recipient_archetype_training.parquet
├── train.parquet
├── validation.parquet
├── test.parquet
├── edge_cases.parquet
└── source_manifest.json
```

After training, it also creates:

```text
data/processed/
├── fusion_training.parquet
├── scored_transactions.parquet
└── scored_transactions.pkl
```

### Canonical transaction schema

Every raw source is first mapped to:

```text
transaction_id
timestamp
payer_id
payee_id
amount
currency
payment_channel
merchant_category
source_dataset
source_partition
transaction_fraud_label
laundering_label
```

The two source labels remain separate. A laundering label is never silently treated as an ordinary transaction-fraud label.

## 3. Install and authenticate Kaggle

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the correct PyTorch build from the official PyTorch selector, then:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Install/configure Kaggle if needed:

```powershell
pip install kaggle
```

In Kaggle, open **Settings → API → Create New Token**. Move the downloaded file to:

```text
C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
```

Confirm authentication:

```powershell
kaggle datasets list -s paysim
```

## 4. Download the three required datasets

The repository downloader retrieves PaySim, BankSim and only the four IBM Small files required for this prototype.

```powershell
python scripts/download_data.py all
```

Equivalent separate commands:

```powershell
python scripts/download_data.py paysim
python scripts/download_data.py banksim
python scripts/download_data.py ibm_aml
```

Expected raw layout:

```text
data/raw/
├── paysim/
│   └── PS_20174392719_1491204439457_log.csv
├── banksim/
│   └── <BankSim transaction CSV>
└── ibm_aml/
    ├── HI-Small_Trans.csv
    ├── HI-Small_Patterns.txt
    ├── LI-Small_Trans.csv
    └── LI-Small_Patterns.txt
```

Validate the downloads:

```powershell
python scripts/verify_multisource_data.py
```

The IBM pattern files are downloaded and retained for later typology-level evaluation. The current integrated model trains from the labelled transaction CSVs.

## 5. Prepare all canonical and stage-specific products

Start with the laptop-friendly prototype size:

```powershell
python -m src.data.prepare_multisource `
  --paysim-rows 30000 `
  --banksim-rows 30000 `
  --ibm-negative-ratio 10 `
  --ibm-max-negatives-per-file 20000
```

What the command does:

1. Standardises PaySim, BankSim, IBM HI-Small and IBM LI-Small separately.
2. Preserves source labels in separate columns.
3. Prefixes entity IDs so accounts from different datasets never collide.
4. Creates the UPI-like transaction product from PaySim and BankSim.
5. Adds synthetic UPI device, QR, VPA, location, complaint and session context.
6. Adds hard-negative cases such as auto, hospital, travel and legitimate high-fan-in merchants.
7. Samples all IBM laundering rows plus a controlled number of legitimate rows.
8. Derives IBM money-flow features such as fan-in, fan-out, flow-through ratio and holding time.
9. Creates temporal train/validation/test splits for the UPI-like transaction product.
10. Creates event sequences for the time-aware Transformer.

Inspect the manifest:

```powershell
Get-Content data/processed/source_manifest.json
```

Inspect source/scenario counts:

```powershell
python -c "import pandas as pd; d=pd.read_parquet('data/processed/upi_transaction_training.parquet'); print(d['source_dataset'].value_counts()); print(d['scenario'].value_counts())"
```

## 6. Train the complete model stack

Training does not require Kafka or Redis yet.

```powershell
python -m src.training.train_all --skip-prepare
```

Training order:

1. LightGBM transaction-risk expert
2. LightGBM legitimate-novelty expert
3. Time-aware Transformer
4. Money-flow HGT using UPI-like rows + IBM HI/LI graph rows
5. Identity/device HGT using UPI-like rows only
6. Context/trust HGT using UPI-like rows only
7. Multi-view graph attention
8. TGN-style temporal money-flow model, trained without IBM LI-Small
9. Graph anomaly model
10. Complaint intelligence
11. Mixture-of-experts fusion
12. Uncertainty and policy decisions

IBM HI-Small contributes to graph training. IBM LI-Small is excluded from TGN training and is reserved as a lower-illicit-ratio holdout. The HGT money-flow view also uses source-aware masks that place HI accounts in training and LI accounts in testing.

Optional external LLM complaint extraction:

```powershell
python -m src.training.train_all --skip-prepare --use-llm
```

## 7. Evaluate

```powershell
python -m src.training.evaluate
```

Outputs:

```text
models/metrics.json
models/evaluation_report.json
```

The report includes:

- overall PR-AUC, ROC-AUC and Brier score;
- false-positive performance on legitimate novelty;
- action distribution;
- scenario-level results;
- PaySim-versus-BankSim source-level results.

## 8. Start the online simulation

Start Kafka and Redis:

```powershell
docker compose up -d
docker compose ps
```

Load trained recipient and sequence scores into Redis:

```powershell
python -m src.training.load_online_cache
```

### Terminal 1 — API

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs`.

### Terminal 2 — rolling-feature consumer

```powershell
python -m src.streaming.feature_consumer
```

### Terminal 3 — graph refresh worker

```powershell
python -m src.graph.graph_worker --interval-seconds 60
```

### Terminal 4 — feedback consumer

```powershell
python -m src.streaming.feedback_consumer
```

### Terminal 5 — dashboard

```powershell
streamlit run dashboard/app.py
```

### Terminal 6 — transaction replay

```powershell
python -m src.streaming.producer --input data/processed/test.parquet --limit 200 --sleep 0.05
```

## 9. Run the five important demonstrations

```powershell
python scripts/demo_requests.py auto
python scripts/demo_requests.py hospital
python scripts/demo_requests.py qr_scam
python scripts/demo_requests.py account_takeover
python scripts/demo_requests.py mule
```

Expected behaviour:

| Scenario | Expected action |
|---|---|
| Long-distance auto payment | ALLOW or CONFIRM |
| Emergency hospital payment | ALLOW or CONFIRM |
| Fake-refund QR scam | WARN, STEP_UP or HOLD |
| Account takeover | STEP_UP, HOLD or BLOCK |
| Mule recipient | HOLD or BLOCK |

## 10. Scale up only after the first successful run

The graph models currently use full-graph HGT layers, so do not start with millions of rows on a laptop.

A larger second run can use:

```powershell
python -m src.data.prepare_multisource `
  --paysim-rows 100000 `
  --banksim-rows 100000 `
  --ibm-negative-ratio 10 `
  --ibm-max-negatives-per-file 50000
```

For the full datasets, first replace full-batch HGT with neighbour sampling or cluster mini-batching.
