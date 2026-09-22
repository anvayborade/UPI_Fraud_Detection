# React Frontend Integration — Safe Frontend-Only Patch

This patch adds only a new `frontend/` folder. It does not replace or edit any existing Python, model, Redis, Kafka, Docker, training, policy, threshold, or data-preparation file.

## Apply

Extract this ZIP into the root of:

```text
UPI_Fraud_Multisource_Data_Project
```

After extraction you should have:

```text
UPI_Fraud_Multisource_Data_Project/frontend/package.json
```

## First run

Keep the existing backend running. Open a new PowerShell terminal:

```powershell
cd "C:\Users\Anvay Borade\OneDrive\Desktop\UPI_Fraud_Multisource_Data_Project\frontend"
npm install
py -3.11 scripts/export_frontend_data.py
npm run dev
```

Open:

```text
http://localhost:5173
```

## Normal later runs

If the model/data have not been retrained, you normally only need:

```powershell
cd "C:\Users\Anvay Borade\OneDrive\Desktop\UPI_Fraud_Multisource_Data_Project\frontend"
npm run dev
```

Re-run the exporter after retraining or when you want the gallery/graphs/timelines to reflect new scored data:

```powershell
py -3.11 scripts/export_frontend_data.py
```

The existing FastAPI URL is `http://localhost:8000`. The React dev UI uses port `5173`.
