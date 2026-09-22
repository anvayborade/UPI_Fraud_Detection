#!/usr/bin/env bash
set -euo pipefail

python scripts/verify_multisource_data.py
python -m src.data.prepare_multisource \
  --paysim-rows "${PAYSIM_ROWS:-30000}" \
  --banksim-rows "${BANKSIM_ROWS:-30000}" \
  --ibm-negative-ratio "${IBM_NEGATIVE_RATIO:-10}" \
  --ibm-max-negatives-per-file "${IBM_MAX_NEGATIVES_PER_FILE:-20000}"
python -m src.training.train_all --skip-prepare
python -m src.training.evaluate

echo "Offline training and evaluation are complete."
echo "Next run: docker compose up -d"
echo "Then: python -m src.training.load_online_cache"
echo "Start the API and dashboard in separate terminals:"
echo "  uvicorn src.api.main:app --reload"
echo "  streamlit run dashboard/app.py"
