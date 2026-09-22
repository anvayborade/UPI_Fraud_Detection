$ErrorActionPreference = "Stop"

python scripts/verify_multisource_data.py
python -m src.data.prepare_multisource `
  --paysim-rows $(if ($env:PAYSIM_ROWS) { $env:PAYSIM_ROWS } else { "30000" }) `
  --banksim-rows $(if ($env:BANKSIM_ROWS) { $env:BANKSIM_ROWS } else { "30000" }) `
  --ibm-negative-ratio $(if ($env:IBM_NEGATIVE_RATIO) { $env:IBM_NEGATIVE_RATIO } else { "10" }) `
  --ibm-max-negatives-per-file $(if ($env:IBM_MAX_NEGATIVES_PER_FILE) { $env:IBM_MAX_NEGATIVES_PER_FILE } else { "20000" })
python -m src.training.train_all --skip-prepare
python -m src.training.evaluate

Write-Host "Offline training and evaluation are complete."
Write-Host "Next run: docker compose up -d"
Write-Host "Then: python -m src.training.load_online_cache"
Write-Host "Start the API and dashboard in separate terminals:"
Write-Host "  uvicorn src.api.main:app --reload"
Write-Host "  streamlit run dashboard/app.py"
