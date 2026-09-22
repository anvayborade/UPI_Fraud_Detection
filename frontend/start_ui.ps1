$ErrorActionPreference = "Stop"
Write-Host "[1/3] Exporting current model/demo data..." -ForegroundColor Cyan
py -3.11 scripts/export_frontend_data.py

if (-not (Test-Path "node_modules")) {
    Write-Host "[2/3] Installing frontend packages..." -ForegroundColor Cyan
    npm install
} else {
    Write-Host "[2/3] node_modules already present." -ForegroundColor DarkGray
}

Write-Host "[3/3] Starting React UI at http://localhost:5173" -ForegroundColor Green
npm run dev
