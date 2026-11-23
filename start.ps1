# Lance l'application complète : API (FastAPI, port 8000) + interface web (Next.js, port 3000).
# Usage, depuis la racine du projet :   powershell -ExecutionPolicy Bypass -File .\start.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Environnement Python introuvable (.venv). Installez-le d'abord :" -ForegroundColor Red
    Write-Host "  python -m venv .venv ; .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path (Join-Path $root "models\agent_profil.joblib"))) {
    Write-Host "Modèles introuvables : exécutez d'abord le notebook notebooks\Projet_MultiAgent_Portefeuille_v4.ipynb" -ForegroundColor Red
    exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "Node.js / npm introuvable : installez Node.js (https://nodejs.org)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $root "web\node_modules"))) {
    Write-Host "Première utilisation : installation des paquets de l'interface (quelques minutes)..." -ForegroundColor Yellow
    Push-Location (Join-Path $root "web"); npm install; Pop-Location
}

Write-Host "Démarrage de l'API sur http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; & '$python' -m uvicorn api.main:app --port 8000"

Write-Host "Démarrage de l'interface sur http://localhost:3000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\web'; npm run dev"

Start-Sleep -Seconds 8
Start-Process "http://localhost:3000"
Write-Host "Application lancée. Fermez les deux fenêtres PowerShell pour l'arrêter." -ForegroundColor Green
