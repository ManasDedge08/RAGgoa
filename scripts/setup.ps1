# One-command setup on Windows. The PowerShell counterpart of setup.sh.
#
#   .\scripts\setup.ps1
#   .\scripts\setup.ps1 -Rebuild
#
# Creates the virtualenv, installs dependencies, builds the corpus and index,
# calibrates the thresholds, and runs the self-tests. Safe to re-run.
#
# Dataset files already in data\dataset are used as-is; anything missing is
# downloaded from the Hub. See data\dataset\README.md.

[CmdletBinding()]
param(
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Indic text in console output needs a UTF-8 code page on Windows.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

$venvPython = ".venv\Scripts\python.exe"
$venvPip = ".venv\Scripts\pip.exe"

Step "Python environment"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $python) {
    throw "Python was not found on PATH. Install Python 3.12 or newer from python.org."
}

$version = & $python.Source -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "found Python $version"
if ([version]$version -lt [version]"3.10") {
    throw "Python $version is too old; 3.10 or newer is required."
}

if (-not (Test-Path $venvPython)) {
    & $python.Source -m venv .venv
    Write-Host "created .venv"
} else {
    Write-Host "reusing .venv"
}
& $venvPip install --quiet --upgrade pip
& $venvPip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
Write-Host "dependencies installed"

Step "OpenMP"
# torch and faiss each bundle an OpenMP DLL. Unlike macOS these cannot be
# symlinked together, so rag/__init__.py sets KMP_DUPLICATE_LIB_OK and pins
# faiss to one thread. Nothing to do here beyond confirming the import works.
& $venvPython -c "import rag, faiss, torch; print('torch', torch.__version__, '| faiss', faiss.__version__)"
if ($LASTEXITCODE -ne 0) { throw "torch and faiss could not be imported together" }

Step "Credentials"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "created .env from the example - add SARVAM_API_KEY to it."
    Write-Host "Without a key the pipeline still runs; speech and Tier 2 use stubs."
} else {
    Write-Host ".env present"
}

Step "Dataset"
$found = @(Get-ChildItem -Path "data\dataset" -Filter *.parquet -Recurse -ErrorAction SilentlyContinue).Count
Write-Host "$found parquet file(s) in data\dataset - anything missing is downloaded."

Step "Corpus"
if ($Rebuild -or -not (Test-Path "data\corpus\passages.parquet")) {
    & $venvPython scripts\prepare_data.py
    if ($LASTEXITCODE -ne 0) { throw "corpus build failed" }
} else {
    Write-Host "data\corpus exists - pass -Rebuild to regenerate"
}

Step "Index (slowest step: roughly 4 minutes per language, longer without a GPU)"
if ($Rebuild) {
    & $venvPython -m rag.index.build --force
    if ($LASTEXITCODE -ne 0) { throw "index build failed" }
    & $venvPython -m rag.index.build_baseline
} elseif (-not (Test-Path "data\index\stats.json")) {
    & $venvPython -m rag.index.build
    if ($LASTEXITCODE -ne 0) { throw "index build failed" }
    & $venvPython -m rag.index.build_baseline
} else {
    Write-Host "data\index exists - pass -Rebuild to regenerate"
}

Step "Calibration"
if ($Rebuild -or -not (Test-Path "reports\confidence.json")) {
    & $venvPython scripts\calibrate.py 200
} else {
    Write-Host "reports\confidence.json exists - pass -Rebuild to recalibrate"
}

Step "Self-tests"
& $venvPython scripts\selftest.py
if ($LASTEXITCODE -ne 0) { throw "self-tests failed" }

Step "Frontend"
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location web
    npm install --silent
    Pop-Location
    Write-Host "web dependencies installed"
} else {
    Write-Host "npm not found - skipping the frontend. Install Node 20+ from nodejs.org."
}

Write-Host ""
Write-Host "Ready. To run it:" -ForegroundColor Green
Write-Host "  .\scripts\run.ps1"
