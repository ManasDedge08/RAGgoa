# Start the API and the UI together on Windows. Counterpart of run.sh.
#
#   .\scripts\run.ps1
#   .\scripts\run.ps1 -ApiPort 8010 -UiPort 5174
#
# Ctrl-C stops both. Logs go to .run\api.log and .run\ui.log.

[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 5173
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"

New-Item -ItemType Directory -Force -Path ".run" | Out-Null

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "No .venv - run .\scripts\setup.ps1 first."
}
if (-not (Test-Path "data\index\stats.json")) {
    throw "No index in data\index - run .\scripts\setup.ps1 first."
}

$api = $null
$ui = $null

function Stop-All {
    Write-Host ""
    Write-Host "stopping..."
    foreach ($p in @($script:api, $script:ui)) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    Write-Host "starting API on :$ApiPort ..."
    $api = Start-Process -FilePath ".venv\Scripts\python.exe" `
        -ArgumentList "-m", "uvicorn", "rag.server:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput ".run\api.log" -RedirectStandardError ".run\api.err.log"

    # The encoder and indexes load before the first request, so this is not instant.
    while ($true) {
        if ($api.HasExited) {
            Write-Host "API failed to start. Last lines of .run\api.err.log:" -ForegroundColor Red
            Get-Content ".run\api.err.log" -Tail 20 -ErrorAction SilentlyContinue
            exit 1
        }
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2 -UseBasicParsing | Out-Null
            break
        } catch { Start-Sleep -Seconds 2 }
    }

    $meta = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/meta" -TimeoutSec 10
    $voice = if ($meta.mock_voice) { "stubbed (no SARVAM_API_KEY)" } else { "live" }
    $names = ($meta.languages | ForEach-Object { $_.name }) -join ", "
    Write-Host "  corpus  $($meta.corpus.passages) passages, $($meta.corpus.sentences) sentences"
    Write-Host "  voice   $voice"
    Write-Host "  langs   $($meta.languages.Count) - $names"

    Write-Host "starting UI on :$UiPort ..."
    $env:VITE_API_BASE = "http://127.0.0.1:$ApiPort"
    $npx = if (Get-Command npx.cmd -ErrorAction SilentlyContinue) { "npx.cmd" } else { "npx" }
    $ui = Start-Process -FilePath $npx `
        -ArgumentList "vite", "--port", "$UiPort", "--host", "127.0.0.1" `
        -WorkingDirectory (Join-Path (Get-Location) "web") `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput "..\.run\ui.log" -RedirectStandardError "..\.run\ui.err.log"

    while ($true) {
        if ($ui.HasExited) {
            Write-Host "UI failed to start. Last lines of .run\ui.err.log:" -ForegroundColor Red
            Get-Content ".run\ui.err.log" -Tail 20 -ErrorAction SilentlyContinue
            exit 1
        }
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:$UiPort" -TimeoutSec 2 -UseBasicParsing | Out-Null
            break
        } catch { Start-Sleep -Seconds 1 }
    }

    Write-Host ""
    Write-Host "  Open  http://127.0.0.1:$UiPort" -ForegroundColor Green
    Write-Host ""
    Write-Host "  The microphone needs localhost or HTTPS - 127.0.0.1 qualifies, so it works."
    Write-Host "  API docs at http://127.0.0.1:$ApiPort/docs"
    Write-Host ""
    Write-Host "  Ctrl-C to stop both."

    while (-not $api.HasExited -and -not $ui.HasExited) { Start-Sleep -Seconds 1 }
}
finally {
    Stop-All
}
