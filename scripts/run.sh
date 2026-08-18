#!/usr/bin/env bash
# Start the API and the UI together, and wait until both answer.
#
#   ./scripts/run.sh              # API on :8000, UI on :5173
#   ./scripts/run.sh 8010 5174    # pick ports
#
# Ctrl-C stops both. Logs go to .run/api.log and .run/ui.log.

set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${1:-8000}"
UI_PORT="${2:-5173}"
# LAN=1 binds every interface. No authentication on the API, so trusted
# networks only.
BIND_HOST=$([[ "${LAN:-0}" == "1" ]] && echo "0.0.0.0" || echo "127.0.0.1")
mkdir -p .run

if [[ ! -d .venv ]]; then
  echo "No .venv — run ./scripts/setup.sh first." >&2
  exit 1
fi
if [[ ! -f data/index/stats.json ]]; then
  echo "No index in data/index — run ./scripts/setup.sh first." >&2
  exit 1
fi

cleanup() {
  echo
  echo "stopping..."
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "${UI_PID:-}" ]] && kill "$UI_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "starting API on :$API_PORT ..."
TOKENIZERS_PARALLELISM=false \
  .venv/bin/uvicorn rag.server:app --host "$BIND_HOST" --port "$API_PORT" \
  > .run/api.log 2>&1 &
API_PID=$!

# The encoder and indexes load before the first request, so this is not instant.
until curl -sf -m 2 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API failed to start. Last lines of .run/api.log:" >&2
    tail -20 .run/api.log >&2
    exit 1
  fi
  sleep 2
done

.venv/bin/python - "$API_PORT" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/meta") as r:
    m = json.load(r)
voice = "stubbed (no SARVAM_API_KEY)" if m["mock_voice"] else "live"
names = ", ".join(l["name"] for l in m["languages"])
print(f"  corpus  {m['corpus']['passages']} passages, {m['corpus']['sentences']} sentences")
print(f"  voice   {voice}")
print(f"  langs   {len(m['languages'])} — {names}")
PY

echo "starting UI on :$UI_PORT ..."
# VITE_API_BASE here is the proxy target for the dev server, not a value
# baked into the client bundle.
(cd web && VITE_API_BASE="http://127.0.0.1:$API_PORT" \
  npx vite --port "$UI_PORT" --host 127.0.0.1 > ../.run/ui.log 2>&1) &
UI_PID=$!
until curl -sf -m 2 "http://127.0.0.1:$UI_PORT" >/dev/null 2>&1; do
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    echo "UI failed to start. Last lines of .run/ui.log:" >&2
    tail -20 .run/ui.log >&2
    exit 1
  fi
  sleep 1
done

cat <<EOF

  Open  http://127.0.0.1:$UI_PORT

  The microphone needs localhost or HTTPS — 127.0.0.1 qualifies, so it works.
  API docs at http://127.0.0.1:$API_PORT/docs

  Ctrl-C to stop both.
EOF

wait
