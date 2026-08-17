#!/usr/bin/env bash
# One-command setup on a fresh machine.
#
#   ./scripts/setup.sh
#
# Creates the virtualenv, installs dependencies, works around the macOS OpenMP
# clash, builds the corpus and index, calibrates the thresholds, and runs the
# self-tests. Safe to re-run: existing artefacts are reused unless --rebuild is
# passed.
#
# Dataset files already on disk in data/dataset/ are used as-is; anything
# missing is downloaded from the Hub. See data/dataset/README.md.

set -euo pipefail
cd "$(dirname "$0")/.."

REBUILD=0
[[ "${1:-}" == "--rebuild" ]] && REBUILD=1

PYTHON="${PYTHON:-python3}"
VENV=.venv

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "Python environment"
if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
  echo "created $VENV"
else
  echo "reusing $VENV"
fi
"$VENV/bin/pip" -q install --upgrade pip
"$VENV/bin/pip" -q install -r requirements.txt
echo "dependencies installed"

step "macOS OpenMP fix"
# torch, faiss and scikit-learn each vendor a libomp; two in one process abort
# at faiss's first parallel region. No-op on Linux.
"$VENV/bin/python" scripts/fix_macos_openmp.py

step "Credentials"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "created .env from the example — add SARVAM_API_KEY to it."
  echo "Without a key the pipeline still runs; speech and Tier 2 use stubs."
else
  echo ".env present"
fi

step "Dataset"
found=$(find data/dataset -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
echo "$found parquet file(s) in data/dataset — anything missing is downloaded."

step "Corpus"
if [[ $REBUILD -eq 1 || ! -f data/corpus/passages.parquet ]]; then
  "$VENV/bin/python" scripts/prepare_data.py
else
  echo "data/corpus exists — pass --rebuild to regenerate"
fi

step "Index (slowest step: roughly 4 minutes per language)"
if [[ $REBUILD -eq 1 ]]; then
  "$VENV/bin/python" -m rag.index.build --force
  "$VENV/bin/python" -m rag.index.build_baseline
elif [[ ! -f data/index/stats.json ]]; then
  "$VENV/bin/python" -m rag.index.build
  "$VENV/bin/python" -m rag.index.build_baseline
else
  echo "data/index exists — pass --rebuild to regenerate"
fi

step "Calibration"
if [[ $REBUILD -eq 1 || ! -f reports/confidence.json ]]; then
  "$VENV/bin/python" scripts/calibrate.py 200
else
  echo "reports/confidence.json exists — pass --rebuild to recalibrate"
fi

step "Self-tests"
"$VENV/bin/python" scripts/selftest.py

step "Frontend"
if command -v npm >/dev/null 2>&1; then
  (cd web && npm install --silent)
  echo "web dependencies installed"
else
  echo "npm not found — skipping the frontend"
fi

cat <<'DONE'

Ready. To run it:

  .venv/bin/uvicorn rag.server:app --port 8000     # API
  cd web && npm run dev                            # UI on :5173

  .venv/bin/python scripts/smoke.py                # four paths incl. a refusal
  .venv/bin/python benchmark.py --baseline         # writes reports/latency_report.md
DONE
