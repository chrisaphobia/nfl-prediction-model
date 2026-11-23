#!/usr/bin/env bash
# Runs the whole project from start to finish: download data, score the models, print a summary.
# Usage:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! python -c "import nflreadpy, sklearn, pyarrow" 2>/dev/null; then
  echo "== Installing requirements"
  pip install -q -r requirements.txt
fi

echo "== Step 1 of 3: downloading NFL data (takes a few minutes the first time)"
python src/data_prep.py

echo "== Step 2 of 3: scoring every model on seasons it never saw (writes results/metrics.md)"
python src/evaluate.py

echo "== Step 3 of 3: plain-language summary with example predictions"
python src/model.py

echo
echo "Done. Results are in results/metrics.md. Live 2026 picks update automatically every morning in picks/2026/scorecard.md."
