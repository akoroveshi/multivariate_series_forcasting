#!/usr/bin/env bash
# Reproduce every experiment reported in the paper.
#
#   bash scripts/run_all.sh                 # everything
#   THREADS=4 bash scripts/run_all.sh       # on a smaller machine
#
# Stages:
#   1. reference baselines on the local splits                        (seconds)
#   2. all 22 training runs, in parallel under a CPU thread budget:
#        2 main models + 16 ablations + 4 additional-dataset runs
#   3. the two submission members retrained on the full public history
#   4. blending ensemble, report tables and figures
#
# Everything runs on CPU. On 6 physical cores the whole pipeline takes roughly
# 3-4 hours of wall clock; scripts/run_parallel.py prints per-job timings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"
THREADS="${THREADS:-8}"
mkdir -p "$ROOT/results" "$ROOT/runs" "$ROOT/logs"

echo "=== 1/4 reference baselines ==="
python "$ROOT/scripts/run_local_baselines.py" --train "$DATA/train.csv" \
  --out "$ROOT/results/baselines_local.json" 2>&1 | tee "$ROOT/logs/baselines.log"

echo "=== 2/4 main models, ablations and additional dataset (parallel) ==="
python "$ROOT/scripts/run_parallel.py" --data-dir "$DATA" --threads "$THREADS" \
  --stage main ablations jena 2>&1 | tee "$ROOT/logs/parallel.log"

echo "=== 3/5 submission members on the full public history (parallel) ==="
python "$ROOT/scripts/run_parallel.py" --data-dir "$DATA" --threads "$THREADS" \
  --stage full 2>&1 | tee "$ROOT/logs/parallel_full.log"

# The RevIN ablation is decisive enough to change the architecture, so both
# families are re-trained without it and the validation split picks the winner.
echo "=== 4/5 RevIN-free variants at full budget (parallel) ==="
python "$ROOT/scripts/run_parallel.py" --data-dir "$DATA" --threads "$THREADS" \
  --stage variants 2>&1 | tee "$ROOT/logs/parallel_variants.log"

echo "=== 5/5 ensemble, tables, figures ==="
BEST_PATCH=$(python -c "import sys,json;sys.path.insert(0,'$ROOT/scripts');from pathlib import Path;from variants import select_best;print(select_best(Path('$ROOT/results'))['patchtst'])")
BEST_LSTM=$(python -c "import sys,json;sys.path.insert(0,'$ROOT/scripts');from pathlib import Path;from variants import select_best;print(select_best(Path('$ROOT/results'))['lstm_attention'])")
python "$ROOT/scripts/build_ensemble.py" --train "$DATA/train.csv" \
  --members "$ROOT/runs/$BEST_PATCH/checkpoint.pt" "$ROOT/runs/$BEST_LSTM/checkpoint.pt" \
  --out "$ROOT/results/ensemble.json" 2>&1 | tee "$ROOT/logs/ensemble.log"
python "$ROOT/scripts/make_report_tables.py"
python "$ROOT/scripts/make_figures.py"

echo
echo "next steps:"
echo "  python scripts/make_submission.py     # build final_submission.zip"
echo "  python scripts/verify_submission.py   # rehearse the private evaluation"
echo "  cd report && tectonic -X compile report.tex"
