#!/usr/bin/env bash
# Reproduce the CPU half of the paper: baselines, both model families, the full
# ablation study and the additional dataset.
#
#   bash scripts/run_all.sh                 # everything
#   THREADS=4 bash scripts/run_all.sh       # on a smaller machine
#
# Stages:
#   1. reference baselines on the local splits                        (seconds)
#   2. 22 training runs, in parallel under a CPU thread budget:
#        2 main models + 16 ablations + 4 additional-dataset runs
#   3. the submission members retrained on the full public history
#   4. RevIN-free variants at full budget
#   5. blend, report tables and figures
#
# Everything here runs on CPU. On 6 physical cores the pipeline takes roughly
# 3-4 hours of wall clock; scripts/run_parallel.py prints per-job timings.
#
# The *submitted* models are not produced here. Under this budget an epoch sees a
# few per cent of the available training windows, which is enough to rank each
# ablation against its own reference but not enough to rank the two architectures
# against each other -- see finding #3 in the README. Run scripts/run_gpu_grid.py
# on a GPU for the models that actually shipped; stage 5 below then picks them up
# automatically, because it blends the best runs present in results/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"
THREADS="${THREADS:-8}"
mkdir -p "$ROOT/results" "$ROOT/runs" "$ROOT/logs"

echo "=== 1/5 reference baselines ==="
python "$ROOT/scripts/run_local_baselines.py" --train "$DATA/train.csv" \
  --out "$ROOT/results/baselines_local.json" 2>&1 | tee "$ROOT/logs/baselines.log"

echo "=== 2/5 main models, ablations and additional dataset (parallel) ==="
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
# Hand the greedy search the eight best runs that have a checkpoint on disk rather
# than one per family: it assigns zero weight to members it cannot use, so a larger
# pool costs search time and nothing else, and members that are individually weaker
# still earn weight when their errors are decorrelated.
mapfile -t MEMBERS < <(python - "$ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "scripts"))
from variants import load_runs  # noqa: E402

runs = load_runs(root / "results")
for tag, _ in sorted(runs.items(), key=lambda kv: kv[1]["val"]["wape"])[:8]:
    checkpoint = root / "runs" / tag / "checkpoint.pt"
    if checkpoint.exists():
        print(checkpoint)
PY
)
if [ "${#MEMBERS[@]}" -eq 0 ]; then
  echo "no run checkpoints found under $ROOT/runs; nothing to blend" >&2
  exit 1
fi
printf 'blending %d candidate members:\n' "${#MEMBERS[@]}"
printf '  %s\n' "${MEMBERS[@]}"
python "$ROOT/scripts/build_ensemble.py" --train "$DATA/train.csv" \
  --members "${MEMBERS[@]}" \
  --out "$ROOT/results/ensemble.json" 2>&1 | tee "$ROOT/logs/ensemble.log"
python "$ROOT/scripts/make_report_tables.py"
python "$ROOT/scripts/make_figures.py"

echo
echo "next steps:"
echo "  python scripts/make_submission.py     # build final_submission.zip"
echo "  python scripts/verify_submission.py   # rehearse the private evaluation"
echo "  cd report && tectonic -X compile report.tex"
