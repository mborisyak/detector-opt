#!/usr/bin/env bash
# Confirm a concurrency K with REAL bo.py jobs before committing a campaign to it. Run ON bo.
#
#   scripts/confirm_concurrency.sh <K> [budget] [task]
#   scripts/confirm_concurrency.sh 12 262144 enzyme_extremes
#
# WHY THIS EXISTS SEPARATELY FROM scripts/probe_mps.sh. That sweep times the TRAINER, and a campaign
# job is not only its trainer: between designs `bo.py` fits the GP and maximises expected improvement
# -- scipy L-BFGS-B over the design dimension, with Ryser permanents for the permutation-invariant
# kernel -- and that work is pure HOST CPU. The decision log records the ACQUISITION, not the kernel,
# as the cost driver at higher design dimensions. So the trainer sweep is blind to precisely the load
# that would bite at high K, where every one of K processes wants cores for it at once. A K chosen
# from the sweep alone is a K chosen from a workload the campaign does not run.
#
# The unit here is a real `bo.py` run with a REDUCED `training.budget` (the pool it must fill), every
# other setting at campaign values. Budget is the only knob touched, and it decides how many designs
# a run funds, not how any of them are trained -- so the per-design cost, the growth schedule, the
# convergence procedure and the BO layer are all the campaign's own.
#
# Compare the reported aggregate against the sweep's prediction at the same K. If it falls short, the
# acquisition is contending and K should come down; the sweep's number was measured on less work than
# the campaign does.
#
# It writes under output/concurrency-confirm/ and never touches a campaign tree.
set -euo pipefail

K=${1:?usage: confirm_concurrency.sh <K> [budget] [task]}
BUDGET=${2:-262144}
TASK=${3:-enzyme_extremes}

REPO=$HOME/detector-opt
OUTDIR=$REPO/output/concurrency-confirm/k$K
export CUDA_MPS_PIPE_DIRECTORY="$HOME/.mps"

[ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] || { echo "MPS is DOWN -- start it first, or this measures time-slicing" >&2; exit 1; }

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
cd "$REPO"

nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used --format=csv,noheader -l 5 \
  > "$OUTDIR/gpu.csv" 2>/dev/null &
SAMPLER=$!

START=$(date +%s)
WORKERS=()
for i in $(seq 1 "$K"); do
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
    python scripts/bo.py "=$TASK" \
      output="$OUTDIR/w$i" seed="$((900000 + i))" nn_init_strategy=from_scratch \
      training.budget="$BUDGET" \
      > "$OUTDIR/w$i.log" 2>&1 &
  WORKERS+=($!)
done
FAILED=0
for pid in "${WORKERS[@]}"; do
  wait "$pid" || FAILED=$((FAILED + 1))
done
END=$(date +%s)
kill "$SAMPLER" 2>/dev/null || true

ELAPSED=$((END - START))
echo "K=$K budget=$BUDGET wall=${ELAPSED}s failed=$FAILED"
"$HOME/venv/bin/python" - "$OUTDIR" "$ELAPSED" "$K" <<'PY'
import glob
import json
import os
import sys

outdir, elapsed, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
calls, designs, runs = 0, 0, 0
for path in sorted(glob.glob(os.path.join(outdir, "w*", "*.json"))):
  if os.path.basename(path) not in ("results.json", "partial.json"):
    continue
  with open(path) as handle:
    record = json.load(handle)
  calls += int(record.get("detector_calls_used", 0))
  designs += int(record.get("n_iterations_completed", 0))
  runs += 1
print(f"runs with a trajectory: {runs}/{k}")
print(f"designs completed:      {designs}  ({designs / max(elapsed, 1) * 3600:.1f} per hour aggregate)")
print(f"detector calls:         {calls}  ({calls / max(elapsed, 1):.0f} per second aggregate)")
PY
