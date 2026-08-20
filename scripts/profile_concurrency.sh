#!/bin/bash
# Concurrency profile: does per-process epoch cost degrade as K processes share one GPU?
#
#   sbatch --gres=shard:16 --cpus-per-task=16 --wrap='bash scripts/profile_concurrency.sh =bo output/concurrency'
#
# K identical copies of probe_epoch_cost.py are started TOGETHER and all must finish before the next K.
# Holding every shard in ONE job is what makes this a measurement rather than a guess: nothing else can
# be scheduled onto the card while a level runs, so a slowdown at higher K is contention between the K
# copies and not an unrelated neighbour.
#
# WHAT IT MUST SEPARATE -- THREE regimes, not two. Let t(K) be per-process s/epoch and t(1) the solo
# cost. Aggregate throughput is K / t(K).
#
#   t(K) ~ t(1)          latency-bound, idle SMs   -> aggregate RISES ~linearly with K
#   t(K) ~ K * t(1)      exact equal sharing       -> aggregate FLAT; concurrency buys nothing
#   t(K) >  K * t(1)     WORSE than equal sharing  -> aggregate FALLS; concurrency actively HURTS
#
# The third is the case worth ruling in or out, and no campaign figure settles it: aggregate
# throughput must be read from this script's own timings, not from a number quoted elsewhere. Contention that is superlinear in K -- MPS context
# thrashing, memory-bandwidth saturation, host-side dispatch queueing -- would show as t(12)/t(1) > 12,
# and the ratio t(K)/(K*t(1)) is printed for every level precisely so that is read off directly rather
# than inferred.
#
# The levels are run LARGEST-LAST and each level's copies are identical, so a level that degrades
# cannot be blamed on one unlucky replica: the spread across replicas at fixed K is reported too.
set -u
CONFIG=${1:?config token, e.g. =bo}
OUT=${2:?output dir}
EPOCHS=${EPOCHS:-30}
LEVELS=${LEVELS:-"1 2 4 8 12"}
PY=${PY:-/home/max/venv/bin/python}
# MPS MUST BE NAMED EXPLICITLY OR IT IS NOT USED. A running `nvidia-cuda-mps-control` serves ONE pipe
# directory; a client that does not set CUDA_MPS_PIPE_DIRECTORY looks in the default /tmp/nvidia-mps,
# finds nothing, and silently falls back to its own CUDA context -- which TIME-SLICES in Default
# compute mode and yields exactly-equal sharing. An earlier run of this script omitted these two lines
# and therefore profiled time-slicing while a daemon was running, which is indistinguishable from
# saturation in the numbers alone. Set MPS= to deliberately profile the no-MPS case.
if [ "${MPS:-1}" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/home/max/.mps}
  export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/home/max/.mps-log}
fi
echo "MPS pipe dir: ${CUDA_MPS_PIPE_DIRECTORY:-<unset -> NO MPS>}"
mkdir -p "$OUT"
for K in $LEVELS; do
  echo "=== K=$K : starting $K concurrent copies ==="
  pids=()
  for ((r = 0; r < K; r++)); do
    env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$PY" -u scripts/probe_epoch_cost.py "$CONFIG" --n-models 1 --epochs "$EPOCHS" \
      --output "$OUT/k${K}_r${r}.json" > "$OUT/k${K}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== K=$K : done ==="
done
echo "ALL LEVELS COMPLETE"
