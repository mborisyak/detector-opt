#!/bin/bash
# 3070 concurrency: SHiP workload (=bo), no-MPS pass then MPS pass, same levels.
#
# --budget 262144 IS A MEMORY KNOB, NOT A WORKLOAD CHANGE. At the config budget of 1048576 one SHiP
# process holds ~4.7 GB of an 8 GB card, so K=2 would OOM and there would be no curve to measure.
# `budget` sizes the event POOLS only; this probe fills and evaluates exactly `iteration_limit`
# (262144) rows and runs `iteration_limit // batch` = 1024 steps an epoch. Setting budget to
# iteration_limit is therefore the smallest pool that still holds one full window, and every measured
# quantity -- steps per epoch, eval window, batch, network -- is unchanged.
#
# LEVELS stop at 3: even shrunk, a process needs ~2.3 GB and four would not fit in 8 GB.
# Two passes so K=1 appears in BOTH -- that pair is the MPS overhead for a SINGLE process, which is
# otherwise invisible and which the bo sweep never measured.
set -u
cd /home/max/dev/detector-opt
PY=/home/max/opt/pyenv/versions/3.11.9/envs/py3/bin/python
CFG="=bo"
LEVELS="1 2 3"
run_pass() {  # $1 = tag, $2 = outdir
  for K in $LEVELS; do
    echo "=== $1 K=$K ==="
    pids=()
    for ((r=0;r<K;r++)); do
      env XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -u scripts/probe_epoch_cost.py "$CFG" \
        --n-models 1 --epochs 30 --budget 262144 --output "$2/k${K}_r${r}.json" > "$2/k${K}_r${r}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
  done
}
mkdir -p output/ship3070-nomps output/ship3070-mps
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
echo "PASS 1: NO MPS (pipe dir unset)"
run_pass nomps output/ship3070-nomps
export CUDA_MPS_PIPE_DIRECTORY=/home/max/.mps CUDA_MPS_LOG_DIRECTORY=/home/max/.mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
sleep 2
if ! echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>/dev/null | grep -qE '[0-9]'; then
  echo "FATAL: MPS not answering on $CUDA_MPS_PIPE_DIRECTORY -- pass 2 aborted"; exit 1
fi
echo "PASS 2: MPS OK on $CUDA_MPS_PIPE_DIRECTORY"
run_pass mps output/ship3070-mps
echo "BOTH PASSES COMPLETE"
