#!/usr/bin/env bash
# Concurrency sweep for the cloud campaign box: how many campaign jobs should share this GPU?
#
#   scripts/probe_mps.sh <outdir> on|off <n0> <n_increment> <designs> <level> [<level> ...]
#
#   scripts/probe_mps.sh output/mps-probe/off off 2048 1024 1 1 6
#   scripts/probe_mps.sh output/mps-probe/on  on  2048 1024 1 1 2 4 6 8 12 16
#
# Each level launches N identical workers (scripts/probe_mps.py, same seed, so every worker does the
# SAME work) and waits for all of them. The comparison is therefore "how long does one fixed unit
# take when N of them run at once", and aggregate throughput is N * t_1 / t_N.
#
# WHAT IT MUST SEPARATE is stated in scripts/probe_mps.py's docstring: SM occupancy (MPS is the fix),
# host CPU (24 cores at 4 per job = 6, so a saturation exactly at 6 means the GPU was never the
# constraint), and device memory (expected not to bind at 96 GiB, measured anyway). Running the sweep
# with `off` and with `on` is what separates the first from the other two: without an MPS server the
# processes time-slice and aggregate throughput is flat in N regardless of what else is true.
#
# MPS IS STARTED AND STOPPED BY THIS SCRIPT for the level of the sweep it was asked for, so an `off`
# run cannot accidentally inherit a server left up by an earlier `on` run -- which would read as
# "MPS makes no difference" and is exactly the kind of silent contamination the probe exists to
# avoid. It leaves the server RUNNING after an `on` sweep, because that is the state a campaign
# wants.
#
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is read from the environment and passed to the workers, so the
# partitioned variant is a second sweep rather than a second script:
#
#   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=17 scripts/probe_mps.sh output/mps-probe/on-17 on 2048 1024 1 6
#
# nvidia-smi is sampled every 2 s for the duration of each level into gpu.csv.
#
# READING `nvidia-smi` UNDER MPS. In the process table the Type column shows `M+C` for an MPS CLIENT
# and plain `C` for the MPS server itself (and for any process that is NOT going through MPS). That
# makes it the most direct check that MPS is actually in use: a worker showing `C` where `M+C` was
# expected has silently bypassed the server -- almost always a missing CUDA_MPS_PIPE_DIRECTORY in its
# environment -- and is time-slicing. On Volta and later, clients have isolated address spaces and
# their device memory is reported PER CLIENT, so `nvidia-smi` is usable for per-process footprint
# here; on older hardware it was all lumped under the server. Each worker's probe.json carries the
# device allocator's own figures as well, which are what a memory ceiling should be computed from.
set -euo pipefail

OUTDIR=$1
MPS_MODE=$2
N0=$3
N_INCREMENT=$4
DESIGNS=$5
shift 5
LEVELS=("$@")

CONFIG=${CONFIG:-enzyme_extremes}
SEED=${SEED:-20260816}
export CUDA_MPS_PIPE_DIRECTORY="$HOME/.mps"
export CUDA_MPS_LOG_DIRECTORY="$HOME/.mps/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$OUTDIR"

mps_up() { [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; }

stop_mps() {
  if mps_up; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    sleep 2
  fi
}

start_mps() {
  if ! mps_up; then
    nvidia-cuda-mps-control -d
    sleep 2
  fi
  echo "MPS server: $(echo get_server_list | nvidia-cuda-mps-control 2>&1 | tr '\n' ' ')"
}

if [ "$MPS_MODE" = "on" ]; then
  start_mps
else
  stop_mps
  unset CUDA_MPS_PIPE_DIRECTORY
fi

echo "sweep: config=$CONFIG mps=$MPS_MODE n0=$N0 n_increment=$N_INCREMENT designs=$DESIGNS levels=${LEVELS[*]}"

for N in "${LEVELS[@]}"; do
  LEVELDIR="$OUTDIR/n$N"
  rm -rf "$LEVELDIR"
  mkdir -p "$LEVELDIR"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used \
             --format=csv,noheader -l 2 > "$LEVELDIR/gpu.csv" 2>/dev/null &
  SAMPLER=$!
  START=$(date +%s)
  WORKERS=()
  for i in $(seq 1 "$N"); do
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
      python scripts/probe_mps.py "=$CONFIG" \
        output="$LEVELDIR/w$i" seed="$SEED" designs="$DESIGNS" \
        training.n0="$N0" training.n_increment="$N_INCREMENT" \
        > "$LEVELDIR/w$i.log" 2>&1 &
    WORKERS+=($!)
  done
  FAILED=0
  for pid in "${WORKERS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
  done
  END=$(date +%s)
  kill "$SAMPLER" 2>/dev/null || true
  printf '%s %s %s\n' "$N" "$((END - START))" "$FAILED" >> "$OUTDIR/wall.txt"
  echo "level N=$N: $((END - START)) s wall, $FAILED worker(s) failed"
done

echo "done. aggregate with: python scripts/probe_mps_report.py $OUTDIR"
