#!/bin/bash
# Wait for the draining linear orchestrator to exit, then relaunch it under the MPS profile.
#
#   scripts/resubmit_linear_mps.sh
#
# WHY A WAITER. `snakemake` under SIGTERM stops submitting but keeps running jobs to completion, so
# the moment to relaunch is "orchestrator gone", not "some time from now". Relaunching early would
# put two orchestrators on the same working directory and the second would fail on the lock.
#
# ⚠️ MPS IS EXPORTED HERE, NOT IN THE SNAKEFILE. `sbatch` propagates the submitting environment, so
# the pipe directory has to be in THIS shell for the cells to become MPS clients. The snakefile names
# no machine and must not learn about MPS.
set -u
cd "$(dirname "$0")/.." || exit 90

exec 200>/tmp/resubmit-linear.lock
flock -n 200 || { echo "another copy is running -- exiting"; exit 0; }

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PATH="$PATH"

while ps -eo args | grep -q "[s]nakemake -s linear.snake"; do
  echo "[$(date +%H:%M:%S)] draining: $(squeue -h -o '%i %j %T' | awk '$2 ~ /^cf7ef182/ && $3=="RUNNING"' | wc -l) cell(s) still running"
  sleep 60
done
echo "[$(date +%H:%M:%S)] drain complete"

if ! pgrep -u "$USER" -x nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] MPS control daemon is NOT running -- starting it"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  sleep 3
fi
echo "[$(date +%H:%M:%S)] MPS thread pct: $(echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>&1 | head -1)"

snakemake -s linear.snake --unlock >/dev/null 2>&1
echo "[$(date +%H:%M:%S)] relaunching under profiles/workstation (shard:3, local_gpu=4)"
exec flock -n /tmp/linear-local.lock \
  snakemake -s linear.snake --profile profiles/workstation output/linear/selected.txt
