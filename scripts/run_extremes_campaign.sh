#!/bin/bash
# Launch the extremes campaign (extremes.snake) on the workstation's SLURM, detached, under one lock.
#
#   setsid nohup scripts/run_extremes_campaign.sh > logs/extremes-local.log 2>&1 < /dev/null &
#   scripts/run_extremes_campaign.sh -n                                # dry run, in the foreground
#   scripts/run_extremes_campaign.sh output/extremes/selected.txt      # stop after the selection
#
# The twin of scripts/run_linear_campaign.sh: everything machine-bound is in profiles/workstation-extremes/
# config.yaml (shards per cell, memory), and this script only guards the launch -- MPS must be running (its
# absence is silent at the GPU and ruinous for throughput), and `flock -n` keeps a second orchestrator from
# submitting every cell twice. The lock is its own, so the linear and extremes orchestrators can coexist.
# A stale snakemake lock from a killed orchestrator is cleared with `scripts/run_extremes_campaign.sh --unlock`.
#
# Arguments are passed through to snakemake (targets, -n, --unlock, ...).
set -u
cd "$(dirname "$0")/.." || exit 90
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
if [ -z "$(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null)" ]; then
  echo "no MPS server: start it first (see profiles/workstation/config.yaml)" >&2
  exit 91
fi
mkdir -p logs
exec flock -n /tmp/extremes-snakemake.lock snakemake -s extremes.snake --profile profiles/workstation-extremes "$@"
