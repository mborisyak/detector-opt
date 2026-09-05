#!/bin/bash
# Launch the linear-ladder campaign (linear.snake) on the workstation's SLURM, detached, under one lock.
#
#   setsid nohup scripts/run_linear_campaign.sh > logs/linear-local.log 2>&1 < /dev/null &
#   scripts/run_linear_campaign.sh -n                                # dry run, in the foreground
#   scripts/run_linear_campaign.sh output/linear/selected.txt        # stop after the selection
#
# Everything machine-bound is in profiles/workstation/config.yaml (shards per cell, memory, MPS notes); this
# script only guards the launch. It is the workstation twin of scripts/cern/run_campaign.sh.
#
# ⛔️ MPS MUST BE RUNNING (see the profile). Its absence is silent at the GPU and ruinous for throughput, so the
# launcher checks the control socket and refuses to start without a server. The pipe directory is exported here
# so sbatch propagates it to every cell.
#
# ⛔️ `flock -n` SO A SECOND ORCHESTRATOR CANNOT START; two would submit every cell twice. Snakemake's own lock under
# .snakemake/ is the second guard; a stale one from a killed orchestrator is cleared with
# `snakemake -s linear.snake --profile profiles/workstation --unlock`.
#
# Arguments are passed through to snakemake (targets, -n, --unlock, ...). Rerun-incomplete and keep-going come
# from the profile.
set -u
cd "$(dirname "$0")/.." || exit 90
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
if [ -z "$(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null)" ]; then
  echo "no MPS server: start it first (see profiles/workstation/config.yaml)" >&2
  exit 91
fi
mkdir -p logs
exec flock -n /tmp/linear-snakemake.lock snakemake -s linear.snake --profile profiles/workstation "$@"
