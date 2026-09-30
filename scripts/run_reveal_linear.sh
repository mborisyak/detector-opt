#!/bin/bash
# Launch the linear reveal ablation (ablation_reveal_linear.snake) on the workstation's SLURM, detached, under its
# own lock, beside the campaign's orchestrator.
#
#   setsid nohup scripts/run_reveal_linear.sh > logs/ablation-reveal-linear.log 2>&1 < /dev/null &
#   scripts/run_reveal_linear.sh -n                                # dry run, in the foreground
#
# The twin of scripts/run_linear_campaign.sh: the same MPS guard and the same profile (shards, memory, squeue status
# command), a lock of its own, and `--nolock` because the campaign's snakemake holds the working directory's lock
# (the two DAGs share no output path). PYTHONPATH pins the sourceless gearup snapshot the linear campaign runs on
# (output/gearup-snapshot-2026-09-07/README.md): the editable gearup in ~/dev/gearup is mid-edit and broken.
set -u
cd "$(dirname "$0")/.." || exit 90
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PYTHONPATH="$PWD/output/gearup-snapshot-2026-09-07${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "$(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null)" ]; then
  echo "no MPS server: start it first (see profiles/workstation/config.yaml)" >&2
  exit 91
fi
mkdir -p logs
exec flock -n /tmp/reveal-linear-snakemake.lock \
  snakemake -s ablation_reveal_linear.snake --profile profiles/workstation --nolock "$@"
