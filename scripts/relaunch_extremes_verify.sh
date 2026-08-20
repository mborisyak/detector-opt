#!/bin/bash
# Overnight: verify the extremes campaign cells that are still unverified.
#
# Skips any cell that already has a verification.json, so it is safe to re-run: two cells
# (1244111331/from_scratch and /meta) completed before the sweep was cancelled and are NOT redone.
# One job per cell, 2 shards, so two run at a time. `verify.n_points: 10` and `epochs: 200` in
# config/enzyme_extremes_cont_p8e3.yaml make each cell 1.5-2 h -- budget 9-12 h for the ten.
set -u
cd /home/max/dev/detector-opt
PY=${PY:-/home/max/opt/pyenv/versions/3.11.9/envs/py3/bin/python}
CPUS=${CPUS:-4}
for d in output/extremes-p8e3/*/*/; do
  run=${d%/}
  [ -f "$run/verification.json" ] && { echo "  skip $(echo $run | cut -d/ -f3-4) (already verified)"; continue; }
  seed=$(echo "$run" | cut -d/ -f3)
  id=$(sbatch --parsable --gres=shard:1 --cpus-per-task="$CPUS" --mem-per-cpu=2500 --time=04:00:00 \
    -J "vfy-ext-$(basename "$run")-$seed" -o "$run/verify.log" \
    --wrap="env XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u scripts/verify_trajectory.py \
'=enzyme_extremes_cont_p8e3' trajectory=$run seed=$seed progress=plain")
  echo "  job $id  $(echo $run | cut -d/ -f3-4)"
done
