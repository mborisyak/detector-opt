#!/usr/bin/env bash
# Submit one 10-seed x 4-arm `linear` campaign to the local SLURM cluster, one cell per job.
#
#   scripts/launch_linear_ladder.sh <config> <output_dir> <job_prefix> [count] [skip]
#   scripts/launch_linear_ladder.sh linear_d1n2_fine output/linear-d1n2-10seed D1
#   scripts/launch_linear_ladder.sh linear_d3n4_fine output/linear-10seed L 10 10
#
# THE SEEDS ARE DERIVED, NOT LISTED, AND THE GENERATOR IS THE SNAKEFILE'S: `random.Random(123456)`
# with `randint(0, 2**31 - 1)`, which is `SUPER_SEED` and the exact call at `Snakefile:215-217`.
# Every campaign in this repository draws from that one stream, so seed k means the same thing
# everywhere and the rungs of the dimension ladder are paired seed for seed.
#
# `count` is how many seeds to submit (default 10) and `skip` how many leading draws to DISCARD
# (default 0). EXTENDING a campaign is `skip = the number of seeds it already has`: the stream is
# replayed from the top and the seeds already banked are stepped over, so the new cells continue the
# sequence instead of restarting it. Verify with the printed list before letting it run.
#
# THE ARMS AND WHAT EACH IS SHOWN. `from_scratch`, `continue` and `closest` train one network per
# design and run at `training.reveal=none`; `meta` carries one network across designs and runs at
# `design`. That split is what `output/linear-10seed` recorded and it is set here rather than in the
# config because one config serves all four arms.
#
# RESOURCES MATCH THE COMPLETED CAMPAIGN. `--gres=shard:6` so exactly two cells share the card and
# the rest queue; `--mem=12000` for `meta` and 6000 for the others, because `meta` completes the most
# BO iterations and accumulates the most host state -- four `meta` cells were host-OOM-killed at 2500
# on 2026-08-21 with no CUDA error of any kind. `--time` is MANDATORY: a job without one is
# unlimited, is never backfilled, and stalls with the node half idle.
#
# ⚠️ MPS STAYS OFF. Two GPU MMU faults on 2026-08-21 were each amplified by MPS into a multi-cell
# kill; the blast radius was exactly the MPS client set. This launcher never starts the daemon.
#
# ⚠️ `sbatch -o` OVERWRITES. A resubmission of a cell destroys that cell's previous log, so copy any
# log worth keeping before re-running a seed.
set -e
CONFIG=$1
OUTPUT=$2
PREFIX=$3
COUNT=${4:-10}
SKIP=${5:-0}
[ -n "$CONFIG" ] && [ -n "$OUTPUT" ] && [ -n "$PREFIX" ] || { echo "usage: $0 <config> <output_dir> <job_prefix> [count] [skip]"; exit 2; }
[ -f "config/$CONFIG.yaml" ] || { echo "no config/$CONFIG.yaml"; exit 2; }

SEEDS=$(python3 -c "
import random
rng = random.Random(123456)
draws = [rng.randint(0, 2 ** 31 - 1) for _ in range($SKIP + $COUNT)]
print(' '.join(str(seed) for seed in draws[$SKIP:]))
")
echo "config=$CONFIG output=$OUTPUT skip=$SKIP count=$COUNT"
echo "seeds: $SEEDS"
mkdir -p "$OUTPUT/logs"

for SEED in $SEEDS; do
  for ARM in from_scratch continue closest meta; do
    if [ "$ARM" = "meta" ]; then REVEAL=design; MEM=12000; else REVEAL=none; MEM=6000; fi
    sbatch --gres=shard:6 --cpus-per-task=2 --mem=$MEM --time=06:00:00 \
           -J "$PREFIX$SEED-$ARM" -o "$OUTPUT/logs/$SEED-$ARM.log" \
           --wrap="python scripts/bo.py =$CONFIG output=$OUTPUT/$SEED/$ARM seed=$SEED nn_init_strategy=$ARM training.reveal=$REVEAL"
  done
done
echo "submitted $((COUNT * 4)) cells for $CONFIG -> $OUTPUT"
