#!/usr/bin/env bash
# Queue ONE SLURM JOB PER ARM. Each job runs `optimizer_bakeoff.py` for a single cell and exits.
#
# WHY ONE PROCESS PER ARM. An arm restores budget-sized event pools onto the GPU -- measured at 5.1 GiB
# of the 8 GiB device for an `enzyme_extremes` trainer -- and running several inside one interpreter
# OOMs on the third, because the previous arm's device buffers are still held when the next allocates.
# That is what killed job 990. Process exit is the one reclamation that is guaranteed.
#
# WHY `--gres=shard:2`. The node advertises `shard:2` against a single 8 GiB card, and SLURM counts
# SLOTS -- it does not partition VRAM. Two arms at 5.1 GiB do not fit, so an arm asks for BOTH shards,
# which is the honest statement that it needs the whole card. SLURM then serialises the queue for us,
# and the jobs can be submitted all at once instead of chained by hand.
#
# `optimizer_bakeoff.py` skips a cell whose json already exists, so re-running this is free and only
# fills gaps. Each cell writes its own log, so a failure is attributable to an arm.
#
#   scripts/optimizer_bakeoff.sh                 # queue every cell
#   scripts/optimizer_bakeoff.sh adan/constant   # queue one
set -u

OUTPUT=${OUTPUT:-output/optimizer-bakeoff}
RUN=${RUN:-output/enzyme_extremes/1244111331/meta}
RUN_SEED=${RUN_SEED:-1244111331}
CONFIG=${CONFIG:-=enzyme_extremes}
LOGS=${LOGS:-logs}

# SIX CELLS, not ten: the five optimisers under the constant rate, and the SCHEDULE tested on `adamw`
# alone. The second factor does not need repeating on every optimiser to answer whether the decay
# helps -- `adamw` is the reference arm, so it carries it, and the schedule comparison is then
# adamw-against-adamw, paired on everything else.
DEFAULT_CELLS=(
  adamw/constant adamaxw/constant adan/constant nadamw/constant amsgrad/constant
  adamw/hyperbolic
)

if [ $# -gt 0 ]; then
  CELLS=("$@")
else
  CELLS=("${DEFAULT_CELLS[@]}")
fi

mkdir -p "${LOGS}"
for cell in "${CELLS[@]}"; do
  name=${cell/\//-}
  sbatch --gres=shard:2 --cpus-per-task=4 --mem=12000 \
         -J "bake-${name}" -o "${LOGS}/bakeoff-${name}.log" \
         --wrap="python -u scripts/optimizer_bakeoff.py ${CONFIG} output=${OUTPUT} run=${RUN} \
                 run_seed=${RUN_SEED} arms=${cell}"
done

squeue -o '%.7i %.18j %.2t %.8M %R'
