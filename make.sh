#!/usr/bin/env bash
# Run all four BO training strategies, verify each trajectory independently, and plot both.
#
#   ./make.sh [output_root] [seed] [stage] [--config <name>]
#
# Without a `stage`, both stages run and NOTHING already computed is redone: a completed
# `results.json` makes bo.py skip that strategy, and the points already in `verification.json` (under
# identical settings) are reused. Naming a stage means "redo this one": it runs that stage ALONE and
# passes --force to it, so the corresponding results are overwritten.
#
#   ./make.sh out 42          -> BO + verification, skipping whatever is already computed
#   ./make.sh out 42 bo       -> the four BO runs only, recomputed from scratch
#   ./make.sh out 42 verify   -> the four verifications only, recomputed from scratch
#   ./make.sh out 42 all      -> both stages, everything recomputed from scratch
#
# `--config <name>` names a run config under config/ and defaults to `bo` (the SST spectrometer over
# config/bo.yaml); `--config enzyme` runs the same comparison on the enzymatic single-batch design of
# experiments (config/enzyme.yaml).
#
# Strategies (same config + seed, separate output dirs; overridden via gearup CLI):
#   from_scratch / continue / closest  -> per-design DesignTrainer (continue & closest
#       warm-start its weights from a previous design);
#   meta                               -> persistent ContinualTrainer with current+history
#       (50/50) replay -- the proposal's meta-model.
# All four are design-conditioned. Each runs until the shared budget pool fills; if the GPU runs
# out of memory, lower `training.budget` in the run config.
#
# Then, per strategy, scripts/verify_trajectory.py re-scores <=verify.n_points designs of the
# trajectory -- the INCUMBENTS, the iterations where the best-so-far loss improved -- by restoring
# the network the run reported each of them with and continuing it on a fresh 6:2:2 split, for a
# held-out test score the optimizer never saw. That verifies the NETWORK, and nothing else:
# integration accuracy is not its business, since the solver estimates its own error from a dt and a
# dt/2 chain on every solve and every call asserts on it, with scripts/check_stability.py measuring
# the step against the scheme's stability boundary and scripts/verify_lsoda.py cross-checking the
# scheme against LSODA.
#
# Everything is written as JSON (results.json, verification.json, convergence_all.json), so the
# final plot regenerates with just the last command.
set -euo pipefail

# Progress here is meant to be watched, and it is usually watched through a redirect (`./make.sh
# ... > run.log`) or a pipe -- where Python block-buffers stdout and a healthy run looks silent for
# many minutes. Keep it line-buffered.
export PYTHONUNBUFFERED=1

CONFIG="bo"
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"}

OUT="${1:-output/compare}"
SEED="${2:-42}"
STAGE="${3:-}"

case "${STAGE}" in
  "")     RUN_BO=1; RUN_VERIFY=1; FORCE=() ;;
  bo)     RUN_BO=1; RUN_VERIFY=0; FORCE=(--force) ;;
  verify) RUN_BO=0; RUN_VERIFY=1; FORCE=(--force) ;;
  all)    RUN_BO=1; RUN_VERIFY=1; FORCE=(--force) ;;
  *) echo "unknown stage '${STAGE}' -- expected bo, verify, all, or nothing at all" >&2; exit 2 ;;
esac

STRATEGIES=(from_scratch continue closest meta)

if [ "${RUN_BO}" -eq 1 ]; then
  for strategy in "${STRATEGIES[@]}"; do
    echo "==================== BO: ${strategy} ===================="
    python scripts/bo.py "=${CONFIG}" output="${OUT}/${strategy}" seed="${SEED}" \
      nn_init_strategy="${strategy}" ${FORCE[@]+"${FORCE[@]}"}
  done
fi

if [ "${RUN_VERIFY}" -eq 1 ]; then
  for strategy in "${STRATEGIES[@]}"; do
    echo "==================== verify: ${strategy} ===================="
    python scripts/verify_trajectory.py "=${CONFIG}" trajectory="${OUT}/${strategy}" seed="${SEED}" \
      ${FORCE[@]+"${FORCE[@]}"}
  done
fi

# Overlay: self-evaluated (dashed) vs verified (solid, with error bars), + the plotted values.
python scripts/compare_strategies.py --output "${OUT}"
echo "Done. Combined convergence -> ${OUT}/convergence_all.png (values in convergence_all.json)"
