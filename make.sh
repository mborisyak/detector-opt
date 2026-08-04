#!/usr/bin/env bash
# Run all four BO training strategies, verify each trajectory independently, and plot both.
#
#   ./make.sh [output_root] [seed] [config]
#
# `config` names a run config under config/ and defaults to `bo` (the SST spectrometer over
# config/bo.yaml); `enzyme` runs the same comparison on the enzymatic single-batch design of
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
# Then, per strategy, scripts/verify_trajectory.py re-scores <=verify.n_points designs along the
# trajectory with a fresh regressor, a 6:2:2 split and a held-out test score the optimizer never
# saw. That verifies the NETWORK, and nothing else -- integration accuracy is not its business: the
# solver estimates its own error from a dt and a dt/2 chain on every solve and every call asserts on
# it, with scripts/check_stability.py measuring the step against the scheme's stability boundary and
# scripts/verify_lsoda.py cross-checking the scheme against LSODA.
#
# Everything is written as JSON (results.json, verification.json, convergence_all.json), so the
# final plot regenerates with just the last command.
set -euo pipefail

# Progress here is meant to be watched, and it is usually watched through a redirect (`./make.sh
# ... > run.log`) or a pipe -- where Python block-buffers stdout and a healthy run looks silent for
# many minutes. Keep it line-buffered.
export PYTHONUNBUFFERED=1

OUT="${1:-output/compare}"
SEED="${2:-42}"
CONFIG="${3:-bo}"

STRATEGIES=(from_scratch continue closest meta)

for strategy in "${STRATEGIES[@]}"; do
  echo "==================== BO: ${strategy} ===================="
  python scripts/bo.py "=${CONFIG}" output="${OUT}/${strategy}" seed="${SEED}" \
    nn_init_strategy="${strategy}"
done

for strategy in "${STRATEGIES[@]}"; do
  echo "==================== verify: ${strategy} ===================="
  python scripts/verify_trajectory.py "=${CONFIG}" trajectory="${OUT}/${strategy}" seed="${SEED}"
done

# Overlay: self-evaluated (dashed) vs verified (solid, with error bars), + the plotted values.
python scripts/compare_strategies.py --output "${OUT}"
echo "Done. Combined convergence -> ${OUT}/convergence_all.png (values in convergence_all.json)"
