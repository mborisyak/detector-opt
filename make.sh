#!/usr/bin/env bash
# Run all four BO training strategies (config/bo.yaml) and overlay their
# convergence on one plot.
#
#   ./make.sh [output_root] [seed]
#
# Strategies (same config + seed, separate output dirs; overridden via gearup CLI):
#   from_scratch / continue / closest  -> per-design, design-blind DesignTrainer
#       (continue & closest warm-start its weights from a previous design);
#   meta                               -> persistent, design-conditioned
#       ContinualTrainer with current+history (50/50) replay.
# Each runs until the shared budget pool fills. If the GPU runs out of memory,
# lower `training.budget` in config/bo.yaml.
set -euo pipefail

OUT="${1:-output/compare}"
SEED="${2:-42}"

for strategy in from_scratch continue closest meta; do
  echo "==================== strategy: ${strategy} ===================="
  python scripts/bo.py output="${OUT}/${strategy}" seed="${SEED}" \
    nn_init_strategy="${strategy}"
done

# Overlay the four best-so-far loss curves vs cumulative detector calls.
python scripts/compare_strategies.py --output "${OUT}"
echo "Done. Combined convergence -> ${OUT}/convergence_all.png"

# Independently verify each strategy's trajectory: <= 10 designs uniform in detector calls
# (always the last), full budget resampled per design, 6:2:2 train/val/test, an 8-member
# ensemble regressor, best-val network scored ONCE on the held-out test buffer.
for strategy in from_scratch continue closest meta; do
  echo "==================== verify: ${strategy} ===================="
  python scripts/verify_trajectory.py trajectory="${OUT}/${strategy}" seed="${SEED}" \
    regressor.set-regressor.n_models=8
done
echo "Done. Verification -> ${OUT}/<strategy>/verification.{json,png} + plots/"

# Refresh the comparison now that the verifications exist (adds ${OUT}/comparison.txt).
python scripts/compare_strategies.py --output "${OUT}"
echo "Comparison -> ${OUT}/comparison.txt"
