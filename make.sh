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

# Cross-strategy comparison of each trajectory's FINAL verified design (verification always
# includes the last point) + per-strategy step statistics (BO iterations, detector calls per
# iteration avg+-std). Writes an ASCII table to ${OUT}/comparison.txt (+ stdout).
python3 - "${OUT}" <<'EOF'
import json, os, sys
from statistics import mean, pstdev

out = sys.argv[1]
rows = []
for strategy in ("from_scratch", "continue", "closest", "meta"):
  vpath = os.path.join(out, strategy, "verification.json")
  rpath = os.path.join(out, strategy, "results.json")
  if not (os.path.exists(vpath) and os.path.exists(rpath)):
    continue
  d = json.load(open(vpath))
  spent = [int(r["spent"]) for r in json.load(open(rpath))["results"]]
  p = d["points"][-1]  # points sorted by trajectory index; the final design is always verified
  rows.append((
    strategy, len(spent), mean(spent), pstdev(spent), int(p["detector_calls"]), p["reported_loss"], p["val_loss"],
    p["test_loss"], p["test_sem"], p["test_loss"] - p["reported_loss"]
  ))
lines = [
  f"{'strategy':<14}{'steps':>6}{'calls/step':>16}{'calls':>9}{'reported':>10}{'val':>8}{'test':>8}{'sem':>8}{'delta':>9}",
  "-" * 88,
]
for r in rows:
  cps = f"{r[2]:.0f}+-{r[3]:.0f}"
  lines.append(f"{r[0]:<14}{r[1]:>6}{cps:>16}{r[4]:>9}{r[5]:>10.4f}{r[6]:>8.4f}{r[7]:>8.4f}{r[8]:>8.4f}{r[9]:>+9.4f}")
text = "\n".join(lines) + "\n"
with open(os.path.join(out, "comparison.txt"), "w") as f:
  f.write(text)
print(text, end="")
EOF
echo "Comparison -> ${OUT}/comparison.txt"
