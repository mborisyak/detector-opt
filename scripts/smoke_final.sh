#!/bin/bash
# PLUMBING smoke test for the docs/final.md chain -- NOT a measurement, and no number it produces
# means anything. It proves that the strategy configs resolve, that `probe_retention.py` can replay a
# validation trajectory and restore each arm's state, that it records a usable convergence curve, and
# that `select_retention.py` ranks and writes.
#
# THE LADDER IS TINY AND THE PRECISION IS SLACKENED TO 2.0e-2, deliberately, and the value is chosen
# rather than guessed: at 5.0e-2 every design converges at n0, giving ONE window and nothing for the
# interpolation to do; at the real 1.0e-2 no window this small can converge at all, because the task
# genuinely needs 150-470k training rows (measured over 735 banked designs). 2.0e-2 lands mid-ladder,
# so the best-so-far curves have several growth rounds of real structure. `iteration_limit` is 12x
# `n0` here against 8x in the real config, so there is headroom against a cap.
#
# ⛔️ SLACKENING THE BAR IS LEGITIMATE HERE AND ONLY HERE. The campaign's exit test is untouched.
set -eu
cd /home/max/dev/detector-opt
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
OUT=output/smoke/angle
SEED=282522616
SMALL="training.n0=8192 training.n_increment=4096 training.iteration_limit=98304 training.budget=262144 training.patience=4 training.warmup_epochs=1 training.loss_precision=0.02"

for ARM in from_scratch continue meta; do
  echo "### validation $ARM"
  python scripts/bo.py =angle strategy=angle-$ARM-norewind output=$OUT/validation/$SEED/$ARM seed=$SEED $SMALL
done

for ARM in from_scratch continue meta; do
  for V in norewind rewind-025 sp-l06-s1e2; do
    echo "### probe $ARM $V"
    python scripts/probe_retention.py =angle strategy=angle-$ARM-$V \
      trajectory=$OUT/validation/$SEED/$ARM/results.json seed=$SEED \
      output=$OUT/probe/$SEED/$ARM/$V.json $SMALL
  done
done

echo "### select"
python scripts/select_retention.py --task angle --probes $OUT/probe --report $OUT/selection.txt
echo "### SMOKE OK"
