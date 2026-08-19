#!/usr/bin/env python3
"""BO driver for the STRICT-GROWTH training procedure.

    python scripts/bo_strict.py =linear_d2n3_strict output=output/strict seed=1 \
        nn_init_strategy=from_scratch

IDENTICAL to `scripts/bo.py` in everything the OPTIMISER does -- same proposal loop, same seed
contract, same resume, same checkpoints, same plots, same four strategies. It reuses that driver and
substitutes ONLY the two trainer classes, because a copied 400-line driver drifts and a drifted driver
makes two experiments incomparable for reasons nobody can see.

WHAT DIFFERS, and all of it lives in `detopt/nn/trainer/strict.py`:

    warmup        `warmup_epochs` epochs after EVERY data addition, no test at all.
    then, each epoch:
      (1) `gap + err > loss_precision`            -> ADD DATA (round resets, warmup runs again)
      (2) `patience` epochs with no improvement   -> CONVERGED, score and return
      (3) otherwise                               -> keep training

The SCORING gate is unchanged from `bo.py` -- `gap + err <= loss_precision`, objective `val`, noise
`gap + err`. Only the CONVERGENCE test differs: "has the loss stopped improving" replaces the Bayesian
"is the fitted slope small". A slow steady descent cannot satisfy the strict rule, because it keeps
producing new bests and keeps resetting the counter.

⚠️ Runs under this driver are NOT comparable to runs under `bo.py` -- different stopping rule, hence
different data per design and a different reported value. The matched control is the SAME arm and seed
under both drivers.

`detopt/nn/trainer/design.py` is neither modified nor imported for its loop, and remains the only
implementation of the Bayesian procedure.
"""

import sys

import gearup

import bo
from detopt.nn.trainer.strict import StrictContinualTrainer, StrictDesignTrainer

# THE ONE SUBSTITUTION. `bo.bo()` reads `bo.TRAINERS` when it selects a trainer, so rebinding the
# mapping here is the whole difference between the drivers. Asserted rather than assumed: if that seam
# moves, this fails loudly instead of silently running the Bayesian procedure.
if set(bo.TRAINERS) != {"per_design", "meta"}:
  raise RuntimeError(f"bo.TRAINERS changed shape ({sorted(bo.TRAINERS)}); bo_strict's substitution is stale")
bo.TRAINERS = {"per_design": StrictDesignTrainer, "meta": StrictContinualTrainer}


def main():
  arguments = ["force=yes" if a == "--force" else a for a in sys.argv[1:]]
  print(
    "[bo_strict] STRICT-GROWTH: warmup, then add data while gap+err > loss_precision, "
    "converge on `patience` epochs without improvement", flush=True
  )
  gearup.gearup(bo.bo).with_config("config/bo.yaml")(arguments)


if __name__ == "__main__":
  main()
