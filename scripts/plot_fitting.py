#!/usr/bin/env python3
"""What each training regime does DURING fitting, with the data additions marked.

    python scripts/plot_fitting.py --task intersect --runs output/final-bo/intersect/select \
        --strategy from_scratch --iteration 3

WHAT THIS SHOWS, and how it differs from `plot_learning_curves.py`. That script probes the weights a
regime LEFT BEHIND, by retraining them under one standard optimiser. This one is the regime's own
training run: the per-epoch series `bo.py` writes to `plots/iter_NNN_history.npz` for every design,
including `train_budget_per_epoch`, which is where the growth procedure INJECTED DATA. Each dashed
line is one such injection -- and for shrink-and-perturb it is also where the weights were partially
resampled, so the loss jumps and has to re-converge. That sawtooth is the whole cost of the regime
and it is invisible in any end-of-design number.

One panel per regime, one design (`--iteration`), all seeds that have it. The x axis is the epoch
within that design, so panels are NOT the same width: a regime that needs three times the epochs to
cross the same data is exactly what this plot is for.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def additions(budget_per_epoch):
  """Epoch indices where the training window grew -- the data injections."""
  b = np.asarray(budget_per_epoch)
  return np.nonzero(np.diff(b) > 0)[0] + 1


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--runs", required=True)
  parser.add_argument("--strategy", default="from_scratch")
  parser.add_argument("--iteration", type=int, default=3)
  parser.add_argument("--out", default="output/plots")
  parser.add_argument("--name", default=None)
  args = parser.parse_args()

  found = {}
  for path in sorted(glob.glob(os.path.join(args.runs, "*", "*", "*", "plots", f"iter_{args.iteration:03d}_history.npz"))):
    seed, strategy, regime = path.split(os.sep)[-5:-2]
    if strategy != args.strategy:
      continue
    found.setdefault(regime, []).append((seed, np.load(path)))
  if len(found) == 0:
    raise SystemExit(f"plot_fitting: no iter_{args.iteration:03d} history for strategy={args.strategy} under {args.runs}")

  regimes = sorted(found)
  figure, axes = plt.subplots(1, len(regimes), figsize=(4.6 * len(regimes), 4.4), squeeze=False, sharey=True)
  for axis, regime in zip(axes[0], regimes):
    for seed, z in found[regime]:
      val = z["val_loss_per_epoch"]
      train = z["train_loss_per_epoch"]
      epochs = np.arange(len(val))
      line, = axis.plot(epochs, val, linewidth=1.1, label=f"{seed[:6]} val")
      axis.plot(epochs, train, linewidth=0.8, alpha=0.45, linestyle="--", color=line.get_color())
      for e in additions(z["train_budget_per_epoch"]):
        axis.axvline(e, color=line.get_color(), alpha=0.20, linewidth=0.7)
    n_add = int(np.mean([len(additions(z["train_budget_per_epoch"])) for _, z in found[regime]]))
    n_ep = int(np.mean([len(z["val_loss_per_epoch"]) for _, z in found[regime]]))
    axis.set_title(f"{regime}\n{n_ep} epochs, {n_add} data additions", fontsize=10)
    axis.set_xlabel("epoch within the design")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(fontsize=6)
  axes[0][0].set_ylabel("loss (solid = validation, dashed = train)")
  figure.suptitle(
    f"{args.task} / {args.strategy}: fitting design {args.iteration}; vertical lines are DATA ADDITIONS", fontsize=11
  )
  figure.tight_layout()
  os.makedirs(args.out, exist_ok=True)
  name = args.name if args.name is not None else f"fitting-{args.task}-{args.strategy}-iter{args.iteration:03d}.png"
  path = os.path.join(args.out, name)
  figure.savefig(path, dpi=150)
  print(f"wrote {path}")
  for regime in regimes:
    for seed, z in found[regime]:
      add = additions(z["train_budget_per_epoch"])
      print(
        "  %-11s %-11s epochs=%4d additions=%2d  window %d -> %d  final val=%.4f" % (
          regime, seed[:9], len(z["val_loss_per_epoch"]), len(add), int(z["train_budget_per_epoch"][0]),
          int(z["train_budget_per_epoch"][-1]), float(z["val_loss_per_epoch"][-1])
        )
      )


if __name__ == "__main__":
  main()
