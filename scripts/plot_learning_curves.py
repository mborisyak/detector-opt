#!/usr/bin/env python3
"""Per-regime learning curves from the verification retrain -- a STANDARDISED probe of what each
training regime left behind.

    python scripts/plot_learning_curves.py --task intersect \
        --runs output/final-bo/intersect/select --out output/plots

WHAT THIS IS, AND WHY IT IS A FAIR COMPARISON. `verify_trajectory.py` does NOT apply the training
regime: it restores the weights the run reported for a design and retrains them for a fixed
`verify.epochs` with one standard optimiser, identical for every regime. So the only thing that
differs between these curves is the WEIGHTS THE REGIME PRODUCED. Epoch 0 is the restored network's
held-out loss -- the regime's own answer -- and the descent after it is how much a fixed retrain
budget can still extract from that starting point.

WHAT IT IS NOT. It is not the regime's own training dynamics; those are not recorded per epoch. A
regime that looks bad at epoch 0 and catches up by epoch 64 is one whose weights were merely
under-trained, not badly placed -- which is a different diagnosis from one that starts low and stays
flat, and separating those two is the point of plotting the whole curve rather than the endpoint.

One curve per regime, averaged over the cells that reached a complete verification, taken at the
LAST trajectory point (the run's final incumbent). The band is the standard error across cells.
"""
import argparse
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(runs, use_last=True):
  """``{(strategy, regime): [np.ndarray of shape (epochs, 3)]}`` from complete verifications."""
  out = collections.defaultdict(list)
  for path in sorted(glob.glob(os.path.join(runs, "*", "*", "*", "verification.json"))):
    directory = os.path.dirname(path)
    if not os.path.exists(os.path.join(directory, "verified.txt")):
      continue
    _, strategy, regime = directory.split(os.sep)[-3:]
    with open(path) as handle:
      payload = json.load(handle)
    points = [p for p in payload.get("points", []) if p.get("history")]
    if len(points) == 0:
      continue
    points.sort(key=lambda p: p.get("detector_calls", 0))
    chosen = [points[-1]] if use_last else points
    for point in chosen:
      out[(strategy, regime)].append(np.asarray(point["history"], dtype=float))
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--runs", required=True)
  parser.add_argument("--out", default="output/plots")
  parser.add_argument("--name", default=None)
  parser.add_argument("--all-points", action="store_true", help="average every trajectory point, not just the last")
  args = parser.parse_args()

  data = load(args.runs, use_last=not args.all_points)
  if len(data) == 0:
    raise SystemExit(f"plot_learning_curves: no complete verifications under {args.runs}")
  strategies = sorted({s for s, _ in data})

  figure, axes = plt.subplots(2, len(strategies), figsize=(6.0 * len(strategies), 8.4), squeeze=False, sharex="col")
  for column, strategy in enumerate(strategies):
    upper, lower = axes[0][column], axes[1][column]
    for regime in sorted(r for s, r in data if s == strategy):
      stack = data[(strategy, regime)]
      length = min(len(h) for h in stack)
      block = np.stack([h[:length] for h in stack])
      epochs = block[0, :, 0]
      for axis, index, label in ((upper, 2, "held-out validation"), (lower, 1, "train")):
        mean = np.nanmean(block[:, :, index], axis=0)
        error = (np.nanstd(block[:, :, index], axis=0, ddof=1) / np.sqrt(len(stack)) if len(stack) > 1 else np.zeros_like(mean))
        line, = axis.plot(epochs, mean, linewidth=1.6, label=f"{regime} (n={len(stack)})")
        axis.fill_between(epochs, mean - error, mean + error, alpha=0.15, color=line.get_color())
    upper.set_title(f"{args.task} / {strategy}")
    upper.set_ylabel("validation loss (retrain)")
    lower.set_ylabel("train loss (retrain)")
    lower.set_xlabel("retrain epoch")
    for pane in (upper, lower):
      pane.grid(alpha=0.25, linewidth=0.5)
      pane.legend(fontsize=7)

  figure.suptitle(f"{args.task}: what each regime left behind -- fixed retrain, identical for all regimes", fontsize=11)
  figure.tight_layout()
  os.makedirs(args.out, exist_ok=True)
  name = args.name if args.name is not None else f"learning-{args.task}.png"
  path = os.path.join(args.out, name)
  figure.savefig(path, dpi=150)
  print(f"wrote {path}")
  for (strategy, regime), stack in sorted(data.items()):
    block = np.stack([h[:min(len(x) for x in stack)] for h in stack])
    print(
      "  %-13s %-11s n=%d  epoch0 val=%.4f  final val=%.4f" %
      (strategy, regime, len(stack), np.nanmean(block[:, 0, 2]), np.nanmean(block[:, -1, 2]))
    )


if __name__ == "__main__":
  main()
