#!/usr/bin/env python3
"""Per-epoch curves for the rewind doses -- one figure per design, doses overlaid.

    python scripts/plot_rewind_doses.py --seed 1 --output-dir docs/figures

WHAT IT DRAWS AND WHY. `scripts/probe_precision.py` stores the full per-epoch history (`per_epoch`),
which is the only view that shows what a rewind actually DOES: at each data addition the parameters are
pulled part of the way back toward the network the run started from, the loss jumps, and it recovers
over the following epochs. The per-round collapse cannot show that -- it keeps one point per round.

Three panels per design, sharing the epoch axis:

  (a) training loss (solid) and validation loss (dashed), one colour per dose. The gap between a
      matched pair of lines IS the quantity the stopping rule watches.
  (b) `|val - train| + err`, the quantity that must fall below the target for the run to be accepted,
      with the target drawn. Log scale, because it spans two decades.
  (c) the data the run is training on, as a step. Every step IS a data addition, so this panel is the
      x-axis legend for the other two: where a dose's curve ends is where it converged.

A ZOOM figure repeats (a) and (b) over the first epochs, where the additions are frequent and the
rewind's sawtooth is visible; at full scale that detail is compressed into the left margin.

Every dose is the SAME design, the SAME seed and the SAME initial network, so a vertical difference
between two curves at the same epoch is a paired difference.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

DOSES = (
  ("no rewind", "output/screen/s18-mix000.json", "tab:red"),
  ("rewind 0.125", "output/screen/s18-mix0125.json", "tab:purple"),
  ("rewind 0.25", "output/screen/s18-mix025.json", "tab:blue"),
  ("rewind 0.5", "output/screen/s18-mix05.json", "tab:green"),
)


def load(path):
  if not os.path.isfile(path):
    return {}
  with open(path) as f:
    return {(r["design"], int(r["seed"])): r for r in json.load(f)["rows"]}


def curves(row):
  """``(train, val, uncertainty, window)`` per epoch, or ``None`` if the row predates the storage."""
  per_epoch = row.get("per_epoch")
  if per_epoch is None:
    return None
  train = np.asarray(per_epoch["train"], np.float64)
  val = np.asarray(per_epoch["val"], np.float64)
  err = np.hypot(np.asarray(per_epoch["train_sem"], np.float64), np.asarray(per_epoch["val_sem"], np.float64))
  return train, val, np.abs(val - train) + err, np.asarray(per_epoch["window"], np.int64)


def draw(doses, design, seed, precision, path, last_epoch=None):
  present = [(label, cells[(design, seed)], colour) for label, cells, colour in doses
             if (design, seed) in cells and curves(cells[(design, seed)]) is not None]
  if len(present) == 0:
    print(f"no per-epoch curves for {design} seed {seed}; skipping")
    return False
  span = "" if last_epoch is None else f", first {last_epoch} epochs"
  figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 1.4]})
  figure.suptitle(f"Rewind doses at {design}, seed {seed}{span}\n"
                  "solid: training loss   dashed: validation loss", fontsize=12)
  for label, row, colour in present:
    train, val, uncertainty, window = curves(row)
    stop = len(train) if last_epoch is None else min(len(train), int(last_epoch))
    epochs = np.arange(1, stop + 1)
    tag = f"{label} ({row['status']}, {row['n_epochs']} epochs)"
    axes[0].plot(epochs, train[:stop], color=colour, linewidth=1.0, label=tag)
    axes[0].plot(epochs, val[:stop], color=colour, linewidth=1.0, linestyle="--", alpha=0.6)
    axes[1].plot(epochs, uncertainty[:stop], color=colour, linewidth=1.0, label=label)
    axes[2].step(epochs, window[:stop], color=colour, linewidth=1.0, where="post", label=label)
  axes[0].set_ylabel("loss")
  axes[0].legend(fontsize=8, loc="best")
  axes[1].set_ylabel("|val - train| + error")
  axes[1].set_yscale("log")
  axes[1].axhline(precision, color="k", linestyle="-.", linewidth=1)
  axes[1].text(1, precision * 1.08, f"target {precision:g} -- a run is accepted when it falls below this",
               fontsize=8, va="bottom")
  axes[2].set_ylabel("training events")
  axes[2].set_xlabel("epoch (each step in the bottom panel is a data addition)")
  for axis in axes:
    axis.grid(alpha=0.25)
  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  print(f"wrote {path}")
  return True


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--designs", nargs="+", default=["sobol-best", "sobol-rank6", "sobol-median"])
  parser.add_argument("--precision", type=float, default=8.0e-3)
  parser.add_argument("--zoom", type=int, default=400, help="epochs shown in the companion zoom figure")
  parser.add_argument("--output-dir", default="docs/figures")
  arguments = parser.parse_args()

  os.makedirs(arguments.output_dir, exist_ok=True)
  doses = [(label, load(path), colour) for label, path, colour in DOSES]
  missing = [label for label, cells, _ in doses if len(cells) == 0]
  if len(missing) > 0:
    print(f"[warning] nothing written yet for {missing}; drawing the rest")
  for design in arguments.designs:
    draw(doses, design, arguments.seed, arguments.precision,
         os.path.join(arguments.output_dir, f"doses-{design}.png"))
    draw(doses, design, arguments.seed, arguments.precision,
         os.path.join(arguments.output_dir, f"doses-{design}-zoom.png"), last_epoch=arguments.zoom)


if __name__ == "__main__":
  main()
