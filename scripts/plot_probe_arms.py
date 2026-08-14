#!/usr/bin/env python3
"""Trajectory figures for the probe arms -- `scripts/probe_precision.py` outputs.

    python scripts/plot_probe_arms.py --seed 1 --output-dir docs/figures

WHAT IT DRAWS, and why these and not per-epoch curves. `probe_precision` stores the epoch history
COLLAPSED to one row per data-addition round (`probe_dropout._per_window`), so the x axis available is
the DATA USED, not the epoch. That is the better axis anyway for this comparison: the whole question is
what a schedule does to the score AT A GIVEN AMOUNT OF DATA, and plotting against epochs would put the
growth arm's 8000 passes beside the fixed arm's 300 and hide it.

  Figure 1  train and validation loss against data used, one panel per design.
            The fixed-window arm holds ONE data amount, so it is a marker rather than a line -- that is
            the point of it: it is where a schedule-free run of the same size lands.
  Figure 2  the two terms the convergence test adds: the train/validation gap and the statistical
            error, against data used, with the target drawn. A run stops when their sum crosses it.

Every line is ONE (design, seed) measurement, and the arms are matched cell-for-cell on their initial
network, so the vertical distance between two lines at the same x is a paired difference and not a
draw from two populations.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

# The arms, in the order they should be drawn, with the style each carries through both figures.
ARMS = (
  ("no rewind", "output/screen/s13-grow.json", "tab:red", "-"),
  ("rewind 0.25", "output/screen/s13-mix025.json", "tab:blue", "-"),
  ("rewind 0.5", "output/screen/s13-mix05.json", "tab:purple", "-"),
  ("rewind 1.0", "output/screen/s13-mix10.json", "tab:orange", "-"),
  ("all at once", "output/screen/s13-fixed.json", "tab:green", "--"),
)


def load(path):
  """``{(design, seed): row}`` for one arm, or ``{}`` if it has not been written yet."""
  if not os.path.isfile(path):
    return {}
  with open(path) as f:
    return {(r["design"], int(r["seed"])): r for r in json.load(f)["rows"]}


def trajectory(row):
  """``(data, train, val, gap, err)`` over the run's data-addition rounds.

  The stored `window` is the TRAIN half; the run also draws validation events in a fixed ratio, so the
  simulation actually spent is larger. The train half is what the loss is a mean over, so it is what
  the x axis says, and the conversion is stated in the caption rather than folded in silently.
  """
  per_window = row.get("per_window", [])
  if len(per_window) == 0:
    return None
  data = np.asarray([w["window"] for w in per_window], np.float64)
  return (data, np.asarray([w["train"] for w in per_window], np.float64),
          np.asarray([w["val"] for w in per_window], np.float64),
          np.asarray([w["diff"] for w in per_window], np.float64),
          np.asarray([w["err"] for w in per_window], np.float64))


def panels(designs, title, ylabel, figsize=(13, 4.2)):
  figure, axes = plt.subplots(1, len(designs), figsize=figsize, sharex=True)
  axes = np.atleast_1d(axes)
  figure.suptitle(title, fontsize=12)
  for axis in axes:
    axis.set_xscale("log")
    axis.set_xlabel("training events used")
    axis.grid(alpha=0.25, which="both")
  axes[0].set_ylabel(ylabel)
  return figure, axes


def draw_losses(arms, designs, seed, path):
  """Figure 1: train (solid) and validation (faint) loss against data used."""
  figure, axes = panels(designs, f"Score against data used, seed {seed}  "
                        "(solid: training loss, faint: validation loss)", "loss")
  for axis, design in zip(axes, designs):
    for label, cells, colour, style in arms:
      row = cells.get((design, seed))
      if row is None:
        continue
      curve = trajectory(row)
      if curve is None:
        continue
      data, train, val, _, _ = curve
      if data.shape[0] == 1:  # the fixed-window arm: one data amount, so a marker not a line
        axis.plot(data, train, marker="o", color=colour, markersize=7, linestyle="none",
                  label=f"{label} ({row['status']})")
        axis.plot(data, val, marker="o", color=colour, markersize=7, alpha=0.35, linestyle="none")
      else:
        axis.plot(data, train, color=colour, linestyle=style, label=f"{label} ({row['status']})")
        axis.plot(data, val, color=colour, linestyle=style, alpha=0.35)
    axis.set_title(design, fontsize=10)
    axis.legend(fontsize=8, loc="best")
  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  print(f"wrote {path}")


def draw_terms(arms, designs, seed, precision, path):
  """Figure 2: the gap and the statistical error, the two terms the stopping test adds."""
  figure, axes = panels(designs, f"The two terms of the convergence test, seed {seed}  "
                        "(solid: train/validation gap, dotted: statistical error)", "loss units")
  for axis, design in zip(axes, designs):
    for label, cells, colour, style in arms:
      row = cells.get((design, seed))
      if row is None:
        continue
      curve = trajectory(row)
      if curve is None:
        continue
      data, _, _, gap, err = curve
      marker = "o" if data.shape[0] == 1 else None
      line = "none" if data.shape[0] == 1 else style
      axis.plot(data, gap, color=colour, linestyle=line, marker=marker, markersize=7, label=label)
      axis.plot(data, err, color=colour, linestyle="none" if marker else ":", marker=marker,
                markersize=5, alpha=0.5)
    axis.axhline(precision, color="k", linestyle="-.", linewidth=1)
    axis.text(axis.get_xlim()[0], precision * 1.06, f"target {precision:g}", fontsize=8, va="bottom")
    axis.set_yscale("log")
    axis.set_title(design, fontsize=10)
    axis.legend(fontsize=8, loc="best")
  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  print(f"wrote {path}")


def draw_mm(path, no_information):
  """The kinetic-constant task: the network against the stand-in that ranks its designs."""
  cells = load("output/screen/s16-mm-floor.json")
  if len(cells) == 0:
    print("no kinetic-constant measurements yet; skipping that figure")
    return
  order, labels, network, proxy = [], [], [], []
  for (design, seed), row in sorted(cells.items(), key=lambda kv: kv[1]["proxy_loss"]):
    order.append(len(order))
    labels.append(f"{design}\nseed {seed}")
    network.append(0.5 * (row["train"] + row["val"]))
    proxy.append(row["proxy_loss"])
  figure, axis = plt.subplots(figsize=(1.6 * len(order) + 3.0, 4.2))
  width = 0.38
  axis.bar([o - width / 2 for o in order], proxy, width, label="tree stand-in", color="tab:orange")
  axis.bar([o + width / 2 for o in order], network, width, label="network, all data at once",
           color="tab:blue")
  axis.axhline(no_information, color="k", linestyle="-.", linewidth=1)
  axis.text(-0.45, no_information * 1.01, f"knowing nothing = {no_information:g}", fontsize=8)
  axis.set_xticks(order)
  axis.set_xticklabels(labels, fontsize=8)
  axis.set_ylabel("error")
  axis.set_title("Kinetic-constant task: the network is the weaker estimator", fontsize=12)
  axis.legend(fontsize=9)
  axis.grid(alpha=0.25, axis="y")
  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  print(f"wrote {path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=1, help="the seed to draw (all arms share it)")
  parser.add_argument("--designs", nargs="+", default=["sobol-best", "sobol-rank6", "sobol-median"])
  parser.add_argument("--precision", type=float, default=8.0e-3, help="the target, drawn as a rule")
  parser.add_argument("--no-information", type=float, default=0.3340,
                      help="the kinetic-constant task's no-information error")
  parser.add_argument("--output-dir", default="docs/figures")
  arguments = parser.parse_args()

  os.makedirs(arguments.output_dir, exist_ok=True)
  arms = [(label, load(path), colour, style) for label, path, colour, style in ARMS]
  present = [(label, cells, colour, style) for label, cells, colour, style in arms if len(cells) > 0]
  missing = [label for label, cells, _, _ in arms if len(cells) == 0]
  if len(missing) > 0:
    print(f"[warning] no measurements for {missing}; drawing the rest")

  draw_losses(present, arguments.designs, arguments.seed,
              os.path.join(arguments.output_dir, "schedules-loss.png"))
  draw_terms(present, arguments.designs, arguments.seed, arguments.precision,
             os.path.join(arguments.output_dir, "schedules-terms.png"))
  draw_mm(os.path.join(arguments.output_dir, "kinetic-floor.png"), arguments.no_information)


if __name__ == "__main__":
  main()
