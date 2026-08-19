#!/usr/bin/env python3
"""Figures for the convergence-criterion bias benchmark.

    python scripts/plot_criterion_bench.py --plots-dir output/criterion-bench/plots

Reads the JSONs the probes wrote and renders the figures the argument needs. Every input is optional:
a missing file prints a line and skips its figure rather than failing, so the script can be re-run as
cells land.

THE FIGURES, and what each one has to show.

  1 excess_precision   `excess` against `loss_precision`, one line per arm, log x. H1/H2: the bar is a
                       purchase order for data, and if it is what makes the bias arm-dependent the two
                       lines must converge as the bar tightens.
  2 patience           `excess` and exit window against `patience`, one line per arm per task. H3.
  3 exit_decomposition `gap` and `err` STACKED against the `loss_precision` bar, per arm. Shows which
                       term is binding at the exit and whether the arms split it differently.
  4 curves             per-epoch train and validation loss for one representative design per cell,
                       with the last data addition and the stop marked. This is what makes a premature
                       stop visible rather than asserted.
  5 remaining          remaining descent from the fitted tail against exit window, per arm.
  6 rewind             the restart cells: within-design parameter displacement by position, and exit
                       window / excess per restart cell. H7/H7a.

COLOUR IS BY ARM AND NEVER BY RANK: `from_scratch` is slot 1 and `meta` is slot 2 of the validated
categorical palette in every figure, so a figure that drops an arm does not repaint the survivor. The
two exit components are slots 3 and 4 of the same order; both sit below 3:1 against the light surface,
so they carry direct value labels (the relief rule) rather than relying on the fill alone.

THIS SCRIPT MEASURES NOTHING. It reads the JSONs the probes wrote.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import matplotlib

matplotlib.use("AGG")

import matplotlib.pyplot as plt
import numpy as np

from report_bias_cross import enrich

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dedcd6"
ARM_COLOUR = {"from_scratch": "#2a78d6", "meta": "#eb6834"}
ARM_MARKER = {"from_scratch": "o", "meta": "s"}
COMPONENT_COLOUR = {"gap": "#1baf7a", "err": "#eda100"}


def style(axes, xlabel, ylabel, title):
  axes.set_facecolor(SURFACE)
  axes.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
  axes.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
  axes.set_title(title, color=INK, fontsize=10, loc="left")
  axes.grid(True, color=GRID, linewidth=0.6, zorder=0)
  axes.set_axisbelow(True)
  for side in ("top", "right"):
    axes.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axes.spines[side].set_color(GRID)
  axes.tick_params(colors=INK_SECONDARY, labelsize=8)


def figure(width=7.0, height=4.2, columns=1):
  fig, axes = plt.subplots(1, columns, figsize=(width * columns, height))
  fig.patch.set_facecolor(SURFACE)
  return fig, (axes if columns > 1 else [axes])


def save(fig, path):
  fig.tight_layout()
  fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
  plt.close(fig)
  print(f"  -> {path}")


def load(path):
  if not os.path.isfile(path):
    print(f"  (missing) {path}")
    return []
  with open(path) as f:
    rows = json.load(f)["rows"]
  rows = [r for r in rows if "per_epoch" in r]
  for row in rows:
    enrich(row)
  return rows


def bias_key(rows):
  """`excess_paired` where the closed form exists, else the held-out `test`."""
  if any(r.get("excess_paired") is not None for r in rows):
    return "excess_paired", "excess against bayes_risk (loss units)"
  if any(r.get("excess_reported") is not None for r in rows):
    return "excess_reported", "reported minus bayes_risk (loss units)"
  return "test", "held-out loss (loss units; floor unknown)"


def aggregate(rows, group, value):
  """`{group_value: (median, low, high)}` with the range being the min and max over the cell."""
  buckets = collections.defaultdict(list)
  for row in rows:
    if row.get(value) is None or not np.isfinite(row[value]):
      continue
    buckets[row[group]].append(float(row[value]))
  return {k: (float(np.median(v)), float(np.min(v)), float(np.max(v))) for k, v in buckets.items() if len(v) > 0}


def line_by_arm(axes, rows, group, value, logx=False):
  for arm in sorted({r["arm"] for r in rows}):
    subset = [r for r in rows if r["arm"] == arm]
    points = aggregate(subset, group, value)
    if len(points) == 0:
      continue
    x = sorted(points)
    median = [points[k][0] for k in x]
    low = [points[k][1] for k in x]
    high = [points[k][2] for k in x]
    axes.fill_between(x, low, high, color=ARM_COLOUR[arm], alpha=0.14, linewidth=0, zorder=1)
    axes.plot(
      x, median, color=ARM_COLOUR[arm], linewidth=2.0, marker=ARM_MARKER[arm], markersize=8, markeredgecolor=SURFACE,
      markeredgewidth=1.5, label=arm, zorder=3
    )
  if logx:
    axes.set_xscale("log")
  axes.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)


def plot_precision(rows, plots_dir):
  if len(rows) == 0:
    return
  key, label = bias_key(rows)
  fig, (left, right) = figure(width=5.6, columns=2)
  line_by_arm(left, rows, "loss_precision", key, logx=True)
  style(
    left, "loss_precision bar (loss units, log)", label, "H1/H2  bias against the bar\nband = min-max over designs and seeds"
  )
  left.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":", zorder=2)
  line_by_arm(right, rows, "loss_precision", "window", logx=True)
  right.set_yscale("log")
  style(right, "loss_precision bar (loss units, log)", "exit window (train rows, log)", "exit window against the bar")
  save(fig, os.path.join(plots_dir, "1_excess_vs_precision.png"))


def plot_patience(datasets, plots_dir):
  present = [(name, rows) for name, rows in datasets if len(rows) > 0 and len({r["patience"] for r in rows}) > 1]
  if len(present) == 0:
    print("  (no patience sweep with more than one level yet)")
    return
  fig, axes = plt.subplots(2, len(present), figsize=(5.2 * len(present), 7.4), squeeze=False)
  fig.patch.set_facecolor(SURFACE)
  for column, (name, rows) in enumerate(present):
    key, label = bias_key(rows)
    line_by_arm(axes[0][column], rows, "patience", key)
    axes[0][column].set_xscale("log", base=2)
    style(axes[0][column], "patience (epochs, log2)", label, f"H3  {name}: bias against patience")
    line_by_arm(axes[1][column], rows, "patience", "window")
    axes[1][column].set_xscale("log", base=2)
    style(axes[1][column], "patience (epochs, log2)", "exit window (train rows)", f"{name}: exit window against patience")
  fig.tight_layout()
  fig.savefig(os.path.join(plots_dir, "2_patience.png"), dpi=160, facecolor=SURFACE)
  plt.close(fig)
  print(f"  -> {os.path.join(plots_dir, '2_patience.png')}")


def plot_exit_decomposition(rows, plots_dir, axis="loss_precision"):
  """`gap` and `err` stacked at the exit, one bar per (level, arm), against the bar they must clear.

  Both component hues sit below 3:1 against the light surface, so every segment carries a direct
  value label -- the relief rule -- and the arm is named in the tick label rather than by colour.
  """
  rows = [r for r in rows if r.get(axis) is not None and r.get("loss_precision") is not None]
  if len(rows) == 0:
    print("  (no rows carry loss_precision for the exit decomposition)")
    return
  levels = sorted({r[axis] for r in rows})
  arms = sorted({r["arm"] for r in rows})
  cells, labels = [], []
  for level in levels:
    for arm in arms:
      subset = [r for r in rows if r["arm"] == arm and r[axis] == level]
      if len(subset) == 0:
        continue
      cells.append((
        float(np.median([r["diff"] for r in subset])), float(np.median([r["err"]
                                                                        for r in subset])), float(subset[0]["loss_precision"]),
      ))
      labels.append(f"{level:g}\n{arm.replace('from_scratch', 'scratch')}")
  fig, (axes, ) = figure(width=max(7.0, 0.9 * len(cells)))
  positions = np.arange(len(cells), dtype=float)
  gaps = [c[0] for c in cells]
  errs = [c[1] for c in cells]
  bars = [c[2] for c in cells]
  axes.bar(positions, gaps, 0.62, color=COMPONENT_COLOUR["gap"], zorder=3, label="gap = |val - train|")
  axes.bar(
    positions, errs, 0.62, bottom=gaps, color=COMPONENT_COLOUR["err"], zorder=3, label="err = hypot(train_sem, val_sem)",
    edgecolor=SURFACE, linewidth=2.0
  )
  for position, gap, err, bar in zip(positions, gaps, errs, bars):
    axes.text(position, gap / 2, f"{gap:.4f}", ha="center", va="center", fontsize=6.5, color=INK)
    axes.text(position, gap + err / 2, f"{err:.4f}", ha="center", va="center", fontsize=6.5, color=INK)
    axes.text(
      position, gap + err, f"  {100 * err / (gap + err):.0f}% err", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY
    )
    axes.plot([position - 0.42, position + 0.42], [bar, bar], color=INK, linewidth=1.6, linestyle="--", zorder=4)
  axes.plot([], [], color=INK, linewidth=1.6, linestyle="--", label="loss_precision bar")
  axes.set_xticks(positions)
  axes.set_xticklabels(labels, fontsize=7)
  style(
    axes, f"{axis} and arm", "exit-time slack components (loss units)",
    "Exit decomposition: which term is binding\nsum must sit at or below the dashed bar"
  )
  axes.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left", bbox_to_anchor=(1.01, 1.0))
  save(fig, os.path.join(plots_dir, "3_exit_decomposition.png"))


def plot_curves(datasets, plots_dir, rank=None):
  panels = []
  for name, rows in datasets:
    for arm in sorted({r["arm"] for r in rows}):
      subset = [r for r in rows if r["arm"] == arm and (rank is None or r.get("rank") == rank)]
      if len(subset) == 0:
        continue
      subset.sort(key=lambda r: (r["seed"], r.get("rank", r.get("position", 0))))
      panels.append((f"{name} / {arm}", subset[len(subset) // 2]))
  if len(panels) == 0:
    print("  (no rows for the per-epoch curves yet)")
    return
  columns = min(3, len(panels))
  rowcount = (len(panels) + columns - 1) // columns
  fig, axes = plt.subplots(rowcount, columns, figsize=(5.0 * columns, 3.5 * rowcount), squeeze=False)
  fig.patch.set_facecolor(SURFACE)
  for index, (label, row) in enumerate(panels):
    axis = axes[index // columns][index % columns]
    history = row["per_epoch"]
    train = np.asarray(history["train"], np.float64)
    validation = np.asarray(history["val"], np.float64)
    window = np.asarray(history["window"], np.int64)
    epochs = np.arange(train.size)
    changed = np.flatnonzero(np.diff(window) != 0)
    axis.plot(epochs, train, color=ARM_COLOUR[row["arm"]], linewidth=2.0, label="train", zorder=3)
    axis.plot(epochs, validation, color=ARM_COLOUR[row["arm"]], linewidth=2.0, linestyle="--", label="validation", zorder=3)
    if changed.size > 0:
      axis.axvline(int(changed[-1]) + 1, color=INK_SECONDARY, linewidth=1.2, linestyle=":", zorder=2)
      axis.text(int(changed[-1]) + 1, axis.get_ylim()[1], " last addition", fontsize=7, color=INK_SECONDARY, va="top")
    axis.axvline(train.size - 1, color=INK, linewidth=1.2, zorder=2)
    if row.get("bayes_risk") is not None:
      axis.axhline(row["bayes_risk"], color=INK, linewidth=1.0, linestyle="-.", zorder=2)
      axis.text(0, row["bayes_risk"], " bayes_risk", fontsize=7, color=INK, va="bottom")
    style(axis, "epoch", "loss (loss units)", f"{label}\nwindow {row['window']}, {train.size} epochs, stop at the solid line")
    axis.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
  for index in range(len(panels), rowcount * columns):
    axes[index // columns][index % columns].axis("off")
  fig.tight_layout()
  fig.savefig(os.path.join(plots_dir, "4_curves.png"), dpi=160, facecolor=SURFACE)
  plt.close(fig)
  print(f"  -> {os.path.join(plots_dir, '4_curves.png')}")


def plot_remaining(datasets, plots_dir):
  rows = [
    r for _name, subset in datasets for r in subset
    if r.get("remaining_train") is not None and r.get("loss_precision") is not None
  ]
  if len(rows) == 0:
    print("  (no fitted tails yet)")
    return
  fig, (axes, ) = figure(width=7.0)
  for arm in sorted({r["arm"] for r in rows}):
    subset = [r for r in rows if r["arm"] == arm]
    axes.scatter([r["window"] for r in subset], [r["remaining_train"] for r in subset], s=44, color=ARM_COLOUR[arm],
                 edgecolor=SURFACE, linewidth=1.2, label=arm, zorder=3)
  precision = float(np.median([r["loss_precision"] for r in rows]))
  axes.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":", zorder=2)
  axes.axhline(0.1 * precision, color=INK, linewidth=1.0, linestyle="--", zorder=2)
  axes.text(min(r["window"] for r in rows), 0.1 * precision, " 10% of loss_precision", fontsize=7, color=INK, va="bottom")
  axes.set_xscale("log")
  style(
    axes, "exit window (train rows, log)", "remaining descent, final minus fitted asymptote (loss units)",
    "Did it stop mid-descent? Tail fit A*exp(alpha*t)+c after the last addition"
  )
  axes.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
  save(fig, os.path.join(plots_dir, "5_remaining_descent.png"))


def plot_rewind(rows, plots_dir):
  if len(rows) == 0:
    return
  cells = sorted({r["cell"] for r in rows})
  arms = sorted({r["arm"] for r in rows})
  fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.2))
  fig.patch.set_facecolor(SURFACE)

  for arm in arms:
    subset = [r for r in rows if r["arm"] == arm]
    points = aggregate(subset, "position", "displacement")
    if len(points) == 0:
      continue
    x = sorted(points)
    axes[0].fill_between(
      x, [points[k][1] for k in x], [points[k][2] for k in x], color=ARM_COLOUR[arm], alpha=0.14, linewidth=0
    )
    axes[0].plot(
      x, [points[k][0] for k in x], color=ARM_COLOUR[arm], linewidth=2.0, marker=ARM_MARKER[arm], markersize=8,
      markeredgecolor=SURFACE, markeredgewidth=1.5, label=arm, zorder=3
    )
  axes[0].axhline(0.01, color=INK, linewidth=1.0, linestyle="--", zorder=2)
  axes[0].text(1, 0.011, " 1% of the parameter norm: below this the rewind is a no-op", fontsize=7, color=INK)
  axes[0].set_yscale("log")
  style(
    axes[0], "position in the design sequence", "||p - q|| / ||p|| at the design's end",
    "H7  within-design displacement the rewind acts on\nposition 1: meta's carried network is untrained, so both arms match"
  )
  axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

  positions = np.arange(len(cells), dtype=float)
  width = 0.36
  for offset, arm in zip((-width / 2 - 0.02, width / 2 + 0.02), arms):
    for panel, value, name in ((axes[1], "window", "exit window (train rows)"), (axes[2], "excess_reported",
                                                                                 "reported minus bayes_risk (loss units)")):
      heights = []
      for cell in cells:
        subset = [r[value] for r in rows if r["arm"] == arm and r["cell"] == cell and r.get(value) is not None]
        heights.append(float(np.median(subset)) if len(subset) > 0 else float("nan"))
      panel.bar(positions + offset, heights, width, color=ARM_COLOUR[arm], zorder=3, label=arm)
      panel.set_xticks(positions)
      panel.set_xticklabels(cells, fontsize=7, rotation=20, ha="right")
      style(panel, "restart cell", name, f"H7a  {name.split(' (')[0]} per restart cell")
      panel.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
  save(fig, os.path.join(plots_dir, "6_rewind.png"))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--plots-dir", default="output/criterion-bench/plots")
  parser.add_argument("--precision", default="output/biascross/linear_precision.json")
  parser.add_argument("--linear", default="output/biascross/linear.json")
  parser.add_argument("--mm", default="output/biascross/mm.json")
  parser.add_argument("--rewind", default="output/criterion-bench/rewind.json")
  arguments = parser.parse_args()

  os.makedirs(arguments.plots_dir, exist_ok=True)
  precision = load(arguments.precision)
  linear = load(arguments.linear)
  mm = load(arguments.mm)
  rewind = load(arguments.rewind)

  plot_precision(precision, arguments.plots_dir)
  plot_patience([("linear", linear), ("MM", mm)], arguments.plots_dir)
  decomposition = next((d for d in (precision, mm, linear, rewind) if len(d) > 0), [])
  plot_exit_decomposition(decomposition, arguments.plots_dir)
  plot_curves([("linear", linear), ("MM", mm), ("rewind", rewind)], arguments.plots_dir)
  plot_remaining([("linear", linear), ("MM", mm), ("precision", precision), ("rewind", rewind)], arguments.plots_dir)
  plot_rewind(rewind, arguments.plots_dir)


if __name__ == "__main__":
  main()
