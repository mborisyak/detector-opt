#!/usr/bin/env python3
"""Plot what `scripts/optimizer_bakeoff.py` measured: five optimisers under two learning-rate schedules.

THREE OUTPUTS, and the third is not decoration. `curves.png` is the loss against epoch, as a 2x2 of
small multiples -- rows train / validation, columns the two schedules -- on ONE shared y-axis, so every
panel is read against the same scale and no pair of arms is compared across different axes.
`final.png` is the headline magnitude: the objective each arm converged to. `summary.md` is the TABLE
VIEW, which the palette check requires (two of the five hues sit under 3:1 against the surface, and the
rule for that is that the numbers must be legible somewhere that is not the colour).

The five hues are the reference categorical palette's slots 1, 8, 3, 7, 4, validated as a set for this
chart's all-pairs case: every hard check passes, with the aqua/red pair in the 6-8 CVD band. That band
is legal only WITH secondary encoding, so each optimiser also carries its own dash pattern -- which is
what makes the figure survive greyscale printing and colour-blind readers, not the colour alone.

Arms that did not converge are drawn (the curve is still what happened) and marked in the table; arms
with no file yet are skipped, so this can be run against a sweep still in flight.

    srun --cpus-per-task=2 --mem=1800 -u python scripts/plot_optimizer_bakeoff.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPTIMIZERS = ("adamw", "adamaxw", "adan", "nadamw", "amsgrad")
SCHEDULES = ("constant", "hyperbolic")
COLOR = {
  "adamw": "#2a78d6",
  "adamaxw": "#e34948",
  "adan": "#1baf7a",
  "nadamw": "#4a3aa7",
  "amsgrad": "#eda100",
}
DASH = {
  "adamw": (None, None),
  "adamaxw": (5, 2),
  "adan": (1.5, 1.5),
  "nadamw": (7, 2, 1.5, 2),
  "amsgrad": (3, 1.5, 1.5, 1.5),
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8f8e88"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"


def load(directory):
  """Every arm on disk, keyed ``(optimizer, schedule)``, plus the sweep's meta."""
  arms = {}
  for optimizer in OPTIMIZERS:
    for schedule in SCHEDULES:
      path = os.path.join(directory, f"{optimizer}-{schedule}.json")
      if os.path.exists(path):
        with open(path) as f:
          arms[(optimizer, schedule)] = json.load(f)
  meta_path = os.path.join(directory, "meta.json")
  meta = {}
  if os.path.exists(meta_path):
    with open(meta_path) as f:
      meta = json.load(f)
  return arms, meta


def style(axes):
  """Recessive frame: the data is the only thing at full strength."""
  axes.set_facecolor(SURFACE)
  axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
  axes.set_axisbelow(True)
  for side in ("top", "right"):
    axes.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axes.spines[side].set_color(GRID)
  axes.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)


def curves(arms, meta, path):
  """Left column: the five optimisers at the constant rate. Right column: the SCHEDULE, on `adamw`
    alone, which is the only arm that carries both -- so the right panel is adamw against itself,
    paired on everything else. Colour stays with the optimiser in both, and the schedule is carried by
    the line style plus a direct label, so no hue means two different things across the figure."""
  figure, panels = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex="col", sharey=True)
  figure.patch.set_facecolor(SURFACE)

  values = [v for record in arms.values() for key in ("train_loss_per_epoch", "val_loss_per_epoch")
            for v in record[key]]
  if len(values) > 0:
    low, high = float(np.min(values)), float(np.percentile(values, 99.5))
    margin = 0.05 * (high - low)
    panels[0][0].set_ylim(low - margin, high + margin)

  for row, series in enumerate(("train", "val")):
    axes = panels[row][0]
    style(axes)
    for optimizer in OPTIMIZERS:
      record = arms.get((optimizer, "constant"))
      if record is None or len(record[f"{series}_loss_per_epoch"]) == 0:
        continue
      y = record[f"{series}_loss_per_epoch"]
      line, = axes.plot(np.arange(1, len(y) + 1), y, color=COLOR[optimizer], linewidth=1.8,
                        solid_capstyle="round", zorder=3, label=optimizer)
      if DASH[optimizer][0] is not None:
        line.set_dashes(DASH[optimizer])
    axes.set_ylabel(f"{'training' if series == 'train' else 'validation'} loss",
                    color=INK_SECONDARY, fontsize=10)

    axes = panels[row][1]
    style(axes)
    for schedule, dashes in (("constant", None), ("hyperbolic", (2, 2))):
      record = arms.get(("adamw", schedule))
      if record is None or len(record[f"{series}_loss_per_epoch"]) == 0:
        continue
      y = record[f"{series}_loss_per_epoch"]
      line, = axes.plot(np.arange(1, len(y) + 1), y, color=COLOR["adamw"], linewidth=1.8,
                        solid_capstyle="round", zorder=3)
      if dashes is not None:
        line.set_dashes(dashes)
      axes.annotate(schedule, xy=(len(y), y[-1]), xytext=(5, 0), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=9, va="center", annotation_clip=False)
    axes.margins(x=0.14)

  k = meta.get("k", "K")
  panels[0][0].set_title("constant learning rate", color=INK, fontsize=11, pad=10, loc="left")
  panels[0][1].set_title(f"adamw only: constant vs alpha / (t / {k} + 1)", color=INK, fontsize=11,
                         pad=10, loc="left")
  for column in (0, 1):
    panels[1][column].set_xlabel("epoch", color=INK_SECONDARY, fontsize=10)

  handles, labels = panels[0][0].get_legend_handles_labels()
  legend = figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.075, 0.935), ncol=5,
                         frameon=False, fontsize=10, handlelength=3.0)
  for text in legend.get_texts():
    text.set_color(INK_SECONDARY)

  figure.suptitle("Optimisers on one design, from `meta`'s own network and pool",
                  color=INK, fontsize=13, x=0.008, ha="left", y=0.988)
  if "best_loss_reported" in meta:
    figure.text(0.008, 0.955,
                f"best design of the run: iteration {meta['best_iteration']}, reported "
                f"{meta['best_loss_reported']:.4f}   ·   n0 {meta.get('n0')}, increment "
                f"{meta.get('n_increment')}   ·   alpha {meta.get('learning_rate')}, "
                f"weight decay {meta.get('weight_decay')}",
                color=INK_SECONDARY, fontsize=9.5, ha="left")
  figure.tight_layout(rect=(0, 0, 1, 0.87))
  figure.savefig(path, dpi=160, facecolor=SURFACE)
  plt.close(figure)


def final(arms, path):
  """The headline magnitude: what each arm converged to. The schedule arm sits beside its own optimiser,
    in that optimiser's colour and hatched, because it is the same entity under a second condition."""
  cells = [(o, "constant") for o in OPTIMIZERS if (o, "constant") in arms]
  cells += [(o, "hyperbolic") for o in OPTIMIZERS if (o, "hyperbolic") in arms]
  cells = [c for c in cells if arms[c]["objective"] is not None]
  if len(cells) == 0:
    return

  figure, axes = plt.subplots(figsize=(9.5, 4.6))
  figure.patch.set_facecolor(SURFACE)
  style(axes)

  for index, (optimizer, schedule) in enumerate(cells):
    record = arms[(optimizer, schedule)]
    axes.bar(index, record["objective"], 0.68, color=COLOR[optimizer], edgecolor=SURFACE, linewidth=1.2,
             hatch=None if schedule == "constant" else "///", zorder=3)
    axes.text(index, record["objective"], f" {record['objective']:.4f}", ha="center", va="bottom",
              fontsize=8.5, color=INK_SECONDARY, rotation=90)

  axes.set_xticks(range(len(cells)))
  axes.set_xticklabels([o if s == "constant" else f"{o}\nhyperbolic" for o, s in cells],
                       color=INK_SECONDARY, fontsize=10)
  axes.set_ylabel("objective (validation loss at convergence)", color=INK_SECONDARY, fontsize=10)
  axes.set_title("solid: constant learning rate    hatched: hyperbolic decay",
                 color=INK_SECONDARY, fontsize=10, loc="left", pad=8)
  figure.tight_layout()
  figure.savefig(path, dpi=160, facecolor=SURFACE)
  plt.close(figure)


def summary(arms, meta, path):
  """The table view. Two of the five hues are under 3:1 on this surface, so the numbers must be
    readable somewhere that is not the colour -- and a table is also what anyone reruns against."""
  lines = ["# Optimizer bake-off", ""]
  if len(meta) > 0:
    lines += [
      f"Run `{meta['run']}`, {meta['n_completed']} designs completed. Trained on that run's BEST design",
      f"(iteration {meta['best_iteration']}, reported {meta['best_loss_reported']:.4f}), from its persistent",
      f"network and both budget pools. alpha = {meta['learning_rate']:g}, weight decay =",
      f"{meta['weight_decay']:g}, K = {meta['k']}.", "",
    ]
  lines += [
    "| optimizer | schedule | status | objective | +- | epochs | rounds | spent | wall (min) |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  for schedule in SCHEDULES:
    for optimizer in OPTIMIZERS:
      record = arms.get((optimizer, schedule))
      if record is None:
        continue
      objective = "" if record["objective"] is None else f"{record['objective']:.4f}"
      error = "" if record["objective_std"] is None else f"{record['objective_std']:.4f}"
      spent = "" if record["spent"] is None else f"{record['spent']}"
      rounds = len(set(record["train_budget_per_epoch"]))
      lines.append(
        f"| {optimizer} | {schedule} | {record['status']} | {objective} | {error} | "
        f"{record['n_epochs']} | {rounds} | {spent} | {record['wall_s'] / 60.0:.1f} |"
      )
  with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--directory", default="output/optimizer-bakeoff")
  arguments = parser.parse_args()

  arms, meta = load(arguments.directory)
  if len(arms) == 0:
    raise SystemExit(f"{arguments.directory}: no arm files yet")
  print(f"{len(arms)}/10 arms on disk")

  curves(arms, meta, os.path.join(arguments.directory, "curves.png"))
  final(arms, os.path.join(arguments.directory, "final.png"))
  print(summary(arms, meta, os.path.join(arguments.directory, "summary.md")))


if __name__ == "__main__":
  main()
