#!/usr/bin/env python3
"""Median best-so-far against DESIGN INDEX, not detector calls.

`median_convergence.py` plots against cumulative detector calls, which credits an arm for making its
designs cheap -- `meta` spends 74.5k calls a design against the others' ~92k, so it travels further
right per design. Against the design index the question is the other one: what does an arm get per
DESIGN PROBED, with cost set aside. Both are worth having and neither is the whole answer.

Seeds are median-ed pointwise and each arm is truncated at the shortest of its seeds, so every plotted
point is a median over the SAME number of runs.
"""
import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = ("from_scratch", "continue", "closest", "meta")
COLOR = {"from_scratch": "#2a78d6", "continue": "#e34948", "closest": "#1baf7a", "meta": "#4a3aa7"}
DASH = {"from_scratch": None, "continue": (5, 2), "closest": (1.5, 1.5), "meta": (7, 2, 1.5, 2)}
INK, INK_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e3e2dd"


def best_so_far(path):
  with open(path) as f:
    rows = json.load(f)["results"]
  return np.minimum.accumulate([r["loss"] for r in rows if r.get("loss") is not None])


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--task", default="output/enzyme_extremes")
  parser.add_argument("--seeds", nargs="+", default=["1244111331", "126382657"])
  parser.add_argument("--output", default="output/enzyme_extremes/per_design.png")
  arguments = parser.parse_args()

  figure, axes = plt.subplots(figsize=(9.5, 5.6))
  figure.patch.set_facecolor(SURFACE)
  axes.set_facecolor(SURFACE)
  axes.grid(True, color=GRID, linewidth=0.8)
  axes.set_axisbelow(True)
  for side in ("top", "right"):
    axes.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axes.spines[side].set_color(GRID)
  axes.tick_params(colors=INK_SECONDARY, labelsize=9)

  summary = {}
  for strategy in STRATEGIES:
    curves = []
    for seed in arguments.seeds:
      path = os.path.join(arguments.task, seed, strategy, "results.json")
      if os.path.exists(path):
        curves.append(best_so_far(path))
    if len(curves) == 0:
      continue
    length = min(len(c) for c in curves)  # every point a median over the same runs
    stacked = np.stack([c[:length] for c in curves])
    median = np.median(stacked, axis=0)
    summary[strategy] = {"designs": length, "final": float(median[-1]),
                         "per_seed_lengths": [int(len(c)) for c in curves]}
    line, = axes.plot(np.arange(1, length + 1), median, color=COLOR[strategy], linewidth=1.9,
                      solid_capstyle="round", label=strategy, zorder=3)
    if DASH[strategy] is not None:
      line.set_dashes(DASH[strategy])

  axes.set_yscale("log")
  axes.set_xlabel("designs probed", color=INK_SECONDARY, fontsize=10)
  axes.set_ylabel("median best-so-far loss", color=INK_SECONDARY, fontsize=10)
  axes.set_title("BO convergence per DESIGN (median across seeds, self-evaluated)", color=INK, fontsize=12,
                 loc="left", pad=10)
  legend = axes.legend(frameon=False, fontsize=10, handlelength=3.0)
  for text in legend.get_texts():
    text.set_color(INK_SECONDARY)
  figure.tight_layout()
  figure.savefig(arguments.output, dpi=160, facecolor=SURFACE)

  print(f"{'strategy':13s} {'designs plotted':>15s} {'per-seed':>12s} {'median final':>13s}")
  for strategy, row in summary.items():
    print(f"{strategy:13s} {row['designs']:15d} {str(row['per_seed_lengths']):>12s} {row['final']:13.4f}")
  print(f"-> {arguments.output}")


if __name__ == "__main__":
  main()
