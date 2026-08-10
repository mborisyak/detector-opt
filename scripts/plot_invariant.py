#!/usr/bin/env python3
"""Read the analytic-benchmark shards and report what they support -- and only that.

Every statistic here is the one this project learned to use the hard way (docs/enzyme-gbdt-log.md,
the 2026-08-10 audits):

* p-values are TWO-SIDED. There is no pre-registered direction: the whole question is which kernel
  wins, and a one-sided test assumes the answer.
* the BO-vs-null tests within one (objective, m) cell share ONE null, so they are not independent;
  Holm across the four is reported next to the raw value.
* the headline is a ratio of medians, so it gets a bootstrap CI rather than being quoted bare.
* kernel-vs-kernel is PAIRED by seed, which is legitimate here for the reason it was legitimate on
  the enzyme benchmark: every arm shares its seed's initial design, verified rather than assumed.

    python scripts/plot_invariant.py --directory output/invariant --output output/invariant/summary
"""

import argparse
import glob
import json

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import mannwhitneyu, wilcoxon

ORDER = ["sorting-rbf", "normalised-invariant-rbf", "permutation-invariant-rbf", "ard-rbf", "random"]
COLOUR = {"sorting-rbf": "#7d3ac1", "normalised-invariant-rbf": "#2a9d3a",
          "permutation-invariant-rbf": "#eb6834", "ard-rbf": "#2a78d6", "random": "#888888"}


def load(directory):
  curves, meta = {}, None
  for path in sorted(glob.glob(f"{directory}/*.json")):
    with open(path) as f:
      shard = json.load(f)
    # This script writes its own summary.json into the same directory, so a bare glob re-reads it on
    # the next run. Skip anything that is not a shard rather than crashing on it.
    if not isinstance(shard, dict) or "curves" not in shard:
      continue
    meta = meta or shard
    for key, value in shard["curves"].items():
      curves[key] = np.asarray(value)
  if meta is None:
    raise SystemExit(f"no shard JSON under {directory}")
  return curves, meta


def holm(raw):
  order, out, running = sorted(raw, key=raw.get), {}, 0.0
  for i, key in enumerate(order):
    running = out[key] = min(1.0, max(running, (len(order) - i) * raw[key]))
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--directory", default="output/invariant")
  parser.add_argument("--output", default="output/invariant/summary")
  arguments = parser.parse_args()

  curves, meta = load(arguments.directory)
  objectives = sorted({k.split("|")[0] for k in curves})
  dimensions = sorted({int(k.split("|")[1]) for k in curves})
  kernels = [k for k in ORDER if k != "random"]

  rows = []
  for objective in objectives:
    for m in dimensions:
      null = curves.get(f"{objective}|{m}|random")
      if null is None:
        continue
      raw = {}
      for kernel in kernels:
        arm = curves.get(f"{objective}|{m}|{kernel}")
        if arm is None:
          continue
        raw[kernel] = mannwhitneyu(arm[:, -1], null[:, -1], alternative="two-sided").pvalue
      adjusted = holm(raw)
      for kernel in raw:
        arm = curves[f"{objective}|{m}|{kernel}"]
        rng = np.random.default_rng(0)
        boot = [np.median(rng.choice(null[:, -1], null.shape[0])) / max(np.median(rng.choice(arm[:, -1], arm.shape[0])), 1e-300)
                for _ in range(2000)]
        rows.append({
          "objective": objective, "m": m, "kernel": kernel, "n_seeds": arm.shape[0],
          "best": float(np.median(arm[:, -1])), "null": float(np.median(null[:, -1])),
          "ratio": float(np.median(null[:, -1]) / max(np.median(arm[:, -1]), 1e-300)),
          "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
          "p_raw": float(raw[kernel]), "p_holm": float(adjusted[kernel]),
        })

  print(f"{'objective':11s} {'m':>2} {'kernel':26s} {'best@N':>9} {'null/BO':>8} {'95% CI':>16} "
        f"{'p':>7} {'Holm':>7}")
  for row in rows:
    print(f"{row['objective']:11s} {row['m']:2d} {row['kernel']:26s} {row['best']:9.5f} "
          f"{row['ratio']:7.3f}x [{row['ci'][0]:5.2f},{row['ci'][1]:6.2f}] {row['p_raw']:7.4f} {row['p_holm']:7.4f}")

  print("\nsorting vs the group average, PAIRED by seed (two-sided):")
  for objective in objectives:
    for m in dimensions:
      a = curves.get(f"{objective}|{m}|sorting-rbf")
      b = curves.get(f"{objective}|{m}|permutation-invariant-rbf")
      c = curves.get(f"{objective}|{m}|normalised-invariant-rbf")
      if a is None or b is None:
        continue
      line = (f"  {objective:11s} m={m}: sort<group {int((a[:, -1] < b[:, -1]).sum())}/{a.shape[0]}"
              f" p={wilcoxon(a[:, -1], b[:, -1]).pvalue:.3f}")
      if c is not None:
        line += (f"   norm<group {int((c[:, -1] < b[:, -1]).sum())}/{c.shape[0]}"
                 f" p={wilcoxon(c[:, -1], b[:, -1]).pvalue:.3f}")
      print(line)

  # Lay the panels out as a GRID over the (objective, m) cells that actually have data, rather than
  # as objectives x dimensions. A sweep restricted to one dimension would otherwise degenerate to a
  # single tall column, which clips the legend and wastes the page.
  cells = [(objective, m) for objective in objectives for m in dimensions
           if f"{objective}|{m}|random" in curves]
  columns = int(np.ceil(np.sqrt(len(cells)))) if len(cells) > 0 else 1
  n_rows = int(np.ceil(len(cells) / columns))
  figure = Figure(figsize=(5.0 * columns, 3.8 * n_rows), dpi=130)
  axes = np.atleast_2d(figure.subplots(n_rows, columns, squeeze=False))
  for axis in axes.ravel()[len(cells):]:
    axis.set_visible(False)
  for index, (objective, m) in enumerate(cells):
      i, j = divmod(index, columns)
      axis = axes[i, j]
      for kernel in ORDER:
        arm = curves.get(f"{objective}|{m}|{kernel}")
        if arm is None:
          continue
        iterations = np.arange(1, arm.shape[1] + 1)
        median = np.median(arm, axis=0)
        axis.plot(iterations, median, color=COLOUR[kernel], lw=1.6,
                  ls="--" if kernel == "random" else "-", label=kernel if index == 0 else None)
        low, high = np.percentile(arm, [25, 75], axis=0)
        axis.fill_between(iterations, low, high, color=COLOUR[kernel], alpha=0.10, lw=0)
      axis.set_yscale("log")
      axis.set_title(f"{objective}, m = {m}", fontsize=10)
      axis.set_xlabel("BO iteration", fontsize=9)
      if j == 0:
        axis.set_ylabel("best-so-far (median, IQR)", fontsize=9)
      axis.tick_params(labelsize=8)
      axis.grid(alpha=0.25, lw=0.5)
  figure.legend(loc="lower center", ncol=5, fontsize=9, frameon=False)
  figure.suptitle(f"Permutation-invariant analytic objectives: {meta['seeds']} seeds, "
                  f"{meta['iterations']} iterations, bands are the interquartile range", fontsize=11)
  figure.tight_layout(rect=(0, 0.05, 1, 0.96))
  figure.savefig(f"{arguments.output}.png")
  with open(f"{arguments.output}.json", "w") as f:
    json.dump(rows, f, indent=2)
  print(f"\nwrote {arguments.output}.png and {arguments.output}.json")


if __name__ == "__main__":
  main()
