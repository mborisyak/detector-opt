#!/usr/bin/env python3
"""Two figures from the SHiP `loss_precision` campaigns, read from a local copy of the run tree.

    python scripts/plot_ship_campaign.py --tree output/ship-prec --out output/ship-prec/plots

`ship-campaign-median.png` -- median best-so-far against cumulative detector calls, one panel per bar.
Partial cells have different lengths, so every curve is interpolated onto a common call grid and the
arm is drawn ONLY while EVERY one of its seeds is still alive. A median over a CHANGING set of seeds
is not monotone -- when the lowest cell ends the median jumps UP, which on a best-so-far axis reads as
the optimiser getting worse and is an artefact. Holding the set fixed costs range and buys a curve
that means one thing along its whole length.

`ship-arm-spend-paired.png` -- the axis on which a warm start should show. At a fixed `loss_precision`
the exit test drives every converged design to the same bar, so arms can differ in SPEND but barely in
LOSS. Pairing is only legitimate on the `n_init` initial proposals, which are shared across arms at a
seed; beyond those BO diverges and index-matching compares unlike designs. Design 0 is excluded
because `meta` has no history there and IS `from_scratch` by construction. Design identity is checked
before any pair is used.

⚠️ EVERY RATIO IS DRAWN WITH ITS SPREAD. Per-design spend varies by nearly an order of magnitude
WITHIN one cell, so a geometric mean without an interval is not a result.
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, INK_3 = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
ARMS = ['from_scratch', 'continue', 'closest', 'meta']
BARS = [('prec1e2', 'loss_precision 1.0e-2'), ('prec5e3', 'loss_precision 5.0e-3')]
N_INIT = 5

plt.rcParams.update({
  'figure.facecolor': SURFACE,
  'axes.facecolor': SURFACE,
  'savefig.facecolor': SURFACE,
  'axes.edgecolor': INK_3,
  'axes.linewidth': 0.8,
  'axes.labelcolor': INK_2,
  'text.color': INK,
  'xtick.color': INK_2,
  'ytick.color': INK_2,
  'xtick.labelsize': 8,
  'ytick.labelsize': 8,
  'axes.labelsize': 9,
  'axes.titlesize': 10,
  'legend.fontsize': 8,
  'legend.frameon': False,
  'grid.color': '#e6e5e0',
  'grid.linewidth': 0.7,
  'lines.linewidth': 2.0,
})


def load(tree, bar):
  """``{arm: {seed: [result rows]}}`` for one bar."""
  out = {}
  for path in sorted(glob.glob(os.path.join(tree, bar, '*', '*', 'results.json'))):
    seed, arm = path.split(os.sep)[-3:-1]
    with open(path) as handle:
      rows = json.load(handle)['results']
    if len(rows) > 0:
      out.setdefault(arm, {})[seed] = rows
  return out


def median_trajectories(tree, out_dir):
  figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
  grid = np.linspace(3.0e4, 1.05e6, 220)
  for axis, (bar, title) in zip(axes, BARS):
    data = load(tree, bar)
    for slot, arm in enumerate(ARMS):
      seeds = data.get(arm, {})
      if len(seeds) == 0:
        continue
      curves = []
      for rows in seeds.values():
        calls = np.cumsum([row['spent'] for row in rows])
        best = np.minimum.accumulate([row['loss'] for row in rows])
        curve = np.full_like(grid, np.nan)
        live = grid <= calls[-1]
        curve[live] = np.interp(grid[live], calls, best)
        curves.append(curve)
      stack = np.vstack(curves)
      contributing = np.sum(~np.isnan(stack), axis=0)
      drawn = contributing == stack.shape[0]
      if not np.any(drawn):
        continue
      usable = stack[:, drawn]
      median = np.median(usable, axis=0)
      low, high = np.min(usable, axis=0), np.max(usable, axis=0)
      axis.fill_between(grid[drawn], low, high, color=SERIES[slot], alpha=0.10, linewidth=0)
      axis.plot(grid[drawn], median, color=SERIES[slot], label=f'{arm}  ({len(curves)} seeds)')
    axis.axvline(1048576, color=INK_3, linewidth=1.0, linestyle=(0, (4, 3)))
    axis.grid(True, alpha=0.9)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
      axis.spines[side].set_visible(False)
    axis.set_xlabel('cumulative detector calls')
    axis.set_title(title, loc='left', pad=8)
    axis.set_xlim(0, 1.15e6)
  axes[0].set_ylabel('median best loss so far  (band = min-max across seeds)')
  for axis in axes:
    axis.legend(loc='upper right', title='arm (cells contributing)', title_fontsize=8)
  figure.suptitle(
    'SHiP campaigns, PARTIAL -- median across seeds, drawn only while EVERY seed is still alive', x=0.01, ha='left',
    fontsize=10, color=INK
  )
  figure.tight_layout(rect=(0, 0, 1, 0.93))
  target = os.path.join(out_dir, 'ship-campaign-median.png')
  figure.savefig(target, dpi=160)
  plt.close(figure)
  return target


def paired_spend(tree, out_dir, bar='prec1e2'):
  data = load(tree, bar)
  seeds = sorted({seed for arm in ARMS for seed in data.get(arm, {})})
  pairs = {arm: [] for arm in ARMS}
  for seed in seeds:
    if not all(seed in data.get(arm, {}) for arm in ARMS):
      continue
    rows = {arm: data[arm][seed] for arm in ARMS}
    for index in range(1, N_INIT):
      if any(len(rows[arm]) <= index for arm in ARMS):
        continue
      designs = {tuple(np.round(rows[arm][index]['x_scaled'], 6)) for arm in ARMS}
      if len(designs) != 1:
        continue
      for arm in ARMS:
        pairs[arm].append(rows[arm][index]['spent'])

  reference = np.asarray(pairs['from_scratch'], float)
  n = reference.shape[0]
  spread = max(float(np.max(np.abs(np.log2(np.asarray(pairs[arm], float) / reference)))) for arm in ARMS[1:])
  label_x = spread + 0.25
  figure, axis = plt.subplots(figsize=(9.6, 3.6))
  for slot, arm in enumerate(ARMS[1:], start=1):
    ratios = np.log2(np.asarray(pairs[arm], float) / reference)
    jitter = (np.arange(n) % 5 - 2) * 0.045
    axis.scatter(ratios, np.full(n, slot) + jitter, s=26, color=SERIES[slot], alpha=0.55, linewidths=0)
    mean = float(np.mean(ratios))
    half = 1.96 * float(np.std(ratios, ddof=1)) / np.sqrt(n)
    axis.plot([mean - half, mean + half], [slot, slot], color=SERIES[slot], linewidth=2.5, solid_capstyle='butt')
    axis.plot([mean], [slot], marker='|', markersize=16, color=INK, markeredgewidth=2.0)
    axis.text(
      label_x, slot, f'x{2 ** mean:.2f}  [{2 ** (mean - half):.2f}, {2 ** (mean + half):.2f}]', fontsize=8, color=INK_2,
      va='center', ha='left', fontfamily='monospace'
    )
  axis.axvline(0.0, color=INK_3, linewidth=1.0)
  axis.set_xlim(-spread - 0.15, label_x + 1.5)
  axis.set_yticks(range(1, len(ARMS)))
  axis.set_yticklabels(ARMS[1:])
  axis.set_ylim(0.4, len(ARMS) - 0.4)
  axis.set_xlabel('log2( spend / from_scratch spend ), same design, same seed   <- warm start cheaper | dearer ->')
  axis.grid(True, axis='x', alpha=0.9)
  axis.set_axisbelow(True)
  for side in ('top', 'right', 'left'):
    axis.spines[side].set_visible(False)
  figure.suptitle(
    f'SHiP {bar}: paired spend against a cold start -- {n} PAIRED DESIGNS ({n // (N_INIT - 1)} seeds x designs 1-{N_INIT - 1})',
    x=0.01, ha='left', fontsize=10, color=INK
  )
  figure.tight_layout(rect=(0, 0, 1, 0.90))
  target = os.path.join(out_dir, 'ship-arm-spend-paired.png')
  figure.savefig(target, dpi=160)
  plt.close(figure)
  return target, n


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tree', default='output/ship-prec')
  parser.add_argument('--out', default='output/ship-prec/plots')
  arguments = parser.parse_args()
  os.makedirs(arguments.out, exist_ok=True)
  print(f'wrote {median_trajectories(arguments.tree, arguments.out)}')
  target, n = paired_spend(arguments.tree, arguments.out)
  print(f'wrote {target}  (n={n} paired cells)')


if __name__ == '__main__':
  main()
