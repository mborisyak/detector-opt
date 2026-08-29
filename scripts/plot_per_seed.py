#!/usr/bin/env python3
"""Per-seed best-so-far curves: one panel per SEED, one line per arm.

    python scripts/plot_per_seed.py output/ship-addr-prec1e2 --out output/render --name ship_seeds.png

The median and mean plots in `scripts/plot_median.py` aggregate ACROSS seeds and therefore hide the
quantity that decides whether an arm separates: the seed-to-seed spread. This shows every seed
separately, so a single adverse seed is visible as itself rather than as inflated error bars.

Complete cells only by default; `--partial` includes running cells, whose best-so-far is an upper
bound that will keep falling. Colour follows the ARM, matching `plot_median.py`, and each panel is
titled with the seed and the final ordering, because two of the four arm colours sit below 3:1
contrast against the chart surface and identity must not rest on colour alone.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, INK_3 = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8e58c9']
ARMS = ('from_scratch', 'continue', 'closest', 'meta', 'meta_reinit')
# `zip` TRUNCATES SILENTLY. A four-colour SERIES against five ARMS left `meta_reinit` with no entry
# and the plot died on a KeyError only when a tree actually contained that arm. Assert instead.
assert len(SERIES) >= len(ARMS), f'{len(ARMS)} arms need at least that many colours, got {len(SERIES)}'
ARM_COLOUR = dict(zip(ARMS, SERIES))

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
  'xtick.labelsize': 7,
  'ytick.labelsize': 7,
  'axes.labelsize': 8,
  'axes.titlesize': 8,
  'legend.fontsize': 8,
  'legend.frameon': False,
  'grid.color': '#e8e6e1',
  'axes.grid': True,
  'grid.linewidth': 0.6,
})


def load(tree, partial, exclude=()):
  cells = {}
  for path in glob.glob(os.path.join(tree, '*', '*', 'results.json')):
    parts = path.split(os.sep)
    seed, arm = parts[-3], parts[-2]
    if arm not in ARMS:
      continue
    with open(path) as handle:
      payload = json.load(handle)
    rows = payload['results']
    if f'{seed}/{arm}' in exclude:
      continue
    if len(rows) == 0 or not (payload.get('completed') or partial):
      continue
    calls = np.cumsum([row['spent'] for row in rows])
    best = np.minimum.accumulate([row['loss'] for row in rows])
    cells[(seed, arm)] = (calls, best)
  return cells


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('tree')
  parser.add_argument('--out', default=None)
  parser.add_argument('--name', default='per_seed.png')
  parser.add_argument('--partial', action='store_true')
  parser.add_argument(
    '--exclude', nargs='*', default=(),
    help='cells to drop outright, as seed/arm; a CAPPED cell carries completed: false and would otherwise be '
    'plotted by --partial as though it were still running'
  )
  parser.add_argument('--paired', action='store_true', help='only seeds complete in EVERY arm')
  arguments = parser.parse_args()

  cells = load(arguments.tree, arguments.partial, exclude=set(arguments.exclude))
  seeds = sorted({seed for seed, _ in cells}, key=int)
  if arguments.paired:
    seeds = [s for s in seeds if all((s, a) in cells for a in ARMS)]
  if len(seeds) == 0:
    raise SystemExit('no seeds to plot')

  columns = min(4, len(seeds))
  rows_n = (len(seeds) + columns - 1) // columns
  figure, axes = plt.subplots(rows_n, columns, figsize=(4.0 * columns, 3.2 * rows_n), squeeze=False)
  for index, seed in enumerate(seeds):
    axis = axes[index // columns][index % columns]
    finals = {}
    for arm in ARMS:
      if (seed, arm) not in cells:
        continue
      calls, best = cells[(seed, arm)]
      axis.step(calls, best, where='post', color=ARM_COLOUR[arm], linewidth=1.8, label=arm)
      finals[arm] = best[-1]
    order = sorted(finals, key=finals.get)
    axis.set_title(f'{seed}   ' + ' < '.join(a[:4] for a in order), color=INK, loc='left')
    axis.set_xlabel('cumulative detector calls')
    axis.set_ylabel('best loss so far')
  for index in range(len(seeds), rows_n * columns):
    axes[index // columns][index % columns].axis('off')
  handles = [plt.Line2D([], [], color=ARM_COLOUR[a], linewidth=2, label=a) for a in ARMS]
  figure.legend(handles=handles, loc='upper right', ncol=4)
  figure.suptitle(
    f'{os.path.basename(arguments.tree.rstrip("/"))} — best-so-far per seed '
    f'({len(seeds)} seeds)', x=0.01, ha='left', fontsize=10, color=INK
  )
  figure.tight_layout(rect=(0, 0, 1, 0.95))
  out = arguments.out or arguments.tree
  os.makedirs(out, exist_ok=True)
  target = os.path.join(out, arguments.name)
  figure.savefig(target, dpi=140)
  print(f'wrote {target}')


if __name__ == '__main__':
  main()
