"""Per-seed read-out: one subplot per (tree, seed), one best-so-far curve per arm, plus every banked design.

    python scripts/plot_per_seed.py output/plots/extremes-2026-09-07/trees/select-optimal \
        output/plots/extremes-2026-09-07/trees/test-reported --out output/plots/extremes-2026-09-07 --name per-seed

The companion of `plot_median.py` for the question it cannot answer: WHICH seeds carry an aggregate. A
tree is the same ``<seed>/<arm>/results.json`` layout, cells are complete trajectories only, and the
curve is the same best-so-far against cumulative detector calls, carried to the configured budget.
Each subplot is titled ``<tree name> <seed>``; faint dots are the individual designs' reported losses,
so a cell that found a low design once and never again reads differently from one that kept
improving. Colours, palette and the budget check are imported from `plot_median.py`, so an arm has
the same colour in both figures. The y axis is shared across subplots.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_median import load, budget_of, colour_of, INK_2


def designs_of(tree, seed, arm):
  """``(cumulative calls, loss)`` of every banked design of one cell."""
  with open(os.path.join(tree, seed, arm, 'results.json')) as handle:
    rows = json.load(handle)['results']
  return np.cumsum([row['spent'] for row in rows]), np.asarray([row['loss'] for row in rows])


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('trees', nargs='+')
  parser.add_argument('--out', default=None, help='directory for the png; defaults beside the first tree')
  parser.add_argument('--name', default='per-seed')
  parser.add_argument('--cols', type=int, default=5)
  args = parser.parse_args()

  panels = []
  for tree in args.trees:
    label = os.path.basename(os.path.normpath(tree))
    budget = budget_of(tree)
    arms = load(tree)
    for seed in sorted({seed for cells in arms.values() for seed in cells}, key=int):
      panels.append((tree, label, seed, budget, {arm: cells[seed] for arm, cells in arms.items() if seed in cells}))
  if len(panels) == 0:
    raise SystemExit('no complete cells in ' + ', '.join(args.trees))

  cols = min(args.cols, len(panels))
  rows = int(np.ceil(len(panels) / cols))
  figure, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), sharey=True, squeeze=False)
  ordered_arms = sorted({arm for *_, cells in panels for arm in cells})
  for axis, (tree, label, seed, budget, cells) in zip(axes.flat, panels):
    for slot, arm in enumerate(ordered_arms):
      if arm not in cells:
        continue
      x, y = cells[arm]
      colour = colour_of(arm, slot)
      if budget is not None and x[-1] < budget:
        x, y = np.append(x, budget), np.append(y, y[-1])
      axis.step(x, y, where='post', color=colour, linewidth=1.6, label=f'{arm} ({y[-1]:.3f})')
      dx, dy = designs_of(tree, seed, arm)
      axis.plot(dx, dy, linestyle='none', marker='o', markersize=2.4, color=colour, alpha=0.35)
    axis.set_title(f'{label}  {seed}', fontsize=9, loc='left')
    axis.legend(fontsize=6.5, frameon=False, loc='upper right')
    axis.grid(True, linewidth=0.4)
    axis.tick_params(labelsize=7)
  for axis in axes.flat[len(panels):]:
    axis.set_visible(False)
  for axis in axes[-1]:
    axis.set_xlabel('cumulative detector calls', fontsize=8, color=INK_2)
  for axis in axes[:, 0]:
    axis.set_ylabel('loss (line: best so far; dots: designs)', fontsize=8, color=INK_2)
  figure.suptitle(
    'Per-seed trajectories, one subplot per seed; legend holds the best loss at the budget', fontsize=10, x=0.01, ha='left'
  )
  figure.tight_layout(rect=(0, 0, 1, 0.97))
  out_dir = args.out if args.out is not None else os.path.dirname(os.path.normpath(args.trees[0]))
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f'{args.name}.png')
  figure.savefig(path, dpi=160)
  print(f'wrote {path} ({len(panels)} panels)')


if __name__ == '__main__':
  main()
