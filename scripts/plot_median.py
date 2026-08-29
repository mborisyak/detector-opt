#!/usr/bin/env python3
"""THE median plot. Median best-so-far against cumulative detector calls, across seeds, per arm.

    python scripts/plot_median.py output/linear-median-revealed output/linear-median-withheld
    python scripts/plot_median.py output/ship-prec/prec1e2 output/ship-prec/prec5e3 --out <dir>

One panel per TREE, one line per arm. A tree is any directory of ``<seed>/<arm>/results.json``.

This is the standard read-out for every campaign in this repository. Use it rather than writing
another one.

TWO RULES IT ENFORCES, both learned the hard way:

* ONE SEED CANNOT SUPPORT A BEST-SO-FAR CLAIM. BO proposes design ``n_init`` from a GP fitted to the
  observed losses, so a float-level perturbation moves the proposal and the trajectories diverge from
  there. Two `linear` cells with identical configs banked 20 and 15 designs with bests 0.0787 and
  0.1205 -- a larger gap than most arm effects. An arm drawn from a single seed is labelled as such.
* THE MEDIAN IS DRAWN ONLY WHILE EVERY SEED IS STILL ALIVE. A median over a CHANGING set of seeds is
  not monotone: when the lowest cell ends, it jumps UP, which on a best-so-far axis reads as the
  optimiser getting worse and is an artefact.

--mean adds a SECOND ROW of panels holding the mean with standard-error bars, and removes the
min-max band from both rows: the request there is the sampling uncertainty of the centre, which a
min-max envelope over 10 seeds does not show and visually swamps.

Every curve is direct-labelled at its right end because two of the four arm colours fall below 3:1
contrast against the chart surface, so identity must not rest on colour alone.

EVERY TRAJECTORY IS CARRIED TO THE CONFIGURED BUDGET, by appending one point holding its last
best-so-far. This is ON BY DEFAULT (`--no-extend` turns it off) because best-so-far at budget B is
the minimum over designs with spend <= B, so the value at the budget is DEFINED for every cell -- a
cell whose final design never completed simply holds its last value. Leaving it off clips each arm at
its shortest seed and silently drops the point at the budget, which is the point being compared.

``--paired`` keeps only the seeds completed in EVERY arm. Use it on a campaign that is still running:
cells finish at different times, so one arm can hold a seed the others do not, and comparing curves
built on different seed sets compares different experiments. It is OFF by default so the unfiltered
read-out is unchanged.
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
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']

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


def budget_of(tree, exclude=()):
  """The configured ``training.budget`` the cells of ``tree`` were run at.

  ``exclude`` is honoured here as well as in :func:`load`, because a tree can legitimately hold cells
  from two runs: a budget EXTENSION leaves the not-yet-restarted cells carrying the previous run's
  `results.json`, so the tree mixes budgets until every cell has been rerun. Excluding those cells has
  to silence this guard too, or the excluded cells still decide whether the plot may be drawn.
  """
  budgets = set()
  for path in glob.glob(os.path.join(tree, '*', '*', 'results.json')):
    seed, arm = path.split(os.sep)[-3:-1]
    if f'{seed}/{arm}' in exclude:
      continue
    # A DOT IN AN ARM NAME MARKS A SIDELINED COPY, NOT AN ARM. `ship-addr-prec1e2` holds
    # `750143450/from_scratch.stopped-10designs`, a stopped pre-refactor run; plotting it added a
    # sixth "arm" to the 1M figure in almost the same colour as `meta`. Canonical arm names never
    # contain a dot, so this is a structural guard rather than a name to remember.
    if '.' in arm:
      continue
    with open(path) as handle:
      payload = json.load(handle)
    budgets.add(int(payload['config']['training']['budget']))
  if len(budgets) > 1:
    raise ValueError(f'{tree} mixes budgets {sorted(budgets)}; a shared x axis would be meaningless')
  return budgets.pop() if len(budgets) == 1 else None


def load(tree, partial=False, exclude=()):
  """``{arm: {seed: (cumulative calls, best so far)}}``. ``partial`` also admits cells still running.

  ``exclude`` holds ``seed/arm`` cells to drop outright. A CAPPED cell -- one whose design exhausted
  ``iteration_limit`` without meeting the precision -- carries ``completed: false`` and is therefore
  indistinguishable here from a cell that is merely still running, so ``partial`` would plot its
  trajectory as if it were in flight. A capped cell is a FAILURE and is never a result, so it has to
  be named.
  """
  out = {}
  for path in sorted(glob.glob(os.path.join(tree, '*', '*', 'results.json'))):
    seed, arm = path.split(os.sep)[-3:-1]
    if f'{seed}/{arm}' in exclude:
      continue
    if '.' in arm:
      continue  # sidelined copy, not an arm -- see budget_of
    with open(path) as handle:
      payload = json.load(handle)
    rows = payload['results']
    # COMPLETE TRAJECTORIES ONLY. A cell still running has spent part of its budget, so its
    # best-so-far is an upper bound that will keep falling; mixing it into a median makes the median
    # a statement about which cells happened to get a GPU first.
    if len(rows) == 0 or not (payload.get('completed') or partial):
      continue
    out.setdefault(arm, {}
                   )[seed] = (np.cumsum([row['spent'] for row in rows]), np.minimum.accumulate([row['loss'] for row in rows]))
  return out


ARM_COLOUR = {'from_scratch': SERIES[0], 'continue': SERIES[1], 'closest': SERIES[2], 'meta': SERIES[3]}

YLABEL = {'median': 'median best loss so far', 'mean': 'mean best loss so far  (bars = standard error)'}


def colour_of(arm, slot):
  """Colour follows the ARM, not its rank, so a tree missing an arm does not repaint the others."""
  return ARM_COLOUR.get(arm, SERIES[slot % len(SERIES)])


def place_labels(axis, labels, x):
  """Label each curve at its right end, pushed apart where finals nearly coincide.

  Required rather than decorative: two of the four series sit below 3:1 contrast against the chart
  surface, so identity must not rest on line colour alone."""
  if len(labels) == 0:
    return
  bottom, top = axis.get_ylim()
  gap = 0.05 * (top - bottom)
  labels = sorted(labels, key=lambda item: item[1])
  for index in range(1, len(labels)):
    arm, height = labels[index]
    if height - labels[index - 1][1] < gap:
      labels[index] = (arm, labels[index - 1][1] + gap)
  left, right = axis.get_xlim()
  axis.set_xlim(left, right + 0.09 * (right - left))
  for arm, height in labels:
    axis.annotate(
      arm, xy=(x, height), xytext=(6, 0), textcoords='offset points', va='center', ha='left', fontsize=7.5, color=INK_2
    )


def panel(axis, tree, title, statistic='median', spread=True, paired=False, extend=False, partial=False, exclude=()):
  data = load(tree, partial=partial, exclude=exclude)
  if paired and len(data) > 0:
    # ARMS MUST BE READ ON THE SAME SEEDS. Cells finish at different times, so mid-campaign one arm
    # can hold a seed the others do not -- and a curve over a different seed set is a different
    # experiment, not a comparable one. This is the cross-ARM twin of the rule the median already
    # obeys along x.
    shared = set.intersection(*[set(seeds) for seeds in data.values()])
    data = {arm: {seed: curve for seed, curve in seeds.items() if seed in shared} for arm, seeds in data.items()}
    data = {arm: seeds for arm, seeds in data.items() if len(seeds) > 0}
  if len(data) == 0:
    axis.set_title(f'{title}\n(no results yet)', loc='left', pad=8)
    return
  longest = max(calls[-1] for arm in data for calls, _ in data[arm].values())
  shortest = min(calls[0] for arm in data for calls, _ in data[arm].values())
  # Best-so-far at budget B is the minimum over designs with spend <= B, so a cell that banked nothing
  # after its last design holds that value to the end of its budget. Without this the curve is clipped
  # at the arm's shortest seed and valid points are dropped.
  # ONE duplicate per TRAJECTORY, at the CONFIGURED budget, holding that trajectory's last
  # best-so-far. Cells stop banking rows short of the budget because the final design they paid for
  # never completed, and each cell stops at a different point; without this every arm is clipped at
  # its shortest seed.
  if extend:
    full = budget_of(tree, exclude=exclude)
    if full is not None:
      longest = max(longest, float(full))
      data = {
        arm: {
          seed: ((np.append(calls, full), np.append(best, best[-1])) if calls[-1] < full else (calls, best))
          for seed, (calls, best) in seeds.items()
        }
        for arm, seeds in data.items()
      }
  grid = np.linspace(shortest, longest, 300)
  labels, edge = [], shortest
  for slot, arm in enumerate(sorted(data)):
    stack = []
    for calls, best in data[arm].values():
      curve = np.full_like(grid, np.nan)
      live = grid <= calls[-1]
      curve[live] = np.interp(grid[live], calls, best)
      stack.append(curve)
    stack = np.vstack(stack)
    drawn = np.sum(~np.isnan(stack), axis=0) == stack.shape[0]
    if not np.any(drawn):
      continue
    usable = stack[:, drawn]
    colour = colour_of(arm, slot)
    if spread and stack.shape[0] > 1:
      axis.fill_between(grid[drawn], usable.min(axis=0), usable.max(axis=0), color=colour, alpha=0.12, linewidth=0)
    centre = np.mean(usable, axis=0) if statistic == 'mean' else np.median(usable, axis=0)
    axis.plot(grid[drawn], centre, color=colour, label=f'{arm}  ({stack.shape[0]})')
    if statistic == 'mean' and stack.shape[0] > 1:
      error = np.std(usable, axis=0, ddof=1) / np.sqrt(stack.shape[0])
      mark = np.unique(np.linspace(0, centre.size - 1, 12).astype(int))
      axis.errorbar(
        grid[drawn][mark], centre[mark], yerr=error[mark], fmt='none', ecolor=colour, elinewidth=1.2, capsize=2.5, capthick=1.2
      )
    labels.append((arm, float(centre[-1])))
    edge = max(edge, longest if extend else float(grid[drawn][-1]))
  axis.set_xlabel('cumulative detector calls')
  axis.set_title(f'{title}   (arm, seeds)', loc='left', pad=8)
  axis.grid(True, alpha=0.9)
  axis.set_axisbelow(True)
  axis.legend(loc='upper right')
  for side in ('top', 'right'):
    axis.spines[side].set_visible(False)
  place_labels(axis, labels, edge)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('trees', nargs='+')
  parser.add_argument('--out', default=None, help='directory for the png; defaults beside the first tree')
  parser.add_argument('--name', default='median.png')
  parser.add_argument(
    '--mean', action='store_true',
    help='add a second row of panels holding the MEAN with standard-error bars, and drop the min-max band from both'
  )
  parser.add_argument(
    '--partial', action='store_true',
    help='also plot cells that are still RUNNING; their best-so-far is an upper bound that will keep falling'
  )
  parser.add_argument(
    '--paired', action='store_true', help='keep only the seeds completed in EVERY arm, so the arms are read on one seed set'
  )
  parser.add_argument(
    '--exclude', nargs='*', default=(),
    help='cells to drop outright, as seed/arm; use for CAPPED cells, which carry completed: false and would '
    'otherwise be plotted by --partial as though they were still running'
  )
  parser.add_argument(
    '--no-extend', dest='extend', action='store_false',
    help='do NOT carry each trajectory to the configured budget; the curve then stops at its last banked design'
  )
  parser.set_defaults(extend=True)
  arguments = parser.parse_args()

  rows = ('median', 'mean') if arguments.mean else ('median', )
  spread = not arguments.mean
  size = (5.7 * len(arguments.trees), 4.4 * len(rows))
  figure, axes = plt.subplots(len(rows), len(arguments.trees), figsize=size, sharey=True, squeeze=False)
  for row, statistic in enumerate(rows):
    for axis, tree in zip(axes[row], arguments.trees):
      panel(
        axis, tree, os.path.basename(tree.rstrip('/')), statistic=statistic, spread=spread, paired=arguments.paired,
        extend=arguments.extend, partial=arguments.partial, exclude=set(arguments.exclude)
      )
    label = YLABEL[statistic] + ('  (band = min-max across seeds)' if spread else '')
    axes[row][0].set_ylabel(label)
  heading = 'Median (top) and mean with standard error (bottom), across seeds' if arguments.mean else 'Median across seeds'
  figure.suptitle(heading, x=0.01, ha='left', fontsize=10, color=INK)
  figure.tight_layout(rect=(0, 0, 1, 0.93 if len(rows) == 1 else 0.96))
  out = arguments.out or os.path.join(arguments.trees[0], 'plots')
  os.makedirs(out, exist_ok=True)
  target = os.path.join(out, arguments.name)
  figure.savefig(target, dpi=160)
  print(f'wrote {target}')


if __name__ == '__main__':
  main()
