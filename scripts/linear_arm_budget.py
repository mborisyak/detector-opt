"""Best TRUE risk as a function of cumulative detector calls, per arm, on `linear`.

The matched-design-count comparison answers "which arm proposes better designs"; it does not answer
"which arm should a campaign run", because the arms buy different numbers of designs with the same
budget. This reads the same `results.json` files and reports the best `bayes_risk` reached by the
time a given number of detector calls has been spent -- budget-matched rather than count-matched --
alongside the count-matched figure, so the two can be read against each other.
"""

import argparse
import collections
import json
import os

import numpy as np


def bayes_risk(flat, n_dimensions, noise):
  flat = np.asarray(flat, np.float64).reshape(-1)
  n_probes = flat.size // n_dimensions
  probe = flat.reshape(n_dimensions, n_probes).T
  rows = np.concatenate([probe, np.ones((n_probes, 1), np.float64)], axis=-1)
  precision = rows.T @ rows / noise**2 + np.eye(n_dimensions + 1)
  return float(np.trace(np.linalg.inv(precision)) / (n_dimensions + 1))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True)
  parser.add_argument('--dimensions', type=int, required=True)
  parser.add_argument('--noise', type=float, required=True)
  parser.add_argument('--budgets', type=float, nargs='+', required=True)
  args = parser.parse_args()

  runs = collections.defaultdict(dict)
  for dirpath, _, filenames in os.walk(args.root):
    if 'results.json' not in filenames:
      continue
    arm = os.path.basename(dirpath)
    seed = os.path.basename(os.path.dirname(dirpath))
    with open(os.path.join(dirpath, 'results.json')) as handle:
      results = json.load(handle)['results']
    true = np.array([bayes_risk(r['design'], args.dimensions, args.noise) for r in results])
    cumulative = np.cumsum([r['spent'] for r in results])
    runs[seed][arm] = (true, cumulative)

  arms = sorted({a for seed in runs for a in runs[seed]})
  print(f'root={args.root}  d={args.dimensions}  noise={args.noise}')
  print('BEST TRUE bayes_risk BY CUMULATIVE DETECTOR CALLS (budget-matched)')
  for budget in args.budgets:
    usable = [s for s in runs if all(a in runs[s] and runs[s][a][1][-1] >= budget for a in arms)]
    if len(usable) == 0:
      print(f'  budget={budget:.0f}: no seed reached it in every arm')
      continue
    print(f'  budget={budget:9.0f}   seeds={len(usable)}')
    per_arm = {}
    for arm in arms:
      values, designs = [], []
      for seed in usable:
        true, cumulative = runs[seed][arm]
        reached = int(np.searchsorted(cumulative, budget, side='right'))
        values.append(true[:max(reached, 1)].min())
        designs.append(reached)
      per_arm[arm] = np.array(values)
      print(f'      {arm.ljust(14)} median {np.median(values):.6f}  mean {np.mean(values):.6f}  '
            f'median designs {np.median(designs):5.1f}')
    if 'meta' in per_arm and 'from_scratch' in per_arm:
      delta = per_arm['meta'] - per_arm['from_scratch']
      print(f'      meta - from_scratch: median {np.median(delta):+.6f}  '
            f'meta better on {int(np.sum(delta < 0))}/{len(usable)} seeds')
    print()


if __name__ == '__main__':
  main()
