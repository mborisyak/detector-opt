"""Gap-to-floor comparison of the `linear` BO arms at MATCHED design count.

Reads the `results.json` of finished runs, scores every proposed design against the closed-form
`LinearDetector.bayes_risk` (recomputed here in numpy from the same formula, checked against the
documented 0.004975 / 0.501 at d = 1), and reports, per arm and per seed, the BEST TRUE risk among
the first k designs. Comparing true risk removes the reported-loss bias entirely, and matching k
removes the selection advantage `meta` gets from exiting at smaller windows and scoring more designs.

Also reports the per-design cost (`spent`) distribution per arm, which is what the design count and
therefore the whole budget question turns on.
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


def load_run(path):
  with open(path) as handle:
    payload = json.load(handle)
  return payload


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True)
  parser.add_argument('--dimensions', type=int, required=True)
  parser.add_argument('--noise', type=float, required=True)
  parser.add_argument('--cuts', type=int, nargs='+', default=[5, 10, 20, 40, 80])
  args = parser.parse_args()

  runs = collections.defaultdict(dict)
  for dirpath, _, filenames in os.walk(args.root):
    if 'results.json' not in filenames:
      continue
    arm = os.path.basename(dirpath)
    seed = os.path.basename(os.path.dirname(dirpath))
    runs[seed][arm] = os.path.join(dirpath, 'results.json')

  table = {}
  for seed in sorted(runs):
    for arm, path in sorted(runs[seed].items()):
      payload = load_run(path)
      results = payload['results']
      true_risk = np.array([bayes_risk(r['design'], args.dimensions, args.noise) for r in results])
      reported = np.array([r['loss'] for r in results])
      spent = np.array([r['spent'] for r in results], np.float64)
      table[(seed, arm)] = dict(true_risk=true_risk, reported=reported, spent=spent,
                                n=len(results), calls=payload.get('detector_calls_used'))

  arms = sorted({arm for _, arm in table})
  seeds = sorted({seed for seed, _ in table})
  counts = {key: value['n'] for key, value in table.items()}
  print(f'root={args.root}  d={args.dimensions}  noise={args.noise}')
  print(f'arms={arms}  seeds={seeds}')
  print()
  print('DESIGN COUNT and MEDIAN CALLS PER DESIGN')
  header = 'seed'.ljust(12) + ''.join(a.ljust(26) for a in arms)
  print(header)
  for seed in seeds:
    row = seed.ljust(12)
    for arm in arms:
      value = table.get((seed, arm))
      if value is None:
        row += '-'.ljust(26)
      else:
        row += f'{value["n"]:4d} designs {np.median(value["spent"]):8.0f}'.ljust(26)
    print(row)
  print()

  print('BEST TRUE bayes_risk AMONG FIRST k DESIGNS (matched k)')
  for k in args.cuts:
    usable = [s for s in seeds if all(counts.get((s, a), 0) >= k for a in arms)]
    if len(usable) == 0:
      continue
    print(f'  k={k:4d}   seeds={len(usable)}')
    per_arm = {}
    for arm in arms:
      values = np.array([table[(s, arm)]['true_risk'][:k].min() for s in usable])
      per_arm[arm] = values
      print(f'      {arm.ljust(14)} median {np.median(values):.6f}  mean {values.mean():.6f}  '
            f'min {values.min():.6f}  max {values.max():.6f}')
    if 'meta' in per_arm and 'from_scratch' in per_arm:
      delta = per_arm['meta'] - per_arm['from_scratch']
      wins = int(np.sum(delta < 0))
      print(f'      meta - from_scratch: median {np.median(delta):+.6f}  '
            f'meta better on {wins}/{len(usable)} seeds')
    print()

  print('REPORTED LOSS BIAS: mean(reported - true) over all designs')
  for arm in arms:
    biases, spends, ratios = [], [], []
    for seed in seeds:
      value = table.get((seed, arm))
      if value is None:
        continue
      biases.append(np.mean(value['reported'] - value['true_risk']))
      spends.append(np.median(value['spent']))
      ratios.append(np.mean((value['reported'] - value['true_risk']) / value['true_risk']))
    print(f'  {arm.ljust(14)} bias {np.mean(biases):+.6f}  relative {np.mean(ratios):+.4f}  '
          f'median spent {np.median(spends):8.0f}')


if __name__ == '__main__':
  main()
