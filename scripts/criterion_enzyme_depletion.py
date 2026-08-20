#!/usr/bin/env python3
"""The doubling criterion for `enzyme_depletion`, pooled over `scripts/enzyme_depletion_bo.py` reports.

    python scripts/criterion_enzyme_depletion.py output/mm-retune/bo-nominated-*.json

    P( loss@2n designs < loss@n designs ),  THE TWO FROM DIFFERENT, INDEPENDENT RUNS.

Read off ONE trajectory the comparison is monotone by construction, so the within-run probability is
1 whatever the task does. Between independent runs it is 0.5 exactly when doubling the search buys
nothing, which is what a saturated task fails to beat. It is the two-sample AUC (Mann-Whitney) of
`loss@2n` against `loss@n`, over all R(R-1) ORDERED pairs of distinct runs, ties at 0.5.

The standard error is a delete-one JACKKNIFE over RUNS -- one run is one draw -- because the pairs
share runs and `sqrt(P(1-P)/R)` is the error bar of a proportion of R independent trials, which this
is not. The naive figure is printed beside it so the difference stays visible.

AGAINST THE NULL, ALWAYS: the random-search arm shares seeds with BO, so the excess is jackknifed
PAIRED over the same delete-one blocks rather than by differencing two independent error bars.

THE QUANTITY IS `reevaluated`, the incumbent DESIGN re-scored on an independent event block, never
`observed` (whose running minimum carries the winner's curse, and whose bias grows with the number of
designs -- the very axis the criterion varies). Seeds are pooled by `(measurement_noise, seed)`, so a
seed appearing in two report blocks is counted once.

This is the single-substrate twin of `scripts/criterion_enzyme_depletion_bi.py`; the payload keys
differ (`concentration_bounds`, not the two bi-substrate ones) and there is no plot.
"""
import argparse
import glob
import json

import numpy as np


def load(paths):
  """`{noise: {arm: (n_seeds, n_checkpoints) re-evaluated incumbent loss}}`, the checkpoints, settings."""
  by_noise, checkpoints, settings = {}, None, {}
  for path in sorted({p for pattern in paths for p in glob.glob(pattern)}):
    with open(path) as handle:
      report = json.load(handle)
    noise = float(report['measurement_noise'])
    if checkpoints is None:
      checkpoints = [int(c) for c in report['checkpoints']]
    elif [int(c) for c in report['checkpoints']] != checkpoints:
      raise ValueError(f'{path} has checkpoints {report["checkpoints"]}, expected {checkpoints}')
    seeds = int(report['seed_start']) + np.arange(int(report['n_seeds']))
    bucket = by_noise.setdefault(noise, {})
    settings.setdefault(
      noise, {
        'n_experiments': report['n_experiments'],
        'duration': report['duration'],
        'concentration_bounds': report['concentration_bounds'],
        'michaelis_bounds': report['michaelis_bounds'],
        'velocity_bounds': report['velocity_bounds'],
        'kernel': report['kernel'],
        'n_events': report['n_events'],
        'no_information_loss': report['no_information_loss'],
      }
    )
    for arm, block in report['arms'].items():
      values = np.stack([np.asarray(block['reevaluated'][str(c)], float) for c in checkpoints], axis=-1)
      store = bucket.setdefault(arm, {})
      for seed, row in zip(seeds, values):
        store[int(seed)] = row
  out = {}
  for noise, arms in by_noise.items():
    shared = sorted(set.intersection(*[set(v) for v in arms.values()]))
    out[noise] = {arm: np.stack([arms[arm][s] for s in shared]) for arm in arms}
  return out, checkpoints, settings


def auc(at_2n, at_n):
  """`P(loss@2n < loss@n)` over ORDERED pairs of DISTINCT runs, ties at 0.5."""
  n = len(at_2n)
  comparison = (at_2n[:, None] < at_n[None, :]).astype(float) + 0.5 * (at_2n[:, None] == at_n[None, :])
  np.fill_diagonal(comparison, 0.0)
  return float(comparison.sum() / (n * (n - 1)))


def jackknife(statistic, n):
  """`(value, standard error)` of `statistic(keep_mask)` by delete-one over runs."""
  everything = np.ones(n, bool)
  full = statistic(everything)
  partial = np.empty(n)
  for k in range(n):
    keep = everything.copy()
    keep[k] = False
    partial[k] = statistic(keep)
  return full, float(np.sqrt((n - 1) / n * np.sum((partial - partial.mean())**2)))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('reports', nargs='+')
  arguments = parser.parse_args()

  curves, checkpoints, settings = load(arguments.reports)
  pairs = [(i, j) for i in range(len(checkpoints)) for j in range(len(checkpoints)) if checkpoints[j] == 2 * checkpoints[i]]
  if len(pairs) == 0:
    raise SystemExit(f'no n -> 2n pair among the checkpoints {checkpoints}')
  for noise in sorted(curves):
    setting = settings[noise]
    guess = float(setting['no_information_loss'])
    print(
      f'noise {noise:g} mM, m={setting["n_experiments"]}, box {setting["concentration_bounds"]} mM, '
      f'K {setting["michaelis_bounds"]} mM, q {setting["velocity_bounds"]} mM/s, '
      f'T {setting["duration"] / 3600:.2f} h, {setting["n_events"]} events, kernel {setting["kernel"]}'
    )
    for lower, upper in pairs:
      arms = {}
      for arm in sorted(curves[noise]):
        values = curves[noise][arm]
        n_seeds = values.shape[0]
        point, error = jackknife(lambda keep, v=values: auc(v[keep, upper], v[keep, lower]), n_seeds)
        arms[arm] = point
        naive = float(np.sqrt(point * (1.0 - point) / n_seeds))
        gain = float(np.mean(values[:, lower]) - np.mean(values[:, upper])) / guess
        gain_error = float(np.std(values[:, lower] - values[:, upper]) / np.sqrt(n_seeds) / guess)
        print(
          f'  {checkpoints[lower]:2d}->{checkpoints[upper]:<3d} {arm:7s} '
          f'P = {point:.4f} +- {error:.4f} (jackknife; naive {naive:.4f}), {n_seeds} runs, '
          f'E[loss@{checkpoints[lower]}] = {np.mean(values[:, lower]):.4f}, '
          f'E[loss@{checkpoints[upper]}] = {np.mean(values[:, upper]):.4f}, '
          f'bonus = {gain:.4f} +- {gain_error:.4f}'
        )
      if 'bo' in arms and 'random' in arms:

        def difference(keep, block=curves[noise]):
          return (
            auc(block['bo'][keep, upper], block['bo'][keep, lower]) -
            auc(block['random'][keep, upper], block['random'][keep, lower])
          )

        delta, delta_error = jackknife(difference, curves[noise]['bo'].shape[0])
        print(
          f'  {checkpoints[lower]:2d}->{checkpoints[upper]:<3d} excess over null: '
          f'{delta:+.4f} +- {delta_error:.4f} (paired jackknife)'
        )


if __name__ == '__main__':
  main()
