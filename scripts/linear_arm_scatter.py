"""Reported-loss FIDELITY per arm on `linear`, against the closed-form floor.

A constant offset in the reported loss is absorbed by the GP's constant mean and costs BO nothing;
what misleads the surrogate is SCATTER -- the part of `reported - bayes_risk` that varies from design
to design. This script separates the two: it fits `reported = a * true + b` per run and reports the
slope, the offset and the residual spread, plus the Spearman rank agreement between the reported
ordering and the true ordering, which is what the acquisition function actually consumes.
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


def spearman(a, b):
  ra = np.argsort(np.argsort(a)).astype(np.float64)
  rb = np.argsort(np.argsort(b)).astype(np.float64)
  ra -= ra.mean()
  rb -= rb.mean()
  denominator = np.sqrt(np.sum(ra**2) * np.sum(rb**2))
  return float(np.sum(ra * rb) / denominator) if denominator > 0 else float('nan')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True)
  parser.add_argument('--dimensions', type=int, required=True)
  parser.add_argument('--noise', type=float, required=True)
  args = parser.parse_args()

  per_arm = collections.defaultdict(list)
  for dirpath, _, filenames in os.walk(args.root):
    if 'results.json' not in filenames:
      continue
    arm = os.path.basename(dirpath)
    with open(os.path.join(dirpath, 'results.json')) as handle:
      results = json.load(handle)['results']
    if len(results) < 6:
      continue
    true = np.array([bayes_risk(r['design'], args.dimensions, args.noise) for r in results])
    reported = np.array([r['loss'] for r in results])
    reported_std = np.array([r['loss_std'] for r in results])
    slope, offset = np.polyfit(true, reported, 1)
    residual = reported - (slope * true + offset)
    per_arm[arm].append(dict(slope=slope, offset=offset, residual_sd=residual.std(ddof=2),
                             rank=spearman(true, reported), n=len(results),
                             claimed_sd=float(np.median(reported_std)),
                             relative_residual=float(np.std((reported - true) / true, ddof=1))))

  print(f'root={args.root}  d={args.dimensions}  noise={args.noise}')
  print(f'{"arm".ljust(14)}{"runs":>5}{"slope":>9}{"offset":>10}{"resid sd":>11}'
        f'{"claimed sd":>12}{"rel resid":>11}{"rank corr":>11}')
  for arm in sorted(per_arm):
    rows = per_arm[arm]
    print(f'{arm.ljust(14)}{len(rows):5d}'
          f'{np.median([r["slope"] for r in rows]):9.4f}'
          f'{np.median([r["offset"] for r in rows]):10.5f}'
          f'{np.median([r["residual_sd"] for r in rows]):11.5f}'
          f'{np.median([r["claimed_sd"] for r in rows]):12.5f}'
          f'{np.median([r["relative_residual"] for r in rows]):11.4f}'
          f'{np.median([r["rank"] for r in rows]):11.4f}')


if __name__ == '__main__':
  main()
