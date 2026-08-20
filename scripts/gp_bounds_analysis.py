#!/usr/bin/env python3
"""Summarise the two probes that decide the GP's prior bounds.

`fits` reads `scripts/probe_gp_bounds.py --mode fits` output and counts, per iteration and over the
run, how often the fitted amplitude and lengthscales sit ON a prior bound and at which end, plus how
far the fit moves between consecutive iterations. Pinning is a hyperparameter within `--tolerance`
of a bound in LOG units, which is what L-BFGS-B returns when it has run into one.

`criterion` reads `scripts/enzyme_depletion_bo.py` output and computes the pre-registered headline:
`P(loss @ 2n < loss @ n)` as the all-pairs U-statistic between the two checkpoints' distributions
over INDEPENDENT seeds (the diagonal is dropped, so no pair shares a run), with a delete-one
jackknife standard error over seeds. Two files compared with `--against` are assumed to share their
seeds, so the difference gets a jackknife SE of its own rather than the sum of two independent ones.
"""
import argparse
import json

import numpy as np


def u_statistic(at_n, at_2n, exclude_diagonal=True):
  """`P(loss @ 2n < loss @ n)` over independent runs; ties count a half."""
  at_n, at_2n = np.asarray(at_n, dtype=float), np.asarray(at_2n, dtype=float)
  wins = (at_2n[:, None] < at_n[None, :]).astype(float) + 0.5 * (at_2n[:, None] == at_n[None, :])
  if not exclude_diagonal:
    return float(wins.mean())
  size = wins.shape[0]
  return float((wins.sum() - np.trace(wins)) / (size * (size - 1)))


def jackknife(function, size):
  """Delete-one estimate and standard error of `function(keep_mask_indices)` over `size` seeds."""
  full = function(np.arange(size))
  values = np.asarray([function(np.delete(np.arange(size), k)) for k in range(size)])
  spread = float(np.sqrt((size - 1) / size * np.sum((values - values.mean())**2)))
  return full, spread


def report_criterion(arguments):
  payload = json.load(open(arguments.file))
  first, last = str(payload['checkpoints'][0]), str(payload['checkpoints'][-1])
  other = json.load(open(arguments.against)) if arguments.against is not None else None
  print(
    f"{arguments.file}  m={payload['n_experiments']}  kernel={payload['kernel']}  "
    f"l={payload.get('log_lengthscale_prior_bounds')}  a={payload.get('log_amplitude_prior_bounds')}"
  )
  if other is not None:
    print(
      f"{arguments.against}  m={other['n_experiments']}  kernel={other['kernel']}  "
      f"l={other.get('log_lengthscale_prior_bounds')}  a={other.get('log_amplitude_prior_bounds')}"
    )
  for arm, data in payload['arms'].items():
    at_n = np.asarray(data['reevaluated'][first], dtype=float)
    at_2n = np.asarray(data['reevaluated'][last], dtype=float)
    size = at_n.size
    value, error = jackknife(lambda keep: u_statistic(at_n[keep], at_2n[keep]), size)
    print(
      f"  {arm:8s} n={size:4d}  loss@{first} = {at_n.mean():.5f} +- {at_n.std(ddof=1) / np.sqrt(size):.5f}"
      f"  loss@{last} = {at_2n.mean():.5f} +- {at_2n.std(ddof=1) / np.sqrt(size):.5f}"
      f"  P(@{last} < @{first}) = {value:.4f} +- {error:.4f}"
    )
    if other is None or arm not in other['arms']:
      continue
    other_n = np.asarray(other['arms'][arm]['reevaluated'][first], dtype=float)
    other_2n = np.asarray(other['arms'][arm]['reevaluated'][last], dtype=float)
    if other_n.size != size:
      print(f"    (not paired: {arguments.against} has {other_n.size} seeds)")
      continue
    delta, delta_error = jackknife(
      lambda keep: u_statistic(other_n[keep], other_2n[keep]) - u_statistic(at_n[keep], at_2n[keep]), size
    )
    paired = other_2n - at_2n
    print(
      f"    paired vs --against: dP = {delta:+.4f} +- {delta_error:.4f}"
      f"   d(loss@{last}) = {paired.mean():+.5f} +- {paired.std(ddof=1) / np.sqrt(size):.5f}"
      f"   better in {int((paired < 0).sum())}/{size} seeds"
    )


def group_fits(paths):
  """Chunks of one campaign share their detector, arm and prior box; keyed on those, they merge."""
  groups = {}
  for path in paths:
    payload = json.load(open(path))
    key = (
      payload['n_experiments'], payload['arm'], tuple(payload['length_scale_bounds']),
      tuple(payload['amplitude_squared_bounds'])
    )
    if key not in groups:
      groups[key] = {k: v for k, v in payload.items() if k != 'records'}
      groups[key]['records'] = []
      groups[key]['parts'] = []
    groups[key]['records'].extend(payload['records'])
    groups[key]['parts'].append(path)
  return groups


def report_fits(arguments):
  for payload in group_fits(arguments.files).values():
    path = ' '.join(payload['parts'])
    low, high = payload['length_scale_bounds']
    amplitude_low, amplitude_high = payload['amplitude_squared_bounds']
    tolerance = arguments.tolerance
    rows = []
    for record in payload['records']:
      for fit in record['fits']:
        rows.append((fit['n'], fit['amplitude_squared'], fit['length_scale'], fit['y_variance']))
    counts = np.asarray([row[0] for row in rows])
    amplitude = np.asarray([row[1] for row in rows])
    length = np.asarray([row[2] for row in rows], dtype=float)
    y_variance = np.asarray([row[3] for row in rows])
    at_low = np.log(length) - np.log(low) < tolerance
    at_high = np.log(high) - np.log(length) < tolerance
    amplitude_at_low = np.log(amplitude) - np.log(amplitude_low) < tolerance
    amplitude_at_high = np.log(amplitude_high) - np.log(amplitude) < tolerance
    print(
      f"{path}  m={payload['n_experiments']}  d={payload['design_dim']}  arm={payload['arm']}  "
      f"l in [{low:.4g}, {high:.4g}]  amplitude^2 in [{amplitude_low:.4g}, {amplitude_high:.4g}]"
    )
    print(f"  {len(payload['records'])} seeds, {len(rows)} fits, {length.shape[1]} lengthscales each")
    print(
      f"  lengthscale : at LOW {at_low.mean() * 100:5.1f}%   at HIGH {at_high.mean() * 100:5.1f}%"
      f"   median {np.median(length):.4g}   geomean {np.exp(np.log(length).mean()):.4g}"
      f"   [p5 {np.percentile(length, 5):.4g}, p95 {np.percentile(length, 95):.4g}]"
    )
    print(
      f"  fits with ANY coordinate pinned {np.mean(at_low.any(axis=1) | at_high.any(axis=1)) * 100:5.1f}%"
      f"   with ALL pinned at HIGH {at_high.all(axis=1).mean() * 100:5.1f}%"
      f"   with ALL pinned at LOW {at_low.all(axis=1).mean() * 100:5.1f}%"
      f"   mean pinned coordinates {(at_low | at_high).sum(axis=1).mean():.2f} of {length.shape[1]}"
    )
    print(
      f"  amplitude^2 : at LOW {amplitude_at_low.mean() * 100:5.1f}%   at HIGH {amplitude_at_high.mean() * 100:5.1f}%"
      f"   median {np.median(amplitude):.4g}   [p5 {np.percentile(amplitude, 5):.4g}, p95 {np.percentile(amplitude, 95):.4g}]"
    )
    print(
      f"  observed y variance: median {np.median(y_variance):.4g}"
      f"   [p5 {np.percentile(y_variance, 5):.4g}, p95 {np.percentile(y_variance, 95):.4g}]"
      f"   amplitude^2 / var median {np.median(amplitude / np.maximum(y_variance, 1e-30)):.4g}"
    )
    print('  by observation count:')
    for n in sorted(set(counts.tolist())):
      select = counts == n
      print(
        f"    n={n:3d}  low {at_low[select].mean() * 100:5.1f}%  high {at_high[select].mean() * 100:5.1f}%"
        f"  l geomean {np.exp(np.log(length[select]).mean()):.4g}"
        f"  amp^2 median {np.median(amplitude[select]):.4g}"
        f"  amp low {amplitude_at_low[select].mean() * 100:5.1f}% high {amplitude_at_high[select].mean() * 100:5.1f}%"
        f"  var median {np.median(y_variance[select]):.4g}"
      )
    jumps = []
    for record in payload['records']:
      series = np.log(np.asarray([fit['length_scale'] for fit in record['fits']], dtype=float))
      if series.shape[0] > 1:
        jumps.append(np.abs(np.diff(series, axis=0)).ravel())
    jumps = np.concatenate(jumps)
    print(
      f"  |d log l| between consecutive iterations: median {np.median(jumps):.4g}"
      f"  mean {jumps.mean():.4g}  p95 {np.percentile(jumps, 95):.4g}"
      f"  fraction > 1 (a factor e): {np.mean(jumps > 1.0) * 100:.1f}%"
    )


def main():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='command', required=True)
  fits = subparsers.add_parser('fits')
  fits.add_argument('files', nargs='+')
  fits.add_argument('--tolerance', type=float, default=1e-4)
  criterion = subparsers.add_parser('criterion')
  criterion.add_argument('file')
  criterion.add_argument('--against', default=None)
  arguments = parser.parse_args()
  if arguments.command == 'fits':
    report_fits(arguments)
  else:
    report_criterion(arguments)


if __name__ == '__main__':
  main()
