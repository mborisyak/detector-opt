#!/usr/bin/env python3
"""BO on the `enzyme_depletion_bi` objective -- the ANALYTIC instrument, many seeds, no network.

    python scripts/enzyme_depletion_bi_bo.py --detector-config config/detector/enzyme_depletion_bi_m2.yaml \
        --n-seeds 64 --n-iterations 20 --output output/enzyme-depletion-bi/m2.json

There is no regressor to train here: the objective IS the estimator's parameter error, and the
estimator (`EnzymeDepletionBiDetector.estimate`, a grid posterior mean over (ln q, ln K_A, ln K_B))
costs under a second. So the question a neural campaign can only ask five times is asked here over
dozens of independent runs.

THE OBJECTIVE. For one seed, ONE block of events is drawn and every design in that seed's run is
scored on the SAME block (common random numbers), so the objective is a deterministic function of
the design and the design comparison within a run is paired. The reported `loss @ n` is NOT the best
value observed -- a running minimum over noisy evaluations is optimistically biased -- but the
incumbent design's loss RE-EVALUATED on an independent block of events.

THE PRIMARY FIGURE OF MERIT

    P( loss@2n  <  loss@n ),  over ORDERED PAIRS of DIFFERENT runs

i.e. `loss@2n` from run j against `loss@n` from run i, for every i != j. Different runs, because
within one run the incumbent is monotone by construction and the comparison returns 1 trivially.
This measures what the task exists to show: whether doubling the iteration budget reliably pays,
rather than paying on average once enough seeds are averaged.

The bonus figure is `E[loss@n] - E[loss@2n]`, and the same pair of numbers divided by the detector's
closed-form no-information level is the campaign criterion the sibling task pre-registered. Every
arm is run on the SAME seeds, so `--arms bo,random` gives the random-search null paired run by run.

The surrogate is `sorting-rbf` with `exchangeable: n_experiments`: the design is a SET of (A0, B0)
PAIRS, and `_exchangeable_blocks` hands the kernel both design fields so a permutation moves each
experiment's two concentrations together. Trust regions are banned on this project and none is used.
"""
import argparse
import itertools
import json
import os
import time
import warnings

import numpy as np

import detopt.detector
import detopt.utils.config
from detopt.bo import BayesianOptimizer, kernel_from_config
from detopt.detector import EnzymeDepletionBiDetector

GP = {
  'n_folds': 5,
  'n_restarts': 5,
  'n_steps': 40,
  'log_lengthscale_prior_bounds': [-2.0, 1.0],
  'log_amplitude_prior_bounds': [-6.0, 1.5],
}


def build_detector(arguments):
  """The detector under test: a DETECTOR CONFIG when one is given, the class defaults otherwise."""
  import yaml

  if arguments.detector_config is None:
    return EnzymeDepletionBiDetector(n_experiments=arguments.n_experiments, measurement_noise=arguments.measurement_noise)
  with open(arguments.detector_config) as handle:
    config = yaml.safe_load(handle)
  overrides = list(arguments.overrides)
  if arguments.n_experiments is not None:
    overrides.append(f'enzyme_depletion_bi.n_experiments={arguments.n_experiments}')
  if arguments.measurement_noise is not None:
    overrides.append(f'enzyme_depletion_bi.measurement_noise={arguments.measurement_noise}')
  if len(overrides) > 0:
    config = detopt.utils.config.override(config, overrides)
  return detopt.detector.from_config(config)


def build_optimiser(detector, kernel_name, n_init):
  kernel = kernel_from_config({kernel_name: {'exchangeable': detector.n_experiments}}, detector, GP)
  return BayesianOptimizer(detector.design_dim(), gp=GP, ei={'n_restarts': 16, 'n_steps': 60}, kernel=kernel, n_init=n_init)


def objective(detector, scaled, event_index):
  """Mean loss of the analytic instrument over `event_index`, at a SCALED design."""
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(scaled, np.float32))
  _, event, _, target = detector(design, event_index)
  predicted = detector.estimate(design, event)
  return float(jnp.mean(detector.loss(predicted, detector.normalize_target(target))))


def run(arm, detector, seed, arguments, checkpoints):
  """One seed of one arm: the observed curve, and the re-evaluated incumbent at each checkpoint."""
  rng = np.random.default_rng(seed)
  train_index = np.arange(seed * arguments.n_events, (seed + 1) * arguments.n_events)
  holdout_index = np.arange(10_000_000 + seed * arguments.n_events, 10_000_000 + (seed + 1) * arguments.n_events)
  optimiser = build_optimiser(detector, arguments.kernel, arguments.n_init)
  observed, designs, reevaluated = [], [], {}
  for iteration in range(arguments.n_iterations):
    if arm == 'random':
      point = rng.random(detector.design_dim())
    else:
      point = np.asarray(optimiser.propose(int(seed) * 1000 + iteration), dtype=float)
    value = objective(detector, point, train_index)
    optimiser.append(point, value, noise=1e-6)
    observed.append(value)
    designs.append(point)
    if (iteration + 1) in checkpoints:
      reevaluated[iteration + 1] = objective(detector, designs[int(np.argmin(observed))], holdout_index)
  return observed, reevaluated, designs[int(np.argmin(observed))].tolist()


def probability_of_improvement(early, late, n_bootstrap=2000, seed=0):
  """`P(late < early)` over ORDERED PAIRS of DIFFERENT runs, with a seed-level bootstrap interval.

  Pairing runs i != j is what makes this a statement about run-to-run reliability rather than about
  the monotone incumbent inside one run."""
  early, late = np.asarray(early, float), np.asarray(late, float)
  pairs = [(i, j) for i, j in itertools.product(range(early.size), range(late.size)) if i != j]
  index_i = np.array([p[0] for p in pairs])
  index_j = np.array([p[1] for p in pairs])
  estimate = float(np.mean(late[index_j] < early[index_i]))
  rng = np.random.default_rng(seed)
  draws = []
  for _ in range(n_bootstrap):
    take = rng.integers(0, early.size, early.size)
    sample_early, sample_late = early[take], late[take]
    grid = sample_late[None, :] < sample_early[:, None]
    np.fill_diagonal(grid, False)
    draws.append(grid.sum() / (early.size * (early.size - 1)))
  return estimate, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def acceptance(payload, early, late, loss_precision):
  """`docs/benchmark-acceptance.md` section 2.1, evaluated per seed and printed as PASS/FAIL.

  `baseline` is the MEDIAN RANDOM DESIGN, taken from the random arm's own observed values in this
  same run, so it is measured on the same objective and the same event blocks. Without a random arm
  there is no baseline and the clauses that use it are skipped."""
  if 'random' not in payload['arms']:
    return None
  baseline = float(np.median(np.asarray(payload['arms']['random']['observed'])))
  report = {'baseline': baseline, 'arms': {}}
  for arm, data in payload['arms'].items():
    first = np.asarray(data['reevaluated'][str(early)])
    second = np.asarray(data['reevaluated'][str(late)])
    strong = ((first < baseline) & (second < first) & ((first - second) > 0.2 * (baseline - first))
              & ((first - second) > 10.0 * loss_precision))
    weak = (first < baseline) & (second < first)
    passed = bool(strong.sum() > first.size / 2) and bool(weak.all())
    report['arms'][arm] = {
      'n_strong': int(strong.sum()),
      'n_seeds': int(first.size),
      'weak_all': bool(weak.all()),
      'pass': passed
    }
    print(
      f'    {arm:>7}: baseline (median random design) = {baseline:.4f};  strong {int(strong.sum())}/{first.size} '
      f'(need > {first.size / 2:.1f});  weak on every seed: {bool(weak.all())}  ->  '
      f'{"PASS" if passed else "FAIL"}'
    )
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--detector-config', default=None, help='a DETECTOR config yaml (config/detector/*.yaml)')
  parser.add_argument('--n-experiments', type=int, default=None)
  parser.add_argument('--measurement-noise', type=float, default=None)
  parser.add_argument(
    '--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
    help='dotted override on the detector config, e.g. `--set enzyme_depletion_bi.measurement_noise=0.025`'
  )
  parser.add_argument('--n-seeds', type=int, default=64)
  parser.add_argument('--seed-start', type=int, default=0)
  parser.add_argument('--n-iterations', type=int, default=20)
  parser.add_argument('--n-events', type=int, default=1024)
  parser.add_argument('--n-init', type=int, default=5)
  parser.add_argument('--kernel', default='sorting-rbf')
  parser.add_argument('--arms', default='bo,random')
  parser.add_argument('--checkpoints', default='5,10,20')
  parser.add_argument(
    '--loss-precision', type=float, default=5.0e-3,
    help='the resolution floor of acceptance clause (d), `error`. On a NEURAL objective this is the '
    'trainer`s convergence slack; on THIS analytic objective there is no trainer, so it is the '
    'objective`s own reproducibility -- MEASURED as the standard deviation of one design`s loss over '
    'independent event blocks of 1024, which is 0.0031 at a good design and 0.0073 at a bad one, hence '
    'the 5.0e-3 default. Using the trainer`s 1.25e-2 here would put clause (d)`s bar at 0.125, above '
    'the entire loss range of the task, and fail every candidate by construction.'
  )
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()

  warnings.simplefilter('ignore')
  checkpoints = tuple(int(c) for c in arguments.checkpoints.split(','))
  detector = build_detector(arguments)
  payload = {
    'n_experiments': int(detector.n_experiments),
    'measurement_noise': float(detector.measurement_noise),
    'concentration_a_bounds': list(detector.concentration_a_bounds),
    'concentration_b_bounds': list(detector.concentration_b_bounds),
    'velocity_bounds': list(detector.velocity_bounds),
    'michaelis_a_bounds': list(detector.michaelis_a_bounds),
    'michaelis_b_bounds': list(detector.michaelis_b_bounds),
    'duration': float(detector.duration),
    'n_measurements': int(detector.n_measurements),
    'n_grid': int(detector.n_grid),
    'n_seeds': arguments.n_seeds,
    'seed_start': arguments.seed_start,
    'n_iterations': arguments.n_iterations,
    'n_events': arguments.n_events,
    'kernel': arguments.kernel,
    'checkpoints': list(checkpoints),
    'no_information_loss': detector.no_information_loss(),
    'arms': {},
  }
  for arm in arguments.arms.split(','):
    curves, reevaluated, incumbents = [], {c: [] for c in checkpoints}, []
    start = time.time()
    for seed in range(arguments.seed_start, arguments.seed_start + arguments.n_seeds):
      observed, holdout, incumbent = run(arm, detector, seed, arguments, checkpoints)
      curves.append(observed)
      incumbents.append(incumbent)
      for checkpoint in checkpoints:
        reevaluated[checkpoint].append(holdout[checkpoint])
      done = seed + 1 - arguments.seed_start
      if done % 8 == 0:
        print(f'  {arm} {done}/{arguments.n_seeds}  ({time.time() - start:.0f} s)', flush=True)
    payload['arms'][arm] = {
      'observed': curves,
      'reevaluated': {
        str(c): reevaluated[c]
        for c in checkpoints
      },
      'incumbents': incumbents,
    }

  os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
  with open(arguments.output, 'w') as handle:
    json.dump(payload, handle)

  guess = detector.no_information_loss()
  doubles = [(a, b) for a, b in itertools.product(checkpoints, checkpoints) if b == 2 * a]
  for arm, data in payload['arms'].items():
    print(f'{arm}:')
    for checkpoint in checkpoints:
      values = np.asarray(data['reevaluated'][str(checkpoint)])
      print(f'    E[loss @ {checkpoint:2d}] = {values.mean():.4f} +- {values.std() / np.sqrt(values.size):.4f}')
    for early, late in doubles:
      first = np.asarray(data['reevaluated'][str(early)])
      second = np.asarray(data['reevaluated'][str(late)])
      estimate, low, high = probability_of_improvement(first, second)
      print(
        f'    n = {early:2d} -> {late:2d}:  P(loss@2n < loss@n) = {estimate:.3f} [{low:.3f}, {high:.3f}]   '
        f'E[loss@n] - E[loss@2n] = {first.mean() - second.mean():+.4f}   '
        f'criterion = {(first.mean() - second.mean()) / guess:+.4f}'
      )
  for early, late in doubles:
    print(
      f'acceptance test (docs/benchmark-acceptance.md 2.1) at n = {early} -> {late}, '
      f'loss_precision = {arguments.loss_precision:g}:'
    )
    acceptance(payload, early, late, arguments.loss_precision)
  print(f'wrote {arguments.output}')


if __name__ == '__main__':
  main()
