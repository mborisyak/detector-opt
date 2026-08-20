#!/usr/bin/env python3
"""BO on the `enzyme_depletion` objective -- the ANALYTIC instrument, many seeds, no network.

    python scripts/enzyme_depletion_bo.py --n-experiments 4 --n-seeds 256 --n-iterations 20 \
        --output output/enzyme-depletion/m4.json

There is no regressor to train here: the objective IS the estimator's parameter error, and the
estimator (`EnzymeDepletionDetector.estimate`, a grid posterior mean) costs milliseconds. So the
question a neural campaign can only ask five times is asked here over hundreds of seeds.

THE OBJECTIVE. For one seed, ONE block of events is drawn and every design in that seed's run is
scored on the SAME block (common random numbers), so the objective is a deterministic function of
the design and the design comparison is paired. The reported `loss @ n designs` is NOT the best
value observed -- a running minimum over noisy evaluations is optimistically biased -- but the
incumbent design's loss RE-EVALUATED on an independent block of events.

THE CRITERION, as pre-registered:

    ( E[loss @ 10 designs] - E[loss @ 20 designs] ) / loss_of_a_guess

with `loss_of_a_guess` the detector's closed-form no-information level (1/3) and the expectation
over seeds. `--arms bo,random` runs the random-search null beside it on the same seeds.

The surrogate is `sorting-rbf` with `exchangeable: n_experiments`: the design is a SET of initial
concentrations, so permuting the experiments names the same experiment. Trust regions are banned
on this project and none is used.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

import detopt.bo
import detopt.detector
import detopt.utils.config
from detopt.bo import BayesianOptimizer, kernel_from_config
from detopt.detector import EnzymeDepletionDetector


def build_detector(arguments):
  """The detector under test: a DETECTOR CONFIG when one is given, the class defaults otherwise.

  `--n-experiments` / `--measurement-noise` stay honoured in both cases, so one config file serves a
  whole batch-size and noise sweep."""
  import yaml

  if arguments.detector_config is None:
    if arguments.n_experiments is None or arguments.measurement_noise is None:
      raise SystemExit('without --detector-config both --n-experiments and --measurement-noise are required')
    return EnzymeDepletionDetector(n_experiments=arguments.n_experiments, measurement_noise=arguments.measurement_noise)
  with open(arguments.detector_config) as handle:
    config = yaml.safe_load(handle)
  overrides = list(arguments.overrides)
  if arguments.n_experiments is not None:
    overrides.append(f'enzyme_depletion.n_experiments={arguments.n_experiments}')
  if arguments.measurement_noise is not None:
    overrides.append(f'enzyme_depletion.measurement_noise={arguments.measurement_noise}')
  if len(overrides) > 0:
    config = detopt.utils.config.override(config, overrides)
  return detopt.detector.from_config(config)


def build(detector, gp, kernel_name, n_init):
  kernel = kernel_from_config({kernel_name: {'exchangeable': detector.n_experiments}}, detector, gp)
  return BayesianOptimizer(detector.design_dim(), gp=gp, ei={'n_restarts': 16, 'n_steps': 60}, kernel=kernel, n_init=n_init)


def objective(detector, scaled, event_index):
  """Mean loss of the analytic instrument over `event_index`, at a SCALED design."""
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(scaled, np.float32))
  _, event, _, target = detector(design, event_index)
  predicted = detector.estimate(design, event)
  return float(jnp.mean(detector.loss(predicted, detector.normalize_target(target))))


def run(arm, detector, seed, n_iterations, n_events, gp, kernel_name, n_init, checkpoints):
  """One seed of one arm. Returns the observed curve and the re-evaluated incumbent loss at each
  checkpoint (an independent event block, so the number carries no selection bias)."""
  rng = np.random.default_rng(seed)
  train_index = np.arange(seed * n_events, (seed + 1) * n_events)
  holdout_index = np.arange(10_000_000 + seed * n_events, 10_000_000 + (seed + 1) * n_events)
  optimiser = build(detector, gp, kernel_name, n_init)
  observed, designs, reevaluated = [], [], {}
  for iteration in range(n_iterations):
    if arm == 'random':
      x = rng.random(detector.design_dim())
    else:
      x = np.asarray(optimiser.propose(int(seed) * 1000 + iteration), dtype=float)
    value = objective(detector, x, train_index)
    optimiser.append(x, value, noise=1e-6)
    observed.append(value)
    designs.append(x)
    if (iteration + 1) in checkpoints:
      incumbent = designs[int(np.argmin(observed))]
      reevaluated[iteration + 1] = objective(detector, incumbent, holdout_index)
  return observed, reevaluated, designs[int(np.argmin(observed))].tolist()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n-experiments', type=int, default=None)
  parser.add_argument('--measurement-noise', type=float, default=None)
  parser.add_argument(
    '--detector-config', default=None,
    help='a DETECTOR config yaml (config/detector/*.yaml) supplying the prior, the design box and the '
    'window; without it the detector defaults are used and only the two arguments above may move'
  )
  parser.add_argument(
    '--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
    help='dotted override on the detector config, the repo\'s usual form, e.g. '
    '`--set enzyme_depletion.measurement_noise=0.0125`'
  )
  parser.add_argument('--n-seeds', type=int, default=256)
  parser.add_argument(
    '--seed-start', type=int, default=0, help='first seed; chunks of one campaign use disjoint blocks and are merged'
  )
  parser.add_argument('--n-iterations', type=int, default=20)
  parser.add_argument('--n-events', type=int, default=1024)
  parser.add_argument('--n-init', type=int, default=5)
  parser.add_argument('--kernel', default='sorting-rbf')
  parser.add_argument('--arms', default='bo,random')
  parser.add_argument('--checkpoints', default='10,20')
  parser.add_argument(
    '--log-lengthscale-bounds', type=float, nargs=2, default=(-2.0, 1.0),
    help='GP lengthscale prior box in LOG units, against the SCALED cube [0, 1]'
  )
  parser.add_argument(
    '--log-amplitude-bounds', type=float, nargs=2, default=(-6.0, 1.5),
    help='GP amplitude prior box in LOG units; the kernel carries amplitude^2, so these are halved logs'
  )
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()

  warnings.simplefilter('ignore')
  checkpoints = tuple(int(c) for c in arguments.checkpoints.split(','))
  detector = build_detector(arguments)
  gp = {
    'n_folds': 5,
    'n_restarts': 5,
    'n_steps': 40,
    'log_lengthscale_prior_bounds': list(arguments.log_lengthscale_bounds),
    'log_amplitude_prior_bounds': list(arguments.log_amplitude_bounds)
  }
  payload = {
    'n_experiments': int(detector.n_experiments),
    'measurement_noise': float(detector.measurement_noise),
    'concentration_bounds': list(detector.concentration_bounds),
    'michaelis_bounds': list(detector.michaelis_bounds),
    'velocity_bounds': list(detector.velocity_bounds),
    'duration': float(detector.duration),
    'n_seeds': arguments.n_seeds,
    'seed_start': arguments.seed_start,
    'n_iterations': arguments.n_iterations,
    'n_events': arguments.n_events,
    'kernel': arguments.kernel,
    'log_lengthscale_prior_bounds': list(arguments.log_lengthscale_bounds),
    'log_amplitude_prior_bounds': list(arguments.log_amplitude_bounds),
    'checkpoints': list(checkpoints),
    'no_information_loss': detector.no_information_loss(),
    'concentration_bounds': list(detector.concentration_bounds),
    'arms': {},
  }
  for arm in arguments.arms.split(','):
    curves, reevaluated, incumbents = [], {c: [] for c in checkpoints}, []
    start = time.time()
    for seed in range(arguments.seed_start, arguments.seed_start + arguments.n_seeds):
      observed, holdout, incumbent = run(
        arm, detector, seed, arguments.n_iterations, arguments.n_events, gp, arguments.kernel, arguments.n_init, checkpoints
      )
      curves.append(observed)
      incumbents.append(incumbent)
      for c in checkpoints:
        reevaluated[c].append(holdout[c])
      if (seed + 1 - arguments.seed_start) % 16 == 0:
        print(f'  {arm} {seed + 1 - arguments.seed_start}/{arguments.n_seeds}  ({time.time() - start:.0f} s)', flush=True)
    payload['arms'][arm] = {
      'observed': curves,
      'reevaluated': {
        str(c): reevaluated[c]
        for c in checkpoints
      },
      'incumbents': incumbents
    }

  os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
  with open(arguments.output, 'w') as handle:
    json.dump(payload, handle)

  guess = detector.no_information_loss()
  for arm, data in payload['arms'].items():
    at = {c: float(np.mean(data['reevaluated'][str(c)])) for c in checkpoints}
    criterion = (at[checkpoints[0]] - at[checkpoints[-1]]) / guess
    error = float(
      np.std(np.asarray(data['reevaluated'][str(checkpoints[0])]) - np.asarray(data['reevaluated'][str(checkpoints[-1])])) /
      np.sqrt(arguments.n_seeds) / guess
    )
    print(
      f'{arm}: E[loss @ {checkpoints[0]}] = {at[checkpoints[0]]:.4f}, '
      f'E[loss @ {checkpoints[-1]}] = {at[checkpoints[-1]]:.4f}, '
      f'criterion = {criterion:.4f} +- {error:.4f}'
    )
  print(f'wrote {arguments.output}')


if __name__ == '__main__':
  main()
