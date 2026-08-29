#!/usr/bin/env python3
"""What scale the analytic enzyme objective actually has, and what the BO surrogate fits for it.

CPU-only, seconds per seed: the objective is `EnzymeDepletionDetector.estimate` (a grid posterior
mean), the same instrument `scripts/enzyme_depletion_bo.py` optimises, so the GP's hyperparameters
can be watched over hundreds of runs instead of five.

Two modes.

`scales` measures the quantities a prior bound should be derived FROM, none of them from a finished
run:

  * the spread of the objective over the design cube, which is what an amplitude models;
  * the lengthscale and amplitude the SAME kernel picks when the fit is well determined -- a
    marginal-likelihood fit on a large uniform sample under deliberately inert bounds;
  * the nearest-neighbour spacing of the observations a real run has, which is the shortest
    lengthscale that is distinguishable from white noise, and the diameter of the searched region,
    which is the longest that is distinguishable from a constant.

`fits` re-runs the driver's OWN fit at every BO iteration and records the fitted amplitude and
per-coordinate lengthscales, the log marginal likelihood, the variance of the observations, and the
distance of each hyperparameter to its prior bound in log units. Pinning is then counted, not
asserted; the iteration-to-iteration jump in log-lengthscale is recorded beside it, because a fit
that swings without ever touching a bound is unstable too.

`resolution` fixes a design and varies only the event block, so its spread is the instrument's own
Monte-Carlo scatter -- the bar any difference measured with it has to clear.

The searched frame is the kernel's, not the cube's: `sorting-rbf` compares designs after sorting the
exchangeable block, so every distance here is measured after the same sort.
"""
import argparse
import json
import os
import warnings

import numpy as np
import yaml

import detopt.detector
import detopt.utils.config
from detopt.bo import BayesianOptimizer, kernel_from_config


def build_detector(path, n_experiments):
  with open(path) as handle:
    config = yaml.safe_load(handle)
  if n_experiments is not None:
    config = detopt.utils.config.override(config, [f'enzyme_depletion.n_experiments={n_experiments}'])
  return detopt.detector.from_config(config)


def gp_config(log_lengthscale_bounds, log_amplitude_bounds, n_restarts):
  return {
    'n_folds': 5,
    'n_restarts': n_restarts,
    'n_steps': 40,
    'log_lengthscale_prior_bounds': list(log_lengthscale_bounds),
    'log_amplitude_prior_bounds': list(log_amplitude_bounds)
  }


def build_optimiser(detector, gp, kernel_name, n_init):
  kernel = kernel_from_config({kernel_name: {'exchangeable': detector.n_experiments}}, detector, gp)
  return BayesianOptimizer(detector.design_dim(), gp=gp, ei={'n_restarts': 16, 'n_steps': 60}, kernel=kernel, n_init=n_init)


def objective(detector, scaled, event_index):
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(scaled, np.float32))
  _, event, _, target = detector(design, event_index)
  predicted = detector.estimate(design, event)
  return float(jnp.mean(detector.loss(predicted, detector.normalize_target(target))))


def sorted_frame(X):
  """The frame `sorting-rbf` compares in: the exchangeable block ascending."""
  return np.sort(np.atleast_2d(np.asarray(X, dtype=float)), axis=1)


def nearest_neighbour_spacing(rng, d, n_points, n_replicates):
  """Mean and median nearest-neighbour distance of `n_points` uniform designs, in the sorted frame."""
  values = []
  for _ in range(n_replicates):
    X = sorted_frame(rng.random((n_points, d)))
    gaps = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))
    np.fill_diagonal(gaps, np.inf)
    values.append(gaps.min(axis=1))
  values = np.concatenate(values)
  return float(values.mean()), float(np.median(values)), float(np.percentile(values, 10))


def fit_hyperparameters(detector, gp, kernel_name, X, y, noise, seed):
  """The driver's own fit on the given observations; returns the fitted kernel and its likelihood."""
  optimiser = build_optimiser(detector, gp, kernel_name, n_init=1)
  optimiser.append(X, y, noise)
  model = optimiser._fit(optimiser.X, optimiser.y - float(np.mean(optimiser.y)), optimiser.noise, int(seed))
  return model


def run_scales(arguments):
  rng = np.random.default_rng(arguments.seed)
  detector = build_detector(arguments.detector_config, arguments.n_experiments)
  d = detector.design_dim()
  payload = {'mode': 'scales', 'n_experiments': int(detector.n_experiments), 'design_dim': int(d)}

  # 1. The objective over the whole cube, one event block per replicate.
  samples, spreads, means, minima, maxima = [], [], [], [], []
  for replicate in range(arguments.n_blocks):
    index = np.arange(replicate * arguments.n_events, (replicate + 1) * arguments.n_events)
    X = rng.random((arguments.n_designs, d))
    y = np.asarray([objective(detector, x, index) for x in X])
    samples.append((X, y))
    spreads.append(float(np.var(y)))
    means.append(float(np.mean(y)))
    minima.append(float(np.min(y)))
    maxima.append(float(np.max(y)))
  payload['uniform'] = {
    'n_designs': arguments.n_designs,
    'n_blocks': arguments.n_blocks,
    'variance_mean': float(np.mean(spreads)),
    'variance_sem': float(np.std(spreads) / np.sqrt(len(spreads))),
    'mean_mean': float(np.mean(means)),
    'min_mean': float(np.mean(minima)),
    'max_mean': float(np.mean(maxima)),
  }

  # 2. What the SAME kernel fits when the fit is well determined: a large uniform sample under
  # bounds wide enough to be inert, so the answer is the data's and not the box's.
  wide = gp_config(arguments.wide_log_lengthscale_bounds, arguments.wide_log_amplitude_bounds, arguments.n_restarts)
  well_determined = []
  for X, y in samples:
    model = fit_hyperparameters(
      detector, wide, arguments.kernel, X, y, np.full(y.shape, arguments.well_determined_noise), arguments.seed
    )
    well_determined.append({
      'amplitude_squared': float(model.kernel_.constant_value),
      'length_scale': np.atleast_1d(model.kernel_.length_scale).astype(float).tolist(),
      'log_marginal_likelihood': float(model.log_marginal_likelihood_value_),
    })
  payload['well_determined'] = {
    'n_designs': arguments.n_designs,
    'noise': arguments.well_determined_noise,
    'wide_log_lengthscale_bounds': list(arguments.wide_log_lengthscale_bounds),
    'wide_log_amplitude_bounds': list(arguments.wide_log_amplitude_bounds),
    'fits': well_determined,
  }

  # 3. The geometry of the observation set a real run has.
  mean_gap, median_gap, p10_gap = nearest_neighbour_spacing(rng, d, arguments.n_observations, arguments.n_replicates)
  corners = sorted_frame(np.stack([np.zeros(d), np.ones(d)]))
  payload['geometry'] = {
    'n_observations': arguments.n_observations,
    'cube_diameter': float(np.sqrt(d)),
    'sorted_region_diameter': float(np.linalg.norm(corners[1] - corners[0])),
    'nearest_neighbour_mean': mean_gap,
    'nearest_neighbour_median': median_gap,
    'nearest_neighbour_p10': p10_gap,
  }
  return payload


def run_resolution(arguments):
  """What the instrument can resolve: the objective's scatter at a FIXED design.

  One design, many INDEPENDENT event blocks, so the spread is the read-out's own Monte-Carlo
  scatter and nothing else -- the number every difference in this study has to beat."""
  rng = np.random.default_rng(arguments.seed)
  detector = build_detector(arguments.detector_config, arguments.n_experiments)
  designs = rng.random((arguments.n_designs, detector.design_dim()))
  rows = []
  for design in designs:
    values = []
    for replicate in range(arguments.n_blocks):
      index = np.arange(replicate * arguments.n_events, (replicate + 1) * arguments.n_events)
      values.append(objective(detector, design, index))
    values = np.asarray(values)
    rows.append({
      'design': design.tolist(),
      'mean': float(values.mean()),
      'std': float(values.std(ddof=1)),
      'values': values.tolist(),
    })
  return {
    'mode': 'resolution',
    'n_experiments': int(detector.n_experiments),
    'design_dim': int(detector.design_dim()),
    'n_events': arguments.n_events,
    'n_blocks': arguments.n_blocks,
    'designs': rows,
  }


def run_fits(arguments):
  detector = build_detector(arguments.detector_config, arguments.n_experiments)
  d = detector.design_dim()
  gp = gp_config(arguments.log_lengthscale_bounds, arguments.log_amplitude_bounds, arguments.n_restarts)
  length_low, length_high = float(np.exp(gp['log_lengthscale_prior_bounds'][0])
                                  ), float(np.exp(gp['log_lengthscale_prior_bounds'][1]))
  amplitude_low, amplitude_high = float(np.exp(2 * gp['log_amplitude_prior_bounds'][0])
                                        ), float(np.exp(2 * gp['log_amplitude_prior_bounds'][1]))

  records = []
  for seed in range(arguments.seed_start, arguments.seed_start + arguments.n_seeds):
    rng = np.random.default_rng(seed)
    index = np.arange(seed * arguments.n_events, (seed + 1) * arguments.n_events)
    optimiser = build_optimiser(detector, gp, arguments.kernel, arguments.n_init)
    observed, per_iteration = [], []
    for iteration in range(arguments.n_iterations):
      if arguments.arm == 'random':
        x = rng.random(d)
      else:
        x = np.asarray(optimiser.propose(int(seed) * 1000 + iteration), dtype=float)
      value = objective(detector, x, index)
      optimiser.append(x, value, noise=arguments.noise)
      observed.append(value)
      if optimiser.X.shape[0] >= max(arguments.n_init, 2):
        model = optimiser._fit(
          optimiser.X, optimiser.y - float(np.mean(optimiser.y)), optimiser.noise,
          int(seed) * 1000 + iteration + 1
        )
        length_scale = np.atleast_1d(model.kernel_.length_scale).astype(float)
        per_iteration.append({
          'n': int(optimiser.X.shape[0]),
          'amplitude_squared': float(model.kernel_.constant_value),
          'length_scale': length_scale.tolist(),
          'log_marginal_likelihood': float(model.log_marginal_likelihood_value_),
          'y_variance': float(np.var(optimiser.y)),
        })
    records.append({'seed': int(seed), 'observed': [float(v) for v in observed], 'fits': per_iteration})
  return {
    'mode': 'fits',
    'arm': arguments.arm,
    'n_experiments': int(detector.n_experiments),
    'design_dim': int(d),
    'kernel': arguments.kernel,
    'log_lengthscale_prior_bounds': list(arguments.log_lengthscale_bounds),
    'log_amplitude_prior_bounds': list(arguments.log_amplitude_bounds),
    'length_scale_bounds': [length_low, length_high],
    'amplitude_squared_bounds': [amplitude_low, amplitude_high],
    'n_init': arguments.n_init,
    'n_iterations': arguments.n_iterations,
    'records': records,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', choices=('scales', 'fits', 'resolution'), required=True)
  parser.add_argument('--detector-config', default='config/detector/mm_hk_m2.yaml')
  parser.add_argument('--n-experiments', type=int, default=None)
  parser.add_argument('--kernel', default='sorting-rbf')
  parser.add_argument('--noise', type=float, default=1e-6)
  parser.add_argument('--n-events', type=int, default=1024)
  parser.add_argument('--n-restarts', type=int, default=5)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--n-designs', type=int, default=256)
  parser.add_argument('--n-blocks', type=int, default=8)
  parser.add_argument('--n-observations', type=int, default=20)
  parser.add_argument('--n-replicates', type=int, default=2000)
  parser.add_argument('--well-determined-noise', type=float, default=1e-4)
  parser.add_argument('--wide-log-lengthscale-bounds', type=float, nargs=2, default=(-6.0, 4.0))
  parser.add_argument('--wide-log-amplitude-bounds', type=float, nargs=2, default=(-8.0, 4.0))
  parser.add_argument('--log-lengthscale-bounds', type=float, nargs=2, default=(-2.0, 1.0))
  parser.add_argument('--log-amplitude-bounds', type=float, nargs=2, default=(-6.0, 1.5))
  parser.add_argument('--arm', default='bo')
  parser.add_argument('--seed-start', type=int, default=0)
  parser.add_argument('--n-seeds', type=int, default=64)
  parser.add_argument('--n-iterations', type=int, default=20)
  parser.add_argument('--n-init', type=int, default=5)
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()

  warnings.simplefilter('ignore')
  payload = {'scales': run_scales, 'fits': run_fits, 'resolution': run_resolution}[arguments.mode](arguments)
  os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
  with open(arguments.output, 'w') as handle:
    json.dump(payload, handle)
  print(f'wrote {arguments.output}')


if __name__ == '__main__':
  main()
