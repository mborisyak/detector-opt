#!/usr/bin/env python3
"""Shape and difficulty of the `enzyme_depletion` task under one candidate setting.

    JAX_PLATFORMS=cpu python scripts/mm_task_profile.py --detector-config config/detector/mm_hk_m2.yaml \
        --set enzyme_depletion.concentration_bounds='[0.004, 8.0]' --output output/mm-retune/candidate.json

Two measurements, both on the ANALYTIC instrument (`estimate`, the grid posterior mean), both under
common random numbers so every design in one report is scored on the SAME event block and the
comparison between designs is paired.

  * **profile** -- the loss of a REPLICATED design (every experiment at the same initial
    concentration) over a log grid of A0. This is the 1-D landscape: it locates the good region,
    shows where the loss stops responding to the design (a plateau), and separates the two ways of
    being bad, since it reports the (ln q, ln K) components apart.
  * **random** -- uniform draws in the scaled cube, the quantity the acceptance conditions are
    stated on: the loss distribution, the Q-Q deviation from `C ||x - x*||^2` in the SET metric
    (per-experiment coordinates sorted, as `scripts/mm_distance_vs_loss.py` defines it), and the
    same losses in interpretable units.

INTERPRETABLE UNITS. `reduction` is `sqrt(no_information_loss / loss)`, the factor by which the
posterior standard deviation beats the prior's. `ln_rmse` is the per-parameter root mean squared
error as a natural-log factor on the physical value, `sqrt(component) * span / 2`, and `relative` is
`exp(ln_rmse) - 1`, the fractional error on q or K.

No web access, no installs; CPU only.
"""
import argparse
import json
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import math

import numpy as np
import yaml

import detopt.detector
import detopt.utils.config
from mm_distance_vs_loss import sorted_batch


def evaluate(detector, scaled, event_index):
  """Per-event loss and per-parameter squared error of one SCALED design, as numpy arrays."""
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(scaled, np.float32))
  _, event, _, target = detector(design, event_index)
  predicted = detector.estimate(design, event)
  metric = detector.metric(predicted, detector.normalize_target(target))
  return {name: np.asarray(jnp.asarray(value), dtype=np.float64) for name, value in metric.items()}


def units(detector, loss, velocity, michaelis):
  """A loss plus its two components -> uncertainty reduction and per-parameter relative error."""
  spans = (
    math.log(detector.velocity_bounds[1] / detector.velocity_bounds[0]),
    math.log(detector.michaelis_bounds[1] / detector.michaelis_bounds[0])
  )
  velocity_ln = math.sqrt(max(velocity, 0.0)) * 0.5 * spans[0]
  michaelis_ln = math.sqrt(max(michaelis, 0.0)) * 0.5 * spans[1]
  return {
    'loss': float(loss),
    'velocity': float(velocity),
    'michaelis': float(michaelis),
    'reduction': float(math.sqrt(detector.no_information_loss() / loss)) if loss > 0.0 else float('inf'),
    'velocity_ln_rmse': velocity_ln,
    'michaelis_ln_rmse': michaelis_ln,
    'velocity_relative': float(math.exp(velocity_ln) - 1.0),
    'michaelis_relative': float(math.exp(michaelis_ln) - 1.0),
  }


def profile(detector, event_index, low, high, n_points):
  """Replicated-design landscape over NOMINAL A0 in [low, high], log spaced, outside the box allowed."""
  bounds = detector.concentration_bounds
  log_low, log_high = math.log(bounds[0]), math.log(bounds[1])
  rows = []
  for nominal in np.exp(np.linspace(math.log(low), math.log(high), n_points)):
    scaled = np.full(detector.design_dim(), (math.log(nominal) - log_low) / (log_high - log_low))
    metric = evaluate(detector, scaled, event_index)
    row = units(detector, metric['loss'].mean(), metric['velocity'].mean(), metric['michaelis'].mean())
    row['nominal'] = float(nominal)
    row['scaled'] = float(scaled[0])
    row['loss_sem'] = float(metric['loss'].std(ddof=1) / math.sqrt(metric['loss'].size))
    rows.append(row)
  return rows


def random_designs(detector, event_index, n_designs, seed):
  """Uniform draws in the scaled cube: losses, components and the design matrix."""
  rng = np.random.default_rng(seed)
  designs = rng.random((n_designs, detector.design_dim()))
  losses, velocity, michaelis, sem = [], [], [], []
  for design in designs:
    metric = evaluate(detector, design, event_index)
    losses.append(metric['loss'].mean())
    velocity.append(metric['velocity'].mean())
    michaelis.append(metric['michaelis'].mean())
    sem.append(metric['loss'].std(ddof=1) / math.sqrt(metric['loss'].size))
  return designs, np.array(losses), np.array(velocity), np.array(michaelis), np.array(sem)


def quantile_deviation(losses, distances):
  """Max |quantile(loss) - quantile(distance^2)| after min-max normalisation, on 101 levels."""

  def normalise(values):
    low, high = float(np.min(values)), float(np.max(values))
    return (values - low) / (high - low) if high > low else np.zeros_like(values)

  levels = np.linspace(0.0, 1.0, 101)
  loss_q = normalise(np.quantile(losses, levels))
  square_q = normalise(np.quantile(distances**2, levels))
  return float(np.max(np.abs(loss_q - square_q))), loss_q.tolist(), square_q.tolist()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--detector-config', required=True)
  parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE')
  parser.add_argument('--n-events', type=int, default=4096)
  parser.add_argument('--n-designs', type=int, default=1024)
  parser.add_argument('--profile-points', type=int, default=49)
  parser.add_argument('--profile-low', type=float, default=None)
  parser.add_argument('--profile-high', type=float, default=None)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--event-offset', type=int, default=0)
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()

  with open(arguments.detector_config) as handle:
    config = yaml.safe_load(handle)
  if len(arguments.overrides) > 0:
    config = detopt.utils.config.override(config, arguments.overrides)
  detector = detopt.detector.from_config(config)
  event_index = np.arange(arguments.event_offset, arguments.event_offset + arguments.n_events)

  low = arguments.profile_low if arguments.profile_low is not None else detector.concentration_bounds[0]
  high = arguments.profile_high if arguments.profile_high is not None else detector.concentration_bounds[1]
  landscape = profile(detector, event_index, low, high, arguments.profile_points)

  designs, losses, velocity, michaelis, sem = random_designs(detector, event_index, arguments.n_designs, arguments.seed)
  best = int(np.argmin(losses))
  worst = int(np.argmax(losses))
  median = int(np.argsort(losses)[losses.size // 2])
  reference = sorted_batch(designs[best], detector.n_experiments)
  distances = np.array([np.linalg.norm(sorted_batch(x, detector.n_experiments) - reference) for x in designs])
  deviation, loss_q, square_q = quantile_deviation(losses, distances)

  guess = detector.no_information_loss()
  summary = {
    'detector_config': arguments.detector_config,
    'overrides': arguments.overrides,
    'settings': {
      'n_experiments': int(detector.n_experiments),
      'n_measurements': int(detector.n_measurements),
      'duration': float(detector.duration),
      'measurement_noise': float(detector.measurement_noise),
      'concentration_bounds': list(detector.concentration_bounds),
      'michaelis_bounds': list(detector.michaelis_bounds),
      'velocity_bounds': list(detector.velocity_bounds),
      'n_grid': int(detector.n_grid),
      'steps_per_measurement': int(detector.steps_per_measurement),
      'n_events': arguments.n_events,
      'n_designs': arguments.n_designs,
      'seed': arguments.seed,
    },
    'no_information_loss': guess,
    'profile': landscape,
    'profile_minimum': min(landscape, key=lambda row: row['loss']),
    'profile_maximum': max(landscape, key=lambda row: row['loss']),
    'random': {
      'best': units(detector, losses[best], velocity[best], michaelis[best]),
      'median': units(detector, losses[median], velocity[median], michaelis[median]),
      'worst': units(detector, losses[worst], velocity[worst], michaelis[worst]),
      'best_design_scaled': designs[best].tolist(),
      'best_design_nominal': np.asarray(detector.to_nominal(designs[best].astype(np.float32))).tolist(),
      'median_design_nominal': np.asarray(detector.to_nominal(designs[median].astype(np.float32))).tolist(),
      'quantiles': {
        str(level): float(np.quantile(losses, level))
        for level in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
      },
      'loss_sem_median': float(np.median(sem)),
      'at_no_information_fraction': float(np.mean(losses > 0.95 * guess)),
      'at_profile_ceiling_fraction': float(np.mean(losses > 0.95 * losses.max())),
      'qq_max_abs_deviation_from_diagonal': deviation,
      'qq_loss_quantiles': loss_q,
      'qq_square_quantiles': square_q,
    },
  }
  os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
  with open(arguments.output, 'w') as handle:
    json.dump(summary, handle, indent=2)

  print(f'{arguments.detector_config}  {arguments.overrides}')
  print(
    f'  box {detector.concentration_bounds} mM, K {detector.michaelis_bounds} mM, '
    f'q {detector.velocity_bounds} mM/s, T {detector.duration / 3600:.2f} h, '
    f'noise {detector.measurement_noise} mM, m={detector.n_experiments}'
  )
  print('  profile (replicated design):')
  for row in landscape:
    print(
      f'    A0 = {row["nominal"]:9.4f} mM  loss {row["loss"]:.4f} +- {row["loss_sem"]:.4f}  '
      f'reduction {row["reduction"]:5.2f}x  ln q {row["velocity"]:.4f}  ln K {row["michaelis"]:.4f}'
    )
  for name in ('best', 'median', 'worst'):
    row = summary['random'][name]
    print(
      f'  {name:6s} loss {row["loss"]:.4f}  reduction {row["reduction"]:5.2f}x  '
      f'q +-{100 * row["velocity_relative"]:5.1f}%  K +-{100 * row["michaelis_relative"]:5.1f}%'
    )
  print(
    f'  qq deviation {deviation:.4f}   at-no-information {summary["random"]["at_no_information_fraction"]:.3f}   '
    f'at-ceiling {summary["random"]["at_profile_ceiling_fraction"]:.3f}'
  )
  print(f'wrote {arguments.output}')


if __name__ == '__main__':
  main()
