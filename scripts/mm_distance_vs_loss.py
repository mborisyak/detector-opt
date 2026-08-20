"""Tuning diagnostic for the enzyme-depletion (MM) task: distance-to-optimum against loss.

Samples uniform random SCALED designs, scores each with the same `objective` the BO arms use, takes
the sample minimiser as x*, and plots ||x - x*|| against loss. The question it answers is the shape
one: a well-posed BO task should look like a bowl, so the random-design loss distribution should
resemble that of ||x||^2 -- no flat plateau (dead space), no needle.

The design is a SET (the surrogate is `sorting-rbf` with `exchangeable: n_experiments`), so distance
is computed between designs whose per-experiment coordinates have been SORTED. Comparing unsorted
vectors would report a large distance between two designs that are the same batch in another order.

Left panel is the scatter with a least-squares quadratic in the distance overlaid. Right panel is a
quantile-quantile plot of the loss against ||x - x*||^2, both min-max normalised: a straight diagonal
means the profile is quadratic in the set metric, a curve that leaves the diagonal at the right-hand
end is a plateau, and one that hugs the axis near zero is a needle.

No web access, no installs; everything here is local.
"""
import argparse
import json

import numpy as np
import yaml

import detopt.detector
import detopt.utils.config
from enzyme_depletion_bo import objective


def sorted_batch(scaled, n_experiments):
  """`scaled` reshaped to (n_experiments, -1) with rows sorted lexicographically, then flattened."""
  matrix = np.asarray(scaled, dtype=float).reshape(n_experiments, -1)
  order = np.lexsort(matrix.T[::-1])
  return matrix[order].reshape(-1)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--detector-config', required=True)
  parser.add_argument('--n-designs', type=int, default=2048)
  parser.add_argument('--n-events', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()

  with open(arguments.detector_config) as handle:
    config = yaml.safe_load(handle)
  detector = detopt.detector.from_config(config)
  n_experiments = detector.n_experiments
  dimension = detector.design_dim()

  rng = np.random.default_rng(arguments.seed)
  event_index = np.arange(arguments.n_events)
  designs = rng.random((arguments.n_designs, dimension))
  losses = np.array([objective(detector, x, event_index) for x in designs])

  best = int(np.argmin(losses))
  reference = sorted_batch(designs[best], n_experiments)
  distances = np.array([np.linalg.norm(sorted_batch(x, n_experiments) - reference) for x in designs])

  coefficients = np.polyfit(distances, losses, 2)
  grid = np.linspace(0.0, distances.max(), 200)

  def normalise(values):
    low, high = float(np.min(values)), float(np.max(values))
    return (values - low) / (high - low) if high > low else np.zeros_like(values)

  quantiles = np.linspace(0.0, 1.0, 101)
  loss_q = normalise(np.quantile(losses, quantiles))
  square_q = normalise(np.quantile(distances**2, quantiles))

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))
  left.scatter(distances, losses, s=7, alpha=0.35, color='tab:blue', edgecolors='none')
  left.plot(grid, np.polyval(coefficients, grid), color='crimson', lw=2, label='least-squares quadratic')
  left.axhline(detector.no_information_loss(), color='k', ls=':', lw=1.5, label='no-information loss')
  left.plot(0.0, losses[best], marker='*', ms=16, color='gold', mec='k', mew=.8, label='x* (sample minimiser)')
  left.set_xlabel('||x - x*||  (set metric: per-experiment coordinates sorted)')
  left.set_ylabel('loss')
  left.set_title(
    f'{arguments.detector_config}\n{arguments.n_designs} uniform random designs, '
    f'{arguments.n_events} events, m={n_experiments}'
  )
  left.legend(fontsize=8)
  left.grid(alpha=.3)

  right.plot(square_q, loss_q, color='tab:purple', lw=2)
  right.plot([0, 1], [0, 1], color='k', ls='--', lw=1, label='exact quadratic profile')
  right.set_xlabel('normalised quantiles of ||x - x*||^2')
  right.set_ylabel('normalised quantiles of loss')
  right.set_title('profile shape against ||x||^2\ndiagonal = quadratic; flat tail at right = plateau')
  right.legend(fontsize=8)
  right.grid(alpha=.3)

  figure.tight_layout()
  figure.savefig(arguments.output, dpi=110)

  fraction_within = float(np.mean(losses > 0.95 * detector.no_information_loss()))
  summary = {
    'detector_config': arguments.detector_config,
    'n_designs': arguments.n_designs,
    'n_events': arguments.n_events,
    'n_experiments': int(n_experiments),
    'design_dim': int(dimension),
    'no_information_loss': float(detector.no_information_loss()),
    'loss_min': float(losses.min()),
    'loss_median': float(np.median(losses)),
    'loss_max': float(losses.max()),
    'loss_at_no_information_fraction': fraction_within,
    'best_design_scaled': designs[best].tolist(),
    'best_design_nominal': np.asarray(detector.to_nominal(designs[best].astype(np.float32))).tolist(),
    'quadratic_coefficients': coefficients.tolist(),
    'qq_max_abs_deviation_from_diagonal': float(np.max(np.abs(loss_q - square_q))),
  }
  with open(arguments.output.replace('.png', '.json'), 'w') as handle:
    json.dump(summary, handle, indent=2)
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
