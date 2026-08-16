#!/usr/bin/env python3
"""The `linear` landscape in d DIMENSIONS, in closed form -- stage 1 of `docs/linear-highdim.md`.

NO TRAINING HAPPENS HERE. Every number is the Bayes risk
``tr[(X^T X / sigma^2 + I)^-1] / (d + 1)`` with rows ``(x_i, 1)``, ``x_i`` in ``[-1, 1]^d``, and the
target ``(w, b)`` with ``d + 1`` standard-normal components -- the direct generalisation of
``LinearDetector.bayes_risk``, which is the 1-dimensional case of exactly this expression. The
detector is NOT imported: this script must be runnable while the 1-dimensional campaign owns
``detopt/detector/linear.py``, and the formula is short enough to state twice.

WHAT IT ANSWERS. Acceptance criterion (d) of `docs/benchmark-acceptance.md` wants
``loss@n - loss@2n > 10 * loss_precision``. The n -> 2n improvement can never exceed the whole
distance from the baseline (the MEDIAN random design) to the optimum, so

    gain := median_random - optimum                       an UPPER BOUND on what (d) can measure
    loss_precision >= max over designs of (diff + err)    the pre-registered slack protocol

is a NECESSARY condition on the landscape alone, before any network is trained. ``err`` is the
statistical term and it is derivable: the per-event loss of the Bayes predictor is
``sum_k mu_k z_k^2 / (d + 1)`` with ``mu_k`` the eigenvalues of the posterior covariance and
``z_k`` standard normal, so

    E[loss]  = sum mu_k / (d + 1)                          == the level itself
    sd[loss] = sqrt(2 sum mu_k^2) / (d + 1)
    sd/mean  = sqrt(2 sum mu_k^2) / sum mu_k               in [sqrt(2/(d+1)), sqrt(2)]

⚠️ The 1-dimensional shorthand "the standard deviation equals the mean" is a SPECIAL CASE, not a
general fact: it holds when the eigenvalues are equal and there are exactly two of them. With d + 1
components the ratio moves between ``sqrt(2/(d+1))`` (all directions equally uncertain) and
``sqrt(2)`` (one direction dominating), so it is computed per design here rather than assumed. The
Monte-Carlo check (`--monte-carlo`) draws the posterior residual directly and confirms the formula.

The trainer reports ``err = hypot(train_sem, val_sem)`` over a window of ``W`` training rows and
``W * val_fraction / (1 - val_fraction)`` validation rows, i.e. ``W / 3`` at the settled
``val_fraction = 0.25``, so ``err = sd * sqrt(1/W + 3/W) = 2 sd / sqrt(W)``.

USAGE

    srun --cpus-per-task=2 -u python scripts/linear_nd_landscape.py \
        --output output/linear-nd/landscape.json

Reported per (d, n_probes, sigma): the optimum and how it is reached, the structural ceiling, the
rank deficiency when ``n_probes < d + 1``, the random-design quantiles, the gain, the
``loss_precision`` criterion (d) then demands, and the intrinsic ``err`` at the worst design for
several window caps.
"""

import argparse
import itertools
import json
import os

import numpy as np
import scipy.optimize


def risk(probes, sigma, dimension):
  """Bayes risk of a batch of designs. ``probes`` is ``(..., n_probes, d)`` NOMINAL, in ``[-1, 1]^d``.

  Returned per design: ``tr[(X^T X / sigma^2 + I)^-1] / (d + 1)`` with rows ``(x_i, 1)``. Computed
  from the eigenvalues of ``X^T X`` rather than by inverting a matrix, so a rank-deficient design
  (which is the whole point when ``n_probes <= d``) costs no accuracy: a zero eigenvalue contributes
  exactly 1, i.e. one prior direction that the read-out never touches.
  """
  probes = np.asarray(probes, np.float64)
  rows = np.concatenate([probes, np.ones(probes.shape[:-1] + (1, ), np.float64)], axis=-1)
  gram = np.einsum('...ij,...ik->...jk', rows, rows)
  eigenvalues = np.linalg.eigvalsh(gram)
  eigenvalues = np.clip(eigenvalues, 0.0, None)
  return np.sum(1.0 / (1.0 + eigenvalues / sigma**2), axis=-1) / (dimension + 1)


def posterior_eigenvalues(probes, sigma):
  """The eigenvalues ``mu_k`` of the posterior covariance ``(X^T X / sigma^2 + I)^-1`` at ONE design."""
  probes = np.asarray(probes, np.float64)
  rows = np.concatenate([probes, np.ones((probes.shape[0], 1), np.float64)], axis=-1)
  eigenvalues = np.clip(np.linalg.eigvalsh(rows.T @ rows), 0.0, None)
  return 1.0 / (1.0 + eigenvalues / sigma**2)


def loss_spread(probes, sigma):
  """``(level, sd, sd / level)`` of the PER-EVENT loss of the Bayes predictor at one design."""
  mu = posterior_eigenvalues(probes, sigma)
  level = float(np.sum(mu) / mu.size)
  sd = float(np.sqrt(2.0 * np.sum(mu**2)) / mu.size)
  return level, sd, sd / level


def monte_carlo_spread(probes, sigma, n_events, seed):
  """The same two moments by SIMULATION, as a check that the closed form is the right one.

  Draws ``(w, b)`` from the prior, the read-out noise, forms the exact posterior mean and reports the
  mean and standard deviation of the realised per-event squared error. It is the definition of the
  quantity the trainer's ``err`` is the standard error OF.
  """
  probes = np.asarray(probes, np.float64)
  n_probes, dimension = probes.shape
  rows = np.concatenate([probes, np.ones((n_probes, 1), np.float64)], axis=-1)
  precision = rows.T @ rows / sigma**2 + np.eye(dimension + 1)
  covariance = np.linalg.inv(precision)
  generator = np.random.default_rng(seed)
  theta = generator.standard_normal((n_events, dimension + 1))
  observations = theta @ rows.T + sigma * generator.standard_normal((n_events, n_probes))
  posterior_mean = (covariance @ (rows.T @ observations.T) / sigma**2).T
  squared = np.mean(np.square(theta - posterior_mean), axis=-1)
  return float(np.mean(squared)), float(np.std(squared))


def vertex_designs(dimension, n_probes):
  """Every design whose probes all sit on CORNERS of the box, as ``(n_combinations, n_probes, d)``.

  ``(2^d)^n_probes`` of them, which is 64 at (d=2, n=3) and 4096 at (d=3, n=4) -- exhaustive rather
  than argued, so the claim "the optimum is a corner design" is checked and not assumed.
  """
  corners = np.array(list(itertools.product([-1.0, 1.0], repeat=dimension)), np.float64)
  index = np.array(list(itertools.product(range(corners.shape[0]), repeat=n_probes)), np.int64)
  return corners[index]


def extreme(dimension, n_probes, sigma, sign, n_restarts, seed):
  """Multi-start L-BFGS-B on the box for the minimum (``sign = +1``) or the maximum (``sign = -1``).

  Reported alongside the exhaustive corner enumeration, never instead of it: gradient descent on a
  landscape with a measure-zero degenerate set finds what it is started near, and the two together
  are what says whether the corner claim holds.
  """
  generator = np.random.default_rng(seed)
  bounds = [(-1.0, 1.0)] * (n_probes * dimension)

  def objective(flat):
    return sign * float(risk(flat.reshape(n_probes, dimension), sigma, dimension))

  best_value, best_point = np.inf, None
  starts = generator.uniform(-1.0, 1.0, (n_restarts, n_probes * dimension))
  starts = np.concatenate([starts, np.zeros((1, n_probes * dimension))], axis=0)
  for start in starts:
    result = scipy.optimize.minimize(objective, start, method='L-BFGS-B', bounds=bounds)
    if float(result.fun) < best_value:
      best_value, best_point = float(result.fun), np.asarray(result.x, np.float64)
  return sign * best_value, best_point.reshape(n_probes, dimension)


def coincident_candidates(dimension, n_probes, n_common, seed):
  """Designs with EVERY probe at one common point: the corners, the origin, and random interior
  points. These are the structurally degenerate designs -- rank 1 whatever ``n_probes`` is -- and they
  are the ones `scripts/probe_precision.py` names `centre`, `corner-low` and `corner-high`."""
  generator = np.random.default_rng(seed)
  corners = np.array(list(itertools.product([-1.0, 1.0], repeat=dimension)), np.float64)
  common = np.concatenate([corners, np.zeros((1, dimension)), generator.uniform(-1.0, 1.0, (n_common, dimension))], axis=0)
  return np.repeat(common[:, None, :], n_probes, axis=1)


def cell(dimension, n_probes, sigma, n_random, n_restarts, seed, windows, diff_over_level, sd_inflation):
  """Everything stage 1 owes for one ``(d, n_probes, sigma)``."""
  corner_batch = vertex_designs(dimension, n_probes)
  corner_losses = risk(corner_batch, sigma, dimension)
  corner_best = float(np.min(corner_losses))
  corner_best_design = corner_batch[int(np.argmin(corner_losses))]

  continuous_best, continuous_best_design = extreme(dimension, n_probes, sigma, +1.0, n_restarts, seed)
  optimum = min(corner_best, continuous_best)
  optimum_design = corner_best_design if corner_best <= continuous_best else continuous_best_design

  coincident = coincident_candidates(dimension, n_probes, 8, seed + 1)
  coincident_losses = risk(coincident, sigma, dimension)
  coincident_worst = float(np.max(coincident_losses))
  coincident_worst_design = coincident[int(np.argmax(coincident_losses))]
  continuous_worst, continuous_worst_design = extreme(dimension, n_probes, sigma, -1.0, n_restarts, seed + 2)
  ceiling = max(coincident_worst, continuous_worst)
  ceiling_design = coincident_worst_design if coincident_worst >= continuous_worst else continuous_worst_design

  generator = np.random.default_rng(seed + 3)
  random_designs = generator.uniform(-1.0, 1.0, (n_random, n_probes, dimension))
  random_losses = risk(random_designs, sigma, dimension)
  p50 = float(np.median(random_losses))
  p90 = float(np.quantile(random_losses, 0.90))
  p_max = float(np.max(random_losses))
  median_design = random_designs[int(np.argsort(random_losses)[n_random // 2])]

  level_ceiling, sd_ceiling, ratio_ceiling = loss_spread(ceiling_design, sigma)
  level_median, sd_median, ratio_median = loss_spread(median_design, sigma)
  level_optimum, sd_optimum, ratio_optimum = loss_spread(optimum_design, sigma)

  gain = p50 - optimum
  demanded_precision = gain / 10.0

  # The rank deficiency, stated as the share of the prior that NO design can reach. With `n_probes`
  # rows in `d + 1` unknowns the read-out spans at most `n_probes` directions, so `d + 1 - n_probes`
  # of them keep their full prior variance of 1 -- a floor on the loss that is independent of sigma.
  deficiency = max(0, (dimension + 1) - n_probes)
  structural_floor = deficiency / (dimension + 1)

  # `err` at the WORST design, which is the binding one: it is the largest level in the box and both
  # slack terms scale with the level. The Bayes value is a FLOOR (the network's residual is wider than
  # the posterior's); `sd_inflation` carries the 1-dimensional campaign's measured excess.
  errors = {}
  for window in windows:
    floor = 2.0 * sd_ceiling / np.sqrt(window)
    errors[str(window)] = {
      'err_bayes_floor': float(floor),
      'err_inflated': float(floor * sd_inflation),
      'ten_err_inflated': float(10.0 * floor * sd_inflation),
      'gain_over_ten_err': float(gain / (10.0 * floor * sd_inflation))
    }

  # The margin criterion (d) leaves once BOTH slack terms are paid at the worst design. `diff` is an
  # overfitting bias with no closed form, so it enters as the 1-dimensional campaign's measured
  # `diff / level` extrapolated to this cell -- an ESTIMATE, and exactly what stage 2 measures.
  margins = {}
  for window in windows:
    err = errors[str(window)]['err_inflated']
    slack = diff_over_level * level_ceiling + err
    margins[str(window)] = {
      'predicted_slack': float(slack),
      'ten_slack': float(10.0 * slack),
      'margin': float(gain - 10.0 * slack),
      'gain_over_ten_slack': float(gain / (10.0 * slack))
    }

  return {
    'd': dimension,
    'n_probes': n_probes,
    'sigma': sigma,
    'design_dim': n_probes * dimension,
    'optimum': optimum,
    'optimum_is_corner': bool(corner_best <= continuous_best + 1e-12),
    'optimum_corner_enumeration': corner_best,
    'optimum_continuous': continuous_best,
    'optimum_design': optimum_design.tolist(),
    'ceiling': ceiling,
    'ceiling_is_coincident': bool(coincident_worst >= continuous_worst - 1e-12),
    'ceiling_design': ceiling_design.tolist(),
    'structural_floor': structural_floor,
    'rank_deficiency': deficiency,
    'no_information_level': 1.0,
    'p50': p50,
    'p90': p90,
    'max_random': p_max,
    'gain': gain,
    'demanded_loss_precision': demanded_precision,
    'sd_over_level_ceiling': ratio_ceiling,
    'sd_over_level_median': ratio_median,
    'sd_over_level_optimum': ratio_optimum,
    'level_ceiling': level_ceiling,
    'sd_ceiling': sd_ceiling,
    'level_median': level_median,
    'level_optimum': level_optimum,
    'errors': errors,
    'margins': margins
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dimensions', type=int, nargs='+', default=[1, 2, 3])
  parser.add_argument(
    '--sigmas', type=float, nargs='+', default=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    help='read-out noise standard deviations to sweep'
  )
  parser.add_argument('--n-random', type=int, default=200000, help='uniform designs per cell for the quantiles')
  parser.add_argument('--n-restarts', type=int, default=64, help='L-BFGS-B starts for each extreme')
  parser.add_argument(
    '--windows', type=int, nargs='+', default=[65536, 262144, 1048576],
    help='training-window caps (iteration_limit) at which to price the intrinsic `err`'
  )
  parser.add_argument(
    '--diff-over-level', type=float, default=0.018,
    help='ESTIMATE: the train/validation gap as a fraction of the loss level, carried from the '
    '1-dimensional campaign (slack 0.0155 at the coincident designs, of which err ~0.0067, '
    'against a level of 0.50)'
  )
  parser.add_argument(
    '--sd-inflation', type=float, default=1.42,
    help='ESTIMATE: how much wider the trained network`s per-event loss is than the Bayes '
    'posterior`s, from the same 1-dimensional measurement (err 0.0067 at a window of 45056 '
    'implies sd 0.71 against the Bayes 0.50)'
  )
  parser.add_argument('--seed', type=int, default=20260815)
  parser.add_argument('--monte-carlo', type=int, default=200000, help='events for the sd/mean cross-check (0 disables)')
  parser.add_argument('--output', default='output/linear-nd/landscape.json')
  arguments = parser.parse_args()

  cells = []
  for dimension in arguments.dimensions:
    for n_probes in (dimension, dimension + 1):
      for sigma in arguments.sigmas:
        row = cell(
          dimension, n_probes, sigma, arguments.n_random, arguments.n_restarts, arguments.seed, arguments.windows,
          arguments.diff_over_level, arguments.sd_inflation
        )
        cells.append(row)
        print(
          f"d={dimension} n={n_probes} sigma={sigma:<4g} | opt {row['optimum']:.4f} "
          f"p50 {row['p50']:.4f} p90 {row['p90']:.4f} ceil {row['ceiling']:.4f} | "
          f"gain {row['gain']:.4f} demands LP<{row['demanded_loss_precision']:.5f} | "
          f"sd/level@ceiling {row['sd_over_level_ceiling']:.3f} | "
          f"margin@2^18 {row['margins'][str(arguments.windows[1])]['margin']:+.4f}", flush=True
        )

  check = []
  if arguments.monte_carlo > 0:
    for dimension in arguments.dimensions:
      for n_probes in (dimension, dimension + 1):
        for sigma in (0.1, 0.3, 0.7):
          probes = np.repeat(np.zeros((1, dimension)), n_probes, axis=0)
          level, sd, ratio = loss_spread(probes, sigma)
          mc_level, mc_sd = monte_carlo_spread(probes, sigma, arguments.monte_carlo, arguments.seed)
          check.append({
            'd': dimension,
            'n_probes': n_probes,
            'sigma': sigma,
            'closed_form_level': level,
            'monte_carlo_level': mc_level,
            'closed_form_sd': sd,
            'monte_carlo_sd': mc_sd,
            'closed_form_ratio': ratio,
            'monte_carlo_ratio': mc_sd / mc_level
          })
          print(
            f"[check] d={dimension} n={n_probes} sigma={sigma} coincident-at-origin: "
            f"level {level:.5f} vs MC {mc_level:.5f} | sd {sd:.5f} vs MC {mc_sd:.5f} | "
            f"sd/level {ratio:.4f} vs MC {mc_sd / mc_level:.4f}", flush=True
          )

  os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
  with open(arguments.output, 'w') as f:
    json.dump({'settings': vars(arguments), 'cells': cells, 'monte_carlo_check': check}, f, indent=2, default=float)
  print(f"\nwrote {arguments.output} ({len(cells)} cells)")


if __name__ == '__main__':
  main()
