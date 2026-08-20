#!/usr/bin/env python3
"""Calibration evidence for the `enzyme_depletion` detector -- every number in its config header.

    python scripts/calibrate_enzyme_depletion.py --section all --output-dir output/enzyme-depletion

Sections, in the order the task was calibrated:

  profile         what the depletion curve does as A0 sweeps -- the three information regimes
  visibility      the knee-visibility curve over the parameter prior, and the design box it implies
  measurements    the half-decay width of a semi-optimal design -> the sampling count
  integrator      RKC2 against the closed-form solution over the WHOLE prior box, and its cost
  estimator       the analytic integrated-Michaelis-Menten fit against the grid posterior
  parameterisation  whether the estimator's error is scale-free in the value or in its log
  landscape       the loss profile over random designs, per experiment count and noise

The reference solution is closed form and INDEPENDENT of the detector: integrating the ODE gives
`A + K ln A = A0 + K ln A0 - q t`, i.e. `w + ln w = s` with `w = A/K`, solved by Newton in `ln w`.
"""
import argparse
import math
import os

import numpy as np

VELOCITY_BOUNDS = (1.0e-4, 1.0e-3)
MICHAELIS_BOUNDS = (0.01, 20.0)
CONCENTRATION_BOUNDS = (0.45, 4.0)
DURATION = 21600.0
N_MEASUREMENTS = 8
# Okabe-Ito, a colourblind-safe categorical order used in fixed order (never cycled).
PALETTE = ('#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9')


def exact_concentration(t, initial, velocity, michaelis, n_newton=60):
  """Closed-form `[A](t)` in float64: Newton on `e^v + v = s`, `v = ln(A/K)`."""
  t, initial, velocity, michaelis = np.broadcast_arrays(*[np.asarray(x, np.float64) for x in (t, initial, velocity, michaelis)])
  s = np.log(initial / michaelis) + (initial - velocity * t) / michaelis
  v = np.where(s > 1.0, np.log(np.maximum(s - np.log(np.maximum(s, 1.0 + 1e-12)), 1e-300)), s)
  for _ in range(n_newton):
    ev = np.exp(np.clip(v, -700.0, 700.0))
    v = v - (ev + v - s) / (ev + 1.0)
  return michaelis * np.exp(np.clip(v, -700.0, 700.0))


def draw_prior(n, seed):
  rng = np.random.default_rng(seed)
  return (np.exp(rng.uniform(*np.log(VELOCITY_BOUNDS), n)), np.exp(rng.uniform(*np.log(MICHAELIS_BOUNDS), n)))


def knee_time(initial, velocity, michaelis):
  """When `[A]` reaches `K`; infinite when it never does (`A0 <= K`)."""
  with np.errstate(divide='ignore', invalid='ignore'):
    t = ((initial - michaelis) + michaelis * np.log(initial / michaelis)) / velocity
  return np.where(initial > michaelis, t, np.inf)


def visible_fraction(initial, velocity, michaelis, margin=0.2, duration=DURATION):
  """Fraction of prior draws whose knee falls inside the window with `margin` to spare."""
  return float(np.mean(knee_time(initial, velocity, michaelis) <= (1.0 - margin) * duration))


# --------------------------------------------------------------------------------------------- #
def section_profile(arguments):
  print('=== 1. concentration profile: the three information regimes as A0 sweeps ===')
  print('    A flat curve identifies nothing; a straight one identifies q alone; only a curve whose')
  print('    knee is inside the window identifies q AND K. The sweep spans 0.003 .. 300 mM so all')
  print('    three regimes are present -- a sweep that showed only one could not separate them.')
  velocity, michaelis = draw_prior(arguments.n_draws, 0)
  print(f'    {"A0 (mM)":>9} {"P(<10% depleted)":>17} {"P(no knee, depleted)":>21} {"P(knee visible)":>16}')
  for initial in np.geomspace(0.003, 300.0, 13):
    left = exact_concentration(DURATION, initial, velocity, michaelis) / initial
    late = knee_time(initial, velocity, michaelis) > 0.8 * DURATION
    print(
      f'    {initial:9.3f} {float(np.mean(left > 0.9)):17.3f} '
      f'{float(np.mean((left <= 0.9) & late)):21.3f} '
      f'{visible_fraction(initial, velocity, michaelis):16.3f}'
    )


def section_visibility(arguments):
  print('=== 2/3. knee visibility, and the design box it restricts us to ===')
  print(f'    Requirement: for ANY design in the box the knee is visible for 50%+ of prior draws.')
  velocity, michaelis = draw_prior(arguments.n_draws, 0)
  grid = np.geomspace(0.1, 12.0, 31)
  fraction = np.array([visible_fraction(a, velocity, michaelis) for a in grid])
  for a, f in zip(grid, fraction):
    print(f'    A0={a:7.3f}  visible={f:.3f}{"  *" if f >= 0.5 else ""}')
  inside = grid[fraction >= 0.5]
  print(
    f'    --> visibility >= 0.50 on A0 in [{inside.min():.3f}, {inside.max():.3f}]; '
    f'peak {fraction.max():.3f} at A0 = {grid[int(fraction.argmax())]:.2f}'
  )
  low, high = CONCENTRATION_BOUNDS
  at_low, at_high = visible_fraction(low, velocity, michaelis), visible_fraction(high, velocity, michaelis)
  print(
    f'    chosen concentration_bounds = [{low}, {high}]: worst design in the box = '
    f'{min(at_low, at_high):.3f} (ends {at_low:.3f} / {at_high:.3f})'
  )


def section_measurements(arguments):
  print('=== 4. sampling count: the half-decay must span about 2 measurements ===')
  velocity = math.sqrt(VELOCITY_BOUNDS[0] * VELOCITY_BOUNDS[1])
  michaelis = math.sqrt(MICHAELIS_BOUNDS[0] * MICHAELIS_BOUNDS[1])
  print(f'    prior medians: q = {velocity:.3e} mM/s, K = {michaelis:.3f} mM; duration {DURATION:.0f} s')
  for initial in (1.5, 2.0, 2.5, 2.9, 3.5):
    width = (0.5 * initial + michaelis * math.log(3.0)) / velocity  # t(0.25 A0) - t(0.75 A0)
    print(
      f'    A0={initial:4.1f} mM: half-decay 0.75A0 -> 0.25A0 spans {width:7.0f} s '
      f'-> n_measurements = 2 T / width = {2 * DURATION / width:5.2f}'
    )
  print(
    f'    chosen n_measurements = {N_MEASUREMENTS}; at A0 = 2.5 mM the transition covers '
    f'{(0.5 * 2.5 + michaelis * math.log(3.0)) / velocity / (DURATION / N_MEASUREMENTS):.2f} sampling intervals'
  )


def section_integrator(arguments):
  import jax
  import jax.numpy as jnp
  from detopt.detector.enzyme import rkc2_chain, rkc2_coefficients

  print('=== RKC2 against the closed form, over the WHOLE prior box ===')
  print('    A corner list is not enough: 7 stages / 32 steps passed on corners and then tripped the')
  print('    detector assert on a random design, so this scans a dense grid and says where the worst is.')
  initial = np.geomspace(*CONCENTRATION_BOUNDS, 12)
  velocity = np.geomspace(*VELOCITY_BOUNDS, 12)
  michaelis = np.geomspace(*MICHAELIS_BOUNDS, 24)
  aa, qq, kk = [x.reshape(-1) for x in np.meshgrid(initial, velocity, michaelis, indexing='ij')]
  times = np.arange(1, N_MEASUREMENTS + 1) * DURATION / N_MEASUREMENTS
  truth = exact_concentration(times[None, :], aa[:, None], qq[:, None], kk[:, None])
  print(f'    {aa.size} (A0, q, K) points')
  print(
    f'    {"stages":>7} {"steps/meas":>11} {"dt (s)":>8} {"rate evals":>11} {"max|err|":>11} '
    f'{"max monitor":>12}   worst at'
  )
  for n_stages in (5, 7, 9):
    coefficients = rkc2_coefficients(n_stages, 2.0 / 13.0)
    for steps in (32, 48, 64, 96):
      dt = DURATION / (N_MEASUREMENTS * steps)

      def one(a, q, k):
        rate = lambda concentration: -q * concentration / (concentration + k)
        coarse = rkc2_chain(coefficients, rate, a, dt=dt, n_steps=steps, n_intervals=N_MEASUREMENTS)
        fine = rkc2_chain(coefficients, rate, a, dt=0.5 * dt, n_steps=2 * steps, n_intervals=N_MEASUREMENTS)
        return coarse, jnp.max(jnp.abs(fine - coarse))

      coarse, monitor = jax.jit(jax.vmap(one)
                                )(jnp.asarray(aa, jnp.float32), jnp.asarray(qq, jnp.float32), jnp.asarray(kk, jnp.float32))
      error = np.max(np.abs(np.asarray(coarse, np.float64) - truth), axis=1)
      i = int(np.nanargmax(error))
      print(
        f'    {n_stages:7d} {steps:11d} {dt:8.1f} {3 * n_stages * steps * N_MEASUREMENTS:11d} '
        f'{np.nanmax(error):11.3e} {float(jnp.max(monitor)):12.3e}   '
        f'A0={aa[i]:.2f} q={qq[i]:.1e} K={kk[i]:.3f}'
      )


# --------------------------------------------------------------------------------------------- #
# The REJECTED analytic instrument, kept here so its rejection is reproducible.
#
#   q t = (A0 - A) + K ln(A0/A)   ->   (A0 - A) = q t + K ln(A/A0)
# which is linear in (q, K), so ordinary least squares over every measurement of every experiment
# returns both in closed form. Three variants are measured: plain OLS, generalised least squares
# with the residual variance the errors-in-variables algebra predicts (sigma^2 (1 + K/A)^2), and
# GLS with the fitted curve substituted back into the regressor.
# --------------------------------------------------------------------------------------------- #
def _truncated_mean(mu, sd):
  from scipy.stats import norm

  sd = np.maximum(sd, 1e-12)
  a, b = (-1.0 - mu) / sd, (1.0 - mu) / sd
  mass = norm.cdf(b) - norm.cdf(a)
  safe = mass > 1e-12
  return np.where(
    safe, np.clip(mu + sd * (norm.pdf(a) - norm.pdf(b)) / np.where(safe, mass, 1.0), -1.0, 1.0), np.clip(mu, -1.0, 1.0)
  )


def _weighted_fit(x1, x2, response, weight):
  s11 = (weight * x1 * x1).sum(-1)
  s12 = (weight * x1 * x2).sum(-1)
  s22 = (weight * x2 * x2).sum(-1)
  b1 = (weight * x1 * response).sum(-1)
  b2 = (weight * x2 * response).sum(-1)
  determinant = s11 * s22 - s12 * s12
  usable = ((weight > 0).sum(-1) >= 3) & (determinant > 1e-12 * np.maximum(s11 * s22, 1e-300))
  safe = np.where(usable, determinant, 1.0)
  return (s22 * b1 - s12 * b2) / safe, (s11 * b2 - s12 * b1) / safe, s11 / safe, s22 / safe, usable


def _to_unit(value, bounds):
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return 2.0 * (np.log(value) - low) / (high - low) - 1.0


def analytic_estimate(initial, times, readings, noise, detection_limit, n_passes=1, substitute=False):
  """The rejected instrument: linearised integrated Michaelis-Menten, closed form."""
  initial_b = np.asarray(initial, float)[None, :, None]
  keep = (readings > detection_limit).astype(float).reshape(readings.shape[0], -1)
  clipped = np.maximum(readings, detection_limit)
  response = (initial_b - clipped).reshape(readings.shape[0], -1)
  x1 = np.broadcast_to(times[None, None, :], readings.shape).reshape(readings.shape[0], -1)
  x2 = np.log(clipped / initial_b).reshape(readings.shape[0], -1)
  flat = clipped.reshape(readings.shape[0], -1)
  velocity, michaelis, s11d, s22d, usable = _weighted_fit(x1, x2, response, keep)
  for _ in range(n_passes):
    clipped_k = np.clip(np.nan_to_num(michaelis, nan=MICHAELIS_BOUNDS[0]), *MICHAELIS_BOUNDS)[:, None]
    if substitute:
      clipped_q = np.clip(np.nan_to_num(velocity, nan=VELOCITY_BOUNDS[0]), *VELOCITY_BOUNDS)[:, None]
      model = exact_concentration(
        times[None, None, :],
        np.asarray(initial, float)[None, :, None], clipped_q[:, :, None], clipped_k[:, :, None]
      ).reshape(readings.shape[0], -1)
      x2 = np.log(np.maximum(model, 1e-12) / np.broadcast_to(initial_b, readings.shape).reshape(readings.shape[0], -1))
    velocity, michaelis, s11d, s22d, usable = _weighted_fit(x1, x2, response, keep / (1.0 + clipped_k / flat)**2)
  clipped_q = np.clip(np.nan_to_num(velocity, nan=VELOCITY_BOUNDS[0]), *VELOCITY_BOUNDS)
  clipped_k = np.clip(np.nan_to_num(michaelis, nan=MICHAELIS_BOUNDS[0]), *MICHAELIS_BOUNDS)
  span_q = math.log(VELOCITY_BOUNDS[1] / VELOCITY_BOUNDS[0])
  span_k = math.log(MICHAELIS_BOUNDS[1] / MICHAELIS_BOUNDS[0])
  return (
    _truncated_mean(
      np.where(usable, _to_unit(clipped_q, VELOCITY_BOUNDS), 0.0),
      np.where(usable, 2 * noise * np.sqrt(np.maximum(s22d, 0.0)) / (clipped_q * span_q), 1e6)
    ),
    _truncated_mean(
      np.where(usable, _to_unit(clipped_k, MICHAELIS_BOUNDS), 0.0),
      np.where(usable, 2 * noise * np.sqrt(np.maximum(s11d, 0.0)) / (clipped_k * span_k), 1e6)
    )
  )


def _readings(initial, velocity, michaelis, noise, seed, n_measurements=N_MEASUREMENTS):
  times = np.arange(1, n_measurements + 1) * DURATION / n_measurements
  clean = exact_concentration(
    times[None, None, :],
    np.asarray(initial, float)[None, :, None], velocity[:, None, None], michaelis[:, None, None]
  )
  return times, clean + noise * np.random.default_rng(seed).standard_normal(clean.shape)


def section_estimator(arguments):
  import jax.numpy as jnp
  from detopt.detector import EnzymeDepletionDetector

  print('=== 5a. the ANALYTIC instrument against the grid posterior (m = 1, A0 = 1.7 mM) ===')
  print('    The linearised form is exact in the parameters in exact arithmetic, but two things break')
  print('    it: readings at or below the detection limit carry no logarithm (visible in the 1e-9')
  print('    row), and with noise the SAME reading appears in the response and in the regressor, so')
  print('    the errors-in-variables bias does not average away. No information = 0.3333.')
  print(f'    {"noise":>7} {"n_meas":>7} {"OLS":>8} {"GLS":>8} {"GLS+substitute":>15} {"grid posterior":>15}')
  for n_measurements in (8, 16):
    for noise in (1e-9, 0.01, 0.02, 0.05, 0.1):
      velocity, michaelis = draw_prior(arguments.n_events, 2)
      times, readings = _readings([1.7], velocity, michaelis, noise, 991, n_measurements)
      truth = np.stack([_to_unit(velocity, VELOCITY_BOUNDS), _to_unit(michaelis, MICHAELIS_BOUNDS)], -1)
      row = []
      for passes, substitute in ((0, False), (1, False), (3, True)):
        estimate = np.stack(
          analytic_estimate([1.7], times, readings, max(noise, 1e-9), max(noise, 1e-9), passes, substitute), -1
        )
        row.append(float(np.mean((estimate - truth)**2)))
      detector = EnzymeDepletionDetector(n_experiments=1, n_measurements=n_measurements, measurement_noise=max(noise, 1e-6))
      design = detector.to_nominal(np.asarray(detector.to_scaled({'initial_concentration': [1.7]})))
      _, event, _, target = detector(design, np.arange(arguments.n_events))
      grid = float(jnp.mean(detector.loss(detector.estimate(design, event), detector.normalize_target(target))))
      print(f'    {noise:7.0e} {n_measurements:7d} {row[0]:8.4f} {row[1]:8.4f} {row[2]:15.4f} {grid:15.4f}')
  print('    --> the analytic instrument sits ABOVE the no-information level and gets WORSE with more')
  print('        measurements. Rejected; the detector ships the grid posterior mean.')


def section_parameterisation(arguments):
  import numpy as np
  from detopt.detector import EnzymeDepletionDetector

  print('=== target parameterisation: (q, K), (q, ln K) or (ln q, ln K)? ===')
  print('    Binned by the TRUE value at a good design: if the ABSOLUTE error tracks the parameter')
  print('    while the LOG error does not, the log is the coordinate the measurement supports.')
  detector = EnzymeDepletionDetector(n_experiments=2, measurement_noise=0.05)
  design = {'initial_concentration': [0.6, 3.0]}
  _, event, _, target = detector(design, np.arange(arguments.n_events))
  predicted = np.asarray(detector.estimate(design, event), np.float64)
  physical = np.asarray(detector.denormalize_predictions(predicted).kinetics, np.float64)
  truth = np.asarray(target.kinetics, np.float64)
  for column, (name, bounds) in enumerate((('q  ', VELOCITY_BOUNDS), ('K  ', MICHAELIS_BOUNDS))):
    edges = np.geomspace(bounds[0], bounds[1], 7)
    print(f'    {name}')
    for i in range(6):
      mask = (truth[:, column] >= edges[i]) & (truth[:, column] < edges[i + 1])
      if int(mask.sum()) < 30:
        continue
      absolute = float(np.sqrt(np.mean((physical[mask, column] - truth[mask, column])**2)))
      logarithmic = float(np.sqrt(np.mean((np.log(physical[mask, column]) - np.log(truth[mask, column]))**2)))
      print(
        f'      [{edges[i]:9.4g}, {edges[i + 1]:9.4g})  n={int(mask.sum()):5d}  '
        f'rmse(value)={absolute:10.4g}  rmse(ln)={logarithmic:6.3f}'
      )


def section_landscape(arguments):
  import warnings

  import jax.numpy as jnp
  from scipy.stats import qmc
  from detopt.detector import EnzymeDepletionDetector

  warnings.simplefilter('ignore')
  print('=== 5b. the loss profile over RANDOM designs, per experiment count and noise ===')
  print('    A flat profile means the task is trivial or hopeless; the target signature is a smooth')
  print(f'    bowl. No information = {1.0 / 3.0:.4f}.')
  index = np.arange(arguments.n_events)
  print(f'    {"m":>3} {"noise":>7} {"min":>8} {"25%":>8} {"median":>8} {"75%":>8} {"max":>8} {"max/min":>8}')
  for n_experiments in (1, 2, 4):
    for noise in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
      detector = EnzymeDepletionDetector(n_experiments=n_experiments, measurement_noise=noise)
      points = qmc.Sobol(n_experiments, scramble=True, seed=0).random(arguments.n_designs)
      values = []
      for point in points:
        design = detector.to_nominal(jnp.asarray(point, jnp.float32))
        _, event, _, target = detector(design, index)
        values.append(float(jnp.mean(detector.loss(detector.estimate(design, event), detector.normalize_target(target)))))
      p = np.percentile(values, [0, 25, 50, 75, 100])
      print(
        f'    {n_experiments:3d} {noise:7.3f} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f} {p[3]:8.4f} {p[4]:8.4f} '
        f'{p[4] / max(p[0], 1e-9):8.1f}'
      )


SECTIONS = {
  'profile': section_profile,
  'visibility': section_visibility,
  'measurements': section_measurements,
  'integrator': section_integrator,
  'estimator': section_estimator,
  'parameterisation': section_parameterisation,
  'landscape': section_landscape,
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--section', default='all', help=f'one of {sorted(SECTIONS)} or `all`')
  parser.add_argument('--n-draws', type=int, default=200000)
  parser.add_argument('--n-events', type=int, default=4096)
  parser.add_argument('--n-designs', type=int, default=48)
  parser.add_argument('--output-dir', default='output/enzyme-depletion')
  arguments = parser.parse_args()
  os.makedirs(arguments.output_dir, exist_ok=True)
  names = list(SECTIONS) if arguments.section == 'all' else [arguments.section]
  for name in names:
    SECTIONS[name](arguments)
    print()


if __name__ == '__main__':
  main()
