#!/usr/bin/env python3
"""Figures for the `enzyme_depletion` task: one per experiment count, plus the calibration panel.

    python scripts/plot_enzyme_depletion.py --run output/enzyme-depletion/m1.json \
        output/enzyme-depletion/m2.json output/enzyme-depletion/m4.json \
        --output-dir output/enzyme-depletion

Each per-count figure carries three panels: the read-out the tuned design actually produces, the
loss landscape over designs, and the BO-against-random convergence over seeds. Colours are the
Okabe-Ito colourblind-safe order, used in a fixed order and never cycled; magnitude is a single-hue
sequential ramp; every panel with two series carries a legend.
"""
import argparse
import json
import os
import warnings

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, VERMILLION, GREEN, ORANGE, PURPLE = '#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7'
GRID = {'color': '#D9D9D9', 'linewidth': 0.6}
INK, MUTED = '#1A1A1A', '#666666'


def style(axis, title, xlabel, ylabel):
  axis.set_title(title, fontsize=10, color=INK, loc='left')
  axis.set_xlabel(xlabel, fontsize=9, color=MUTED)
  axis.set_ylabel(ylabel, fontsize=9, color=MUTED)
  axis.tick_params(labelsize=8, colors=MUTED, length=3)
  axis.grid(True, **GRID)
  axis.set_axisbelow(True)
  for side in ('top', 'right'):
    axis.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    axis.spines[side].set_color('#BFBFBF')


def evaluate(detector, scaled, index):
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(scaled, np.float32))
  _, event, _, target = detector(design, index)
  predicted = detector.estimate(design, event)
  return float(jnp.mean(detector.loss(predicted, detector.normalize_target(target))))


def panel_readout(axis, detector, best_scaled):
  """The measurement the tuned design produces, for variants across the K prior."""
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(best_scaled, np.float32))
  initial = np.asarray(design.initial_concentration, np.float64)
  index = np.arange(4096)
  _, event, _, target = detector(design, index)
  kinetics = np.asarray(target.kinetics, np.float64)
  times = np.arange(1, detector.n_measurements + 1) * detector.duration / detector.n_measurements
  # three variants spanning the K prior, at the median velocity
  wanted = np.geomspace(detector.michaelis_bounds[0] * 3, detector.michaelis_bounds[1] / 3, 3)
  chosen = [int(np.argmin(np.abs(np.log(kinetics[:, 1] / w)) + np.abs(np.log(kinetics[:, 0] / 3.16e-4)))) for w in wanted]
  dense = np.linspace(0.0, detector.duration, 400)
  for colour, i in zip((BLUE, VERMILLION, GREEN), chosen):
    q, michaelis = kinetics[i]
    for j in range(detector.n_experiments):
      curve = _exact(dense, initial[j], q, michaelis)
      axis.plot(
        dense / 3600.0, curve, color=colour, linewidth=1.4, alpha=0.9, label=f'K = {michaelis:.2f} mM' if j == 0 else None
      )
      axis.plot(
        times / 3600.0, np.asarray(event.concentration[i, j], np.float64), 'o', color=colour, markersize=4.5,
        markeredgecolor='white', markeredgewidth=0.7
      )
  axis.axhline(0.0, color='#BFBFBF', linewidth=0.8)
  style(
    axis, f'read-out at the tuned design  (A0 = {", ".join(f"{a:.2f}" for a in np.sort(initial))} mM)', 'time (h)', '[A] (mM)'
  )
  axis.legend(fontsize=8, frameon=False, labelcolor=INK)


def _exact(t, initial, velocity, michaelis, n_newton=60):
  t, initial, velocity, michaelis = np.broadcast_arrays(*[np.asarray(x, np.float64) for x in (t, initial, velocity, michaelis)])
  s = np.log(initial / michaelis) + (initial - velocity * t) / michaelis
  v = np.where(s > 1.0, np.log(np.maximum(s - np.log(np.maximum(s, 1.0 + 1e-12)), 1e-300)), s)
  for _ in range(n_newton):
    ev = np.exp(np.clip(v, -700.0, 700.0))
    v = v - (ev + v - s) / (ev + 1.0)
  return michaelis * np.exp(np.clip(v, -700.0, 700.0))


def panel_landscape(axis, detector, guess, n_designs, n_events, seed=7):
  """m = 1: the loss profile over A0. m > 1: the |x|^2 bowl around the best design found."""
  from scipy.stats import qmc

  index = np.arange(n_events)
  if detector.n_experiments == 1:
    low, high = detector.concentration_bounds
    grid = np.geomspace(low, high, 25)
    values = [evaluate(detector, detector.to_scaled({'initial_concentration': [float(a)]}), index) for a in grid]
    axis.plot(
      grid, values, color=BLUE, linewidth=1.8, marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.6,
      label='estimator loss'
    )
    axis.axhline(guess, color=VERMILLION, linewidth=1.4, linestyle='--', label='no information (1/3)')
    axis.set_xscale('log')
    axis.set_yscale('log')
    style(axis, 'loss over the design box', 'initial concentration A0 (mM)', 'MSE on (ln q, ln K)')
    axis.legend(fontsize=8, frameon=False, labelcolor=INK)
    return
  points = qmc.Sobol(detector.n_experiments, scramble=True, seed=seed).random(n_designs)
  values = np.array([evaluate(detector, x, index) for x in points])
  star = points[int(values.argmin())]
  distance = np.linalg.norm(np.sort(points, axis=1) - np.sort(star)[None, :], axis=1)
  axis.scatter(distance, values, s=16, color=BLUE, alpha=0.65, edgecolor='none', label='random designs')
  edges = np.quantile(distance, np.linspace(0.0, 1.0, 9))
  centres = 0.5 * (edges[:-1] + edges[1:])
  binned = [values[(distance >= edges[i]) & (distance <= edges[i + 1])].mean() for i in range(8)]
  axis.plot(centres, binned, color=VERMILLION, linewidth=2.0, label='bin mean')
  axis.axhline(guess, color=MUTED, linewidth=1.2, linestyle='--', label='no information (1/3)')
  axis.set_yscale('log')
  style(axis, 'loss against distance from the best design found', '|x - x*| in the scaled cube (sorted)', 'MSE on (ln q, ln K)')
  axis.legend(fontsize=8, frameon=False, labelcolor=INK)


def panel_convergence(axis, payload):
  guess = payload['no_information_loss']
  for colour, arm in ((BLUE, 'bo'), (VERMILLION, 'random')):
    if arm not in payload['arms']:
      continue
    curves = np.minimum.accumulate(np.asarray(payload['arms'][arm]['observed'], float), axis=1)
    steps = np.arange(1, curves.shape[1] + 1)
    median = np.median(curves, axis=0)
    lower, upper = np.percentile(curves, [25, 75], axis=0)
    axis.plot(steps, median, color=colour, linewidth=2.0, label=f'{arm} (median of {curves.shape[0]} seeds)')
    axis.fill_between(steps, lower, upper, color=colour, alpha=0.16, linewidth=0)
  axis.axhline(guess, color=MUTED, linewidth=1.2, linestyle='--', label='no information (1/3)')
  for point in payload['checkpoints']:
    axis.axvline(point, color='#BFBFBF', linewidth=0.9, linestyle=':')
  axis.set_yscale('log')
  style(axis, 'best-so-far against designs evaluated', 'designs evaluated', 'MSE on (ln q, ln K)')
  axis.legend(fontsize=8, frameon=False, labelcolor=INK)


def criterion(payload, arm='bo'):
  first, last = payload['checkpoints'][0], payload['checkpoints'][-1]
  values = payload['arms'][arm]['reevaluated']
  early, late = np.asarray(values[str(first)], float), np.asarray(values[str(last)], float)
  guess = payload['no_information_loss']
  return ((early.mean() - late.mean()) / guess, float(np.std(early - late) / np.sqrt(early.size) / guess), early.mean(),
          late.mean())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run', nargs='+', required=True)
  parser.add_argument('--n-designs', type=int, default=192)
  parser.add_argument('--n-events', type=int, default=2048)
  parser.add_argument('--output-dir', default='output/enzyme-depletion')
  arguments = parser.parse_args()
  warnings.simplefilter('ignore')
  os.makedirs(arguments.output_dir, exist_ok=True)

  from detopt.detector import EnzymeDepletionDetector

  for path in arguments.run:
    with open(path) as handle:
      payload = json.load(handle)
    n_experiments = payload['n_experiments']
    detector = EnzymeDepletionDetector(n_experiments=n_experiments, measurement_noise=payload['measurement_noise'])
    reevaluated = np.asarray(payload['arms']['bo']['reevaluated'][str(payload['checkpoints'][-1])], float)
    best = np.asarray(payload['arms']['bo']['incumbents'], float)[int(reevaluated.argmin())]

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    figure.patch.set_facecolor('white')
    panel_readout(axes[0], detector, best)
    panel_landscape(axes[1], detector, payload['no_information_loss'], arguments.n_designs, arguments.n_events)
    panel_convergence(axes[2], payload)
    value, error, early, late = criterion(payload)
    figure.suptitle(
      f'enzyme_depletion, {n_experiments} experiment(s), read-out noise {payload["measurement_noise"]} mM   '
      f'|   (E[loss @ {payload["checkpoints"][0]}] - E[loss @ {payload["checkpoints"][-1]}]) / (1/3) = '
      f'{value:.4f} +- {error:.4f}   ({early:.4f} -> {late:.4f}, {payload["n_seeds"]} seeds)', fontsize=10.5, color=INK
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(arguments.output_dir, f'm{n_experiments}.png')
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f'wrote {os.path.abspath(out)}')


if __name__ == '__main__':
  main()
