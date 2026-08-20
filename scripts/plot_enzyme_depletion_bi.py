#!/usr/bin/env python3
"""Decision-grade figures for the `enzyme_depletion_bi` calibration.

    python scripts/plot_enzyme_depletion_bi.py --input-dir output/enzyme-depletion-bi

Every figure is EVIDENCE for one config number, not an illustration of it: it plots the quantity the
decision actually turned on, draws the decision threshold explicitly, shows the REJECTED alternatives
on the same axes, and is captioned with what it settles and what would falsify it. Figures whose
input json is missing are skipped, so a partial calibration still plots.

Colours are Okabe-Ito in FIXED order (never cycled), every series also carries its own marker, and
magnitude maps use a single-hue sequential ramp -- so identity is never carried by colour alone.
"""
import argparse
import json
import math
import os

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ('#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9')
MARKERS = ('o', 's', '^', 'D', 'v', 'P')
THRESHOLD = '#4d4d4d'
GRID = {'color': '#d9d9d9', 'linewidth': 0.6}


def _finish(figure, path, caption):
  """Save with the caption placed BELOW the axes, so it never lands on the tick labels.

  `bbox_inches='tight'` grows the saved canvas to include figure text drawn outside `[0, 1]`, which
  is what keeps a multi-line caption clear of the x labels whatever the subplot geometry is."""
  figure.text(
    0.5, -0.02 - 0.028 * caption.count('\n'), caption, ha='center', va='top', fontsize=8, color='#3d3d3d', linespacing=1.5
  )
  figure.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(figure)
  print(f'  {path}')


def _load(input_dir, name):
  path = os.path.join(input_dir, f'{name}.json')
  if not os.path.exists(path):
    return None
  with open(path) as handle:
    return json.load(handle)


def _style(axis, xlabel, ylabel, title):
  axis.set_xlabel(xlabel)
  axis.set_ylabel(ylabel)
  axis.set_title(title, fontsize=10)
  axis.grid(True, **GRID)
  axis.set_axisbelow(True)
  for side in ('top', 'right'):
    axis.spines[side].set_visible(False)


def plot_reference(input_dir, output_dir, tolerance):
  """THE INVERSION, validated -- not the quadrature, and not a claim of ground truth."""
  data = _load(input_dir, 'reference')
  if data is None:
    return
  errors = np.asarray(data['errors'])
  figure, axis = plt.subplots(figsize=(7.4, 3.9))
  axis.hist(np.log10(np.maximum(errors, 1e-16)), bins=20, color=PALETTE[0], edgecolor='white', linewidth=0.6)
  axis.axvline(math.log10(tolerance), color=THRESHOLD, linestyle='--', linewidth=1.5)
  axis.text(
    math.log10(tolerance),
    axis.get_ylim()[1] * 0.95, f'  integration_tolerance {tolerance:g} mM', color=THRESHOLD, fontsize=8, va='top'
  )
  diagonal = data.get('diagonal_worst')
  if diagonal is not None:
    axis.axvline(math.log10(diagonal), color=PALETTE[1], linestyle=':', linewidth=2)
    axis.text(math.log10(diagonal), axis.get_ylim()[1] * 0.55, '  worst DIAGONAL case', color=PALETTE[1], fontsize=8, va='top')
  _style(
    axis, 'log10 max |exact_extent(t) - LSODA(rtol 1e-12)| at the read-out times (mM)', 'random (A0, B0, q, K_A, K_B) points',
    'The INVERSION, against an independent stiff solver'
  )
  round_trip = data.get('round_trip_worst')
  extra = '' if round_trip is None else f' Round trip t(x(t)) closes to {round_trip:.1e} relative, which separates\nthe root-find from the derivation: a bad derivation agrees there and disagrees here, a bad root-find fails both.'
  _finish(
    figure, os.path.join(output_dir, 'reference.png'),
    'SETTLES: `exact_extent` may be used as an INDEPENDENT NUMERICAL ROUTE for checking RKC2. It is NOT ground truth and\n'
    'NOT a closed form -- separation gives t(x) explicitly, but x(t) is not elementary and is recovered by bisection plus\n'
    'Newton, so it carries root-finder error of its own (D157). What is validated here is that INVERSION, on the CORRECTED\n'
    f'product-form law: worst {errors.max():.2e} mM, {tolerance / errors.max():.0e}x below the tolerance the detector asserts.\n'
    'The DIAGONAL is forced into the sweep because a product of two random ranges never lands on A0 == B0, which is how the\n'
    'original NaN survived undetected.' + extra + '\nFALSIFIED BY: any point approaching the dashed line, which would mean '
    'this route, not RKC2, sets the error floor.'
  )


def plot_degeneracy(input_dir, output_dir):
  """THE CLAIM THE LAW CORRECTION OVERTURNED: the diagonal is weak, not degenerate."""
  data = _load(input_dir, 'degeneracy')
  if data is None:
    return
  rows = data['rows']
  noise = data['noise']
  initial_a = np.array([r[0] for r in rows])
  ratio = np.array([r[1] / r[0] for r in rows])
  separation = np.array([r[2] for r in rows])
  figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

  on_diagonal = np.isclose(ratio, 1.0)
  axes[0].scatter(
    ratio[~on_diagonal], separation[~on_diagonal], s=70, color=PALETTE[0], marker=MARKERS[0], zorder=3, label='A0 != B0'
  )
  axes[0].scatter(
    ratio[on_diagonal], separation[on_diagonal], s=90, color=PALETTE[1], marker=MARKERS[1], zorder=3,
    label='A0 == B0 (was EXACTLY 0 under ping-pong)'
  )
  axes[0].axhline(noise, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axes[0].text(
    0.02, noise * 1.15, f'read-out noise {noise:g} mM', color=THRESHOLD, fontsize=8, transform=axes[0].get_yaxis_transform()
  )
  axes[0].set_xscale('log')
  axes[0].set_yscale('log')
  axes[0].legend(frameon=False, fontsize=8, loc='upper left')
  _style(
    axes[0], 'B0 / A0', "max |curve(K_A, K_B) - curve(K_A', K_B')|  (mM)",
    'Two parameter sets with the SAME K_A + K_B: can the design split them?'
  )

  # The diagonal alone, against concentration, because this is where the fraction and the
  # signal-to-noise disagree and only the second is what the estimator sees.
  order = np.argsort(initial_a[on_diagonal])
  diagonal_a = initial_a[on_diagonal][order]
  diagonal_separation = separation[on_diagonal][order]
  axes[1].plot(
    diagonal_a, diagonal_separation / noise, marker=MARKERS[1], color=PALETTE[1], linewidth=2, markersize=8,
    label='separation / read-out noise (what is RESOLVED)'
  )
  axes[1].plot(
    diagonal_a, diagonal_separation / diagonal_a, marker=MARKERS[2], color=PALETTE[2], linewidth=2, markersize=8,
    linestyle='--', label='separation / A0 (the FRACTION -- misleading here)'
  )
  axes[1].axhline(1.0, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axes[1].text(0.02, 1.06, 'one noise sigma', color=THRESHOLD, fontsize=8, transform=axes[1].get_yaxis_transform())
  axes[1].set_xscale('log')
  axes[1].legend(frameon=False, fontsize=8, loc='upper right')
  _style(axes[1], 'A0 = B0 (mM)', 'diagonal separation, two normalisations', 'ON the diagonal: the two readings disagree')

  figure.tight_layout()
  _finish(
    figure, os.path.join(output_dir, 'degeneracy.png'),
    'SETTLES: the diagonal is WEAK BUT NOT DEGENERATE, so the box has no reason to exclude it. Under the PING-PONG law the\n'
    'rate on A0 == B0 collapsed to q[A]/([A]+K_A+K_B) and equal-sum pairs agreed to 0.000e+00; the product form keeps the\n'
    'K_A K_B term, which is not a function of the sum, and they separate at 0.7-1.8x the read-out noise. Off-diagonal still\n'
    'separates 1.3-4.6x better than the BEST diagonal design, so symmetry breaking is still worth designing for.\n'
    'RIGHT PANEL: the FRACTION of the extent rises monotonically as concentration falls, which invites "the diagonal is most\n'
    'informative at low concentration". The read-out noise is ABSOLUTE, so what is resolved is the separation in mM, and that\n'
    'peaks MID-BOX and falls at both ends. FALSIFIED BY: any diagonal point returning to the floor, which would mean the\n'
    'K_A K_B term had been dropped from the denominator again.'
  )


def plot_measurements(input_dir, output_dir):
  data = _load(input_dir, 'measurements')
  if data is None:
    return
  rows = data['bound_rows']
  counts = np.array([r[0] for r in rows])
  bound = np.array([r[1] for r in rows])
  measured = np.array([r[2] for r in rows])
  figure, axis = plt.subplots(figsize=(6.4, 3.8))
  axis.plot(
    counts, bound, color=PALETTE[1], marker=MARKERS[1], linewidth=2, markersize=9, label='ceiling  log10(0.8 n)  (arithmetic)'
  )
  axis.plot(
    counts, measured, color=PALETTE[0], marker=MARKERS[0], linewidth=2, markersize=9, label='measured best f_A over the plane'
  )
  axis.axhline(0.90, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axis.text(counts.min(), 0.905, ' protocol requirement 0.90', color=THRESHOLD, fontsize=8)
  axis.axvline(10, color=THRESHOLD, linestyle=':', linewidth=1.2)
  axis.text(
    10.08, 0.04, 'shipped n_measurements = 10', color=THRESHOLD, fontsize=8, rotation=90, va='bottom',
    transform=axis.get_xaxis_transform()
  )
  axis.legend(frameon=False, fontsize=8, loc='lower right')
  _style(
    axis, 'n_measurements', 'best achievable knee visibility for K_A', 'The sampling count is set by arithmetic, not by taste'
  )
  _finish(
    figure, os.path.join(output_dir, 'measurements.png'),
    'SETTLES: n_measurements = 10. t_knee = F(design, K)/q EXACTLY, so a fixed design catches the knee inside\n'
    '[t_1, 0.8T] only for q in a window of ratio 0.8n; against a decade-wide log-uniform q prior that CAPS visibility at\n'
    'log10(0.8n). The measured curve sits on the ceiling, so 8 (=0.806) cannot reach 0.90 and 10 (=0.903) is the\n'
    'smallest count that can. FALSIFIED BY: the measured curve falling below the ceiling, meaning something other than\n'
    'the window is binding, or 8 measurements reaching 0.90.'
  )


def plot_window(input_dir, output_dir, duration):
  data = _load(input_dir, 'window')
  if data is None:
    return
  rows = np.asarray(data['rows'])
  figure, axis = plt.subplots(figsize=(6.4, 3.8))
  for index, (column, label) in enumerate(
    ((1, 'best f_A  (glucose knee)'), (2, 'best f_B  (ATP knee)'), (5, 'coverage: both knees somewhere in the box'))):
    axis.plot(rows[:, 0], rows[:, column], color=PALETTE[index], marker=MARKERS[index], linewidth=2, markersize=8, label=label)
  axis.axhline(0.90, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axis.text(rows[0, 0], 0.905, ' protocol requirement 0.90', color=THRESHOLD, fontsize=8)
  axis.axvline(duration, color=THRESHOLD, linestyle=':', linewidth=1.2)
  axis.text(
    duration * 1.04, 0.04, f'shipped duration = {duration:.0f} s', color=THRESHOLD, fontsize=8, rotation=90, va='bottom',
    transform=axis.get_xaxis_transform()
  )
  axis.set_xscale('log')
  axis.legend(frameon=False, fontsize=8, loc='lower right')
  _style(axis, 'duration (s)', 'fraction of parameter draws', 'The time window: both knees must be reachable')
  _finish(
    figure, os.path.join(output_dir, 'window.png'),
    f'SETTLES: duration = {duration:.0f} s -- the SHORTEST window at which BOTH knees reach 0.90 and coverage is complete\n'
    '(every parameter draw has some design in the box showing each knee). Shorter windows lose the ATP knee first,\n'
    'because K_B is ~5x K_A so B must be depleted further. It is also exactly the one hour of config/detector/enzyme.yaml.\n'
    'FALSIFIED BY: f_B reaching 0.90 at a shorter window, or coverage below 1 at the shipped one.'
  )


def plot_visibility(input_dir, output_dir, box_a, box_b):
  data = _load(input_dir, 'visibility')
  if data is None:
    return
  a_values = np.asarray(data['a_values'])
  b_values = np.asarray(data['b_values'])
  figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
  for axis, name, title in zip(axes, ('f_A', 'f_B', 'both'),
                               ('K_A knee (glucose)', 'K_B knee (ATP)', 'BOTH knees in one experiment')):
    grid = np.asarray(data['grids'][name])
    mesh = axis.pcolormesh(b_values, a_values, grid, cmap='Blues', vmin=0.0, vmax=1.0, shading='nearest')
    contour = axis.contour(
      b_values, a_values, grid, levels=[0.25, 0.90], colors=[THRESHOLD], linewidths=1.4, linestyles=['--', '-']
    )
    axis.clabel(contour, fmt={0.25: '0.25', 0.90: '0.90'}, fontsize=7)
    axis.add_patch(
      plt.Rectangle((box_b[0], box_a[0]), box_b[1] - box_b[0], box_a[1] - box_a[0], fill=False, edgecolor=PALETTE[1],
                    linewidth=2.2, zorder=5)
    )
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_title(title, fontsize=10)
    axis.set_xlabel('B0, initial ATP (mM)')
  axes[0].set_ylabel('A0, initial glucose (mM)')
  figure.colorbar(mesh, ax=axes, label='fraction of parameter draws with the knee inside [t_1, 0.8T]', fraction=0.02)
  _finish(
    figure, os.path.join(output_dir, 'visibility.png'),
    'SETTLES: the design box (orange rectangle), and the decision to score the two knees SEPARATELY. Inside the box every\n'
    'clears the 0.25 contour on at least one knee, and the box contains designs above the 0.90 contour for K_A and for\n'
    'K_B separately. The right panel is why they are scored separately: both knees appear in ONE experiment only in a\n'
    'narrow band on the diagonal, which is exactly where the two constants are exactly degenerate (see degeneracy.png).\n'
    'FALSIFIED BY: a point inside the box below 0.25 on both knees, or no point above 0.90 for either.'
  )


def plot_profile(input_dir, output_dir):
  data = _load(input_dir, 'profile')
  if data is None:
    return
  rows = np.asarray(data['rows'])
  a_values = np.unique(rows[:, 0])
  b_values = np.unique(rows[:, 1])
  figure, axes = plt.subplots(1, 4, figsize=(16.5, 3.9), sharey=True)
  titles = ('total loss', 'q', 'K_A', 'K_B')
  for index, axis in enumerate(axes):
    grid = rows[:, 2 + index].reshape(a_values.size, b_values.size)
    mesh = axis.pcolormesh(b_values, a_values, grid, cmap='Blues', vmin=0.0, vmax=1.0 / 3.0, shading='nearest')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_title(titles[index], fontsize=10)
    axis.set_xlabel('B0 (mM)')
    for i, initial_a in enumerate(a_values):
      for j, initial_b in enumerate(b_values):
        axis.text(
          initial_b, initial_a, f'{grid[i, j]:.2f}', ha='center', va='center', fontsize=6,
          color='white' if grid[i, j] > 0.20 else '#1a1a1a'
        )
  axes[0].set_ylabel('A0 (mM)')
  figure.colorbar(mesh, ax=axes, label='mean squared error (no information = 1/3)', fraction=0.012)
  _finish(
    figure, os.path.join(output_dir, 'profile.png'),
    'SETTLES: what a SINGLE experiment can and cannot do, and hence why the batch is the design. B0 >> A0 saturates ATP\n'
    'and identifies K_A while abandoning K_B; A0 >> B0 does the mirror; the diagonal gets neither. No single (A0, B0)\n'
    'gets both, so a batch must break the A/B symmetry in both directions.\n'
    'FALSIFIED BY: a single cell where K_A and K_B are both far below 1/3 -- then m = 1 would already suffice.'
  )


def plot_integrator(input_dir, output_dir, tolerance, shipped):
  data = _load(input_dir, 'integrator')
  if data is None:
    return
  rows = np.asarray(data['rows'])
  figure, axis = plt.subplots(figsize=(6.6, 4.0))
  for index, n_stages in enumerate(sorted(set(rows[:, 0]))):
    chosen = rows[rows[:, 0] == n_stages]
    order = np.argsort(chosen[:, 1])
    axis.plot(
      chosen[order, 1], chosen[order, 3], color=PALETTE[index], marker=MARKERS[index], linewidth=2, markersize=8,
      label=f'{int(n_stages)} stages: true error vs closed form'
    )
    axis.plot(
      chosen[order, 1], chosen[order, 4], color=PALETTE[index], marker=MARKERS[index], linewidth=1.2, markersize=5,
      linestyle=':', alpha=0.75, label=f'{int(n_stages)} stages: dt vs dt/2 monitor'
    )
  axis.axhline(tolerance, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axis.text(rows[:, 1].min(), tolerance * 1.3, f' integration_tolerance {tolerance:g} mM', color=THRESHOLD, fontsize=8)
  axis.axvline(shipped, color=THRESHOLD, linestyle=':', linewidth=1.2)
  axis.text(
    shipped * 1.05, 0.04, f'shipped steps = {shipped}', color=THRESHOLD, fontsize=8, rotation=90, va='bottom',
    transform=axis.get_xaxis_transform()
  )
  axis.set_xscale('log')
  axis.set_yscale('log')
  axis.legend(frameon=False, fontsize=7, ncol=2, loc='lower left')
  _style(
    axis, 'steps per measurement', 'worst error over the prior box (mM)',
    'RKC2 against the closed form, over the whole (A0, B0, q, K_A, K_B) box'
  )
  _finish(
    figure, os.path.join(output_dir, 'integrator.png'),
    'SETTLES: n_stages and steps_per_measurement, and that the dt-vs-dt/2 MONITOR the detector asserts on is a valid\n'
    'proxy for the true error (dotted tracks solid). The shipped setting is the cheapest that clears the tolerance over\n'
    'the whole box, not just at corners.\n'
    'FALSIFIED BY: the dotted monitor falling below the solid true error -- the assert would then pass on a wrong answer.'
  )


def plot_estimator(input_dir, output_dir):
  data = _load(input_dir, 'estimator')
  if data is None:
    return
  rows = np.asarray(data['grid_rows'])
  agreement = np.asarray(data['agreement'])
  if rows.ndim != 2 or rows.shape[1] < 4:
    return  # an older run that recorded only the plateau; the discrimination is what this plots
  figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
  axes[0].plot(rows[:, 0], rows[:, 2], color=PALETTE[0], marker=MARKERS[0], linewidth=2, markersize=9, label='a strong batch')
  axes[0].plot(rows[:, 0], rows[:, 3], color=PALETTE[1], marker=MARKERS[1], linewidth=2, markersize=9, label='a weak batch')
  axes[0].legend(frameon=False, fontsize=8)
  _style(axes[0], 'n_grid on each Michaelis axis', 'loss', 'A plateau proves nothing on its own')
  width = 0.35
  index = np.arange(agreement.shape[0])
  axes[1].bar(index - width / 2, agreement[:, 1], width, color=PALETTE[0], label='detector: RKC2, float32')
  axes[1].bar(index + width / 2, agreement[:, 2], width, color=PALETTE[2], label='reference: closed form, float64')
  axes[1].set_xticks(index)
  axes[1].set_xticklabels([f'm = {int(v)}' for v in agreement[:, 0]])
  axes[1].legend(frameon=False, fontsize=8)
  for position, (detector_value, closed_value) in zip(index, agreement[:, 1:]):
    axes[1].text(
      position,
      max(detector_value, closed_value) * 1.02, f'{abs(detector_value - closed_value):.4f}', ha='center', fontsize=7,
      color='#3d3d3d'
    )
  _style(axes[1], '', 'loss', 'The shipped instrument against an integration-error-free twin')
  _finish(
    figure, os.path.join(output_dir, 'estimator.png'),
    'SETTLES: that the landscape reports INFORMATION rather than integration error, and that a PLATEAU IS NOT EVIDENCE.\n'
    'LEFT: refining the Michaelis axes flattens out at every size tried -- including sizes the Bayes-bound check in\n'
    'resolution.png rejects outright -- because a quantised posterior stops moving just as happily as a converged one.\n'
    'The lattice is settled by resolution.png, not here. RIGHT: the same posterior built from the float64 closed form\n'
    'agrees with the shipped float32/RKC2 one (gap printed above each pair), so the objective is not reporting\n'
    'integration error. FALSIFIED BY: a right-hand gap comparable to the design-to-design span in landscape.png.'
  )


def plot_resolution(input_dir, output_dir, shipped_grid, shipped_noise):
  data = _load(input_dir, 'resolution')
  if data is None:
    return
  rows = np.asarray(data['rows'])
  if rows.ndim != 2 or rows.shape[1] < 6:
    return  # an older run that swept only isotropic lattices, before the anisotropy was found
  figure, axis = plt.subplots(figsize=(7.4, 4.4))
  keys = sorted({(int(r[0]), int(r[1])) for r in rows}, key=lambda k: (k[0] == k[1], k))
  for index, (n_velocity, n_grid) in enumerate(keys):
    chosen = rows[(rows[:, 0] == n_velocity) & (rows[:, 1] == n_grid)]
    order = np.argsort(chosen[:, 3])
    isotropic = n_velocity == n_grid
    axis.plot(
      chosen[order, 3], chosen[order, 5], color=PALETTE[index % len(PALETTE)], marker=MARKERS[index % len(MARKERS)],
      linewidth=2 if not isotropic else 1.4, markersize=8, linestyle='--' if isotropic else '-',
      label=f'({n_velocity}, {n_grid}, {n_grid}) = {int(chosen[0, 2])} nodes' +
      ('  [isotropic, rejected]' if isotropic else '')
    )
  axis.axhline(1.0 / 3.0, color=THRESHOLD, linestyle='--', linewidth=1.6)
  axis.text(
    0.02, 1.0 / 3.0 * 1.03, 'prior variance 1/3 -- a Bayes estimator CANNOT exceed this', color=THRESHOLD, fontsize=8,
    transform=axis.get_yaxis_transform()
  )
  axis.axvline(shipped_noise, color=THRESHOLD, linestyle=':', linewidth=1.2)
  axis.text(
    shipped_noise * 1.04, 0.04, f'shipped noise = {shipped_noise:g}', color=THRESHOLD, fontsize=8, rotation=90, va='bottom',
    transform=axis.get_xaxis_transform()
  )
  axis.set_xscale('log')
  axis.legend(frameon=False, fontsize=7, ncol=2)
  _style(
    axis, 'measurement_noise (mM)', 'worst PER-PARAMETER loss over sampled designs',
    'Is the lattice fine enough to be a Bayes estimator at all?'
  )
  _finish(
    figure, os.path.join(output_dir, 'resolution.png'),
    'SETTLES: n_grid and measurement_noise JOINTLY. The posterior mean is the Bayes estimator, so its risk cannot exceed\n'
    'the prior variance -- any point above the dashed line is PROOF that the lattice is too coarse, needing no reference\n'
    'value and no extrapolation. At high signal-to-noise the posterior is narrower than the grid spacing, the softmax\n'
    'collapses onto one node, and the estimate is worse than a guess. This is the trap the sibling task hit with a\n'
    'closed-form estimator. FALSIFIED BY: the shipped (noise, n_grid) pair sitting above the line at any design.'
  )


def plot_parameterisation(input_dir, output_dir):
  data = _load(input_dir, 'parameterisation')
  if data is None:
    return
  figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
  for index, (name, record) in enumerate(data.items()):
    centres = np.asarray(record['centres'])
    axes[0].plot(
      centres,
      np.asarray(record['value']) / np.asarray(record['value']).min(), color=PALETTE[index], marker=MARKERS[index], linewidth=2,
      markersize=8, label=name
    )
    axes[1].plot(
      centres,
      np.asarray(record['log']) / np.asarray(record['log']).min(), color=PALETTE[index], marker=MARKERS[index], linewidth=2,
      markersize=8, label=name
    )
  for axis, title in zip(axes, ('error on the VALUE', 'error on the LOG')):
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.axhline(1.0, color=THRESHOLD, linestyle='--', linewidth=1.2)
    axis.legend(frameon=False, fontsize=8)
    _style(axis, 'true parameter value (own units)', 'RMSE in the bin, / the smallest bin', title)
  spreads = ',  '.join(
    f'{name}: value {max(record["value"]) / min(record["value"]):.1f}x, log {max(record["log"]) / min(record["log"]):.1f}x'
    for name, record in data.items()
  )
  _finish(
    figure, os.path.join(output_dir, 'parameterisation.png'),
    'SETTLES: the target is (ln q, ln K_A, ln K_B) and not the raw values. Binned by the TRUE value, the error on the\n'
    'VALUE climbs with the parameter across decades while the error on the LOG varies less, so the log is the coordinate\n'
    'on which the estimator is closer to scale-free and a squared-error loss there weights the prior more evenly.\n'
    f'Spread across bins (max / min) -- {spreads}.\n'
    'FALSIFIED BY: the log spread exceeding the value spread for any parameter. (The U shape on the log axis is the\n'
    'estimator shrinking toward the prior at the ends of the range, not a failure of the parameterisation.)'
  )


def plot_landscape(input_dir, output_dir, shipped_noise):
  import glob

  rows, precision = [], None
  for path in sorted(glob.glob(os.path.join(input_dir, 'landscape*.json'))):
    with open(path) as handle:
      data = json.load(handle)
    rows.extend(data['rows'])
    precision = data['loss_precision']
  if len(rows) == 0:
    return
  counts = sorted({r['n_experiments'] for r in rows})
  figure, axes = plt.subplots(1, len(counts), figsize=(4.6 * len(counts), 4.2), sharey=True, squeeze=False)
  for axis, n_experiments in zip(axes[0], counts):
    chosen = [r for r in rows if r['n_experiments'] == n_experiments]
    for index, record in enumerate(chosen):
      values = np.asarray(record['values'])
      jitter = np.random.default_rng(index).normal(0.0, 0.06, values.size)
      axis.scatter(
        np.full(values.size, index) + jitter, values, s=12, alpha=0.55,
        color=PALETTE[0] if record['noise'] != shipped_noise else PALETTE[1], zorder=3
      )
      axis.plot([index - 0.3, index + 0.3], [np.median(values)] * 2, color=THRESHOLD, linewidth=2, zorder=4)
      axis.plot([index - 0.3, index + 0.3], [values.min()] * 2, color=THRESHOLD, linewidth=1, linestyle='--', zorder=4)
    axis.set_xticks(range(len(chosen)))
    axis.set_xticklabels([f'{r["noise"]:g}' for r in chosen])
    axis.axhline(1.0 / 3.0, color=THRESHOLD, linestyle='-.', linewidth=1.2)
    axis.axhline(0.95 / 3.0, color=THRESHOLD, linestyle=':', linewidth=1.0)
    axis.text(-0.4, 1.0 / 3.0 * 1.01, 'no information', color=THRESHOLD, fontsize=8)
    _style(axis, 'measurement_noise (mM)', 'loss over random designs', f'm = {n_experiments} experiments')
  _finish(
    figure, os.path.join(output_dir, 'landscape.png'),
    f'SETTLES: measurement_noise, and whether the task can be RESOLVED at all. The bar the optimiser must clear is the\n'
    f'absolute gap between median (solid bar) and best (dashed) against loss_precision = {precision:g}; the shipped noise is\n'
    'orange. A noise that squeezes the scatter onto\n'
    'the no-information line is hopeless and one that collapses it to a point is trivial. Scatter is shown, not just\n'
    'central tendency, because the claim is about a span. FALSIFIED BY: median-minus-best of order loss_precision.'
  )

  figure, axes = plt.subplots(1, len(counts), figsize=(4.6 * len(counts), 4.0), sharey=True, squeeze=False)
  for axis, n_experiments in zip(axes[0], counts):
    chosen = [r for r in rows if r['n_experiments'] == n_experiments and r['noise'] == shipped_noise]
    if len(chosen) == 0:
      continue
    record = chosen[0]
    for index, (key, label) in enumerate(((('values'), 'measured loss profile'), ('quadratic', '||x - x*||^2 reference'))):
      values = np.sort(np.asarray(record[key]))
      normalised = (values - values.min()) / max(values.max() - values.min(), 1e-12)
      axis.plot(
        normalised, np.linspace(0.0, 1.0, values.size), color=PALETTE[index], marker=MARKERS[index], markersize=4, linewidth=2,
        label=label
      )
    axis.legend(frameon=False, fontsize=8, loc='lower right')
    _style(axis, 'min-max normalised value', 'empirical CDF over random designs', f'm = {n_experiments} experiments')
  _finish(
    figure, os.path.join(output_dir, 'shape.png'),
    'SETTLES: the objective has the SHAPE of a bowl rather than a plateau or a needle. The reference is ||x - x*||^2 for\n'
    'x uniform on the same scaled cube with x* at the best design found -- what a clean quadratic would give. A curve\n'
    'hugging the left edge is a needle (nearly every design as good as the best, so luck beats modelling); one hugging\n'
    'the right is dead space. FALSIFIED BY: the measured CDF departing far from the quadratic reference.'
  )


def plot_traces(output_dir, criterion_files):
  """Best-so-far against DESIGN INDEX, one thin line per independent run, per arm.

  THE CHEAP SCREEN, and the one that kills a bad setting fastest. A needle-on-a-plateau objective
  shows a drop in the first few designs and then a dead-flat incumbent: a lucky early draw found the
  needle and nothing afterwards improves, because there is no gradient to follow. On such a task an
  arm comparison measures which arm's initial random designs happened to land well. A quadratic-like
  bowl instead keeps yielding improvements, which is exactly what makes doubling the budget pay."""
  for path in criterion_files:
    if not os.path.exists(path):
      continue
    with open(path) as handle:
      payload = json.load(handle)
    arms = list(payload['arms'])
    figure, axes = plt.subplots(1, len(arms), figsize=(5.4 * len(arms), 4.2), sharey=True, squeeze=False)
    flat_fraction = {}
    for index, (axis, arm) in enumerate(zip(axes[0], arms)):
      curves = np.asarray(payload['arms'][arm]['observed'])
      best = np.minimum.accumulate(curves, axis=1)
      for row in best:
        axis.plot(np.arange(1, row.size + 1), row, color=PALETTE[index], linewidth=1.0, alpha=0.45)
      axis.plot(
        np.arange(1, best.shape[1] + 1), np.median(best, axis=0), color=THRESHOLD, linewidth=2.5, label='median over runs'
      )
      half = max(best.shape[1] // 2, 1)
      flat_fraction[arm] = float(np.mean(best[:, half - 1] <= best[:, -1] + 1e-12))
      axis.axvline(half, color=THRESHOLD, linestyle=':', linewidth=1.2)
      axis.text(
        half * 1.03, 0.04, f'half the budget ({half})', color=THRESHOLD, fontsize=8, rotation=90, va='bottom',
        transform=axis.get_xaxis_transform()
      )
      axis.set_yscale('log')
      axis.legend(frameon=False, fontsize=8)
      _style(axis, 'design index', 'best-so-far loss', f'{arm}  ({curves.shape[0]} independent runs)')
    summary = ',  '.join(f'{arm} {value:.2f}' for arm, value in flat_fraction.items())
    tag = os.path.splitext(os.path.basename(path))[0]
    _finish(
      figure, os.path.join(output_dir, f'traces-{tag}.png'),
      'SETTLES: whether the objective is a smooth bowl or a NEEDLE ON A PLATEAU. Each thin line is one independent run;\n'
      'the incumbent must keep descending past half the budget rather than flat-lining after an early lucky draw.\n'
      f'Fraction of runs whose incumbent NEVER improves after half the budget: {summary} -- high values mean the arm\n'
      'comparison would be measuring luck. FALSIFIED BY: flat lines from an early design index in any arm.'
    )


def plot_criterion(input_dir, output_dir, criterion_files):
  for path in criterion_files:
    if not os.path.exists(path):
      continue
    with open(path) as handle:
      payload = json.load(handle)
    checkpoints = payload['checkpoints']
    doubles = [(a, b) for a in checkpoints for b in checkpoints if b == 2 * a]
    if len(doubles) == 0:
      continue
    early, late = doubles[-1]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    summary = []
    for index, (arm, data) in enumerate(payload['arms'].items()):
      first = np.asarray(data['reevaluated'][str(early)])
      second = np.asarray(data['reevaluated'][str(late)])
      axes[0].scatter(first, second, s=34, alpha=0.75, color=PALETTE[index], marker=MARKERS[index], label=arm, zorder=3)
      summary.append((arm, float(np.mean(second < first)), first.mean() - second.mean()))
    limits = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]), max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
    axes[0].plot(limits, limits, color=THRESHOLD, linestyle='--', linewidth=1.4)
    axes[0].text(limits[0], limits[1], ' below the line = doubling helped', color=THRESHOLD, fontsize=8, va='top')
    axes[0].legend(frameon=False, fontsize=8)
    _style(axes[0], f'loss @ {early} iterations', f'loss @ {late} iterations', 'Paired WITHIN a run (monotone by construction)')

    for index, (arm, data) in enumerate(payload['arms'].items()):
      values = [np.asarray(data['reevaluated'][str(c)]) for c in checkpoints]
      means = [v.mean() for v in values]
      errors = [v.std() / math.sqrt(v.size) for v in values]
      axes[1].errorbar(
        checkpoints, means, yerr=errors, color=PALETTE[index], marker=MARKERS[index], linewidth=2, markersize=8, capsize=3,
        label=arm
      )
    axes[1].legend(frameon=False, fontsize=8)
    _style(axes[1], 'BO iterations', 'E[re-evaluated incumbent loss]', 'BO against the random-search null')
    tag = os.path.splitext(os.path.basename(path))[0]
    caption = (
      f'SETTLES: whether doubling the iteration budget pays RELIABLY (m = {payload["n_experiments"]}). '
      f'P(loss@{late} < loss@{early}) over ordered pairs of\nINDEPENDENT runs: ' +
      ',  '.join(f'{arm} {probability:.3f} (gap {gap:+.4f})' for arm, probability, gap in summary) + '.\n'
      'The left panel is the WITHIN-run view and is monotone by construction, so it is shown only to expose the spread; '
      'the number\nquoted is the between-run one. FALSIFIED BY: BO not exceeding the random null, or P near 0.5.'
    )
    _finish(figure, os.path.join(output_dir, f'criterion-{tag}.png'), caption)


def plot_identifiability(input_dir, output_dir):
  """THE NOISE DECISION. Per-parameter, because the aggregate loss cannot see a shed target."""
  record = _load(input_dir, 'identifiability')
  if record is None:
    return
  rows = record['rows']
  shipped = [r for r in rows if (r['n_grid_velocity'], r['n_grid']) == (241, 15)]
  finer = [r for r in rows if (r['n_grid_velocity'], r['n_grid']) == (241, 21)]
  noises = [r['noise'] for r in shipped]
  figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

  names = ('ln q', 'ln K_A', 'ln K_B')
  for index, name in enumerate(names):
    axes[0].plot(
      noises, [100.0 * r['recovered'][index] for r in shipped], marker=MARKERS[index], color=PALETTE[index], linewidth=2,
      markersize=7, label=name
    )
  axes[0].axhline(25.0, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axes[0].annotate(
    '25% bar on K_A', (noises[-1], 25.0), textcoords='offset points', xytext=(-4, 6), ha='right', fontsize=8, color=THRESHOLD
  )
  axes[0].axvline(0.02, color=PALETTE[4], linestyle=':', linewidth=2)
  axes[0].annotate('nominated 0.02', (0.02, 92.0), textcoords='offset points', xytext=(5, 0), fontsize=8, color=PALETTE[4])
  axes[0].axvline(0.05, color=PALETTE[1], linestyle=':', linewidth=1.5)
  axes[0].annotate('0.05 REJECTED', (0.05, 60.0), textcoords='offset points', xytext=(5, 0), fontsize=8, color=PALETTE[1])
  axes[0].set_xscale('log')
  _style(axes[0], 'read-out noise (mM)', '% of prior VARIANCE removed', 'What each parameter still retains')
  axes[0].legend(fontsize=9)

  # Any noise whose worst per-parameter loss breaches 1/3 is quantisation, not information, so the
  # panels that read as a noise result must say where they stop being one.
  unusable = [r['noise'] for r in shipped if r['worst_part'] > record['no_information']]
  for panel in (0, 1):
    for noise in unusable:
      axes[panel].axvspan(noise * 0.86, noise * 1.16, color='#bdbdbd', alpha=0.35, zorder=0)
      axes[panel].annotate(
        'lattice-limited', (noise, axes[panel].get_ylim()[0]), textcoords='offset points', xytext=(0, 8), ha='center',
        rotation=90, fontsize=7, color='#4d4d4d'
      )

  prior_sd = record['ln_range'][1] / math.sqrt(12.0)
  axes[1].plot([r['noise'] for r in shipped], [r['sd_ln_michaelis_a_mean'] for r in shipped], marker='o', color=PALETTE[1],
               linewidth=2, markersize=7, label='average design')
  axes[1].plot([r['noise'] for r in shipped], [r['sd_ln_michaelis_a_best'] for r in shipped], marker='s', color=PALETTE[2],
               linewidth=2, markersize=7, label='best design in box')
  axes[1].axhline(prior_sd, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axes[1].annotate(
    f'prior sd = {prior_sd:.3f} (measuring nothing)', (noises[-1], prior_sd), textcoords='offset points', xytext=(-4, -12),
    ha='right', fontsize=8, color=THRESHOLD
  )
  axes[1].axvline(0.02, color=PALETTE[4], linestyle=':', linewidth=2)
  axes[1].set_xscale('log')
  _style(axes[1], 'read-out noise (mM)', 'posterior sd(ln K_A), nats', 'K_A is the binding parameter')
  axes[1].legend(fontsize=9)

  axes[2].plot(
    noises, [r['worst_part'] for r in shipped], marker='o', color=PALETTE[0], linewidth=2, markersize=7,
    label='lattice (241, 15, 15) -- shipped'
  )
  axes[2].plot([r['noise'] for r in finer], [r['worst_part'] for r in finer], marker='s', color=PALETTE[3], linewidth=2,
               markersize=7, linestyle='--', label='lattice (241, 21, 21)')
  axes[2].axhline(record['no_information'], color=THRESHOLD, linestyle='--', linewidth=1.5)
  axes[2].annotate(
    '1/3 = the prior; a Bayes estimator CANNOT exceed it', (noises[-1], record['no_information']), textcoords='offset points',
    xytext=(-4, 5), ha='right', fontsize=8, color=THRESHOLD
  )
  axes[2].set_xscale('log')
  _style(axes[2], 'read-out noise (mM)', 'worst PER-PARAMETER loss', 'Above 1/3 is a LATTICE verdict, not a noise one')
  axes[2].legend(fontsize=9, loc='upper right')

  figure.tight_layout()
  _finish(
    figure, os.path.join(output_dir, 'identifiability.png'),
    'SETTLES `measurement_noise`, and it is set LAST and from THIS. A shape criterion can be GAMED by shedding a target:\n'
    'raising the noise cleans the landscape over (q, K_B) while K_A quietly stops being measured, and span / plateau /\n'
    'near-best are computed on the AGGREGATE and cannot see it. At 0.05 the average design removes only 12.8% of the K_A\n'
    'prior variance (sd 0.621 against the prior 0.665) -- a three-parameter task identifying two. 0.02 keeps 25.2%.\n'
    'The two lattices agree everywhere at noise >= 0.01, so the shipped (241, 15, 15) is not the limiting factor; at 0.005\n'
    'BOTH breach 1/3, which is why lower noise is not free. FALSIFIED BY: K_A recovery at 0.05 rising to 0.02 levels at\n'
    'm = 2, or the finer lattice separating from the shipped one at the nominated noise.'
  )


def plot_box(input_dir, output_dir):
  """THE BOX DECISION. Dead space = plateau; tight = the LARGEST box with no plateau."""
  record = _load(input_dir, 'box')
  if record is None:
    return
  profiles, boxes = record['profiles'], record['boxes']
  # The face-gradient panel is drawn only when that (much slower) half of the section has run; the
  # line profiles alone already locate the plateaus and are worth plotting on their own.
  n_panels = 3 if len(boxes) > 0 else 2
  figure, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.0))

  for panel, axis_name in ((0, 'A0'), (1, 'B0')):
    selected = [p for p in profiles if p['axis'] == axis_name]
    for index, profile in enumerate(selected):
      other = 'B0' if axis_name == 'A0' else 'A0'
      axes[panel].plot(
        profile['sweep'], profile['loss'], marker=MARKERS[index % len(MARKERS)], color=PALETTE[index % len(PALETTE)],
        linewidth=2, markersize=5, label=f'{other} = {profile["fixed"]:g} mM'
      )
    shipped = (0.8, 10.0) if axis_name == 'A0' else (1.0, 25.0)
    for edge in shipped:
      axes[panel].axvline(edge, color=THRESHOLD, linestyle=':', linewidth=1.5)
    axes[panel].annotate(
      'shipped box', (shipped[1], axes[panel].get_ylim()[1]), textcoords='offset points', xytext=(4, -12), fontsize=8,
      color=THRESHOLD
    )
    axes[panel].set_xscale('log')
    _style(axes[panel], f'{axis_name} (mM)', 'loss at m = 1', f'Loss along {axis_name}: flat ends ARE the dead space')
    axes[panel].legend(fontsize=8)

  if len(boxes) == 0:
    figure.tight_layout()
    _finish(
      figure, os.path.join(output_dir, 'box.png'),
      'SETTLES where the loss STOPS RESPONDING to the design, which is what dead space means: DEAD SPACE = PLATEAU, and a\n'
      'TIGHT box is the MAXIMAL box containing none. Both ends are flat for a reason: below A0 ~ 0.15 the extent is lost in\n'
      'the read-out noise and the loss sits AT the 1/3 prior; once one substrate exceeds the other by ~3x it saturates the\n'
      'enzyme, its own constant stops being identifiable, and moving it further changes nothing. So the ALIVE region is a\n'
      'BAND AROUND THE DIAGONAL -- which INVERTS the original box, built to AVOID the diagonal under the ping-pong\n'
      'degeneracy that no longer exists. NOTE the flat ends sit at loss 0.14-0.23, far BELOW 1/3, so the landscape screen\'s\n'
      '`plateau` statistic (designs within 5% of 1/3) counts NONE of them: "no design scores no-information" and "no region\n'
      'is flat" are different claims. FALSIFIED BY: a face of the nominated box showing a gradient at the level of these\n'
      'flat ends (bound too far out), or a healthy gradient at the outermost point swept (box can still grow).'
    )
    return

  labels, width = [], 0.2
  for index, block in enumerate(boxes):
    faces = block['faces']
    names = list(faces)
    positions = np.arange(len(names)) + (index - 0.5 * (len(boxes) - 1)) * width
    axes[2].bar(
      positions, [100.0 * faces[n]['fraction_of_span'] for n in names], width=width * 0.9, color=PALETTE[index % len(PALETTE)],
      label=f'A0 {block["box_a"]} x B0 {block["box_b"]}  (span {block["span"]:.3f})'
    )
    labels = names
  axes[2].axhline(5.0, color=THRESHOLD, linestyle='--', linewidth=1.5)
  axes[2].annotate(
    'below 5% of span = PLATEAU, bound too far out', (len(labels) - 0.5, 5.0), textcoords='offset points', xytext=(-4, 5),
    ha='right', fontsize=8, color=THRESHOLD
  )
  axes[2].set_xticks(np.arange(len(labels)))
  axes[2].set_xticklabels(labels)
  _style(axes[2], 'face of the box', 'mean |d loss| stepping inward, % of span', 'Face gradients: is this face alive?')
  axes[2].legend(fontsize=7, loc='upper right')

  figure.tight_layout()
  _finish(
    figure, os.path.join(output_dir, 'box.png'),
    'SETTLES the design box. DEAD SPACE = PLATEAU (moving the design does not change the loss); a TIGHT box is the\n'
    'MAXIMAL box containing no plateau, so the bounds are found by EXPANDING outward until the loss goes flat and then\n'
    'backing off -- not by shrinking to something safe. A too-SMALL box is a defect: it compresses the loss range and\n'
    'makes the task needle-like. The right panel measures the definition directly, by stepping each face inward.\n'
    'FALSIFIED BY: a face gradient near zero inside the nominated box (bound too far out), or a healthy gradient at the\n'
    'outermost box tried (the sweep never reached the plateau and the box can still grow).'
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', default='output/enzyme-depletion-bi')
  parser.add_argument('--output-dir', default=None)
  parser.add_argument('--integration-tolerance', type=float, default=5.0e-3)
  parser.add_argument('--steps-per-measurement', type=int, default=64)
  parser.add_argument('--duration', type=float, default=3600.0)
  parser.add_argument('--noise', type=float, default=0.02)
  parser.add_argument('--n-grid', type=int, default=31)
  parser.add_argument('--box-a', type=float, nargs=2, default=(0.8, 10.0))
  parser.add_argument('--box-b', type=float, nargs=2, default=(1.0, 25.0))
  parser.add_argument('--criterion', nargs='*', default=[])
  arguments = parser.parse_args()
  output_dir = arguments.output_dir or os.path.join(arguments.input_dir, 'figures')
  os.makedirs(output_dir, exist_ok=True)
  print('figures:')
  plot_reference(arguments.input_dir, output_dir, arguments.integration_tolerance)
  plot_degeneracy(arguments.input_dir, output_dir)
  plot_measurements(arguments.input_dir, output_dir)
  plot_window(arguments.input_dir, output_dir, arguments.duration)
  plot_visibility(arguments.input_dir, output_dir, arguments.box_a, arguments.box_b)
  plot_profile(arguments.input_dir, output_dir)
  plot_integrator(arguments.input_dir, output_dir, arguments.integration_tolerance, arguments.steps_per_measurement)
  plot_estimator(arguments.input_dir, output_dir)
  plot_resolution(arguments.input_dir, output_dir, arguments.n_grid, arguments.noise)
  plot_parameterisation(arguments.input_dir, output_dir)
  plot_landscape(arguments.input_dir, output_dir, arguments.noise)
  plot_identifiability(arguments.input_dir, output_dir)
  plot_box(arguments.input_dir, output_dir)
  plot_criterion(arguments.input_dir, output_dir, arguments.criterion)
  plot_traces(output_dir, arguments.criterion)


if __name__ == '__main__':
  main()
