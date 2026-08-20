#!/usr/bin/env python3
"""Calibrate the `pixel_density` task: does the sigma range have two BAD ends and a good middle?

    python scripts/probe_pixel_density.py --stage geometry landscape plots \
        --output-dir output/pixel-density
    python scripts/probe_pixel_density.py --stage sobol --output-dir output/pixel-density

FOUR STAGES, and each says what it must be able to separate before it is read.

`geometry` -- no estimator at all, pure counting over the frames. THE FIELD OF VIEW IS FIXED: the
    grid spans the whole frame at every design (the corner probes are pinned to the image's edges),
    so the share of probes inside the frame is identically 1 and cannot be a symptom of anything.
    What moves is the probe DENSITY, and the stage must separate the two ends:
      * the CENTRE-to-EDGE spacing ratio, which orders the whole design space and runs from ~0.07 at
        the small end to ~1 (uniform) at the large one. It is the guard against a parameterisation
        that saturates: it must keep moving over the whole range.
      * the ink-weighted distance from an inked pixel to the NEAREST probe, in pixels, and the worst
        such distance over the inked support. Small sigma fails here: the interior probes crowd into
        the middle, the digit's extremities are left to the two pinned corner probes, and the frame
        the network sees is a magnified crop.
    ⚠️ CAPTURED INK IS A TRAP and is reported only to be dismissed: it is monotone DECREASING in
    sigma, because crowding the probes onto the dense central stroke maximises the intensity read
    out while destroying the shape. The smallest sigma scores the most ink and is unreadable. Only
    the classification loss locates the optimum.

`landscape` -- the objective's shape over the design space, on the gradient-boosted proxy of
    `detopt/bo/gbdt.py` (the repo's screening estimator: same recipe, same stopping rule, same
    `(train + val) / 2` at the validation minimum). It must separate three things: the two ends from
    the middle, one design from its neighbour by more than the proxy's own error bar, and a bowl
    from a plateau. Cross-entropy in nats against the no-information level `ln 10`. A proxy decides
    where the interesting designs are; it does not replace the neural objective, and the
    `loss_precision` a campaign runs at must come from the trainer itself
    (`scripts/probe_precision.py`), never from here. The proxy scores a class target in units of
    `ln K`; every number recorded and plotted here is multiplied back into NATS, so it is on the same
    scale as `detector.loss` and as the `ln 10` reference.

`sobol` -- the same proxy at UNIFORMLY drawn designs rather than on a lattice, because the lattice
    says what the surface looks like and a uniform draw says what a RANDOM design costs -- which is
    the distribution a run's `n_init` starts in. It must separate a climbable bowl from a plateau:
    reported as the sorted cost curve and as the held-out R2 of a GP fitted to those designs alone.

`plots` -- the loss profile and what the grid actually looks like at a small, a good and a large
    sigma. Reads what the other stages wrote.

The proxy needs the 100 sampled values as a flat read-out, which is what `_ProxyReadout` presents:
the detector's own combine, channel 0, under the name `detopt.bo.gbdt` expects. The design planes are
deliberately dropped -- inside one design they are constant and carry nothing.
"""

import argparse
import json
import os
import time
from typing import NamedTuple

_allocated = os.environ.get('SLURM_CPUS_PER_TASK', '2')
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
  os.environ.setdefault(_variable, _allocated)
os.environ.setdefault('XLA_FLAGS', f'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={_allocated}')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax.numpy as jnp

import detopt.detector
import detopt.utils.config
from detopt.detector.mnist import MNISTEvent

BLUE, VERMILLION, GREEN, ORANGE, PURPLE = '#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7'
GRID = {'color': '#D9D9D9', 'linewidth': 0.6}
INK, MUTED = '#1A1A1A', '#666666'
NO_INFORMATION = float(np.log(10.0))


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


class _Readout(NamedTuple):
  """The 100 sampled values under the name the gradient-boosted proxy reads."""
  measurements: np.ndarray


class _ProxyReadout:
  """A shim presenting the sampled grid as a flat read-out, so `detopt.bo.gbdt.score_design` scores
  this task by exactly the recipe it scores every other candidate by. Holds no design of its own."""

  def __init__(self, detector):
    self.detector = detector
    self.n_classes = int(detector.n_classes)

  def __call__(self, design, event_index):
    ground_truth, event, mask, target = self.detector(design, event_index)
    samples = self.detector.combine(event, design, mask=mask)[..., 0]
    return ground_truth, _Readout(measurements=np.asarray(samples, np.float32)), mask, target

  def normalize_target(self, target):
    return self.detector.normalize_target(target)


def build(config_path, overrides=()):
  config = detopt.utils.config.override(detopt.utils.config.load_config(config_path), list(overrides))
  return detopt.detector.from_config(config)


def unit_frame(positions):
  """Probe positions from the detector's symmetric ``[-1, 1]`` frame onto ``[0, 1]``, where pixel
  indices and ink statistics are written."""
  return 0.5 * (np.asarray(positions, np.float64) + 1.0)


def sigma_ladder(detector, n):
  return np.exp(np.linspace(np.log(detector.sigma_min), np.log(detector.sigma_max), int(n)))


def ink_statistics(detector, n_frames):
  """The dataset's ink as a `(n_rows, n_columns)` mean frame, plus its support mask."""
  frames = np.asarray(detector.images[:int(min(n_frames, detector.size()))], np.float32) / 255.0
  mean_ink = frames.mean(axis=0)
  return mean_ink, mean_ink > 0.01 * mean_ink.max()


def geometry(detector, n_sigma, n_frames):
  """What the probe density covers and how finely, at each sigma, with no estimator involved."""
  mean_ink, support = ink_statistics(detector, n_frames)
  rows = (np.arange(detector.n_rows) + 0.5) / detector.n_rows
  columns = (np.arange(detector.n_columns) + 0.5) / detector.n_columns
  total_ink = float(mean_ink.sum())
  frame = MNISTEvent(image=jnp.asarray(mean_ink * 255.0, jnp.float32))
  middle = detector.n_grid // 2
  records = []
  for sigma in sigma_ladder(detector, n_sigma):
    design = {'sigma_x': float(sigma), 'sigma_y': float(sigma)}
    x, y = [unit_frame(a) for a in detector.sample_positions(design)]
    interior = ((columns >= x[1]) & (columns <= x[-2]))[None, :] & ((rows >= y[1]) & (rows <= y[-2]))[:, None]
    distance_columns = np.abs(columns[:, None] - x[None, :]).min(axis=1) * detector.n_columns
    distance_rows = np.abs(rows[:, None] - y[None, :]).min(axis=1) * detector.n_rows
    distance = np.hypot(distance_rows[:, None], distance_columns[None, :])
    on_support = support[np.clip((y * detector.n_rows).astype(int), 0, detector.n_rows - 1)[:, None],
                         np.clip((x * detector.n_columns).astype(int), 0, detector.n_columns - 1)[None, :]]
    spacing = np.diff(x) * detector.n_columns
    records.append({
      'sigma': float(sigma),
      'probes_inside_frame': float(np.mean((x >= 0.0) & (x <= 1.0))),
      'ink_inside_the_interior_probes': float((mean_ink * interior).sum() / total_ink),
      'ink_weighted_nearest_probe_pixels': float((mean_ink * distance).sum() / total_ink),
      'worst_ink_nearest_probe_pixels': float(distance[support].max()),
      'probes_on_inked_support': float(np.mean(on_support)),
      'central_spacing_pixels': float(spacing[middle - 1]),
      'edge_spacing_pixels': float(spacing[0]),
      'centre_to_edge_ratio': float(spacing[middle - 1] / spacing[0]),
      'captured_ink_fraction': float(np.sum(np.asarray(detector.combine(frame, design)[..., 0])) / total_ink),
    })
    print(
      f"  sigma={sigma:7.4f}  ctr/edge={records[-1]['centre_to_edge_ratio']:.3f}  "
      f"central={records[-1]['central_spacing_pixels']:5.2f}px  edge={records[-1]['edge_spacing_pixels']:5.2f}px  "
      f"nearest={records[-1]['ink_weighted_nearest_probe_pixels']:5.2f}px  "
      f"worst={records[-1]['worst_ink_nearest_probe_pixels']:5.2f}px  "
      f"ink_in_interior={records[-1]['ink_inside_the_interior_probes']:.3f}  "
      f"captured_ink={records[-1]['captured_ink_fraction']:.3f}", flush=True
    )
  return records


def landscape(detector, *, n_side, n_slice, n_events, max_learners, seed, n_threads, output):
  """Proxy loss over a log-spaced grid of `(sigma_x, sigma_y)` plus a denser ISOTROPIC slice.

  Written after every design, so a job that is killed leaves a usable file behind. Common random
  numbers: every design is scored on the SAME events, which removes the draw from the comparison."""
  from detopt.bo.gbdt import score_design

  proxy = _ProxyReadout(detector)
  ladder = sigma_ladder(detector, n_side)
  isotropic = sigma_ladder(detector, n_slice)
  designs = [(float(sx), float(sy)) for sy in ladder for sx in ladder]
  designs += [(float(s), float(s)) for s in isotropic]
  rows = []
  for index, (sigma_x, sigma_y) in enumerate(designs):
    started = time.time()
    score = score_design(
      proxy, np.asarray([sigma_x, sigma_y], np.float32), n_events=n_events, event_offset=0, max_learners=max_learners,
      n_threads=n_threads, seed=seed
    )
    rows.append({
      'sigma_x': sigma_x,
      'sigma_y': sigma_y,
      'loss': NO_INFORMATION * float(score.loss),
      'sem': NO_INFORMATION * float(score.sem),
      'train': NO_INFORMATION * float(score.train),
      'val': NO_INFORMATION * float(score.val),
      'train_sem': NO_INFORMATION * float(score.train_sem),
      'val_sem': NO_INFORMATION * float(score.val_sem),
      'n_learners': int(score.n_learners),
      'seconds': float(time.time() - started),
    })
    score = rows[-1]
    resolution = abs(score['val'] - score['train']) + float(np.hypot(score['train_sem'], score['val_sem']))
    print(
      f'  [{index + 1}/{len(designs)}] sigma=({sigma_x:6.3f}, {sigma_y:6.3f})  loss={score["loss"]:.4f}'
      f' +- {score["sem"]:.4f} nats  diff+err={resolution:.4f}  learners={score["n_learners"]}'
      f'  {score["seconds"]:.1f}s', flush=True
    )
    write_landscape(output, detector, rows, n_side, n_slice, n_events)
  return rows


def write_landscape(path, detector, rows, n_side, n_slice, n_events):
  """The npz `scripts/probe_precision.py` reads: SCALED designs beside their losses, plus everything
  the plots and the report need."""
  nominal = np.asarray([[r['sigma_x'], r['sigma_y']] for r in rows], np.float32)
  scaled = np.asarray(detector.to_scaled(nominal), np.float32)
  np.savez(
    path, designs=scaled, losses=np.asarray([r['loss'] for r in rows],
                                            np.float64), nominal=nominal, sem=np.asarray([r['sem'] for r in rows], np.float64),
    train=np.asarray([r['train'] for r in rows], np.float64), val=np.asarray([r['val'] for r in rows], np.float64),
    train_sem=np.asarray([r['train_sem'] for r in rows],
                         np.float64), val_sem=np.asarray([r['val_sem'] for r in rows],
                                                         np.float64), n_learners=np.asarray([r['n_learners'] for r in rows],
                                                                                            np.int32),
    seconds=np.asarray([r['seconds'] for r in rows],
                       np.float64), n_side=int(n_side), n_slice=int(n_slice), n_events=int(n_events),
    sigma_min=float(detector.sigma_min), sigma_max=float(detector.sigma_max), n_grid=int(detector.n_grid)
  )


def plot_profile(detector, landscape_path, geometry_records, path):
  """The loss profile: the map over both coordinates, the isotropic slice with its error bars, and
  the geometry that explains each end."""
  with np.load(landscape_path) as data:
    nominal, loss, sem = np.asarray(data['nominal']), np.asarray(data['losses']), np.asarray(data['sem'])
    n_side, n_slice = int(data['n_side']), int(data['n_slice'])
  n_map = n_side * n_side
  ladder = sigma_ladder(detector, n_side)
  grid_loss = loss[:n_map].reshape(n_side, n_side)
  slice_sigma, slice_loss, slice_sem = nominal[n_map:, 0], loss[n_map:], sem[n_map:]

  figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
  edges = np.exp(
    np.concatenate([[1.5 * np.log(ladder[0]) - 0.5 * np.log(ladder[1])], 0.5 * (np.log(ladder[:-1]) + np.log(ladder[1:])),
                    [1.5 * np.log(ladder[-1]) - 0.5 * np.log(ladder[-2])]])
  )
  mesh = axes[0].pcolormesh(edges, edges, grid_loss, cmap='viridis_r', shading='flat')
  best = int(np.argmin(grid_loss))
  axes[0].plot(ladder[best % n_side], ladder[best // n_side], marker='*', markersize=16, color=VERMILLION, linestyle='none')
  axes[0].set_xscale('log')
  axes[0].set_yscale('log')
  figure.colorbar(mesh, ax=axes[0]).set_label('proxy cross-entropy, nats', fontsize=8, color=MUTED)
  style(
    axes[0], f'loss over the design space (star: best {grid_loss.min():.3f} nats)', 'sigma_x (column axis)',
    'sigma_y (row axis)'
  )

  axes[1].fill_between(slice_sigma, slice_loss - slice_sem, slice_loss + slice_sem, color=BLUE, alpha=0.25, linewidth=0)
  axes[1].plot(slice_sigma, slice_loss, color=BLUE, marker='o', markersize=3, label='isotropic sigma')
  axes[1].axhline(NO_INFORMATION, color=VERMILLION, linestyle='--', linewidth=1.0, label='no information, ln 10')
  axes[1].axvline(1.0, color=MUTED, linestyle=':', linewidth=1.0, label='uniform grid, sigma = 1')
  axes[1].set_xscale('log')
  axes[1].set_yscale('log')
  ticks = [t for t in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, NO_INFORMATION) if t >= 0.8 * slice_loss.min()]
  axes[1].set_yticks(ticks)
  axes[1].set_yticklabels([f'{t:.2f}' for t in ticks])
  axes[1].minorticks_off()
  axes[1].legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='none', loc='upper center')
  style(axes[1], 'the isotropic slice', 'sigma (both axes)', 'proxy cross-entropy, nats')

  sigma = np.asarray([r['sigma'] for r in geometry_records])
  axes[2].plot(
    sigma, [r['ink_weighted_nearest_probe_pixels'] for r in geometry_records], color=ORANGE, marker='^', markersize=3,
    label='ink-weighted nearest probe'
  )
  axes[2].plot(
    sigma, [r['worst_ink_nearest_probe_pixels'] for r in geometry_records], color=VERMILLION, marker='o', markersize=3,
    label='worst inked pixel to nearest probe'
  )
  axes[2].set_xscale('log')
  axes[2].set_yscale('log')
  twin = axes[2].twinx()
  twin.plot(
    sigma, [r['centre_to_edge_ratio'] for r in geometry_records], color=GREEN, marker='v', markersize=3,
    label='centre/edge spacing ratio'
  )
  twin.plot(
    sigma, [r['captured_ink_fraction'] for r in geometry_records], color=MUTED, linestyle='--', marker='s', markersize=3,
    label='captured ink -- NOT the objective'
  )
  twin.set_ylabel('fraction', fontsize=9, color=MUTED)
  twin.tick_params(labelsize=8, colors=MUTED, length=3)
  handles, labels = axes[2].get_legend_handles_labels()
  extra_handles, extra_labels = twin.get_legend_handles_labels()
  axes[2].legend(
    handles + extra_handles, labels + extra_labels, fontsize=8, frameon=True, framealpha=0.9, edgecolor='none',
    loc='center left'
  )
  style(axes[2], 'what the probe density does to the coverage', 'sigma (both axes)', 'pixels')

  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  return path


def sobol_landscape(detector, *, n_designs, n_events, max_learners, seed, n_threads, output):
  """Proxy loss at UNIFORMLY drawn designs -- a low-discrepancy sample of the scaled cube, which is
  what a run's `n_init` draws from and therefore the distribution BO actually starts in.

  The lattice says what the surface looks like; this says what a RANDOM design costs, and the two
  answer different questions. Written after every design, so a killed job leaves usable work."""
  from scipy.stats import qmc

  from detopt.bo.gbdt import score_design

  proxy = _ProxyReadout(detector)
  points = int(np.ceil(np.log2(max(int(n_designs), 2))))
  scaled = qmc.Sobol(d=int(detector.design_dim()), scramble=True, seed=int(seed)).random_base2(points)
  rows = []
  for index, point in enumerate(scaled):
    nominal = np.asarray(detector.flatten_design(detector.to_nominal(point.astype(np.float32))), np.float32)
    started = time.time()
    score = score_design(
      proxy, nominal, n_events=n_events, event_offset=0, max_learners=max_learners, n_threads=n_threads, seed=seed
    )
    rows.append({
      'sigma_x': float(nominal[0]),
      'sigma_y': float(nominal[1]),
      'scaled': [float(v) for v in point],
      'loss': NO_INFORMATION * float(score.loss),
      'sem': NO_INFORMATION * float(score.sem),
      'seconds': float(time.time() - started),
    })
    print(
      f'  [{index + 1}/{len(scaled)}] sigma=({rows[-1]["sigma_x"]:6.3f}, {rows[-1]["sigma_y"]:6.3f})  '
      f'loss={rows[-1]["loss"]:.4f} +- {rows[-1]["sem"]:.4f} nats  {rows[-1]["seconds"]:.1f}s', flush=True
    )
    np.savez(
      output, designs=np.asarray([r['scaled'] for r in rows], np.float32),
      losses=np.asarray([r['loss'] for r in rows], np.float64), nominal=np.asarray([[r['sigma_x'], r['sigma_y']] for r in rows],
                                                                                   np.float32),
      sem=np.asarray([r['sem'] for r in rows], np.float64), seconds=np.asarray([r['seconds'] for r in rows],
                                                                               np.float64), n_events=int(n_events)
    )
  return rows


def plot_random(detector, sobol_path, path, seed):
  """The profile over RANDOM designs: what each one costs, what the distribution of costs looks like,
  and whether a GP fitted to them predicts a design it has not seen."""
  with np.load(sobol_path) as data:
    nominal, loss, sem = np.asarray(data['nominal']), np.asarray(data['losses']), np.asarray(data['sem'])
    scaled = np.asarray(data['designs'])
  figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

  points = axes[0].scatter(nominal[:, 0], loss, c=np.log10(nominal[:, 1]), cmap='viridis', s=22)
  axes[0].errorbar(nominal[:, 0], loss, yerr=sem, linestyle='none', ecolor='#BFBFBF', elinewidth=0.8)
  axes[0].set_xscale('log')
  figure.colorbar(points, ax=axes[0]).set_label('log10 sigma_y', fontsize=8, color=MUTED)
  style(axes[0], f'{len(loss)} uniform designs', 'sigma_x (column axis)', 'proxy cross-entropy, nats')

  order = np.argsort(loss)
  axes[1].plot(np.arange(1, len(loss) + 1), loss[order], color=BLUE, marker='o', markersize=3)
  axes[1].axhline(NO_INFORMATION, color=VERMILLION, linestyle='--', linewidth=1.0, label='no information, ln 10')
  axes[1].legend(fontsize=8, frameon=False, loc='lower right')
  style(axes[1], 'the cost of a random design, sorted', 'rank', 'proxy cross-entropy, nats')

  predicted = gp_predictions(scaled, loss, seed)
  axes[2].plot([loss.min(), loss.max()], [loss.min(), loss.max()], color=MUTED, linestyle=':', linewidth=1.0)
  axes[2].plot(loss, predicted, linestyle='none', marker='o', markersize=4, color=GREEN)
  residual, total = float(np.sum(np.square(loss - predicted))), float(np.sum(np.square(loss - loss.mean())))
  style(
    axes[2], f'a GP on those designs, held out (R2 = {1.0 - residual / total:.3f})', 'measured, nats',
    'predicted out of fold, nats'
  )

  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  return path


def gp_predictions(designs_scaled, losses, seed):
  """Out-of-fold GP predictions over the given designs (5-fold, or as many folds as there are
  designs). Same estimator family as `scripts/screen_task.py`."""
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
  from sklearn.model_selection import KFold

  designs_scaled, losses = np.asarray(designs_scaled, np.float64), np.asarray(losses, np.float64)
  kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.full(designs_scaled.shape[1], 0.3), (1e-2, 1e2)) \
      + WhiteKernel(1e-4, (1e-8, 1e0))
  predicted = np.zeros_like(losses)
  n_splits = min(5, designs_scaled.shape[0])
  for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=int(seed)).split(designs_scaled):
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=int(seed))
    model.fit(designs_scaled[train], losses[train])
    predicted[test] = model.predict(designs_scaled[test])
  return predicted


def gp_diagnostic(designs_scaled, losses, seed):
  """Held-out R2 of a GP fitted to the landscape: is the profile something a surrogate can CLIMB, or
  is it noise?"""
  losses = np.asarray(losses, np.float64)
  predicted = gp_predictions(designs_scaled, losses, seed)
  residual = float(np.sum(np.square(losses - predicted)))
  total = float(np.sum(np.square(losses - losses.mean())))
  return 1.0 - residual / total


def summarise(detector, landscape_path, geometry_records, seed):
  """What the landscape says, as numbers rather than a picture: where the optimum is, its margin over
  each end in units of the proxy's own error bar, and whether a GP can model the profile."""
  with np.load(landscape_path) as data:
    nominal, loss, sem = np.asarray(data['nominal']), np.asarray(data['losses']), np.asarray(data['sem'])
    scaled, train, val = np.asarray(data['designs']), np.asarray(data['train']), np.asarray(data['val'])
    train_sem, val_sem = np.asarray(data['train_sem']), np.asarray(data['val_sem'])
    n_side, n_slice = int(data['n_side']), int(data['n_slice'])
  n_map = n_side * n_side
  best = int(np.argmin(loss))
  low, high = n_map, len(loss) - 1
  margin = lambda end: float((loss[end] - loss[best]) / np.hypot(sem[end], sem[best]))
  isotropic = n_map + int(np.argmin(loss[n_map:]))
  summary = {
    'best_sigma': [float(v) for v in nominal[best]],
    'best_isotropic_sigma':
    float(nominal[isotropic, 0]),
    'best_isotropic_loss':
    float(loss[isotropic]),
    'best_loss':
    float(loss[best]),
    'best_sem':
    float(sem[best]),
    'loss_at_sigma_min':
    float(loss[low]),
    'loss_at_sigma_max':
    float(loss[high]),
    'margin_over_sigma_min_in_sem':
    margin(low),
    'margin_over_sigma_max_in_sem':
    margin(high),
    'no_information':
    NO_INFORMATION,
    'worst_proxy_diff_plus_err':
    float(np.max(np.abs(val - train) + np.hypot(train_sem, val_sem))),
    'median_seconds_per_proxy_fit':
    float(np.median(np.asarray(np.load(landscape_path)['seconds']))),
    'gp_r2_over_the_map':
    gp_diagnostic(scaled[:n_map], loss[:n_map], seed),
    'interior_optimum':
    bool(0 < best % n_side < n_side - 1 and 0 < best // n_side < n_side - 1)
    if best < n_map else bool(0 < best - n_map < n_slice - 1),
  }
  for key, value in summary.items():
    print(f'  {key}: {value}', flush=True)
  return summary


def plot_grids(detector, sigmas, path, event_index=3):
  """What the probe grid IS at a small, a good and a large sigma: the probes over the frame, and the
  10x10 read-out the network receives. The corners are pinned at every design; only the interior
  density moves."""
  index = np.asarray([int(event_index)], np.int64)
  figure, axes = plt.subplots(2, len(sigmas), figsize=(4.2 * len(sigmas), 8.4))
  for column, sigma in enumerate(sigmas):
    design = {'sigma_x': float(sigma), 'sigma_y': float(sigma)}
    _, event, mask, _ = detector(design, index)
    samples = np.asarray(detector.combine(event, design, mask=mask)[0, ..., 0])
    x, y = [unit_frame(a) for a in detector.sample_positions(design)]
    spacing = np.diff(x)
    ratio = spacing[detector.n_grid // 2 - 1] / spacing[0]
    frame = np.asarray(event.image[0], np.float32) / 255.0
    top = axes[0, column]
    top.imshow(frame, cmap='Greys', origin='upper', extent=(0.0, 1.0, 1.0, 0.0), interpolation='nearest')
    points = np.stack(np.meshgrid(x, y, indexing='xy'), axis=-1).reshape(-1, 2)
    top.plot(points[:, 0], points[:, 1], linestyle='none', marker='o', markersize=3, color=VERMILLION)
    top.set_xlim(-0.03, 1.03)
    top.set_ylim(1.03, -0.03)
    style(top, f'sigma = {sigma:.2f}: the 100 probes (centre/edge {ratio:.2f})', 'x (normalised column)', 'y (normalised row)')
    top.grid(False)
    bottom = axes[1, column]
    bottom.imshow(samples, cmap='Greys', origin='upper', interpolation='nearest', vmin=0.0, vmax=1.0)
    style(bottom, f'sigma = {sigma:.2f}: what the network sees', 'grid column', 'grid row')
    bottom.grid(False)
  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  return path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default='config/detector/pixel_density.yaml')
  parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE')
  parser.add_argument('--stage', nargs='+', default=['geometry', 'landscape', 'plots'])
  parser.add_argument('--output-dir', default='output/pixel-density')
  parser.add_argument('--n-sigma', type=int, default=41, help='geometry: points on the sigma ladder')
  parser.add_argument('--n-frames', type=int, default=8192, help='geometry: frames the ink statistics average over')
  parser.add_argument('--n-sobol', type=int, default=64, help='sobol: uniformly drawn designs to score')
  parser.add_argument('--n-side', type=int, default=9, help='landscape: designs per axis of the map')
  parser.add_argument('--n-slice', type=int, default=21, help='landscape: points on the isotropic slice')
  parser.add_argument('--n-events', type=int, default=12288, help='landscape: events per proxy fit')
  parser.add_argument('--max-learners', type=int, default=300)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
    '--grid-sigmas', type=float, nargs=3, default=None,
    help='plots: the small, good and large sigma to picture. Default: the two ENDS of the design '
    'range and the isotropic optimum the landscape measured, so the figure is the design space '
    'rather than three numbers chosen by hand'
  )
  arguments = parser.parse_args()

  os.makedirs(arguments.output_dir, exist_ok=True)
  detector = build(arguments.config, arguments.overrides)
  geometry_path = os.path.join(arguments.output_dir, 'geometry.json')
  landscape_path = os.path.join(arguments.output_dir, 'landscape.npz')
  sobol_path = os.path.join(arguments.output_dir, 'sobol.npz')
  n_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))

  if 'geometry' in arguments.stage:
    print(f'geometry over {arguments.n_sigma} sigmas, ink from {arguments.n_frames} frames', flush=True)
    records = geometry(detector, arguments.n_sigma, arguments.n_frames)
    with open(geometry_path, 'w') as f:
      json.dump({'sigma_min': detector.sigma_min, 'sigma_max': detector.sigma_max, 'records': records}, f, indent=2)
    print(f'wrote {geometry_path}', flush=True)

  if 'landscape' in arguments.stage:
    print(
      f'landscape: {arguments.n_side}x{arguments.n_side} map + {arguments.n_slice} isotropic, '
      f'{arguments.n_events} events per fit', flush=True
    )
    rows = landscape(
      detector, n_side=arguments.n_side, n_slice=arguments.n_slice, n_events=arguments.n_events,
      max_learners=arguments.max_learners, seed=arguments.seed, n_threads=n_threads, output=landscape_path
    )
    worst = max(abs(r['val'] - r['train']) + float(np.hypot(r['train_sem'], r['val_sem'])) for r in rows)
    print(f'wrote {landscape_path}: {len(rows)} designs, worst proxy diff+err {worst:.4f}', flush=True)

  if 'sobol' in arguments.stage:
    print(f'sobol: {arguments.n_sobol} uniform designs, {arguments.n_events} events per fit', flush=True)
    rows = sobol_landscape(
      detector, n_designs=arguments.n_sobol, n_events=arguments.n_events, max_learners=arguments.max_learners,
      seed=arguments.seed, n_threads=n_threads, output=sobol_path
    )
    loss = np.asarray([r['loss'] for r in rows])
    print(
      f'wrote {sobol_path}: {len(rows)} designs, loss {loss.min():.4f} .. {loss.max():.4f} nats, '
      f'median {np.median(loss):.4f}', flush=True
    )
    print(f'wrote {plot_random(detector, sobol_path, os.path.join(arguments.output_dir, "random.png"), arguments.seed)}')

  if 'plots' in arguments.stage:
    with open(geometry_path) as f:
      records = json.load(f)['records']
    summary = summarise(detector, landscape_path, records, arguments.seed)
    with open(os.path.join(arguments.output_dir, 'summary.json'), 'w') as f:
      json.dump(summary, f, indent=2)
    sigmas = arguments.grid_sigmas
    if sigmas is None:
      sigmas = [detector.sigma_min, summary['best_isotropic_sigma'], detector.sigma_max]
    print(f'wrote {plot_profile(detector, landscape_path, records, os.path.join(arguments.output_dir, "profile.png"))}')
    print(f'wrote {plot_grids(detector, sigmas, os.path.join(arguments.output_dir, "grids.png"))}')


if __name__ == '__main__':
  main()
