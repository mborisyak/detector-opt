#!/usr/bin/env python3
"""The `linear` BO loop with the REGRESSOR REPLACED BY THE EXACT BAYES RISK, scanned over read-out noise.

WHAT IT IS. `scripts/bo.py` scores a design by training a network on it and reporting the converged
validation loss. This script runs the SAME driver -- `detopt.bo.BayesianOptimizer`, the
`normalised-invariant-rbf` kernel over the exchangeable probes, the same 5 Sobol initial points, the
same `numpy.random.SeedSequence` split, the `gp`/`ei` blocks read out of `config/linear.yaml` rather
than restated -- and replaces only the scorer: a design is scored by its BAYES RISK, the closed-form
loss a perfect estimator reaches there. No network is built, no event is simulated.

WHY. Every number a campaign reports is the sum of two things: what the SEARCH found and what the
TRAINER could resolve. `docs/linear-highdim.md` had to model the trainer's slack to price criterion (d);
`output/linear/` had to be re-scored by `scripts/verify_trajectory.py` to remove the arms' reported
optimism. Here the second term is exactly zero, so the trajectory is the driver's alone: the reported
loss IS the true loss, it can never fall below the optimum, and the distance from the incumbent design
to the analytic answer is measurable at every iteration.

THE MODEL, in `d` dimensions. A design is `n_probes` positions `x_i` in `[-1, 1]^d`; an event draws
`theta = (w, b)` with `d + 1` standard-normal components and reads out `y_i = w . x_i + b + eps_i`
with `eps_i ~ N(0, sigma^2)`; the loss is the mean squared error over the `d + 1` components, so 1.0
is the no-information level at every `d`. With `X` the `(n_probes, d + 1)` matrix of rows `(x_i, 1)`,

    bayes_risk = tr[(X^T X / sigma^2 + I)^-1] / (d + 1) = sum_k 1 / (1 + lambda_k / sigma^2) / (d + 1)

over the eigenvalues of `X^T X`, which is exact rather than ill-conditioned when a design is rank
deficient. `n_probes = d + 1` is the informative choice: `n_probes = d` leaves a whole prior direction
unmeasured (a constant `1/(d+1)` added to every design, `docs/linear-highdim.md` section 1.2) and
`n_probes > d + 1` only averages noise down, since the model still has `d + 1` unknowns.

⚠️ WHY THE FORMULA IS RESTATED HERE AND NOT IMPORTED. `LinearDetector.bayes_risk` is the `d = 1` case
of exactly this expression, and `detopt/detector/linear.py` was never extended with `n_dimensions`
(D102: stage 2 was conditional on stage 1 finding a viable cell, and it did not). Rather than change a
detector a finished campaign was run on, this script states the general form -- as
`scripts/linear_nd_landscape.py` already does, for the same reason -- and CHECKS it against the
detector at `d = 1` on every run: `--check-detector` (on by default) scores a Sobol sample and the
box corners both ways at each sigma and aborts on any disagreement above 1e-12. So there is one
formula in use and the detector remains its reference.

WHAT THIS PROBE MUST SEPARATE, and the check that it can. It must separate read-out noise levels at
which the design problem is still open after `--n-iterations` evaluations from levels at which it is
already solved. Three things make that legible rather than assumed:

* the answer is verified, not trusted: at every sigma the optimal design is found TWICE -- every
  design whose probes all sit on a box corner, enumerated exhaustively (`(2^d)^n_probes` of them), and
  multi-start L-BFGS-B on the box -- and the better of the two is the reference the distance axis is
  measured against. At `(d, n_probes) = (2, 3)` the A-optimal design is NOT a corner design, so a run
  that assumed corners would measure distance to the wrong point;
* the distance itself respects every symmetry the OBJECTIVE has: the probes are a set, and reflections
  and coordinate swaps of the box leave the risk exactly unchanged, so the optimum is an orbit of
  `2^d d!` designs and the distance is minimised over all of them and over the probe permutations. Not
  doing so reported a perfect `d = 2` run as 1.18 away in a box of width 2;
* a `random` arm runs on the same seeds. If BO does not separate from it, the scan is reporting the
  landscape's own easiness and not the driver's search.

The one thing it CANNOT show is whether a trainer could resolve any of this: with an exact scorer the
convergence slack that decides acceptance criterion (d) does not exist. That question is
`scripts/probe_precision.py`'s and is not reopened here.

USAGE, CPU-only and one job at a time (the GPU shards belong to the neural campaigns):

    srun --cpus-per-task=2 --mem=1800 -u python scripts/linear_exact_bo.py --n-dimensions 2 --n-probes 3 \
        --output-dir output/linear-exact-bo-d2n3
    srun --cpus-per-task=2 --mem=1800 -u python scripts/linear_exact_bo.py --replot --output-dir ...

OUTPUT, under `--output-dir` (never `output/linear/`): `scan.json` holds every trajectory with its
designs and the per-sigma anchors, `scan.png` four panels -- the incumbent loss, the same convergence
in DESIGN space, how many designs the search needs against its random control, and where the incumbent
sits at three reads. Every axis is in the objective's own units; nothing is normalised.
"""

import argparse
import itertools
import json
import os
import time

import matplotlib
import numpy as np
import scipy.optimize
import yaml

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

import detopt.detector  # noqa: E402
from detopt.bo import BayesianOptimizer, NormalisedInvariantRBF  # noqa: E402

SIGMA_RAMP = plt.get_cmap('Blues')
CATEGORICAL = ('#0072B2', '#E69F00', '#009E73')
MARKERS = ('o', 's', '^')
CONTROL = '#8a8a8a'
INK = '#1a1a1a'
MUTED = '#6b6b6b'
SYMLOG_THRESHOLD = 1.0e-3
DESIGN_TARGET = 0.05


def probes_from_scaled(x_scaled, dimension, n_probes):
  """`[0, 1]^(n_probes * d)` -> `(n_probes, d)` in `[-1, 1]^d`.

  The flat layout is ONE FIELD PER COORDINATE, each of length `n_probes` -- `[x_1..x_n, y_1..y_n, ...]`
  -- because that is the layout `detopt.bo._exchangeable_blocks` builds for a design whose probes are
  interchangeable, and the kernel's blocks below are derived from the same convention."""
  return 2.0 * np.asarray(x_scaled, np.float64).reshape(dimension, n_probes).T - 1.0


def risk(probes, sigma, dimension):
  """`tr[(X^T X / sigma^2 + I)^-1] / (d + 1)` at one design, from the eigenvalues of `X^T X`."""
  probes = np.asarray(probes, np.float64).reshape(-1, dimension)
  rows = np.concatenate([probes, np.ones((probes.shape[0], 1), np.float64)], axis=-1)
  eigenvalues = np.clip(np.linalg.eigvalsh(rows.T @ rows), 0.0, None)
  return float(np.sum(1.0 / (1.0 + eigenvalues / sigma**2)) / (dimension + 1))


def check_against_detector(path, sigma, n_probes, n_draws, seed):
  """At `d = 1`, `risk` must equal `LinearDetector.bayes_risk` -- the fixture's own code -- everywhere.

  Checked on the box corners and on a random sample, at the sigma about to be scanned, so the general
  formula this script runs on is tied to the detector rather than merely believed to match it.

  BOTH SIDES ARE FED THE SAME float32 DESIGN. `bayes_risk` flattens through a jax float32 array before
  promoting to float64, so a float64 design reaches it rounded; comparing against an unrounded one
  measures that rounding (8.5e-08, which this check reported the first time it ran) instead of the
  agreement of the two formulae, which is what it is for."""
  with open(path) as f:
    config = yaml.safe_load(f)
  (name, ) = config.keys()
  detector = detopt.detector.from_config({name: dict(config[name], noise=float(sigma), n_probes=int(n_probes))})
  generator = np.random.default_rng(seed)
  sample = np.concatenate([
    np.array(list(itertools.product([-1.0, 1.0], repeat=n_probes)), np.float32),
    generator.uniform(-1.0, 1.0, (int(n_draws), n_probes)).astype(np.float32)
  ])
  worst = max(abs(risk(design.astype(np.float64), sigma, 1) - detector.bayes_risk(design)) for design in sample)
  if not worst < 1.0e-12:
    raise SystemExit(f'linear_exact_bo: `risk` disagrees with LinearDetector.bayes_risk by {worst:.3e} at noise {sigma}')
  return worst


def symmetry_orbit(reference, dimension):
  """Every design the objective cannot tell apart from `reference`, as `(n_equivalent, n_probes, d)`.

  The risk depends on the design only through the eigenvalues of `X^T X` with rows `(x_i, 1)`. A sign
  flip of one coordinate and a permutation of the coordinates both act on those rows as an orthogonal
  `R = diag(Q, 1)`, sending `X^T X -> R^T X^T X R`, which leaves the eigenvalues -- and the box -- alone.
  So the optimum is never a point but an ORBIT of `2^d d!` designs, times the probe permutations, and a
  search that lands on a reflected copy has found the answer. Measuring distance to one arbitrary
  member instead would have reported a perfect `d = 2` run as 1.18 away from the box's own width of 2.
  `verified_optimum` returns one member; this expands it, and `scan` asserts the risk is constant on
  what comes back."""
  reference = np.asarray(reference, np.float64).reshape(-1, dimension)
  orbit = []
  for signs in itertools.product([-1.0, 1.0], repeat=dimension):
    for order in itertools.permutations(range(dimension)):
      orbit.append(reference[:, list(order)] * np.asarray(signs, np.float64))
  return np.asarray(orbit, np.float64)


def design_distance(design, orbit, dimension):
  """Distance from a design to the optimum's ORBIT, minimised over the probes' permutations too.

  The probes are a set, so `(a, b)` and `(b, a)` are one experiment; the orbit carries the reflections
  and coordinate swaps the objective cannot see. Reported as the largest coordinate difference of the
  best matching, in the box's own units (its width is 2)."""
  design = np.asarray(design, np.float64).reshape(-1, dimension)
  orders = list(itertools.permutations(range(design.shape[0])))
  return min(float(np.max(np.abs(design[list(order)] - member))) for member in orbit for order in orders)


def verified_optimum(dimension, n_probes, sigma, n_restarts, seed):
  """`(optimum, design, note)` -- the best design in the box, found TWO ways and reconciled.

  Every design whose probes all sit on a box CORNER is enumerated exhaustively (`(2^d)^n_probes`, 64 at
  `d = 2, n = 3`), and multi-start L-BFGS-B runs on the box beside it. Gradient descent alone finds
  what it is started near; the enumeration alone cannot see an interior optimum, and at
  `(d, n_probes) = (2, 3)` `docs/linear-highdim.md` reports exactly that -- one probe on an edge
  midpoint. The better of the two is the reference, and the note records which won."""
  corners = np.array(list(itertools.product([-1.0, 1.0], repeat=dimension)), np.float64)
  index = np.array(list(itertools.product(range(corners.shape[0]), repeat=n_probes)), np.int64)
  vertex = corners[index]
  values = np.array([risk(design, sigma, dimension) for design in vertex], np.float64)
  best_corner, corner_value = vertex[int(np.argmin(values))], float(np.min(values))

  generator = np.random.default_rng(seed)
  width = n_probes * dimension
  starts = np.concatenate([generator.uniform(-1.0, 1.0, (int(n_restarts), width)), np.zeros((1, width))], axis=0)
  best_interior, interior_value = None, np.inf
  for start in starts:
    result = scipy.optimize.minimize(
      lambda flat: risk(flat.reshape(n_probes, dimension), sigma, dimension), start, method='L-BFGS-B',
      bounds=[(-1.0, 1.0)] * width
    )
    if float(result.fun) < interior_value:
      best_interior, interior_value = np.asarray(result.x, np.float64).reshape(n_probes, dimension), float(result.fun)
  if interior_value < corner_value - 1.0e-9:
    return interior_value, best_interior, f'L-BFGS-B BEAT the best corner design by {corner_value - interior_value:.2e}'
  return corner_value, best_corner, f'the corner design wins; L-BFGS-B came within {interior_value - corner_value:.2e}'


def baseline_median(dimension, n_probes, sigma, draws, seed):
  """The acceptance criterion's baseline: the MEDIAN Bayes risk of a uniform design in the box.

  The spread returned beside it is the asymptotic standard error of a median, `1.2533 sd / sqrt(n)`.
  That formula assumes the distribution is locally normal and this one is strongly skewed, so it is a
  guide to how many draws are enough, not a bound on the error."""
  generator = np.random.default_rng(seed)
  sample = generator.uniform(-1.0, 1.0, size=(int(draws), n_probes, dimension))
  risks = np.array([risk(design, sigma, dimension) for design in sample], np.float64)
  return float(np.median(risks)), 1.2533 * float(np.std(risks)) / np.sqrt(risks.size)


def build_kernel(bo_config, dimension, n_probes):
  """The run config's kernel over `d` BLOCKS of `n_probes` indices, permuted TOGETHER.

  Probe `k` is the pair (triple, ...) of its coordinates, so the coordinate blocks move under one
  shared permutation -- the same structure `detopt.bo._exchangeable_blocks` builds from a detector's
  design fields, and the same arithmetic `detopt.bo.kernel_from_config` applies to the prior bounds."""
  gp_config = bo_config['gp']
  (name, ) = gp_config['kernel'].keys()
  if name != 'normalised-invariant-rbf':
    raise SystemExit(f'linear_exact_bo: this script builds the invariant kernel only, and the config names `{name}`')
  amplitude_low, amplitude_high = gp_config['log_amplitude_prior_bounds']
  length_low, length_high = gp_config['log_lengthscale_prior_bounds']
  blocks = tuple(tuple(range(k * n_probes, (k + 1) * n_probes)) for k in range(dimension))
  return NormalisedInvariantRBF(
    d=dimension * n_probes, blocks=blocks, constant_value=float(np.exp(amplitude_low + amplitude_high)),
    constant_value_bounds=(float(np.exp(2.0 * amplitude_low)), float(np.exp(2.0 * amplitude_high))),
    length_scale=float(np.exp(0.5 * (length_low + length_high))),
    length_scale_bounds=(float(np.exp(length_low)), float(np.exp(length_high)))
  )


def run(arm, bo_config, dimension, n_probes, sigma, seed, n_iterations, observation_noise):
  """One arm, one seed: the exact loss of every design evaluated, and the design itself.

  The seed stream is `scripts/bo.py`'s: one `SeedSequence` split ONCE into a network branch and an
  iteration branch, the k-th iteration seed being the k-th spawn of the latter. The network branch is
  spawned and dropped here -- there is no network -- so that the proposals this script sees are the
  proposals the real driver would have seen at the same seed."""
  gp_config = {k: v for k, v in bo_config['gp'].items() if k != 'kernel'}
  width = dimension * n_probes
  optimiser = BayesianOptimizer(
    width, gp=gp_config, ei=dict(bo_config['ei']), kernel=build_kernel(bo_config, dimension, n_probes),
    n_init=int(bo_config.get('n_init', gp_config['n_folds']))
  )
  _, iteration_sequence = np.random.SeedSequence(int(seed)).spawn(2)
  values, designs = [], []
  for _ in range(int(n_iterations)):
    iteration_seed = int(iteration_sequence.spawn(1)[0].generate_state(1)[0])
    if arm == 'random':
      x_scaled = np.random.default_rng(iteration_seed).random(width)
    else:
      x_scaled = np.asarray(optimiser.propose(iteration_seed), np.float64)
    probes = probes_from_scaled(x_scaled, dimension, n_probes)
    value = risk(probes, sigma, dimension)
    optimiser.append(x_scaled, value, noise=observation_noise)
    values.append(value)
    designs.append([float(v) for v in probes.reshape(-1)])
  return values, designs


def incumbent_traces(values, designs, orbit, dimension):
  """The incumbent's loss and its distance to the optimum's orbit, at every iteration.

  The distance is computed ONCE PER DISTINCT INCUMBENT and then repeated: the incumbent changes a
  handful of times in a run, while the orbit-and-permutation minimisation costs
  `2^d d! n_probes!` comparisons -- 46080 at `(d, n_probes) = (4, 5)`, which is the difference between
  seconds and hours over a scan."""
  values = np.asarray(values, np.float64)
  best = np.minimum.accumulate(values)
  where = np.zeros(values.size, np.int64)
  for index in range(1, values.size):
    where[index] = index if values[index] <= values[where[index - 1]] else where[index - 1]
  cache = {index: design_distance(designs[index], orbit, dimension) for index in set(where.tolist())}
  return best, np.array([cache[index] for index in where.tolist()], np.float64)


def scan(arguments, publish):
  """Every (sigma, arm, seed) run, PUBLISHED AS IT COMPLETES.

  `publish(record)` is called after each run and after each noise level's anchors, so a heavy cell is
  never all-or-nothing: the killed `(d, n_probes) = (4, 5)` job spent 2 h 56 m and left nothing but a
  log, because the record used to be written once at the end. `--resume` reads that published file and
  skips the runs already in it, keyed by `(sigma, arm, seed)` -- fields every row actually carries.
  `scripts/probe_precision.py` keys its resume on an `n0` it never writes into a row, so it never skips
  anything; that is the mistake this avoids."""
  dimension, n_probes = arguments.n_dimensions, arguments.n_probes
  rows, anchors = list(arguments.resume_rows), list(arguments.resume_anchors)
  done = {(r['sigma'], r['arm'], r['seed']) for r in rows}
  known = {a['sigma'] for a in anchors}
  if len(done) > 0:
    print(f'resume: {len(done)} runs and {len(known)} noise levels already published', flush=True)

  def settings():
    return {
      'sigmas': [float(s) for s in arguments.sigmas],
      'seeds': [int(s) for s in arguments.seeds],
      'n_dimensions': int(dimension),
      'n_probes': int(n_probes),
      'n_iterations': int(arguments.n_iterations),
      'observation_noise': float(arguments.observation_noise),
      'optimum_restarts': int(arguments.optimum_restarts),
      'baseline_draws': int(arguments.baseline_draws),
      'baseline_seed': int(arguments.baseline_seed),
      'detector_config': arguments.detector_config,
      'run_config': arguments.run_config,
      'bo': arguments.bo_config
    }

  for sigma in arguments.sigmas:
    if sigma in known:
      anchor = next(a for a in anchors if a['sigma'] == sigma)
      optimal_design = np.asarray(anchor['optimal_design'], np.float64)
      optimum = anchor['optimum']
      orbit = symmetry_orbit(optimal_design, dimension)
      print(f'\n=== noise {sigma:g} | anchors reused from the published record | optimum {optimum:.6f}', flush=True)
    else:
      agreement = None
      if dimension == 1 and arguments.check_detector:
        agreement = check_against_detector(arguments.detector_config, sigma, n_probes, 256, arguments.baseline_seed)
      optimum, optimal_design, note = verified_optimum(
        dimension, n_probes, sigma, arguments.optimum_restarts, arguments.baseline_seed
      )
      orbit = symmetry_orbit(optimal_design, dimension)
      spread = max(abs(risk(member, sigma, dimension) - optimum) for member in orbit)
      if not spread < 1.0e-12:
        raise SystemExit(f'linear_exact_bo: the risk is not constant on the optimum`s orbit (spread {spread:.3e})')
      baseline, baseline_spread = baseline_median(dimension, n_probes, sigma, arguments.baseline_draws, arguments.baseline_seed)
      anchors.append({
        'sigma':
        float(sigma),
        'n_dimensions':
        int(dimension),
        'n_probes':
        int(n_probes),
        'optimum':
        optimum,
        'optimal_design': [[float(v) for v in probe] for probe in np.asarray(optimal_design).reshape(n_probes, dimension)],
        'n_equivalent':
        int(orbit.shape[0]),
        'optimum_note':
        note,
        'detector_agreement':
        agreement,
        'ceiling':
        risk(np.zeros((n_probes, dimension)), sigma, dimension),
        'baseline':
        baseline,
        'baseline_spread':
        baseline_spread,
        'range':
        baseline - optimum
      })
      print(
        f'\n=== noise {sigma:g} | d {dimension}, {n_probes} probes | optimum {optimum:.6f} at '
        f'{np.round(np.asarray(optimal_design).reshape(n_probes, dimension), 4).tolist()} ({note}) | '
        f'baseline {baseline:.6f} +- {baseline_spread:.6f} | ceiling {anchors[-1]["ceiling"]:.6f}'
        f'{"" if agreement is None else f" | detector agrees to {agreement:.1e}"}', flush=True
      )
      publish({'settings': settings(), 'anchors': anchors, 'rows': rows})

    for arm, seed in itertools.product(('bo', 'random'), arguments.seeds):
      if (float(sigma), arm, int(seed)) in done:
        continue
      started = time.time()
      values, designs = run(
        arm, arguments.bo_config, dimension, n_probes, sigma, seed, arguments.n_iterations, arguments.observation_noise
      )
      best, distance = incumbent_traces(values, designs, orbit, dimension)
      rows.append({
        'sigma': float(sigma),
        'arm': arm,
        'seed': int(seed),
        'values': [round(float(v), 12) for v in values],
        'designs': [[round(float(v), 6) for v in d] for d in designs],
        'incumbent': [round(float(v), 12) for v in best],
        'distance': [round(float(v), 6) for v in distance]
      })
      publish({'settings': settings(), 'anchors': anchors, 'rows': rows})
      reading = ' '.join(f'loss@{at} {best[min(at, best.size) - 1]:.6f}' for at in (10, 20, best.size))
      print(
        f'  {arm:<7} seed {seed:>3}: {reading} | excess over the optimum {best[-1] - optimum:+.3e} '
        f'| |d - optimal| {distance[-1]:.4f} | {time.time() - started:.0f} s', flush=True
      )
  return {'settings': settings(), 'anchors': anchors, 'rows': rows}


def median_over_seeds(rows, sigma, arm, field):
  selected = [r[field] for r in rows if r['sigma'] == sigma and r['arm'] == arm]
  return np.median(np.asarray(selected, np.float64), axis=0), np.asarray(selected, np.float64)


def first_at_or_below(trace, target, horizon):
  """How many designs a run evaluated before its incumbent first met `target`; `horizon + 1` if never,
  which keeps the median well defined as long as fewer than half the seeds censor."""
  met = np.flatnonzero(np.asarray(trace, np.float64) <= target)
  return int(met[0]) + 1 if met.size > 0 else int(horizon) + 1


def plot(record, path):
  """Four panels, all in the objective's OWN units -- no normalisation anywhere.

  (a) the incumbent loss against designs evaluated, one curve per read-out noise, each with its own
  analytic optimum drawn as a dotted line of the same shade; (b) the same convergence read in DESIGN
  space, the incumbent's distance to the verified optimal design; (c) how many designs the search needs
  to find that design, with the random control; (d) the incumbent loss at three reads against the noise
  axis, between the analytic optimum and the median random design."""
  anchors = {a['sigma']: a for a in record['anchors']}
  sigmas = sorted(anchors)
  rows = record['rows']
  horizon = int(record['settings']['n_iterations'])
  iterations = np.arange(1, horizon + 1)
  shades = [SIGMA_RAMP(0.30 + 0.65 * k / max(len(sigmas) - 1, 1)) for k in range(len(sigmas))]

  figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
  for axis in axes.ravel():
    axis.grid(True, which='major', color='#e9e9e9', linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
      axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
      axis.spines[side].set_color('#cccccc')
    axis.tick_params(colors=MUTED, labelsize=9)

  panel = axes[0, 0]
  for sigma, shade in zip(sigmas, shades):
    median, _ = median_over_seeds(rows, sigma, 'bo', 'incumbent')
    panel.plot(iterations, median, color=shade, linewidth=2.0, label=f'{sigma:g}')
    panel.axhline(anchors[sigma]['optimum'], color=shade, linewidth=1.0, linestyle=':')
  panel.set_yscale('log')
  panel.set_xlabel('designs evaluated (5 Sobol, then BO)', color=INK, fontsize=10)
  panel.set_ylabel('incumbent loss (MSE on the standard-normal target)', color=INK, fontsize=10)
  panel.set_title('a  the incumbent LOSS, median over seeds', color=INK, fontsize=11, loc='left')
  panel.legend(title='read-out noise', fontsize=8, title_fontsize=8, ncol=2, frameon=False, loc='lower left')
  panel.annotate(
    'dotted: the analytic optimum at that noise', xy=(horizon, anchors[sigmas[0]]['optimum']), xytext=(-4, 6),
    textcoords='offset points', color=MUTED, fontsize=8.5, ha='right'
  )

  panel = axes[0, 1]
  for sigma, shade in zip(sigmas, shades):
    median, _ = median_over_seeds(rows, sigma, 'bo', 'distance')
    panel.plot(iterations, median, color=shade, linewidth=2.0, label=f'{sigma:g}')
  panel.set_yscale('symlog', linthresh=SYMLOG_THRESHOLD, linscale=0.6)
  panel.set_ylim(0.0, 2.0)
  panel.set_xlabel('designs evaluated (5 Sobol, then BO)', color=INK, fontsize=10)
  panel.set_ylabel('|incumbent - nearest optimal design|, box width 2', color=INK, fontsize=10)
  panel.set_title('b  the same convergence in DESIGN space', color=INK, fontsize=11, loc='left')
  panel.legend(title='read-out noise', fontsize=8, title_fontsize=8, ncol=2, frameon=False, loc='center right')
  panel.annotate(
    '0 = an optimal design itself', xy=(horizon, 0.0), xytext=(-4, 6), textcoords='offset points', color=MUTED, fontsize=8.5,
    ha='right'
  )

  panel = axes[1, 0]
  censored_any = False
  for arm, target, colour, marker, style, label in (('bo', 0.0, CATEGORICAL[0], MARKERS[0], '-',
                                                     'BO, the optimal design exactly'),
                                                    ('bo', DESIGN_TARGET, CATEGORICAL[2], MARKERS[2], '--',
                                                     f'BO, within {DESIGN_TARGET:g} of it'),
                                                    ('random', DESIGN_TARGET, CONTROL, MARKERS[1], ':',
                                                     f'random control, within {DESIGN_TARGET:g}')):
    reached, censored = [], []
    for sigma in sigmas:
      _, every = median_over_seeds(rows, sigma, arm, 'distance')
      first = np.array([first_at_or_below(trace, target, horizon) for trace in every], np.float64)
      reached.append(float(np.median(first)))
      censored.append(bool(np.median(first) > horizon))
    panel.plot(sigmas, reached, color=colour, linewidth=2.0, linestyle=style, label=label, zorder=2)
    for sigma, value, is_censored in zip(sigmas, reached, censored):
      panel.plot([sigma], [value], color=colour, marker=marker, markersize=7, zorder=3,
                 markerfacecolor='white' if is_censored else colour)
      censored_any = censored_any or is_censored
  panel.axhline(horizon, color=MUTED, linewidth=1.0, linestyle=':')
  if censored_any:
    panel.annotate(
      f'open marker: the median seed never got there within the {horizon}', xy=(sigmas[0], horizon), xytext=(0, -15),
      textcoords='offset points', color=MUTED, fontsize=8.5
    )
  panel.set_xscale('log')
  panel.set_ylim(0.0, horizon * 1.28)
  panel.set_xlabel('read-out noise (sigma)', color=INK, fontsize=10)
  panel.set_ylabel('designs evaluated before the design is found', color=INK, fontsize=10)
  panel.set_title('c  how long the search takes, median over seeds', color=INK, fontsize=11, loc='left')
  panel.legend(fontsize=8.5, frameon=False, loc='center left')

  panel = axes[1, 1]
  panel.plot(
    sigmas, [anchors[s]['baseline'] for s in sigmas], color=MUTED, linewidth=1.6, linestyle='--', label='median random design'
  )
  reads = sorted({at for at in (10, 20, horizon) if at <= horizon})
  for at, colour, marker, size in zip(reads, CATEGORICAL, MARKERS, (9.0, 7.5, 6.0)):
    values = []
    for sigma in sigmas:
      _, every = median_over_seeds(rows, sigma, 'bo', 'incumbent')
      values.append(float(np.median(every[:, at - 1])))
    panel.plot(
      sigmas, values, color=colour, marker=marker, markersize=size, linewidth=2.0, markeredgecolor='white', markeredgewidth=1.0,
      label=f'after {at} designs'
    )
  panel.plot(sigmas, [anchors[s]['optimum'] for s in sigmas], color=INK, linewidth=1.6, linestyle=':', label='analytic optimum')
  panel.set_xscale('log')
  panel.set_yscale('log')
  panel.set_xlabel('read-out noise (sigma)', color=INK, fontsize=10)
  panel.set_ylabel('incumbent loss (MSE on the standard-normal target)', color=INK, fontsize=10)
  panel.set_title('d  where the incumbent sits at three reads', color=INK, fontsize=11, loc='left')
  panel.legend(fontsize=8.5, frameon=False, loc='upper left')

  dimension = int(record['anchors'][0]['n_dimensions'])
  n_probes = int(record['anchors'][0]['n_probes'])
  figure.suptitle(
    '`linear` under the pipeline\'s BO, with the regressor replaced by the exact Bayes risk', color=INK, fontsize=13, x=0.007,
    y=0.986, ha='left'
  )
  figure.text(
    0.007, 0.945, f'design = {n_probes} probes in [-1, 1]^{dimension}, target ({dimension} slopes, intercept), '
    f'{horizon} designs a run, {len(record["settings"]["seeds"])} seeds; losses are the objective`s own, unnormalised',
    color=MUTED, fontsize=9.5, ha='left'
  )
  figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.928))
  figure.savefig(path, dpi=160, facecolor='white')
  print(f'\nwrote {path}')


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--sigmas', type=float, nargs='+', default=[0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
  parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
  parser.add_argument('--n-dimensions', type=int, default=1, help='d, the number of slope coefficients; the target is d + 1')
  parser.add_argument(
    '--n-probes', type=int, default=2, help='probes per design. `d + 1` is the informative choice: `d` leaves a prior '
    'direction unmeasured and more than `d + 1` only averages the noise down'
  )
  parser.add_argument('--n-iterations', type=int, default=40, help='designs evaluated per run, the 5 Sobol included')
  parser.add_argument('--detector-config', default='config/detector/linear.yaml', help='the d = 1 cross-check`s detector')
  parser.add_argument('--run-config', default='config/linear.yaml', help='supplies the `bo` block, kernel included')
  parser.add_argument(
    '--observation-noise', type=float, default=0.0, help='standard deviation handed to the GP per observation. The '
    'objective is EXACT, so zero is the honest value and the driver floors the kernel diagonal at 1e-12 variance'
  )
  parser.add_argument('--optimum-restarts', type=int, default=64, help='L-BFGS-B starts used to verify the optimum')
  parser.add_argument('--baseline-draws', type=int, default=200000)
  parser.add_argument('--baseline-seed', type=int, default=0)
  parser.add_argument(
    '--no-check-detector', dest='check_detector', action='store_false',
    help='skip the d = 1 equivalence check against LinearDetector.bayes_risk'
  )
  parser.add_argument(
    '--resume', action='store_true', help='keep the runs already in the output`s scan.json and do only the rest. The '
    'settings that define a cell must match, or it refuses rather than mixing two cells in one file'
  )
  parser.add_argument('--output-dir', default='output/linear-exact-bo')
  parser.add_argument('--replot', action='store_true', help='redraw the figure from an existing scan.json')
  arguments = parser.parse_args()

  os.makedirs(arguments.output_dir, exist_ok=True)
  path = os.path.join(arguments.output_dir, 'scan.json')
  if arguments.replot:
    with open(path) as f:
      record = json.load(f)
    plot(record, os.path.join(arguments.output_dir, 'scan.png'))
    return

  with open(arguments.run_config) as f:
    arguments.bo_config = yaml.safe_load(f)['bo']
  arguments.resume_rows, arguments.resume_anchors = [], []
  if arguments.resume and os.path.isfile(path):
    with open(path) as f:
      published = json.load(f)
    for key in ('n_dimensions', 'n_probes', 'n_iterations', 'observation_noise'):
      if published['settings'][key] != getattr(arguments, key):
        raise SystemExit(
          f'linear_exact_bo: --resume refused, {key} is {published["settings"][key]} in {path} and '
          f'{getattr(arguments, key)} on the command line'
        )
    arguments.resume_rows, arguments.resume_anchors = published['rows'], published['anchors']

  def publish(record):
    """Write the record to a temporary file and rename it over the real one, so a kill during the write
    cannot leave a truncated json where a complete one was."""
    temporary = path + '.partial'
    with open(temporary, 'w') as f:
      json.dump(record, f)
    os.replace(temporary, path)

  record = scan(arguments, publish)
  print(f'\nwrote {path}')
  plot(record, os.path.join(arguments.output_dir, 'scan.png'))


if __name__ == '__main__':
  main()
