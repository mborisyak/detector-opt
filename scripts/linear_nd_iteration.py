#!/usr/bin/env python3
"""Does the d-dimensional `linear` landscape CONVERT ITERATIONS? -- stage 1b of `docs/linear-highdim.md`.

Stage 1a (`scripts/linear_nd_landscape.py`) prices criterion (d) against
``gain = median_random - optimum``, which is only an UPPER BOUND on ``loss@n - loss@2n``: it is the
whole distance from the baseline to the floor, and the doubling only ever buys part of it. This
script measures the part, on the SAME closed-form objective, with the repo's own BO driver and the
kernel the run config would use. No network is trained and no detector is imported.

THE PROXY, and what it can and cannot show. The objective is the Bayes risk
``tr[(X^T X / sigma^2 + I)^-1] / (d + 1)`` plus a Gaussian draw whose standard deviation is the
convergence slack PREDICTED at that design,

    slack(design) = diff_over_level * level(design) + 2 * sd(design) / sqrt(window)

-- heteroscedastic exactly as the real task is, because both slack terms scale with the loss level
and the level spans the whole landscape. It is OPTIMISTIC in one stated way: it assumes the network
reaches the Bayes floor at every design. The 1-dimensional campaign measured that excess at
0.0002-0.0005 near the optimum, and a level-proportional excess at the bad end is MONOTONE in the
level, so it re-scales the landscape without reordering it.

PRE-REGISTERED BEFORE ANY CELL WAS RUN (NO TRICKS, `docs/benchmark-acceptance.md`):

    seeds                 1 .. 10                    (10, the doc's ">= 10 seeds")
    n -> 2n               20 -> 40                   (the pair config/linear.yaml already declares)
    n_init                5 Sobol points             (the doc's "BO with 5 initial Sobol points")
    kernel                normalised-invariant-rbf, d blocks of n_probes indices -- the design is a
                          SET of probes and each probe is a d-vector, so the d coordinate blocks are
                          permuted TOGETHER, exactly as `_exchangeable_blocks` builds them
    gp / ei               copied verbatim from config/enzyme_extremes.yaml
    baseline              the analytic MEDIAN over uniform designs (stage 1a, 200000 draws)
    loss_precision        the predicted maximum slack, i.e. `slack` at the CEILING design
    diff_over_level       0.018   ESTIMATE from the 1-dimensional campaign (slack 0.0155 at the
                          coincident designs, of which err 0.0067, over a level of 0.50)
    window                262144  the iteration_limit the run config declares
    arms                  `bo` and `random`, paired by seed (criterion 3, the bonus)
    cells                 CANDIDATES  (d, n_probes) in {(1,2), (2,3), (3,4)} x sigma in {0.3,0.5,0.7}
                          CONTROLS    (2,2) and (3,3) at sigma 0.7 -- rank-deficient by construction,
                          expected to FAIL, and included so a pass is read against something

WHAT THIS PROBE MUST BE ABLE TO SEPARATE, checked before it is believed: a landscape that converts
20 -> 40 into a real improvement from one that does not. The controls are the check that it can --
(d, n_probes) = (d, d) leaves a whole prior direction unmeasurable, so its entire landscape spans
``1/(d+1)`` to ``d/(d+1)`` with the informative part a small ripple on top; if the controls pass, the
probe is not measuring what it claims and nothing here means anything.
"""

import argparse
import itertools
import json
import os

import numpy as np

from detopt.bo import BayesianOptimizer, NormalisedInvariantRBF

GP = {
  'n_folds': 5,
  'n_restarts': 5,
  'n_steps': 40,
  'log_lengthscale_prior_bounds': [-2.0, 1.0],
  'log_amplitude_prior_bounds': [-6.0, 1.5]
}
EI = {'n_restarts': 32, 'n_steps': 100}


def probes_from_scaled(x_scaled, dimension, n_probes):
  """``[0, 1]^(n_probes * d)`` -> ``(n_probes, d)`` in ``[-1, 1]^d``.

  The flat layout is ONE FIELD PER COORDINATE, each of length ``n_probes`` -- ``[x_1..x_n, y_1..y_n,
  ...]`` -- because that is the layout `detopt.bo._exchangeable_blocks` needs to find ``d`` blocks of
  ``n_probes`` indices to permute together.
  """
  return 2.0 * np.asarray(x_scaled, np.float64).reshape(dimension, n_probes).T - 1.0


def posterior_eigenvalues(probes, sigma):
  rows = np.concatenate([probes, np.ones((probes.shape[0], 1), np.float64)], axis=-1)
  eigenvalues = np.clip(np.linalg.eigvalsh(rows.T @ rows), 0.0, None)
  return 1.0 / (1.0 + eigenvalues / sigma**2)


def level_and_slack(probes, sigma, diff_over_level, window):
  """``(bayes_risk, predicted_slack)`` at one design -- the loss and the noise it is reported with."""
  mu = posterior_eigenvalues(probes, sigma)
  level = float(np.sum(mu) / mu.size)
  sd = float(np.sqrt(2.0 * np.sum(mu**2)) / mu.size)
  return level, diff_over_level * level + 2.0 * sd / np.sqrt(window)


def build_kernel(dimension, n_probes):
  design_dim = dimension * n_probes
  blocks = tuple(tuple(range(k * n_probes, (k + 1) * n_probes)) for k in range(dimension))
  amplitude_low, amplitude_high = GP['log_amplitude_prior_bounds']
  length_low, length_high = GP['log_lengthscale_prior_bounds']
  return NormalisedInvariantRBF(
    d=design_dim, blocks=blocks, constant_value=float(np.exp(amplitude_low + amplitude_high)),
    constant_value_bounds=(float(np.exp(2 * amplitude_low)), float(np.exp(2 * amplitude_high))),
    length_scale=float(np.exp(0.5 * (length_low + length_high))),
    length_scale_bounds=(float(np.exp(length_low)), float(np.exp(length_high)))
  )


def run(mode, dimension, n_probes, sigma, seed, n_iterations, diff_over_level, window):
  """One arm, one seed: the sequence of REPORTED losses, and the sequence of true Bayes risks."""
  design_dim = dimension * n_probes
  generator = np.random.default_rng(seed)
  optimiser = BayesianOptimizer(design_dim, gp=dict(GP), ei=dict(EI), kernel=build_kernel(dimension, n_probes), n_init=5)
  reported, truth = [], []
  for iteration in range(n_iterations):
    if mode == 'random':
      x_scaled = generator.random(design_dim)
    else:
      x_scaled = np.asarray(optimiser.propose(int(seed) + iteration), np.float64)
    probes = probes_from_scaled(x_scaled, dimension, n_probes)
    level, slack = level_and_slack(probes, sigma, diff_over_level, window)
    value = float(level + slack * generator.standard_normal())
    optimiser.append(x_scaled, value, noise=slack)
    reported.append(value)
    truth.append(level)
  return reported, truth


def criteria(reported, n, baseline, loss_precision):
  """The user's criterion, section 2.1, on ONE seed's sequence of reported losses."""
  at_n = float(np.min(reported[:n]))
  at_2n = float(np.min(reported[:2 * n]))
  improvement = at_n - at_2n
  return {
    'loss_at_n': at_n,
    'loss_at_2n': at_2n,
    'improvement': improvement,
    'a': bool(at_n < baseline),
    'b': bool(at_2n < at_n),
    'c': bool(improvement > 0.2 * (baseline - at_n)),
    'd': bool(improvement > 10.0 * loss_precision),
    'ten_loss_precision': 10.0 * loss_precision
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--landscape', default='output/linear-nd/landscape.json', help='stage 1a output: supplies the baseline')
  parser.add_argument(
    '--cells', nargs='+', default=[
      '1:2:0.3', '1:2:0.5', '1:2:0.7', '2:3:0.3', '2:3:0.5', '2:3:0.7', '3:4:0.3', '3:4:0.5', '3:4:0.7', '2:2:0.7', '3:3:0.7'
    ], metavar='D:N_PROBES:SIGMA'
  )
  parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
  parser.add_argument('--n', type=int, default=20, help='the criterion`s n; the run goes to 2n')
  parser.add_argument('--diff-over-level', type=float, default=0.018)
  parser.add_argument('--window', type=int, default=262144)
  parser.add_argument('--output', default='output/linear-nd/iteration.json')
  parser.add_argument('--resume', action='store_true')
  arguments = parser.parse_args()

  with open(arguments.landscape) as f:
    landscape = json.load(f)['cells']
  by_cell = {(int(c['d']), int(c['n_probes']), float(c['sigma'])): c for c in landscape}

  rows = []
  done = set()
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)['rows']
    done = {(r['d'], r['n_probes'], r['sigma'], r['mode'], r['seed']) for r in rows}
    print(f'resume: {len(done)} runs already in {arguments.output}')

  for spec in arguments.cells:
    dimension, n_probes, sigma = spec.split(':')
    dimension, n_probes, sigma = int(dimension), int(n_probes), float(sigma)
    reference = by_cell.get((dimension, n_probes, sigma))
    if reference is None:
      raise SystemExit(f'linear_nd_iteration: no landscape cell for d={dimension} n={n_probes} sigma={sigma}')
    baseline = float(reference['p50'])
    ceiling_probes = np.asarray(reference['ceiling_design'], np.float64)
    _, loss_precision = level_and_slack(ceiling_probes, sigma, arguments.diff_over_level, arguments.window)
    print(
      f"\n=== d={dimension} n_probes={n_probes} sigma={sigma} | design_dim={dimension * n_probes} "
      f"baseline={baseline:.4f} optimum={reference['optimum']:.4f} "
      f"loss_precision(predicted)={loss_precision:.5f} bar=10x={10 * loss_precision:.4f}", flush=True
    )
    for mode, seed in itertools.product(('bo', 'random'), arguments.seeds):
      if (dimension, n_probes, sigma, mode, seed) in done:
        continue
      reported, truth = run(
        mode, dimension, n_probes, sigma, seed, 2 * arguments.n, arguments.diff_over_level, arguments.window
      )
      # The per-`n` criteria stay INSIDE `criteria` and are never merged up into the row. Criterion
      # (d) is keyed `'d'` and so is the dimension: flattening them writes the boolean over the
      # dimension, which cost this study its cell labels once already.
      row = {
        'd': dimension,
        'n_probes': n_probes,
        'design_dim': dimension * n_probes,
        'sigma': sigma,
        'mode': mode,
        'seed': seed,
        'baseline': baseline,
        'optimum': float(reference['optimum']),
        'loss_precision': loss_precision,
        'best_true': float(np.min(truth)),
        'reported': [round(float(v), 6) for v in reported],
        'truth': [round(float(v), 6) for v in truth],
        # BOTH readings the doc allows, from the SAME run: `n >= 10` is its only requirement, and a
        # landscape that BO has already solved by 20 can still show its payoff over 10 -> 20. Nothing
        # is re-run -- `loss @ m` is a prefix minimum, so both are read off one 2n-long sequence.
        'criteria': {
          str(n): criteria(reported, n, baseline, loss_precision)
          for n in (10, arguments.n)
        }
      }
      rows.append(row)
      long, short = row['criteria'][str(arguments.n)], row['criteria']['10']
      print(
        f"  {mode:6s} seed {seed:3d}: loss@{arguments.n} {long['loss_at_n']:.4f} "
        f"loss@{2 * arguments.n} {long['loss_at_2n']:.4f} improvement {long['improvement']:+.4f} "
        f"| a{int(long['a'])} b{int(long['b'])} c{int(long['c'])} d{int(long['d'])} "
        f"|| n=10: {short['loss_at_n']:.4f} -> {short['loss_at_2n']:.4f} "
        f"({short['improvement']:+.4f}) a{int(short['a'])} b{int(short['b'])} "
        f"c{int(short['c'])} d{int(short['d'])}", flush=True
      )
      os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
      with open(arguments.output, 'w') as f:
        json.dump({'settings': vars(arguments), 'rows': rows}, f, default=float)

  print('\n=== VERDICT PER CELL (PASS <=> #strong > seeds/2 AND weak (a,b) on every seed) ===')
  for spec in arguments.cells:
    dimension, n_probes, sigma = spec.split(':')
    dimension, n_probes, sigma = int(dimension), int(n_probes), float(sigma)
    selected = [r for r in rows if (r['d'], r['n_probes'], r['sigma'], r['mode']) == (dimension, n_probes, sigma, 'bo')]
    if len(selected) == 0:
      continue
    paired = [r for r in rows if (r['d'], r['n_probes'], r['sigma'], r['mode']) == (dimension, n_probes, sigma, 'random')]
    for n in ('10', str(arguments.n)):
      strong = [r for r in selected if all(r['criteria'][n][k] for k in 'abcd')]
      weak = [r for r in selected if r['criteria'][n]['a'] and r['criteria'][n]['b']]
      bo_best = {r['seed']: r['criteria'][n]['loss_at_2n'] for r in selected}
      random_best = {r['seed']: r['criteria'][n]['loss_at_2n'] for r in paired}
      wins = sum(1 for s in bo_best if s in random_best and bo_best[s] < random_best[s])
      passed = len(strong) > len(selected) / 2 and len(weak) == len(selected)
      print(
        f"d={dimension} n_probes={n_probes} sigma={sigma} n={n}: strong {len(strong)}/{len(selected)} "
        f"weak {len(weak)}/{len(selected)} -> {'PASS' if passed else 'FAIL'} "
        f"| median improvement {np.median([r['criteria'][n]['improvement'] for r in selected]):+.4f} "
        f"vs bar {10 * selected[0]['loss_precision']:.4f} "
        f"| median loss@{n} {np.median([r['criteria'][n]['loss_at_n'] for r in selected]):.4f} "
        f"vs optimum {selected[0]['optimum']:.4f} "
        f"| BO beats random on {wins}/{len(random_best)} seeds"
      )
  print(f'\nwrote {arguments.output} ({len(rows)} runs)')


if __name__ == '__main__':
  main()
