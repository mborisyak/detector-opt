#!/usr/bin/env python3
"""Screen `linear` operating points against P(best@2n < best@n) > 0.75 on INDEPENDENT runs.

    srun --cpus-per-task=2 --mem=1800 -u python scripts/linear_criterion.py \
        --cells 3,4 4,5 4,6 --sigmas 0.3 0.5 0.7 1.0 --n-runs 64 \
        --output output/linear-criterion/screen.json

THE STATISTIC. A draw is ONE BO run of `--n-iterations` designs. `best@n` is that run's best-so-far
after `n` designs. The criterion compares TWO DIFFERENT runs: `P = P(best@2n from run j < best@n from
run i)`, `i != j`. Within one run the comparison is degenerate -- best-so-far is monotone -- so every
pair here is drawn from a different pair of runs, and the estimator is the ordered-pair U-statistic
over all `R (R - 1)` such pairs, which is unbiased because runs i and j are independent. Ties count a
half. Its standard error is a CLUSTER bootstrap over runs, which is what accounts for `best@n` and
`best@2n` being read off the same run for the diagonal that is excluded.

THE BONUS RATIOS, both read over the same independent pairs and both reported because the wording is
ambiguous:

    (a)  sd(best@n - best@2n) / sd(best@n)  =  sqrt(var_n + var_2n) / sd_n
    (b)  mean(best@n - best@2n) / sd(best@n)

(a) is >= 1 by construction for independent draws and grows when the spread at 2n grows, so it does
not read as a signal-to-noise ratio; (b) does. Both are printed side by side.

THE SCORER IS THE EXACT BAYES RISK -- `scripts/linear_exact_bo.py`'s `risk`, imported rather than
restated, and that script cross-checks it against `LinearDetector.bayes_risk` at d = 1. So this screen
measures the SEARCH alone and is OPTIMISTIC about the trainer: a neural run's reported loss carries a
convergence slack this does not have. It narrows candidates; it does not decide one.

THE LANDSCAPE anchors reported beside each cell: the optimum (multi-start L-BFGS-B on the box, plus
exhaustive corner enumeration where `(2^d)^n_probes` is small enough to enumerate), the ceiling (all
probes coincident at the origin, which `docs/linear-highdim.md` 1.3 proves is the worst design), and
the quantiles of a uniform design sample, whose median is the acceptance criterion's baseline. `span`
is baseline - optimum: a cell whose floor barely moves across the box cannot reward optimisation.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import scipy.optimize
import yaml

from linear_exact_bo import build_kernel, probes_from_scaled, risk

MAX_CORNER_ENUMERATION = 300000


def batch_risk(probes, sigma, dimension):
  """`risk` over a stack of designs `(..., n_probes, d)`, batched through `eigvalsh`."""
  probes = np.asarray(probes, np.float64)
  rows = np.concatenate([probes, np.ones(probes.shape[:-1] + (1, ), np.float64)], axis=-1)
  gram = np.einsum('...ij,...ik->...jk', rows, rows)
  eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
  return np.sum(1.0 / (1.0 + eigenvalues / sigma**2), axis=-1) / (dimension + 1)


def landscape(dimension, n_probes, sigma, n_restarts, n_draws, seed):
  """Optimum, ceiling and the uniform-design quantiles at one cell."""
  generator = np.random.default_rng(seed)
  width = n_probes * dimension
  starts = np.concatenate([generator.uniform(-1.0, 1.0, (int(n_restarts), width)), np.zeros((1, width))], axis=0)
  best, best_value = None, np.inf
  for start in starts:
    result = scipy.optimize.minimize(
      lambda flat: risk(flat.reshape(n_probes, dimension), sigma, dimension), start, method='L-BFGS-B',
      bounds=[(-1.0, 1.0)] * width
    )
    if float(result.fun) < best_value:
      best, best_value = np.asarray(result.x, np.float64).reshape(n_probes, dimension), float(result.fun)

  corner_value = float('nan')
  if (2**dimension)**n_probes <= MAX_CORNER_ENUMERATION:
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=dimension)), np.float64)
    index = np.array(list(itertools.product(range(corners.shape[0]), repeat=n_probes)), np.int64)
    corner_value = float(np.min(batch_risk(corners[index], sigma, dimension)))

  sample = generator.uniform(-1.0, 1.0, size=(int(n_draws), n_probes, dimension))
  risks = batch_risk(sample, sigma, dimension)
  quantiles = {f'p{int(100 * q):02d}': float(np.quantile(risks, q)) for q in (0.01, 0.10, 0.50, 0.90)}
  return {
    'optimum': best_value,
    'optimum_design': [[float(v) for v in probe] for probe in best],
    'optimum_corner_enumeration': corner_value,
    'optimum_is_corner': bool(np.all(np.abs(np.abs(best) - 1.0) < 1.0e-6)),
    'ceiling': risk(np.zeros((n_probes, dimension)), sigma, dimension),
    'baseline': quantiles['p50'],
    'quantiles': quantiles,
    'span': quantiles['p50'] - best_value,
    'structural_floor': max(0.0, (dimension + 1 - n_probes)) / (dimension + 1.0)
  }


def bo_run(bo_config, dimension, n_probes, sigma, seed, n_iterations, observation_noise, arm):
  """One independent run: the exact loss of every design it evaluated, in order.

  The seed stream is `scripts/bo.py`'s -- one `SeedSequence` split into a network branch and an
  iteration branch -- so a run here sees the proposals the real driver would see at the same seed."""
  from detopt.bo import BayesianOptimizer

  gp_config = {k: v for k, v in bo_config['gp'].items() if k != 'kernel'}
  width = dimension * n_probes
  optimiser = BayesianOptimizer(
    width, gp=gp_config, ei=dict(bo_config['ei']), kernel=build_kernel(bo_config, dimension, n_probes),
    n_init=int(bo_config.get('n_init', gp_config['n_folds']))
  )
  _, iteration_sequence = np.random.SeedSequence(int(seed)).spawn(2)
  values = []
  for _ in range(int(n_iterations)):
    iteration_seed = int(iteration_sequence.spawn(1)[0].generate_state(1)[0])
    if arm == 'random':
      x_scaled = np.random.default_rng(iteration_seed).random(width)
    else:
      x_scaled = np.asarray(optimiser.propose(iteration_seed), np.float64)
    value = risk(probes_from_scaled(x_scaled, dimension, n_probes), sigma, dimension)
    optimiser.append(x_scaled, value, noise=observation_noise)
    values.append(float(value))
  return values


def pair_probability(at_n, at_2n):
  """P(best@2n < best@n) over every ORDERED pair of DIFFERENT runs; ties count a half."""
  at_n, at_2n = np.asarray(at_n, np.float64), np.asarray(at_2n, np.float64)
  wins = (at_2n[None, :] < at_n[:, None]).astype(np.float64) + 0.5 * (at_2n[None, :] == at_n[:, None])
  diagonal = float(np.sum(np.diagonal(wins)))
  size = at_n.size
  return float((np.sum(wins) - diagonal) / (size * (size - 1)))


def criterion(at_n, at_2n, n_bootstrap, seed):
  """The probability with its cluster-bootstrap standard error, and both bonus ratios."""
  at_n, at_2n = np.asarray(at_n, np.float64), np.asarray(at_2n, np.float64)
  probability = pair_probability(at_n, at_2n)
  generator = np.random.default_rng(seed)
  draws = np.array([
    pair_probability(at_n[index], at_2n[index])
    for index in generator.integers(0, at_n.size, size=(int(n_bootstrap), at_n.size))
  ], np.float64)
  spread_n = float(np.std(at_n, ddof=1))
  spread_2n = float(np.std(at_2n, ddof=1))
  difference = float(np.mean(at_n) - np.mean(at_2n))
  return {
    'n_runs': int(at_n.size),
    'P': probability,
    'P_se': float(np.std(draws, ddof=1)),
    'P_lo': float(np.quantile(draws, 0.025)),
    'P_hi': float(np.quantile(draws, 0.975)),
    'median_at_n': float(np.median(at_n)),
    'median_at_2n': float(np.median(at_2n)),
    'mean_at_n': float(np.mean(at_n)),
    'mean_at_2n': float(np.mean(at_2n)),
    'sd_at_n': spread_n,
    'sd_at_2n': spread_2n,
    'ratio_a': float(np.hypot(spread_n, spread_2n) / spread_n) if spread_n > 0.0 else float('nan'),
    'ratio_b': float(difference / spread_n) if spread_n > 0.0 else float('nan'),
    'mean_gain': difference
  }


def normal_probability(at_n, at_2n):
  """`P` again, under a NORMAL model for the two run distributions, with a delta-method error.

  The U-statistic saturates: when every run's `best@2n` is below every run's `best@n` it returns
  exactly 1 and the cluster bootstrap returns the degenerate interval [1, 1], which says the effect is
  large but cannot be compared against 0.75. `P = Phi(delta / sqrt(var_n + var_2n))` is finite there.
  It is a MODEL and is reported beside the model-free estimate, never instead of it."""
  import scipy.stats

  at_n, at_2n = np.asarray(at_n, np.float64), np.asarray(at_2n, np.float64)
  size = at_n.size
  variance_n, variance_2n = float(np.var(at_n, ddof=1)), float(np.var(at_2n, ddof=1))
  spread = float(np.sqrt(variance_n + variance_2n))
  if not spread > 0.0:
    return {'P_normal': float('nan'), 'P_normal_se': float('nan')}
  z = (float(np.mean(at_n)) - float(np.mean(at_2n))) / spread
  # var(z) from the means (var/size each) and from the two variances (2 var^2/(size - 1) each),
  # propagated through z = delta / sqrt(v_n + v_2n).
  variance_of_delta = variance_n / size + variance_2n / size
  variance_of_spread_squared = 2.0 * variance_n**2 / (size - 1) + 2.0 * variance_2n**2 / (size - 1)
  variance_of_z = variance_of_delta / spread**2 + (z / (2.0 * spread**2))**2 * variance_of_spread_squared
  return {'P_normal': float(scipy.stats.norm.cdf(z)), 'P_normal_se': float(scipy.stats.norm.pdf(z) * np.sqrt(variance_of_z))}


def read_runs(directory, arm):
  """Every finished `scripts/bo.py` trajectory under `<directory>/<seed>/<arm>/results.json`.

  Returns `(seeds, losses)` with `losses` a list of per-design reported losses in evaluation order --
  the same number a campaign reports, not a re-scoring."""
  import glob

  import detopt.utils.io as io

  seeds, losses = [], []
  for path in sorted(glob.glob(os.path.join(directory, '*', arm, 'results.json'))):
    with open(path) as f:
      data = json.load(f)
    rows = io.complete_results(data['results'])
    seeds.append(path.split(os.sep)[-3])
    losses.append([float(r['loss']) for r in rows])
  return seeds, losses


def report_runs(directory, arm, pairs, n_bootstrap, seed):
  """The criterion over a tree of finished runs, printed and returned."""
  seeds, losses = read_runs(directory, arm)
  if len(losses) == 0:
    raise SystemExit(f'linear_criterion: no {arm} runs under {directory}')
  lengths = [len(row) for row in losses]
  print(f'{arm}: {len(losses)} runs under {directory}, designs per run {min(lengths)}-{max(lengths)}')
  out = {'directory': directory, 'arm': arm, 'seeds': seeds, 'n_designs': lengths, 'criterion': {}}
  for n in pairs:
    usable = [row for row in losses if len(row) >= 2 * n]
    short = len(losses) - len(usable)
    if len(usable) < 4:
      print(f'  {n} -> {2 * n}: only {len(usable)} runs reached {2 * n} designs -- not evaluated')
      continue
    at_n = np.array([min(row[:n]) for row in usable], np.float64)
    at_2n = np.array([min(row[:2 * n]) for row in usable], np.float64)
    result = criterion(at_n, at_2n, n_bootstrap, seed + n)
    result.update(normal_probability(at_n, at_2n))
    result['n_short'] = short
    result['at_n'] = [float(v) for v in at_n]
    result['at_2n'] = [float(v) for v in at_2n]
    out['criterion'][str(n)] = result
    print(
      f"  {n:>2} -> {2 * n:<3} R {result['n_runs']:>3} (short {short})  P {result['P']:.3f} +- {result['P_se']:.3f} "
      f"[{result['P_lo']:.3f}, {result['P_hi']:.3f}]  P_normal {result['P_normal']:.3f} +- {result['P_normal_se']:.3f}\n"
      f"            median {result['median_at_n']:.5f} -> {result['median_at_2n']:.5f}  "
      f"mean gain {result['mean_gain']:.5f}  sd@n {result['sd_at_n']:.5f}  sd@2n {result['sd_at_2n']:.5f}  "
      f"ratio(a) {result['ratio_a']:.3f}  ratio(b) {result['ratio_b']:.3f}"
    )
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--from-runs', default=None, metavar='DIRECTORY',
    help='score FINISHED neural runs instead of running the exact scorer: reads every '
    '`<DIRECTORY>/<seed>/<arm>/results.json` and applies the same statistic to their '
    'reported losses'
  )
  parser.add_argument('--arms', nargs='+', default=['from_scratch'], help='arms to score under --from-runs')
  parser.add_argument('--cells', nargs='+', default=['3,4', '4,5'], metavar='D,N_PROBES')
  parser.add_argument('--sigmas', type=float, nargs='+', default=[0.3, 0.5, 0.7, 1.0])
  parser.add_argument('--n-runs', type=int, default=64, help='INDEPENDENT BO runs per cell')
  parser.add_argument('--seed0', type=int, default=1000, help='run seeds are seed0 .. seed0 + n_runs - 1')
  parser.add_argument(
    '--seed-list', type=int, nargs='+', default=None,
    help='the exact run seeds, overriding `--seed0`/`--n-runs`. Pass a neural campaign`s own seeds to '
    'get a PAIRED comparison in which only the scorer differs'
  )
  parser.add_argument('--n-iterations', type=int, default=40, help='designs per run; must cover the largest 2n')
  parser.add_argument('--pairs', type=int, nargs='+', default=[5, 10, 20], help='the n of each n -> 2n doubling')
  parser.add_argument('--arm', default='bo', choices=['bo', 'random'])
  parser.add_argument('--observation-noise', type=float, default=0.0)
  parser.add_argument('--run-config', default='config/linear.yaml', help='supplies the `bo` block, kernel included')
  parser.add_argument(
    '--log-lengthscale-bounds', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
    help='override `bo.gp.log_lengthscale_prior_bounds`. NATURAL log (jax_gp.py exponentiates it), '
    'so the spread must satisfy HIGH - LOW >= ln 2 = 0.6931'
  )
  parser.add_argument('--optimum-restarts', type=int, default=192)
  parser.add_argument('--baseline-draws', type=int, default=100000)
  parser.add_argument('--n-bootstrap', type=int, default=4000)
  parser.add_argument('--landscape-only', action='store_true')
  parser.add_argument('--output', default='output/linear-criterion/screen.json')
  arguments = parser.parse_args()

  if arguments.from_runs is not None:
    scored = [
      report_runs(arguments.from_runs, arm, arguments.pairs, arguments.n_bootstrap, arguments.seed0) for arm in arguments.arms
    ]
    os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
    with open(arguments.output, 'w') as f:
      json.dump({'from_runs': arguments.from_runs, 'arms': scored}, f, indent=1)
    print(f'\nwrote {arguments.output}')
    return

  with open(arguments.run_config) as f:
    bo_config = yaml.safe_load(f)['bo']
  if arguments.log_lengthscale_bounds is not None:
    low, high = (float(v) for v in arguments.log_lengthscale_bounds)
    if not high - low >= np.log(2.0):
      raise SystemExit(f'linear_criterion: log-lengthscale spread {high - low:.4f} is below ln 2 = {np.log(2.0):.4f}')
    bo_config['gp']['log_lengthscale_prior_bounds'] = [low, high]

  seeds = ([int(s) for s in arguments.seed_list]
           if arguments.seed_list is not None else [arguments.seed0 + k for k in range(arguments.n_runs)])

  os.makedirs(os.path.dirname(arguments.output) or '.', exist_ok=True)
  record = {
    'settings': {
      'seeds': seeds,
      'cells': arguments.cells,
      'sigmas': [float(s) for s in arguments.sigmas],
      'n_runs': int(arguments.n_runs),
      'seed0': int(arguments.seed0),
      'n_iterations': int(arguments.n_iterations),
      'pairs': [int(n) for n in arguments.pairs],
      'arm': arguments.arm,
      'observation_noise': float(arguments.observation_noise),
      'run_config': arguments.run_config,
      'bo': bo_config
    },
    'cells': []
  }

  def publish():
    temporary = arguments.output + '.partial'
    with open(temporary, 'w') as f:
      json.dump(record, f)
    os.replace(temporary, arguments.output)

  for spec in arguments.cells:
    dimension, n_probes = (int(v) for v in spec.split(','))
    for sigma in arguments.sigmas:
      started = time.time()
      anchors = landscape(
        dimension, n_probes, float(sigma), arguments.optimum_restarts, arguments.baseline_draws, arguments.seed0
      )
      print(
        f"\n=== d {dimension} | {n_probes} probes | sigma {sigma:g} | design dim {dimension * n_probes}\n"
        f"    optimum {anchors['optimum']:.6f} (corner {anchors['optimum_corner_enumeration']:.6f}, "
        f"all-corner design {anchors['optimum_is_corner']}) | baseline {anchors['baseline']:.6f} | "
        f"ceiling {anchors['ceiling']:.6f} | span {anchors['span']:.6f} | "
        f"structural floor {anchors['structural_floor']:.4f} | {time.time() - started:.0f} s", flush=True
      )
      cell = {'n_dimensions': dimension, 'n_probes': n_probes, 'sigma': float(sigma), 'landscape': anchors}
      record['cells'].append(cell)
      publish()
      if arguments.landscape_only:
        continue

      started = time.time()
      traces = []
      for offset, run_seed in enumerate(seeds):
        one = time.time()
        traces.append(
          bo_run(
            bo_config, dimension, n_probes, float(sigma), run_seed, arguments.n_iterations, arguments.observation_noise,
            arguments.arm
          )
        )
        print(
          f"    run {offset + 1:>3}/{len(seeds)} seed {run_seed} "
          f"best {min(traces[-1]):.6f} {time.time() - one:.0f} s", flush=True
        )
        # PUBLISHED AFTER EVERY RUN, not once at the end: this machine is shared and a cell that is
        # cut short must still leave the runs it paid for, keyed by how many went into them.
        incumbent = np.minimum.accumulate(np.asarray(traces, np.float64), axis=1)
        cell['incumbent'] = [[round(float(v), 12) for v in row] for row in incumbent]
        cell['criterion'] = {}
        if len(traces) >= 4:
          for n in arguments.pairs:
            if 2 * n > arguments.n_iterations:
              continue
            cell['criterion'][
              str(n)] = criterion(incumbent[:, n - 1], incumbent[:, 2 * n - 1], arguments.n_bootstrap, arguments.seed0 + n)
        publish()
      for n, result in cell['criterion'].items():
        print(
          f"    {n:>2} -> {2 * int(n):<3}  P {result['P']:.3f} +- {result['P_se']:.3f} "
          f"[{result['P_lo']:.3f}, {result['P_hi']:.3f}]  "
          f"median {result['median_at_n']:.5f} -> {result['median_at_2n']:.5f}  "
          f"sd@n {result['sd_at_n']:.5f}  ratio(a) {result['ratio_a']:.3f}  ratio(b) {result['ratio_b']:.3f}", flush=True
        )
      print(f"    {len(seeds)} runs in {time.time() - started:.0f} s", flush=True)

  print(f"\nwrote {arguments.output}")


if __name__ == '__main__':
  main()
