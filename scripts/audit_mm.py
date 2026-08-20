"""READ-ONLY audit of the MM (Michaelis-Menten) detector and its campaign. Runs NOTHING that trains.

Sections, selected with ``--section``:

* ``logs``    -- parses every ``run.log`` of a campaign directory into per-design records (window, the
                 convergence rule's own ``diff``/``err``/posterior probabilities, growth rounds) and
                 compares the arms. Pure text, no jax.
* ``ceiling`` -- the no-information level of the normalised target, measured rather than asserted:
                 the mean and variance of ``normalize_target`` over events drawn straight from the
                 detector, against the 0 and 1/3 a uniform prior on [-1, 1] gives.
* ``determinism`` -- the detector is a function of ``(design, event_index)`` alone: same index twice,
                 same index under two designs (target must not move), and a re-run in a separate
                 process (checked by writing a digest the caller compares).
* ``guard``   -- the dt / dt-half integration guard: the reported error over the design-box corners
                 at the shipped step count, and whether coarsening the step count makes it FIRE.

All calls are CPU-only by construction of the caller's environment; nothing here touches a GPU or a
checkpoint. Design coordinates are always the detector's own ``_to_nominal_flat`` of a scaled point,
so no bound is hard-coded here.
"""

import argparse
import glob
import json
import math
import os
import re

import jax.nn
import numpy as np

CONVERGED = re.compile(
  r'\[converged/(?P<rule>\w+)\] train=(?P<train>[-\d.]+) val=(?P<val>[-\d.]+) diff=(?P<diff>[-\d.]+) '
  r'err=(?P<err>[-\d.]+) prec=(?P<prec>[-\d.]+) \| P\(gap>LP\)=(?P<p_gap>[-\d.]+) '
  r'P\(settled\)=(?P<p_settled>[-\d.]+) \| window=(?P<window>\d+)'
)
GROW = re.compile(r'\[grow\] window -> (?P<window>\d+), pool (?P<current>\d+)/(?P<capacity>\d+)')
HEADER = re.compile(r'budget=(?P<budget>\d+) detector calls, n_init=(?P<n_init>\d+), d=(?P<d>\d+)')


def parse_log(path):
  """One ``run.log`` -> ``(header, [per-design record])``. A design's record is the ``[grow]`` lines
  that precede its ``[converged]`` line plus that line's own fields."""
  header, records, grows = {}, [], []
  with open(path) as handle:
    for line in handle:
      match = HEADER.search(line)
      if match is not None:
        header = {key: int(value) for key, value in match.groupdict().items()}
        continue
      match = GROW.search(line)
      if match is not None:
        grows.append({key: int(value) for key, value in match.groupdict().items()})
        continue
      match = CONVERGED.search(line)
      if match is not None:
        record = {key: (value if key == 'rule' else float(value)) for key, value in match.groupdict().items()}
        record['window'] = int(record['window'])
        record['n_grow'] = len(grows)
        record['pool_capacity'] = grows[-1]['capacity'] if len(grows) > 0 else None
        records.append(record)
        grows = []
  return header, records


def quantiles(values):
  values = np.asarray(values, np.float64)
  if values.size == 0:
    return 'empty'
  return (
    f'n={values.size:3d} min {values.min():9.4f} p50 {np.median(values):9.4f} '
    f'mean {values.mean():9.4f} max {values.max():9.4f}'
  )


def section_logs(arguments):
  for root in arguments.campaign:
    print('=' * 108)
    print(f'LOGS  {root}')
    print('=' * 108)
    by_arm = {}
    for path in sorted(glob.glob(os.path.join(root, '*', '*', 'run.log'))):
      arm = os.path.basename(os.path.dirname(path))
      seed = os.path.basename(os.path.dirname(os.path.dirname(path)))
      header, records = parse_log(path)
      if len(records) == 0:
        continue
      by_arm.setdefault(arm, []).extend((seed, record) for record in records)
      results_path = os.path.join(os.path.dirname(path), 'results.json')
      completed = json.load(open(results_path))['completed'] if os.path.exists(results_path) else None
      print(
        f'  {seed:>11s} / {arm:<12s} budget={header.get("budget")} pool_cap='
        f'{records[0]["pool_capacity"]} designs={len(records)} completed={completed}'
      )
    print()
    for arm, entries in sorted(by_arm.items()):
      records = [record for _, record in entries]
      rounds = np.array([1 + record['n_grow'] for record in records], np.float64)
      window = np.array([record['window'] for record in records], np.float64)
      print(f'  --- {arm} ---')
      print(f'    rounds   {quantiles(rounds)}')
      print(f'    window   {quantiles(window)}')
      for key in ('train', 'val', 'diff', 'err', 'p_gap', 'p_settled'):
        print(f'    {key:<9s}{quantiles([record[key] for record in records])}')
      first_round = int((rounds == 1).sum())
      print(f'    stopped at round 1: {first_round}/{len(records)}')
      at_bar = np.array([record['err'] for record in records], np.float64)
      bar = records[0]['prec']
      print(f'    err within 10% of prec {bar}: {int((at_bar > 0.9 * bar).sum())}/{len(records)}')
      binding = int(sum(1 for record in records if record['err'] > record['diff']))
      print(f'    err > diff (error term binds): {binding}/{len(records)}')
      print()


def build_detector(config_path):
  import detopt.detector
  from detopt.utils.config import load_config
  config = load_config(config_path)
  (name, ) = [key for key in config if key != 'defaults']
  return detopt.detector.from_config({name: config[name]})


def section_ceiling(arguments):
  import jax.numpy as jnp
  detector = build_detector(arguments.config)
  scaled = np.full((int(detector.design_dim()), ), 0.5, np.float32)
  design = np.asarray(detector._to_nominal_flat(jnp.asarray(scaled)), np.float32)
  index = np.arange(arguments.n_events, dtype=np.int64)
  _ground_truth, _event, _mask, target = detector(design, index)
  normalised = np.asarray(detector.normalize_target(target), np.float64)
  print('=' * 108)
  print(f'CEILING  {arguments.config}  target={detector.target_parameters}  n={arguments.n_events}')
  print('=' * 108)
  print(f'  target_bounds       {detector.target_bounds}')
  print(f'  prior ranges        {[dict(detector.parameter_ranges)[n] for n in detector.target_parameters]}')
  print(f'  normalised min/max  {normalised.min(axis=0)} / {normalised.max(axis=0)}')
  print(f'  normalised mean     {normalised.mean(axis=0)}   (uniform prior: 0)')
  print(f'  normalised variance {normalised.var(axis=0)}   (uniform prior: {1/3:.4f})')
  constant = float(np.mean(np.square(normalised)))
  print(f'  loss of predicting 0 (the prior mean): {constant:.6f}')
  best_constant = float(np.mean(np.square(normalised - normalised.mean(axis=0))))
  print(f'  loss of the best CONSTANT on this sample: {best_constant:.6f}')
  spread = np.std(np.square(normalised).mean(axis=1)) / math.sqrt(normalised.shape[0])
  print(f'  standard error of the sample ceiling at n={arguments.n_events}: {spread:.6f}')
  print(f'  correlation of the normalised target components:\n{np.corrcoef(normalised.T)}')


def section_determinism(arguments):
  detector = build_detector(arguments.config)
  import jax.numpy as jnp
  rng = np.random.default_rng(0)
  scaled_a = rng.random(int(detector.design_dim())).astype(np.float32)
  scaled_b = rng.random(int(detector.design_dim())).astype(np.float32)
  design_a = np.asarray(detector._to_nominal_flat(jnp.asarray(scaled_a)), np.float32)
  design_b = np.asarray(detector._to_nominal_flat(jnp.asarray(scaled_b)), np.float32)
  index = np.arange(arguments.n_events, dtype=np.int64)

  print('=' * 108)
  print(f'DETERMINISM  {arguments.config}  n={arguments.n_events}')
  print('=' * 108)
  truth_1, event_1, _m, target_1 = detector(design_a, index)
  truth_2, event_2, _m, target_2 = detector(design_a, index)
  same = bool(np.array_equal(np.asarray(event_1.measurements), np.asarray(event_2.measurements)))
  print(f'  same design, same indices, twice -> events bit-identical: {same}')
  _t3, event_3, _m, target_3 = detector(design_b, index)
  moved = float(np.max(np.abs(np.asarray(target_1.coefficients) - np.asarray(target_3.coefficients))))
  print(f'  design changed -> max |target moved|: {moved:.3g}   (must be exactly 0)')
  gt_moved = float(np.max(np.abs(np.asarray(truth_1.parameters) - np.asarray(truth_2.parameters))))
  print(f'  ground truth reproducible: max |diff| {gt_moved:.3g}')
  shuffled = index[::-1].copy()
  _t4, event_4, _m, _target_4 = detector(design_a, shuffled)
  reordered = float(np.max(np.abs(np.asarray(event_4.measurements)[::-1] - np.asarray(event_1.measurements))))
  print(f'  index order does not matter: max |diff| {reordered:.3g}')
  digest = float(np.sum(np.asarray(event_1.measurements, np.float64)))
  print(f'  CROSS-PROCESS DIGEST (compare across separate runs): {digest!r}')
  print(f'  event[0] first row: {np.asarray(event_1.measurements)[0, 0, :4]}')


def section_guard(arguments):
  import jax.numpy as jnp
  import itertools
  detector = build_detector(arguments.config)
  bounds = [pair for _, pair, _ in detector._design_layout]
  m = detector.n_experiments
  print('=' * 108)
  print(
    f'GUARD  {arguments.config}  tolerance={detector.integration_tolerance}  '
    f'n_measurements={detector.n_measurements} n_steps_per_measurement={detector.n_steps_per_measurement} '
    f'total_steps={detector.n_measurements * detector.n_steps_per_measurement} dt={detector.measurement_dt:.3e}'
  )
  print('=' * 108)
  index = np.arange(arguments.n_events, dtype=np.int64)

  def measure(det, design):
    _fraction = None
    fraction, temperature, initial_A, initial_B = det._resolve_design(jnp.asarray(design), len(index))
    _meas, _coef, _par, _half, error = det._generate(fraction, temperature, initial_A, initial_B, jnp.asarray(index, jnp.int32))
    return float(jnp.max(error))

  worst, worst_name = -1.0, None
  for choice in itertools.product((0, 1), repeat=len(bounds)):
    name = ''.join('H' if side == 1 else 'L' for side in choice)
    design = np.concatenate([np.full(m, bounds[k][side], np.float32) for k, side in enumerate(choice)])
    error = measure(detector, design)
    flag = 'FIRES' if error > detector.integration_tolerance else ''
    print(f'  corner {name}  error {error:.4e}   margin x{detector.integration_tolerance / max(error, 1e-30):8.1f}  {flag}')
    if error > worst:
      worst, worst_name = error, name
  rng = np.random.default_rng(arguments.seed)
  randoms = rng.random((arguments.n_designs, int(detector.design_dim()))).astype(np.float32)
  errors = [measure(detector, np.asarray(detector._to_nominal_flat(jnp.asarray(row)), np.float32)) for row in randoms]
  print(f'  {arguments.n_designs} random designs: max {max(errors):.4e}  median {np.median(errors):.4e}')
  print(
    f'  WORST CORNER {worst_name}: {worst:.4e} against tolerance {detector.integration_tolerance:.4e} '
    f'(margin x{detector.integration_tolerance / worst:.1f})'
  )

  print('\n  step-count scan at the worst corner -- does the guard FIRE when dt is coarsened?')
  design = np.concatenate([np.full(m, bounds[k][1 if letter == 'H' else 0], np.float32) for k, letter in enumerate(worst_name)])
  for steps in arguments.step_scan:
    coarse = build_detector(arguments.config)
    coarse.n_steps_per_measurement = int(steps)
    coarse.measurement_dt = coarse.duration / (coarse.n_measurements * steps)
    import jax
    coarse._generate = jax.jit(jax.vmap(coarse._event))
    error = measure(coarse, design)
    flag = 'FIRES' if error > detector.integration_tolerance else 'passes'
    print(f'    n_steps_per_measurement {steps:5d} (total {steps * coarse.n_measurements:6d})  '
          f'error {error:.4e}  {flag}')


def section_identifiability(arguments):
  """Per-component posterior variance of the target over a SOBOL SAMPLE of the design box.

  The GOOD/BAD pair of ``scripts/validate_mm.py --mode fim`` answers "is the target measurable at a
  design written down from the prior"; this answers "is any component near-unidentifiable EVERYWHERE",
  which is what pins the loss near the ceiling regardless of design. Same Fisher construction, same
  marginalisation of the nuisances against their own prior."""
  import jax.numpy as jnp
  from scipy.stats import qmc
  sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
  if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)
  from validate_mm import fisher_information, marginal_posterior

  detector = build_detector(arguments.config)
  names = list(detector.target_parameters)
  print('=' * 108)
  print(
    f'IDENTIFIABILITY  {arguments.config}  target={tuple(names)}  '
    f'{arguments.n_designs} Sobol designs x {arguments.n_fim_draws} prior draws'
  )
  print('=' * 108)
  scaled = qmc.Sobol(d=int(detector.design_dim()), scramble=True, seed=arguments.seed).random(arguments.n_designs)
  rows, predicted = [], []
  latent_names = None
  for i, point in enumerate(scaled.astype(np.float32)):
    design = np.asarray(detector._to_nominal_flat(jnp.asarray(point)), np.float32)
    latent_names, information = fisher_information(detector, design, arguments.n_fim_draws, arguments.seed)
    index = [latent_names.index(name) for name in names]
    posterior = np.mean([marginal_posterior(single, index)[1] for single in information], axis=0)
    rows.append(posterior)
    predicted.append(posterior.mean())
  rows = np.asarray(rows)
  print(f'  prior variance of every component: {1/3:.4f}  (a component AT it was not measured at all)')
  for k, name in enumerate(names):
    column = rows[:, k]
    print(
      f'    {name:<10s} min {column.min():.4f}  p10 {np.percentile(column, 10):.4f}  '
      f'p50 {np.median(column):.4f}  max {column.max():.4f}   best reduction '
      f'{100 * (1 - column.min() / (1/3)):5.1f}%'
    )
  predicted = np.asarray(predicted)
  print(
    f'  PREDICTED LOSS over the box: min {predicted.min():.4f}  p10 {np.percentile(predicted, 10):.4f}  '
    f'p50 {np.median(predicted):.4f}  max {predicted.max():.4f}'
  )
  floor = rows.min(axis=0).sum() / len(names)
  print(f'  loss floor if every component reached its OWN best design simultaneously: {floor:.4f}')
  best = int(np.argmin(predicted))
  print(
    f'  best single design: predicted loss {predicted[best]:.4f}  components ' +
    '  '.join(f'{name} {rows[best, k]:.4f}' for k, name in enumerate(names))
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--section', nargs='*', default=['logs'], choices=['logs', 'ceiling', 'determinism', 'guard', 'identifiability']
  )
  parser.add_argument('--n-fim-draws', type=int, default=16)
  parser.add_argument('--campaign', nargs='*', default=['output/campaign-mm-p5e3'])
  parser.add_argument('--config', default='config/detector/enzyme_mm_sym_m3.yaml')
  parser.add_argument('--n-events', type=int, default=4096)
  parser.add_argument('--n-designs', type=int, default=16)
  parser.add_argument('--step-scan', type=int, nargs='*', default=[10, 20, 40, 80, 160])
  parser.add_argument('--seed', type=int, default=0)
  arguments = parser.parse_args()
  for section in arguments.section:
    {
      'logs': section_logs,
      'ceiling': section_ceiling,
      'determinism': section_determinism,
      'guard': section_guard,
      'identifiability': section_identifiability
    }[section](arguments)


if __name__ == '__main__':
  main()
