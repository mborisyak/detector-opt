#!/usr/bin/env python3
"""Bayesian optimisation of the binary inhibitor-MECHANISM design on the SEMI-ANALYTIC instrument.

    python scripts/analytic_extremes_bo.py --detector-config config/detector/extremes_tuned_m2.yaml \
        --n-seeds 64 --seed-start 0 --n-iterations 20 --output output/extremes-analytic/m2-bo.json

CPU only, no network, no trained model. One evaluation is a batch of detector calls plus a matrix
product, so the same question can be asked over hundreds of seeds; a neural evaluation of one design
costs the better part of an hour, which is the whole reason this driver exists.

The reported figure of merit is

    (E[loss @ n_first designs] - E[loss @ n_second designs]) / loss_of_a_guess

with the expectation over seeds and ``loss_of_a_guess = 1.0`` exactly -- the binary task's uniform
prediction scores 1.0 on every sample by construction. Two versions are written: ``observed``, the
best value the run itself saw (what a campaign's ``results.json`` reports), and ``rescored``, the same
winning design re-evaluated on an INDEPENDENT event stream, which carries no winner's curse.

Every arm sees the same instrument at the same seed, so ``bo`` and ``random`` are paired.
"""

import argparse
import json
import math
import os
import time

import numpy as np

import jax

jax.config.update('jax_platform_name', 'cpu')

import detopt.bo
from detopt.analytic import MechanismInstrument, build_detector
from detopt.bo import BayesianOptimizer, kernel_from_config
from detopt.utils.config import load_config

GP = {
  'n_folds': 5,
  'n_restarts': 5,
  'n_steps': 40,
  'log_lengthscale_prior_bounds': [-2.0, 1.0],
  'log_amplitude_prior_bounds': [-6.0, 1.5]
}
EI = {'n_restarts': 32, 'n_steps': 100}


def parse_overrides(items):
  overrides = {}
  for item in items:
    key, _, value = item.partition('=')
    try:
      parsed = json.loads(value)
    except json.JSONDecodeError:
      parsed = value
    overrides[key] = tuple(parsed) if isinstance(parsed, list) else parsed
  return overrides


def make_instrument(detector, arguments, seed):
  """One instrument on a SHARED detector: the detector's jitted kernels compile once per process,
  so a new one per seed would spend more time in the compiler than in the objective."""
  return MechanismInstrument(
    detector, noise=arguments.noise, n_events=arguments.n_events, n_library=arguments.n_library, n_noise=arguments.n_noise,
    hedge=arguments.hedge, seed=seed
  )


def run_arm(arm, detector, instrument, arguments, seed):
  dimension = detector.design_dim()
  generator = np.random.default_rng(0x5EED0000 + seed)
  kernel = kernel_from_config({arguments.kernel: {'exchangeable': detector.n_experiments}}, detector, GP)
  optimiser = BayesianOptimizer(dimension, gp=GP, ei=EI, kernel=kernel, n_init=arguments.n_init)
  designs, losses, errors = [], [], []
  for iteration in range(arguments.n_iterations):
    if arm == 'random':
      design = generator.random(dimension)
    else:
      design = np.asarray(optimiser.propose(int(seed) * 100003 + iteration), np.float64)
    result = instrument.evaluate(design)
    optimiser.append(design, result.loss, noise=max(result.standard_error, 1.0e-6))
    designs.append(np.asarray(design, np.float64).tolist())
    losses.append(result.loss)
    errors.append(result.standard_error)
  return {'designs': designs, 'losses': losses, 'standard_errors': errors}


def rescore(record, audit, cutoffs):
  """The design that was best after each cutoff, re-evaluated on an independent event stream."""
  losses = np.asarray(record['losses'])
  out = {}
  for cutoff in cutoffs:
    index = int(np.argmin(losses[:cutoff]))
    out[str(cutoff)] = {
      'observed': float(losses[index]),
      'index': index,
      'rescored': float(audit.evaluate(np.asarray(record['designs'][index])).loss)
    }
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--detector-config', required=True)
  parser.add_argument('--override', action='append', default=[])
  parser.add_argument('--noise', type=float, required=True)
  parser.add_argument('--n-iterations', type=int, default=20)
  parser.add_argument('--cutoffs', default='10,20')
  parser.add_argument('--n-seeds', type=int, default=64)
  parser.add_argument('--seed-start', type=int, default=0)
  parser.add_argument('--arms', default='bo,random')
  parser.add_argument('--kernel', default='sorting-rbf')
  parser.add_argument('--n-init', type=int, default=5)
  parser.add_argument('--n-events', type=int, default=512)
  parser.add_argument('--n-library', type=int, default=4096)
  parser.add_argument('--n-noise', type=int, default=2)
  parser.add_argument('--hedge', type=float, default=1.0e-3)
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()

  config = load_config(arguments.detector_config)
  overrides = parse_overrides(arguments.override)
  cutoffs = [int(value) for value in arguments.cutoffs.split(',')]
  payload = {
    'detector_config': arguments.detector_config,
    'overrides': {
      k: list(v) if isinstance(v, tuple) else v
      for k, v in overrides.items()
    },
    'noise': arguments.noise,
    'kernel': arguments.kernel,
    'n_iterations': arguments.n_iterations,
    'cutoffs': cutoffs,
    'loss_of_a_guess': 1.0,
    'n_events': arguments.n_events,
    'n_library': arguments.n_library,
    'n_noise': arguments.n_noise,
    'runs': {}
  }
  started = time.time()
  detector = build_detector(config, measurement_noise=0.0, **overrides)
  payload['n_experiments'] = detector.n_experiments
  payload['n_measurements'] = detector.n_measurements
  arms = arguments.arms.split(',')
  for arm in arms:
    payload['runs'][arm] = []
  for offset in range(arguments.n_seeds):
    seed = arguments.seed_start + offset
    instrument = make_instrument(detector, arguments, seed)
    audit = make_instrument(detector, arguments, seed + 500000)
    for arm in arms:
      record = run_arm(arm, detector, instrument, arguments, seed)
      record['seed'] = seed
      record['cutoffs'] = rescore(record, audit, cutoffs)
      payload['runs'][arm].append(record)
      marks = ', '.join('@{}={:.4f}'.format(cutoff, record['cutoffs'][str(cutoff)]['observed']) for cutoff in cutoffs)
      print(f'{arm} seed {seed}: best {min(record["losses"]):.4f} ({marks}) [{time.time() - started:.0f}s]', flush=True)
      write(payload, arguments.output)

  summarise(payload)
  print(f'wrote {arguments.output}')


def write(payload, path):
  """Rewrite the trajectory file after every seed, so a shard killed by its time limit still leaves
  the seeds it finished. Staged through a temporary file so a kill cannot truncate the json."""
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  staged = path + '.staged'
  with open(staged, 'w') as handle:
    json.dump(payload, handle)
  os.replace(staged, path)


def summarise(payload):
  first, second = payload['cutoffs'][0], payload['cutoffs'][-1]
  for arm, runs in payload['runs'].items():
    for kind in ('observed', 'rescored'):
      early = np.asarray([run['cutoffs'][str(first)][kind] for run in runs])
      late = np.asarray([run['cutoffs'][str(second)][kind] for run in runs])
      gain = early - late
      print(
        f'{arm:8s} {kind:9s}: E[loss@{first}] {early.mean():.4f}  E[loss@{second}] {late.mean():.4f}  '
        f'criterion {gain.mean():.4f} +- {gain.std(ddof=1) / math.sqrt(len(gain)):.4f}  '
        f'P(improve) {float(np.mean(late < early)):.3f}  n={len(gain)}'
      )


if __name__ == '__main__':
  main()
