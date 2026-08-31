#!/usr/bin/env python3
"""Mid-trajectory FIXED-DESIGN probe of one architecture under one arm, TRAINED BY THE RUN'S OWN
GROWTH PROCEDURE.

    python scripts/probe_semi_hyper.py =emnist_alphahyper output=... seed=... arm=meta \\
        measure_at='[5, 7, 9]' reference=output/campaign-emnist/330924253/meta/results.json

Three designs are taken from a finished run's trajectory, every condition is measured on THOSE SAME
THREE, and only the ``regressor`` block may differ from ``baseline`` -- asserted, not assumed.

EVERY DESIGN IS TRAINED THROUGH ``trainer.train``. That call is where the growth decision, the exit
test and the REWIND live, and a probe that reimplements the epoch loop silently drops all three:

* ``rewind`` (the rewind) fires AT EVERY DATA ADDITION -- ``params = initial + (1 - lambda) *
  (current - initial)`` -- and also resets the optimiser moments and rewinds the parameter average.
  Without it the fitted weights carry across every window growth, the window is memorised, and the
  loss-versus-samples curve becomes a sawtooth that describes the missing rewind rather than the
  architecture.
* without the exit test there is no ``is_plateaued`` and no Bayesian gap check, so NO cell may be
  called converged.
* without the growth decision the window grows on a schedule the run would never have chosen, at a
  spend the run never paid.

WHAT IS RECONSTRUCTED, AND WHY. A mid-trajectory design is not a design trained in isolation: the
shared pool already holds the earlier designs' events, and a continual arm draws part of every
minibatch from them and carries its network across the boundary. So the probe walks the recorded
sequence from index 0:

* a design that is not measured is REPLAYED BY SAMPLING ALONE, at its RECORDED ``spent``, and that
  is true for EVERY arm. The context is sampled, not trained: the continual arm then replays out of
  that pool while training the design under test, which is the state the run's own trainer would
  have been in. Training the prefix instead would let each architecture grow the pool by a DIFFERENT
  amount, so two architectures would never reach the measured design on the same data -- and
  rewinding the pool afterwards to hide that would re-issue events the prefix network had already
  trained on, which is a leak;
* index 0 is never measured: at ``start == 0`` there is no replay history and the continual arm is
  indistinguishable from a per-design one.

THE READOUT. Conditions exit where their own procedure exits, so they exit at DIFFERENT ``spent`` --
that is the measurement, not a defect. Every epoch's train and validation loss, their standard errors
and the WINDOW are recorded, so "samples needed" is read afterwards as the first crossing of a FIXED
LOSS LEVEL. Never the loss at exit, and never a ``min`` over the whole curve: the window grows during
a design, so such a minimum is taken across different validation sets and is a biased order statistic
that flatters whichever condition was noisiest. ``spent``, ``window``, ``epochs`` and the
converged/capped flag are recorded beside every objective.

A CAPPED CELL IS A RESULT. It is the only cell that reports where the gap actually sits; a converged
one proves ``gap + err <= loss_precision`` and no more.
"""

import json
import os
import sys
import time

import numpy as np

import detopt
import detopt.detector
from detopt.nn.trainer import ContinualTrainer, DesignTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_batch_mix import replay_pool

ARMS = {'from_scratch': DesignTrainer, 'meta': ContinualTrainer}
CAP_MESSAGE = 'did not reach precision within iteration_limit'


def _expanded(path):
  """A config file loaded and reference-expanded exactly as the CLI does it."""
  from gearup.config import read_config

  _, config = read_config([], path)
  return config


def _check_only_regressor_differs(baseline_path, config):
  """Fail unless ``config`` matches ``baseline_path`` outside the ``regressor`` block."""
  baseline = _expanded(baseline_path)
  keys = sorted(set(baseline) | set(config))
  differing = [k for k in keys if k != 'regressor' and baseline.get(k) != config.get(k)]
  if len(differing) > 0:
    raise ValueError(f'config differs from {baseline_path} outside `regressor`, in {differing}')
  same = baseline.get('regressor') == config.get('regressor')
  print(f'[config] regressor {"is identical to" if same else "is the only difference from"} {baseline_path}', flush=True)


def curve_row(snapshot):
  """The per-epoch arrays of one ``trainer.train`` call, as plain lists, with the last epoch's scalars
  beside them. ``window_per_epoch`` is what makes the GROWTH visible."""
  train = snapshot['train_loss_per_epoch']
  validation = snapshot['val_loss_per_epoch']
  train_sem = snapshot['train_sem_per_epoch']
  validation_sem = snapshot['val_sem_per_epoch']
  return {
    'epochs': int(len(train)),
    'window': int(snapshot['final_train_budget']),
    'train': float(train[-1]),
    'validation': float(validation[-1]),
    'train_sem': float(train_sem[-1]),
    'validation_sem': float(validation_sem[-1]),
    'gap': abs(float(validation[-1]) - float(train[-1])),
    'err': float(np.hypot(train_sem[-1], validation_sem[-1])),
    'train_per_epoch': [float(v) for v in train],
    'validation_per_epoch': [float(v) for v in validation],
    'train_sem_per_epoch': [float(v) for v in train_sem],
    'validation_sem_per_epoch': [float(v) for v in validation_sem],
    'window_per_epoch': [int(v) for v in snapshot['train_budget_per_epoch']],
  }


def train_design(trainer, design_scaled, design_index, training_seed, save, allow_cap):
  """One design trained by ``trainer.train`` -- growth, exit test and rewind included -- returned as a
  row. ``save`` is called after every epoch, so a cell can be read live and a killed one still leaves
  its curve."""
  window_start = (trainer.train_pool.current, trainer.val_pool.current)
  latest = {}

  def on_epoch(snapshot, sink=latest):
    sink.clear()
    sink.update(snapshot)
    save(False)

  started = time.time()
  capped = False
  try:
    result = trainer.train(
      np.asarray(design_scaled, dtype=np.float32), int(training_seed), on_epoch=on_epoch, step=int(design_index)
    )
  except RuntimeError as error:
    if not allow_cap or CAP_MESSAGE not in str(error):
      raise
    capped, result = True, None
  elapsed = time.time() - started
  if result is None and not capped:
    raise RuntimeError(f'design {design_index}: the budget pool filled, so this design could not be scored')
  if len(latest) == 0:
    raise RuntimeError(f'design {design_index}: no epoch completed, so there is no curve to report')

  row = curve_row(latest)
  if capped:
    row['objective'] = 0.5 * (row['train'] + row['validation'])
    row['objective_std'] = float(np.hypot(row['gap'] / np.sqrt(12.0), 0.5 * row['err']))
    row['spent'] = (trainer.train_pool.current - window_start[0]) + (trainer.val_pool.current - window_start[1])
  else:
    row['objective'] = float(result.objective_loss)
    row['objective_std'] = float(result.objective_std)
    row['spent'] = int(result.spent)
  row['converged'] = not capped
  row['seconds'] = elapsed
  row['seed'] = int(training_seed)
  row['iteration'] = int(design_index)
  return row


def probe(
  output, reference, seed: int, arm: str = 'meta', measure_at=(5, 7, 9), baseline: str = 'config/emnist.yaml',
  allow_cap: bool = True, **config
):
  if arm not in ARMS:
    raise ValueError(f'arm {arm!r} not in {sorted(ARMS)}')
  measured = sorted(int(k) for k in measure_at)
  if measured[0] < 1:
    raise ValueError(f'index 0 carries no replay history, so it is never a measurement: got {measured}')
  _check_only_regressor_differs(baseline, config)

  with open(reference) as f:
    recorded = json.load(f)['results']
  if len(recorded) <= measured[-1]:
    raise ValueError(f'{reference} holds {len(recorded)} designs, too few for index {measured[-1]}')

  os.makedirs(output, exist_ok=True)
  detector = detopt.detector.from_config(config['detector'])
  # The seed split `scripts/bo.py` performs: a network branch (the regressor draw and the run's
  # train/validation event stream) and an iteration branch (one seed per design).
  network_seq, iteration_seq = np.random.SeedSequence(int(seed)).spawn(2)
  trainer = ARMS[arm].from_config(
    detector, config, checkpoint_dir=os.path.join(output, 'checkpoints'), seed=int(network_seq.generate_state(1)[0])
  )

  rows = []

  def _save(completed):
    payload = {
      'arm': arm,
      'seed': int(seed),
      'reference': reference,
      'measure_at': measured,
      'growth_procedure': True,
      'batch': int(trainer.batch),
      'steps_per_epoch': int(trainer.steps_per_epoch),
      'loss_precision': float(config['training']['loss_precision']),
      'rewind': float(config['training'].get('rewind', 0.0)),
      'regressor': config['regressor'],
      'config': config,
      'completed': completed,
      'designs': rows,
    }
    staged = os.path.join(output, 'probe.json.new')
    with open(staged, 'w') as f:
      json.dump(payload, f, indent=2, default=float)
      f.flush()
      os.fsync(f.fileno())
    os.replace(staged, os.path.join(output, 'probe.json'))

  print(
    f'probe: arm={arm} seed={seed} measure_at={measured} GROWTH PROCEDURE (rewind='
    f'{config["training"].get("rewind", 0.0)}, loss_precision={config["training"]["loss_precision"]}) <- {reference}',
    flush=True
  )
  for step in range(measured[-1] + 1):
    x_scaled = np.asarray(recorded[step]['x_scaled'], np.float32)
    iteration_seed = int(iteration_seq.spawn(1)[0].generate_state(1)[0])
    started = time.time()

    if step not in measured:
      replay_pool(trainer, detector, [x_scaled], [int(recorded[step]['spent'])])
      print(
        f'[prefix {step}] sampled {recorded[step]["spent"]} events, pool at '
        f'{trainer.train_pool.current}/{trainer.val_pool.current}, time={time.time() - started:.1f}s', flush=True
      )
      continue

    row = train_design(trainer, x_scaled, step, iteration_seed, _save, allow_cap)
    design_phys = np.asarray(detector.flatten_design(detector.to_nominal(x_scaled)), np.float32).tolist()
    penalty = detector.design_penalty(design_phys)
    row['x_scaled'] = x_scaled.tolist()
    row['design'] = design_phys
    row['design_penalty'] = None if penalty is None else float(penalty)
    rows.append(row)
    _save(False)
    print(
      f'[design {step}] objective={row["objective"]:.5f} +- {row["objective_std"]:.5f} gap={row["gap"]:.5f} '
      f'err={row["err"]:.5f} epochs={row["epochs"]} window={row["window"]} spent={row["spent"]} '
      f'{"CONVERGED" if row["converged"] else "CAPPED"} ({row["seconds"]:.0f}s)', flush=True
    )
  _save(True)
  print(f'probe: wrote {os.path.join(output, "probe.json")}', flush=True)


if __name__ == '__main__':
  import sys

  import gearup

  gearup.gearup(probe).with_config('config/root.yaml')(sys.argv[1:])
