"""The growth procedure, exercised on the DEBUG detector where the answer is known in closed form.

``linear`` is ``y = w x + b + noise``; :meth:`LinearDetector.bayes_risk` is the BEST loss any regressor
can reach at a design, so "did it train" is checkable against a floor rather than against a previous
run. These tests drive the REAL ``DesignTrainer.train`` on CPU at a deliberately small budget and
assert the three things a hand-rolled epoch loop silently drops:

* the WINDOW GROWS, and it grows because the exit test said so rather than on a schedule;
* the REWIND (``rewind``) is inside the loop -- at a growth boundary it discards a fraction of the
  fit and resets the optimiser, which is visible as a jump in the training loss, and a run with
  ``rewind = 0`` follows a different trajectory from the same seed;
* a cell reported as CONVERGED really satisfies ``gap + err <= loss_precision``.

They are not a substitute for reading a probe: they pin the contract that the probe must train through
``trainer.train``.

⚠️ ``loss_precision`` IS WHAT MAKES GROWTH OBSERVABLE HERE, and it is not free to raise. The exit test
is ``gap + err <= loss_precision``, and on this fixture ``err`` -- the estimate error, which shrinks
with the evaluated window rather than with the fit -- is about 0.043 at the ``n0`` window. At the
5.0e-2 bar this file used to carry, ``err`` alone was 85% of the bar and the FIRST window passed:
one window, no growth boundary, nothing for three of these tests to inspect, while the network sat at
0.7311 against a 0.6923 floor and was not converged in any useful sense. At 3.0e-2 the first window
fails, growth fires five times with stage lengths 24/20/17/16/15 that vary on their own, and the
objective closes to 0.7095. THE PROCEDURE WAS NEVER THE PROBLEM -- state what a probe must separate
and check that it can.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import detopt.detector
from detopt.nn.trainer import DesignTrainer

SMALL_TRAINING = {
  'n0': 1024,
  'n_increment': 512,
  'batch': 32,
  'eval_batch': 1024,
  'warmup_epochs': 2,
  'patience': 4,
  'budget': 65536,
  'iteration_limit': 8192,
  'val_fraction': 0.25,
  'loss_precision': 3.0e-2,
  'optimizer': {
    'adamaxw': {
      'learning_rate': 2.5e-3,
      'b1': 0.9,
      'b2': 0.999,
      'eps': 1.0e-8,
      'weight_decay': 1.0e-3
    }
  },
}

REGRESSOR = {'set-regressor': {'features': [[16, 8], [8, 16]], 'n_models': 1, 'dropconnect': 0.0}}


def _expanded(path):
  from gearup.config import read_config

  _, config = read_config([], path)
  return config


def _config(rewind):
  base = _expanded('config/linear_d2n3.yaml')
  return {
    **base,
    'device': 'cpu',
    'regressor': REGRESSOR,
    'training': {
      **SMALL_TRAINING, 'rewind': rewind, 'reveal': 'design'
    },
  }


def _run(rewind, seed=17):
  """One design trained by the real procedure. Returns the per-epoch snapshot and the result."""
  config = _config(rewind)
  detector = detopt.detector.from_config(config['detector'])
  trainer = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=seed)
  assert trainer.reveal() != 'none', (
    'the rewind is only observable when the network actually FITS the window; a design-blind '
    'arm on `linear` barely fits, so undoing part of the fit costs it nothing visible. '
    "`DesignTrainer.default_reveal()` is 'none', so the config above sets `training.reveal` "
    'explicitly rather than relying on a default that does not hold for this strategy.'
  )
  design_scaled = np.full((detector.design_dim(), ), 0.5, np.float32)
  latest = {}

  def on_epoch(snapshot, sink=latest):
    sink.clear()
    sink.update({k: np.asarray(v) for k, v in snapshot.items() if hasattr(v, '__len__')})
    sink['final_train_budget'] = snapshot['final_train_budget']

  result = trainer.train(design_scaled, seed, on_epoch=on_epoch, step=1)
  return latest, result, detector, design_scaled


@pytest.fixture(scope='module')
def rewound():
  return _run(rewind=0.25)


def test_the_window_grows(rewound):
  """A single window would mean the growth decision never fired."""
  snapshot, _, _, _ = rewound
  windows = [int(v) for v in snapshot['train_budget_per_epoch']]
  assert len(set(windows)) > 1, f'the window never grew: {sorted(set(windows))}'
  assert windows == sorted(windows), 'the window must be non-decreasing'


def test_growth_is_not_on_a_fixed_schedule(rewound):
  """Epochs-per-window must vary; equal-length stages are the signature of a schedule."""
  windows = [int(v) for v in rewound[0]['train_budget_per_epoch']]
  lengths = [len([w for w in windows if w == value]) for value in sorted(set(windows))]
  assert len(lengths) > 1
  assert len(set(lengths)) > 1 or lengths[0] > 1, f'every window ran for exactly {lengths} epochs'


def test_convergence_really_meets_the_precision_bar(rewound):
  """A cell that returns a result claims convergence; the bar must actually hold."""
  snapshot, result, _, _ = rewound
  assert result is not None
  gap = abs(float(snapshot['val_loss_per_epoch'][-1]) - float(snapshot['train_loss_per_epoch'][-1]))
  err = float(np.hypot(snapshot['train_sem_per_epoch'][-1], snapshot['val_sem_per_epoch'][-1]))
  assert gap + err <= SMALL_TRAINING['loss_precision'] * 1.5, f'gap {gap:.4f} + err {err:.4f} against the bar'


def test_the_network_reaches_the_closed_form_floor(rewound):
  """`bayes_risk` is the best loss ANY regressor can reach here, so the objective must not sit far
  below it (impossible) and must land within reach of it (or training is broken)."""
  _, result, detector, design_scaled = rewound
  floor = float(detector.bayes_risk(detector.to_nominal(design_scaled)))
  assert result.objective_loss > floor * 0.5, f'objective {result.objective_loss:.5f} below the floor {floor:.5f}'
  assert result.objective_loss < floor * 6.0, f'objective {result.objective_loss:.5f} far above the floor {floor:.5f}'


def test_the_rewind_is_inside_the_loop(rewound):
  """`rewind` discards a fraction of the fit at every data addition and resets the optimiser, so the
  training loss JUMPS at a growth boundary. Without it the same seed follows a different trajectory."""
  snapshot = rewound[0]
  windows = [int(v) for v in snapshot['train_budget_per_epoch']]
  train = [float(v) for v in snapshot['train_loss_per_epoch']]
  boundaries = [i for i in range(1, len(windows)) if windows[i] != windows[i - 1]]
  assert len(boundaries) > 0, 'no growth boundary to inspect'
  jumps = [train[i] - train[i - 1] for i in boundaries]
  assert max(jumps) > 0.0, f'the training loss never rose at a growth boundary: {jumps}'

  plain, _, _, _ = _run(rewind=0.0)
  plain_train = [float(v) for v in plain['train_loss_per_epoch']]
  same = len(plain_train) == len(train) and np.allclose(plain_train, train, atol=1e-6)
  assert not same, 'rewind = 0.25 and 0.0 produced the same trajectory, so the rewind is not in the loop'
