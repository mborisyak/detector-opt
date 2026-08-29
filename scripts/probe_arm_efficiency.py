"""Loss versus SAMPLES for the warm-start arms at a FIXED design, off the BO loop.

The campaign question -- does a carried network make later designs cheaper -- is confounded inside a
BO run, because each arm scores DIFFERENT designs: "later designs are cheaper" mixes "the network
learned something" with "BO walked into an easier region". This probe removes the confound by
holding the design fixed and varying only the state the arm starts from.

Two phases, one design SEQUENCE (the recorded ``x_scaled`` of an existing trajectory).

``phase=replay`` re-trains designs ``0 .. n_prefix-1`` of that sequence under ONE arm and persists
the trainer after every design (``persist``/``commit``, the same pair a resumed run reads). That
rebuilds the state the arm would hold at each boundary: the persistent network plus the pool of
earlier designs' events for ``meta``, the chain of per-design checkpoints for ``continue``.

``phase=measure`` restores that state at one design index and trains THAT design under each arm,
``repeats`` times, varying ONLY the training seed -- the network draw, the event stream and the
restored state are identical across repeats, so the spread is training randomness alone. Every epoch
is recorded as ``(window, train loss, val loss)``: the loss-versus-samples curve, which says whether
an arm is uniformly faster or merely stops earlier. Samples-to-a-fixed-LEVEL is read off it
afterwards; the convergence rule is a slope test that never looks at a level, so where it fires is
not a measure of cost.

``phase=poolcheck`` verifies the one efficiency this design relies on: that the pool can be filled
by SAMPLING events at a design without training on them.

The arms are exactly ``scripts/bo.py``'s: ``from_scratch`` and ``continue`` are ``DesignTrainer``
(``continue`` warm-starting its params from the previous design's checkpoint), ``meta`` is
``ContinualTrainer``, which carries its own network and replays from the pool.
"""

import json
import os
import shutil

import matplotlib

matplotlib.use('AGG')

import numpy as np

import detopt
import detopt.utils.io
from detopt.nn.trainer import ContinualTrainer, DesignTrainer

ARMS = ('from_scratch', 'continue', 'meta')
TRAINERS = {'from_scratch': DesignTrainer, 'continue': DesignTrainer, 'meta': ContinualTrainer}
STRATEGY_KNOBS = {'replay_weight': ('meta', ), 'current_replay_ratio': ()}


def _trainer_config(config, arm):
  """The run config with the knobs ``arm``'s trainer does not accept dropped, loudly."""
  training = dict(config['training'])
  for knob, owners in STRATEGY_KNOBS.items():
    if arm not in owners and knob in training:
      del training[knob]
      print(f'[config] dropped `training.{knob}` for arm `{arm}`', flush=True)
  return {'training': training, 'regressor': config['regressor'], 'device': config.get('device')}


def _designs(trajectory, count):
  """The first ``count`` scaled designs of a recorded ``results.json``."""
  with open(trajectory) as stream:
    rows = json.load(stream)['results']
  if len(rows) < count:
    raise ValueError(f'{trajectory} holds {len(rows)} designs, {count} requested')
  return [np.asarray(rows[i]['x_scaled'], dtype=np.float32) for i in range(count)]


def _seeds(seed):
  """``(trainer_seed, iteration_seeds)`` derived exactly as ``scripts/bo.py`` derives them, so a
  replay of a recorded sequence sees the seeds that sequence was measured under."""
  network_seq, iteration_seq = np.random.SeedSequence(int(seed)).spawn(2)
  return int(network_seq.generate_state(1)[0]), iteration_seq


def _history(snapshot):
  """One recorded curve: per-epoch window size and losses, as plain lists."""
  if snapshot is None:
    return None
  return {
    'window': [int(v) for v in snapshot['train_budget_per_epoch']],
    'train_loss': [float(v) for v in snapshot['train_loss_per_epoch']],
    'val_loss': [float(v) for v in snapshot['val_loss_per_epoch']],
    'train_sem': [float(v) for v in snapshot['train_sem_per_epoch']],
    'val_sem': [float(v) for v in snapshot['val_sem_per_epoch']],
  }


def _train_one(trainer, design_scaled, train_seed, *, init_params, step):
  """Train one design, returning ``(record, snapshot)``. A design that cannot reach precision inside
  ``iteration_limit`` and a budget exhaustion are recorded as themselves, never as a measurement."""
  keep = {'snapshot': None}

  def on_epoch(snapshot):
    keep['snapshot'] = snapshot

  try:
    result = trainer.train(design_scaled, int(train_seed), init_params=init_params, on_epoch=on_epoch, step=int(step))
  except RuntimeError as error:
    text = str(error).replace('\n', ' ')
    if 'did not reach precision within iteration_limit' not in text:
      raise
    return {'status': 'unconverged', 'error': text}, keep['snapshot']
  if result is None:
    return {'status': 'budget_exhausted'}, keep['snapshot']
  return {
    'status': 'converged',
    'loss': float(result.objective_loss),
    'loss_std': float(result.objective_std),
    'spent': int(result.spent),
  }, keep['snapshot']


def replay(output, seed, arm, trajectory, n_prefix, config):
  """Rebuild ``arm``'s state along the recorded design sequence, persisting after every design."""
  detector = detopt.detector.from_config(config['detector'])
  designs = _designs(trajectory, int(n_prefix))
  trainer_seed, iteration_seq = _seeds(seed)
  checkpoints = os.path.join(output, 'checkpoints')
  trainer = TRAINERS[arm].from_config(
    detector, _trainer_config(config, arm), checkpoint_dir=checkpoints, seed=trainer_seed
  )
  rows = []
  for step, design_scaled in enumerate(designs):
    train_seed = int(iteration_seq.spawn(1)[0].generate_state(1)[0])
    init_params = None
    if arm == 'continue' and step > 0:
      init_params = trainer.restore_design_parameters(step - 1)
    print(f'[replay {arm}] design {step} ...', flush=True)
    record, snapshot = _train_one(trainer, design_scaled, train_seed, init_params=init_params, step=step)
    record.update({'design': step, 'arm': arm, 'history': _history(snapshot)})
    rows.append(record)
    if record['status'] != 'converged':
      print(f'[replay {arm}] design {step} did NOT converge: {record}', flush=True)
      break
    state_path = os.path.join(output, f'state_{step:04d}.npz')
    trainer.persist(state_path)
    detopt.utils.io.commit([state_path])
    print(f"[replay {arm}] design {step}: loss={record['loss']:.4f} spent={record['spent']} "
          f"pool={trainer.train_pool.current}", flush=True)
    with open(os.path.join(output, 'replay.json'), 'w') as stream:
      json.dump({'arm': arm, 'seed': int(seed), 'trajectory': trajectory, 'rows': rows}, stream)


def measure(output, seed, at, repeats, trajectory, replay_root, config):
  """Train design ``at`` under every arm, ``repeats`` times, from each arm's reconstructed state."""
  detector = detopt.detector.from_config(config['detector'])
  designs = _designs(trajectory, int(at) + 1)
  design_scaled = designs[int(at)]
  trainer_seed, _ = _seeds(seed)
  rows = []
  for arm in ARMS:
    source = os.path.join(replay_root, 'replay-continue' if arm in ('from_scratch', 'continue') else 'replay-meta')
    state_path = os.path.join(source, f'state_{int(at) - 1:04d}.npz')
    for repeat in range(int(repeats)):
      scratch = os.path.join(output, f'scratch_{arm}_{repeat}')
      shutil.rmtree(scratch, ignore_errors=True)
      os.makedirs(scratch, exist_ok=True)
      if arm == 'continue':
        previous = f'design_{int(at) - 1:04d}'
        shutil.copytree(os.path.join(source, 'checkpoints', previous), os.path.join(scratch, previous))
      trainer = TRAINERS[arm].from_config(
        detector, _trainer_config(config, arm), checkpoint_dir=scratch, seed=trainer_seed
      )
      trainer.restore(state_path)
      init_params = trainer.restore_design_parameters(int(at) - 1) if arm == 'continue' else None
      train_seed = int(np.random.SeedSequence([int(seed), int(at), repeat]).generate_state(1)[0])
      print(f'[measure] design {at} arm {arm} repeat {repeat} (pool {trainer.train_pool.current}) ...', flush=True)
      record, snapshot = _train_one(trainer, design_scaled, train_seed, init_params=init_params, step=int(at))
      record.update({'design': int(at), 'arm': arm, 'repeat': repeat, 'history': _history(snapshot)})
      rows.append(record)
      print(f"[measure] design {at} arm {arm} repeat {repeat}: {record['status']} "
            f"loss={record.get('loss', float('nan')):.4f} spent={record.get('spent', -1)}", flush=True)
      shutil.rmtree(scratch, ignore_errors=True)
      with open(os.path.join(output, 'measure.json'), 'w') as stream:
        json.dump({'design': int(at), 'seed': int(seed), 'trajectory': trajectory, 'rows': rows}, stream)


def poolcheck(output, seed, trajectory, config):
  """Verify the separation the reconstruction relies on: events can be SAMPLED into the pool at a
  design without training on them, and the fill lands where a trained design would have left it."""
  detector = detopt.detector.from_config(config['detector'])
  designs = _designs(trajectory, 2)
  trainer_seed, _ = _seeds(seed)
  trainer = DesignTrainer.from_config(detector, _trainer_config(config, 'from_scratch'), checkpoint_dir=None,
                                      seed=trainer_seed)
  before = (trainer.train_pool.current, trainer.val_pool.current)
  design = detector.to_nominal(np.asarray(designs[0], dtype=np.float32))
  added = trainer._sample_round(design, before[0], before[1], trainer.n0)
  after = (trainer.train_pool.current, trainer.val_pool.current)
  print(f'[poolcheck] _sample_round returned {added}; pool {before} -> {after}', flush=True)
  with open(os.path.join(output, 'poolcheck.json'), 'w') as stream:
    json.dump({'added': int(added), 'before': list(before), 'after': list(after)}, stream)


def probe(
  output, seed: int, phase: str = 'measure', arm: str = 'meta', trajectory: str = '', replay_root: str = '',
  n_prefix: int = 10, at: int = 6, repeats: int = 3, **config
):
  os.makedirs(output, exist_ok=True)
  if phase == 'replay':
    replay(output, seed, arm, trajectory, n_prefix, config)
  elif phase == 'measure':
    measure(output, seed, at, repeats, trajectory, replay_root, config)
  elif phase == 'poolcheck':
    poolcheck(output, seed, trajectory, config)
  else:
    raise ValueError(f'phase must be replay/measure/poolcheck, got {phase!r}')


if __name__ == '__main__':
  import sys
  import gearup

  gearup.gearup(probe).with_config('config/bo.yaml')(sys.argv[1:])
