"""Does replaying a trajectory rebuild EXACTLY what the saved state file restores?

    python scripts/verify_replay.py <run directory> [--device cpu]

``Trainer.replay`` refills the event pools by re-simulating the designs in ``results.json`` instead of
loading ``trainer.npz``. That is only sound if the two agree leaf for leaf, so this builds the same
trainer twice from the run's own config -- once through ``restore(trainer.npz)``, once through
``replay(rows)`` -- and compares the train and validation pools (contents AND cursors) plus whatever
the strategy carries across designs.

Needs a run that still HAS its ``trainer.npz``; it is the reference the replay is checked against.
Exits non-zero on any mismatch, so it can gate the removal of the state file.
"""

import argparse
import json
import os
import sys

import numpy as np


def build(detector, config, output, seed, strategy):
  from detopt.nn.trainer import ContinualRatioTrainer, ContinualTrainer, DesignTrainer

  trainers = {"per_design": DesignTrainer, "meta": ContinualTrainer, "meta_ratio": ContinualRatioTrainer}
  trainer_cls = trainers.get(strategy, trainers["per_design"])
  return trainer_cls.from_config(detector, config, checkpoint_dir=os.path.join(output, "checkpoints"), seed=seed)


def compare_pools(name, reference, replayed, failures):
  if reference.current != replayed.current:
    failures.append(f"{name} pool cursor: restore={reference.current} replay={replayed.current}")
    return
  import jax

  ref_leaves = jax.tree.leaves(reference.buffers())
  new_leaves = jax.tree.leaves(replayed.buffers())
  if len(ref_leaves) != len(new_leaves):
    failures.append(f"{name} pool leaf count: restore={len(ref_leaves)} replay={len(new_leaves)}")
    return
  filled = reference.current
  for index, (a, b) in enumerate(zip(ref_leaves, new_leaves)):
    a, b = np.asarray(a)[:filled], np.asarray(b)[:filled]
    if a.shape != b.shape:
      failures.append(f"{name} pool leaf {index} shape: {a.shape} vs {b.shape}")
    elif not np.array_equal(a, b):
      worst = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
      failures.append(f"{name} pool leaf {index}: {int(np.sum(a != b))} differing entries, max|diff|={worst:.3e}")
  print(f"  {name} pool: {filled} events, {len(ref_leaves)} leaves compared")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('run', help='a run directory holding results.json, trainer.npz and checkpoints/')
  parser.add_argument('--device', default='cpu')
  arguments = parser.parse_args()

  with open(os.path.join(arguments.run, 'results.json')) as handle:
    payload = json.load(handle)
  rows = payload['results']
  config = dict(payload['config'])
  config['device'] = arguments.device
  strategy = payload.get('nn_init_strategy', 'per_design')
  state_path = os.path.join(arguments.run, 'trainer.npz')
  if not os.path.exists(state_path):
    sys.exit(f"{state_path} is missing; this check needs the saved state as its reference")
  if any('spent_train' not in row for row in rows):
    sys.exit(f"{arguments.run}: rows predate spent_train/spent_val, so the split cannot be replayed")

  import detopt.detector
  detector = detopt.detector.from_config(config['detector'])
  seed = int(np.load(state_path)['seed'])
  print(f"run={arguments.run} strategy={strategy} rows={len(rows)} seed={seed}")

  reference = build(detector, config, arguments.run, seed, strategy)
  reference.restore(state_path)
  replayed = build(detector, config, arguments.run, seed, strategy)
  replayed.replay(rows)

  failures = []
  compare_pools('train', reference.train_pool, replayed.train_pool, failures)
  compare_pools('validation', reference.val_pool, replayed.val_pool, failures)

  carried_reference = getattr(reference, '_running', None)
  carried_replay = getattr(replayed, '_running', None)
  if carried_reference is None:
    print('  carried state: none (per-design strategy)')
  else:
    import jax
    ref_leaves = jax.tree.leaves(carried_reference)
    new_leaves = jax.tree.leaves(carried_replay)
    if len(ref_leaves) != len(new_leaves):
      failures.append(f"carried network leaf count: {len(ref_leaves)} vs {len(new_leaves)}")
    else:
      worst = 0.0
      for index, (a, b) in enumerate(zip(ref_leaves, new_leaves)):
        a, b = np.asarray(jax.random.key_data(a) if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key) else a), \
               np.asarray(jax.random.key_data(b) if jax.dtypes.issubdtype(b.dtype, jax.dtypes.prng_key) else b)
        if a.shape != b.shape:
          failures.append(f"carried leaf {index} shape: {a.shape} vs {b.shape}")
        else:
          worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size > 0 else 0.0)
      print(f"  carried network: {len(ref_leaves)} leaves, max|diff|={worst:.3e}")
      if worst > 0.0:
        failures.append(f"carried network differs, max|diff|={worst:.3e}")

  if len(failures) > 0:
    print('\nMISMATCH:')
    for line in failures:
      print(f"  {line}")
    sys.exit(1)
  print('\nOK: replay reproduces the restored state exactly')


if __name__ == '__main__':
  main()
