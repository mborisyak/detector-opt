"""`ContinualRatioTrainer` against `ContinualTrainer`.

THE POINT OF THIS FILE is the first test: at `current_replay_ratio = 1.0` the new trainer must be
NUMERICALLY IDENTICAL to the one it generalises, so a campaign that changes only the ratio measures
the ratio and nothing else. It is checked element-by-element on the sampled indices and the row
weights, and end-to-end on two trained designs' objectives and parameters -- not argued from the
source.
"""

import jax
import jax.numpy as jnp
import numpy as np

from analytic import analytic_detector
from detopt.nn.trainer import ContinualRatioTrainer, ContinualTrainer

BATCH = 64


def _optimizer():
  import optax

  return optax.adamw(learning_rate=1e-3, weight_decay=1e-3)


def _pair(seed, *, replay_weight=0.25, current_replay_ratio=1.0, budget=20_000):
  """One `ContinualTrainer` and one `ContinualRatioTrainer` built from the SAME arguments and seed."""
  detector = analytic_detector()
  arguments = dict(
    regressor_config={"set-regressor": {
      "features": [[16, 16]],
      "n_models": 4,
      "dropconnect": 0.05
    }}, optimizer=_optimizer(), batch=BATCH, n0=256, n_increment=128, iteration_limit=512, warmup_epochs=2, patience=3,
    loss_precision=0.5, budget=budget, val_fraction=0.25, eval_batch=128, device=None, seed=seed,
  )
  reference = ContinualTrainer(detector, replay_weight=replay_weight, **arguments)
  candidate = ContinualRatioTrainer(
    detector, replay_weight=replay_weight, current_replay_ratio=current_replay_ratio, **arguments
  )
  return detector, reference, candidate


def test_ratio_one_matches_continual_sampling():
  """Same indices and same row weights, element for element, at every window position."""
  _, reference, candidate = _pair(0)

  reference_weights = np.asarray(reference._sample_weights())
  candidate_weights = np.asarray(candidate._sample_weights())
  assert reference_weights.shape == candidate_weights.shape
  assert np.array_equal(reference_weights, candidate_weights)

  for key_seed in (0, 1, 17):
    key = jax.random.PRNGKey(key_seed)
    for start, count in ((0, 200), (1000, 200), (37, 129)):
      reference_index = np.asarray(reference._sample_indices(key, jnp.int32(start), jnp.int32(count)))
      candidate_index = np.asarray(candidate._sample_indices(key, jnp.int32(start), jnp.int32(count)))
      assert reference_index.shape == candidate_index.shape
      assert np.array_equal(reference_index, candidate_index)


def test_ratio_one_matches_continual_training(seed):
  """Two designs trained through both trainers: identical objectives, identical carried parameters.

  Two rather than one, because the first design has no replay history (``start == 0``) and so cannot
  show a composition difference at all -- the second one draws from the first's rows.
  """
  detector, reference, candidate = _pair(seed)
  design = np.zeros(detector.design_dim(), dtype=np.float32)

  for step in (0, 1):
    reference_result = reference.train(design, seed + step, step=step)
    candidate_result = candidate.train(design, seed + step, step=step)
    assert reference_result is not None and candidate_result is not None
    assert reference_result.objective_loss == candidate_result.objective_loss
    assert reference_result.objective_std == candidate_result.objective_std
    assert reference_result.spent == candidate_result.spent

  reference_leaves = jax.tree.leaves(reference._running[0])
  candidate_leaves = jax.tree.leaves(candidate._running[0])
  assert len(reference_leaves) == len(candidate_leaves) and len(reference_leaves) > 0
  for reference_leaf, candidate_leaf in zip(reference_leaves, candidate_leaves):
    assert np.array_equal(np.asarray(reference_leaf), np.asarray(candidate_leaf))
  assert reference.train_pool.current == candidate.train_pool.current


def test_ratio_splits_the_batch():
  """``batch // (1 + ratio)`` replay rows, the remainder current; the batch size never changes."""
  for ratio, expected_replay in ((1.0, BATCH // 2), (3.0, BATCH // 4), (7.0, BATCH // 8)):
    _, _, candidate = _pair(0, current_replay_ratio=ratio)
    assert candidate._replay_rows() == expected_replay
    index = np.asarray(candidate._sample_indices(jax.random.PRNGKey(3), jnp.int32(1000), jnp.int32(200)))
    members = candidate.n_ensemble if candidate.n_ensemble is not None else 1
    assert index.shape == (members * BATCH, )
    per_member = index.reshape(members, BATCH)
    current, past = per_member[:, :BATCH - expected_replay], per_member[:, BATCH - expected_replay:]
    assert (current >= 1000).all() and (current < 1200).all()
    assert (past >= 0).all() and (past < 1000).all()
    weights = np.asarray(candidate._sample_weights())
    assert weights.shape == (members, BATCH)
    assert abs(float(weights.mean()) - 1.0) < 1e-6
    assert weights[:, :BATCH - expected_replay].min() > weights[:, BATCH - expected_replay:].max()


def test_ratio_rejects_non_positive():
  import pytest

  with pytest.raises(ValueError):
    _pair(0, current_replay_ratio=0.0)
