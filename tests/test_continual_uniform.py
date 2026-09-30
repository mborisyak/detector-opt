"""`ContinualUniformTrainer`: a uniform draw over the whole pool, unit weights, and the continual arm's mechanics.

The first test pins the contract that makes the ablation an ablation: at every window position the
indices cover ``[0, start + count)`` and nothing else, the current design's share of the batch tracks
its share of the pool instead of being fixed at one half, and every row weighs 1. The second trains
two designs end to end through the growth procedure and checks the network is carried across them.
"""
import jax
import jax.numpy as jnp
import numpy as np

from analytic import analytic_detector
from detopt.nn.trainer import ContinualUniformTrainer

BATCH = 64


def _optimizer():
  import optax
  return optax.adamw(learning_rate=1e-3, weight_decay=1e-3)


def _trainer(seed, budget=20_000):
  detector = analytic_detector()
  trainer = ContinualUniformTrainer(
    detector, regressor_config={"set-regressor": {
      "features": [[16, 16]],
      "n_models": 4,
      "dropconnect": 0.05
    }}, optimizer=_optimizer(), batch=BATCH, n0=256, n_increment=128, iteration_limit=512, warmup_epochs=2, patience=3,
    loss_precision=0.5, budget=budget, val_fraction=0.25, eval_batch=128, device=None, seed=seed,
  )
  return detector, trainer


def test_uniform_draw_covers_the_whole_pool():
  """Indices in ``[0, start + count)``, current share ~ ``count / (start + count)``, unit weights."""
  _, trainer = _trainer(0)
  members = trainer.n_ensemble if trainer.n_ensemble is not None else 1
  weights = np.asarray(trainer._sample_weights())
  assert weights.shape == (members, BATCH) and np.array_equal(weights, np.ones_like(weights))
  for start, count in ((0, 200), (1000, 200), (37, 129), (3000, 1000)):
    draws = []
    for key_seed in range(64):
      index = np.asarray(trainer._sample_indices(jax.random.PRNGKey(key_seed), jnp.int32(start), jnp.int32(count)))
      assert index.shape == (members * BATCH, )
      assert (index >= 0).all() and (index < start + count).all()
      draws.append(index)
    current_share = float(np.mean(np.concatenate(draws) >= start))
    expected = count / (start + count)
    assert abs(current_share - expected) < 0.05, (start, count, current_share, expected)


def test_uniform_trains_and_carries_the_network():
  """Two designs through the growth procedure: a result each time, and the carried network changes."""
  detector, trainer = _trainer(1)
  design = np.zeros(detector.design_dim(), dtype=np.float32)
  before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(trainer._running[0])]
  for step in (0, 1):
    result = trainer.train(design, 1 + step, step=step)
    assert result is not None and result.spent > 0
  after = jax.tree.leaves(trainer._running[0])
  assert len(before) == len(after) and len(after) > 0
  assert any(not np.array_equal(b, np.asarray(a)) for b, a in zip(before, after))
  assert trainer.train_pool.current > 0
