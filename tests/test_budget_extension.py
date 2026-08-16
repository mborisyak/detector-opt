"""Resuming a run under a RAISED budget: the pool grows, and no event is issued twice.

The failure these guard against is silent in both directions. Before the fix, `Pool.load` took its
capacity from the restored arrays, so a raised budget was discarded and the run ended on its first
round with "pool exhausted", having added nothing. And had the capacity alone been fixed, the
train/val cut -- `stream[:train_budget]` / `stream[train_budget:]` -- MOVES with the budget, so the
resumed training pool would refill from exactly the stream positions the validation pool already
held: at 2097152 -> 3145728 that is every one of the 524288 stored validation events reappearing as
training data, with nothing raising and every reported loss quietly invalid.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from detopt.utils.pools import Pool


def _specs(width=3):
  return (jnp.zeros((width, ), jnp.float32), )


def _state(rows, capacity, width=3):
  slots = (np.arange(capacity * width, dtype=np.float32).reshape(capacity, width), )
  return {"slots": slots, "current": rows}


def test_pool_load_defaults_to_the_snapshot_capacity():
  pool = Pool.load(_state(4, 8), _specs())
  assert pool.capacity == 8
  assert pool.current == 4


def test_pool_load_grows_to_the_requested_capacity_and_keeps_the_rows():
  stored = _state(4, 8)
  pool = Pool.load(stored, _specs(), capacity=16)
  assert pool.capacity == 16
  assert pool.current == 4
  kept = np.asarray(pool.buffers()[0])[:4]
  np.testing.assert_array_equal(kept, stored["slots"][0][:4])
  assert pool.capacity - pool.current == 12


def test_pool_load_refuses_to_shrink_below_the_fill():
  with pytest.raises(ValueError, match="below the snapshot's fill"):
    Pool.load(_state(6, 8), _specs(), capacity=4)


def _layout(generations, size, seed):
  """The train/val index arrays a trainer would build for these generations."""
  from detopt.utils.events import shuffled_event_index

  total = sum(train + val for train, val in generations)
  stream = shuffled_event_index(size, total, seed)
  train_parts, val_parts, at = [], [], 0
  for train, val in generations:
    train_parts.append(stream[at:at + train])
    at += train
    val_parts.append(stream[at:at + val])
    at += val
  return np.concatenate(train_parts), np.concatenate(val_parts)


def test_raising_the_budget_keeps_every_consumed_position_and_stays_disjoint():
  seed, size = 12345, None
  train0, val0 = 1572864, 524288
  extra_train, extra_val = 786432, 262144

  before_train, before_val = _layout([(train0, val0)], size, seed)
  after_train, after_val = _layout([(train0, val0), (extra_train, extra_val)], size, seed)

  # Everything already consumed keeps the event it had: a resumed pool's cursor still points at the
  # same place in the same stream.
  np.testing.assert_array_equal(after_train[:train0], before_train)
  np.testing.assert_array_equal(after_val[:val0], before_val)

  # And the increment does not re-issue the other pool's stored events. The bar is the BACKGROUND
  # rate, not zero: an analytic source draws WITH REPLACEMENT from [0, 2**31), so two disjoint blocks
  # of n1 and n2 draws still share about n1*n2/2**31 values by chance -- ~192 here. The moving-cut
  # layout shares 524225, i.e. the entire stored validation set, so 1% of it separates the two cases
  # by three orders of magnitude and cannot be met by accident.
  assert len(np.intersect1d(after_train[train0:], before_val)) < val0 // 100
  assert len(np.intersect1d(after_val[val0:], before_train)) < val0 // 100


def test_the_moving_cut_layout_would_have_failed_this():
  """The bug, written as a test, so the fix is not mistaken for a no-op."""
  from detopt.utils.events import shuffled_event_index

  seed = 12345
  train0, val0 = 1572864, 524288
  train1 = 2359296

  small = shuffled_event_index(None, train0 + val0, seed)
  stored_val = small[train0:train0 + val0]
  large = shuffled_event_index(None, train1 + 786432, seed)
  naive_train_increment = large[train0:train1]

  overlap = np.intersect1d(naive_train_increment, stored_val)
  # Not exactly val0: the source draws with replacement, so `stored_val` holds ~64 duplicate values
  # and intersect1d counts uniques. "Essentially all of it" is the claim, and 99% states it without
  # pretending to a precision the sampling does not have.
  assert len(overlap) > 0.99 * val0, "the moving cut re-issues the whole stored validation set as training data"
