"""The cosine-floor schedule: an optax transform whose counter a data addition resets, leaving the inner optimiser alone."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from detopt.nn.trainer.schedule import cosine_floor_schedule, rephase, schedule_count, with_cosine_floor


def _step(optimizer, state, params, grads):
  updates, state = optimizer.update(grads, state, params)
  return optax.apply_updates(params, updates), state


def test_schedule_values():
  schedule = cosine_floor_schedule(steps_per_epoch=8, epochs=4, peak=4.0)
  assert float(schedule(0)) == pytest.approx(4.0)
  assert float(schedule(16)) == pytest.approx(2.5)
  assert float(schedule(32)) == pytest.approx(1.0)
  assert float(schedule(10_000)) == pytest.approx(1.0)


def test_rephase_resets_the_schedule_and_nothing_else():
  optimizer = with_cosine_floor(optax.adamaxw(1e-3, weight_decay=1e-3), cosine_floor_schedule(4, 2, 4.0))
  params = {'w': jnp.ones(3)}
  state = optimizer.init(params)
  grads = {'w': jnp.full(3, 0.5)}
  for _ in range(5):
    params, state = _step(optimizer, state, params, grads)
  assert schedule_count(state) == 5
  adam_before = state[0][0]
  state = rephase(state)
  assert schedule_count(state) == 0
  adam_after = state[0][0]
  assert int(adam_after.count) == 5
  np.testing.assert_array_equal(np.asarray(adam_after.mu['w']), np.asarray(adam_before.mu['w']))
  np.testing.assert_array_equal(np.asarray(adam_after.nu['w']), np.asarray(adam_before.nu['w']))


def test_first_step_after_rephase_is_peak_times_the_inner_update():
  inner = optax.adamaxw(1e-3, weight_decay=1e-3)
  optimizer = with_cosine_floor(inner, cosine_floor_schedule(4, 2, 4.0))
  params = {'w': jnp.ones(3)}
  grads = {'w': jnp.full(3, 0.5)}
  inner_updates, _ = inner.update(grads, inner.init(params), params)
  updates, _ = optimizer.update(grads, optimizer.init(params), params)
  np.testing.assert_allclose(np.asarray(updates['w']), 4.0 * np.asarray(inner_updates['w']), rtol=1e-6)


def test_refuses_an_inner_optimiser_with_its_own_schedule():
  inner = optax.adamaxw(optax.cosine_decay_schedule(1e-3, 10))
  with pytest.raises(ValueError):
    with_cosine_floor(inner, cosine_floor_schedule(4, 2, 4.0))


def test_schedule_count_requires_exactly_one_counter():
  state = optax.adamaxw(1e-3).init({'w': jnp.ones(2)})
  with pytest.raises(ValueError):
    schedule_count(state)
