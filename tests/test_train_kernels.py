"""`build_train_epoch`: the scan-folded and donated per-step kernels must agree."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from detopt.utils.train import build_train_epoch

STEPS = 8


def _fixture(seed=0):
  """A tiny least-squares problem: params (4,), data (64, 4), targets (64,)."""
  rng = np.random.default_rng(seed)
  data = jnp.asarray(rng.standard_normal((64, 4)), jnp.float32)
  truth = jnp.asarray(rng.standard_normal(4), jnp.float32)
  target = data @ truth

  def sample(key, data, target):
    index = jax.random.randint(key, (16, ), 0, data.shape[0])
    return data[index], target[index]

  def loss_fn(params, state, drop_key, batch_x, batch_y):
    return jnp.mean(jnp.square(batch_x @ params - batch_y)), state

  return data, target, sample, loss_fn


def _run(scan, donate, seed=0):
  data, target, sample, loss_fn = _fixture(seed)
  optimizer = optax.adam(1e-2)
  params = jnp.zeros(4, jnp.float32)
  state = jnp.zeros((), jnp.float32)  # a stand-in for non-param network state
  epoch = build_train_epoch(loss_fn, optimizer, STEPS, sample, scan=scan, donate=donate)
  params, state, _opt, losses = epoch(params, state, optimizer.init(params), jax.random.PRNGKey(1), data, target)
  return np.asarray(params), np.asarray(losses)


def test_scan_and_step_take_the_same_trajectory():
  """Both modes split the same key the same way, so they must land on identical parameters -- this is
    what makes a donated per-step run substitutable for the scan-folded one."""
  scan_params, scan_losses = _run(scan=True, donate=False)
  step_params, step_losses = _run(scan=False, donate=True)
  assert np.allclose(scan_params, step_params, atol=1e-6)
  assert np.allclose(scan_losses, step_losses, atol=1e-6)


def test_donation_does_not_change_the_result():
  donated, _ = _run(scan=False, donate=True)
  plain, _ = _run(scan=False, donate=False)
  assert np.array_equal(donated, plain)


def test_it_actually_trains():
  _params, losses = _run(scan=True, donate=False)
  assert losses[-1] < losses[0]


def test_donated_buffers_are_invalidated():
  """The hazard the docstring warns about, asserted so it cannot be forgotten: after a donated call the
    caller's array is gone, and anything aliasing it is gone too."""
  data, target, sample, loss_fn = _fixture()
  optimizer = optax.adam(1e-2)
  params = jnp.zeros(4, jnp.float32)
  alias = params
  epoch = build_train_epoch(loss_fn, optimizer, 1, sample, scan=False, donate=True)
  epoch(params, jnp.zeros(()), optimizer.init(params), jax.random.PRNGKey(0), data, target)
  with pytest.raises(RuntimeError):
    np.asarray(alias)
