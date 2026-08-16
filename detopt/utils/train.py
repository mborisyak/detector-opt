"""Training-loop kernels: an epoch as one scan-folded dispatch, or as donated per-step calls.

``scan=True`` folds ``steps`` SGD steps into a single ``jax.lax.scan`` under one ``jit`` -- one dispatch
per epoch, all data resident on the device. ``scan=False`` jits ONE step with ``donate_argnums`` over
the parameters, the network state and the optimiser state, and drives it from python.

⚠️ DONATION INVALIDATES THE CALLER'S BUFFERS. After a donated call the arrays passed in must not be
read again; only the returned ones are valid. That includes ALIASES -- a caller holding a second
reference to the same parameters (a rewind target, a checkpoint about to be written) loses it too, so
either copy before the loop or pass ``donate=False``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax

__all__ = ["build_train_epoch"]


def build_train_epoch(loss_fn, optimizer, steps, sample, *, scan: bool = True, donate: bool = True,
                      aggregate=None):
  """An ``epoch(params, state, opt_state, key, *data) -> (params, state, opt_state, losses)`` kernel.

    ``loss_fn(params, state, drop_key, *batch) -> (loss, new_state)`` and
    ``sample(index_key, *data) -> batch`` are the caller's; ``data`` is whatever the sampler needs
    (buffers, window offsets) and is passed through untouched.

    Both modes consume the SAME key sequence -- ``jax.random.split(key, steps)`` -- so they take the
    same trajectory and can be differenced. ``aggregate`` reduces the stacked per-step losses; the
    default returns them untouched.
    """
  reduce = (lambda x: x) if aggregate is None else aggregate

  def one_step(params, state, opt_state, key, *data):
    key_index, key_drop = jax.random.split(key)
    batch = sample(key_index, *data)
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, key_drop, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), state, opt_state, loss

  if scan:

    @jax.jit
    def epoch(params, state, opt_state, key, *data):
      def body(carry, k):
        params, state, opt_state = carry
        params, state, opt_state, loss = one_step(params, state, opt_state, k, *data)
        return (params, state, opt_state), loss

      (params, state, opt_state), losses = jax.lax.scan(body, (params, state, opt_state), jax.random.split(key, steps))
      return params, state, opt_state, reduce(losses)

    return epoch

  # By NAME, not position: a positional list silently donates the wrong buffer if the signature moves.
  donated = ("params", "state", "opt_state") if donate else ()
  step = partial(jax.jit, donate_argnames=donated)(one_step)

  def epoch(params, state, opt_state, key, *data):
    losses = []
    for k in jax.random.split(key, steps):
      params, state, opt_state, loss = step(params, state, opt_state, k, *data)
      losses.append(loss)
    return params, state, opt_state, reduce(jnp.stack(losses))

  return epoch
