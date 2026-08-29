"""The CONTINUAL strategy's hooks, as module functions.

`detopt/nn/trainer/common.py` already establishes this idiom for the per-design lifecycle
(`window_sample_indices`, `fresh_design_network`): a strategy IMPLEMENTS the abstract `Trainer` hooks
by DELEGATING to a module function, rather than by overriding a concrete default. This module does the
same for replay, so the several training PROCEDURES that offer a continual arm share one
implementation instead of each carrying its own copy.

⚠️ `ContinualTrainer` in :mod:`.continual` still carries its own copy of this logic and is NOT changed
here -- it is the growth procedure's arm and is in flight. The two are equivalent today; an audit
verified them token-for-token. If either is edited, the other must follow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
  "replay_sample_indices", "replay_sample_weights", "persistent_network", "carried_network_state", "load_carried_network_state"
]


def replay_sample_indices(trainer, key, start, count):
  """Half of each member's batch from the current design's window ``[start, start + count)``, half
  drawn uniformly from all past designs' rows ``[0, start)``.

  With no history (``start == 0``) BOTH halves come from the current window, so the first design of a
  run is statistically identical to the per-design strategy and carries no arm signal. ``start`` and
  ``count`` are dynamic 0-d int32 ``jax.Array``. Indices are laid out ``(members, batch)`` and
  raveled row-major so the loss's reshape recovers one minibatch per member.
  """
  members = trainer.n_ensemble if trainer.n_ensemble is not None else 1
  half = trainer.batch // 2
  n_current = trainer.batch - half
  key_current, key_past = jax.random.split(key)
  current = start + jax.random.randint(key_current, (members, n_current), 0, jnp.maximum(count, 1))
  past = jnp.where(
    start > 0, jax.random.randint(key_past, (members, half), 0, jnp.maximum(start, 1)),
    start + jax.random.randint(key_past, (members, half), 0, jnp.maximum(count, 1)),
  )
  return jnp.concatenate([current, past], axis=1).reshape(-1)


def replay_sample_weights(trainer):
  """Row weights matching :func:`replay_sample_indices`' layout: 1 for the current design's half,
  ``trainer.replay_weight`` for the replay half, normalised to mean 1 so the loss keeps the scale
  ``loss_precision`` is stated against."""
  members = trainer.n_ensemble if trainer.n_ensemble is not None else 1
  half = trainer.batch // 2
  n_current = trainer.batch - half
  row = jnp.concatenate([jnp.ones((n_current, ), jnp.float32), jnp.full((half, ), trainer.replay_weight, jnp.float32)])
  row = row / jnp.mean(row)
  return jnp.broadcast_to(row, (members, trainer.batch))


def persistent_network(trainer, seed):
  """The ONE network a continual arm carries, built eagerly -- never lazily behind a ``None``."""
  _, params, state = trainer._build_regressor(seed)
  device = trainer.device
  return (jax.device_put(params, device), jax.device_put(state, device))


def carried_network_state(running):
  """``(params, state)`` as something numpy can hold. Buffer state holds TYPED PRNG KEYS, which have
  no numpy representation, so they are stored as raw key data and re-wrapped on load."""
  from .continual import _key_data

  params, state = running
  payload = {}
  for name, tree in (("param", params), ("buffer", state)):
    for index, leaf in enumerate(jax.tree.leaves(tree)):
      payload[f"net_{name}_{index}"] = np.asarray(_key_data(leaf))
  return payload


def load_carried_network_state(running, data, device):
  """The inverse of :func:`carried_network_state`, against the freshly built network's own dtypes."""
  from .continual import _like

  params, state = running
  restored = []
  for name, tree in (("param", params), ("buffer", state)):
    leaves, structure = jax.tree.flatten(tree)
    loaded = [_like(leaf, data[f"net_{name}_{index}"]) for index, leaf in enumerate(leaves)]
    restored.append(jax.device_put(jax.tree.unflatten(structure, loaded), device))
  return (restored[0], restored[1])


def replay_carried_network(trainer, rows):
  """The persistent network, read back from the LAST replayed design's checkpoint.

  The continual strategies carry one network across designs and it is already saved per design, so a
  replayed run recovers it from that checkpoint rather than from a state file. The restored trees come
  back as PURE DICTS; they are poured into the live network's own structure so leaf types, dtypes and
  device placement are the running network's, not the checkpoint's."""
  from ...utils import io

  if len(rows) == 0:
    return trainer._running
  manager = trainer._checkpoint_manager(len(rows) - 1)
  if manager is None:
    raise ValueError(
      "a continual strategy cannot be replayed without checkpoints: the persistent "
      "network crosses design boundaries and is not derivable from the trajectory"
    )
  params, state = trainer._running
  restored_params, restored_state, _design, _aux = io.restore_training_checkpoint(manager, regressor=(params, state))
  manager.close()
  recovered = []
  for live, loaded in ((params, restored_params), (state, restored_state)):
    leaves, structure = jax.tree.flatten(live)
    recovered.append(jax.device_put(jax.tree.unflatten(structure, jax.tree.leaves(loaded)), trainer.device))
  return recovered[0], recovered[1]
