"""Continual training with the minibatch drawn UNIFORMLY from every row in the pool.

A SIBLING of :class:`~detopt.nn.trainer.continual.ContinualTrainer`, never a subclass: the growth
procedure in `_DesignBase.train` is concrete and concrete methods are final here, so this class
implements the same abstract hooks itself, exactly as `continual_ratio.py` does.

WHAT IS UNDER TEST. `ContinualTrainer` fixes the batch composition at 1:1 -- half the rows from the
current design's window, half replayed from earlier designs -- whatever the pool holds. This trainer
has no composition at all: every row of the minibatch is drawn uniformly from ``[0, start + count)``,
the whole pool, so the current design's share of a batch is its share of the data, ``count / (start +
count)``, and shrinks as the history grows. No replay weight either: every row weighs 1. A design that
is expensive to fit therefore gets a smaller and smaller fraction of the gradient the later it comes
in the trajectory, which is the effect the ablation measures against the 1:1 batch.

At ``start == 0`` (the first design of a run) every row is a current-design row and the batch is what
`ContinualTrainer` draws too, so the difference between the two trainers is confined to designs with
a history, as it must be.

Everything else -- the persistent network with a fresh optimiser at every design boundary, the
exact-resume payload, the revealed design -- is the continual arm's, unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .common import design_init_sequence
from .design import _DesignBase
from .replay import carried_network_state, load_carried_network_state, persistent_network, replay_carried_network

__all__ = ["ContinualUniformTrainer"]


class ContinualUniformTrainer(_DesignBase):
  """ONE persistent network across designs; each minibatch is a uniform draw from the whole pool."""

  def spent_calls(self):
    """Reserves nothing, so the pools' fill IS the spend."""

    return self.train_pool.current + self.val_pool.current

  def _round_extra(self, n_train):
    """Trains on the proposed designs alone, so a round costs exactly what it asked for."""

  def default_reveal(self):
    """A continual strategy: the batch mixes designs, so the design is what tells the rows apart."""
    return 'design'

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._running = persistent_network(self, int(design_init_sequence(self.seed, 0).generate_state(1)[0]))

  def _sample_indices(self, key, start, count):
    """``batch`` rows per ensemble member drawn uniformly from ``[0, start + count)`` -- the past
    designs' rows and the current window together, with no composition imposed.

    ``start`` and ``count`` are dynamic 0-d int32 ``jax.Array``. Indices are laid out ``(members,
    batch)`` and raveled row-major so the loss's reshape recovers one minibatch per member.
    """
    members = self.n_ensemble if self.n_ensemble is not None else 1
    total = jnp.maximum(start + count, 1)
    return jax.random.randint(key, (members, self.batch), 0, total).reshape(-1)

  def _sample_weights(self):
    """Every row weighs 1: a uniform draw has no replay half to reweight."""
    members = self.n_ensemble if self.n_ensemble is not None else 1
    return jnp.ones((members, self.batch), jnp.float32)

  def _init_design_network(self, init_seq, init_params):
    """The persistent network with a FRESH optimiser state; warm-start arguments are ignored."""
    params, state = self._running
    return params, state, self.optimizer.init(params)

  def _persist_network(self, params, state, opt_state):
    """Keep the trained network for the next design; the optimiser state is discarded, not stored."""
    self._running = (params, state)

  def _carried_state(self):
    return carried_network_state(self._running)

  def _load_carried_state(self, data):
    self._running = load_carried_network_state(self._running, data, self.device)

  def _replay_carried_state(self, rows):
    self._running = replay_carried_network(self, rows)
