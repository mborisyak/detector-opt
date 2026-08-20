"""Continual training at a CONFIGURABLE current:replay batch composition.

A SIBLING of :class:`~detopt.nn.trainer.continual.ContinualTrainer`, never a subclass: the growth
procedure in `_DesignBase.train` is concrete and concrete methods are final here, so this class
implements the same abstract hooks itself. `continual.py` is not modified and its class is not
imported.

WHAT IS UNDER TEST. `ContinualTrainer` splits every minibatch 50/50 between the current design's
window and replay, while the number the trainer must drive under `loss_precision` -- and the number
it hands to BO -- is evaluated on the CURRENT design's window alone. `current_replay_ratio` makes
that split a knob: `r` current-design rows per replay row, so `1.0` is the 50/50 batch and larger
values spend more of the gradient on the window that is actually scored.

THE SPLIT, AND WHAT AN UNEVEN BATCH DOES. `n_replay = batch // (1 + r)` and the current design takes
the REMAINDER, `batch - n_replay`, so the batch size is exactly `batch` at every ratio and any
rounding is spent on the current design rather than on replay. At `r = 1` that is `batch // 2` replay
rows and `batch - batch // 2` current rows -- element-for-element the `ContinualTrainer` layout, which
`tests/test_continual_ratio.py` checks against it directly rather than by argument. A ratio large
enough to round `n_replay` to 0 is legal and means a persistent network trained with no replay at all.

`replay_weight` keeps its meaning: what a replay row is worth against a current-design row in the
gradient, with the row weights normalised to mean 1 so the loss keeps the scale `loss_precision` is
stated against. Composition and weight are independent -- one changes how many replay rows there are,
the other what each is worth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .common import design_init_sequence
from .design import _DesignBase
from .replay import carried_network_state, load_carried_network_state, persistent_network

__all__ = ["ContinualRatioTrainer"]


class ContinualRatioTrainer(_DesignBase):
  """ONE persistent network across designs, with replay at a configurable current:replay ratio.

  Everything except the batch composition matches :class:`~detopt.nn.trainer.continual.ContinualTrainer`:
  the same growth procedure, the same persistent network with the optimiser restarted at every design
  boundary, and the same exact-resume payload.
  """

  def __init__(self, *args, replay_weight: float = 1.0, current_replay_ratio: float = 1.0, **kwargs):
    self.replay_weight = float(replay_weight)
    self.current_replay_ratio = float(current_replay_ratio)
    if self.current_replay_ratio <= 0.0:
      raise ValueError(
        f"current_replay_ratio must be > 0 (current-design rows per replay row), "
        f"got {self.current_replay_ratio}"
      )
    super().__init__(*args, **kwargs)
    self._running = persistent_network(self, int(design_init_sequence(self.seed, 0).generate_state(1)[0]))

  def _replay_rows(self):
    """Replay rows per ensemble member: ``batch // (1 + current_replay_ratio)``; the current design
    takes the remaining ``batch - n_replay``."""
    return int(self.batch // (1.0 + self.current_replay_ratio))

  def _sample_indices(self, key, start, count):
    """``n_current`` rows from the current design's window ``[start, start + count)`` and ``n_replay``
    drawn uniformly from all past designs' rows ``[0, start)``.

    With no history (``start == 0``) the replay rows come from the current window too, so the first
    design of a run carries no arm signal. ``start`` and ``count`` are dynamic 0-d int32 ``jax.Array``.
    Indices are laid out ``(members, batch)`` and raveled row-major so the loss's reshape recovers one
    minibatch per member.
    """
    members = self.n_ensemble if self.n_ensemble is not None else 1
    n_replay = self._replay_rows()
    n_current = self.batch - n_replay
    key_current, key_past = jax.random.split(key)
    current = start + jax.random.randint(key_current, (members, n_current), 0, jnp.maximum(count, 1))
    past = jnp.where(
      start > 0, jax.random.randint(key_past, (members, n_replay), 0, jnp.maximum(start, 1)),
      start + jax.random.randint(key_past, (members, n_replay), 0, jnp.maximum(count, 1)),
    )
    return jnp.concatenate([current, past], axis=1).reshape(-1)

  def _sample_weights(self):
    """Row weights matching :meth:`_sample_indices`' layout: 1 for the current design's rows,
    ``replay_weight`` for the replay rows, normalised to mean 1."""
    members = self.n_ensemble if self.n_ensemble is not None else 1
    n_replay = self._replay_rows()
    n_current = self.batch - n_replay
    row = jnp.concatenate([jnp.ones((n_current, ), jnp.float32), jnp.full((n_replay, ), self.replay_weight, jnp.float32)])
    row = row / jnp.mean(row)
    return jnp.broadcast_to(row, (members, self.batch))

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
