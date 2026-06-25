"""Continual trainer: one persistent network across designs, with replay."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .design import _DesignBase

__all__ = ["ContinualTrainer"]


class ContinualTrainer(_DesignBase):
    """Continually trains ONE network across all designs, with experience replay.

    Same shared budget pools, design-conditioned features, and convergence
    procedure as :class:`DesignTrainer`; two things differ:

    * **Persistent network** -- params, non-param state and optimiser state are
      kept on the instance and carried from one :meth:`train` call to the next
      (the network is created once, never re-initialised per design; warm-start
      arguments are ignored).
    * **Replay sampling** -- each minibatch is half from the current design's
      window ``[w0, w0 + count)`` and half drawn uniformly from all past
      iterations' data ``[0, w0)``. The first design (``w0 == 0``, no history)
      draws both halves from its own window. Past events carry their own encoded
      design in the pool, so ``combine`` handles the mixed-design batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ONE persistent network, continued across every design (the continual strategy IS the warm
        # start). Built eagerly here -- never lazily on the first train() call -- so no jax array is
        # cached behind a None. _persist_network writes the continued net back after each design.
        _, params, state = self._build_regressor(self.seed)
        opt_state = self.optimizer.init(params)
        d = self.device
        self._running = (jax.device_put(params, d), jax.device_put(state, d), jax.device_put(opt_state, d))

    def _init_design_network(self, init_seq, init_params):
        # The persistent net is the network for every design; warm-start args are ignored.
        return self._running

    def _persist_network(self, params, state, opt_state):
        self._running = (params, state, opt_state)

    def _sample_indices(self, key, start, count):
        # Half of each member's batch from the current window [start, start+count),
        # half from past iterations [0, start). With no history (start == 0) both
        # halves come from the current window. start/count are dynamic int32
        # jax.Array. Indices are laid out per member -- ``(members, batch)`` raveled
        # row-major -- so the loss's reshape recovers one replay batch per member.
        members = self.n_ensemble or 1
        half = self.batch // 2
        n_cur = self.batch - half
        k_cur, k_past = jax.random.split(key)
        cur = start + jax.random.randint(k_cur, (members, n_cur), 0, jnp.maximum(count, 1))
        past = jnp.where(
            start > 0,
            jax.random.randint(k_past, (members, half), 0, jnp.maximum(start, 1)),
            start + jax.random.randint(k_past, (members, half), 0, jnp.maximum(count, 1)),
        )
        return jnp.concatenate([cur, past], axis=1).reshape(-1)
