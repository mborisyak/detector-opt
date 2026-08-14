"""Continual trainer: one persistent network across designs, with replay."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .design import _DesignBase

__all__ = ["ContinualTrainer"]


def _is_key(leaf):
    return jnp.issubdtype(jnp.asarray(leaf).dtype, jax.dtypes.prng_key)


def _key_data(leaf):
    """A leaf as something numpy can hold: a typed PRNG key becomes its raw key data."""
    return jax.random.key_data(leaf) if _is_key(leaf) else leaf


def _like(reference, stored):
    """``stored`` back in ``reference``'s own dtype -- re-wrapping key data into a typed key, and using
    the reference's key IMPLEMENTATION rather than the default, so a network built under a non-default
    one restores as itself."""
    if _is_key(reference):
        return jax.random.wrap_key_data(jnp.asarray(stored, jnp.uint32), impl=jax.random.key_impl(reference))
    return jnp.asarray(stored, reference.dtype)


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
      draws both halves from its own window. Past events carry their own scaled
      design in the pool, so ``combine`` handles the mixed-design batch.
    """

    def __init__(self, *args, replay_weight: float = 1.0, **kwargs):
        # What a REPLAY row is worth against a current-design row in the gradient. 1.0 reproduces the
        # original behaviour exactly (a plain mean over the 50/50 batch).
        #
        # WHY IT IS A KNOB AT ALL. The batch is half current design and half replay, but the number
        # this trainer must drive below `loss_precision` -- and the number it hands to BO -- is
        # evaluated on the CURRENT design's window ALONE (`_eval_train` reads from `w0_train`
        # forward). So at `replay_weight = 1.0` half of every gradient is spent on rows that
        # contribute to no measured quantity, a fixed 2x dilution of the current design's signal from
        # the second design onward, however large the pool grows.
        #
        # MEASURED consequence at 1.0 (three-class campaign, designs 6/13/19, paired, 65536 held-out
        # events): the continual arm reports +0.0184 / +0.0214 / +0.0326 of loss above a fresh network
        # on the same design's data -- 2.3 to 4.1 x `loss_precision`, at 23-41 sigma. It is a BIAS,
        # not a data shortage: doubling every per-design count recovers only 0 / 41 / 24% of it.
        self.replay_weight = float(replay_weight)
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

    def _carried_state(self):
        """The persistent network -- PARAMS AND BUFFER STATE, NOT THE OPTIMISER MOMENTS (user,
        2026-08-14). The moments are re-initialised from the restored params on resume, so a resumed
        continual run differs from an uninterrupted one by one design's worth of Adam warm-up on the
        design that follows the restart. Say so rather than let it pass as an exact resume.

        The buffer state holds TYPED PRNG KEYS, which have no numpy representation; they are stored as
        their raw key data and re-wrapped on load against the freshly built network's own dtypes."""
        params, state, _ = self._running
        payload = {}
        for name, tree in (("param", params), ("buffer", state)):
            for i, leaf in enumerate(jax.tree.leaves(tree)):
                payload[f"net_{name}_{i}"] = np.asarray(_key_data(leaf))
        return payload

    def _load_carried_state(self, data):
        params, state, _ = self._running
        restored = []
        for name, tree in (("param", params), ("buffer", state)):
            leaves, structure = jax.tree.flatten(tree)
            loaded = [_like(leaf, data[f"net_{name}_{i}"]) for i, leaf in enumerate(leaves)]
            restored.append(jax.device_put(jax.tree.unflatten(structure, loaded), self.device))
        self._running = (restored[0], restored[1], self.optimizer.init(restored[0]))

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

    def _sample_weights(self):
        """Row weights matching ``_sample_indices``' layout: 1 for the current design, ``replay_weight``
        for the replay half. Normalised to mean 1 so the loss keeps the scale ``loss_precision`` is
        stated against, which means the convergence rule and the reported objective are unaffected by
        the choice of weight."""
        members = self.n_ensemble or 1
        half = self.batch // 2
        n_cur = self.batch - half
        row = jnp.concatenate([jnp.ones((n_cur,), jnp.float32),
                               jnp.full((half,), self.replay_weight, jnp.float32)])
        row = row / jnp.mean(row)
        return jnp.broadcast_to(row, (members, self.batch))
