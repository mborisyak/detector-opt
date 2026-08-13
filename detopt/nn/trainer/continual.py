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
      draws both halves from its own window. Past events carry their own scaled
      design in the pool, so ``combine`` handles the mixed-design batch.
    """

    def __init__(self, *args, replay_weight: float = 1.0, replay_alpha: float = 1.0, **kwargs):
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
        # How much a HISTORICAL row counts as evidence when the parameter prior is weighed against the
        # data -- see `_prior_count`. Only read when `prior_scale > 0`; with the prior off it is inert.
        self.replay_alpha = float(replay_alpha)
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

    def _prior_count(self, start, count):
        """``count + replay_alpha * start`` -- the current window PLUS the history the replay half is
        drawn from, since this network is fitted to both.

        WHY IT IS NOT ``count``. The prior's weight against the data is ``1/N``, and ``N`` is the
        evidence the objective actually represents. A continual network's objective is a mixture: half
        of every minibatch comes from ``[0, start)``, so those rows are being fitted too. Charging it
        ``1/count`` would weigh the same prior against only the current design's window -- and since
        ``start`` grows with every design while ``count`` does not, the over-regularisation would GROW
        with the campaign, reaching 20x or more by design 20. `from_scratch` would meanwhile be charged
        correctly. That is an arm-asymmetric handicap manufactured by the trainer, on top of the
        handicap the campaign is trying to measure.

        ``replay_alpha`` is how much a historical row counts against a current one, and it belongs with
        ``replay_weight``: at ``replay_weight = 1.0`` a replayed row has the same say in the gradient as
        a current one, so it is the same kind of evidence and ``replay_alpha = 1.0`` matches. Setting
        ``replay_weight = 0`` (no replay) should be paired with ``replay_alpha = 0``, which recovers the
        per-design denominator exactly.
        """
        return count + self.replay_alpha * start

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
