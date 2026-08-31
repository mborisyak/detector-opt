"""Replay trainers: design revealed, minibatch half current design and half earlier ones.

`ContinualTrainer` (`meta`) additionally CARRIES the trained network across designs;
`ContinualReinitTrainer` (`meta_reinit`) re-initialises it at every design, so the pair
isolates what the carry is worth."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .common import design_init_sequence, fresh_design_network
from .design import _DesignBase

__all__ = ["ContinualTrainer", "ContinualReinitTrainer"]


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


class _ReplayBase(_DesignBase):
  """Design-REVEALED training with experience replay, shared by the strategies that carry a network
    across designs (:class:`ContinualTrainer`) and the one that re-initialises at every design
    (:class:`ContinualReinitTrainer`).

    What lives here is what those two SHARE: the revealed design, the replay minibatch and its row
    weights. The network LIFECYCLE stays abstract, so neither subclass overrides a concrete body.

    Continually trains ONE network across all designs, with experience replay.

    Same shared budget pools and convergence procedure as :class:`DesignTrainer`,
    but NOT the same features: this strategy REVEALS the design to the network
    (:meth:`reveals_design`), where the per-design one withholds it. Two other
    things differ:

    * **Persistent network** -- params and non-param state are kept on the
      instance and carried from one :meth:`train` call to the next (the network
      is created once, never re-initialised per design; warm-start arguments are
      ignored). The OPTIMISER IS RESTARTED AT EVERY DESIGN BOUNDARY: what is
      continued is the network, not Adam's moment estimates, which are a
      property of the window just finished rather than of what has been learned.
      Within a design the moments are likewise rebuilt at every ``rewind``
      rewind (``design.py``), for the same reason.

      That makes the carried state COMPLETE -- params and buffers are all there
      is -- so :meth:`_carried_state` persists everything a design boundary
      needs and a resumed run is EXACT, not approximate.
    * **Replay sampling** -- each minibatch is half from the current design's
      window ``[w0, w0 + count)`` and half drawn uniformly from all past
      iterations' data ``[0, w0)``. The first design (``w0 == 0``, no history)
      draws both halves from its own window. Past events carry their own raw
      physical design in the pool, so ``combine`` handles the mixed-design batch
      and the revealed design is what tells a replay row from a current one.
    """

  def spent_calls(self):
    """Reserves nothing, so the pools' fill IS the spend."""

    return self.train_pool.current + self.val_pool.current

  def _round_extra(self, n_train):
    """Trains on the proposed designs alone, so a round costs exactly what it asked for."""

  def default_reveal(self):
    """A continual strategy: ONE network across many designs, its batch mixing the current design
      with replay from earlier ones, so the design is what tells those rows apart."""
    return 'design'

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
      start > 0, jax.random.randint(k_past, (members, half), 0, jnp.maximum(start, 1)),
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
    row = jnp.concatenate([jnp.ones((n_cur, ), jnp.float32), jnp.full((half, ), self.replay_weight, jnp.float32)])
    row = row / jnp.mean(row)
    return jnp.broadcast_to(row, (members, self.batch))


class ContinualTrainer(_ReplayBase):
  """Replay training that CARRIES one network across every design -- the `meta` strategy.

    The persistent network is the warm start. See :class:`_ReplayBase` for the replay batch, and
    the notes below for why the optimiser moments are NOT among the things carried."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # ONE persistent network, continued across every design (the continual strategy IS the warm
    # start). Built eagerly here -- never lazily on the first train() call -- so no jax array is
    # cached behind a None. _persist_network writes the continued net back after each design.
    _, params, state = self._build_regressor(int(design_init_sequence(self.seed, 0).generate_state(1)[0]))
    d = self.device
    self._running = (jax.device_put(params, d), jax.device_put(state, d))

  def _init_design_network(self, init_seq, init_params):
    """The persistent network, with a FRESH optimiser state. Warm-start arguments are ignored --
        the continual strategy IS the warm start. See the class docstring for why the moments are not
        among the things carried."""
    params, state = self._running
    return params, state, self.optimizer.init(params)

  def _persist_network(self, params, state, opt_state):
    """Keep the trained network for the next design. The optimiser state is DISCARDED here rather
        than stored and ignored, so nothing on the instance can be mistaken for carried moments."""
    self._running = (params, state)

  def _carried_state(self):
    """The persistent network -- PARAMS AND BUFFER STATE, which is ALL that crosses a design
        boundary (user, 2026-08-14; the optimiser restart of the class docstring is what makes that
        true). A resumed continual run therefore starts the next design from exactly the state an
        uninterrupted one would: same params, same buffers, and moments that both rebuild from
        scratch.

        The buffer state holds TYPED PRNG KEYS, which have no numpy representation; they are stored as
        their raw key data and re-wrapped on load against the freshly built network's own dtypes."""
    params, state = self._running
    payload = {}
    for name, tree in (("param", params), ("buffer", state)):
      for i, leaf in enumerate(jax.tree.leaves(tree)):
        payload[f"net_{name}_{i}"] = np.asarray(_key_data(leaf))
    return payload

  def _load_carried_state(self, data):
    params, state = self._running
    restored = []
    for name, tree in (("param", params), ("buffer", state)):
      leaves, structure = jax.tree.flatten(tree)
      loaded = [_like(leaf, data[f"net_{name}_{i}"]) for i, leaf in enumerate(leaves)]
      restored.append(jax.device_put(jax.tree.unflatten(structure, loaded), self.device))
    self._running = (restored[0], restored[1])

  def _replay_carried_state(self, rows):
    from .replay import replay_carried_network

    self._running = replay_carried_network(self, rows)


class ContinualReinitTrainer(_ReplayBase):
  """`meta` with the network carry REMOVED: design revealed, replay batch kept, but a FRESH random
    network at every design.

    This isolates WHICH half of `meta` does the work. `meta` bundles three things -- the design is
    revealed to the network, the minibatch replays earlier designs, and the trained network is
    carried across the design boundary. This strategy keeps the first two and drops the third, so a
    difference against `meta` is attributable to the carry alone, and a difference against
    `from_scratch` to the reveal-plus-replay alone.

    It is NOT `from_scratch`: that arm withholds the design and samples only its own window.
    """

  def _init_design_network(self, init_seq, init_params):
    """A fresh random network per design. ``init_params`` is IGNORED -- a warm start is exactly
        what this strategy exists to remove, so honouring it would defeat the ablation."""
    return fresh_design_network(self, init_seq, None)

  def _persist_network(self, params, state, opt_state):
    """Nothing is kept: the next design draws its own network."""

  def _carried_state(self):
    return {}  # nothing survives a design boundary

  def _load_carried_state(self, data):
    """Nothing to load -- see :meth:`_carried_state`."""

  def _replay_carried_state(self, rows):
    """Nothing crosses a design boundary here -- see :meth:`_carried_state`."""
