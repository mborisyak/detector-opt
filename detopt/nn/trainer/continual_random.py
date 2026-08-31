"""Continual replay training with a THIRD source: events at uniformly random designs.

WHAT IS UNDER TEST. `ContinualTrainer` mixes each minibatch from two sources -- the current design's
window and replay from the designs BO already proposed. Both are self-selected: replay holds only
what the search chose to look at, which concentrates near wherever it is already heading. This
trainer adds a source that is not self-selected at all, events simulated at designs drawn uniformly
over the box, so the network's design-conditioning is trained on support the search never visits.
The trade is paid in the SHARED BUDGET (user, 2026-08-29): random events are detector calls like any
other, so the arm reaches its first design having spent `alpha` of the run on designs BO never scores.

MAXIMAL VARIATION: ONE EVENT, ONE DESIGN. The random rows do not share designs -- every random event
carries its own draw. `_fill_pool` already hands the detector a per-event design tree and merely
fills it with copies, so this costs nothing measurable: a 256-event batch takes 17.1 ms with
all-distinct designs against 16.6 ms with identical ones.

WHERE THE RANDOM EVENTS LIVE, and why not in a pool of their own. The training kernel indexes ONE
buffer -- `train_step` gathers from the `buffers` argument, which `_DesignBase.train` always fills
from `train_pool` -- and both are concrete, so a separate `Pool` is not reachable from the sampler.
The random events therefore occupy a RESERVED PREFIX `[0, random_capacity)` of the train pool, and
the observed data appends after it. That keeps three sources addressable by index arithmetic over the
one buffer, needs no change to any shared method, and has the side benefit that `bo.py` counts the
random events in `detector_calls_used` for free, so the plotted x-axis is the true spend.

RANDOM DATA ARRIVES WITH THE ROUND. `Trainer._round_extra` is called by `_sample_round` the moment a
round's own events land, so "the trainer asked for more data" and "the extra data arrived" are ONE
event and the pools can never disagree about how much of the budget is spent.

THE PREFIX FILL IS A FUNCTION OF THE OBSERVED FILL, never a stored cursor. `train_epoch` is jitted
ONCE in `_build_kernels`, so anything the sampler closes over is baked at trace time and a growing
cursor would go stale. Instead the filler and the sampler compute the same `_random_fill(observed)` --
host-side from `train_pool.current`, inside the kernel from `start + count`, which are the same number
because a round ends with the window running to the pool's fill. The frozen leaf returns the whole
prefix; the online leaf returns a fixed ratio of the observed fill.

WEIGHTS ARE PER-SOURCE MEANS, NOT PER-ROW. Each source's rows are averaged and the three means are
combined at `1 : replay_weight : random_weight`, so composition sets the VARIANCE of each estimate
and the weights alone set the effective say in the gradient. `ContinualTrainer` needs no such
distinction because its batch is symmetric -- at 50/50 the two schemes coincide exactly -- but at
2:1:1 they differ (8:1:1 against 4:1:1), so the choice has to be explicit.

⚠️ VALIDATION UNDERFILLS, BY CONSTRUCTION. Random events have no validation counterpart (user), yet
they consume train-pool capacity, so `_sample_round` generates val events for the observed rows only.
The run still ends when the train pool fills; the val pool simply stops short. At `alpha = 0.125` and
`val_fraction = 0.25` the arm spends about 0.958 of the configured budget rather than all of it,
which is visible as a slightly shorter curve and is NOT a censored result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .common import design_init_sequence, shuffled_event_index
from .design import _DesignBase
from .replay import carried_network_state, load_carried_network_state, persistent_network, replay_carried_network

__all__ = ["ContinualRandomFrozenTrainer", "ContinualRandomOnlineTrainer"]

# Entropy for the random-design stream. A TWO-ELEMENT root, because `design_init_sequence` already
# consumes `SeedSequence(self.seed).spawn(k)` indexed by BO step: taking child k here would hand the
# random designs the same state as design k's network initialisation and silently correlate them.
# A list entropy has an empty spawn key and cannot equal any child of the scalar root.
RANDOM_STREAM_TAG = 0x52414E44


class _RandomReplayBase(_DesignBase):
  """Persistent network + replay + a reserved prefix of events at uniformly random designs.

  A SIBLING of the continual trainers, never a subclass: the growth procedure is concrete and
  concrete methods are final here, so this class implements the same abstract hooks itself. When the
  random prefix is filled is the one thing left abstract (:meth:`_random_fill`), and the two leaves
  below answer it.
  """

  def default_reveal(self):
    """A continual strategy: one network across many designs and a batch that mixes three sources, so
    the design is what tells those rows apart."""
    return 'design'

  def _random_fill(self, observed):
    """How many rows of the random prefix are filled once ``observed`` events sit in the train pool.
    ABSTRACT -- the schedule IS the experiment.

    Called in two places that must agree EXACTLY: host-side from ``_round_extra`` with
    ``train_pool.current``, and inside the jitted sampler with ``start + count``. Those are the same
    number -- a round ends with the window running to the pool's fill -- which is what lets the fill be
    derived rather than stored, and a stored cursor would be baked in at trace time and go stale."""
    raise NotImplementedError()

  def __init__(self, *args, alpha: float = 0.125, replay_weight: float = 0.125, random_weight: float = 0.125, **kwargs):
    self.alpha = float(alpha)
    if not 0.0 < self.alpha < 1.0:
      raise ValueError(f"alpha is the fraction of the budget spent on random designs, in (0, 1); got {self.alpha}")
    # What a replay / random row is worth against a current-design row, as a PER-SOURCE weight: the
    # rows of each source are averaged first, so these are the effective say in the gradient and the
    # 2:1:1 composition only sets how noisy each of the three means is.
    self.replay_weight = float(replay_weight)
    self.random_weight = float(random_weight)
    budget = int(kwargs["budget"])
    super().__init__(*args, **kwargs)

    self.random_capacity = int(round(budget * self.alpha))
    if not 0 < self.random_capacity < self.train_pool.capacity:
      raise ValueError(
        f"alpha={self.alpha} reserves {self.random_capacity} of a {self.train_pool.capacity}-event "
        f"train pool; it must leave room for the observed designs"
      )
    # The observed data starts AFTER the reserved prefix. Advancing the cursor is what reserves it:
    # `_fill_pool` writes at `pool.current`, so every observed append lands past the random rows.
    self.train_pool.current = self.random_capacity

    # The random designs come from their own stream so a requeued cell reproduces them with NOTHING
    # stored -- the same contract the rest of the run uses, where resuming is replaying a sequence
    # rather than restoring a generator position.
    self._random_seq = np.random.SeedSequence([int(self.seed), RANDOM_STREAM_TAG])
    # Random events take the DISJOINT TAIL of a longer draw from the run's own shuffled stream, so no
    # event is ever used twice: `shuffled_event_index` shares its prefix with a shorter draw, which is
    # exactly the property that makes the tail free of the observed pools' positions.
    observed = self.train_pool.capacity + self.val_pool.capacity
    stream = shuffled_event_index(self.detector.size(), observed + self.random_capacity, self.seed)
    self._random_index = stream[observed:]
    self._random_current = 0

    self._running = persistent_network(self, int(design_init_sequence(self.seed, 0).generate_state(1)[0]))
    # The prefix is never empty when the first design opens: frozen fills it whole here, online fills
    # the share its opening round has already earned.
    self._fill_random(self._random_fill(self.train_pool.current))

  # ------------------------------------------------------------------ #
  # The random prefix.
  # ------------------------------------------------------------------ #
  def _draw_random_designs(self, n):
    """``n`` independent designs, uniform on the SCALED cube, as a batched physical design tree.

    Drawn from the trainer's own random stream at the position the prefix has already reached, so the
    draw is a function of (seed, prefix offset) alone and a replayed run reproduces it row for row."""
    child = self._random_seq.spawn(1)[0]
    bits = np.random.default_rng(np.random.SeedSequence([int(child.generate_state(1)[0]), self._random_current]))
    scaled = bits.random((int(n), self.detector.design_dim())).astype(np.float32)
    return self.detector.to_nominal(jnp.asarray(scaled))

  def _fill_random(self, target):
    """Fill the reserved prefix up to ``target`` rows, one fresh design per event.

    Writes through ``Pool.assign`` rather than ``append``: the pool's own cursor belongs to the
    observed data, which is already past the prefix."""
    target = int(min(int(target), self.random_capacity))
    pool = self.train_pool
    while self._random_current < target:
      k = min(256, target - self._random_current)
      at = self._random_current
      design_b = self._draw_random_designs(k)
      _gt, event, mask, target_b = self.detector(design_b, self._random_index[at:at + k])
      index = jnp.arange(at, at + k, dtype=jnp.int32)
      pool._buffers = pool.assign(pool._buffers, index, (event, mask, target_b, design_b))
      self._random_current = at + k

  def spent_calls(self):
    """The pools' fill MINUS the reserved prefix rows not yet simulated.

    The cursor is what reserves the prefix, so ``train_pool.current`` counts the whole of it from
    construction while the online schedule has paid for only part. Reporting the fill would claim
    budget with no detector call behind it -- up to ``alpha`` of the run at its start."""
    unsimulated = self.random_capacity - self._random_current
    return self.train_pool.current + self.val_pool.current - unsimulated

  def random_spent(self):
    """Detector calls this trainer has spent on random designs. What ``bo.py`` records per iteration so
    the cumulative-calls axis is the true spend."""
    return int(self._random_current)

  # ------------------------------------------------------------------ #
  # Sampling: three sources over the ONE train buffer.
  # ------------------------------------------------------------------ #
  def _source_rows(self):
    """Rows per ensemble member for (current, historical, random), composing 2:1:1 with any rounding
    spent on the current design -- the source whose window is the one actually scored."""
    quarter = self.batch // 4
    return self.batch - 2 * quarter, quarter, quarter

  def _sample_indices(self, key, start, count):
    """Indices into the train buffer for one step: the current window ``[start, start + count)``, the
    observed history ``[random_capacity, start)``, and the random prefix ``[0, _random_fill(start))``.

    WITH NO HISTORY the historical rows come from the current window, matching what
    ``ContinualTrainer`` does at its first design so design 1 stays a matched comparison. The random
    prefix is never empty -- it is filled before the first design opens in both schedules."""
    members = self.n_ensemble or 1
    n_current, n_history, n_random = self._source_rows()
    reserved = jnp.int32(self.random_capacity)
    k_cur, k_hist, k_rand = jax.random.split(key, 3)

    current = start + jax.random.randint(k_cur, (members, n_current), 0, jnp.maximum(count, 1))
    has_history = start > reserved
    history = jnp.where(
      has_history, reserved + jax.random.randint(k_hist, (members, n_history), 0, jnp.maximum(start - reserved, 1)),
      start + jax.random.randint(k_hist, (members, n_history), 0, jnp.maximum(count, 1)),
    )
    filled = jnp.maximum(jnp.int32(self._random_fill(start + count)), 1)
    random = jax.random.randint(k_rand, (members, n_random), 0, filled)
    return jnp.concatenate([current, history, random], axis=1).reshape(-1)

  def _sample_weights(self):
    """Row weights giving PER-SOURCE MEANS combined at ``1 : replay_weight : random_weight``.

    Each source's rows carry ``weight / n_rows``, so the weighted mean over the batch is the weighted
    mean OF THE THREE SOURCE MEANS and the composition drops out of the effective weight. Normalised
    to mean 1 so the loss keeps the scale ``loss_precision`` is stated against."""
    members = self.n_ensemble or 1
    counts = self._source_rows()
    weights = (1.0, self.replay_weight, self.random_weight)
    row = jnp.concatenate([jnp.full((n, ), w / n, jnp.float32) for n, w in zip(counts, weights)])
    row = row / jnp.mean(row)
    return jnp.broadcast_to(row, (members, self.batch))

  # ------------------------------------------------------------------ #
  # Network lifecycle: one persistent network, as the continual strategies.
  # ------------------------------------------------------------------ #
  def _round_extra(self, n_train):
    """Top the random prefix up to the share this much observed data has earned.

    Called by ``_sample_round`` immediately after the round's own events land, so the random events
    arrive WITH the data the trainer asked for and the sampler's ``_random_fill(start + count)`` always
    describes rows that have actually been simulated."""
    self._fill_random(self._random_fill(self.train_pool.current))

  def _init_design_network(self, init_seq, init_params):
    """The persistent network with a FRESH optimiser state. Warm-start arguments are ignored -- the
    continual strategy IS the warm start."""
    params, state = self._running
    return params, state, self.optimizer.init(params)

  def _persist_network(self, params, state, opt_state):
    self._running = (params, state)

  def _carried_state(self):
    return carried_network_state(self._running)

  def _load_carried_state(self, data):
    self._running = load_carried_network_state(self._running, data, self.device)

  def _replay_carried_state(self, rows):
    self._running = replay_carried_network(self, rows)


class ContinualRandomFrozenTrainer(_RandomReplayBase):
  """The random prefix is filled IN FULL before the first design -- `meta_random_frozen`.

  The whole random cost is paid up front, so the arm reaches design 1 having already spent ``alpha``
  of the budget and every design after it sees the same, complete random support."""

  def _random_fill(self, observed):
    return self.random_capacity


class ContinualRandomOnlineTrainer(_RandomReplayBase):
  """The random prefix grows with the observed data -- `meta_random_online`.

  Held at ``alpha / (1 - alpha)`` of the observed fill, so the random cost is spread across the run
  instead of paid up front and the two leaves differ ONLY in when the spend happens: the data
  distribution they draw from is identical."""

  def _random_fill(self, observed):
    share = self.random_capacity / max(self.train_pool.capacity - self.random_capacity, 1)
    # `observed` counts the reserved prefix, so the earned share is measured on the OBSERVED rows alone.
    # `jnp.clip` serves both callers: the host passes an int, the sampler a tracer.
    return jnp.clip(jnp.int32((observed - self.random_capacity) * share), 1, self.random_capacity)
