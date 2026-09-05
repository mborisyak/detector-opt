"""Shared machinery for the design trainers (:mod:`detopt.nn.trainer`).

:class:`Trainer` holds everything the concrete trainers share: the budget event
pools, the JIT train/eval kernels, event sampling, the per-design network
lifecycle, and checkpointing. Each concrete trainer adds only its ``__init__``
(which knobs it takes) and its ``train`` loop:

* :class:`~detopt.nn.trainer.design.DesignTrainer` -- per-design, data-growing
  convergence;
* :class:`~detopt.nn.trainer.continual.ContinualTrainer` -- one persistent network
  across designs, with experience replay;
* :class:`~detopt.nn.trainer.full_budget.FullBudgetTrainer` -- one design, full
  budget up front, fixed number of epochs.

Pool / window model
-------------------
Each trainer owns ONE pair of event pools (train + val) sized to the whole
detector-call budget, allocated once and *accumulated* across calls. A ``train``
call occupies a contiguous **window** ``[w0, w0 + n)`` (``w0`` = the pool fill when
it started); training and evaluation address it by a runtime ``start`` offset, so a
single compiled kernel serves any window position/fill without recompiling.

Every event carries its own **raw physical design** in the pool, and ``combine``
always receives that TRUE design: each event is merged with its own (``combine``
scales it, then ``combine_scaled`` un-scales + gathers per-hit), so a mixed-design
batch (e.g. replay) is handled per event. Pools store RAW records
(events/targets/design); ``combine`` + ``normalize_target`` run per batch inside the
kernels, not at fill time.

Whether the NETWORK is told the design is a separate question, answered per
strategy by the abstract :meth:`Trainer.reveals_design` and threaded into
``combine`` by :meth:`Trainer._combine`. Withholding narrows the features; it never
changes what the detector measured.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from ...utils.pools import Pool
from ...utils.events import shuffled_event_index

__all__ = ["Trainer", "TrainResult", "regressor_rngs"]


class TrainResult(NamedTuple):
  objective_loss: float  # (mean_train + mean_val) / 2 at convergence
  objective_std: float  # |val - train| + hypot(train_sem, val_sem) -- spread + error of the means
  spent: int  # detector calls this design added to the pools (train + val)
  params: object  # trained regressor params (for warm-starting later designs)
  spent_train: int = 0  # of `spent`, how many landed in the TRAIN pool; 0 = not reported by this trainer
  spent_val: int = 0  # of `spent`, how many landed in the VALIDATION pool; 0 = not reported by this trainer


def _round_down(n0: int, n_increment: int, limit: int) -> int:
  """Largest ``n0 + k*n_increment`` (k >= 0) not exceeding ``limit``."""
  if limit < n0:
    return n0
  return n0 + ((limit - n0) // n_increment) * n_increment


# ---------------------------------------------------------------------------- #
# The PER-DESIGN lifecycle (shared by DesignTrainer + FullBudgetTrainer), as module functions so each
# strategy IMPLEMENTS the abstract Trainer hooks by delegating here -- rather than overriding a concrete
# default that the continual strategy would then have to re-override.
# ---------------------------------------------------------------------------- #
def window_sample_indices(trainer, key, start, count):
  """Uniform minibatch indices over the current design's filled window ``[start, start+count)`` --
    HISTORY-IGNORING. ``start`` / ``count`` are 0-d int32 ``jax.Array`` (dynamic, so the kernel never
    recompiles as the window moves/grows). Draws ``members * batch`` (``trainer.draw_batch``): for an
    ensemble the loss reshapes these into ``members`` i.i.d. minibatches over the same window."""
  return start + jax.random.randint(key, (trainer.draw_batch, ), 0, jnp.maximum(count, 1))


def regressor_rngs(seed):
  """The rng streams a regressor is BUILT with -- the single definition, because a network restored
  under a different set of streams does not match its own checkpoint.

  THREE NAMED STREAMS, not one. With a single ``default`` stream, constructing an ``nnx.Dropout``
  forks it -- ``rngs['dropout']`` falls back to ``default`` -- which advances the counter every kernel
  is drawn from, so merely ENABLING dropout reshuffles every parameter after it. MEASURED: checksum
  2201.172396 against 2218.083384, initialisation noise of 0.003-0.009 landed on top of whatever the
  regulariser does. Naming the streams makes each consumer draw from its own, so an arm with dropout
  and an arm without start from the identical network. Parameter draws are unchanged from the
  single-stream form when dropout is off, so no existing seed is renumbered.

  Anything that RESTORES a trained network must build its target with this too: the saved state
  carries one counter per stream, and `nnx.replace_by_pure_dict` raises on a key the target lacks.
  """
  return nnx.Rngs(params=seed, dropout=seed, dropconnect=seed)


def cosine_optimizer(optimizer_config, total_steps):
  """The run's optimiser with its learning rate wrapped in a single-cycle cosine decay (peak ->
    ~0) over ``total_steps``. Shared by the fixed-epoch trainers (full-budget confirmation and
    verification), which train for a known number of steps and so can anneal."""
  from ...utils.config import split

  name, arguments = split(optimizer_config)
  arguments = dict(arguments)
  peak = arguments.pop("learning_rate")
  schedule = optax.cosine_decay_schedule(init_value=peak, decay_steps=int(total_steps))
  return getattr(optax, name)(learning_rate=schedule, **arguments)


def design_init_sequence(run_seed, step):
  """Network-initialisation ``SeedSequence`` for design ``step``, rooted at the RUN-level seed.

  ONE sequence per run, indexed by the BO iteration. The continual trainer and the warm-started arms
  take element 0 and carry the network onward from there; the per-design trainer takes element
  ``step``, so it still draws a fresh network every design. The point is that at step 0 EVERY arm
  starts from the identical weights, which makes the first design a matched comparison instead of a
  difference of initialisation -- previously the continual arm drew from the run seed while the
  per-design arm drew from a sequence rooted at that design's own seed, and the two never agreed.

  ``SeedSequence.spawn`` is deterministic on a fresh root, so ``spawn(step + 1)[step]`` is the
  ``step``-th child however it is reached. A SEQUENCE is returned rather than an integer so that every
  caller derives the network the same way -- ``generate_state`` on this object -- and so the rewind
  path can still ``spawn`` further children from it."""
  return np.random.SeedSequence(int(run_seed)).spawn(step + 1)[step]


def fresh_design_network(trainer, init_seq, init_params):
  """A FRESH network per design (optionally warm-started), optimiser reset -- ``(params, state,
    opt_state)`` on the trainer's device. Warm-start carries PARAMS only; the non-param buffer
    state (rng counters) is always freshly built -- it is never read during train (the dropout key
    is threaded fresh) or eval (deterministic), so carrying it forward would be meaningless."""
  _, params, state = trainer._build_regressor(int(init_seq.generate_state(1)[0]))
  if init_params is not None:
    params = init_params
  opt_state = trainer.optimizer.init(params)
  d = trainer.device
  return (jax.device_put(params, d), jax.device_put(state, d), jax.device_put(opt_state, d))


REVEAL = ('none', 'design', 'zeros', 'append')


class Trainer:
  """Shared base: budget pools, JIT kernels, sampling, network lifecycle, checkpoints.

    A concrete trainer computes its ``optimizer`` and the per-call window caps
    (``iteration_limit`` / ``val_iteration_limit``) from its own knobs and passes
    them to ``__init__``, which sizes the pools and builds the kernels.
    """

  def __init__(
    self, detector, *, regressor_config: dict, optimizer: optax.GradientTransformation, batch: int, budget: int,
    iteration_limit: int, val_iteration_limit: int, val_fraction: float = 0.25, eval_batch: int | None = None, device=None,
    checkpoint_dir: str | None = None, seed: int = 0, reveal: str | None = None,
  ):
    """Store shared state, size the budget pools, and build the JIT kernels.

        ``iteration_limit`` is the per-call train window (also the epoch length:
        ``iteration_limit // batch`` steps); ``val_iteration_limit`` the val window.
        The concrete trainer derives both, and ``optimizer``, from its own knobs.
        """
    self.detector = detector
    if reveal is not None and reveal not in REVEAL:
      raise ValueError(f'reveal must be one of {sorted(REVEAL)} or None for the strategy default, got {reveal!r}')
    self._reveal = reveal
    if self.default_reveal() == 'design' and self.reveal() != 'design':
      # A strategy whose default is 'design' needs it: its batch MIXES designs, and the design is the
      # only thing telling a replay row from a current one. Withheld or zeroed, it is fitting a
      # mixture it cannot separate -- legitimate as an experiment, never as a default.
      warnings.warn(
        f'{type(self).__name__} defaults to reveal=\'design\' because its batch mixes the current '
        f'design with replay from earlier ones; running it at reveal={self.reveal()!r} leaves the '
        f'network unable to tell those rows apart. This is a deliberate experiment or a mistake -- '
        f'it is not a neutral setting.', RuntimeWarning, stacklevel=2
      )
    self.regressor_config = regressor_config
    self.optimizer = optimizer
    self.batch = int(batch)
    self.val_fraction = float(val_fraction)
    # Evaluation needs no gradients, so a larger batch is cheaper.
    self.eval_batch = int(eval_batch) if eval_batch else 8 * self.batch
    self.device = device
    self.checkpoint_dir = checkpoint_dir  # only trainers that checkpoint set this
    self.seed = int(seed)
    self._val_ratio = self.val_fraction / (1.0 - self.val_fraction)
    # NO ADAPTIVE PRIOR HERE, deliberately (user, 2026-08-14). A MAP term `prior_scale *
    # regularization() / N` was tried and REMOVED: its weight moves with the window, so it is
    # strongest exactly when the window is smallest, and the convergence procedure exits on
    # "training loss settled AND gap small" -- both of which a pulled-down network satisfies
    # cheaply. MEASURED: it converged `sobol-best`/seed 1 at a window of 28672 reporting 0.6279
    # where the same cell at the full window reaches 0.6039. That is a bias in the objective BO
    # consumes, not a change to the precision it is quoted at. Regularisation is AdamW's weight
    # decay alone, set in the run config.

    # Per-call window caps -- the shared kernels and ``_sample_round`` use these.
    self.iteration_limit = int(iteration_limit)
    self.val_iteration_limit = int(val_iteration_limit)
    self.steps_per_epoch = max(1, self.iteration_limit // self.batch)

    # ONE pool pair sized to the whole budget; the architecture is fixed, so the
    # train + train/val eval kernels are built once.
    self.train_pool, self.val_pool, train_budget, val_budget = self._make_pools(detector, budget, device)
    # Shuffled, DISJOINT train/val event indices over the budget -- the script-side ownership of the
    # split (the detector is a deterministic function of (design, event_index), no internal pools).
    # Built ONCE; designs accumulate into the pools, consuming these indices in fill order (oversampled
    # by wrapping when budget > detector.size()).
    # GENERATIONS, not one cut. Each entry is a (train, val) block of the seeded event stream, taken
    # in order; with a single entry this is exactly `stream[:train]` and `stream[train:train+val]`.
    # The list exists so a run RESUMED UNDER A RAISED BUDGET keeps every position it has already
    # consumed and takes the increment from fresh stream positions instead. Cutting the stream at
    # `train_budget` alone cannot do that: the cut MOVES when the budget changes, so the resumed
    # training pool would refill from exactly the positions the validation pool already holds -- at
    # 2097152 -> 3145728 that is all 524288 stored validation events reappearing as training data.
    self._generations = [(int(train_budget), int(val_budget))]
    self._rebuild_event_index()
    self._build_kernels(seed)

  def _rebuild_event_index(self):
    """Build ``_train_index`` / ``_val_index`` from ``_generations`` over the run's seeded stream.

        The stream is a function of (detector size, seed) alone, so a longer draw shares its prefix
        with a shorter one and every already-consumed position keeps the event it had.
        """
    total = sum(train + val for train, val in self._generations)
    stream = shuffled_event_index(self.detector.size(), total, self.seed)
    train_parts, val_parts, at = [], [], 0
    for train, val in self._generations:
      train_parts.append(stream[at:at + train])
      at += train
      val_parts.append(stream[at:at + val])
      at += val
    self._train_index = np.concatenate(train_parts) if len(train_parts) > 1 else train_parts[0]
    self._val_index = np.concatenate(val_parts) if len(val_parts) > 1 else val_parts[0]

  def _make_pools(self, detector, budget, device):
    """Allocate the train + val event pools, split by ``val_fraction``.

        Returns ``(train_pool, val_pool, train_budget, val_budget)``.
        """
    budget = int(budget)
    val_budget = round(budget * self.val_fraction)
    train_budget = budget - val_budget
    # Pool slots (positional, pytree): raw Event, mask, raw Target, raw physical Design / event.
    mask_dim = int(jax.tree.leaves(detector.event_spec())[0].shape[0])  # M (per-hit count)
    mask_spec = jax.ShapeDtypeStruct((mask_dim, ), jnp.int32)
    specs = (detector.event_spec(), mask_spec, detector.target_spec(), detector.design_spec())
    train_pool = Pool(train_budget, specs, device)
    val_pool = Pool(val_budget, specs, device)
    return train_pool, val_pool, train_budget, val_budget

  # ------------------------------------------------------------------ #
  # Regressor + kernel construction
  # ------------------------------------------------------------------ #
  def _build_regressor(self, seed):
    """Build a fresh regressor and split it into ``(graphdef, params, state)``; the streams come from
        :func:`regressor_rngs`, which is where they are defined and why."""
    from detopt.nn import from_config

    reg = from_config(self.detector, config=self.regressor_config, rngs=regressor_rngs(seed), design=self.reveal())
    return nnx.split(reg, nnx.Param, nnx.Variable)

  def _build_kernels(self, seed):
    """Build the train-epoch + train/val eval kernels once (architecture fixed).

        Requires ``optimizer`` / ``steps_per_epoch`` / ``iteration_limit`` /
        ``val_iteration_limit`` to be set first. Also queries the regressor's
        ``ensemble()`` size: an ``n``-member ensemble trains on ``n`` independent
        minibatches per step (``n * batch`` indices drawn from the same window) and
        averages member predictions at evaluation; a single model (``None``) takes
        the plain one-batch path.
        """
    from detopt.nn import from_config

    reg = from_config(self.detector, config=self.regressor_config, rngs=regressor_rngs(seed), design=self.reveal())
    self.n_ensemble = reg.ensemble()
    self.draw_batch = self.batch * (self.n_ensemble or 1)  # indices drawn per train step
    reg_def = nnx.split(reg, nnx.Param, nnx.Variable)[0]
    self._train_epoch = self._build_train_epoch(reg_def)
    self._eval_train = self._build_eval(reg_def, self.iteration_limit)
    self._eval_val = self._build_eval(reg_def, self.val_iteration_limit)

  # ------------------------------------------------------------------ #
  # JIT kernels. Buffers are (event, mask, target, design); ``design`` holds each
  # event's RAW PHYSICAL design (``combine`` scales it, then ``combine_scaled``
  # decodes + gathers). The window is addressed by a runtime ``start`` offset +
  # ``count`` (one compiled kernel serves any window). ``combine`` + ``normalize_target``
  # run here, per batch, on the raw records (not at pool-fill time).
  # ------------------------------------------------------------------ #
  def default_reveal(self):
    """What this strategy shows the network ABSENT an explicit setting: ``'design'`` or ``'none'``.
    ABSTRACT, and the TRAINER'S call.

    ⚠️ CONSTANT IS NOT THE SAME AS FREE. A per-design strategy sees one design across its whole batch,
    but where the design fixes the measurement geometry a network denied it has to INFER that geometry
    from the readings, which costs data. The default is what a run gets when it says nothing, not a
    claim that withholding is free."""
    raise NotImplementedError()

  def reveal(self):
    """What this RUN shows the network -- ``training.reveal`` when set, else :meth:`default_reveal`.
    FINAL; a strategy varies the default, not this.

    ``'design'`` -- the design reaches the network, resolved into the features as the detector sees fit.

    ``'none'`` -- WITHHELD. The features are NARROWER: the detector emits what it can say without the
    design, and the regressor is built for that width. The measurement is unchanged -- a task whose
    measurement depends on its design still applies it.

    ``'zeros'`` -- revealed in SHAPE but not in CONTENT: full feature width, full regressor input, and
    ``zeros_like(design)`` handed to ``combine``. The CAPACITY-MATCHED control; ``'none'`` is the
    narrow-input one.

    ``'append'`` -- the design reaches the network as RAW NUMBERS rather than resolved into the
    features: ``'none'``'s layout with the scaled design concatenated onto every element. It splits
    what ``'design'`` conflates -- the design's INFORMATION from the detector's OWN way of folding it
    into per-element geometry. Against ``'design'`` it isolates the value of that resolution; against
    ``'none'`` / ``'zeros'`` it isolates the value of the information.

    ⛔️ ZEROS ARE A POINT, NOT AN ABSENCE. They are the NOMINAL zero, which for most detectors lies
    outside the design box, and where the measurement depends on the design the detector applies it --
    the visible window is degenerate at zero extent. Read a ``'zeros'`` arm as "conditioned on one
    fixed, possibly unphysical design", never as "unconditioned"."""
    return self.default_reveal() if self._reveal is None else self._reveal

  def _combine(self, event, design, mask):
    """``detector.combine`` as the TRAINING PROCEDURE performs it, under :meth:`reveal`."""
    reveal = self.reveal()
    if reveal == 'zeros':
      design = jax.tree.map(jnp.zeros_like, design)
    if reveal == 'append':
      # The design-free features with the SCALED design concatenated onto every element. Generic by
      # construction: it acts on `combine`'s OUTPUT, so the same rule serves an element set (M, F) and
      # an image (H, W, C) alike, and no detector implements anything for it.
      features = self.detector.combine(event, design, mask=mask, reveal_design=False)
      scaled = jnp.asarray(self.detector.to_scaled(design), features.dtype)
      return jnp.concatenate([features, jnp.broadcast_to(scaled, (*features.shape[:-1], scaled.shape[-1]))], axis=-1)
    return self.detector.combine(event, design, mask=mask, reveal_design=reveal != 'none')

  def _make_loss_fn(self, reg_def):
    detector = self.detector
    combine = self._combine
    members = self.n_ensemble
    batch = self.batch
    weights = self._sample_weights()  # None = uniform; built once, static per trainer

    def loss_fn(params, state, drop_key, event_b, mask_b, design_b, target_b):
      # deterministic=False -> dropout ACTIVE; the rng is threaded in
      # explicitly (fresh per step) so it lives at the current trace level.
      reg = nnx.merge(reg_def, params, state)
      features = combine(event_b, design_b, mask_b)
      emask = detector.element_mask(event_b, mask_b)  # per-element mask (== hit mask, unless layer-wise)
      target = detector.normalize_target(target_b)
      # The MODEL owns the forward (reg.loss) so it can inject net-specific loss terms.
      if members is None:
        per = reg.loss(
          detector.loss, features, emask, target, deterministic=False,
          rngs=nnx.Rngs(dropout=jax.random.fold_in(drop_key, 0), dropconnect=jax.random.fold_in(drop_key, 1))
        )
      else:
        # event_b holds ``members * batch`` independent draws from the window;
        # split into one minibatch per member -> (N, batch, ...). Each member
        # trains on its own batch; combine stays batch-flat.
        feats_e = features.reshape((members, batch) + features.shape[1:])
        mask_e = emask.reshape((members, batch) + emask.shape[1:])
        target_e = target.reshape((members, batch) + target.shape[1:])
        per = reg.loss(
          detector.loss, feats_e, mask_e, target_e, deterministic=False,
          rngs=nnx.Rngs(dropout=jax.random.fold_in(drop_key, 0), dropconnect=jax.random.fold_in(drop_key, 1))
        )
      if weights is None:
        loss = jnp.mean(per)
      else:
        # WEIGHTED mean over the minibatch. The strategy that builds the indices also says
        # what each row is worth -- the continual trainer's batch is half current design and
        # half replay, and those two halves do not deserve equal say in a gradient whose
        # result is judged ONLY on the current design's window (`_eval_train` evaluates from
        # `w0_train` forward, so replay rows contribute to the update and to nothing that is
        # measured). `weights` is pre-normalised to mean 1 by the implementer, so the loss
        # keeps the same scale as the unweighted case and `loss_precision` still means what
        # it meant.
        wb = weights.reshape(weights.shape + (1, ) * (per.ndim - weights.ndim))
        loss = jnp.mean(per * wb)
      _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
      # NO PARAMETER-PRIOR TERM. The only regularisation is AdamW's weight decay, which lives in
      # the optimiser and never touches this loss -- so the number the trainer reports is the
      # unpenalised loss of the network, and the convergence procedure reads it unbiased.
      return loss, new_state

    return loss_fn

  def _sample_indices(self, key, start, count):
    """Minibatch indices for one train step (``members * batch``). ABSTRACT -- the strategy implements
        it: the per-design trainers draw uniformly over the current window (:func:`window_sample_indices`),
        the continual trainer mixes in replay from past iterations."""
    raise NotImplementedError()

  def _sample_weights(self):
    """Per-row weights for one train step, matching ``_sample_indices``' layout, or ``None`` for a
        uniform mean. ABSTRACT -- the strategy that decides WHICH rows a batch holds also decides what
        each is worth. Must be pre-normalised to mean 1 so the loss keeps its scale."""
    raise NotImplementedError()

  def _build_train_epoch(self, reg_def):
    optimizer = self.optimizer
    steps = self.steps_per_epoch
    loss_fn = self._make_loss_fn(reg_def)
    sample_indices = self._sample_indices  # may be overridden (e.g. replay)

    def train_step(carry, key):
      params, state, opt_state, start, count, buffers = carry
      event_buf, mask_buf, target_buf, design_buf = buffers
      key_idx, key_drop = jax.random.split(key)
      idx = sample_indices(key_idx, start, count)
      (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params,
        state,
        key_drop,
        jax.tree.map(lambda a: a[idx], event_buf),  # raw Event minibatch (pytree)
        mask_buf[idx],
        jax.tree.map(lambda a: a[idx], design_buf),  # raw Design minibatch (pytree)
        jax.tree.map(lambda a: a[idx], target_buf),  # raw Target minibatch (pytree)
      )
      updates, new_opt_state = optimizer.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return (new_params, new_state, new_opt_state, start, count, buffers), loss

    @jax.jit
    def train_epoch(params, state, opt_state, key, start, count, buffers):
      keys = jax.random.split(key, steps)
      carry0 = (params, state, opt_state, start, count, buffers)
      final, losses = jax.lax.scan(train_step, carry0, keys)
      params, state, opt_state, *_ = final
      return params, state, opt_state, losses

    return train_epoch

  def _build_eval(self, reg_def, window):
    """Eval kernel scanning ``window`` rows from ``start`` (gradient-free)."""
    detector = self.detector
    combine = self._combine
    eval_batch = self.eval_batch
    members = self.n_ensemble
    n_chunks = -(-window // eval_batch)  # ceil

    @jax.jit
    def eval_pass(params, state, buffers, start):
      event_buf, mask_buf, target_buf, design_buf = buffers
      pool_size = jax.tree.leaves(event_buf)[0].shape[0]
      reg = nnx.merge(reg_def, params, state)

      def body(_carry, c):
        idxs = start + c * eval_batch + jnp.arange(eval_batch, dtype=jnp.int32)
        safe = jnp.clip(idxs, 0, pool_size - 1)
        ev = jax.tree.map(lambda a: a[safe], event_buf)
        mask_b = mask_buf[safe]
        features = combine(ev, jax.tree.map(lambda a: a[safe], design_buf), mask_b)
        emask = detector.element_mask(ev, mask_b)
        target_b = detector.normalize_target(jax.tree.map(lambda a: a[safe], target_buf))
        if members is None:
          # MODEL owns the forward (net-specific loss terms apply, e.g. deep supervision).
          per = reg.loss(detector.loss, features, emask, target_b, deterministic=True)
        else:
          # Every member sees the SAME eval batch; average member PREDICTIONS, then loss.
          feats_e = jnp.broadcast_to(features, (members, ) + features.shape)
          mask_e = jnp.broadcast_to(emask, (members, ) + emask.shape)
          pred = reg(feats_e, mask_e, deterministic=True).mean(axis=0)  # (E, T)
          per = detector.loss(pred, target_b)
        return None, per

      _, losses = jax.lax.scan(body, None, jnp.arange(n_chunks))
      return losses.reshape(-1)[:window]  # per-event losses over the window

    return eval_pass

  # ------------------------------------------------------------------ #
  # Event sampling -- the only place the detector is called.
  # ------------------------------------------------------------------ #
  def _fill_pool(self, design, pool, n_to_add, index_array):
    """Generate ``n_to_add`` events at the physical ``Design`` and append them.

        The event indices are the next slice of ``index_array`` taken at the pool's CURRENT fill
        (``pool.n_current`` is the cursor), so the detector call ``detector(design, event_index)`` is
        deterministic. The detector gets the (broadcast) physical ``Design``; the stored per-event design
        is that same raw ``Design`` record (``combine`` encodes it per batch). Events/targets/design are
        stored as raw namedtuple records (the pool is pytree-aware).
        """
    added = 0
    chunk = min(256, n_to_add)
    while added < n_to_add:
      k = min(chunk, n_to_add - added)
      start = pool.current  # cursor into index_array (the pool fill advances it via append)
      ev_idx = index_array[start:start + k]
      design_b = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a)[None], (k, ) + jnp.asarray(a).shape), design)
      _gt, event, mask, target = self.detector(design_b, ev_idx)
      pool.append(event, mask, target, design_b)
      added += k

  def _sample_round(self, design, w0_train, w0_val, n_requested):
    """Append one round of train+val events into the shared budget pools (from the disjoint
        ``self._train_index`` / ``self._val_index``).

        Returns the number of *train* events added (> 0), ``0`` if this design's
        window is full (it needs more than ``iteration_limit`` -> caller crashes),
        or ``None`` if the shared budget pool is full (run is over).
        """
    tp, vp = self.train_pool, self.val_pool
    # Window cap (a partial final add to land exactly on iteration_limit is OK).
    n_train = min(int(n_requested), self.iteration_limit - (tp.current - w0_train))
    if n_train <= 0:
      return 0  # window full: design needs > iteration_limit -> caller crashes
    n_val = round(n_train * self._val_ratio)
    n_val = max(0, min(n_val, self.val_iteration_limit - (vp.current - w0_val)))
    # Budget: the whole round must fit the pools, else the run is over.
    if n_train > tp.capacity - tp.current or n_val > vp.capacity - vp.current:
      return None
    self._fill_pool(design, tp, n_train, self._train_index)
    if n_val > 0:
      self._fill_pool(design, vp, n_val, self._val_index)
    # WHAT ELSE THIS STRATEGY BUYS WITH THE SAME ROUND. A strategy whose training set is not only the
    # designs BO proposed adds its own events HERE, alongside the round, so "the trainer asked for more
    # data" and "the extra data arrived" are one event and the pools can never disagree about how much
    # of the budget has been spent.
    self._round_extra(n_train)
    return n_train

  def spent_calls(self):
    """Detector calls this trainer has actually SIMULATED. ABSTRACT -- the strategy that decides what
        its pools hold also decides how to count them.

        Not the same as the pools' fill: a strategy may RESERVE pool slots it has not paid for yet, and
        `bo.py` reports this as the run's spend, so a reserve counted here would be budget claimed
        without a detector call behind it."""
    raise NotImplementedError()

  def _round_extra(self, n_train):
    """Events this strategy appends ALONGSIDE a round of ``n_train`` observed ones. ABSTRACT -- the
        strategy that decides what its training set holds also decides what a data request costs.

        Most strategies train on the proposed designs alone and add nothing."""
    raise NotImplementedError()

  # ------------------------------------------------------------------ #
  # Network lifecycle (overridden by ContinualTrainer to persist the net).
  # ------------------------------------------------------------------ #
  def _init_design_network(self, init_seq, init_params):
    """The network for a design: ``(params, state, opt_state)``. ABSTRACT -- the strategy implements it:
        a FRESH per-design net (:func:`fresh_design_network`) or the persistent continual net."""
    raise NotImplementedError()

  def _persist_network(self, params, state, opt_state):
    """Base trainer keeps nothing -- each design is independent."""

  def restore_design_parameters(self, iteration):
    """The PARAMETERS this run trained for ``iteration``, read back from that design's checkpoint.

        THIS IS WHAT A WARM START READS, and reading it from disk rather than from a list on the
        driver is what makes the warm-start strategies resumable. The per-design checkpoint is written
        ONCE, at convergence, holding exactly the network whose loss the run reported (:meth:`train`
        saves it immediately before it returns), so warm-starting from the checkpoint is warm-starting
        from the same arrays the design ended on -- with the difference that they survive the process.
        A run resumed at a design boundary can therefore still warm-start from every design it measured
        before the interruption, instead of from the ones it happens to still hold in memory.

        The checkpoint stores FLAT leaves, so a freshly built regressor supplies the structure they are
        poured back into; every one of its own random values is overwritten, which is why the seed it
        is built with does not matter.
        """
    from ...utils import io

    if self.checkpoint_dir is None or len(self.checkpoint_dir) == 0:
      raise ValueError(
        "a warm start reads the per-design checkpoints, so a run that warm-starts must "
        "have been given a checkpoint_dir"
      )
    path = os.path.join(self.checkpoint_dir, f"design_{iteration:04d}")
    if not os.path.isdir(path):
      raise FileNotFoundError(
        f"no checkpoint at {path} -- a warm start continues the network this run "
        f"reported for design {iteration}, so that design's checkpoint must be kept"
      )
    manager = io.get_checkpointer(path)
    if manager.latest_step() is None:
      raise ValueError(f"{path} holds no saved epoch")
    _, params, state = self._build_regressor(self.seed)
    parameters, _state, _design, _aux = io.restore_training_checkpoint(manager, regressor=(params, state))
    manager.close()
    nnx.replace_by_pure_dict(params, parameters)
    return jax.device_put(params, self.device)

  # ------------------------------------------------------------------ #
  # Resume state. `scripts/bo.py` restarts an interrupted run at the design boundary: the design it
  # died on is REDONE from its start, so nothing mid-design is kept -- no epoch counter, no
  # convergence history, no optimiser moments. What must survive is everything that carries ACROSS
  # designs, because it cannot be recomputed without re-spending the detector budget.
  # ------------------------------------------------------------------ #
  def persist(self, path):
    """STAGE the pools (contents AND fill cursors) plus whatever the strategy carries between designs.

        Staged rather than published: the driver stages the optimiser and the trainer and then commits
        the SET, so a run killed mid-save can never resume with new evidence beside a stale pool
        (:func:`detopt.utils.io.stage` / :func:`~detopt.utils.io.commit`).

        The CURSORS matter as much as the rows: they decide where the next design's window opens and
        therefore which slice of the run's event index it consumes, so restoring rows without them
        would silently re-issue events that have already been paid for. ``seed`` is stored to be
        CHECKED on restore -- it fixes the train/val split and the event order, so resuming a run
        under a different one would reuse the pool against a different index.
        """
    from ...utils import io

    payload = {
      "seed": np.int64(self.seed),
      # The event-index LAYOUT, not just the pools. Without it a resumed run cannot tell which stream
      # positions its stored events came from, and a raised budget would re-issue them.
      "generations": np.asarray(self._generations, dtype=np.int64),
    }
    for name, pool in (("train", self.train_pool), ("val", self.val_pool)):
      payload[f"{name}_current"] = np.int64(pool.current)
      for i, leaf in enumerate(jax.tree.leaves(pool.buffers())):
        payload[f"{name}_leaf_{i}"] = np.asarray(leaf)
    payload.update(self._carried_state())
    io.stage(path, payload)

  def restore(self, path):
    """Load a :meth:`persist` snapshot, honouring an interrupted multi-file commit."""
    from ...utils import io

    source = io.restore_path(str(path))
    if source is None:
      raise FileNotFoundError(f"no trainer state at {path} (nor {path}.old)")
    with np.load(source) as data:
      if int(data["seed"]) != self.seed:
        raise ValueError(
          f"{source}: state was written at seed {int(data['seed'])}, this trainer is at "
          f"{self.seed}; the seed fixes the train/val split and the event order"
        )
      # THE LAYOUT COMES BACK FROM THE STATE, and a raised budget EXTENDS it rather than recutting it.
      #
      # ⚠️ A STATE WRITTEN BEFORE `generations` EXISTED MUST BE READ AS THE BLOCK IT ACTUALLY WAS,
      # which is the SAVED pools' capacity -- taken from the leading dimension of the saved arrays --
      # NOT this trainer's configured capacity. Reading it as the configured one makes `extra` zero
      # at any budget, so the single block silently takes the NEW sizes and the cut moves after all:
      # precisely the train/val overlap generations exist to prevent, on precisely the trees that
      # predate them, which are the only trees that will ever take this branch.
      if "generations" in data:
        stored = [(int(train), int(val)) for train, val in np.asarray(data["generations"]).reshape(-1, 2)]
      else:
        saved = tuple(int(np.shape(data[f"{name}_leaf_0"])[0]) for name in ("train", "val"))
        stored = [saved]
        print(
          f"[migrate] {source}: written before the event-index layout was recorded; reading it as one "
          f"block of {saved[0]}+{saved[1]} events (the saved pools' own capacity)."
        )
      spent_train = sum(train for train, _ in stored)
      spent_val = sum(val for _, val in stored)
      extra_train = self.train_pool.capacity - spent_train
      extra_val = self.val_pool.capacity - spent_val
      if extra_train < 0 or extra_val < 0:
        raise ValueError(
          f"{source}: state was written for a budget of {spent_train}+{spent_val} events and "
          f"this trainer is configured for {self.train_pool.capacity}+{self.val_pool.capacity}; "
          f"a budget may be RAISED between runs but never lowered -- the smaller pools cannot "
          f"hold events already paid for"
        )
      self._generations = stored if extra_train == 0 and extra_val == 0 else stored + [(extra_train, extra_val)]
      self._rebuild_event_index()
      # Release the allocated pool BEFORE building the restored one, and keep the leaves on the HOST
      # until `Pool.load` places them -- either way round, two pools on the device at once doubles the
      # high-water mark permanently.
      for name, attribute in (("train", "train_pool"), ("val", "val_pool")):
        pool = getattr(self, attribute)
        structure = jax.tree.structure(pool.buffers())
        leaves = [data[f"{name}_leaf_{i}"] for i in range(structure.num_leaves)]
        # THIS trainer's configured capacity, not the snapshot's -- a run resumed under a RAISED
        # budget must come back with the larger pool, or the raise is silently discarded and the run
        # ends immediately on a pool that is already full.
        specs, device, capacity = pool.specs, pool.device, pool.capacity
        for buffer in jax.tree.leaves(pool.buffers()):
          buffer.delete()
        setattr(
          self, attribute,
          Pool.load({
            "slots": jax.tree.unflatten(structure, leaves),
            "current": int(data[f"{name}_current"]),
          }, specs, capacity=capacity, device=device)
        )
      self._load_carried_state(data)

  def replay(self, rows):
    """Refill the pools by RE-SIMULATING the trajectory, instead of loading a saved snapshot.

        ``rows`` are the COMMITTED results entries in order; each carries its design in the scaled cube
        plus the ``spent_train`` / ``spent_val`` it added. ``detector(design, event_index)`` is
        deterministic and the event index is a function of (detector size, seed, generations), so
        replaying the rows reproduces the pools exactly. What it costs is the simulation, not the
        science -- which is why the run no longer has to carry a budget-sized state file.

        The pools must be EMPTY: this rebuilds them, it does not top them up.
        """
    if self.train_pool.current > 0 or self.val_pool.current > 0:
      raise ValueError(
        f"replay() rebuilds the pools and needs them empty, but they hold "
        f"{self.train_pool.current}+{self.val_pool.current} events"
      )
    for row in rows:
      design = self.detector.to_nominal(np.asarray(row["x_scaled"], dtype=np.float32))
      n_train, n_val = int(row["spent_train"]), int(row["spent_val"])
      if n_train > self.train_pool.capacity - self.train_pool.current or \
          n_val > self.val_pool.capacity - self.val_pool.current:
        raise ValueError(
          f"replaying iteration {row.get('iteration')} needs {n_train}+{n_val} events but the "
          f"pools hold only {self.train_pool.capacity - self.train_pool.current}+"
          f"{self.val_pool.capacity - self.val_pool.current} more; the budget was LOWERED"
        )
      self._fill_pool(design, self.train_pool, n_train, self._train_index)
      if n_val > 0:
        self._fill_pool(design, self.val_pool, n_val, self._val_index)
    self._replay_carried_state(rows)

  def _replay_carried_state(self, rows):
    """Recover whatever this strategy carries ACROSS designs, given the replayed trajectory. ABSTRACT.

        A per-design strategy carries nothing. A continual one carries its persistent network, which is
        already written per design by :meth:`_save_checkpoint`, so it reads it back from there."""
    raise NotImplementedError()

  def _carried_state(self):
    """Arrays this strategy carries ACROSS designs, as a ``name -> array`` payload for :meth:`persist`.
        ABSTRACT -- the strategy that decides what survives a design boundary also decides what a
        resumed run must be handed back."""
    raise NotImplementedError()

  def _load_carried_state(self, data):
    """Restore what :meth:`_carried_state` wrote, from a loaded ``npz``. ABSTRACT."""
    raise NotImplementedError()

  # ------------------------------------------------------------------ #
  # Checkpointing (orbax imported lazily to keep `import detopt` light).
  # ------------------------------------------------------------------ #
  def _checkpoint_manager(self, step):
    if not self.checkpoint_dir:
      return None
    from ...utils import io

    return io.get_checkpointer(os.path.join(self.checkpoint_dir, f"design_{step:04d}"))

  def _save_checkpoint(self, manager, epoch, params, state, design_tree, train, val):
    if manager is None:
      return
    from ...utils import io

    io.save_training_checkpoint(
      manager, epoch, config={"regressor": self.regressor_config}, parameters=params, state=state, design=design_tree, aux={
        "train_loss": train,
        "val_loss": val
      },
    )

  @staticmethod
  def _snapshot(train_hist, val_hist, train_sem_hist, val_sem_hist, window_hist, final_train, final_val):
    return {
      "train_loss_per_epoch": np.asarray(train_hist, dtype=np.float64),
      "val_loss_per_epoch": np.asarray(val_hist, dtype=np.float64),
      "train_sem_per_epoch": np.asarray(train_sem_hist, dtype=np.float64),
      "val_sem_per_epoch": np.asarray(val_sem_hist, dtype=np.float64),
      "train_budget_per_epoch": np.asarray(window_hist, dtype=np.int64),
      "final_train_budget": int(final_train),
      "final_val_pool_size": int(final_val),
    }
