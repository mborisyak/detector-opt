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

Every event carries its own **raw physical design** in the pool, and ``combine`` is
always **design-conditioned**: each event is merged with its own design
(``combine`` encodes it, then ``combine_encoded`` decodes + gathers per-hit), so the
network sees the true detector geometry -- and a mixed-design batch (e.g. replay) is
handled per event. Pools store RAW records (events/targets/design); ``combine`` +
``normalize_target`` run per batch inside the kernels, not at fill time.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from ...utils.pools import Pool
from ...utils.events import shuffled_event_index

__all__ = ["Trainer", "TrainResult"]


class TrainResult(NamedTuple):
    objective_loss: float  # (mean_train + mean_val) / 2 at convergence
    objective_std: float  # 0.5 * sqrt(train_sem^2 + val_sem^2) -- est_sem
    spent: int  # detector calls this design added to the pools (train + val)
    params: object  # trained regressor params (for warm-starting later designs)


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
    return start + jax.random.randint(key, (trainer.draw_batch,), 0, jnp.maximum(count, 1))


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


class Trainer:
    """Shared base: budget pools, JIT kernels, sampling, network lifecycle, checkpoints.

    A concrete trainer computes its ``optimizer`` and the per-call window caps
    (``iteration_limit`` / ``val_iteration_limit``) from its own knobs and passes
    them to ``__init__``, which sizes the pools and builds the kernels.
    """

    def __init__(
        self,
        detector,
        *,
        regressor_config: dict,
        optimizer: optax.GradientTransformation,
        batch: int,
        budget: int,
        iteration_limit: int,
        val_iteration_limit: int,
        val_fraction: float = 0.25,
        eval_batch: int | None = None,
        device=None,
        checkpoint_dir: str | None = None,
        seed: int = 0,
    ):
        """Store shared state, size the budget pools, and build the JIT kernels.

        ``iteration_limit`` is the per-call train window (also the epoch length:
        ``iteration_limit // batch`` steps); ``val_iteration_limit`` the val window.
        The concrete trainer derives both, and ``optimizer``, from its own knobs.
        """
        self.detector = detector
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
        budget_index = shuffled_event_index(detector.size(), train_budget + val_budget, self.seed)
        self._train_index = budget_index[:train_budget]
        self._val_index = budget_index[train_budget:]
        self._build_kernels(seed)

    def _make_pools(self, detector, budget, device):
        """Allocate the train + val event pools, split by ``val_fraction``.

        Returns ``(train_pool, val_pool, train_budget, val_budget)``.
        """
        budget = int(budget)
        val_budget = round(budget * self.val_fraction)
        train_budget = budget - val_budget
        # Pool slots (positional, pytree): raw Event, mask, raw Target, raw physical Design / event.
        mask_dim = int(jax.tree.leaves(detector.event_spec())[0].shape[0])  # M (per-hit count)
        mask_spec = jax.ShapeDtypeStruct((mask_dim,), jnp.int32)
        specs = (detector.event_spec(), mask_spec, detector.target_spec(), detector.design_spec())
        train_pool = Pool(train_budget, specs, device)
        val_pool = Pool(val_budget, specs, device)
        return train_pool, val_pool, train_budget, val_budget

    # ------------------------------------------------------------------ #
    # Regressor + kernel construction
    # ------------------------------------------------------------------ #
    def _build_regressor(self, seed):
        """Build a fresh regressor and split it into ``(graphdef, params, state)``."""
        from detopt.nn import from_config

        reg = from_config(self.detector, config=self.regressor_config, rngs=nnx.Rngs(seed))
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

        reg = from_config(self.detector, config=self.regressor_config, rngs=nnx.Rngs(seed))
        self.n_ensemble = reg.ensemble()
        self.draw_batch = self.batch * (self.n_ensemble or 1)  # indices drawn per train step
        reg_def = nnx.split(reg, nnx.Param, nnx.Variable)[0]
        self._train_epoch = self._build_train_epoch(reg_def)
        self._eval_train = self._build_eval(reg_def, self.iteration_limit)
        self._eval_val = self._build_eval(reg_def, self.val_iteration_limit)

    # ------------------------------------------------------------------ #
    # JIT kernels. Buffers are (event, mask, target, design); ``design`` holds each
    # event's RAW PHYSICAL design (``combine`` encodes it, then ``combine_encoded``
    # decodes + gathers). The window is addressed by a runtime ``start`` offset +
    # ``count`` (one compiled kernel serves any window). ``combine`` + ``normalize_target``
    # run here, per batch, on the raw records (not at pool-fill time).
    # ------------------------------------------------------------------ #
    def _make_loss_fn(self, reg_def):
        detector = self.detector
        members = self.n_ensemble
        batch = self.batch

        def loss_fn(params, state, drop_key, event_b, mask_b, design_b, target_b):
            # deterministic=False -> dropout ACTIVE; the rng is threaded in
            # explicitly (fresh per step) so it lives at the current trace level.
            reg = nnx.merge(reg_def, params, state)
            features = detector.combine(event_b, design_b, mask=mask_b)  # design_b: per-event PHYSICAL design
            emask = detector.element_mask(event_b, mask_b)  # per-element mask (== hit mask, unless layer-wise)
            target = detector.normalize_target(target_b)
            # The MODEL owns the forward (reg.loss) so it can inject net-specific loss terms.
            if members is None:
                per = reg.loss(detector.loss, features, emask, target, deterministic=False, rngs=nnx.Rngs(drop_key))
            else:
                # event_b holds ``members * batch`` independent draws from the window;
                # split into one minibatch per member -> (N, batch, ...). Each member
                # trains on its own batch; combine stays batch-flat.
                feats_e = features.reshape((members, batch) + features.shape[1:])
                mask_e = emask.reshape((members, batch) + emask.shape[1:])
                target_e = target.reshape((members, batch) + target.shape[1:])
                per = reg.loss(detector.loss, feats_e, mask_e, target_e, deterministic=False, rngs=nnx.Rngs(drop_key))
            loss = jnp.mean(per)
            _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
            return loss, new_state

        return loss_fn

    def _sample_indices(self, key, start, count):
        """Minibatch indices for one train step (``members * batch``). ABSTRACT -- the strategy implements
        it: the per-design trainers draw uniformly over the current window (:func:`window_sample_indices`),
        the continual trainer mixes in replay from past iterations."""
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
                features = detector.combine(ev, jax.tree.map(lambda a: a[safe], design_buf), mask=mask_b)
                emask = detector.element_mask(ev, mask_b)
                target_b = detector.normalize_target(jax.tree.map(lambda a: a[safe], target_buf))
                if members is None:
                    # MODEL owns the forward (net-specific loss terms apply, e.g. deep supervision).
                    per = reg.loss(detector.loss, features, emask, target_b, deterministic=True)
                else:
                    # Every member sees the SAME eval batch; average member PREDICTIONS, then loss.
                    feats_e = jnp.broadcast_to(features, (members,) + features.shape)
                    mask_e = jnp.broadcast_to(emask, (members,) + emask.shape)
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
            design_b = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a)[None], (k,) + jnp.asarray(a).shape), design)
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
        return n_train

    # ------------------------------------------------------------------ #
    # Network lifecycle (overridden by ContinualTrainer to persist the net).
    # ------------------------------------------------------------------ #
    def _init_design_network(self, init_seq, init_params):
        """The network for a design: ``(params, state, opt_state)``. ABSTRACT -- the strategy implements it:
        a FRESH per-design net (:func:`fresh_design_network`) or the persistent continual net."""
        raise NotImplementedError()

    def _persist_network(self, params, state, opt_state):
        """Base trainer keeps nothing -- each design is independent."""

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
            manager,
            epoch,
            config={"regressor": self.regressor_config},
            parameters=params,
            state=state,
            design=design_tree,
            aux={"train_loss": train, "val_loss": val},
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
