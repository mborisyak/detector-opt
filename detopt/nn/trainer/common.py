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

Every event carries its own **encoded design** in the pool, and ``combine`` is
always **design-conditioned**: each event is merged with its own encoded design
(``combine`` decodes it internally), so the network sees the true detector
geometry -- and a mixed-design batch (e.g. replay) is handled per event.
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

__all__ = ["Trainer", "TrainResult"]


class TrainResult(NamedTuple):
    objective_loss: float  # (mean_train + mean_val) / 2 at convergence
    objective_std: float  # 0.5 * sqrt(train_sem^2 + val_sem^2) -- est_sem
    spent: int  # detector calls this design added to the pools (train + val)
    params: object  # trained regressor params (for warm-starting later designs)
    state: object  # trained regressor non-param state


def _round_down(n0: int, n_increment: int, limit: int) -> int:
    """Largest ``n0 + k*n_increment`` (k >= 0) not exceeding ``limit``."""
    if limit < n0:
        return n0
    return n0 + ((limit - n0) // n_increment) * n_increment


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
        self.train_pool, self.val_pool, _, _ = self._make_pools(detector, budget, device)
        self._build_kernels(seed)

    def _make_pools(self, detector, budget, device):
        """Allocate the train + val event pools, split by ``val_fraction``.

        Returns ``(train_pool, val_pool, train_budget, val_budget)``.
        """
        budget = int(budget)
        val_budget = round(budget * self.val_fraction)
        train_budget = budget - val_budget
        measurement_shape = tuple(detector.event_shape())
        mask_shape = measurement_shape[:-1]
        target_shape = tuple(detector.target_shape())
        design_shape = tuple(detector.encoded_design_shape())  # encoded design / event
        train_pool = Pool(train_budget, measurement_shape, mask_shape, target_shape, design_shape, device)
        val_pool = Pool(val_budget, measurement_shape, mask_shape, target_shape, design_shape, device)
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
    # JIT kernels. Buffers are (X, mask, targets, designs); ``designs`` holds each
    # event's ENCODED design (``combine`` decodes it). The window is addressed by a
    # runtime ``start`` offset + ``count`` (one compiled kernel serves any window).
    # ------------------------------------------------------------------ #
    def _make_loss_fn(self, reg_def):
        detector = self.detector
        members = self.n_ensemble
        batch = self.batch

        def loss_fn(params, state, drop_key, X_b, mask_b, design_b, targets_b):
            # deterministic=False -> dropout ACTIVE; the rng is threaded in
            # explicitly (fresh per step) so it lives at the current trace level.
            reg = nnx.merge(reg_def, params, state)
            X_norm = detector.normalize(X_b)
            features = detector.combine(X_norm, design_b)  # design_b: per-event ENCODED design
            if members is None:
                pred = reg(features, mask_b, deterministic=False, rngs=nnx.Rngs(drop_key))
            else:
                # X_b holds ``members * batch`` independent draws from the window;
                # split into one minibatch per member -> (N, batch, ...). Each member
                # trains on its own batch; combine/normalize stay batch-flat.
                feats_e = features.reshape((members, batch) + features.shape[1:])
                mask_e = mask_b.reshape((members, batch) + mask_b.shape[1:])
                pred = reg(feats_e, mask_e, deterministic=False, rngs=nnx.Rngs(drop_key))  # (N, batch, T)
                pred = pred.reshape((members * batch,) + pred.shape[2:])
            loss = jnp.mean(detector.loss(pred, targets_b))
            _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
            return loss, new_state

        return loss_fn

    def _sample_indices(self, key, start, count):
        """Minibatch indices for one train step.

        Base trainer: uniform over the current design's filled window
        ``[start, start + count)`` -- it **ignores history**. ``start`` / ``count``
        are 0-d int32 ``jax.Array`` (dynamic, so the kernel never recompiles as the
        window moves or grows). Subclasses override to mix in past iterations.

        Draws ``members * batch`` indices: for an ensemble the loss reshapes these
        into ``members`` independent minibatches (all i.i.d. over the same window),
        one per member; for a single model ``members == 1`` (just ``batch``).
        """
        return start + jax.random.randint(key, (self.draw_batch,), 0, jnp.maximum(count, 1))

    def _build_train_epoch(self, reg_def):
        optimizer = self.optimizer
        steps = self.steps_per_epoch
        loss_fn = self._make_loss_fn(reg_def)
        sample_indices = self._sample_indices  # may be overridden (e.g. replay)

        def train_step(carry, key):
            params, state, opt_state, start, count, buffers = carry
            X_buf, mask_buf, targets_buf, design_buf = buffers
            key_idx, key_drop = jax.random.split(key)
            idx = sample_indices(key_idx, start, count)
            (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params,
                state,
                key_drop,
                X_buf[idx],
                mask_buf[idx],
                design_buf[idx],
                targets_buf[idx],
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
            X_buf, mask_buf, targets_buf, design_buf = buffers
            pool_size = X_buf.shape[0]
            reg = nnx.merge(reg_def, params, state)

            def body(_carry, c):
                idxs = start + c * eval_batch + jnp.arange(eval_batch, dtype=jnp.int32)
                safe = jnp.clip(idxs, 0, pool_size - 1)
                X_norm = detector.normalize(X_buf[safe])
                features = detector.combine(X_norm, design_buf[safe])  # per-event ENCODED design
                mask_b = mask_buf[safe]
                if members is None:
                    pred = reg(features, mask_b, deterministic=True)  # dropout OFF
                else:
                    # Every member sees the SAME eval batch; average their predictions.
                    feats_e = jnp.broadcast_to(features, (members,) + features.shape)
                    mask_e = jnp.broadcast_to(mask_b, (members,) + mask_b.shape)
                    pred = reg(feats_e, mask_e, deterministic=True).mean(axis=0)  # (E, T)
                return None, detector.loss(pred, targets_buf[safe])

            _, losses = jax.lax.scan(body, None, jnp.arange(n_chunks))
            return losses.reshape(-1)[:window]  # per-event losses over the window

        return eval_pass

    # ------------------------------------------------------------------ #
    # Event sampling -- the only place the detector is called.
    # ------------------------------------------------------------------ #
    def _fill_pool(self, design_phys, design_enc, pool, split, n_to_add, seed_seq):
        """Generate ``n_to_add`` events at ``design_phys`` and append them.

        The detector is called with the *physical* design; the stored per-event
        design is the *encoded* one (``combine`` consumes encoded designs).
        """
        design_phys = np.asarray(design_phys, dtype=np.float32)
        design_enc = np.asarray(design_enc, dtype=np.float32)
        added = 0
        chunk = min(256, n_to_add)
        while added < n_to_add:
            k = min(chunk, n_to_add - added)
            phys_b = np.broadcast_to(design_phys[None, :], (k, design_phys.shape[0]))
            enc_b = np.broadcast_to(design_enc[None, :], (k, design_enc.shape[0]))
            _gt, X, mask, targets = self.detector(seed_seq.spawn(1)[0], phys_b, split=split)
            pool.append(X, mask, targets, enc_b)
            added += k

    def _sample_round(self, design_phys, design_enc, w0_train, w0_val, n_requested, seed_seq):
        """Append one round of train+val events into the shared budget pools.

        Returns the number of *train* events added (> 0), ``0`` if this design's
        window is full (it needs more than ``iteration_limit`` -> caller crashes),
        or ``None`` if the shared budget pool is full (run is over).
        """
        tp, vp = self.train_pool, self.val_pool
        # Window cap (a partial final add to land exactly on iteration_limit is OK).
        n_train = min(int(n_requested), self.iteration_limit - (tp.n_current - w0_train))
        if n_train <= 0:
            return 0  # window full: design needs > iteration_limit -> caller crashes
        n_val = round(n_train * self._val_ratio)
        n_val = max(0, min(n_val, self.val_iteration_limit - (vp.n_current - w0_val)))
        # Budget: the whole round must fit the pools, else the run is over.
        if n_train > tp.n_max - tp.n_current or n_val > vp.n_max - vp.n_current:
            return None
        train_seq, val_seq = seed_seq.spawn(2)
        self._fill_pool(design_phys, design_enc, tp, "train", n_train, train_seq)
        if n_val > 0:
            self._fill_pool(design_phys, design_enc, vp, "val", n_val, val_seq)
        return n_train

    # ------------------------------------------------------------------ #
    # Network lifecycle (overridden by ContinualTrainer to persist the net).
    # ------------------------------------------------------------------ #
    def _init_design_network(self, init_seq, init_params, init_state):
        """A fresh network per design (optionally warm-started); optimiser reset."""
        _, params, state = self._build_regressor(int(init_seq.generate_state(1)[0]))
        if init_params is not None:
            params, state = init_params, init_state
        opt_state = self.optimizer.init(params)
        d = self.device
        return (
            jax.device_put(params, d),
            jax.device_put(state, d),
            jax.device_put(opt_state, d),
        )

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
