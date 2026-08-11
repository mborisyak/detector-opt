"""Per-design trainer with a data-growing convergence procedure.

:class:`DesignTrainer` turns a *design* into a converged objective-loss estimate:
a fresh (or warm-started) network is trained on the design's window, growing the
window until the loss estimate is precise enough.

Convergence procedure
---------------------
With ``err = sqrt(train_sem^2 + val_sem^2)``, after the mandatory warmup, at the
end of each epoch:

  1. ``(val - train) - err > loss_precision`` (gap confidently over precision)
     -> **add data** now, even before convergence;
  2. else not converged (train and val loss not both plateaued over ``patience``
     epochs) -> train;
  3. else ``|val - train| + err >= loss_precision`` -> **add data**;
  4. else -> **return** ``((train+val)/2, 0.5*err)``.

``iteration_limit`` is the per-design upper limit; exceeding it is a hard error.
The shared budget bounds the whole run: when the pool can't fit the next sample,
:meth:`train` returns ``None`` and the BO loop stops.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ...utils.training import is_plateaued, masked_mean_sem
from .common import Trainer, TrainResult, _round_down, window_sample_indices, fresh_design_network

__all__ = ["DesignTrainer"]


class _DesignBase(Trainer):
    """The per-design training LOOP (window-growing convergence) + factory, shared by the per-design
    strategy (:class:`DesignTrainer`) and the continual strategy (:class:`ContinualTrainer`). The network
    LIFECYCLE (``_sample_indices`` / ``_init_design_network``) is ABSTRACT here -- each strategy IMPLEMENTS
    it, so neither overrides a concrete default."""

    @classmethod
    def from_config(cls, detector, config, *, checkpoint_dir=None, seed=0):
        """Build a trainer from a full run-config dict.

        ``detector`` is passed explicitly; the rest comes from ``config`` -- the
        ``training`` block (which holds the data, optimizer and convergence knobs)
        is spread onto the constructor; ``optimizer`` / ``device`` are built and
        ``regressor`` is passed through. ``checkpoint_dir`` and ``seed`` are
        run-level and supplied by the caller.
        """
        from ...utils.config import optimizer as make_optimizer, resolve_device

        training = {k: v for k, v in config["training"].items() if k != "optimizer"}
        return cls(
            detector,
            regressor_config=config["regressor"],
            optimizer=make_optimizer(config["training"]["optimizer"]),
            device=resolve_device(config.get("device")),
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            **training,
        )

    def __init__(
        self,
        detector,
        *,
        regressor_config: dict,
        optimizer: optax.GradientTransformation,
        batch: int,
        n0: int,
        n_increment: int,
        iteration_limit: int,
        warmup_epochs: int,
        patience: int,
        flatness_tol: float,
        loss_precision: float,
        budget: int,
        val_fraction: float = 0.25,
        eval_batch: int | None = None,
        device=None,
        checkpoint_dir: str | None = None,
        seed: int = 0,
    ):
        self.n0 = int(n0)
        self.n_increment = int(n_increment)
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.flatness_tol = float(flatness_tol)
        self.loss_precision = float(loss_precision)

        # Window caps from this trainer's knobs: round the per-design cap so additions
        # land exactly on it (it also fixes the epoch length); the val cap mirrors the
        # train/val ratio. The base ``__init__`` then sizes the pools + builds kernels.
        iteration_limit = _round_down(self.n0, self.n_increment, int(iteration_limit))
        val_iteration_limit = max(int(batch), round(iteration_limit * val_fraction / (1.0 - val_fraction)))
        super().__init__(
            detector,
            regressor_config=regressor_config,
            optimizer=optimizer,
            batch=batch,
            budget=budget,
            iteration_limit=iteration_limit,
            val_iteration_limit=val_iteration_limit,
            val_fraction=val_fraction,
            eval_batch=eval_batch,
            device=device,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
        )

    def train(
        self,
        design_scaled,
        seed_seq,
        *,
        init_params=None,
        on_epoch=None,
        step=0,
    ) -> TrainResult | None:
        """Train a regressor for one *scaled* design.

        Appends this design's events into the shared budget pools (a fresh window)
        and trains within that window. ``seed_seq`` is a
        :class:`numpy.random.SeedSequence`; ``init_params`` optionally warm-starts
        the params from a previously trained design (the buffer state stays fresh).

        Returns a :class:`TrainResult`, or ``None`` if the shared budget pool is
        exhausted (the design did not complete).
        """
        detector = self.detector
        design_scaled = np.asarray(design_scaled, dtype=np.float32)
        design = detector.to_nominal(design_scaled)  # physical Design namedtuple (what the pools store)
        design_phys = np.asarray(detector.flatten_design(design), dtype=np.float32)  # flat, for the checkpoint tree

        init_seq, training_seq = seed_seq.spawn(2)

        # Network for this design (base: fresh / optionally warm-started; the
        # continual trainer keeps and continues the same one across designs).
        params, state, opt_state = self._init_design_network(init_seq, init_params)

        tp, vp = self.train_pool, self.val_pool
        # This design's window starts at the current pool fill. The offsets handed
        # to the kernels are 0-d int32 jax.Array (dynamic -> no recompile).
        w0_train, w0_val = tp.current, vp.current
        w0_train_j, w0_val_j = jnp.int32(w0_train), jnp.int32(w0_val)

        manager = self._checkpoint_manager(step)
        design_tree = {"scaled": design_scaled, "physical": design_phys}

        train_loss_history, val_loss_history, pool_size_history = [], [], []
        train_sem_history, val_sem_history = [], []
        key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))

        # Initial data (n0 <= iteration_limit, so this only fails on a full budget).
        if self._sample_round(design, w0_train, w0_val, self.n0) is None:
            return None

        round_start = 0  # history index where the current (post-add) round began
        epoch_in_round = 0
        objective = None  # (loss, std) set on convergence

        # Per-epoch callback (plotting) on a single background worker so its slow
        # I/O never stalls training; drained on every exit by the context manager.
        plot_ctx = ThreadPoolExecutor(max_workers=1) if on_epoch is not None else nullcontext()
        with plot_ctx as plot_pool:
            while True:
                train_count = tp.current - w0_train  # filled rows of the window
                val_count = vp.current - w0_val
                key, subkey = jax.random.split(key)
                params, state, opt_state, _ = self._train_epoch(
                    params,
                    state,
                    opt_state,
                    subkey,
                    w0_train_j,
                    jnp.int32(train_count),
                    tp.buffers(),
                )

                # Per-epoch losses: sequential masked passes over BOTH windows.
                train_eval = self._eval_train(params, state, tp.buffers(), w0_train_j)
                train_mean, train_sem = masked_mean_sem(train_eval, train_count)
                val_eval = self._eval_val(params, state, vp.buffers(), w0_val_j)
                val_mean, val_sem = masked_mean_sem(val_eval, val_count)
                train_mean = float(train_mean)
                train_sem = float(train_sem)
                val_mean = float(val_mean)
                val_sem = float(val_sem)

                train_loss_history.append(train_mean)
                val_loss_history.append(val_mean)
                train_sem_history.append(train_sem)
                val_sem_history.append(val_sem)
                pool_size_history.append(train_count)
                epoch_in_round += 1

                if on_epoch is not None:
                    plot_pool.submit(
                        on_epoch,
                        self._snapshot(
                            train_loss_history,
                            val_loss_history,
                            train_sem_history,
                            val_sem_history,
                            pool_size_history,
                            train_count,
                            val_count,
                        ),
                    )

                gap_signed = val_mean - train_mean  # val - train (signed)
                err = float(np.hypot(train_sem, val_sem))  # combined SEM of val-train
                diff = abs(gap_signed)

                # Warmup: unconditional training after every data addition.
                if epoch_in_round <= self.warmup_epochs:
                    continue

                # Decision (see docstring): add data only for a confidently large
                # gap (1) or a converged-but-imprecise estimate (3).
                large_gap = gap_signed - err > self.loss_precision
                if not large_gap:
                    # Slope criterion: BOTH the train and val losses are flat over the
                    # patience window (val still improving -> keep training).
                    train_flat = is_plateaued(
                        train_loss_history[round_start:], self.patience, self.flatness_tol, self.loss_precision
                    )
                    val_flat = is_plateaued(
                        val_loss_history[round_start:], self.patience, self.flatness_tol, self.loss_precision
                    )
                    if not (train_flat and val_flat):
                        continue  # (2) train further
                    if diff + err < self.loss_precision:
                        # The objective is (train + val) / 2 and its uncertainty is `diff + err`:
                        # the SPREAD of the two numbers averaged, plus the error of their means. It
                        # is the very quantity the line above just bounded below `loss_precision`,
                        # so what the GP is told about an observation is exactly what made the
                        # design stop -- and never smaller than the tolerance that allowed it.
                        # Passing `err` alone (the old value) claims a precision the estimate does
                        # not have: it ignores the generalisation gap, which is the larger term.
                        objective = (0.5 * (train_mean + val_mean), diff + err)  # (4)
                        print(
                            f"  [converged] train={train_mean:.4f} val={val_mean:.4f} "
                            f"diff={diff:.4f} err={err:.4f} diff+err={diff + err:.4f} "
                            f"prec={self.loss_precision:.4f} | "
                            f"window={train_count} pool={tp.current}/{tp.capacity}"
                        )
                        break
                    # (3) converged but diff+err >= precision -> fall through to grow.

                # (1) large gap, or (3) converged-but-imprecise -> add data.
                n_added = self._sample_round(
                    design,
                    w0_train,
                    w0_val,
                    self.n_increment,
                )
                if n_added is None:
                    return None  # shared budget pool exhausted mid-design
                if n_added == 0:
                    raise RuntimeError(
                        f"design did not reach precision within iteration_limit "
                        f"{self.iteration_limit} (window={train_count}); "
                        f"diff={diff:.4g} err={err:.4g} diff+err={diff + err:.4g} vs "
                        f"precision={self.loss_precision:.4g} (large_gap={large_gap}). "
                        f"A healthy network must reach precision within iteration_limit "
                        f"-- raise it or fix model/regularisation."
                    )
                round_start = len(train_loss_history)
                epoch_in_round = 0
                print(f"  [grow] window -> {tp.current - w0_train}, pool {tp.current}/{tp.capacity}")
                continue

            # ONE checkpoint per design, written at convergence. Saving every epoch wrote 27 files
            # an epoch to retain only the last three (96% of them deleted again), and orbax's
            # async save blocks the NEXT one until its background write lands -- free only while
            # the epoch outlasts the write (~200 ms locally, more on network storage). Nothing
            # read the earlier epochs anyway: both consumers take ``latest_step()``, i.e. exactly
            # this final state (verify_trajectory._restore_design_network, continue_reported).
            self._save_checkpoint(
                manager,
                len(train_loss_history),
                params,
                state,
                design_tree,
                train_mean,
                val_mean,
            )
            if manager is not None:
                manager.wait_until_finished()
            self._persist_network(params, state, opt_state)  # continual: keep it
            objective_loss, objective_std = objective
            spent = (tp.current - w0_train) + (vp.current - w0_val)
            return TrainResult(objective_loss, objective_std, spent, params)


class DesignTrainer(_DesignBase):
    """The standard per-design strategy: a FRESH network per design, minibatches drawn uniformly over the
    current window (history-ignoring)."""

    def _sample_indices(self, key, start, count):
        return window_sample_indices(self, key, start, count)

    def _init_design_network(self, init_seq, init_params):
        return fresh_design_network(self, init_seq, init_params)
