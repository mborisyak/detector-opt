"""Full-budget, fixed-epoch confirmation trainer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ...utils.training import masked_mean_sem
from .common import Trainer, TrainResult, cosine_optimizer, fresh_design_network, window_sample_indices

__all__ = ["FullBudgetTrainer"]


class FullBudgetTrainer(Trainer):
    """Confirm a single design by full-budget, fixed-length training.

    No adaptive procedure: it samples the *entire* detector-call budget under one
    design up front, then trains a freshly initialised regressor for exactly
    ``max_epochs`` epochs (one epoch = one pass over the training budget) with a
    **single-cycle cosine-decayed** learning rate (peak -> ~0). No data growing, no
    plateau / precision gate -- the only stop is ``max_epochs``. Design-conditioned
    like every trainer (``combine`` sees the design's true geometry), via the shared
    kernels.

    This is the confirmation trainer behind ``scripts/confirm.py``: an independent
    estimate of a design's achievable loss, free of warm-starting and of the
    shared/meta network.
    """

    @classmethod
    def from_config(cls, detector, config, *, max_epochs, seed=0):
        """Build from a run-config dict; ``max_epochs`` is the confirmation knob.

        Reuses the run's ``regressor`` and ``training.optimizer`` (same architecture
        and optimiser hyper-parameters as BO), but wraps the learning rate in a
        single-cycle cosine decay spanning the whole run.
        """
        from ...utils.config import resolve_device

        training = config["training"]
        return cls(
            detector,
            regressor_config=config["regressor"],
            optimizer_config=training["optimizer"],
            batch=training["batch"],
            budget=training["budget"],
            max_epochs=max_epochs,
            val_fraction=training.get("val_fraction", 0.25),
            eval_batch=training.get("eval_batch"),
            device=resolve_device(config.get("device")),
            seed=seed,
        )

    def __init__(
        self,
        detector,
        *,
        regressor_config: dict,
        optimizer_config: dict,
        batch: int,
        budget: int,
        max_epochs: int,
        val_fraction: float = 0.25,
        eval_batch: int | None = None,
        device=None,
        seed: int = 0,
    ):
        self.max_epochs = int(max_epochs)

        # The single design fills the whole budget; one epoch is a full pass over the
        # train split (so iteration_limit = train_budget). Size the split arithmetically
        # to build the single-cycle cosine schedule (peak -> ~0 over the whole run);
        # the base ``__init__`` then sizes the pools + builds kernels.
        val_budget = round(budget * val_fraction)
        train_budget = budget - val_budget
        steps_per_epoch = max(1, train_budget // int(batch))
        optimizer = cosine_optimizer(optimizer_config, self.max_epochs * steps_per_epoch)
        super().__init__(
            detector,
            regressor_config=regressor_config,
            optimizer=optimizer,
            batch=batch,
            budget=budget,
            iteration_limit=train_budget,
            val_iteration_limit=val_budget,
            val_fraction=val_fraction,
            eval_batch=eval_batch,
            device=device,
            seed=seed,
        )

    def _prior_count(self, start, count):
        # One window, one fit, no history: the window IS the evidence.
        return count

    def _sample_indices(self, key, start, count):
        return window_sample_indices(self, key, start, count)  # uniform over the full budget window

    def _sample_weights(self):
        return None  # one window, one design: uniform

    def _init_design_network(self, init_seq, init_params):
        return fresh_design_network(self, init_seq, init_params)  # fresh, single design

    def train(self, design_scaled, seed_seq, *, on_epoch=None) -> TrainResult:
        """Train a fresh regressor on the full budget for ``max_epochs`` epochs.

        Samples the entire budget under ``design_scaled`` (one design) up front, then
        runs exactly ``max_epochs`` epochs with the cosine-decayed LR. Returns the
        final ``(train + val) / 2`` loss and its combined SEM.
        """
        detector = self.detector
        design_scaled = np.asarray(design_scaled, dtype=np.float32)
        design = detector.to_nominal(design_scaled)  # physical Design namedtuple (what the pools store)
        init_seq, training_seq = seed_seq.spawn(2)

        # Fresh, randomly initialised regressor + optimiser (cosine-scheduled LR).
        params, state, opt_state = self._init_design_network(init_seq, None)

        # Populate EVERYTHING up front: the whole budget under this single design (the disjoint
        # train/val event indices were drawn once in __init__).
        tp, vp = self.train_pool, self.val_pool
        self._fill_pool(design, tp, tp.capacity, self._train_index)
        self._fill_pool(design, vp, vp.capacity, self._val_index)
        start = jnp.int32(0)
        train_count, val_count = int(tp.current), int(vp.current)

        key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))
        train_hist, val_hist = [], []
        train_sem_hist, val_sem_hist, window_hist = [], [], []
        train_mean = val_mean = train_sem = val_sem = float("nan")

        # on_epoch (plotting) runs on a single background worker so its I/O never
        # stalls training; drained on exit by the context manager.
        plot_ctx = ThreadPoolExecutor(max_workers=1) if on_epoch is not None else nullcontext()
        with plot_ctx as plot_pool:
            for _ in range(self.max_epochs):
                key, subkey = jax.random.split(key)
                params, state, opt_state, _ = self._train_epoch(
                    params,
                    state,
                    opt_state,
                    subkey,
                    start,
                    jnp.int32(train_count),
                    tp.buffers(),
                )
                train_eval = self._eval_train(params, state, tp.buffers(), start)
                train_mean, train_sem = masked_mean_sem(train_eval, train_count)
                val_eval = self._eval_val(params, state, vp.buffers(), start)
                val_mean, val_sem = masked_mean_sem(val_eval, val_count)
                train_mean, train_sem = float(train_mean), float(train_sem)
                val_mean, val_sem = float(val_mean), float(val_sem)
                train_hist.append(train_mean)
                val_hist.append(val_mean)
                train_sem_hist.append(train_sem)
                val_sem_hist.append(val_sem)
                window_hist.append(train_count)
                if on_epoch is not None:
                    plot_pool.submit(
                        on_epoch,
                        self._snapshot(
                            train_hist,
                            val_hist,
                            train_sem_hist,
                            val_sem_hist,
                            window_hist,
                            train_count,
                            val_count,
                        ),
                    )

        objective_loss = 0.5 * (train_mean + val_mean)
        # Same definition as DesignTrainer: the spread of the two averaged losses plus the error
        # of their means, so `objective_std` means one thing across trainers.
        objective_std = float(abs(val_mean - train_mean) + np.hypot(train_sem, val_sem))
        spent = tp.current + vp.current
        return TrainResult(objective_loss, objective_std, spent, params)
