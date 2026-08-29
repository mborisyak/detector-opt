"""STRICT growth. A sibling of :mod:`.design`, never a subclass: `_DesignBase.train` is concrete and
concrete methods are final here. `design.py` is not modified and not imported for its loop.

THE PROCEDURE (user, 2026-08-18):

    warmup `warmup_epochs` epochs after every data addition, no intervention.
    then each epoch:
      gap + err > loss_precision              -> add data
      `patience` epochs without improvement   -> converged, return

`gap = |val - train|`, `err = hypot(train_sem, val_sem)`, improvement is a new best TRAINING loss.
The counter resets on every data addition. The scoring gate is the same as the Bayesian procedure's,
so a difference between the two is a difference between convergence tests alone.
"""

from __future__ import annotations

import threading

import jax
import jax.numpy as jnp
import numpy as np

from ...utils.training import masked_mean_sem
from .common import Trainer, TrainResult, _round_down, fresh_design_network, window_sample_indices
from .replay import (
  carried_network_state, load_carried_network_state, persistent_network, replay_carried_network, replay_sample_indices,
  replay_sample_weights
)

__all__ = ["StrictDesignTrainer", "StrictContinualTrainer"]


class _StrictBase(Trainer):
  """The strict-growth lifecycle. Implements :meth:`train`; the sampling and network hooks stay
  abstract so each strategy supplies its own."""

  @classmethod
  def spent_calls(self):
    """Reserves nothing, so the pools' fill IS the spend."""
    return self.train_pool.current + self.val_pool.current

  def _round_extra(self, n_train):
    """Trains on the proposed designs alone, so a round costs exactly what it asked for."""

  def from_config(cls, detector, config, *, checkpoint_dir=None, seed=0):
    from ...utils.config import optimizer as make_optimizer, resolve_device

    training = {k: v for k, v in config["training"].items() if k != "optimizer"}
    return cls(
      detector, regressor_config=config["regressor"], optimizer=make_optimizer(config["training"]["optimizer"]),
      device=resolve_device(config.get("device")), checkpoint_dir=checkpoint_dir, seed=seed, **training
    )

  def __init__(
    self, detector, *, n0: int, n_increment: int, iteration_limit: int, batch: int, val_fraction: float = 0.25,
    warmup_epochs: int = 32, patience: int = 10, loss_precision: float = 1.0e-2, param_mix: float = 0.0, **kwargs
  ):
    iteration_limit = _round_down(int(n0), int(n_increment), int(iteration_limit))
    val_iteration_limit = max(int(batch), round(iteration_limit * val_fraction / (1.0 - val_fraction)))
    super().__init__(
      detector, batch=batch, iteration_limit=iteration_limit, val_iteration_limit=val_iteration_limit,
      val_fraction=val_fraction, **kwargs
    )
    self.n0 = int(n0)
    self.n_increment = int(n_increment)
    self.warmup_epochs = int(warmup_epochs)
    self.patience = int(patience)
    self.loss_precision = float(loss_precision)
    self.param_mix = float(param_mix)

  def train(self, design_scaled, seed, *, init_params=None, on_epoch=None, step=0) -> TrainResult | None:
    """Train one design. Returns a :class:`TrainResult`, or ``None`` if the shared budget pool is
    exhausted -- the driver reads that as the end of the run."""
    detector = self.detector
    design_scaled = np.asarray(design_scaled, dtype=np.float32)
    design = detector.to_nominal(design_scaled)
    design_phys = np.asarray(detector.flatten_design(design), dtype=np.float32)

    init_seq, training_seq = np.random.SeedSequence(int(seed)).spawn(2)
    params, state, opt_state = self._init_design_network(init_seq, init_params)
    initial_params = params

    tp, vp = self.train_pool, self.val_pool
    w0_train, w0_val = tp.current, vp.current
    w0_train_j, w0_val_j = jnp.int32(w0_train), jnp.int32(w0_val)

    manager = self._checkpoint_manager(step)
    design_tree = {"scaled": design_scaled, "physical": design_phys}
    train_loss_history, val_loss_history, pool_size_history = [], [], []
    train_sem_history, val_sem_history = [], []
    key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))

    if self._sample_round(design, w0_train, w0_val, self.n0) is None:
      return None

    epoch_in_round, since_improvement = 0, 0
    best_train = float("inf")
    objective = None
    train_mean, val_mean = float("nan"), float("nan")

    while True:
      train_count = tp.current - w0_train
      val_count = vp.current - w0_val
      key, subkey = jax.random.split(key)
      params, state, opt_state, _ = self._train_epoch(
        params, state, opt_state, subkey, w0_train_j, jnp.int32(train_count), tp.buffers()
      )

      train_eval = self._eval_train(params, state, tp.buffers(), w0_train_j)
      train_mean, train_sem = masked_mean_sem(train_eval, train_count)
      val_eval = self._eval_val(params, state, vp.buffers(), w0_val_j)
      val_mean, val_sem = masked_mean_sem(val_eval, val_count)
      train_mean, train_sem = float(train_mean), float(train_sem)
      val_mean, val_sem = float(val_mean), float(val_sem)

      train_loss_history.append(train_mean)
      val_loss_history.append(val_mean)
      train_sem_history.append(train_sem)
      val_sem_history.append(val_sem)
      pool_size_history.append(train_count)
      epoch_in_round += 1

      if on_epoch is not None:
        snapshot = self._snapshot(
          train_loss_history, val_loss_history, train_sem_history, val_sem_history, pool_size_history, train_count, val_count
        )
        threading.Thread(target=on_epoch, args=(snapshot, ), daemon=True).start()

      gap = abs(val_mean - train_mean)
      err = float(np.hypot(train_sem, val_sem))
      if train_mean < best_train:
        best_train = train_mean
        since_improvement = 0
      else:
        since_improvement += 1

      if epoch_in_round <= self.warmup_epochs:
        continue

      if gap + err > self.loss_precision:
        n_added = self._sample_round(design, w0_train, w0_val, self.n_increment)
        if n_added is None:
          return None
        if n_added == 0:
          raise RuntimeError(
            f"design did not reach precision within iteration_limit {self.iteration_limit} "
            f"(window={train_count}); gap={gap:.4g} err={err:.4g} gap+err={gap + err:.4g} vs "
            f"precision={self.loss_precision:.4g}."
          )
        if self.param_mix > 0.0:
          mix = self.param_mix
          params = jax.tree.map(lambda p, q: q + (1.0 - mix) * (p - q), params, initial_params)
          opt_state = self.optimizer.init(params)
        epoch_in_round = 0
        since_improvement, best_train = 0, float("inf")
        print(f"  [grow] window -> {tp.current - w0_train}, pool {tp.current}/{tp.capacity}", flush=True)
        continue

      if since_improvement >= self.patience:
        objective = (val_mean, gap + err)
        print(
          f"  [converged/strict] train={train_mean:.4f} val={val_mean:.4f} gap={gap:.4f} err={err:.4f} "
          f"prec={self.loss_precision:.4f} | {since_improvement} epochs without improvement | "
          f"window={train_count} epochs={len(train_loss_history)}", flush=True
        )
        break

    self._save_checkpoint(manager, len(train_loss_history), params, state, design_tree, train_mean, val_mean)
    if manager is not None:
      manager.wait_until_finished()
    self._persist_network(params, state, opt_state)
    objective_loss, objective_std = objective
    spent_train, spent_val = tp.current - w0_train, vp.current - w0_val
    return TrainResult(
      objective_loss, objective_std, spent_train + spent_val, params, spent_train=spent_train, spent_val=spent_val
    )


class StrictDesignTrainer(_StrictBase):
  """A FRESH network per design, minibatches over the current window alone."""

  def default_reveal(self):
    """A fresh network per design, so the design is constant across the whole batch. Withheld by
    default; set ``training.reveal`` to override."""
    return 'none'

  def _sample_indices(self, key, start, count):
    return window_sample_indices(self, key, start, count)

  def _sample_weights(self):
    return None

  def _init_design_network(self, init_seq, init_params):
    return fresh_design_network(self, init_seq, init_params)

  def _carried_state(self):
    return {}

  def _load_carried_state(self, data):
    """Nothing to load -- see :meth:`_carried_state`."""

  def _replay_carried_state(self, rows):
    """Nothing crosses a design boundary here -- see :meth:`_carried_state`."""


class StrictContinualTrainer(_StrictBase):
  """ONE persistent network across designs, with experience replay."""

  def default_reveal(self):
    """One network across every design with replay, so the design is what tells a replay row from a
    current one."""
    return 'design'

  def __init__(self, *args, replay_weight: float = 1.0, **kwargs):
    self.replay_weight = float(replay_weight)
    super().__init__(*args, **kwargs)
    self._running = persistent_network(self, self.seed)

  def _sample_indices(self, key, start, count):
    return replay_sample_indices(self, key, start, count)

  def _sample_weights(self):
    return replay_sample_weights(self)

  def _init_design_network(self, init_seq, init_params):
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
