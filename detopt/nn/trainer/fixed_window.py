"""Fixed-window training: one design, the dataset fixed up front, no growth.

THE PROCEDURE (user, 2026-08-23):

    fill the pools with ``window`` train events and ``val_window`` validation events under one
    design, from position 0 of the run's own event index;
    ``warmup_epochs`` epochs unconditionally;
    then each epoch, return when the TRAINING LOSS HAS PLATEAUED, by the growth trainer's own test --
    ``P(train change over +patience < loss_precision / 2) > 0.9`` on the Bayesian trend of the
    post-warmup training history, with the measured per-epoch SEMs as known.

It is the CONTROL for a growth run: the same events that run ended with, presented at once, so the
only difference is the path taken to the dataset. THE GROWTH TRAINER CANNOT SERVE AS THIS CONTROL --
its procedure is entitled to ask for more data, and at a fixed window that request is a hard error
(``design.py``), which is how two of nine cells died mid-descent on 2026-08-22. There is no
data-addition branch here, so the training-loss plateau is the only exit and the gap rule that fired
there does not exist.

``train_offset`` / ``val_offset`` skip that many entries of the run's event index before filling, so a
control can be drawn INDEPENDENTLY of the growth run rather than from the same events: the index is a
permutation, so a disjoint slice of it is an independent sample of the same population. At offset 0 the
control holds exactly the events the growth run held.

The window is the epoch: ``iteration_limit = window``, so an epoch is ``window // batch`` steps, and
``patience`` is that many epochs. A growth run's epoch is ITS ``iteration_limit``, which is larger --
the step counts of the two runs are not equal and are reported rather than matched.

Sibling of :mod:`.design` and :mod:`.strict`, never a subclass: their ``train`` is concrete and
concrete methods are final.

The result carries the FOUR fields every copy of this library has, and the caller reads the split spend
off the pool cursors. CERN runs an older copy whose ``TrainResult`` has no ``spent_train`` /
``spent_val``, and it stays pinned there because the runs this control is compared against came from it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np

from ...utils.training import bayesian_trend, masked_mean_sem, probability_change_below
from .common import Trainer, TrainResult, design_init_sequence, fresh_design_network, window_sample_indices

__all__ = ["FixedWindowTrainer", "PLATEAU_MESSAGE"]

PLATEAU_MESSAGE = "training loss did not plateau within max_epochs"


class FixedWindowTrainer(Trainer):
  """One design, a window fixed at construction, exit on the training-loss plateau."""

  @classmethod
  def spent_calls(self):
    """Reserves nothing, so the pools' fill IS the spend."""
    return self.train_pool.current + self.val_pool.current

  def from_config(cls, detector, config, *, window, val_window, max_epochs, seed=0, train_offset=0, val_offset=0):
    """Build from a run-config dict. ``window`` / ``val_window`` are the growth run's own final
        spend, so this trainer draws exactly the events that run held; ``budget`` stays the run's, so
        the shuffled event index -- and therefore those events -- are identical."""
    from ...utils.config import optimizer as make_optimizer, resolve_device

    training = config["training"]
    return cls(
      detector, regressor_config=config["regressor"], optimizer=make_optimizer(training["optimizer"]), batch=training["batch"],
      budget=training["budget"], window=window, val_window=val_window, warmup_epochs=training["warmup_epochs"],
      patience=training["patience"], loss_precision=training["loss_precision"], max_epochs=max_epochs,
      val_fraction=training.get("val_fraction",
                                0.25), eval_batch=training.get("eval_batch"), device=resolve_device(config.get("device")),
      seed=seed, reveal=training.get("reveal"), train_offset=train_offset, val_offset=val_offset,
    )

  def __init__(
    self, detector, *, regressor_config: dict, optimizer, batch: int, budget: int, window: int, val_window: int,
    warmup_epochs: int, patience: int, loss_precision: float, max_epochs: int, val_fraction: float = 0.25,
    eval_batch: int | None = None, device=None, seed: int = 0, reveal: str | None = None, train_offset: int = 0,
    val_offset: int = 0,
  ):
    self.train_offset = int(train_offset)
    self.val_offset = int(val_offset)
    self.warmup_epochs = int(warmup_epochs)
    self.patience = int(patience)
    self.loss_precision = float(loss_precision)
    self.max_epochs = int(max_epochs)
    super().__init__(
      detector, regressor_config=regressor_config, optimizer=optimizer, batch=batch, budget=budget, iteration_limit=int(window),
      val_iteration_limit=int(val_window), val_fraction=val_fraction, eval_batch=eval_batch, device=device, seed=seed,
      reveal=reveal,
    )

  def default_reveal(self):
    """One network per design at a fixed window, so the design is constant across every batch it
        draws. Withheld by default, as the per-design growth trainer withholds it."""
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

  def train(self, design_scaled, seed, *, init_params=None, on_epoch=None, step=0) -> TrainResult:
    """Train one design on the fixed window until the training loss plateaus.

        ``seed`` drives the training keys; the NETWORK comes from ``(self.seed, step)`` exactly as the
        growth trainer draws it, so the two start from the same parameters. Raises
        :data:`PLATEAU_MESSAGE` if ``max_epochs`` passes without the test firing -- an unfinished run
        is not a result and is never reported as one."""
    detector = self.detector
    design_scaled = np.asarray(design_scaled, dtype=np.float32)
    design = detector.to_nominal(design_scaled)
    # THE SAME NETWORK THE GROWTH RUN STARTED FROM. `DesignTrainer` draws its network from
    # `design_init_sequence(self.seed, step)` and its training keys from the `seed` argument; both are
    # mirrored here EXACTLY, so a control built at the same trainer seed and the same `step` begins at a
    # bit-identical network. Deriving the draw any other way would confound the path with the init.
    _, training_seq = np.random.SeedSequence(int(seed)).spawn(2)
    init_seq = design_init_sequence(self.seed, int(step))
    params, state, opt_state = self._init_design_network(init_seq, init_params)

    tp, vp = self.train_pool, self.val_pool
    for name, offset, count, index in (("train", self.train_offset, self.iteration_limit, self._train_index),
                                       ("validation", self.val_offset, self.val_iteration_limit, self._val_index)):
      if offset + count > len(index):
        raise ValueError(
          f"{name}: offset {offset} + window {count} exceeds the {len(index)}-event index; "
          f"an independently sampled control needs a budget that holds both slices"
        )
    self._fill_pool(design, tp, self.iteration_limit, self._train_index[self.train_offset:])
    self._fill_pool(design, vp, self.val_iteration_limit, self._val_index[self.val_offset:])
    start = jnp.int32(0)
    train_count, val_count = int(tp.current), int(vp.current)

    key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))
    train_history, val_history = [], []
    train_sem_history, val_sem_history, window_history = [], [], []

    plot_ctx = ThreadPoolExecutor(max_workers=1) if on_epoch is not None else nullcontext()
    with plot_ctx as plot_pool:
      for _ in range(self.max_epochs):
        key, subkey = jax.random.split(key)
        params, state, opt_state, _ = self._train_epoch(
          params, state, opt_state, subkey, start, jnp.int32(train_count), tp.buffers()
        )
        train_mean, train_sem = masked_mean_sem(self._eval_train(params, state, tp.buffers(), start), train_count)
        val_mean, val_sem = masked_mean_sem(self._eval_val(params, state, vp.buffers(), start), val_count)
        train_mean, train_sem = float(train_mean), float(train_sem)
        val_mean, val_sem = float(val_mean), float(val_sem)
        train_history.append(train_mean)
        val_history.append(val_mean)
        train_sem_history.append(train_sem)
        val_sem_history.append(val_sem)
        window_history.append(train_count)
        if on_epoch is not None:
          plot_pool.submit(
            on_epoch,
            self._snapshot(
              train_history, val_history, train_sem_history, val_sem_history, window_history, train_count, val_count
            ),
          )

        if len(train_history) <= self.warmup_epochs:
          continue
        train_series = np.asarray(train_history[self.warmup_epochs:], dtype=np.float64)
        val_series = np.asarray(val_history[self.warmup_epochs:], dtype=np.float64)
        train_series_sem = np.asarray(train_sem_history[self.warmup_epochs:], dtype=np.float64)
        if train_series.shape[0] < 3:
          continue
        prior_sigma = max(float(train_series[0]), float(val_series[0])) / 3.0
        trend_mean, trend_cov = bayesian_trend(train_series, train_series_sem, prior_sigma)
        settled = probability_change_below(trend_mean, trend_cov, self.patience, 0.5 * self.loss_precision)
        if settled > 0.9:
          diff = abs(val_mean - train_mean)
          err = float(np.hypot(train_sem, val_sem))
          print(
            f"  [plateau/fixed-window] train={train_mean:.4f} val={val_mean:.4f} diff={diff:.4f} err={err:.4f} "
            f"| P(settled)={settled:.3f} | window={train_count} epochs={len(train_history)}", flush=True
          )
          return TrainResult(
            0.5 * (train_mean + val_mean), float(np.hypot(diff / np.sqrt(12.0), 0.5 * err)), train_count + val_count, params
          )

    raise RuntimeError(
      f"{PLATEAU_MESSAGE} {self.max_epochs} (window={train_count}); "
      f"train={train_history[-1]:.4g} val={val_history[-1]:.4g}"
    )
