#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from flax import nnx
from gpytorch.mlls import ExactMarginalLogLikelihood

matplotlib.use("AGG")

import detopt

# Make load_hnl_data importable when running this script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_hnl_data import HNLDataLoader

VALID_INIT_STRATEGIES = ("from_scratch", "continue", "closest")


def _resolve_device(name):
    """Resolve a backend string (``"cpu"`` / ``"gpu"`` / ``"tpu"``) to a JAX device.

    ``None`` falls back to JAX's default (no explicit placement).
    """
    if name is None:
        return None
    name = str(name).lower()
    try:
        devices = jax.devices(name)
    except RuntimeError as exc:
        raise RuntimeError(f"JAX backend {name!r} not available: {exc}") from exc
    if not devices:
        raise RuntimeError(f"No JAX devices found for backend {name!r}")
    return devices[0]


# ----------------------------- statistics ---------------------------------- #


def check_loss_mean_agreement(train_losses, val_losses, max_mean_difference, z=1.96):
    """Two-sample equivalence test on the difference of means."""
    train_losses = np.asarray(train_losses, dtype=np.float64)
    val_losses = np.asarray(val_losses, dtype=np.float64)

    train_mean = np.mean(train_losses)
    val_mean = np.mean(val_losses)

    train_std = np.std(train_losses, ddof=1)
    val_std = np.std(val_losses, ddof=1)

    n_train = len(train_losses)
    n_val = len(val_losses)

    delta_mu = val_mean - train_mean
    sigma_delta_mu = np.sqrt(train_std**2 / n_train + val_std**2 / n_val)

    lower = delta_mu - z * sigma_delta_mu
    upper = delta_mu + z * sigma_delta_mu

    agrees = lower >= -max_mean_difference and upper <= max_mean_difference
    return agrees, delta_mu, sigma_delta_mu


# ------------------------------ data layer --------------------------------- #


def _load_data_loader(config):
    """Build the HNL data loader from the top-level ``data`` config block."""
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "selected_data")
    max_particles = int(data_cfg.get("max_particles", config["detector"]["straw"].get("max_particles", 2)))
    val_fraction = float(data_cfg.get("val_fraction", 0.2))
    split_seed = int(data_cfg.get("split_seed", 42))
    return HNLDataLoader(
        data_dir=data_dir,
        max_particles=max_particles,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )


def _random_batch(loader, indices, batch_size, rng):
    """Random batch (with replacement) drawn from ``indices``."""
    sampled = rng.choice(len(indices), size=batch_size, replace=True)
    return loader._build_batch_from_indices(indices[sampled])


def _chunk_sparse_hits(
    events,
    layers,
    straws,
    times,
    mask,
    targets,
    n_chunks: int,
    batch_size: int,
    max_hits_per_chunk: int,
):
    """Bucket flat sparse-hit arrays from a single simulator call into
    ``n_chunks`` fixed-shape chunks of ``batch_size`` events each.

    The simulator emits one flat array of length ``2 * (n_chunks * batch_size)
    * max_particles * n_layers`` for each per-hit field, plus a contiguous
    ``targets`` array of shape ``(n_chunks * batch_size, 6)``. We bucket hits
    by ``events // batch_size`` and pad each chunk to ``max_hits_per_chunk``.

    Returns six leading-axis-batched ``np.ndarray``s suitable for
    ``device_put`` then ``jax.lax.scan``.
    """
    valid = mask.astype(bool)
    v_chunk_id = (events[valid] // batch_size).astype(np.int32)
    v_event_local = (events[valid] % batch_size).astype(np.int32)
    v_layers = layers[valid].astype(np.int32)
    v_straws = straws[valid].astype(np.int32)
    v_times = times[valid].astype(np.float32)

    # Sort hits by chunk index so each chunk's hits are contiguous.
    order = np.argsort(v_chunk_id, kind="stable")
    v_chunk_id = v_chunk_id[order]
    v_event_local = v_event_local[order]
    v_layers = v_layers[order]
    v_straws = v_straws[order]
    v_times = v_times[order]

    starts = np.searchsorted(v_chunk_id, np.arange(n_chunks), side="left")
    ends = np.searchsorted(v_chunk_id, np.arange(n_chunks), side="right")

    ce = np.zeros((n_chunks, max_hits_per_chunk), dtype=np.int32)
    cl = np.zeros((n_chunks, max_hits_per_chunk), dtype=np.int32)
    cs = np.zeros((n_chunks, max_hits_per_chunk), dtype=np.int32)
    ct = np.zeros((n_chunks, max_hits_per_chunk), dtype=np.float32)
    cm = np.zeros((n_chunks, max_hits_per_chunk), dtype=np.int32)

    for c in range(n_chunks):
        s, e = int(starts[c]), int(ends[c])
        n_hits = min(e - s, max_hits_per_chunk)
        if n_hits <= 0:
            continue
        ce[c, :n_hits] = v_event_local[s : s + n_hits]
        cl[c, :n_hits] = v_layers[s : s + n_hits]
        cs[c, :n_hits] = v_straws[s : s + n_hits]
        ct[c, :n_hits] = v_times[s : s + n_hits]
        cm[c, :n_hits] = 1

    chunked_targets = np.asarray(targets, dtype=np.float32).reshape(n_chunks, batch_size, -1)
    return ce, cl, cs, ct, cm, chunked_targets


def _sequential_batches(loader, indices, batch_size):
    """Yield ``(daughter_data, hnl_targets)`` chunks covering ``indices`` once."""
    n = len(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield loader._build_batch_from_indices(indices[start:end])


class IterationPool:
    """Per-BO-iteration event pool with stable train/val partitioning.

    Events are drawn without replacement from a global shuffle, partitioned
    80/20 (or whatever ``val_fraction``) into train and val. ``grow`` appends
    new events to either side; events never change partitions.
    """

    def __init__(self, n_total_events: int, val_fraction: float, rng: np.random.Generator):
        self.n_total = int(n_total_events)
        self.val_fraction = float(val_fraction)
        self._order = rng.permutation(self.n_total).astype(np.int32)
        self._consumed = 0
        self.train_indices = np.empty(0, dtype=np.int32)
        self.val_indices = np.empty(0, dtype=np.int32)

    def grow_to(self, n_events: int) -> tuple[int, int]:
        """Ensure the pool contains at least ``n_events`` distinct events."""
        n_events = int(min(max(2, n_events), self.n_total))
        n_to_add = n_events - self._consumed
        if n_to_add > 0:
            new_events = self._order[self._consumed : self._consumed + n_to_add]
            n_new_val = max(0, min(n_to_add, int(round(n_to_add * self.val_fraction))))
            self.val_indices = np.concatenate([self.val_indices, new_events[:n_new_val]])
            self.train_indices = np.concatenate([self.train_indices, new_events[n_new_val:]])
            self._consumed += n_to_add
        return int(len(self.train_indices)), int(len(self.val_indices))

    @property
    def size(self) -> int:
        return int(self._consumed)

    def exhausted(self) -> bool:
        return self._consumed >= self.n_total


# ----------------------------- plotting ------------------------------------ #


def _plot_iteration_losses(history, iteration, design, val_loss, plots_dir):
    """Save a per-epoch training/validation loss plot for one BO iteration."""
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    train_loss_per_epoch = history["train_loss_per_epoch"]
    val_loss_per_epoch = history["val_loss_per_epoch"]
    train_budget_per_epoch = history.get("train_budget_per_epoch")
    final_train_budget = history.get("final_train_budget")
    data_extensions_used = history.get("data_extensions_used", 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, train_loss_per_epoch.shape[0] + 1)

    ax.plot(epochs, train_loss_per_epoch, marker="o", label="train", color="tab:blue")
    if val_loss_per_epoch.size > 0:
        ax.plot(
            epochs[: val_loss_per_epoch.shape[0]],
            val_loss_per_epoch,
            marker="s",
            label="val",
            color="tab:orange",
        )

    if train_budget_per_epoch is not None and train_budget_per_epoch.size > 1:
        diffs = np.diff(train_budget_per_epoch)
        growth_epochs = np.where(diffs > 0)[0] + 1
        for k, idx in enumerate(growth_epochs):
            ax.axvline(
                epochs[idx],
                color="tab:green",
                linestyle="--",
                alpha=0.5,
                label=(f"data grew to {int(train_budget_per_epoch[idx])}" if k == 0 else None),
            )

    title_suffix = f"final val_loss={val_loss:.4f}"
    if final_train_budget is not None:
        title_suffix += f" | n_train={final_train_budget} (+{data_extensions_used} growth)"
    ax.set_title(f"Iter {iteration} - Train/Val convergence ({title_suffix})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(plots_dir, f"iter_{iteration:03d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    design_path = os.path.join(plots_dir, f"iter_{iteration:03d}_design.json")
    with open(design_path, "w") as f:
        json.dump(design, f, indent=2, default=float)

    return out_path


# ------------------------------ regressor ---------------------------------- #


def _init_regressor_state(detector, config, seed):
    """Create a fresh regressor and matching optimizer state."""
    rngs = nnx.Rngs(seed)
    regressor = detopt.nn.from_config(detector, config=config["regressor"], rngs=rngs)
    regressor_def, r_params, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)

    optimizer = detopt.utils.config.optimizer(config["optimizer"])
    opt_state = optimizer.init(r_params)
    return regressor_def, r_params, r_state, opt_state, optimizer


# --------------------------- train & evaluate ------------------------------ #


def train_and_evaluate(
    detector,
    loader,
    config,
    design_params,
    seed,
    init_state=None,
    on_epoch=None,
):
    """Train regressor on ``design_params`` and return (val_loss, final_state, history).

    The BO script owns all data fetching. This function:
      * builds an :class:`IterationPool` of size ``n0`` (from config), with an
        80/20 train/val split;
      * trains by sampling random batches from the pool's train partition;
      * monitors per-epoch val MSE on random batches from the pool's val
        partition (cheap);
      * runs a sliding-window train/val agreement test every epoch;
      * if training plateaus without agreement, grows the pool by
        ``n_increment`` and runs another ``epochs`` block; raises if the
        dataset is exhausted in that state;
      * at the end, computes the BO objective as the exact MSE over **every
        event** in the pool's final val partition (sequential, no random).
    """
    detector.update_from_yaml_design(design_params)
    design = detector.get_encoded_current_design()

    if init_state is None:
        regressor_def, r_params, r_state, opt_state, optimizer = _init_regressor_state(detector, config, seed)
    else:
        regressor_def, r_params, r_state, opt_state = init_state
        optimizer = detopt.utils.config.optimizer(config["optimizer"])

    # --- per-iteration data pool ---
    n0 = int(config.get("n0", 1000))
    n_increment = int(config.get("n_increment", 500))
    val_fraction = float(config.get("data", {}).get("val_fraction", 0.2))
    n_total = int(loader.n_events)
    pool_rng = np.random.default_rng((seed, 0xBEEF))
    pool = IterationPool(n_total, val_fraction, pool_rng)
    pool.grow_to(n0)
    current_train_budget = pool.size
    data_extensions_used = 0

    # --- training plumbing ---
    batch = int(config["batch"])
    steps = int(config["steps"])
    n_val_batches = int(config.get("validation_batches", 10))
    device = _resolve_device(config.get("device"))
    design_array = np.tile(design.reshape(1, -1), (batch, 1))
    design_array_j = jax.device_put(jnp.asarray(design_array), device=device)

    # Co-locate model params/state and optimizer state with the kernels.
    r_params = jax.device_put(r_params, device=device)
    r_state = jax.device_put(r_state, device=device)
    opt_state = jax.device_put(opt_state, device=device)

    # Hits-per-call upper bound used both by the simulator and by chunking.
    max_hits_per_chunk = 2 * batch * detector.max_particles * detector.n_layers

    train_rng = np.random.default_rng((seed, 0xC0DE, 0))
    val_rng = np.random.default_rng((seed, 0xC0DE, 1))

    def loss_fn(x, c, t, params, state):
        reg = nnx.merge(regressor_def, params, state)
        pred = reg(x, c, deterministic=True)
        target_norm = detector.normalize_target(t)
        mse = jnp.mean(jnp.square(target_norm - pred))
        _, _, state = nnx.split(reg, nnx.Param, nnx.Variable)
        return mse, state

    @partial(jax.jit, device=device)
    def train_epoch_kernel(chunked_meas, chunked_targets, params, state, opt_state):
        """Run one epoch's worth of SGD steps in a single fused XLA call.

        ``chunked_meas`` is a tuple of five ``(n_steps, max_hits_per_chunk)``
        arrays (events, layers, straws, times, mask). ``chunked_targets`` has
        shape ``(n_steps, batch, target_dim)``.
        """

        def step_body(carry, inputs):
            params, state, opt_state = carry
            info, tgt = inputs
            (loss, state), grad = jax.value_and_grad(loss_fn, argnums=3, has_aux=True)(info, design_array_j, tgt, params, state)
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, state, opt_state), loss

        (params, state, opt_state), losses = jax.lax.scan(
            step_body,
            init=(params, state, opt_state),
            xs=(chunked_meas, chunked_targets),
        )
        return params, state, opt_state, losses

    @partial(jax.jit, device=device)
    def val_epoch_kernel(chunked_meas, chunked_targets, params, state):
        """Compute per-batch MSE for all validation chunks in one fused call."""

        def body(_, inputs):
            info, tgt = inputs
            reg = nnx.merge(regressor_def, params, state)
            pred = reg(info, design_array_j, deterministic=True)
            target_norm = detector.normalize_target(tgt)
            mse = jnp.mean(jnp.square(target_norm - pred))
            return None, mse

        _, mses = jax.lax.scan(body, init=None, xs=(chunked_meas, chunked_targets))
        return mses

    def _collect_and_chunk(indices, n_chunks, rng, sim_seed):
        """One simulator call for ``n_chunks * batch`` events; chunked for JIT."""
        total = n_chunks * batch
        dd, tg = _random_batch(loader, indices, total, rng)
        big_design = np.tile(design.reshape(1, -1), (total, 1))
        _, info, target, _, _ = detector(
            seed=sim_seed,
            daughter_data=dd,
            hnl_targets=tg,
            configurations=big_design,
        )
        events, layers, straws, times, mask = info
        return _chunk_sparse_hits(
            np.asarray(events),
            np.asarray(layers),
            np.asarray(straws),
            np.asarray(times),
            np.asarray(mask),
            np.asarray(target),
            n_chunks=n_chunks,
            batch_size=batch,
            max_hits_per_chunk=max_hits_per_chunk,
        )

    status = detopt.utils.progress.status_bar(disable=False)

    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 1e-4))
    min_delta_relative = float(config.get("min_delta_relative", 0.01))
    agreement_window = int(config.get("agreement_window", 20))
    max_mean_difference_relative = float(config.get("max_mean_difference_relative", 0.10))
    z_value = float(config.get("z_value", 1.96))
    min_epochs = int(config.get("min_epochs", 3))
    growth_cooldown = int(config.get("growth_cooldown", agreement_window))
    cooldown_remaining = 0

    best_train_loss = float("inf")
    epochs_without_improvement = 0
    train_losses_history = []
    train_losses_history_steps = []
    val_losses_history = []
    train_budget_history = []

    def eval_val_mse(epoch_idx):
        """Per-epoch val MSE: one simulator call + one fused JIT kernel."""
        if len(pool.val_indices) == 0 or n_val_batches <= 0:
            return float("nan")
        ce, cl, cs, ct, cm, ctargets = _collect_and_chunk(
            pool.val_indices,
            n_chunks=n_val_batches,
            rng=val_rng,
            sim_seed=(seed + 1000, epoch_idx),
        )
        mses = val_epoch_kernel(
            (
                jax.device_put(ce, device=device),
                jax.device_put(cl, device=device),
                jax.device_put(cs, device=device),
                jax.device_put(ct, device=device),
                jax.device_put(cm, device=device),
            ),
            jax.device_put(ctargets, device=device),
            r_params,
            r_state,
        )
        return float(jnp.mean(mses))

    early_stopped = False
    last_agrees = None
    last_training_plateaued = False
    global_epoch = 0

    def _emit_epoch():
        """Push the current history snapshot to ``on_epoch`` (if given)."""
        if on_epoch is None:
            return
        snapshot = {
            "train_losses": np.asarray(train_losses_history_steps, dtype=np.float64),
            "train_loss_per_epoch": np.asarray(train_losses_history, dtype=np.float64),
            "val_loss_per_epoch": np.asarray(val_losses_history, dtype=np.float64),
            "train_budget_per_epoch": np.asarray(train_budget_history, dtype=np.int64),
            "val_losses": np.empty(0, dtype=np.float64),
            "data_extensions_used": int(data_extensions_used),
            "final_train_budget": int(current_train_budget),
            "final_val_pool_size": int(len(pool.val_indices)),
            "early_stopped": False,
        }
        try:
            on_epoch(snapshot)
        except Exception as exc:
            # Plotting failures should never abort training.
            print(f"[on_epoch] callback raised {type(exc).__name__}: {exc}")

    while True:
        for epoch in status.epochs(config["epochs"]):
            # One simulator call for the whole epoch's training data; one
            # fused JIT'd scan for all SGD steps.
            ce, cl, cs, ct, cm, ctargets = _collect_and_chunk(
                pool.train_indices,
                n_chunks=steps,
                rng=train_rng,
                sim_seed=(seed, global_epoch),
            )
            r_params, r_state, opt_state, losses = train_epoch_kernel(
                (
                    jax.device_put(ce, device=device),
                    jax.device_put(cl, device=device),
                    jax.device_put(cs, device=device),
                    jax.device_put(ct, device=device),
                    jax.device_put(cm, device=device),
                ),
                jax.device_put(ctargets, device=device),
                r_params,
                r_state,
                opt_state,
            )
            epoch_train_losses = np.asarray(losses).tolist()

            train_losses_history_steps.append(epoch_train_losses)
            mean_train_loss = float(np.mean(epoch_train_losses))
            train_losses_history.append(mean_train_loss)
            train_budget_history.append(current_train_budget)
            val_losses_history.append(eval_val_mse(global_epoch))
            _emit_epoch()

            # Plateau bookkeeping (relative OR absolute improvement counts).
            abs_threshold = best_train_loss - min_delta
            rel_threshold = best_train_loss * (1.0 - min_delta_relative)
            improvement_threshold = min(abs_threshold, rel_threshold)
            if mean_train_loss < improvement_threshold:
                best_train_loss = mean_train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            training_plateaued = epochs_without_improvement >= patience
            last_training_plateaued = training_plateaued

            # Sliding-window agreement test over the last K epochs.
            grow_now = False
            grow_delta_mu = 0.0
            grow_sigma = 0.0
            grow_tau = 0.0
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif len(train_losses_history) >= agreement_window:
                train_window = train_losses_history[-agreement_window:]
                val_window = val_losses_history[-agreement_window:]
                # Relative equivalence band: tau scales with current loss.
                scale = 0.5 * (float(np.mean(train_window)) + float(np.mean(val_window)))
                tau = max_mean_difference_relative * max(scale, 1e-12)
                agrees, delta_mu, sigma_delta_mu = check_loss_mean_agreement(
                    train_window,
                    val_window,
                    max_mean_difference=tau,
                    z=z_value,
                )
                last_agrees = agrees
                if epoch + 1 >= min_epochs and agrees and training_plateaued:
                    print(
                        f"\nEarly stop: train/val agree over last "
                        f"{agreement_window} epochs | tau={tau:.6f} | "
                        f"delta_mu={delta_mu:.6f}, "
                        f"sigma_delta_mu={sigma_delta_mu:.6f}"
                    )
                    early_stopped = True
                    break

                # Grow mid-block when train/val disagree, even if training
                # has not plateaued yet.
                if not agrees and epoch + 1 >= min_epochs:
                    grow_now = True
                    grow_delta_mu = float(delta_mu)
                    grow_sigma = float(sigma_delta_mu)
                    grow_tau = float(tau)

            global_epoch += 1

            if grow_now:
                if n_increment <= 0:
                    continue
                if pool.exhausted():
                    raise RuntimeError(
                        f"Dataset exhausted: train/val disagree at the full "
                        f"pool ({n_total} events) after "
                        f"{data_extensions_used} extensions. Add more data "
                        f"or relax convergence criteria."
                    )
                new_budget = min(current_train_budget + n_increment, n_total)
                print(
                    f"\n[data growth] disagreement mid-block at epoch "
                    f"{global_epoch}: {current_train_budget} -> {new_budget} "
                    f"pool events (ext {data_extensions_used + 1}) | "
                    f"delta_mu={grow_delta_mu:.6f}, sigma={grow_sigma:.6f}, "
                    f"tau={grow_tau:.6f}"
                )
                pool.grow_to(new_budget)
                current_train_budget = pool.size
                data_extensions_used += 1
                best_train_loss = float("inf")
                epochs_without_improvement = 0
                last_agrees = None
                last_training_plateaued = False
                cooldown_remaining = growth_cooldown

        if early_stopped:
            break

        # End of an epochs block without convergence.
        overfit_like = bool(last_training_plateaued) and (last_agrees is False)
        if not overfit_like:
            break

        if n_increment <= 0:
            break

        if pool.exhausted():
            raise RuntimeError(
                f"Dataset exhausted: training plateaued without train/val "
                f"agreement at the full pool ({n_total} events) after "
                f"{data_extensions_used} extensions. Add more data or relax "
                f"convergence criteria."
            )

        new_budget = min(current_train_budget + n_increment, n_total)
        print(
            f"\n[data growth] plateaued without agreement: "
            f"{current_train_budget} -> {new_budget} pool events "
            f"(ext {data_extensions_used + 1})"
        )
        pool.grow_to(new_budget)
        current_train_budget = pool.size
        data_extensions_used += 1
        best_train_loss = float("inf")
        epochs_without_improvement = 0
        last_agrees = None
        last_training_plateaued = False

    # --- final validation: exact MSE over the entire iteration val pool ---
    reg = nnx.merge(regressor_def, r_params, r_state)
    val_per_sample = []
    for dd, tg in _sequential_batches(loader, pool.val_indices, batch):
        n_evt = tg.shape[0]
        c = np.tile(design.reshape(1, -1), (n_evt, 1))
        _, measurements, target, _, _ = detector(
            seed=(seed + 1000, 1, n_evt),
            daughter_data=dd,
            hnl_targets=tg,
            configurations=c,
        )
        pred = reg(measurements, jnp.array(c), deterministic=True)
        target_norm = detector.normalize_target(target)
        mse_per_event = jnp.mean(jnp.square(target_norm - pred), axis=-1)
        val_per_sample.extend(np.asarray(mse_per_event).tolist())
    val_loss = float(np.mean(val_per_sample))

    final_state = (regressor_def, r_params, r_state, opt_state)
    history = {
        "train_losses": np.asarray(train_losses_history_steps, dtype=np.float64),
        "train_loss_per_epoch": np.asarray(train_losses_history, dtype=np.float64),
        "val_loss_per_epoch": np.asarray(val_losses_history, dtype=np.float64),
        "train_budget_per_epoch": np.asarray(train_budget_history, dtype=np.int64),
        "val_losses": np.asarray(val_per_sample, dtype=np.float64),
        "data_extensions_used": int(data_extensions_used),
        "final_train_budget": int(current_train_budget),
        "final_val_pool_size": int(len(pool.val_indices)),
        "early_stopped": bool(early_stopped),
    }
    return val_loss, final_state, history


# ---------------------------------- BO ------------------------------------- #


def optimize(config, output_dir, n_iter=50, n_init=10, seed=42):
    nn_init_strategy = config["nn_init_strategy"]
    if nn_init_strategy not in VALID_INIT_STRATEGIES:
        raise ValueError(f"Unknown nn_init_strategy: {nn_init_strategy!r}. " f"Must be one of {VALID_INIT_STRATEGIES}.")

    detector = detopt.detector.from_config(config["detector"])
    loader = _load_data_loader(config)

    # Define parameter bounds
    n_design_params = detector.yaml_design_shape()[0]
    bounds = torch.stack(
        [
            torch.zeros(n_design_params, dtype=torch.float64),
            torch.ones(n_design_params, dtype=torch.float64),
        ]
    )

    train_X = torch.empty((0, bounds.shape[1]), dtype=torch.float64)
    train_Y = torch.empty((0, 1), dtype=torch.float64)
    results = []
    best_obj, best_design = -np.inf, None

    running_state = None
    state_history: list = []

    print(f"\nBayesian Optimization with BoTorch: {n_iter} iterations")
    print(f"Output directory: {output_dir}")
    print(f"NN init strategy: {nn_init_strategy}")
    print("=" * 80)

    for i in range(n_iter):
        iter_start = time.time()

        # Propose design
        if i < n_init:
            params = torch.rand(1, bounds.shape[1], dtype=torch.float64) * (bounds[1] - bounds[0]) + bounds[0]
        else:
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            EI = ExpectedImprovement(gp, best_f=train_Y.max())
            params, _ = optimize_acqf(
                EI,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
            )

        params_np = params.detach().cpu().squeeze().numpy()
        design = detector.decode_yaml_design(params_np)

        init_state = None
        closest_idx = None
        if nn_init_strategy == "continue":
            init_state = running_state
        elif nn_init_strategy == "closest" and len(state_history) > 0:
            dists = torch.linalg.norm(train_X - params, dim=-1)
            closest_idx = int(torch.argmin(dists).item())
            init_state = state_history[closest_idx]
            print(f"[closest] Warm-starting from iteration {closest_idx} " f"(distance={float(dists[closest_idx]):.4f})")

        print(f"\n[Iteration {i + 1}/{n_iter}] Evaluating design...")

        plots_dir = os.path.join(output_dir, "plots")

        def _on_epoch(snapshot, _i=i, _design=design):
            # Use the most recent per-epoch val MSE as a live proxy in the
            # title; the real BO objective is the exact full-val MSE we
            # compute at the end of train_and_evaluate.
            vlp = snapshot["val_loss_per_epoch"]
            live_val = float(vlp[-1]) if vlp.size > 0 else float("nan")
            _plot_iteration_losses(
                snapshot,
                iteration=_i,
                design=_design,
                val_loss=live_val,
                plots_dir=plots_dir,
            )

        val_loss, final_state, history = train_and_evaluate(
            detector,
            loader,
            config,
            design,
            seed + i,
            init_state=init_state,
            on_epoch=_on_epoch,
        )
        objective = -val_loss

        plot_path = _plot_iteration_losses(
            history,
            iteration=i,
            design=design,
            val_loss=val_loss,
            plots_dir=plots_dir,
        )

        if nn_init_strategy == "continue":
            running_state = final_state
        elif nn_init_strategy == "closest":
            state_history.append(final_state)

        train_X = torch.cat([train_X, params])
        train_Y = torch.cat([train_Y, torch.tensor([[objective]], dtype=torch.float64)])

        iter_time = time.time() - iter_start

        if objective > best_obj:
            best_obj, best_design = objective, design
            print(f"✓ Iter {i + 1}/{n_iter} | Loss: {val_loss:.6f} | " f"Time: {iter_time:.1f}s | BEST ★")
        else:
            print(f"✓ Iter {i + 1}/{n_iter} | Loss: {val_loss:.6f} | " f"Time: {iter_time:.1f}s")

        result_entry = {
            "iteration": i,
            "design": design,
            "objective": float(objective),
            "nn_init_strategy": nn_init_strategy,
            "warm_start_from": closest_idx,
            "final_train_budget": int(history.get("final_train_budget", 0)),
            "final_val_pool_size": int(history.get("final_val_pool_size", 0)),
            "data_extensions_used": int(history.get("data_extensions_used", 0)),
            "early_stopped": bool(history.get("early_stopped", False)),
        }
        results.append(result_entry)

        checkpoint_path = f"{output_dir}/checkpoint_iter_{i:03d}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(
                {
                    "iteration": i,
                    "design": design,
                    "objective": float(objective),
                    "val_loss": float(val_loss),
                    "time": iter_time,
                    "nn_init_strategy": nn_init_strategy,
                    "warm_start_from": closest_idx,
                    "final_train_budget": int(history.get("final_train_budget", 0)),
                    "final_val_pool_size": int(history.get("final_val_pool_size", 0)),
                    "data_extensions_used": int(history.get("data_extensions_used", 0)),
                    "early_stopped": bool(history.get("early_stopped", False)),
                },
                f,
                indent=2,
            )

        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "best_objective": float(best_obj),
                    "best_design": best_design,
                    "n_iterations_completed": i + 1,
                    "method": "BoTorch",
                    "nn_init_strategy": nn_init_strategy,
                },
                f,
                indent=2,
            )

        print(f"Saved: {checkpoint_path}")
        print(f"Saved: {plot_path}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest objective: {best_obj:.6f} (validation loss: {-best_obj:.6f})")
    print(f"Improvement: {best_obj - results[0]['objective']:.6f}")
    print("\nBest design:")
    print(json.dumps(best_design, indent=2))
    print(f"\nResults saved to: {output_dir}/results.json")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="output/bayesian_opt")
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    optimize(config, args.output, args.n_iterations, args.n_initial, args.seed)
