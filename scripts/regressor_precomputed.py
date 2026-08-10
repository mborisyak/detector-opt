# DEPRECATED: not maintained against the typed-records detector API (Event/Target/Design
# namedtuples, combine/combine_scaled, raw buffers). Left untouched on purpose -- do not use
# or refactor; see scripts/regression.py for the current regressor-training entry point.
import math
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

import detopt
from detopt.detector.straw import SparseHits

# NOTE: NOT YET PORTED to the new Detector contract (detector-spec.md).
# Uses the ragged-layout model.combine; incompatible with the padded (B, M, F)
# events from the new detector. Will not run end-to-end until the model is
# ported to the padded layout (see detopt/nn/set_regressor.py).

# Enable JAX compilation logging to detect recompilations
jax.config.update("jax_log_compiles", True)
jax.config.update("jax_explain_cache_misses", True)


def save_deepset_input_label_hists(measurements_list, targets_list, design, model, detector, out_path: str):
    """
    Save histograms using precomputed measurements and targets.

    Args:
        measurements_list: List of measurement batches (SparseHits or dense)
        targets_list: List of target batches
        design: Detector design configuration (single design, will be tiled)
        model: Neural network model
        detector: Detector object (for normalization parameters)
        out_path: Output path for histogram plot
    """
    X_list = []
    y_list = []
    hits_per_event_list = []

    for measurements, y in zip(measurements_list, targets_list):
        batch_size = len(y)
        # Tile design to match batch size
        design_batch = np.tile(design, (batch_size, 1))

        # Normalized per-hit features come from here:
        hit_features, events, _ = model.combine(measurements, design_batch)  # hit_features: (n_hits, 8), events: (n_hits,)

        X_list.append(np.asarray(hit_features))  # move from JAX to numpy
        y_list.append(np.asarray(y))

        hits_per_event_list.append(np.bincount(np.asarray(events), minlength=batch_size))

    X = np.concatenate(X_list, axis=0)  # (total_hits, 8)
    y = np.concatenate(y_list, axis=0)  # (n_batches*batch, ...) targets

    # Normalize targets for visualization
    y = np.asarray(detector.normalize_target(y))

    hits_per_event = np.concatenate(hits_per_event_list)  # (n_batches*batch,)

    # (optional) cap number of hits to keep plots fast & memory sane
    max_hits = 200_000
    if X.shape[0] > max_hits:
        X = X[:max_hits]

    feature_names = [
        "station_norm",
        "view_norm",
        "layer_norm",
        "straw_norm",
        "tdc_norm",
        "pos_norm",
        "angle_norm",
        "B_norm",
    ]

    plots = [(f"x/{feature_names[i]}", X[:, i]) for i in range(X.shape[1])]
    plots.append(("x/hits_per_event", hits_per_event))

    if y.ndim == 2:
        for i in range(y.shape[1]):
            plots.append((f"y/{i}", y[:, i]))
    else:
        plots.append(("y", y))

    ncols = 3
    nrows = int(math.ceil(len(plots) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (name, data) in zip(axes, plots):
        ax.hist(np.asarray(data).reshape(-1), bins=60)
        ax.set_title(name)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(plots) :]:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def regress(
    seed,
    output,
    progress=False,
    restore=True,
    trace=None,
    report=None,
    precomputed_dir=None,
    **config,
):
    """
    Train regressor using precomputed detector responses.

    Args:
        seed: Random seed
        output: Output directory for model checkpoints
        progress: Show progress bars
        restore: Restore from checkpoint (if available)
        trace: Trace file path (unused)
        report: Report file path for plots
        precomputed_dir: Directory containing precomputed chunk files
        **config: Configuration dictionary
    """
    print(f"Configuration: {config}")
    print(f"Report parameter: {report}")
    print(f"Precomputed data directory: {precomputed_dir}")

    if precomputed_dir is None:
        raise ValueError("precomputed_dir must be specified for regressor_precomputed")

    rngs = nnx.Rngs(seed)
    np_rng_train = np.random.default_rng(seed=(seed, 0))
    np_rng_val = np.random.default_rng(seed=(seed, 1))  # Separate RNG for validation

    # Initialize detector (needed for design and normalization parameters)
    detector = detopt.detector.from_config(config["detector"])
    enc = detector.to_scaled(config["design"])  # design ALWAYS from config (detector holds none)
    enc = enc.reshape(1, -1)
    print(f"Scaled design: {enc}")
    print(f"Design parameters: {detector.get_design(enc)}")
    print(f"Spawn probability: {detector.p_spawn_single}")

    # Load precomputed data
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from load_precomputed_data import PrecomputedDataLoader

    data_loader = PrecomputedDataLoader(precomputed_dir)
    train_indices, val_indices = data_loader.get_train_val_split(val_fraction=0.2, seed=seed)
    print(f"\n✓ Loaded {data_loader.n_events} precomputed events")

    # Initialize regressor
    regressor = detopt.nn.from_config(detector, config=config["regressor"], rngs=rngs)

    # Split model into static and dynamic parts for functional JIT
    regressor_def, r_params, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)

    # Create optax optimizer directly for functional updates
    optax_optimizer = detopt.utils.config.optimizer(config["optimizer"])
    opt_state = optax_optimizer.init(r_params)

    epochs, steps = config["epochs"], config["steps"]
    print(f"Training: {epochs} epochs, {steps} steps per epoch")
    batch, validation_batches = config["batch"], config["validation_batches"]

    design = np.tile(enc, (batch, 1))

    def loss_f(x, c, t, r_params, r_state):
        regressor = nnx.merge(regressor_def, r_params, r_state)
        p = regressor(x, c, deterministic=True)

        # DCA
        p_denorm = detector.denormalize_predictions(p)
        t_denorm = t

        # Extract position and momentum: [x, y, z, px, py, pz]
        pos_pred = p_denorm[:, :3]  # (batch, 3)
        mom_pred = p_denorm[:, 3:]  # (batch, 3)
        pos_true = t_denorm[:, :3]
        mom_true = t_denorm[:, 3:]

        mom_pred_norm = jnp.linalg.norm(mom_pred, axis=1, keepdims=True)
        mom_true_norm = jnp.linalg.norm(mom_true, axis=1, keepdims=True)

        mom_pred_hat = mom_pred / jnp.clip(mom_pred_norm, min=1e-6)
        mom_true_hat = mom_true / jnp.clip(mom_true_norm, min=1e-6)

        dca_pred = jnp.linalg.norm(jnp.cross(pos_pred, mom_pred_hat), axis=1)
        dca_true = jnp.linalg.norm(jnp.cross(pos_true, mom_true_hat), axis=1)

        dca_loss = jnp.mean(jnp.square(dca_pred - dca_true))
        # end DCA

        # Standard MSE loss on normalized values
        target_norm = detector.normalize_target(t)
        diff = target_norm - p
        mse = jnp.mean(jnp.square(diff), axis=-1)
        mse_loss = jnp.mean(mse)

        # Combined loss: MSE + weighted DCA loss
        # Scale DCA loss to be comparable to MSE
        loss = mse_loss  # + 0.1 * dca_loss

        _, _, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)
        return loss, r_state

    def metric_f(x, c, t, r_params, r_state):
        regressor = nnx.merge(regressor_def, r_params, r_state)
        p = regressor(x, c, deterministic=True)
        target_norm = detector.normalize_target(t)
        rmse = jnp.sqrt(jnp.mean(jnp.square(target_norm - p), axis=-1))
        metric = jnp.mean(rmse)
        return metric

    @jax.jit
    def step(x, c, t, r_params, r_state, opt_state):
        c, t = jnp.array(c), jnp.array(t)
        (loss, r_state), grad = jax.value_and_grad(loss_f, argnums=3, has_aux=True)(x, c, t, r_params, r_state)
        updates, opt_state = optax_optimizer.update(grad, opt_state, r_params)
        r_params = optax.apply_updates(r_params, updates)
        return loss, r_params, r_state, opt_state

    training_losses = np.ndarray(shape=(epochs, steps))
    validation_losses = np.ndarray(shape=(epochs, 1))  # Single validation score per epoch

    val_predictions = []
    val_targets = []

    # Create fixed validation set (do this once before training loop)
    print("\nCreating fixed validation set...")
    fixed_val_measurements = []
    fixed_val_targets = []
    n_fixed_val_batches = max(50, validation_batches)  # At least 50 batches for stability

    for _ in range(n_fixed_val_batches):
        meas, targ = data_loader.get_batch_from_indices(val_indices, batch, rng=np_rng_val)
        fixed_val_measurements.append(tuple(jnp.asarray(m) for m in meas))
        fixed_val_targets.append(jnp.asarray(targ))

    print(f"✓ Fixed validation set: {n_fixed_val_batches} batches ({n_fixed_val_batches * batch} samples)")

    status = detopt.utils.progress.status_bar(disable=not progress)

    print("\nGenerating input/label histograms...")
    hist_measurements = []
    hist_targets = []
    for i in range(3):
        meas, targ = data_loader.get_batch(batch, rng=np_rng_train)
        hist_measurements.append(meas)
        hist_targets.append(targ)

    save_deepset_input_label_hists(
        hist_measurements,
        hist_targets,
        enc,
        regressor,
        detector,
        out_path="output/hists.png",
    )
    print("✓ Histograms saved to output/hists.png")

    print(f"\nStarting training...")
    for i in status.epochs(epochs):
        epoch_start = time.time()
        step_times = []
        for j in status.training(steps):
            step_start = time.time()
            measurements, target = data_loader.get_batch_from_indices(train_indices, batch, rng=np_rng_train)

            measurements = tuple(jnp.asarray(m) for m in measurements)
            loss, r_params, r_state, opt_state = step(measurements, design, target, r_params, r_state, opt_state)
            training_losses[i, j] = loss
            step_time = time.time() - step_start
            step_times.append(step_time)

        # Validation on fixed set
        val_loss_sum = 0.0
        val_predictions_epoch = []
        val_targets_epoch = []

        for j, (measurements, target) in enumerate(zip(fixed_val_measurements, fixed_val_targets)):
            regressor_merged = nnx.merge(regressor_def, r_params, r_state)
            predictions_norm = regressor_merged(measurements, jnp.array(design))

            # Denormalize predictions to raw units for visualization
            predictions = detector.denormalize_predictions(predictions_norm)

            val_loss = metric_f(measurements, design, target, r_params, r_state)
            val_loss_sum += val_loss

            # Collect ALL predictions for this epoch
            val_predictions_epoch.append(np.array(predictions))
            val_targets_epoch.append(np.array(target))

            # Debug: print first batch only
            if j == 0:
                print(f"\nEpoch {i + 1} Validation (batch 0/{len(fixed_val_measurements)}):")
                print(f"  Loss (first batch): {val_loss:.6f}")
                print(f"  Target[0]: {target[0]}")
                print(f"  Predicted[0]: {predictions[0]}")

        # Average validation loss over all batches
        validation_losses[i, 0] = val_loss_sum / len(fixed_val_measurements)

        # Store ALL validation predictions (not just first batch)
        val_predictions.append(np.concatenate(val_predictions_epoch, axis=0))
        val_targets.append(np.concatenate(val_targets_epoch, axis=0))

        # Print epoch summary (greppable format)
        train_loss_mean = np.mean(training_losses[i])
        val_loss_mean = np.mean(validation_losses[i])
        train_loss_std = np.std(training_losses[i])
        val_loss_std = np.std(validation_losses[i])
        epoch_time = time.time() - epoch_start
        avg_step_time = np.mean(step_times)
        first_step_time = step_times[0] if step_times else 0
        print(
            f"EPOCH: {i + 1}/{epochs} train_loss={train_loss_mean:.6f}±{train_loss_std:.4f} "
            f"val_loss={val_loss_mean:.6f} "
            f"epoch_time={epoch_time:.2f}s avg_step={avg_step_time:.3f}s first_step={first_step_time:.3f}s"
        )

        # Save plot after each epoch
        if report is not None:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot losses up to current epoch
            detopt.utils.viz.losses.plot(training_losses[: i + 1], axes[0, 0])
            axes[0, 0].set_title("Training losses")
            axes[0, 0].set_yscale("log")

            detopt.utils.viz.losses.plot(validation_losses[: i + 1], axes[0, 1])
            axes[0, 1].set_title("Validation losses")
            axes[0, 1].set_yscale("log")

            # Compute position and momentum errors
            if len(val_predictions) > 0:
                pred_array = np.array(val_predictions)  # (epochs, batch, 6)
                targ_array = np.array(val_targets)  # (epochs, batch, 6)

                # Position errors (x, y, z) - first 3 components
                pos_errors = pred_array[:, :, :3] - targ_array[:, :, :3]
                pos_rmse = np.sqrt(np.mean(pos_errors**2, axis=(1, 2)))  # RMSE per epoch

                # Momentum errors (px, py, pz) - last 3 components
                mom_errors = pred_array[:, :, 3:] - targ_array[:, :, 3:]
                mom_rmse = np.sqrt(np.mean(mom_errors**2, axis=(1, 2)))  # RMSE per epoch

                # Plot position precision
                epochs_so_far = np.arange(len(pos_rmse))
                axes[1, 0].plot(epochs_so_far, pos_rmse, "b-", linewidth=2)
                axes[1, 0].set_title("HNL Vertex Position RMSE")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_yscale("log")
                axes[1, 0].set_ylabel("Position RMSE (cm)")
                axes[1, 0].grid(True, alpha=0.3)

                # Plot momentum precision
                axes[1, 1].plot(epochs_so_far, mom_rmse, "r-", linewidth=2)
                axes[1, 1].set_title("HNL Momentum RMSE")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].set_yscale("log")
                axes[1, 1].set_ylabel("Momentum RMSE (GeV/c)")
                axes[1, 1].grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(report)
            plt.close(fig)

    # Save model parameters and state
    import os

    import orbax.checkpoint as ocp

    os.makedirs(output, exist_ok=True)
    manager = ocp.CheckpointManager(
        output,
        checkpointers=ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )

    # Use the updated params and state from training
    parameters = nnx.to_pure_dict(r_params)
    state = nnx.to_pure_dict(r_state)

    # Use the optimizer state from training
    optimizer_state = opt_state

    manager.save(
        0,
        args=ocp.args.Composite(model=ocp.args.PyTreeSave(detopt.utils.io.save_model(parameters, state, optimizer_state))),
    )

    print(f"\n✓ Model saved to {output}")
    print("✓ Training complete!")


if __name__ == "__main__":
    import gearup

    gearup.gearup(regress=regress).with_config("config/regress_precomputed.yaml")()
