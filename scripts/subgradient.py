import os
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import nnx

import detopt

# NOTE: NOT YET PORTED to the new Detector contract (detector-spec.md).
# The ragged-layout regressor is incompatible with the padded (B, M, F)
# events from the new detector. Call sites updated mechanically; will not
# run end-to-end until the regressor is ported (see nn/set_regressor.py).

matplotlib.use("AGG")

MAX_INT = 9223372036854775807

# Enable JAX compilation logging to detect recompilations
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)


def optimize(seed, output, progress=True, restore=True, trace=None, report=None, **config):
    print(f"using {config.get('regressor')} as regressor")
    print("Design optimization mode: Physical parameters (13 params)")

    np_rng = np.random.default_rng(seed=seed)
    get_seed = lambda: np_rng.integers(low=0, high=MAX_INT)
    rng = jax.random.PRNGKey(get_seed())

    checkpointer = detopt.utils.io.get_checkpointer(output)
    if checkpointer.latest_step() is not None and checkpointer.latest_step() >= config["epochs"]:
        return

    detector = detopt.detector.from_config(config["detector"])
    print(detector.get_current_yaml_design())
    print(detector.yaml_to_layer_design(detector.get_current_yaml_design()))

    rng, key_init = jax.random.split(rng, num=2)
    restored = detopt.utils.io.restore_state(checkpointer, detector, config, rngs=nnx.Rngs(key_init), restore=restore)

    starting_epoch = restored["starting_epoch"]

    # Load aux data first (needed for design params)
    aux: dict | None = restored["aux"]

    # Load or initialize design parameters
    if aux is not None and "design_params" in aux:
        design_params_dict = aux["design_params"]
        design = detector.encode_yaml_design(design_params_dict)
        print(f"Restored design parameters from checkpoint")
    else:
        design_params_dict = detector.get_current_yaml_design()
        design = detector.encode_yaml_design(design_params_dict)
        print(f"Using current detector design parameters")
    print(f"Design parameters: {design_params_dict}")
    print(f"Encoded design shape: {design.shape}")
    # Get design optimizer (will be reinitialized for physical parameters)
    design_optimizer = restored["design"]["optimizer"]

    # Reinitialize optimizer state with correct shape for physical parameters
    design_optimizer_state = design_optimizer.init(design)
    print(f"Initialized design optimizer for physical parameters (shape: {design.shape})")

    regressor_def, regressor_optimizer = (
        restored["regressor"]["model"],
        restored["regressor"]["optimizer"],
    )
    regressor_parameters, regressor_state = (
        restored["regressor"]["parameters"],
        restored["regressor"]["state"],
    )
    regressor_optimizer_state = restored["regressor"]["optimizer_state"]

    design_eps = float(config["design_eps"])

    epochs, steps, substeps = config["epochs"], config["steps"], config["substeps"]
    batch, validation_batches = config["batch"], config["validation_batches"]

    reg_coef = config.get("regularization", 1.0e-4)

    @jax.jit
    def loss_f(x, c, t, r_params, r_state):
        regressor = nnx.merge(regressor_def, r_params, r_state)
        p = regressor(x, c, deterministic=False)

        # Standard MSE loss on normalized values
        target_norm = detector.normalize_target(t)
        diff = target_norm - p
        mse = jnp.mean(jnp.square(diff), axis=-1)
        mse_loss = jnp.mean(mse)

        # Regularization
        reg_loss = reg_coef * regressor.regularization() if hasattr(regressor, "regularization") else 0.0

        loss = mse_loss + reg_loss + 1.0e-2 * jnp.mean(jnp.square(p))

        _, _, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)
        return loss, r_state

    @jax.jit
    def metric_f(x, c, t, r_params, r_state):
        regressor = nnx.merge(regressor_def, r_params, r_state)
        p = regressor(x, c, deterministic=True)

        # RMSE on normalized values
        target_norm = detector.normalize_target(t)
        rmse = jnp.sqrt(jnp.mean(jnp.square(target_norm - p), axis=-1))
        metric = jnp.mean(rmse)

        return metric

    @jax.jit
    def metric_per_component_f(x, c, t, r_params, r_state):
        """Compute RMSE for each output component (x, y, z, px, py, pz)."""
        regressor = nnx.merge(regressor_def, r_params, r_state)
        p = regressor(x, c, deterministic=True)

        # Unnormalize predictions and targets to physical units
        p_unnorm = detector.denormalize_predictions(p)

        # Per-component RMSE in physical units
        per_component_rmse = jnp.sqrt(jnp.mean(jnp.square(t - p_unnorm), axis=0))

        return per_component_rmse

    # Use separate RNG for validation to ensure different events from training
    val_rng = np.random.default_rng(seed=(seed + 999, 1))
    print("\nUsing separate RNG for validation data (different events from training)")

    @jax.jit
    def step_regressor(x, c, t, r_params, r_state, opt_state):
        (loss, r_state), grad = jax.value_and_grad(loss_f, argnums=3, has_aux=True)(x, c, t, r_params, r_state)
        updates, opt_state = regressor_optimizer.update(grad, opt_state)
        r_params = optax.apply_updates(r_params, updates)
        grad_check = jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grad)
        return loss, r_params, r_state, opt_state, grad_check

    def step_design(design, x, c, t, r_params, r_state, opt_state):
        """
        Step design for physical parameter optimization.

        Uses finite differences to approximate gradient w.r.t. design parameters.
        Not JIT-compiled since it involves Python control flow for conversions.
        """
        # Convert current design parameters to explicit design (layer-level)
        design_params = detector.decode_yaml_design(np.array(design))
        explicit_design = detector.yaml_to_layer_design(design_params)
        explicit_enc = detector.layer_design_to_array(explicit_design)
        c_base = jnp.broadcast_to(explicit_enc[None, :], (batch, explicit_enc.shape[0]))

        # Compute gradient at explicit design level
        (loss_val, _), explicit_grad = jax.value_and_grad(loss_f, argnums=1, has_aux=True)(x, c_base, t, r_params, r_state)
        explicit_grad = jnp.mean(explicit_grad, axis=0)

        # Approximate design parameter gradient using finite differences
        # For each design parameter, perturb and measure change in explicit design
        design_grad = np.zeros_like(design)
        eps = 1e-4

        for i in range(len(design)):
            design_pert = np.array(design)
            design_pert[i] += eps

            design_params_pert = detector.decode_yaml_design(design_pert)
            explicit_design_pert = detector.yaml_to_layer_design(design_params_pert)
            explicit_enc_pert = detector.layer_design_to_array(explicit_design_pert)

            # Compute derivative of explicit design w.r.t. design parameter
            d_explicit_d_design = (explicit_enc_pert - explicit_enc) / eps

            # Chain rule: grad_design = grad_explicit * d_explicit_d_design
            design_grad[i] = np.dot(np.array(explicit_grad), d_explicit_d_design)

        # Apply optimizer update
        design_grad_jax = jnp.array(design_grad)
        updates, opt_state = design_optimizer.update(design_grad_jax, opt_state, design)
        design_updated = optax.apply_updates(design, updates)

        return loss_val, design_updated, opt_state

    regressor_losses = np.ndarray(shape=(epochs, steps, substeps))
    regressor_validation = np.ndarray(shape=(epochs, validation_batches))
    regressor_validation_per_component = np.ndarray(shape=(epochs, validation_batches, 6))

    if aux is not None:
        regressor_losses[:starting_epoch] = aux["regressor"]["training"][:starting_epoch]
        regressor_validation[:starting_epoch] = aux["regressor"]["validation"][:starting_epoch]
        if "validation_per_component" in aux["regressor"]:
            regressor_validation_per_component[:starting_epoch] = aux["regressor"]["validation_per_component"][:starting_epoch]

    status = detopt.utils.progress.status_bar(disable=not progress)

    @jax.jit
    def check(params, state):
        return jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)]))

    for i in status.epochs(starting_epoch, epochs):
        epoch_start = time.time()
        step_times = []

        for j in status.training(steps):
            step_start = time.time()
            for k in range(substeps):
                # Perturb design parameters for exploration
                design_shape = detector.yaml_design_shape()
                design_batch_params = design[None, :] + design_eps * np_rng.normal(size=(batch, *design_shape)).astype(
                    np.float32
                )

                # Convert to explicit design (layer-level) for detector simulation
                design_batch_explicit = np.zeros((batch, *detector.design_shape()), dtype=np.float32)
                for b in range(batch):
                    design_params = detector.decode_yaml_design(design_batch_params[b])
                    explicit_design = detector.yaml_to_layer_design(design_params)
                    design_batch_explicit[b] = detector.layer_design_to_array(explicit_design)

                _gt, measurements, _mask, target = detector(get_seed(), design_batch_explicit, split="train")

                (
                    regressor_losses[i, j, k],
                    regressor_parameters,
                    regressor_state,
                    regressor_optimizer_state,
                    grad_check,
                ) = step_regressor(
                    measurements,
                    design_batch_explicit,
                    target,
                    regressor_parameters,
                    regressor_state,
                    regressor_optimizer_state,
                )

                if not check(regressor_parameters, regressor_state):
                    print("measurements", np.min(measurements), np.max(measurements))
                    print("target", np.min(target), np.max(target))
                    print(jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), regressor_parameters))
                    print(grad_check)
                    raise ValueError("NaN/Inf detected in regressor parameters")

            # Generate data for design gradient computation
            # Use exact current design parameters (no perturbation)
            design_params_current = detector.decode_yaml_design(design)
            explicit_design_current = detector.yaml_to_layer_design(design_params_current)
            explicit_encoded = detector.layer_design_to_array(explicit_design_current)
            design_batch_explicit = np.broadcast_to(explicit_encoded[None], shape=(batch, *detector.design_shape()))

            _gt, measurements, _mask, target = detector(get_seed(), design_batch_explicit, split="train")

            # Update design parameters
            _, design_updated, design_optimizer_state = step_design(
                design,
                measurements,
                design_batch_explicit,
                target,
                regressor_parameters,
                regressor_state,
                design_optimizer_state,
            )

            if not np.all(np.isfinite(design_updated)):
                print("original:", design)
                print("updated:", design_updated)
                print(
                    "measurements",
                    np.all(np.isfinite(measurements)),
                    np.max(measurements),
                )
                print("target", np.all(np.isfinite(target)), np.max(target))
                raise ValueError("NaN/Inf detected in design update")
            else:
                design = design_updated

            step_time = time.time() - step_start
            step_times.append(step_time)

        # Validation: use current design but with separate validation events
        for j in status.validation(validation_batches):
            # Perturb current design for validation
            design_shape = detector.yaml_design_shape()
            design_batch_params = design[None, :] + design_eps * val_rng.normal(size=(batch, *design_shape)).astype(np.float32)

            # Convert to explicit design
            design_batch_explicit = np.zeros((batch, *detector.design_shape()), dtype=np.float32)
            for b in range(batch):
                design_params = detector.decode_yaml_design(design_batch_params[b])
                explicit_design = detector.yaml_to_layer_design(design_params)
                design_batch_explicit[b] = detector.layer_design_to_array(explicit_design)

            # Generate validation data with separate seed
            _gt, measurements, _mask, target = detector(
                (seed + 999, i, j, 0),
                design_batch_explicit,
                split="val",
            )

            regressor_validation[i, j] = metric_f(
                measurements,
                design_batch_explicit,
                target,
                regressor_parameters,
                regressor_state,
            )

            # Compute per-component RMSE
            regressor_validation_per_component[i, j] = metric_per_component_f(
                measurements,
                design_batch_explicit,
                target,
                regressor_parameters,
                regressor_state,
            )

        # Print epoch summary
        train_loss_mean = np.mean(regressor_losses[i])
        train_loss_std = np.std(regressor_losses[i])
        val_loss_mean = np.mean(regressor_validation[i])
        val_loss_std = np.std(regressor_validation[i])
        epoch_time = time.time() - epoch_start
        avg_step_time = np.mean(step_times)
        first_step_time = step_times[0] if step_times else 0

        # Decode and display design parameters
        design_params_display = detector.decode_yaml_design(design)
        print(
            f"EPOCH: {i + 1}/{epochs} train_loss={train_loss_mean:.6f}±{train_loss_std:.4f} "
            f"val_loss={val_loss_mean:.6f}±{val_loss_std:.4f} "
            f"epoch_time={epoch_time:.2f}s avg_step={avg_step_time:.3f}s first_step={first_step_time:.3f}s"
        )
        print(f"Design Parameters: {design_params_display}")

        aux = {
            "regressor": {
                "training": regressor_losses[: i + 1],
                "validation": regressor_validation[: i + 1],
                "validation_per_component": regressor_validation_per_component[: i + 1],
            },
            "design_params": design_params_display,
        }

        # Save state
        detopt.utils.io.save_state(
            i,
            checkpointer,
            design=design,
            design_optimizer_state=design_optimizer_state,
            regressor_parameters=regressor_parameters,
            regressor_state=regressor_state,
            regressor_optimizer_state=regressor_optimizer_state,
            aux=aux,
        )

        if trace is not None:
            # Save design parameters
            import json

            design_params_path = os.path.join(trace, f"design_params-{i:05d}.json")
            os.makedirs(trace, exist_ok=True)
            with open(design_params_path, "w") as f:
                json.dump(design_params_display, f, indent=2)

            # Also save converted explicit design for visualization compatibility
            explicit_design = detector.yaml_to_layer_design(design_params_display)
            explicit_design_encoded = detector.layer_design_to_array(explicit_design)
            detopt.utils.io.save_design(
                detector,
                os.path.join(trace, f"design-{i:05d}.json"),
                explicit_design_encoded,
            )

        if report is not None:
            import matplotlib.pyplot as plt

            os.makedirs(report, exist_ok=True)

            fig = plot(aux)
            fig.savefig(os.path.join(report, "losses.png"))
            plt.close(fig)

    checkpointer.close()


def plot(aux):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 12))
    axes = fig.subplots(2, 1)

    detopt.utils.viz.losses.plot(aux["regressor"]["training"], axes[0])
    axes[0].set_title("Regressor Training Losses")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    detopt.utils.viz.losses.plot(aux["regressor"]["validation"], axes[1])
    axes[1].set_title("Regressor Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def report(seed, output, progress=True, restore=True, trace=None, report=None, **config):
    import matplotlib.pyplot as plt

    # Use 'output' as checkpoint directory, 'report' as output directory for plots
    checkpoint_dir = output
    output_dir = report if report is not None else "report_output"

    os.makedirs(output_dir, exist_ok=True)

    rng = jax.random.PRNGKey(seed)

    checkpointer = detopt.utils.io.get_checkpointer(checkpoint_dir)
    detector = detopt.detector.from_config(config["detector"])

    restored = detopt.utils.io.restore_state(checkpointer, detector, config, rngs=nnx.Rngs(rng), restore=True)

    aux = restored["aux"]

    fig = plot(aux)
    fig.savefig(os.path.join(output_dir, "losses.png"))
    plt.close(fig)


if __name__ == "__main__":
    import gearup

    gearup.gearup(optimize=optimize, report=report).with_config("config/subgradient.yaml")()
