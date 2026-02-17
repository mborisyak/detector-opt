import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import detopt
from detopt.detector.straw import SparseHits


def regress(
    seed, output, progress=False, restore=True, trace=None, report=None, **config
):
    print(config)
    print(f"Report parameter: {report}")

    rngs = nnx.Rngs(seed)
    np_rng = np.random.default_rng(seed=(seed, 0))

    detector = detopt.detector.from_config(config["detector"])
    print(detector.get_current_design())
    enc = detector.get_encoded_current_design()
    enc = enc.reshape(1, -1)
    print(enc)
    # input("Waiting ")
    print(detector.get_design(enc))
    # input("Waiting ")
    print(detector.p_spawn_single)
    regressor = detopt.nn.from_config(detector, config=config["regressor"], rngs=rngs)
    optimizer = nnx.Optimizer(
        regressor, detopt.utils.config.optimizer(config["optimizer"]), wrt=nnx.Param
    )
    # input("Waiting 1")
    epochs, steps = config["epochs"], config["steps"]
    print(epochs, steps)
    batch, validation_batches = config["batch"], config["validation_batches"]
    # input("Waiting 2")
    design = np.tile(enc, (batch, 1))
    # print(design)
    # print(design.shape)
    # input("wait design")

    @nnx.jit
    def loss_f(model, x, c, t):
        p = model(x, c)
        return jnp.mean(detector.loss(t, p))

    @nnx.jit
    def metric_f(model, x, c, t):
        p = model(x, c)
        return jnp.mean(detector.metric(t, p))

    @nnx.jit
    def step(model, optimizer, x, c, t):
        # x is a dictionary for sparse data, don't convert to array
        c, t = jnp.array(c), jnp.array(t)
        loss, grad = nnx.value_and_grad(loss_f, argnums=0)(model, x, c, t)
        optimizer.update(model, grad)
        return loss

    training_losses = np.ndarray(shape=(epochs, steps))
    validation_losses = np.ndarray(shape=(epochs, validation_batches))

    # Track predictions and targets for precision analysis
    val_predictions = []
    val_targets = []

    status = detopt.utils.progress.status_bar(disable=not progress)
    # input("Waiting start")
    for i in status.epochs(epochs):
        for j in status.training(steps):
            _, measurements, target = detector(
                seed=(seed, 1, 1, 0), configurations=design
            )
            print(measurements[0])
            # input("Waiting for step")
            training_losses[i, j] = step(
                regressor, optimizer, measurements, design, target
            )

        for j in status.validation(validation_batches):
            design = np_rng.normal(size=(batch, *detector.design_shape())).astype(
                np.float32
            )
            _, measurements, target = detector(
                seed=(seed, 1, 1, 1), configurations=design
            )

            # Get predictions for precision analysis (normalized outputs)
            predictions_norm = regressor(measurements, jnp.array(design))

            # Denormalize predictions to raw units for visualization
            predictions = predictions_norm * detector.target_std + detector.target_mean

            validation_losses[i, j] = metric_f(regressor, measurements, design, target)

            # Debug: print first batch of first validation to check values
            if True:
                print(f"\nDEBUG Epoch {i + 1}:")
                print(f"  Target[0]: {target[0]}")
                print(f"  Predicted[0] : {predictions_norm[0]}")
                print(f"  Predicted[0] (denormalized): {predictions[0]}")
                print(f"  Diff: {np.abs(predictions[0] - target[0])}")
                t = target[0]  # physical
                mu = detector.target_mean  # shape (6,)
                sd = detector.target_std  # shape (6,)

                t_norm = (t - mu) / sd  # normalized target
                p_norm = predictions_norm[0]  # normalized prediction

                print("Target_norm[0]:", t_norm)
                print("Pred_norm[0]:", p_norm)
                print("Diff_norm:", t_norm - p_norm)

            # Store predictions and targets for this epoch
            if j == 0:  # Store only first validation batch per epoch
                val_predictions.append(np.array(predictions))
                val_targets.append(np.array(target))

        # Print epoch summary (greppable format)
        train_loss_mean = np.mean(training_losses[i])
        val_loss_mean = np.mean(validation_losses[i])
        train_loss_std = np.std(training_losses[i])
        val_loss_std = np.std(validation_losses[i])
        print(
            f"EPOCH: {i + 1}/{epochs} train_loss={train_loss_mean:.6f}±{train_loss_std:.4f} val_loss={val_loss_mean:.6f}±{val_loss_std:.4f}"
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
                pos_rmse = np.sqrt(
                    np.mean(pos_errors**2, axis=(1, 2))
                )  # RMSE per epoch

                # Momentum errors (px, py, pz) - last 3 components
                mom_errors = pred_array[:, :, 3:] - targ_array[:, :, 3:]
                mom_rmse = np.sqrt(
                    np.mean(mom_errors**2, axis=(1, 2))
                )  # RMSE per epoch

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
            # Update the same file each epoch
            fig.savefig(report)
            plt.close(fig)
            # Silently update plot (no print to keep output clean)

    # Save model parameters and state
    import os

    import orbax.checkpoint as ocp

    os.makedirs(output, exist_ok=True)
    manager = ocp.CheckpointManager(
        output,
        checkpointers=ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )

    _, parameters, state = nnx.split(regressor, nnx.Param, nnx.Variable)
    parameters = nnx.to_pure_dict(parameters)
    state = nnx.to_pure_dict(state)

    # Create fresh optimizer state (can be reinitialized on load)
    optax_optimizer = detopt.utils.config.optimizer(config["optimizer"])
    optimizer_state = optax_optimizer.init(parameters)

    manager.save(
        0,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeSave(
                detopt.utils.io.save_model(parameters, state, optimizer_state)
            )
        ),
    )


if __name__ == "__main__":
    import gearup

    gearup.gearup(regress=regress).with_config("config/regress.yaml")()
