#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from flax import nnx
from gpytorch.mlls import ExactMarginalLogLikelihood

import detopt


def train_and_evaluate(detector, config, design_params, seed):
    """Train regressor on design and return validation loss"""
    detector.update_from_yaml_design(design_params)
    design = detector.get_encoded_current_design()

    rngs = nnx.Rngs(seed)
    regressor = detopt.nn.from_config(detector, config=config["regressor"], rngs=rngs)
    regressor_def, r_params, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)

    optimizer = detopt.utils.config.optimizer(config["optimizer"])
    opt_state = optimizer.init(r_params)

    batch = config["batch"]
    design_array = np.tile(design.reshape(1, -1), (batch, 1))
    target_mean, target_std = detector.target_mean, detector.target_std

    def loss_fn(x, c, t, params, state):
        reg = nnx.merge(regressor_def, params, state)
        pred = reg(x, c, deterministic=True)
        target_norm = (t - target_mean) / target_std
        mse = jnp.mean(jnp.square(target_norm - pred))
        _, _, state = nnx.split(reg, nnx.Param, nnx.Variable)
        return mse, state

    @jax.jit
    def step(x, c, t, params, state, opt_state):
        (loss, state), grad = jax.value_and_grad(loss_fn, argnums=3, has_aux=True)(
            x, c, t, params, state
        )
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, state, opt_state

    # Training with early stopping
    status = detopt.utils.progress.status_bar(disable=False)

    patience = config.get("patience", 5)  # Stop after N epochs without improvement
    min_delta = config.get("min_delta", 1e-4)  # Minimum change to consider improvement
    tol_ratio = config.get("tol_ratio", 1.05)  # val/train ratio tolerance

    best_train_loss = float("inf")
    epochs_without_improvement = 0
    train_losses_history = []

    for epoch in status.epochs(config["epochs"]):
        epoch_train_losses = []
        for step_i in status.training(config["steps"]):
            _, measurements, target, _, _ = detector(
                seed=(seed, epoch, step_i, 0), configurations=design_array
            )
            loss, r_params, r_state, opt_state = step(
                measurements, design_array, target, r_params, r_state, opt_state
            )
            epoch_train_losses.append(float(loss))

        mean_train_loss = np.mean(epoch_train_losses)
        train_losses_history.append(mean_train_loss)

        # Check for plateau (no significant improvement)
        if mean_train_loss < best_train_loss - min_delta:
            best_train_loss = mean_train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping criterion 1: Training loss plateau
        if epochs_without_improvement >= patience:
            print(f"\nEarly stop: training loss plateau ({patience} epochs)")
            break

        # Check validation every few epochs
        if (epoch + 1) % max(1, config["epochs"] // 5) == 0 or epoch == config[
            "epochs"
        ] - 1:
            val_losses_check = []
            for i in range(config.get("validation_batches", 10)):
                _, measurements, target, _, _ = detector(
                    seed=(seed + 1000, i, 1), configurations=design_array
                )
                reg = nnx.merge(regressor_def, r_params, r_state)
                pred = reg(measurements, jnp.array(design_array), deterministic=True)
                target_norm = (target - target_mean) / target_std
                rmse = jnp.sqrt(jnp.mean(jnp.square(target_norm - pred), axis=-1))
                val_losses_check.append(float(jnp.mean(rmse)))

            mean_val_loss = np.mean(val_losses_check)

            # Early stopping criterion 2: val_loss ≈ train_loss
            if mean_val_loss <= mean_train_loss * tol_ratio:
                print(
                    f"\nEarly stop: val_loss ≈ train_loss ({mean_val_loss:.6f} ≈ {mean_train_loss:.6f})"
                )
                break

    # Final validation
    val_losses = []
    for i in status.validation(config.get("validation_batches", 10)):
        _, measurements, target, _, _ = detector(
            seed=(seed + 1000, i, 1), configurations=design_array
        )
        reg = nnx.merge(regressor_def, r_params, r_state)
        pred = reg(measurements, jnp.array(design_array), deterministic=True)
        target_norm = (target - target_mean) / target_std
        rmse = jnp.sqrt(jnp.mean(jnp.square(target_norm - pred), axis=-1))
        val_losses.append(float(jnp.mean(rmse)))

    return np.mean(val_losses)


def optimize(config, output_dir, n_iter=50, n_init=10, seed=42):
    """Run Bayesian optimization using BoTorch"""
    detector = detopt.detector.from_config(config["detector"])

    # Define parameter bounds
    bounds = torch.tensor(
        [
            [8300, 8500, 9200, 9400, 0.05, 1.0, 3.0, 0.0002, 8800, 200],  # Lower
            [8500, 8700, 9400, 9600, 0.12, 3.0, 7.0, 0.001, 9100, 400],  # Upper
        ],
        dtype=torch.float64,
    )

    # Storage
    train_X = torch.empty((0, bounds.shape[1]), dtype=torch.float64)
    train_Y = torch.empty((0, 1), dtype=torch.float64)
    results = []
    best_obj, best_design = -np.inf, None

    print(f"\nBayesian Optimization with BoTorch: {n_iter} iterations")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    for i in range(n_iter):
        iter_start = time.time()

        # Propose design
        if i < n_init:
            # Random initialization
            params = (
                torch.rand(1, bounds.shape[1], dtype=torch.float64)
                * (bounds[1] - bounds[0])
                + bounds[0]
            )
        else:
            # BoTorch: Fit GP and optimize acquisition
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Expected Improvement
            EI = ExpectedImprovement(gp, best_f=train_Y.max())

            # Optimize acquisition function
            params, acq_value = optimize_acqf(
                EI,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
            )

        # Convert to design dict
        params_np = params.squeeze().numpy()
        stereo_angle = float(params_np[4])
        design = {
            "station_z": params_np[:4].tolist(),
            "view_angles": [0.0, stereo_angle, -stereo_angle, 0.0],
            "layer_z_gap": float(params_np[5]),
            "view_z_gap": float(params_np[6]),
            "max_B": float(params_np[7]),
            "z0": float(params_np[8]),
            "B_sigma": float(params_np[9]),
        }

        # Evaluate
        print(f"\n[Iteration {i + 1}/{n_iter}] Evaluating design...")
        val_loss = train_and_evaluate(detector, config, design, seed + i)
        objective = -val_loss  # Higher is better

        # Update BoTorch data
        train_X = torch.cat([train_X, params])
        train_Y = torch.cat([train_Y, torch.tensor([[objective]], dtype=torch.float64)])

        iter_time = time.time() - iter_start

        if objective > best_obj:
            best_obj, best_design = objective, design
            print(
                f"✓ Iter {i + 1}/{n_iter} | Loss: {val_loss:.6f} | Time: {iter_time:.1f}s | BEST ★"
            )
        else:
            print(
                f"✓ Iter {i + 1}/{n_iter} | Loss: {val_loss:.6f} | Time: {iter_time:.1f}s"
            )

        results.append(
            {"iteration": i, "design": design, "objective": float(objective)}
        )

        # Save intermediate checkpoint
        checkpoint_path = f"{output_dir}/checkpoint_iter_{i:03d}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(
                {
                    "iteration": i,
                    "design": design,
                    "objective": float(objective),
                    "val_loss": float(val_loss),
                    "time": iter_time,
                },
                f,
                indent=2,
            )

        # Save cumulative results
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "best_objective": float(best_obj),
                    "best_design": best_design,
                    "n_iterations_completed": i + 1,
                    "method": "BoTorch",
                },
                f,
                indent=2,
            )

        print(f"Saved: {checkpoint_path}")
        print("-" * 80)

    # Final summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest objective: {best_obj:.6f} (validation loss: {-best_obj:.6f})")
    print(f"Improvement: {best_obj - results[0]['objective']:.6f}")
    print(f"\nBest design:")
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

    import os

    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    optimize(config, args.output, args.n_iterations, args.n_initial, args.seed)
