#!/usr/bin/env python3
"""Bayesian Optimization for Detector Geometry"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from scipy.optimize import minimize
from scipy.stats import norm

import detopt


class BayesianOptimizer:
    def __init__(self, bounds, noise=0.01, xi=0.01, length_scale=1.0):
        self.bounds = np.array(bounds)
        self.noise = noise
        self.xi = xi
        self.length_scale = length_scale
        self.X = []
        self.y = []

    def kernel(self, X1, X2):
        """RBF kernel"""
        X1_n = (X1 - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        X2_n = (X2 - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        sqdist = (
            np.sum(X1_n**2, 1).reshape(-1, 1) + np.sum(X2_n**2, 1) - 2 * X1_n @ X2_n.T
        )
        return np.exp(-0.5 * sqdist / self.length_scale**2)

    def predict(self, X_test):
        if len(self.X) == 0:
            return np.zeros(len(X_test)), np.ones(len(X_test))

        X_train = np.array(self.X)
        y_train = np.array(self.y)
        K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        K_s = self.kernel(X_train, X_test)
        K_ss = self.kernel(X_test, X_test)
        K_inv = np.linalg.inv(K)
        mu = K_s.T @ K_inv @ y_train
        sigma = np.sqrt(np.maximum(np.diag(K_ss - K_s.T @ K_inv @ K_s), 1e-10))
        return mu, sigma

    def acquisition(self, X):
        """Expected Improvement"""
        if len(self.y) == 0:
            return np.ones(len(X))
        mu, sigma = self.predict(X)
        y_best = np.max(self.y)
        Z = (mu - y_best - self.xi) / sigma
        ei = (mu - y_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = 0
        return ei

    def propose(self):
        best_x, best_acq = None, -np.inf
        for _ in range(10):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            res = minimize(
                lambda x: -self.acquisition(x.reshape(1, -1))[0],
                x0,
                bounds=self.bounds,
                method="L-BFGS-B",
            )
            if -res.fun > best_acq:
                best_acq, best_x = -res.fun, res.x
        return best_x

    def add(self, x, y):
        """Add observation to GP"""
        self.X.append(x.copy())
        self.y.append(y)


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

    # Training with progress bar
    status = detopt.utils.progress.status_bar(disable=False)
    for epoch in status.epochs(config["epochs"]):
        for step_i in status.training(config["steps"]):
            _, measurements, target, _, _ = detector(
                seed=(seed, epoch, step_i, 0), configurations=design_array
            )
            loss, r_params, r_state, opt_state = step(
                measurements, design_array, target, r_params, r_state, opt_state
            )

    # Validation with progress bar
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


def optimize(config, output_dir, n_iter=50, n_init=10, seed=42, length_scale=1.0):
    """Run Bayesian optimization"""
    detector = detopt.detector.from_config(config["detector"])

    # Define parameter bounds
    bounds = [
        (8300, 8500),
        (8500, 8700),
        (9200, 9400),
        (9400, 9600),  # station_z
        (0.05, 0.12),  # stereo_angle
        (1.0, 3.0),
        (3.0, 7.0),  # gaps
        (0.0002, 0.001),
        (8800, 9100),
        (200, 400),  # B-field
    ]

    bo = BayesianOptimizer(bounds, length_scale=length_scale)
    results = []
    best_obj, best_design = -np.inf, None

    print(f"\nBayesian Optimization: {n_iter} iterations")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    for i in range(n_iter):
        iter_start = time.time()
        # Propose design
        if i < n_init:
            params = np.random.uniform(bo.bounds[:, 0], bo.bounds[:, 1])
        else:
            params = bo.propose()

        # Convert params to design dict with view_angles computed from stereo_angle
        stereo_angle = float(params[4])
        design = {
            "station_z": params[:4].tolist(),
            "view_angles": [0.0, stereo_angle, -stereo_angle, 0.0],
            "layer_z_gap": float(params[5]),
            "view_z_gap": float(params[6]),
            "max_B": float(params[7]),
            "z0": float(params[8]),
            "B_sigma": float(params[9]),
        }

        # Evaluate
        print(f"\n[Iteration {i + 1}/{n_iter}] Evaluating design...")
        val_loss = train_and_evaluate(detector, config, design, seed + i)
        objective = -val_loss  # Higher is better
        bo.add(params, objective)

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

        # Save cumulative results every iteration
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "best_objective": float(best_obj),
                    "best_design": best_design,
                    "n_iterations_completed": i + 1,
                    "length_scale": length_scale,
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
    print(f"Improvement: {best_obj - (-results[0]['objective']):.6f}")
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
    parser.add_argument(
        "--length-scale", type=float, default=1.0, help="GP kernel length scale"
    )
    args = parser.parse_args()

    import os

    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    optimize(
        config,
        args.output,
        args.n_iterations,
        args.n_initial,
        args.seed,
        args.length_scale,
    )
