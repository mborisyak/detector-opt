#!/usr/bin/env python3
"""Bayesian optimisation of the SST detector design.

Thin driver around two reusable wrappers:

* :class:`detopt.bo.BayesianOptimizer` -- the outer GP+EI loop. It owns
  X-normalisation (per-dimension bounds -> unit cube), so the GP/acquisition code
  stays bounds-agnostic. BO searches the *encoded* (unconstrained N(0,1)) design
  space, treated as the wrapper's "nominal" space with per-dimension bounds
  ``[-bound, bound]``.
* :class:`detopt.nn.trainer.DesignTrainer` -- per-design network training. It
  builds JIT train/eval kernels once, preallocates fixed-size GPU event pools,
  runs the data-growing convergence procedure (see its docstring), and
  checkpoints each epoch. The detector-call budget is shared across all designs.

The objective is the converged loss ``(mean_train + mean_val) / 2`` (BO minimises
it directly); each observation's GP noise is the loss estimate's own SEM.
"""

import json
import os
import time

import matplotlib

matplotlib.use("AGG")  # before any pyplot import (detopt.utils.viz pulls it in)

import numpy as np

import detopt
from detopt.bo import BayesianOptimizer
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.utils.viz.bo import plot_iteration, plot_convergence

# from_scratch / continue / closest all use DesignTrainer (a fresh per-design
# network; "continue" and "closest" warm-start its weights from a previous design).
# "meta" is the ContinualTrainer: one persistent network trained with current+history
# replay. All are design-conditioned (combine sees each event's real encoded design).
VALID_INIT_STRATEGIES = ("from_scratch", "continue", "closest", "meta")


def bo(output, seed: int, **config):
    seed = int(seed)
    nn_init_strategy = config.get("nn_init_strategy", "from_scratch")
    if nn_init_strategy not in VALID_INIT_STRATEGIES:
        raise ValueError(f"nn_init_strategy {nn_init_strategy!r} not in {VALID_INIT_STRATEGIES}")

    os.makedirs(output, exist_ok=True)
    plots_dir = os.path.join(output, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    bo_cfg = config["bo"]
    gp_cfg = dict(bo_cfg["gp"])
    ei_cfg = dict(bo_cfg["ei"])
    bound = float(bo_cfg.get("bound", 3.0))
    # Initial random proposals: as many as the GP's CV folds, so the first GP
    # fit has enough points for k-fold cross-validation.
    n_init = int(bo_cfg.get("n_init", gp_cfg["n_folds"]))

    detector = detopt.detector.from_config(config["detector"])
    # BO searches the encoded (unconstrained N(0,1)) design space; per-dimension
    # bounds [-bound, bound] cover ~99.7% of N(0,1). The wrapper maps these to the
    # unit cube where the GP + EI run.
    d = int(detector.design_dim())
    bounds = np.stack([np.full(d, -bound), np.full(d, bound)], axis=1)
    bo_opt = BayesianOptimizer(bounds, gp=gp_cfg, ei=ei_cfg, n_init=n_init, seed=seed)

    trainer_cls = ContinualTrainer if nn_init_strategy == "meta" else DesignTrainer
    trainer = trainer_cls.from_config(
        detector,
        config,
        checkpoint_dir=os.path.join(output, "checkpoints"),
        seed=seed,
    )

    # The trainer owns the budget-sized event pools (train + val); the run ends
    # when they fill. Designs append into them (windowed) across iterations.
    budget = trainer.train_pool.capacity + trainer.val_pool.capacity
    root_seq = np.random.SeedSequence(int(seed))  # per-design RNG via spawn

    proposed_encoded = []  # row-aligned with trained_params (warm-start)
    trained_params = []
    results = []
    best_loss, best_design = np.inf, None

    print(f"BO: running until the budget pool fills " f"(budget={budget} detector calls, n_init={n_init}, d={d})")

    i = 0
    while True:
        iter_start = time.time()
        x_prop = np.asarray(bo_opt.propose(), dtype=np.float32)
        if bo_opt.last_info is not None:
            info = bo_opt.last_info
            print(
                f"  [acq] EI={info['ei']:.4g} "
                f"| log_ls~{info['log_lengthscale_mean']:.3f} "
                f"log_amp={info['log_amplitude']:.3f}"
            )

        design_phys = np.asarray(detector.flatten_design(detector.decode_design(x_prop)), dtype=np.float32).tolist()

        def _on_epoch(snapshot, _i=i, _d=design_phys):
            vlp = snapshot["val_loss_per_epoch"]
            live = float(vlp[-1]) if vlp.size > 0 else float("nan")
            plot_iteration(snapshot, iteration=_i, design=_d, val_loss=live, plots_dir=plots_dir)

        # Network init strategy (todo.md): from_scratch trains fresh; continue
        # warm-starts from the previous design; closest from the nearest (encoded
        # L2) previously trained design.
        # Warm-start applies only to the per-design DesignTrainer strategies; the
        # "meta" ContinualTrainer carries its own persistent network.
        init_params = None
        warm_from = None
        if len(trained_params) > 0 and nn_init_strategy in ("continue", "closest"):
            if nn_init_strategy == "continue":
                warm_from = len(trained_params) - 1
            else:  # closest
                dists = np.linalg.norm(np.asarray(proposed_encoded) - x_prop[None, :], axis=1)
                warm_from = int(np.argmin(dists))
                print(f"  [warm-start] closest = iter {warm_from} (dist={float(dists[warm_from]):.3f})")
            init_params = trained_params[warm_from]

        used = trainer.train_pool.current + trainer.val_pool.current
        print(f"[iter {i+1}] training... ({budget - used} detector calls left)")
        result = trainer.train(
            x_prop,
            root_seq.spawn(1)[0],
            init_params=init_params,
            on_epoch=_on_epoch,
            step=i,
        )
        if result is None:
            print(f"[budget] pool exhausted; finishing BO after {i} completed iterations.")
            break
        loss = float(result.objective_loss)  # BO minimises the loss directly

        bo_opt.append(x_prop, loss, noise=result.objective_std)
        proposed_encoded.append(x_prop)
        trained_params.append(result.params)

        improved = loss < best_loss
        if improved:
            best_loss, best_design = loss, design_phys

        elapsed = time.time() - iter_start
        marker = " BEST" if improved else ""
        print(
            f"[iter {i+1}] loss={result.objective_loss:.5f}±{result.objective_std:.4f} "
            f"spent={result.spent} time={elapsed:.1f}s{marker}"
        )

        results.append(
            {
                "iteration": i,
                "design": design_phys,
                "x_encoded": x_prop.tolist(),
                "loss": float(result.objective_loss),
                "loss_std": float(result.objective_std),
                "spent": int(result.spent),
                "time_s": float(elapsed),
                "nn_init_strategy": nn_init_strategy,
                "warm_start_from": warm_from,
            }
        )
        with open(os.path.join(output, "results.json"), "w") as f:
            json.dump(
                {
                    "results": results,
                    "best_loss": float(best_loss),
                    "best_design": best_design,
                    "n_iterations_completed": i + 1,
                    "detector_calls_used": int(trainer.train_pool.current + trainer.val_pool.current),
                    "method": "JAX-GP+EI",
                },
                f,
                indent=2,
                default=float,
            )

        # Refresh the convergence plot after every completed iteration.
        plot_convergence(results, output)
        i += 1

    plot_convergence(results, output)
    print(f"\nBest loss: {best_loss:.6f}")
    return best_loss, best_design, results


if __name__ == "__main__":
    import gearup

    gearup.gearup(bo).with_config("config/bo.yaml")()
