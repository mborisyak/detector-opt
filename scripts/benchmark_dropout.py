#!/usr/bin/env python3
"""Does dropout help convergence (data-to-precision) for the ENSEMBLE regressor?

Repeats the dropout convergence experiment, but with an ensemble
:class:`detopt.nn.SetRegressor` (``n_models`` independently-trained
members) in place of the single set regressor. One design is trained with the
real :class:`detopt.nn.trainer.DesignTrainer` + the actual stopping criterion,
varying only the dropout rate, and we report how much data / how many epochs each
needs to estimate the loss to precision.

Each member trains on its own independent minibatch drawn from the same window
(no extra detector calls); evaluation averages the members' predictions. Dropout
regularises the train/val gap; the stopping rule grows data while
``|val - train| + err >= precision``, so *less overfitting -> reach precision with
less data* is the hypothesis. A design that can't reach precision within
``iteration_limit`` is reported as CAPACITY (the loud-crash case); a run that
exhausts the shared budget is reported as ``budget``.
"""

import numpy as np
import optax

from detopt.detector import Stereo4Feature
from detopt.nn.trainer import DesignTrainer

DROPOUTS = [0.0, 0.1, 0.3]


def run(seed=0, n_models=4, budget=400_000, iteration_limit=16384, loss_precision=2.0e-2):
    det = Stereo4Feature(engine="simplified")
    # Nominal design: 0.5 is the midpoint of every bound in the SCALED cube (zeros, which meant
    # the midpoint under the old encoding, are now its LOWER CORNER).
    design = np.full(det.design_dim(), 0.5, dtype=np.float32)
    rows = []
    for p in DROPOUTS:
        trainer = DesignTrainer(
            det,
            regressor_config={
                "set-regressor": {
                    "features": [[64, 64], [64, 32]],
                    "n_models": n_models,
                    "p_dropout": p,
                }
            },
            optimizer=optax.adamw(learning_rate=1e-3, weight_decay=1e-3),
            batch=128,
            n0=2048,
            n_increment=1024,
            iteration_limit=iteration_limit,
            warmup_epochs=10,
            patience=10,
            loss_precision=loss_precision,
            budget=budget,
            val_fraction=0.2,
            eval_batch=2048,
            device=None,
            seed=seed,
        )
        epochs = [0]
        try:
            result = trainer.train(
                design,
                seed,
                on_epoch=lambda s, e=epochs: e.__setitem__(0, e[0] + 1),
            )
            if result is None:
                rows.append((p, "budget", epochs[0], 0, float("nan"), float("nan")))
            else:
                train_data = round(result.spent / 1.2)  # spent = train + 0.2/0.8 * train (val_fraction=0.2)
                rows.append((p, "yes", epochs[0], train_data, result.objective_loss, result.objective_std))
        except RuntimeError:
            rows.append((p, "CAPACITY", epochs[0], iteration_limit, float("nan"), float("nan")))

    print(
        f"\nDropout vs convergence (debug detector, ensemble n_models={n_models}, nominal design, "
        f"seed={seed}, precision={loss_precision}, iteration_limit={iteration_limit})"
    )
    print(f"{'p_dropout':<10} {'converged':<10} {'epochs':>7} {'train_data':>11} {'loss':>9} {'est_sem':>9}")
    for p, conv, ep, data, loss, sem in rows:
        print(f"{p:<10} {conv:<10} {ep:>7} {data:>11} {loss:>9.4f} {sem:>9.4f}")


if __name__ == "__main__":
    run()
