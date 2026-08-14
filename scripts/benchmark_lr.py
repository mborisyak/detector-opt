#!/usr/bin/env python3
"""Quick comparison of learning-rate schedules under our real stopping criterion.

Trains ONE design (the debug detector, nominal design at the scaled midpoint) with the actual
:class:`detopt.nn.trainer.DesignTrainer` -- same convergence procedure, same
config numbers -- under three schedules, and reports how much data / how many
epochs each needs to estimate the loss to precision:

* constant  -- fixed lr (the config baseline);
* seesaw    -- SGDR warm restarts: cosine-anneal peak->min over a period, then
               JUMP abruptly back to peak (discontinuous restart);
* gradual   -- smooth cosine cycle peak->min->peak->... (no discontinuity).

All three share the seed (same init + same event stream), so differences are
attributable to the schedule alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from detopt.detector import Stereo4Feature
from detopt.nn.trainer import DesignTrainer

PEAK, FLOOR = 1.0e-3, 1.0e-5
PERIOD_EPOCHS = 5  # one schedule cycle, in epochs


def make_schedules(steps_per_epoch):
    period = PERIOD_EPOCHS * steps_per_epoch

    def constant(step):
        return PEAK

    def seesaw(step):  # SGDR: anneal peak->floor over `period`, then restart at peak
        frac = (step % period) / period
        return FLOOR + 0.5 * (PEAK - FLOOR) * (1.0 + jnp.cos(jnp.pi * frac))

    def gradual(step):  # smooth full cosine cycle peak->floor->peak
        return FLOOR + 0.5 * (PEAK - FLOOR) * (1.0 + jnp.cos(2.0 * jnp.pi * step / period))

    return {"constant": constant, "seesaw": seesaw, "gradual": gradual}


def run(seed=0, n_max_cap=16384):
    device = jax.devices("cuda")[0]
    det = Stereo4Feature(engine="simplified")
    # Config numbers (config/bo_debug.yaml), bounded n_max_cap for a quick test.
    common = dict(
        regressor_config={"set-regressor": {"features": [[64, 64], [64, 32]], "p_dropout": 0.1}},
        batch=256,
        n0=2048,
        n_increment=1024,
        n_max_cap=n_max_cap,
        warmup_epochs=10,
        patience=10,
        loss_precision=1e-2,
        val_fraction=0.2,
        device=device,
        seed=seed,
    )
    steps_per_epoch = n_max_cap // common["batch"]
    schedules = make_schedules(steps_per_epoch)
    design = np.full(det.design_dim(), 0.5, dtype=np.float32)  # nominal design = the scaled midpoint

    rows = []
    for name, sched in schedules.items():
        trainer = DesignTrainer(det, optimizer=optax.adamw(learning_rate=sched, weight_decay=1e-3), **common)
        epochs = [0]
        try:
            result = trainer.train(
                design,
                seed,
                remaining=400_000,
                on_epoch=lambda s, e=epochs: e.__setitem__(0, e[0] + 1),
            )
            if result is None:
                rows.append((name, "budget", epochs[0], 0, float("nan"), float("nan")))
            else:
                train_data = round(result.spent / 1.25)  # spent = train + 0.25*train
                rows.append((name, "yes", epochs[0], train_data, result.objective_loss, result.objective_std))
        except RuntimeError:
            rows.append((name, "CAPACITY", epochs[0], n_max_cap, float("nan"), float("nan")))

    print(f"\nLR schedule comparison (debug detector, nominal design, seed={seed}, " f"precision=1e-2, n_max_cap={n_max_cap})")
    print(f"{'schedule':<10} {'converged':<10} {'epochs':>7} {'train_data':>11} " f"{'loss':>9} {'est_sem':>9}")
    for name, conv, ep, data, loss, sem in rows:
        print(f"{name:<10} {conv:<10} {ep:>7} {data:>11} {loss:>9.4f} {sem:>9.4f}")


if __name__ == "__main__":
    run()
