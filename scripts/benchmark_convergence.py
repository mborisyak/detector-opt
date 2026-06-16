#!/usr/bin/env python3
"""Validate the epoch-stopping criterion (with and without the power-law safeguard)
against +100 more epochs, on a FIXED dataset.

Everything here runs *outside* the trainer's data-growing loop: we draw a single
fixed dataset of ``10 * n0`` train events (plus the matching val split), then run
ONE continuous training run -- Adam state is carried the whole way, never reset --
recording the train/val loss curve. We then locate two stop points on that single
curve:

* ``base``  -- the plateau signal alone (``is_plateaued`` on the train loss);
* ``safe``  -- plateau AND the power-law floor predicts < ``loss_precision`` of
  further descent (``extrapolated_floor``).

and compare each stop's objective ``(train+val)/2`` to the objective 100 epochs
later in the SAME run. A sound criterion should leave little on the table; the
safeguard should stop later than ``base`` and leave even less.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from detopt.detector.debug import DebugDetector
from detopt.nn.trainer import DesignTrainer
from detopt.utils.training import extrapolated_floor, is_plateaued, masked_mean_sem

LOOKAHEAD = 100


def _train_curve(seed, n0, max_epochs):
    """One continuous run on a fixed 10*n0 dataset; returns the per-epoch curve."""
    det = DebugDetector()
    trainer = DesignTrainer(
        det,
        regressor_config={"set-regressor": {"features": [[64, 64], [64, 32]], "p_dropout": 0.1}},
        optimizer=optax.adamw(learning_rate=1e-3, weight_decay=1e-3),
        batch=128,
        n0=n0,
        n_increment=1024,
        iteration_limit=10 * n0,  # one epoch = one pass over the fixed train set
        warmup_epochs=10,
        patience=10,
        flatness_tol=1e-2,
        loss_precision=2e-2,
        budget=16 * n0,  # room for 10*n0 train + its val split
        val_fraction=0.2,
        eval_batch=2048,
        seed=seed,
    )
    design_enc = np.zeros(det.design_dim(), dtype=np.float32)
    design_phys = np.asarray(det.flatten_design(det.decode_design(design_enc)), dtype=np.float32)

    init_seq, train_seq, val_seq, run_seq = np.random.SeedSequence(seed).spawn(4)
    tp, vp = trainer.train_pool, trainer.val_pool
    trainer._fill_pool(design_phys, design_enc, tp, "train", 10 * n0, train_seq)
    trainer._fill_pool(design_phys, design_enc, vp, "val", round(10 * n0 * 0.25), val_seq)
    tcount, vcount = int(tp.n_current), int(vp.n_current)

    params, state, opt_state = trainer._init_design_network(init_seq, None, None)
    zero = jnp.int32(0)
    tc = jnp.int32(tcount)
    key = jax.random.PRNGKey(int(run_seq.generate_state(1)[0]))

    train_h, val_h, tsem_h, vsem_h = [], [], [], []
    for _ in range(max_epochs):
        key, sk = jax.random.split(key)
        # Adam state (opt_state) is threaded through every epoch -- never reset.
        params, state, opt_state, _ = trainer._train_epoch(params, state, opt_state, sk, zero, tc, tp.buffers())
        tr = trainer._eval_train(params, state, tp.buffers(), zero)
        va = trainer._eval_val(params, state, vp.buffers(), zero)
        tm, ts = masked_mean_sem(tr, tcount)
        vm, vs = masked_mean_sem(va, vcount)
        train_h.append(float(tm))
        val_h.append(float(vm))
        tsem_h.append(float(ts))
        vsem_h.append(float(vs))
    return trainer, np.array(train_h), np.array(val_h), np.array(tsem_h), np.array(vsem_h), tcount


def _first_stop(train_h, obj, patience, flat, prec, warmup, with_safeguard):
    """First epoch (1-indexed) at which the (safeguarded?) plateau stop fires."""
    for t in range(warmup + 1, len(train_h) + 1):
        if not is_plateaued(train_h[:t], patience, flat, prec):
            continue
        if with_safeguard:
            remaining = obj[t - 1] - extrapolated_floor(obj[:t])
            if remaining >= prec:
                continue
        return t
    return None


def run(seeds=(0, 1, 2), n0=2048, max_epochs=400):
    rows = []
    for seed in seeds:
        trainer, train_h, val_h, tsem, vsem, tcount = _train_curve(seed, n0, max_epochs)
        prec, patience, flat, warmup = (
            trainer.loss_precision,
            trainer.patience,
            trainer.flatness_tol,
            trainer.warmup_epochs,
        )
        obj = 0.5 * (train_h + val_h)

        print(f"\n[seed {seed}] fixed dataset = {tcount} train events, {max_epochs} epochs, prec={prec}")
        print(
            f"  {'stop':>10} {'epoch':>6} {'train':>8} {'val':>8} {'obj':>8} {'floor':>8} {'remain':>8} {'obj+100':>8} {'d_obj':>8}"
        )
        for name, safeguard in (("base", False), ("safe", True)):
            t = _first_stop(train_h, obj, patience, flat, prec, warmup, safeguard)
            if t is None:
                print(f"  {name:>10}    never triggered within {max_epochs} epochs")
                continue
            floor = extrapolated_floor(obj[:t])
            remaining = obj[t - 1] - floor
            t2 = min(t + LOOKAHEAD, len(obj))
            d = obj[t2 - 1] - obj[t - 1]  # negative = loss kept dropping after the stop
            tag = f"+{t2 - t}" if t2 - t < LOOKAHEAD else "+100"
            print(
                f"  {name:>10} {t:>6} {train_h[t-1]:>8.4f} {val_h[t-1]:>8.4f} {obj[t-1]:>8.4f} "
                f"{floor:>8.4f} {remaining:>8.4f} {obj[t2-1]:>8.4f} {d:>8.4f}  ({tag})"
            )
            rows.append((name, d))

    if rows:
        print(
            "\nMean objective change over +100 epochs after the stop "
            "(negative = loss was still dropping -> stopped too early):"
        )
        for name in ("base", "safe"):
            ds = np.array([d for n, d in rows if n == name])
            if ds.size:
                print(f"  {name:>5}: {ds.mean():+.4f} ± {ds.std():.4f}  (|.| should be << prec if sound)")


if __name__ == "__main__":
    run()
