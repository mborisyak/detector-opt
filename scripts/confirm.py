#!/usr/bin/env python3
"""Check a BO-optimised design two ways (gearup subcommands; both use config/bo.yaml).

``confirm`` -- retrain from scratch on the full budget: pick the best design in a
    run's ``results.json``, sample the entire detector-call budget under it, and
    train a **freshly initialised** regressor for a fixed number of epochs
    (single-cycle cosine LR). An independent estimate of the design's loss, free of
    warm-start / shared-net / data-growing.

``evaluate`` -- load a saved network and score it on fresh data, **no training**:
    restore a regressor from a training checkpoint (e.g. the meta run's
    ``checkpoints/design_<i>``) plus the design it was trained for, and evaluate it
    on a full budget of freshly sampled events at that design. Checks whether a
    reported per-design loss is a genuine held-out number.

Run::

    python scripts/confirm.py confirm results=output/compare/meta/results.json max_epochs=200
    python scripts/confirm.py evaluate checkpoint=output/compare/meta/checkpoints/design_0070

``confirm``'s epoch is a full pass over the budget (heavy) -- lower ``max_epochs``
(or ``training.budget``) for a quicker check.
"""

import json
import os

import numpy as np

import detopt
from detopt.nn.trainer import FullBudgetTrainer
from detopt.utils.viz.bo import plot_confirmation


def confirm(results, max_epochs: int = 100, seed: int = 0, **config):
    seed, max_epochs = int(seed), int(max_epochs)
    with open(results) as f:
        run = json.load(f)

    # Best design = the iteration with the lowest reported loss. We retrain on its
    # *encoded* design (the trainer decodes it to physical for the detector).
    rs = run["results"]
    best = min(rs, key=lambda r: r["loss"])
    design_enc = np.asarray(best["x_encoded"], dtype=np.float32)
    bo_loss, bo_std = float(best["loss"]), float(best.get("loss_std", float("nan")))
    print(f"Best design: iteration {best['iteration']} of {len(rs)} " f"| BO loss = {bo_loss:.4f} ± {bo_std:.4f}")

    detector = detopt.detector.from_config(config["detector"])
    phys = np.asarray(detector.decode_design(design_enc), dtype=np.float32)
    print(f"  encoded  = {design_enc.tolist()}")
    print(f"  physical = {phys.tolist()}")

    trainer = FullBudgetTrainer.from_config(detector, config, max_epochs=max_epochs, seed=seed)
    budget = int(config["training"]["budget"])

    last = {}  # keep the most recent snapshot for the final plot + history dump

    def _progress(snap):
        last["snap"] = snap
        t, v = snap["train_loss_per_epoch"], snap["val_loss_per_epoch"]
        e = len(v)
        if e == 1 or e % 5 == 0 or e == max_epochs:
            print(f"  [epoch {e:>3}/{max_epochs}] train={t[-1]:.4f} val={v[-1]:.4f}")

    print(
        f"Sampling the full budget ({budget} detector calls) under this design and "
        f"training a fresh regressor for {max_epochs} epochs (cosine LR, seed={seed})..."
    )
    result = trainer.train(design_enc, np.random.SeedSequence(seed), on_epoch=_progress)

    print(
        f"\nConfirmed loss = {result.objective_loss:.4f} ± {result.objective_std:.4f} "
        f"(trained on {result.spent} detector calls)"
    )
    print(
        f"BO reported    = {bo_loss:.4f} ± {bo_std:.4f}" f"   | delta (confirm - BO) = {result.objective_loss - bo_loss:+.4f}"
    )

    out_dir = os.path.dirname(os.path.abspath(results))
    snap = last.get("snap")
    record = {
        "best_iteration": best["iteration"],
        "design_encoded": design_enc.tolist(),
        "design_physical": phys.tolist(),
        "bo_loss": bo_loss,
        "bo_loss_std": bo_std,
        "confirmed_loss": float(result.objective_loss),
        "confirmed_std": float(result.objective_std),
        "detector_calls": int(result.spent),
        "max_epochs": max_epochs,
        "seed": seed,
    }
    if snap is not None:  # full per-epoch curves, so the plot is reproducible later
        record["train_loss_per_epoch"] = snap["train_loss_per_epoch"].tolist()
        record["val_loss_per_epoch"] = snap["val_loss_per_epoch"].tolist()
    out = os.path.join(out_dir, "confirmation.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  saved -> {out}")

    if snap is not None:
        plot_confirmation(
            snap,
            bo_loss=bo_loss,
            bo_std=bo_std,
            confirmed_loss=float(result.objective_loss),
            out_path=os.path.join(out_dir, "confirmation.png"),
            title=f"Confirmation: design iter {best['iteration']} — {max_epochs} epochs, full budget",
        )
    return result.objective_loss


def evaluate(checkpoint, seed: int = 0, step=None, **config):
    """Evaluate a *saved* network on a full budget of FRESH events -- no training.

    Restores the regressor from a training ``checkpoint`` (and the design it was
    trained for), then scores it on the whole detector-call budget of freshly
    sampled events at that design (design-conditioned, the design's true encoded
    geometry). Answers: does this network's reported per-design loss hold up on
    fresh held-out data?
    """
    import jax
    from flax import nnx

    from detopt.nn import from_config as regressor_from_config
    from detopt.utils import io
    from detopt.utils.config import resolve_device

    seed = int(seed)
    if step is not None:
        step = int(step)
    device = resolve_device(config.get("device"))
    detector = detopt.detector.from_config(config["detector"])

    manager = io.get_checkpointer(checkpoint)
    used_step = manager.latest_step() if step is None else step
    params_pure, state_pure, design, aux = io.restore_training_checkpoint(manager, step)

    design_enc = np.asarray(design["encoded"], dtype=np.float32)
    phys = np.asarray(detector.decode_design(design_enc), dtype=np.float32)
    print(f"Loaded checkpoint {checkpoint} (step {used_step})")
    print(f"  design encoded  = {design_enc.tolist()}")
    if aux:
        rec_t = float(aux.get("train_loss", float("nan")))
        rec_v = float(aux.get("val_loss", float("nan")))
        print(f"  checkpoint recorded loss: train={rec_t:.4f} val={rec_v:.4f}")

    # Rebuild the architecture and load the saved weights into it (pure dict -> nnx).
    reg = regressor_from_config(detector, config=config["regressor"], rngs=nnx.Rngs(seed))
    graphdef, params, state = nnx.split(reg, nnx.Param, nnx.Variable)
    nnx.replace_by_pure_dict(params, params_pure)
    nnx.replace_by_pure_dict(state, state_pure)
    params, state = jax.device_put(params, device), jax.device_put(state, device)

    @jax.jit
    def eval_batch(params, state, X, mask, targets):
        net = nnx.merge(graphdef, params, state)
        X_norm = detector.normalize(X)
        # The design's true ENCODED design (1-D -> combine broadcasts per event).
        feats = detector.combine(X_norm, design_enc)
        pred = net(feats, mask, deterministic=True)  # dropout off
        return detector.loss(pred, targets)  # per-event (B,)

    budget = int(config["training"]["budget"])
    chunk = int(config["training"].get("eval_batch", 2048))
    phys_b = np.broadcast_to(phys[None, :], (chunk, phys.shape[0]))
    seq = np.random.SeedSequence(seed)
    print(f"Evaluating on ~{budget} fresh events at this design (no training)...")
    total = total_sq = 0.0
    n = 0
    while n < budget:
        _gt, X, mask, targets = detector(seq.spawn(1)[0], phys_b, split="val")
        losses = np.asarray(eval_batch(params, state, X, mask, targets), dtype=np.float64)
        total += losses.sum()
        total_sq += (losses**2).sum()
        n += losses.shape[0]
    mean = total / n
    sem = float(np.sqrt(max(total_sq / n - mean**2, 0.0) / n))
    print(f"\nFresh-data loss = {mean:.4f} ± {sem:.4f}  (over {n} held-out events)")

    out = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), f"evaluation_{os.path.basename(checkpoint)}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "checkpoint": os.path.abspath(checkpoint),
                "step": int(used_step),
                "design_blind": bool(design_blind),
                "design_encoded": design_enc.tolist(),
                "design_physical": phys.tolist(),
                "checkpoint_train_loss": (float(aux.get("train_loss")) if aux and "train_loss" in aux else None),
                "checkpoint_val_loss": (float(aux.get("val_loss")) if aux and "val_loss" in aux else None),
                "fresh_eval_loss": float(mean),
                "fresh_eval_sem": float(sem),
                "n_events": int(n),
                "seed": seed,
            },
            f,
            indent=2,
        )
    print(f"  saved -> {out}")
    return mean


if __name__ == "__main__":
    import gearup

    gearup.gearup(confirm=confirm, evaluate=evaluate).with_config("config/bo.yaml")()
