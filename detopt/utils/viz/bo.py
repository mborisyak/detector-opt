"""Plots for the Bayesian-optimisation driver (``scripts/bo.py``).

* :func:`plot_iteration` -- one design's train/val loss curves (with +/- SEM
  uncertainty bands) over epochs, marking where data was added;
* :func:`plot_convergence` -- best-so-far objective loss vs cumulative detector
  calls across the BO run.
"""

import json
import os

import numpy as np

__all__ = ["plot_iteration", "plot_convergence", "plot_convergence_comparison", "plot_confirmation"]


def plot_iteration(history, iteration, design, val_loss, plots_dir):
    """Plot one design's per-epoch train/val loss with +/- SEM bands.

    ``history`` is the trainer's epoch snapshot (means, SEMs and pool sizes per
    epoch). Saves ``iter_<iteration>.png`` and the design JSON into ``plots_dir``.
    Uses matplotlib's OO ``Figure`` API (no pyplot global state) so it is safe to
    call from a background thread (the trainer runs it off the training loop).
    """
    from matplotlib.figure import Figure

    os.makedirs(plots_dir, exist_ok=True)
    tl = history["train_loss_per_epoch"]
    vl = history["val_loss_per_epoch"]
    tsem = history.get("train_sem_per_epoch")
    vsem = history.get("val_sem_per_epoch")
    budgets = history.get("train_budget_per_epoch")
    epochs = np.arange(1, tl.shape[0] + 1)

    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    # Means with +/- SEM uncertainty bands (lower band floored for the log axis).
    ax.plot(epochs, tl, marker="o", ms=3, color="tab:blue", label="train (mean +/- SEM)")
    if tsem is not None and tsem.size == tl.size:
        ax.fill_between(epochs, np.maximum(tl - tsem, 1e-9), tl + tsem, color="tab:blue", alpha=0.2)
    if vl.size:
        ev = epochs[: vl.shape[0]]
        ax.plot(ev, vl, marker="s", ms=3, color="tab:orange", label="val (mean +/- SEM)")
        if vsem is not None and vsem.size == vl.size:
            ax.fill_between(
                ev,
                np.maximum(vl - vsem, 1e-9),
                vl + vsem,
                color="tab:orange",
                alpha=0.2,
            )

    # One "data added" legend entry; annotate each addition with the new pool size.
    if budgets is not None and budgets.size > 1:
        for k, idx in enumerate(np.where(np.diff(budgets) > 0)[0] + 1):
            ax.axvline(
                epochs[idx],
                color="tab:green",
                ls="--",
                alpha=0.5,
                label="data added" if k == 0 else None,
            )
            ax.annotate(
                f"{int(budgets[idx])}",
                xy=(epochs[idx], 0.98),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=8,
                color="tab:green",
            )

    # Title: the converged estimate (train+val)/2 +/- its SEM, when available.
    final_train = int(history.get("final_train_budget", 0))
    if tsem is not None and vsem is not None and tl.size and vl.size:
        est = 0.5 * (tl[-1] + vl[-1])
        est_sem = 0.5 * float(np.hypot(tsem[-1], vsem[-1]))
        head = f"Iter {iteration}: loss={est:.4f} +/- {est_sem:.4f}"
    else:
        head = f"Iter {iteration}: final val={val_loss:.5f}"
    ax.set_title(f"{head}  (train pool {final_train})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalised)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(plots_dir, f"iter_{iteration:03d}.png")
    fig.savefig(out, dpi=120)
    with open(os.path.join(plots_dir, f"iter_{iteration:03d}_design.json"), "w") as f:
        json.dump(design, f, indent=2, default=float)
    return out


def plot_convergence(results, output_dir):
    """Best-so-far objective loss vs cumulative detector calls (simulated events)."""
    if not results:
        return
    from matplotlib.figure import Figure

    per_iter_calls = np.array([r["spent"] for r in results], dtype=np.float64)
    cumulative_calls = np.cumsum(per_iter_calls)
    val_loss = np.array([r["loss"] for r in results], dtype=np.float64)
    val_loss_std = np.array([r["loss_std"] for r in results], dtype=np.float64)
    best_so_far = np.minimum.accumulate(val_loss)

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.errorbar(
        cumulative_calls,
        val_loss,
        yerr=val_loss_std,
        fmt="o",
        color="tab:gray",
        alpha=0.6,
        capsize=3,
        label="per-iteration loss +/- SEM",
    )
    ax.step(
        cumulative_calls,
        best_so_far,
        where="post",
        color="tab:blue",
        lw=2,
        label="best so far",
    )
    ax.set_xlabel("cumulative detector calls (simulated events)")
    ax.set_ylabel("loss (normalised MSE)")
    ax.set_yscale("log")
    ax.set_title("BO convergence vs detector budget")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(output_dir, "convergence.png")
    fig.savefig(out, dpi=120)
    with open(os.path.join(output_dir, "convergence.json"), "w") as f:
        json.dump(
            {
                "cumulative_detector_calls": cumulative_calls.tolist(),
                "val_loss": val_loss.tolist(),
                "val_loss_std": val_loss_std.tolist(),
                "best_so_far": best_so_far.tolist(),
            },
            f,
            indent=2,
        )
    print(f"  [plot] convergence -> {out}")


def plot_convergence_comparison(runs, out_path, title="BO convergence by strategy"):
    """Overlay best-so-far loss vs cumulative detector calls for several runs.

    ``runs`` maps a label -> that run's ``results`` list (each entry a dict with
    ``spent`` and ``loss``). One step-curve per label on a shared axis.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    for label, results in runs.items():
        if not results:
            continue
        calls = np.cumsum([r["spent"] for r in results], dtype=np.float64)
        loss = np.array([r["loss"] for r in results], dtype=np.float64)
        best = np.minimum.accumulate(loss)
        ax.step(calls, best, where="post", lw=2, marker="o", ms=3, label=label)
    ax.set_xlabel("cumulative detector calls (simulated events)")
    ax.set_ylabel("best-so-far loss (normalised MSE)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  [plot] strategy comparison -> {out_path}")
    return out_path


def plot_confirmation(history, *, bo_loss, out_path, bo_std=0.0, confirmed_loss=None, epochs=None, title=None):
    """Train/val loss vs epoch for a full-budget confirmation retrain.

    Draws the per-epoch train/val curves (+/- SEM bands) plus the BO-reported loss
    as a dashed reference line, so the gap between a fresh full-budget retrain and
    BO's number is visible at a glance. ``history`` is a trainer snapshot
    (``train_loss_per_epoch`` / ``val_loss_per_epoch`` and optional
    ``*_sem_per_epoch``); pass ``epochs`` for non-contiguous x-values.
    """
    from matplotlib.figure import Figure

    tl = np.asarray(history["train_loss_per_epoch"], dtype=np.float64)
    vl = np.asarray(history["val_loss_per_epoch"], dtype=np.float64)
    tsem = history.get("train_sem_per_epoch")
    vsem = history.get("val_sem_per_epoch")
    x = np.asarray(epochs, dtype=np.float64) if epochs is not None else np.arange(1, tl.size + 1)

    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    ax.plot(x, tl, marker="o", ms=3, color="tab:blue", label="train (mean +/- SEM)")
    if tsem is not None and len(tsem) == tl.size:
        tsem = np.asarray(tsem, dtype=np.float64)
        ax.fill_between(x, np.maximum(tl - tsem, 1e-9), tl + tsem, color="tab:blue", alpha=0.2)
    ax.plot(x, vl, marker="s", ms=3, color="tab:orange", label="val (mean +/- SEM)")
    if vsem is not None and len(vsem) == vl.size:
        vsem = np.asarray(vsem, dtype=np.float64)
        ax.fill_between(x, np.maximum(vl - vsem, 1e-9), vl + vsem, color="tab:orange", alpha=0.2)

    # The number we are checking against (what BO claimed for this design).
    ax.axhline(bo_loss, color="tab:red", ls="--", lw=1.5, label=f"BO reported = {bo_loss:.3f}")
    if bo_std:
        ax.axhspan(bo_loss - bo_std, bo_loss + bo_std, color="tab:red", alpha=0.10)

    head = title or "Full-budget confirmation retrain"
    if confirmed_loss is not None:
        head += f": confirmed = {confirmed_loss:.3f}  vs  BO = {bo_loss:.3f}"
    ax.set_title(head)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (normalised MSE)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  [plot] confirmation -> {out_path}")
    return out_path
