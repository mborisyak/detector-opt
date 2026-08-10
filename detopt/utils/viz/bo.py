"""Plots for the Bayesian-optimisation driver (``scripts/bo.py``).

* :func:`plot_iteration` -- one design's train/val loss curves (with +/- SEM
  uncertainty bands) over epochs, marking where data was added;
* :func:`plot_convergence` -- best-so-far objective loss vs cumulative detector
  calls across the BO run.
"""

import json
import os

import numpy as np

__all__ = ["plot_iteration", "plot_convergence", "plot_convergence_comparison",
           "plot_convergence_two_panel", "plot_confirmation"]


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


# Categorical slots of the validated default palette, assigned to strategies in FIXED order (never
# cycled) with a distinct marker each -- the marker is the secondary encoding, so the series stay
# separable without relying on colour.
_STRATEGY_STYLE = (
    ("#2a78d6", "o"),  # blue
    ("#eb6834", "s"),  # orange
    ("#1baf7a", "^"),  # aqua
    ("#eda100", "D"),  # yellow
    ("#e87ba4", "v"),  # magenta
    ("#4a3aa7", "P"),  # violet
)


def plot_strategy_verification(runs, out_path, *, json_path=None, title="BO convergence: self-evaluated vs verified"):
    """Overlay each strategy's self-evaluated and independently verified loss on one axis.

    ``runs`` maps a strategy label to ``{"results": [...], "verification": {...} | None}`` -- the BO
    run's ``results.json`` entries and the ``verification.json`` written by
    ``scripts/verify_trajectory.py``.

    Per strategy, one colour (colour follows the strategy, never its rank) carrying:

    * a **dashed** step -- the run's own best-so-far loss. That number is *self-evaluated*: the
      convergence procedure stopped on the very train/val losses it reports, and the warm-started and
      continual strategies carry a network that has already seen other designs' data;
    * **open markers** on that dashed line -- the raw reported loss of exactly the designs that were
      verified, so the pairing is visible rather than inferred;
    * a **solid** line with error bars -- those same designs re-scored on a held-out test split that
      nothing in the procedure ever looked at.

    The vertical gap between a strategy's open and filled markers is its optimism, and it is not the
    same for all four -- which is why the raw dashed comparison alone would mislead. Everything drawn
    is written to ``json_path`` so the figure regenerates without re-running anything.
    """
    import json

    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    dumped = {}

    for index, (label, run) in enumerate(runs.items()):
        results = run.get("results") or []
        if len(results) == 0:
            continue
        colour, marker = _STRATEGY_STYLE[index % len(_STRATEGY_STYLE)]
        calls = np.cumsum([r["spent"] for r in results], dtype=np.float64)
        reported = np.array([r["loss"] for r in results], dtype=np.float64)
        best = np.minimum.accumulate(reported)
        ax.step(calls, best, where="post", lw=1.8, ls="--", color=colour, alpha=0.9,
                label=f"{label} (self-evaluated)")
        entry = {"calls": calls.tolist(), "reported_loss": reported.tolist(),
                 "self_evaluated_best_so_far": best.tolist()}

        verification = run.get("verification")
        points = (verification or {}).get("points") or []
        if len(points) > 0:
            vx = np.array([p["detector_calls"] for p in points], dtype=np.float64)
            vy = np.array([p["test_loss"] for p in points], dtype=np.float64)
            verr = np.array([p["test_sem"] for p in points], dtype=np.float64)
            vreported = np.array([p["reported_loss"] for p in points], dtype=np.float64)
            # the self-evaluated value of exactly these designs (open markers, on the dashed style)
            ax.plot(vx, vreported, ls="none", marker=marker, ms=7, mfc="none", mec=colour, mew=1.6)
            ax.errorbar(vx, vy, yerr=verr, color=colour, lw=2.2, marker=marker, ms=7, capsize=3,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label=f"{label} (verified, held-out)")
            entry["verification"] = {
                "calls": vx.tolist(), "test_loss": vy.tolist(), "test_sem": verr.tolist(),
                "reported_loss": vreported.tolist(),
                "point": [p["point"] for p in points],
                "val_loss": [p["val_loss"] for p in points],
                "design_physical": [p["design_physical"] for p in points],
            }
        dumped[label] = entry

    ax.set_xlabel("cumulative detector calls (simulated events)")
    ax.set_ylabel("loss (normalised MSE)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, lw=0.6)
    handles, labels = ax.get_legend_handles_labels()
    # A second, style-only legend: what dashed / open / solid mean.
    style = [
        Line2D([], [], color="#52514e", ls="--", lw=2, label="self-evaluated, best-so-far"),
        Line2D([], [], color="#52514e", ls="none", marker="o", ms=7, mfc="none", mew=1.6,
               label="self-evaluated, verified designs"),
        Line2D([], [], color="#52514e", ls="-", lw=2, marker="o", ms=7, markeredgecolor="white",
               label="verified (held-out test split)"),
    ]
    first = ax.legend(handles, labels, loc="upper right", fontsize=9, ncols=2)
    ax.add_artist(first)
    ax.legend(handles=style, loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  [plot] strategy comparison -> {out_path}")

    if json_path is not None:
        with open(json_path, "w") as f:
            json.dump({"runs": dumped}, f, indent=2, default=float)
        print(f"  [plot] plotted values -> {json_path}")
    return out_path


def plot_convergence_two_panel(runs, out_path, *, json_path=None, title="BO convergence by strategy"):
  """Two stacked step-panels sharing an x-axis: TOP = each strategy's self-evaluated best-so-far
  loss, BOTTOM = the same for the independently verified (held-out test) designs. When no strategy
  has a verification, only the single self-evaluated panel is drawn. Both panels use the SAME step
  style (matching :func:`plot_convergence_comparison`); colour+marker follow the strategy, never its
  rank. Everything drawn is written to ``json_path`` so either panel regenerates without re-running.

  ``runs`` maps a strategy label -> ``{"results": [...], "verification": {...} | None}``.
  """
  import json

  from matplotlib.figure import Figure

  has_verif = any((run.get("verification") or {}).get("points") for run in runs.values())
  fig = Figure(figsize=(9, 8) if has_verif else (9, 5))
  if has_verif:
    ax_self, ax_ver = fig.subplots(2, 1, sharex=True)
  else:
    ax_self, ax_ver = fig.subplots(), None

  dumped = {}
  for index, (label, run) in enumerate(runs.items()):
    results = run.get("results") or []
    if len(results) == 0:
      continue
    colour, marker = _STRATEGY_STYLE[index % len(_STRATEGY_STYLE)]
    calls = np.cumsum([r["spent"] for r in results], dtype=np.float64)
    best = np.minimum.accumulate([r["loss"] for r in results])
    ax_self.step(calls, best, where="post", lw=2, marker=marker, ms=3, color=colour, label=label)
    entry = {"self_evaluated": {"calls": calls.tolist(), "best_so_far": best.tolist()}}

    points = sorted(((run.get("verification") or {}).get("points") or []), key=lambda p: p["detector_calls"])
    if ax_ver is not None and len(points) > 0:
      vx = np.array([p["detector_calls"] for p in points], dtype=np.float64)
      vbest = np.minimum.accumulate([p["test_loss"] for p in points])
      ax_ver.step(vx, vbest, where="post", lw=2, marker=marker, ms=3, color=colour, label=label)
      entry["verified"] = {"calls": vx.tolist(), "best_so_far": vbest.tolist(),
                           "test_loss": [float(p["test_loss"]) for p in points],
                           "test_sem": [float(p["test_sem"]) for p in points]}
    dumped[label] = entry

  for ax in (a for a in (ax_self, ax_ver) if a is not None):
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("best-so-far loss (norm. MSE)")
    ax.legend(loc="upper right", fontsize=9)
  ax_self.set_title(f"{title} — self-evaluated")
  if ax_ver is not None:
    ax_ver.set_title("verified (held-out test)")
  (ax_ver or ax_self).set_xlabel("cumulative detector calls (simulated events)")
  fig.tight_layout()
  fig.savefig(out_path, dpi=120)
  print(f"  [plot] strategy comparison (two-panel) -> {out_path}")

  if json_path is not None:
    with open(json_path, "w") as f:
      json.dump({"runs": dumped}, f, indent=2, default=float)
    print(f"  [plot] plotted values -> {json_path}")
  return out_path


def _median_step(curves):
    """Pointwise median of ``where="post"`` step functions, given as ``(x, y)`` pairs with x sorted.

    Evaluated on the union of the curves' x grids, starting at the first point every curve has
    reached (before that the median is over fewer runs and would jump when one joins); beyond its
    last point a curve continues flat -- a best-so-far value persists once found.
    """
    start = max(float(x[0]) for x, _ in curves)
    grid = np.unique(np.concatenate([x for x, _ in curves]))
    grid = grid[grid >= start]
    stack = np.stack([y[np.searchsorted(x, grid, side="right") - 1] for x, y in curves])
    return grid, np.median(stack, axis=0)


def plot_median_convergence(runs, out_path, *, json_path=None, title="BO convergence: median best-so-far across seeds"):
    """Two panels of per-strategy MEDIAN best-so-far (cummin) curves across seeds.

    ``runs`` maps a strategy label to ``{seed_label: {"results": [...], "verification": {...} | None}}``
    -- each seed's ``results.json`` entries and its ``verification.json`` (``scripts/
    verify_trajectory.py``). Left panel: the runs' own self-evaluated reported loss (dashed, the
    same visual language as :func:`plot_strategy_verification`). Right panel: the independently
    verified held-out test loss of the verified designs (solid). Per seed the best-so-far curve is
    a step function in cumulative detector calls; the median over seeds is taken pointwise via
    :func:`_median_step`. Everything drawn is written to ``json_path`` so the figure regenerates
    without re-running anything.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5.5))
    axes = fig.subplots(1, 2, sharex=True)
    dumped = {}

    for index, (label, seeds) in enumerate(runs.items()):
        colour, _ = _STRATEGY_STYLE[index % len(_STRATEGY_STYLE)]
        reported, verified = [], []
        for run in seeds.values():
            results = run.get("results") or []
            if len(results) > 0:
                calls = np.cumsum([r["spent"] for r in results], dtype=np.float64)
                loss = np.array([r["loss"] for r in results], dtype=np.float64)
                reported.append((calls, np.minimum.accumulate(loss)))
            points = sorted((run.get("verification") or {}).get("points") or [],
                            key=lambda p: p["detector_calls"])
            if len(points) > 0:
                vx = np.array([p["detector_calls"] for p in points], dtype=np.float64)
                vy = np.array([p["test_loss"] for p in points], dtype=np.float64)
                verified.append((vx, np.minimum.accumulate(vy)))
        entry = {"seeds": list(seeds)}
        panels = ((axes[0], reported, "--", "self_evaluated"), (axes[1], verified, "-", "verified"))
        for ax, curves, linestyle, key in panels:
            if len(curves) == 0:
                continue
            grid, median = _median_step(curves)
            suffix = "" if len(curves) == len(seeds) else f" (n={len(curves)})"
            ax.step(grid, median, where="post", lw=2.0, ls=linestyle, color=colour, alpha=0.9,
                    label=f"{label}{suffix}")
            entry[key] = {"calls": grid.tolist(), "median_best_so_far": median.tolist(),
                          "n_seeds": len(curves)}
        dumped[label] = entry

    subtitles = ("self-evaluated (reported loss)", "verified (held-out test loss)")
    for ax, subtitle in zip(axes, subtitles):
        ax.set_xlabel("cumulative detector calls (simulated events)")
        ax.set_yscale("log")
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.25, lw=0.6)
        if len(ax.get_lines()) > 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("median best-so-far loss (normalised MSE)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  [plot] median convergence -> {out_path}")

    if json_path is not None:
        with open(json_path, "w") as f:
            json.dump({"runs": dumped}, f, indent=2, default=float)
        print(f"  [plot] plotted values -> {json_path}")
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
