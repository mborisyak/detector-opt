"""Regenerate the SHiP comparison figures from whatever cells are on disk.

Reads `output/ship-tied-rw025` and `output/ship-rw025` (the rw=0.25 tied-init corpus), pairing each
`results.json` with its `verification.json` when one exists. Emits the goal figure (`meta` alone,
every seed) and the four-arm figure, both via `plot_median_convergence`, whose pointwise reduction
starts at the first call count every curve has reached -- a median over a varying subset is not
monotone and has bitten this plot before. Writes the drawn data beside each PNG so the figure
regenerates without re-reading the corpus. `output/ship-4gpu` is never read.
"""
import argparse
import json
import pathlib

from detopt.utils.viz.bo import plot_median_convergence

ARMS = ("from_scratch", "closest", "continue", "meta")
ROOTS = ("output/ship-tied-rw025", "output/ship-rw025")


def load_cells(arms):
  runs = {arm: {} for arm in arms}
  for root in ROOTS:
    base = pathlib.Path(root)
    if not base.is_dir():
      continue
    for cell in sorted(base.glob("*/*")):
      arm = cell.name
      if arm not in runs:
        continue
      results_path = cell / "results.json"
      if not results_path.is_file():
        continue
      results = json.loads(results_path.read_text()).get("results", [])
      verification_path = cell / "verification.json"
      verification = json.loads(verification_path.read_text()) if verification_path.is_file() else None
      runs[arm][cell.parent.name] = {"results": results, "verification": verification}
  return {arm: seeds for arm, seeds in runs.items() if len(seeds) > 0}


def plot_arm_trajectories(runs, out_path, *, json_path=None, title=None):
  """Per-seed best-so-far trajectories for every arm, with each arm's verified endpoint as a dot.

  Thin lines are individual seeds, the bold step is the pointwise median over them (started where
  every seed has arrived, so the median never rises). At the right edge each arm carries its median
  VERIFIED test loss with two error bars: a wide cap for the spread across seeds (standard error of
  the median's inputs, what governs whether two arms differ) and a narrow one for the verification's
  own `test_sem` (what a single re-scoring resolves). Arms are offset horizontally so the dots do
  not overlap. An arm verified on one seed only gets an open marker: it has no across-seed spread.
  """
  import numpy as np
  from matplotlib.figure import Figure
  from detopt.utils.viz.bo import _STRATEGY_STYLE, _reduce_step

  fig = Figure(figsize=(11.5, 6.5))
  ax = fig.subplots()
  dumped = {}
  spans = []

  for index, (label, seeds) in enumerate(runs.items()):
    colour, marker = _STRATEGY_STYLE[index % len(_STRATEGY_STYLE)]
    curves, finals, sems = [], [], []
    for run in seeds.values():
      results = run.get("results") or []
      if len(results) > 0:
        calls = np.cumsum([r["spent"] for r in results], dtype=np.float64)
        loss = np.minimum.accumulate(np.array([r["loss"] for r in results], dtype=np.float64))
        curves.append((calls, loss))
        ax.step(calls, loss, where="post", lw=0.9, color=colour, alpha=0.28, zorder=1)
      points = (run.get("verification") or {}).get("points") or []
      if len(points) > 0:
        last = max(points, key=lambda p: p["detector_calls"])
        finals.append(float(last["test_loss"]))
        sems.append(float(last.get("test_sem", 0.0)))
    if len(curves) == 0:
      continue
    grid, median = _reduce_step(curves, np.median)
    ax.step(grid, median, where="post", lw=2.4, color=colour, alpha=0.95, zorder=3, label=f"{label}  (n={len(curves)})")
    spans.append(float(grid[-1]))
    entry = {"seeds": list(seeds), "calls": grid.tolist(), "median_best_so_far": median.tolist(), "n_seeds": len(curves)}
    if len(finals) > 0:
      entry["verified"] = {
        "values": finals,
        "median": float(np.median(finals)),
        "test_sem": float(np.median(sems)),
        "n": len(finals)
      }
    dumped[label] = entry

  right = max(spans) if len(spans) > 0 else 1.0
  offsets = np.linspace(0.015, 0.055, max(len(dumped), 1))
  for index, (label, entry) in enumerate(dumped.items()):
    verified = entry.get("verified")
    if verified is None:
      continue
    colour, marker = _STRATEGY_STYLE[index % len(_STRATEGY_STYLE)]
    x = right * (1.0 + offsets[index])
    values, centre = np.array(verified["values"], dtype=np.float64), verified["median"]
    spread = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    filled = len(values) > 1
    if spread > 0.0:
      ax.errorbar(x, centre, yerr=spread, color=colour, capsize=7, lw=2.0, zorder=4)
    ax.errorbar(x, centre, yerr=verified["test_sem"], color=colour, capsize=3, lw=3.2, zorder=5)
    ax.plot(
      x, centre, marker=marker, ms=9, color=colour, zorder=6, markerfacecolor=colour if filled else "white", markeredgewidth=1.8
    )
    ax.annotate(
      f"{centre:.4f}", xy=(x, centre), xytext=(0, 13 if index % 2 == 0 else -17), textcoords="offset points", ha="center",
      fontsize=9.0, fontweight="bold", color=colour
    )

  ax.axvline(right, color="#9b9a97", lw=0.8, ls=":", zorder=0)
  ax.set_xlim(right=right * 1.10)
  ax.set_xlabel("cumulative detector calls (simulated events)")
  ax.set_ylabel("best-so-far loss (normalised MSE)")
  ax.grid(True, which="both", alpha=0.25, lw=0.6)
  ax.minorticks_on()
  ax.legend(loc="upper right", fontsize=9, title="thin = seeds, bold = median")
  ax.set_title(title or "SHiP rw=0.25, tied init: per-seed trajectories and verified endpoints")
  fig.text(
    0.5, 0.012, "dots: median verified test loss.  wide cap = across-seed SEM,  narrow bar = "
    "verification test_sem.  open marker = one seed, no spread.", ha="center", fontsize=8.5, color="#52514e"
  )
  fig.tight_layout(rect=(0, 0.035, 1, 1))
  fig.savefig(out_path, dpi=130)
  print(f"  [plot] arm trajectories -> {out_path}")
  if json_path is not None:
    with open(json_path, "w") as handle:
      json.dump({"runs": dumped}, handle, indent=2, default=float)
    print(f"  [plot] plotted values -> {json_path}")
  return out_path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", default="output/ship-comp")
  parser.add_argument("--statistic", default="median", choices=("median", "mean"))
  args = parser.parse_args()
  out = pathlib.Path(args.output)
  out.mkdir(parents=True, exist_ok=True)

  goal = load_cells(("meta", ))
  n_seeds = len(goal.get("meta", {}))
  plot_median_convergence(
    goal, out / "GOAL-rw025-5seed.png", json_path=out / "GOAL-rw025-5seed.json", statistic=args.statistic,
    title=f"SHiP meta, replay_weight=0.25, tied init -- {args.statistic} of {n_seeds} seeds"
  )
  print(f"  GOAL-rw025-5seed.png    meta, {n_seeds} seeds")

  arms = load_cells(ARMS)
  counts = ", ".join(f"{arm}={len(seeds)}" for arm, seeds in arms.items())
  plot_median_convergence(
    arms, out / "median-rw025-vs-arms.png", json_path=out / "median-rw025-vs-arms.json", statistic=args.statistic,
    title=f"SHiP rw=0.25, tied init, by strategy -- {args.statistic} best-so-far"
  )
  print(f"  median-rw025-vs-arms.png   {counts}")

  plot_arm_trajectories(arms, out / "arms-trajectories-verified.png", json_path=out / "arms-trajectories-verified.json")
  print(f"  arms-trajectories-verified.png   {counts}")


if __name__ == "__main__":
  main()
