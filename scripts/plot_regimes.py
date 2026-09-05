#!/usr/bin/env python3
"""Per-strategy regime trajectories for the hyper-parameter selection -- mean and error, one panel
per strategy.

    python scripts/plot_regimes.py --task intersect --runs output/final-bo/intersect/select \
        --verified --out output/plots

WHAT IT SHOWS. One COLUMN per strategy and TWO ROWS, the house form: the MEDIAN best-so-far curve
per regime on top, the MEAN with ERROR BARS below. Not a single row, and not a shaded band -- a band
hides how few trajectories stand behind each point, which is the thing to keep visible while a
campaign is partial. Under `--verified` both rows are built from the held-out verification curves
rather than the losses the runs reported for their own designs. It is the picture behind
`select_retention.py`'s integral rank, and it reads the curves through that script's own functions
so the two cannot drift apart.

⚠️ THE MEAN IS TAKEN OVER A COMPLETE PAIRED BLOCK -- a set of seeds for which every plotted regime
has a run. Regimes finish at different rates, and averaging each regime over whatever seeds it
happens to hold compares regimes across DIFFERENT designs, which is exactly the confound the shared
Sobol sequence was chosen to remove. While a campaign is partial no seed may hold every regime yet,
so the block is found by dropping whichever regime or seed is missing most cells until what remains
is full. What was dropped is printed and named in the title: a panel showing three of five regimes
is a panel that cannot speak about the other two.

Because the panel is a mean over a handful of trajectories, the band is a standard error and not a
claim about a single run. With three seeds it separates a regime that is consistently ahead from one
consistently behind, and nothing finer.
"""
import argparse
import collections
import glob
import itertools
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_retention import best_so_far, left_constant, ranks, verified_curve


def load(roots, use_verified, pooled=False):
  """``(runs, verified, budget)`` gathered from every ``<seed>/<strategy>/<regime>/results.json``.

    ⚠️ A LADDER IS PLOTTED ONE RUNG AT A TIME, never pooled. Rungs carry different budgets and sit
    on different loss scales, so a mean across them would average incomparable trajectories -- point
    ``--runs`` at a single rung directory, which is already this layout.

    The legend carries each regime's AVERAGE INTEGRAL RANK across the block's trajectories,
    plus or minus the standard error across them -- the same number `select_retention.py`
    selects on, computed by the same procedure, so the picture and the decision cannot
    disagree. LOWER IS BETTER. With three trajectories the error bar separates a regime that
    is consistently ahead from one consistently behind and nothing finer; overlapping
    intervals are a tie, not an ordering.
    """
  depth = ("*", "*", "*", "*") if pooled else ("*", "*", "*")
  runs, verified, declared = {}, {}, set()
  for root in roots:
    for path in sorted(glob.glob(os.path.join(root, *depth, "results.json"))):
      if pooled:
        rung, seed, strategy, regime = path.split(os.sep)[-5:-1]
        seed = f"{rung}/{seed}"
      else:
        seed, strategy, regime = path.split(os.sep)[-4:-1]
      with open(path) as handle:
        payload = json.load(handle)
      rows = payload.get("results", [])
      if len(rows) == 0:
        continue
      declared.add(int(payload.get("config", {}).get("training", {}).get("budget", 0)))
      verification = os.path.join(os.path.dirname(path), "verification.json")
      if use_verified and not os.path.exists(verification):
        continue
      runs[(strategy, seed, regime)] = rows
      if use_verified:
        with open(verification) as handle:
          verified[(strategy, seed, regime)] = json.load(handle)
  declared.discard(0)
  if len(declared) > 1:
    raise SystemExit(f"plot_regimes: runs declare different budgets {sorted(declared)}; not comparable")
  if len(declared) == 0:
    raise SystemExit("plot_regimes: no run declares a budget")
  return runs, verified, declared.pop()


def complete_block(runs, strategy, seeds, regimes):
  """Largest complete ``(regimes, seeds)`` block, chosen to maximise regimes times seeds.

    Every subset of the regimes is tried -- there are five of them, so this is exact rather than
    greedy. Fixing the regimes fixes the seeds: a seed is in the block exactly when it holds all of
    them. Ties go to the block with more regimes, since comparing four regimes on two seeds says
    more than two regimes on four. Returns ``(regimes, seeds, dropped)``.
    """
  regimes = list(regimes)
  best = ([], [], 0)
  for size in range(len(regimes), 1, -1):
    for subset in itertools.combinations(regimes, size):
      paired = [k for k in seeds if all((strategy, k, r) in runs for r in subset)]
      score = len(subset) * len(paired)
      if len(paired) > 0 and (score > best[2] or (score == best[2] and len(subset) > len(best[0]))):
        best = (list(subset), paired, score)
  chosen, paired, _ = best
  dropped = [f"regime {r}" for r in regimes if r not in chosen]
  dropped += [f"seed {k}" for k in seeds if k not in paired and len(chosen) > 0]
  return chosen, paired, dropped


def integral_ranks(curves, regimes, paired, budget):
  """``{regime: [per-trajectory integral rank]}`` -- `select_retention`'s procedure, replicated.

    The grid is built PER TRAJECTORY, as there: regimes are ranked only against the regimes of their
    own seed, so a seed that is uniformly harder cannot outvote one that is uniformly easier. Ranks
    are constant between inflection points, so the integral is an exact sum of rank times interval
    width, divided by the span to keep the score in rank units.
    """
  out = {r: [] for r in regimes}
  for seed in paired:
    if any((r, seed) not in curves for r in regimes):
      continue
    grid = sorted({v for r in regimes for v in curves[(r, seed)][0]})
    if len(grid) < 2:
      continue
    sampled = {r: left_constant(grid, *curves[(r, seed)]) for r in regimes}
    widths = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
    span = sum(widths)
    if span <= 0:
      continue
    pointwise = [ranks([sampled[r][i] for r in regimes]) for i in range(len(grid) - 1)]
    for k, regime in enumerate(regimes):
      out[regime].append(sum(p[k] * w for p, w in zip(pointwise, widths)) / span)
  return out


def curve_for(runs, verified, key, budget, use_verified):
  if use_verified:
    payload = verified.get(key)
    return None if payload is None else verified_curve(payload, budget)
  rows = runs.get(key)
  return None if rows is None else best_so_far(rows, budget)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--runs", required=True, nargs="+", help="one or more <seed>/<strategy>/<regime> trees")
  parser.add_argument("--out", default="output/plots")
  parser.add_argument("--name", default=None)
  parser.add_argument(
    "--verified", action="store_true", help="plot the held-out verification curves rather than the losses the runs reported"
  )
  args = parser.parse_args()

  runs, verified, budget = load(args.runs, args.verified)
  if len(runs) == 0:
    raise SystemExit(f"plot_regimes: no runs under {args.runs}")
  strategies = sorted({s for s, _, _ in runs})

  figure, axes = plt.subplots(2, len(strategies), figsize=(6.0 * len(strategies), 8.4), squeeze=False, sharex="col")
  summary = []
  for column, strategy in enumerate(strategies):
    upper, lower = axes[0][column], axes[1][column]
    seeds = sorted({k for s, k, _ in runs if s == strategy})
    regimes = sorted({r for s, _, r in runs if s == strategy})
    held = {r: [k for k in seeds if (strategy, k, r) in runs] for r in regimes}
    regimes, paired, dropped = complete_block(runs, strategy, seeds, regimes)

    grid, curves = set(), {}
    for regime in regimes:
      for seed in paired:
        curve = curve_for(runs, verified, (strategy, seed, regime), budget, args.verified)
        if curve is None:
          continue
        curves[(regime, seed)] = curve
        grid.update(curve[0])
    grid = sorted(g for g in grid if g <= budget)
    if len(grid) == 0 or len(paired) == 0:
      for pane in (upper, lower):
        pane.text(0.5, 0.5, f"{strategy}\nno paired trajectory yet", ha="center", va="center", transform=pane.transAxes)
      upper.set_title(strategy)
      summary.append(f"  {strategy:<12} NO complete block yet (dropped: {', '.join(dropped) or 'nothing'})")
      continue

    rank_of = integral_ranks(curves, regimes, paired, budget)
    marks = np.unique(np.linspace(0, len(grid) - 1, min(len(grid), 12)).astype(int))
    for offset, regime in enumerate(regimes):
      stack = [left_constant(grid, *curves[(regime, seed)]) for seed in paired if (regime, seed) in curves]
      if len(stack) == 0:
        continue
      stack = np.asarray(stack)
      mean = stack.mean(axis=0)
      error = stack.std(axis=0, ddof=1) / np.sqrt(len(stack)) if len(stack) > 1 else np.zeros_like(mean)
      scores = rank_of.get(regime, [])
      if len(scores) > 1:
        rank_error = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        label = f"{regime}  rank {np.mean(scores):.2f}+/-{rank_error:.2f}"
      elif len(scores) == 1:
        label = f"{regime}  rank {scores[0]:.2f} (1 traj)"
      else:
        label = f"{regime}  rank n/a"
      if len(held[regime]) - len(paired) > 0:
        label += f" [+{len(held[regime]) - len(paired)} unpaired]"
      line, = upper.step(grid, np.median(stack, axis=0), where="post", label=label, linewidth=1.6)
      shift = np.asarray(grid, dtype=float)[marks] * (1.0 + 0.004 * (offset - 0.5 * (len(regimes) - 1)))
      lower.errorbar(
        shift, mean[marks], yerr=error[marks], label=label, color=line.get_color(), linewidth=1.4, elinewidth=1.0, capsize=3,
        marker="o", markersize=3.5
      )

    scored = "held-out test loss" if args.verified else "best-so-far loss"
    upper.set_ylabel(f"median {scored}")
    lower.set_ylabel(f"mean {scored}")
    lower.set_xlabel("cumulative detector calls")
    for pane in (upper, lower):
      pane.set_xlim(min(grid), budget)
      pane.grid(alpha=0.25, linewidth=0.5)
      pane.legend(fontsize=7, loc="upper right")
    note = f", dropped {'; '.join(dropped)}" if len(dropped) > 0 else ""
    upper.set_title(f"{args.task} / {strategy} -- {len(regimes)} regimes x {len(paired)} seeds")
    lower.set_title(f"mean +/- standard error over {len(paired)} seeds", fontsize=9)
    summary.append(f"  {strategy:<12} block {len(regimes)} regimes x {len(paired)} seeds{note}")

  kind = "verified" if args.verified else "reported"
  figure.suptitle(f"{args.task}: regime trajectories ({kind}, budget {budget:,})", fontsize=12)
  figure.tight_layout()
  os.makedirs(args.out, exist_ok=True)
  name = args.name if args.name is not None else f"regimes-{args.task}.png"
  path = os.path.join(args.out, name)
  figure.savefig(path, dpi=150)
  plt.close(figure)
  print(f"wrote {path}  ({kind}, budget {budget:,})")
  print("\n".join(summary))


if __name__ == "__main__":
  main()
