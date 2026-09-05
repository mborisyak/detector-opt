#!/usr/bin/env python3
"""Pick each strategy's training regime by INTEGRAL RANK over best-so-far curves -- `docs/final.md`.

    python scripts/select_retention.py --task angle --runs output/angle/select --write

TERMINOLOGY, because the two are easy to swap and the procedure depends on which is which. A
STRATEGY is the arm -- `from_scratch`, `continue`, `meta` -- and governs behaviour BETWEEN designs. A
TRAINING REGIME governs behaviour WITHIN a design, at each data addition -- `norewind`, the `rewind`
values, shrink-and-perturb. This script chooses a REGIME for each STRATEGY; it never compares
strategies with each other.

WHAT IT CONSUMES. One run per (strategy, seed, regime): the same 5-design SOBOL sequence, scored as
if it were a BO sequence. Because `n_init` covers the whole run the GP is never fitted, so every
regime sees the SAME designs in the SAME order -- the comparison is paired, and a difference between
curves is attributable to the regime rather than to which designs a regime happened to sample.

THE PROCEDURE, per strategy and per trajectory (a trajectory is one validation seed, holding all the
regimes' runs for that seed):

  1. BEST-SO-FAR against CUMULATIVE DETECTOR CALLS -- the running minimum of the per-design loss.
  2. A FINAL POINT AT THE BUDGET carrying the best loss the run ever saw. Regimes reach the end of
     the sequence having spent different amounts (`meta` converges cheaper), so without it the curves
     would end at different x and could not be compared over a common range. Carrying the best-seen
     loss flat to the budget states the true thing -- it reached that loss and would still be at it,
     having spent nothing more -- so converging cheaply is CREDITED, not truncated.
  3. LEFT-CONSTANT INTERPOLATION onto the union of all inflection points. Best-so-far is a step
     function: between two scored designs nothing has been learned, so the value cannot move.
     Interpolating linearly would invent a descent that did not happen.
  4. INTEGRAL RANK: rank the regimes at each point, integrate that rank over the budget, divide by
     the span. The ranks are constant between inflection points, so the integral is exact as a sum of
     rank times interval width, and dividing by the span keeps the score in RANK UNITS. This is
     budget-weighted rather than an average over points, which matters because the points are never
     evenly spaced -- designs cost 200k-470k calls apiece and the final point sits far out at the
     budget. Unweighted, the early designs would outvote the whole stretch after every regime has
     finished the sequence.
  5. Average the per-trajectory integral ranks across trajectories, within a strategy. Lowest wins.

⚠️ THE BASIS IS THREE TRAJECTORIES PER STRATEGY. That resolves a regime that is consistently ahead
from one consistently behind; it does not resolve neighbours. The full per-trajectory matrix and the
margin to the runner-up are printed so a selection resting on one seed is visible as one.
"""
import argparse
import collections
import glob
import json
import os


def best_so_far(rows, budget):
  """``(x, y)`` step curve: cumulative detector calls against the running-minimum loss.

    A final point is appended at ``budget`` carrying the best loss seen, so every regime's curve
    spans the same range whatever it spent.
    """
  x, y, spent, best = [], [], 0, float("inf")
  for row in rows:
    if row.get("loss") is None:
      continue
    spent += int(row.get("spent", 0))
    best = min(best, float(row["loss"]))
    x.append(float(spent))
    y.append(best)
  if len(x) == 0:
    return None
  if budget > x[-1]:
    x.append(float(budget))
    y.append(y[-1])
  return x, y


def verified_curve(payload, budget):
  """``(x, y)`` BEST-SO-FAR step curve built from a ``verification.json``: cumulative detector calls
    against the running minimum of the HELD-OUT score of every design of the run.

    Same construction as :func:`best_so_far`, on verified losses instead of reported ones -- that is
    the point. The two curves are compared and ranked against each other, so they must be built the
    same way; a raw (non-monotone) verified curve against a monotone reported one compares two
    different objects and the integral rank of the pair means nothing.

    A final point at ``budget`` carries the best score seen, so regimes that converge cheaply are
    credited rather than truncated.
    """
  points = sorted(payload.get("points", []), key=lambda p: p.get("detector_calls", 0))
  x, y, best = [], [], float("inf")
  for point in points:
    if point.get("test_loss") is None or point.get("detector_calls") is None:
      continue
    best = min(best, float(point["test_loss"]))
    x.append(float(point["detector_calls"]))
    y.append(best)
  if len(x) == 0:
    return None
  if budget > x[-1]:
    x.append(float(budget))
    y.append(y[-1])
  return x, y


def left_constant(grid, x, y):
  """``y`` sampled onto ``grid``, holding the last value at or before each point."""
  out, j = [], 0
  for g in grid:
    while j + 1 < len(x) and x[j + 1] <= g:
      j += 1
    out.append(y[j] if g >= x[0] else y[0])
  return out


def ranks(values):
  """Ranks, 1 = smallest, ties sharing the AVERAGE of the positions they span."""
  order = sorted(range(len(values)), key=lambda i: values[i])
  out = [0.0] * len(values)
  i = 0
  while i < len(order):
    j = i
    while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
      j += 1
    shared = (i + j) / 2.0 + 1.0
    for k in range(i, j + 1):
      out[order[k]] = shared
    i = j + 1
  return out


def provenance(path):
  """The `SELECTED-FROM:` directory recorded in an existing optimal config, or None."""
  if not os.path.exists(path):
    return None
  with open(path) as handle:
    for line in handle:
      if line.startswith("# SELECTED-FROM:"):
        return line.split(":", 1)[1].strip()
      if not line.startswith("#"):
        break
  return None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--runs", required=True, help="directory holding <seed>/<strategy>/<regime>/results.json")
  parser.add_argument("--write", action="store_true")
  parser.add_argument("--report", default=None)
  parser.add_argument("--force-write", action="store_true")
  parser.add_argument(
    "--verified", action="store_true",
    help="rank the HELD-OUT verification curves (verification.json beside each results.json) rather "
    "than the losses the runs reported for their own designs"
  )
  parser.add_argument(
    "--pooled", action="store_true",
    help="runs are <rung>/<seed>/<strategy>/<regime>/; pool rungs into ONE selection, treating "
    "each (rung, seed) as a separate trajectory"
  )
  args = parser.parse_args()

  # POOLED IS ONE EXTRA DIRECTORY LEVEL AND ONE MORE TRAJECTORY KEY, nothing else. The linear ladder
  # selects a single regime per strategy for all four rungs, so its runs live under
  # <rung>/<seed>/<strategy>/<regime> and a trajectory is a (rung, seed) pair -- 4 x 3 = 12 of them,
  # against 3 for a per-task selection. The ranking itself is unchanged: regimes are still compared
  # only WITHIN a trajectory, so rungs of different difficulty are never put on a common loss scale.
  depth = ("*", "*", "*", "*") if args.pooled else ("*", "*", "*")
  runs, verified, declared = {}, {}, set()
  for path in sorted(glob.glob(os.path.join(args.runs, *depth, "results.json"))):
    parts = path.split(os.sep)
    if args.pooled:
      rung, seed, strategy, regime = parts[-5:-1]
      trajectory = f"{rung}/{seed}"
    else:
      seed, strategy, regime = parts[-4:-1]
      trajectory = seed
    with open(path) as handle:
      payload = json.load(handle)
    payload_rows = payload.get("results", [])
    runs[(strategy, trajectory, regime)] = payload_rows
    # THE BUDGET IS THE CONFIGURED ONE, NOT THE SPEND THAT HAPPENED. Deriving it from the runs made the
    # score depend on whichever run spent most, and when the runs stopped on a DESIGN cap far short of
    # their budget the two limits disagreed enough to flip a verdict. Read what the run was told to
    # spend; the runs are budget-limited, so they reach it.
    declared.add(int(payload.get("config", {}).get("training", {}).get("budget", 0)))
    if args.verified:
      verification = os.path.join(os.path.dirname(path), "verification.json")
      if not os.path.exists(verification):
        raise SystemExit(
          f"select_retention: --verified but no verification.json beside {path}; the search runs must "
          f"be verified before they can be ranked on held-out scores"
        )
      with open(verification) as handle:
        payload = json.load(handle)
      # ⛔️ A PARTIAL VERIFICATION IS NOT A SHORT ONE, IT IS A WRONG ONE. A verification killed in
      # flight leaves a payload holding the points it reached, and nothing marks it as unfinished.
      # Ranked, its best-so-far curve is built from one or two points and flattens across the whole
      # budget -- which reads as a regime that converged early and cheaply, the strongest possible
      # result. Three such payloads (1, 1 and 4 points) existed when a spot box died mid-campaign.
      designs = len([row for row in payload_rows if row.get("loss") is not None])
      have = len([p for p in payload.get("points", []) if p.get("test_loss") is not None])
      if have < designs:
        raise SystemExit(
          f"select_retention: {verification} holds {have} verified point(s) but the trajectory has "
          f"{designs} scored design(s); verification scores every design, so this one is INCOMPLETE. "
          f"Re-run it, or exclude the cell -- do not rank on it."
        )
      verified[(strategy, trajectory, regime)] = payload
  if len(runs) == 0:
    raise SystemExit(f"select_retention: no runs under {args.runs}")

  declared.discard(0)
  if len(declared) > 1:
    raise SystemExit(f"select_retention: runs declare different budgets {sorted(declared)}; not comparable")
  budget = declared.pop() if len(declared) == 1 else max(sum(int(r.get("spent", 0)) for r in rows) for rows in runs.values())
  strategies = sorted({s for s, _, _ in runs})
  lines, winners = [], {}

  for strategy in strategies:
    seeds = sorted({k for s, k, _ in runs if s == strategy})
    regimes = sorted({r for s, _, r in runs if s == strategy})
    per_seed = collections.defaultdict(list)
    used = []
    for seed in seeds:
      curves = {}
      for regime in regimes:
        rows = runs.get((strategy, seed, regime))
        if args.verified:
          payload = verified.get((strategy, seed, regime))
          curve = None if payload is None else verified_curve(payload, budget)
        else:
          curve = None if rows is None else best_so_far(rows, budget)
        if curve is not None:
          curves[regime] = curve
      missing = [r for r in regimes if r not in curves]
      if len(missing) > 0:
        lines.append(f"  [skip] {strategy} seed {seed}: missing {missing}")
        continue
      grid = sorted({v for x, _ in curves.values() for v in x})
      sampled = {r: left_constant(grid, *curves[r]) for r in regimes}
      widths = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
      span = sum(widths)
      if span <= 0:
        lines.append(f"  [skip] {strategy} seed {seed}: zero-width support")
        continue
      used.append(seed)
      pointwise = [ranks([sampled[r][i] for r in regimes]) for i in range(len(grid) - 1)]
      for k, regime in enumerate(regimes):
        per_seed[regime].append(sum(p[k] * w for p, w in zip(pointwise, widths)) / span)

    if len(used) == 0:
      lines.append(f"  [skip] {strategy}: no usable trajectory")
      continue

    lines.append(f"\n=== {args.task} / {strategy} -- {len(used)} trajectories, {len(regimes)} regimes ===")
    summary = sorted((sum(v) / len(v), r) for r, v in per_seed.items())
    # THE PER-TRAJECTORY COLUMNS ONLY FIT WHILE THERE ARE FEW OF THEM. A pooled ladder selection has 12
    # trajectories, and a row of 12 twelve-wide columns runs to ~230 characters -- it wraps in a
    # terminal AND in the report file, which makes the permanent record unreadable exactly where the
    # decision is justified. Past six trajectories the spread is summarised and the full matrix follows
    # underneath, one line per trajectory, which stays legible at any width.
    if len(used) > 6:
      lines.append("  %-13s %9s %9s %9s  %13s" % ("regime", "min", "median", "max", "integral rank"))
      for average, regime in summary:
        v = sorted(per_seed[regime])
        lines.append("  %-13s %9.3f %9.3f %9.3f  %13.3f" % (regime, v[0], v[len(v) // 2], v[-1], average))
      lines.append("  per trajectory (regime=rank):")
      for i, label in enumerate(used):
        lines.append("    %-18s %s" % (label, "  ".join("%s=%.2f" % (r, per_seed[r][i]) for _, r in summary)))
    else:
      lines.append("  %-13s %s  %11s" % ("regime", " ".join("%12s" % s for s in used), "integral rank"))
      for average, regime in summary:
        lines.append("  %-13s %s  %11.3f" % (regime, " ".join("%12.3f" % v for v in per_seed[regime]), average))
    best, runner = summary[0], summary[1]
    winners[strategy] = best[1]
    lines.append(
      f"  -> {best[1]}  (integral rank {best[0]:.3f}); RUNNER-UP {runner[1]} at {runner[0]:.3f}, "
      f"margin {runner[0] - best[0]:.3f}"
    )

  text = "\n".join(lines)
  print(text)
  if args.report is not None:
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
      handle.write(text + "\n")

  if args.write and os.path.exists("SELECT_HOLD"):
    raise SystemExit("select_retention: SELECT_HOLD is present -- refusing to WRITE. See the file.")
  if args.write:
    for strategy, regime in winners.items():
      source = f"config/strategy/{args.task}-{strategy}-{regime}.yaml"
      target = f"config/strategy/{args.task}-{strategy}-optimal.yaml"
      previous = provenance(target)
      if previous is not None and previous != args.runs and not args.force_write:
        raise SystemExit(f"select_retention: {target} came from {previous!r}, this run used {args.runs!r}")
      with open(source) as handle:
        body = handle.read()
      with open(target, "w") as handle:
        handle.write(
          f"# SELECTED-FROM: {args.runs}\n"
          f"# Written by scripts/select_retention.py, by integral rank of the best-so-far curves over\n"
          f"# a shared 5-design Sobol sequence. A COPY of {os.path.basename(source)}; edit THAT file\n"
          f"# and re-run the selection -- edits here are discarded on the next run.\n#\n"
        )
        handle.write(body)
      print(f"[write] {source} -> {target}")


if __name__ == "__main__":
  main()
