#!/usr/bin/env python3
"""Pick each arm's retention setting from POINTWISE RANKS along its convergence curve -- stage 3 of
`docs/final.md`.

    python scripts/select_retention.py --task angle --probes output/final/angle/probe --write

THE WHOLE CURVE, NOT ITS ENDPOINT, AND THAT IS THE POINT. There are only three probed designs per
arm, so a procedure that reduces each to one final loss selects among seven settings from three
numbers -- enough to separate a setting that is always first from one that is always last, and
nothing finer. Ranking pointwise along the best-so-far trajectory instead makes every epoch a
comparison, and it asks the question that actually matters: which setting reaches a given loss for
the least data, not merely which one ends lowest.

THE PROCEDURE, per arm:

  1. BEST-SO-FAR. Each probe record carries `objective_per_epoch` -- `(train + val) / 2` epoch by
     epoch, the same quantity `TrainResult.objective_loss` reports at convergence -- against
     `window_per_epoch`, the training rows behind each epoch. The trajectory is the running minimum
     of the first as a function of the second, so it is monotone non-increasing by construction.
     `window` plateaus WITHIN a growth round, so the curve is collapsed to one value per distinct
     window before anything is interpolated.
  2. PSEUDO POINT AND UNION SUPPORT. The settings stop at different windows -- that IS the effect
     under study. Each trajectory gets a pseudo point at the per-design budget (`iteration_limit`)
     carrying its best-seen loss, so a setting that converged cheaply is credited for holding that
     loss rather than truncating everyone else's comparison range. Every curve then spans the same
     range, and the grid is the UNION of all the settings' points -- no extrapolation anywhere.
  3. RANK INTEGRAL. Rank the settings at each grid point, then integrate that rank over the budget
     and divide by the full span, which keeps the score in RANK UNITS (1 to the number of settings).
     The ranks are constant between grid points, so the integral is exact as a sum of rank times
     interval width. This is budget-weighted, not an average over grid points, and the grid is never
     even: growth rounds sit `n_increment` apart while the pseudo point is far out at
     `iteration_limit`. Unweighted, a cluster of early closely-spaced additions would outvote the
     whole stretch of budget after every setting has converged. The per-trajectory integrals are then
     averaged across trajectories. Ranks are competition ranks with ties sharing the smaller rank, so
     two indistinguishable settings cannot be separated by float noise.

THE STATISTIC IS THE MEAN. `docs/final.md`'s stage-3 heading says "median rank" while its bullets say
"average rank" twice; the user settled it as AVERAGE (2026-08-31), which is also what the bullets
specify, so the mean is what selects. The median over grid points and trajectories is still computed
and printed beside it -- not as a competing rule, but because a selection that flips between the two
is one resting on a skewed handful of grid points, and that is worth seeing.

TIES ARE BROKEN DETERMINISTICALLY, in decreasing order of how much the data says: the rank
rank integral, then weighted median rank, then budget-weighted mean best-so-far, then the config
name. A tie that reaches the name is reported as unresolved rather than presented as a choice.

WHAT `--write` DOES: copies the winning `config/strategy/<task>-<arm>-<variant>.yaml` to
`config/strategy/<task>-<arm>-optimal.yaml` with a provenance header. Without it nothing is written.
"""
import argparse
import collections
import glob
import json
import os
import statistics

import numpy as np


def step(grid, window, value):
  """``value`` sampled onto ``grid`` with a ZERO-ORDER HOLD -- the last value at or before each point.

    The best-so-far curve is piecewise CONSTANT in the window: between two data additions the design
    has acquired nothing, so its best loss cannot move. Interpolating linearly would invent a descent
    that did not happen, and would let two settings cross strictly between grid points -- exactly the
    case the rank integral assumes away when it holds each rank across its interval.
    """
  return value[np.clip(np.searchsorted(window, grid, side="right") - 1, 0, len(value) - 1)]


def weighted_median(values, weights):
  """The value at which the cumulative weight first reaches half the total."""
  order = np.argsort(values)
  cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
  return float(np.asarray(values)[order][int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])


def _provenance(path):
  """The `SELECTED-FROM:` probe directory recorded in an existing optimal config, or None."""
  if not os.path.exists(path):
    return None
  with open(path) as handle:
    for line in handle:
      if line.startswith("# SELECTED-FROM:"):
        return line.split(":", 1)[1].strip()
      if not line.startswith("#"):
        break
  return None


def rank(values):
  """Competition ranks, 1 = smallest. Ties share the smaller rank."""
  order = sorted(range(len(values)), key=lambda i: values[i])
  ranks = [0] * len(values)
  i = 0
  while i < len(order):
    j = i
    while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
      j += 1
    for k in range(i, j + 1):
      ranks[order[k]] = i + 1
    i = j + 1
  return ranks


def best_so_far(record):
  """``(window, best_so_far_objective)`` -- one point per distinct window, monotone non-increasing,
    with a PSEUDO POINT at the per-design budget carrying the best loss the trajectory ever saw.

    The pseudo point is what makes the settings comparable. They stop at different windows -- that is
    the effect under study -- and without it a setting that converged early would either truncate
    every other setting's comparison range or have to be extrapolated. Carrying its best-seen loss
    flat to the budget instead states the true thing: it reached that loss and would still be at it,
    having spent nothing more. A setting that converges cheaply is CREDITED by this, not penalised.
    """
  objective = np.asarray(record["objective_per_epoch"], dtype=np.float64)
  window = np.asarray(record["window_per_epoch"], dtype=np.float64)
  if len(objective) == 0 or len(objective) != len(window):
    raise SystemExit(f"select_retention: malformed curve in {record.get('trajectory')} ({record.get('variant')})")
  running = np.minimum.accumulate(objective)
  # The LAST epoch at each distinct window carries the running minimum for that window.
  keep = np.concatenate([window[1:] != window[:-1], [True]])
  window, running = window[keep], running[keep]
  budget = float(record["iteration_limit"])
  if budget > window[-1]:
    window = np.append(window, budget)
    running = np.append(running, running[-1])
  return window, running


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--probes", required=True, help="directory holding <seed>/<arm>/<variant>.json")
  parser.add_argument("--write", action="store_true", help="copy winners to <task>-<arm>-optimal.yaml")
  parser.add_argument("--report", default=None, help="also write the table here")
  parser.add_argument(
    "--force-write", action="store_true", help="overwrite an optimal config that was selected from a DIFFERENT probe directory"
  )
  args = parser.parse_args()

  records = []
  for path in sorted(glob.glob(os.path.join(args.probes, "*", "*", "*.json"))):
    with open(path) as handle:
      record = json.load(handle)
    record["variant"] = os.path.splitext(os.path.basename(path))[0]
    records.append(record)
  if len(records) == 0:
    raise SystemExit(f"select_retention: no probe records under {args.probes}")

  by_arm = collections.defaultdict(list)
  for record in records:
    by_arm[record["arm"]].append(record)

  lines, winners = [], {}
  for arm in sorted(by_arm):
    rows = by_arm[arm]
    variants = sorted({r["variant"] for r in rows})
    designs = sorted({(r["seed"], r["design_index"]) for r in rows})

    per_design_mean, per_design_median, per_design_level, used = {}, {}, {}, []
    spans = {}
    for design in designs:
      cell = {r["variant"]: r for r in rows if (r["seed"], r["design_index"]) == design}
      missing = [v for v in variants if v not in cell]
      if len(missing) > 0:
        lines.append(f"  [skip] {arm} {design}: missing {missing}")
        continue
      curves = {v: best_so_far(cell[v]) for v in variants}
      # THE COMMON SUPPORT IS THE UNION OF ALL POINTS (`docs/final.md`). Every trajectory now runs
      # from n0 to the same budget because of its pseudo point, so the union is well defined and no
      # trajectory is ever extrapolated -- each is evaluated only inside its own range.
      grid = np.unique(np.concatenate([w for w, _ in curves.values()]))
      spans[design] = (
        float(grid[0]), float(grid[-1]), len(grid),
        min(variants, key=lambda v: curves[v][0][-2] if len(curves[v][0]) > 1 else curves[v][0][-1])
      )
      values = np.stack([step(grid, curves[v][0], curves[v][1]) for v in variants])
      pointwise = np.stack([rank(list(column)) for column in values.T])  # (grid, variants)

      # THE RANK INTEGRAL. Each grid point opens an interval of budget running to the next one, and the
      # ranks hold across it, so a setting's score is the AREA under its rank curve DIVIDED BY THE FULL
      # SPAN -- which keeps it in rank units, 1 to len(variants), directly readable as "this setting
      # was on average k-th across the budget".
      #
      # This is a budget-weighted average, NOT an average over grid points, and the distinction is not
      # cosmetic: the grid is never even. Growth rounds sit `n_increment` apart while the pseudo point
      # is far out at `iteration_limit`, so unweighted, a cluster of early closely-spaced additions
      # would outvote the entire stretch of budget after every setting has converged.
      width = np.diff(grid)
      span = float(width.sum())
      if span <= 0.0:
        lines.append(f"  [skip] {arm} {design}: zero-width support")
        continue
      used.append(design)
      for i, variant in enumerate(variants):
        per_design_mean.setdefault(variant, []).append(float(np.dot(pointwise[:-1, i], width) / span))
        per_design_median.setdefault(variant, []).append(weighted_median(pointwise[:-1, i], width))
        per_design_level.setdefault(variant, []).append(float(np.dot(values[i][:-1], width) / span))

    if len(used) == 0:
      lines.append(f"  [skip] {arm}: no usable design")
      continue

    lines.append(f"\n=== {args.task} / {arm} -- {len(used)} design(s), {len(variants)} settings ===")
    lines.append(
      "  %-16s %s  %8s %8s %11s" %
      ("setting", " ".join("%11s" % f"{s}@{i}" for s, i in used), "rank-int", "median", "mean level")
    )
    summary = []
    for variant in variants:
      summary.append((
        statistics.mean(per_design_mean[variant]), statistics.mean(per_design_median[variant]),
        statistics.mean(per_design_level[variant]), variant
      ))
    for mean_rank, median_rank, level, variant in sorted(summary):
      lines.append(
        "  %-16s %s  %8.3f %8.2f %11.5f" %
        (variant, " ".join("%11.3f" % v for v in per_design_mean[variant]), mean_rank, median_rank, level)
      )

    # The support is REPORTED, not assumed. If every setting converged at n0 the union is two points
    # (n0 and the budget) and the comparison rests on almost nothing, which must be visible.
    for design, (low, high, points, earliest) in sorted(spans.items()):
      lines.append(
        "  support %s@%d: windows [%d, %d], %d union point(s)%s; earliest to stop: %s" %
        (design[0], design[1], low, high, points, "  <-- DEGENERATE" if points <= 2 else "", earliest)
      )

    best, runner = sorted(summary)[0], sorted(summary)[1]
    winners[arm] = best[3]
    unresolved = best[:3] == runner[:3]
    lines.append(
      f"  -> {best[3]}  (rank integral {best[0]:.3f}, weighted median {best[1]:.2f})" + (
        f"; RUNNER-UP {runner[3]} at {runner[0]:.3f}, margin {runner[0] - best[0]:.3f}"
        if not unresolved else f"; ⚠️ TIED with {runner[3]} on every criterion -- selection is arbitrary"
      )
    )
    by_median = sorted(summary, key=lambda e: (e[1], e[0]))[0][3]
    if by_median != best[3]:
      lines.append(
        f"  ⚠️ NOT ROBUST TO THE STATISTIC: the rank integral selects {best[3]}, the weighted median would "
        f"select {by_median}. The integral is the rule; a flip means the ranks are skewed across the budget "
        f"rather than consistently favouring one setting."
      )

  text = "\n".join(lines)
  print(text)
  if args.report is not None:
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
      handle.write(text + "\n")

  if args.write:
    for arm, variant in winners.items():
      source = f"config/strategy/{args.task}-{arm}-{variant}.yaml"
      target = f"config/strategy/{args.task}-{arm}-optimal.yaml"
      # ⛔️ THE TARGET PATH CARRIES NO PREFIX, so two campaigns run at different output prefixes
      # resolve to the SAME file. That is forced by `docs/final.md` pinning the name, and snakemake
      # cannot catch it because these are undeclared side effects. So the provenance is written into
      # the file and checked on the way back in: overwriting a selection made from a DIFFERENT probe
      # directory is an error, not a silent replacement. `--force-write` is the deliberate override.
      previous = _provenance(target)
      if previous is not None and previous != args.probes and not args.force_write:
        raise SystemExit(
          f"select_retention: {target} was selected from probes under {previous!r}, but this run used "
          f"{args.probes!r}. Overwriting would silently repoint a campaign at another campaign's "
          f"settings. Pass --force-write if that is intended, or delete the file."
        )
      with open(source) as handle:
        body = handle.read()
      with open(target, "w") as handle:
        handle.write(
          f"# SELECTED-FROM: {args.probes}\n"
          f"# Written by scripts/select_retention.py, by the rank integral along the best-so-far curve\n"
          f"# over the union support. This is a COPY of {os.path.basename(source)}; edit THAT file and\n"
          f"# re-run the selection -- edits here are discarded on the next run.\n"
          f"# The `SELECTED-FROM:` line is load-bearing: it is what stops a second campaign at another\n"
          f"# output prefix from silently overwriting this selection.\n#\n"
        )
        handle.write(body)
      print(f"[write] {source} -> {target}")


if __name__ == "__main__":
  main()
