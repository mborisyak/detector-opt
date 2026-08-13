#!/usr/bin/env python3
"""Judge a finished NEURAL campaign against criterion (1) of docs/benchmark-acceptance.md.

    python scripts/campaign_criterion.py output/campaign-inhib --n 13 --loss-precision 6.0e-3

This is the counterpart of `scripts/compare_screens.py`, which judges the PROXY screen. The criterion
is identical; only the objective differs, and that is the whole point of running the campaign -- the
proxy is a stand-in (Spearman 0.940 on the melt task), and acceptance is stated on the real thing.

WHERE THE BASELINE COMES FROM. Criterion (a) needs "the median loss of random designs" measured by
the SAME objective, and re-running a random-search arm would cost another campaign. It is not needed:
BO's first `n_init` proposals ARE uniform draws in the scaled cube (scripts/bo.py builds them from a
Sobol sequence before any GP exists), so pooling those blocks across seeds gives `n_init x n_seeds`
independent random designs scored by the neural objective, for free and with no extra assumption. The
pool is reported with its size so a thin one cannot masquerade as a solid baseline.

WHAT IT REFUSES TO DO. `--n` is an INPUT, not something chosen here from the curves: it was fixed in
output/screen/addendum-inhibitor-n.md on cost grounds before any run finished. A seed that did not
reach `2n` designs is reported SHORT rather than evaluated at whatever it did reach, because judging
each seed at its own doubling would let the run length be picked per seed after the fact.
"""
import argparse
import glob
import json
import os

import numpy as np


def load(directory):
  """Every completed run under `directory`, as (seed, arm, losses, n_init)."""
  runs = []
  for path in sorted(glob.glob(os.path.join(directory, "*", "*", "results.json"))):
    with open(path) as f:
      data = json.load(f)
    seed, arm = path.split(os.sep)[-3], path.split(os.sep)[-2]
    losses = np.array([r["loss"] for r in data["results"]], dtype=float)
    # `spent` in results.json is PER-DESIGN (verified: it is not monotone, and it sums to the run's
    # total), so the running budget is its cumulative sum. This is what makes an equal-BUDGET
    # comparison between arms possible at all -- see `compare_arms`, where cutting at equal design
    # count would delete the very effect being measured.
    spent = np.cumsum(np.array([r.get("spent", 0) for r in data["results"]], dtype=float))
    runs.append({
      "seed": seed,
      "arm": arm,
      "losses": losses,
      "spent": spent,
      "completed": bool(data.get("completed")),
      "calls": int(data.get("detector_calls_used", 0))
    })
  return runs


def compare_arms(runs, cut_designs, reference_arm):
  """`meta` against `from_scratch` AT EQUAL DETECTOR-CALL BUDGET -- the measurement the campaign is for.

  Criterion (1), which `judge` implements, asks whether a task converts iterations into a better curve
  WITHIN one arm. It cannot answer the question this campaign exists for, which is whether `meta`'s
  known advantage -- more designs per unit detector budget -- turns into a better final objective.

  CUT AT EQUAL SPEND, NOT AT EQUAL DESIGN COUNT. `meta` buys roughly 2.3x the designs for the same
  detector calls precisely BY spending fewer calls on each, so cutting both arms at the same design
  number hands `meta` a budget it never had and deletes the effect. The budget at each cut is taken
  from the REFERENCE arm's own trajectory, per seed, so neither arm is judged at a budget it never
  reached.

  The cut points are an INPUT, declared in docs/tuning-preregistration.md before the campaign ran, and
  every one of them is reported whatever it says. Scanning a trajectory for the budget at which `meta`
  happens to win, and reporting that one, is the threshold-after-the-result move the acceptance
  document bans -- enumerating the cuts in advance is what makes post-hoc cutting honest.
  """
  seeds = sorted({r["seed"] for r in runs})
  arms = sorted({r["arm"] for r in runs})
  if reference_arm not in arms:
    print(f"\n  arm comparison SKIPPED: no '{reference_arm}' arm under this directory (found {arms})")
    return
  others = [a for a in arms if a != reference_arm]
  if len(others) == 0:
    print(f"\n  arm comparison SKIPPED: only the '{reference_arm}' arm is present")
    return

  print(f"\n  ARM COMPARISON at equal detector-call budget   (reference: {reference_arm})")
  for cut in cut_designs:
    rows, wins, differences = [], 0, []
    for seed in seeds:
      by_arm = {r["arm"]: r for r in runs if r["seed"] == seed}
      if reference_arm not in by_arm:
        continue
      reference = by_arm[reference_arm]
      if len(reference["losses"]) < cut:
        rows.append((seed, f"SHORT: {reference_arm} reached {len(reference['losses'])} designs < {cut}"))
        continue
      budget = float(reference["spent"][cut - 1])
      cell = {}
      for arm in arms:
        if arm not in by_arm:
          continue
        run = by_arm[arm]
        within = run["spent"] <= budget
        if int(within.sum()) == 0:
          continue
        cell[arm] = (float(np.min(run["losses"][within])), int(within.sum()))
      if len(cell) < 2:
        rows.append((seed, "only one arm reached this budget"))
        continue
      rows.append((seed, cell))
      for arm in others:
        if arm in cell:
          difference = cell[reference_arm][0] - cell[arm][0]   # > 0 means the other arm is BETTER
          differences.append(difference)
          if difference > 0.0:
            wins += 1

    print(f"\n    cut at {reference_arm} design {cut}:")
    for seed, cell in rows:
      if isinstance(cell, str):
        print(f"      {seed:>12s}  {cell}")
        continue
      rendered = "   ".join(f"{arm} {loss:.4f} ({designs}d)" for arm, (loss, designs) in sorted(cell.items()))
      print(f"      {seed:>12s}  {rendered}")
    if len(differences) > 0:
      differences = np.array(differences, dtype=float)
      print(f"      -> median advantage {np.median(differences):+.4f}"
            f"   better on {wins}/{len(differences)} seeds"
            f"   (positive = the non-reference arm wins)")
    else:
      print("      -> no seed had both arms at this budget")


def judge(runs, n, bar_d, progress_fraction, baseline, label):
  """The four conditions, per seed, at `n -> 2n`. Returns nothing; prints the verdict."""
  n2 = 2 * n
  usable = [r for r in runs if len(r["losses"]) >= n2]
  short = [r for r in runs if len(r["losses"]) < n2]

  print(f"\n  criterion at n = {n} -> {n2}   ({label})")
  for r in short:
    print(f"    SHORT: seed {r['seed']} reached only {len(r['losses'])} designs -- not evaluated")
  if len(usable) == 0:
    print("    no seed reached 2n; nothing to judge")
    return

  at_n = np.array([np.min(r["losses"][:n]) for r in usable])
  at_2n = np.array([np.min(r["losses"][:n2]) for r in usable])
  gains = at_n - at_2n
  bar_c = progress_fraction * (baseline - at_n)

  beats = at_n < baseline
  improves = at_2n < at_n
  progresses = gains > bar_c
  resolves = gains > bar_d
  strong = beats & improves & progresses & resolves
  weak = beats & improves
  k = len(usable)

  print(f"    (a) loss@n < baseline .................. {int(beats.sum()):2d}/{k}")
  print(f"    (b) loss@2n < loss@n ................... {int(improves.sum()):2d}/{k}")
  print(
    f"    (c) gain > {progress_fraction}*(baseline - loss@n) ..... {int(progresses.sum()):2d}/{k}"
    f"   median bar {np.median(bar_c):.4f}"
  )
  print(f"    (d) gain > 10*loss_precision = {bar_d:<8.4f} {int(resolves.sum()):2d}/{k}")
  print(f"    STRONG (all four) ...................... {int(strong.sum()):2d}/{k}"
        f"  (need > {k / 2:g})")
  print(f"    WEAK (a)+(b), EVERY seed ............... {int(weak.sum()):2d}/{k}  (need {k})")
  passed = bool(strong.sum() > k / 2 and weak.sum() == k and len(short) == 0)
  print(f"    -> {'PASS' if passed else 'FAIL'}")
  print(f"    loss@n  {np.array2string(at_n, precision=4, max_line_width=90)}")
  print(f"    gains   {np.array2string(gains, precision=4, max_line_width=90)}")

  span = baseline - float(np.min(at_2n))
  if bar_d > span:
    print(
      f"    !! (d) IS UNSATISFIABLE BY CONSTRUCTION: bar {bar_d:.4f} exceeds the whole"
      f" baseline-to-best range {span:.4f}"
    )
  elif bar_d > 0.5 * span:
    print(f"    !  (d)'s bar {bar_d:.4f} is over half the useful range {span:.4f} -- very demanding")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("directory")
  parser.add_argument(
    "--n", type=int, required=True, help="the DECLARED evaluation point; must have been fixed before the runs finished"
  )
  parser.add_argument(
    "--secondary-n", type=int, default=None, help="a second n printed for comparison ONLY, never used for the verdict"
  )
  parser.add_argument("--loss-precision", type=float, required=True)
  parser.add_argument("--error-multiple", type=float, default=10.0)
  parser.add_argument("--progress-fraction", type=float, default=0.2)
  parser.add_argument("--n-init", type=int, default=5, help="BO's random block, = bo.n_init")
  parser.add_argument("--arm", default=None, help="judge only this arm")
  parser.add_argument(
    "--compare-arms-at", type=int, nargs="*", default=[],
    help="DECLARED design counts at which to compare arms at EQUAL DETECTOR-CALL BUDGET. Every value "
         "given is reported whatever it says; they must have been fixed before the runs finished "
         "(docs/tuning-preregistration.md 7.1)."
  )
  parser.add_argument("--reference-arm", default="from_scratch", help="the arm whose spend defines each budget cut")
  arguments = parser.parse_args()

  runs = load(arguments.directory)
  if arguments.arm is not None:
    runs = [r for r in runs if r["arm"] == arguments.arm]
  if len(runs) == 0:
    raise SystemExit(f"no runs under {arguments.directory}")

  for r in runs:
    flag = "" if r["completed"] else "  (INCOMPLETE -- still running)"
    print(
      f"  {r['seed']:>12s} {r['arm']:<13s} {len(r['losses']):3d} designs  best {r['losses'].min():.4f}"
      f"  {r['calls']} calls{flag}"
    )

  # The baseline: BO's own n_init block is a uniform draw, so pooling it across seeds IS a random
  # sample of designs scored by this objective.
  random_block = np.concatenate([r["losses"][:arguments.n_init] for r in runs])
  baseline = float(np.median(random_block))
  print(
    f"\n  baseline = median of {len(random_block)} random designs (the pooled n_init blocks)"
    f" = {baseline:.4f}   [{random_block.min():.4f}, {random_block.max():.4f}]"
  )

  bar_d = arguments.error_multiple * arguments.loss_precision
  judge(runs, arguments.n, bar_d, arguments.progress_fraction, baseline, "THE VERDICT")
  if arguments.secondary_n is not None:
    judge(
      runs, arguments.secondary_n, bar_d, arguments.progress_fraction, baseline,
      "SECONDARY -- for comparison with the proxy screen, NOT the verdict"
    )
  if len(arguments.compare_arms_at) > 0:
    compare_arms(runs, arguments.compare_arms_at, arguments.reference_arm)


if __name__ == "__main__":
  main()
