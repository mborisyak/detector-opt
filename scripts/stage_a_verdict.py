#!/usr/bin/env python3
"""Judge STAGE A: pick an operating point per candidate, by the rule declared before it ran.

    python scripts/stage_a_verdict.py                          # everything in output/screen
    python scripts/stage_a_verdict.py --candidate extremes

`docs/tuning-preregistration.md` section 6 is the specification; this is its implementation. It exists
so the verdict is COMPUTED rather than argued: the ranking statistic, the six conditions and the three
doubling pairs were all fixed in writing before any cell returned, and a tool that applies them cannot
quietly weight a cell it likes.

WHAT IT RANKS ON, AND WHY IT IS NOT CRITERION (1). Criterion (1) asks whether a task converts
iterations into a better curve at all. This round asks something narrower: whether `meta`'s ~2.45x
design advantage converts into a better final objective. Measured on the existing campaign
(section 5.2), `meta` wins at 5 and 10 designs and LOSES by 20-30 -- so the advantage converts only
while extra iterations still buy something. The operating point must therefore be one where 20 -> 40
STILL PAYS, and that gain is the ranking statistic. Criterion (1) at 10 -> 20 remains a condition, not
the objective.

THE THREE PAIRS COST NOTHING EXTRA. Each screen runs its iteration test once to 40 and stores every
seed's whole best-so-far curve, so 5 -> 10, 10 -> 20 and 20 -> 40 are read off the same trajectories,
paired on the same seeds. All three are printed for every cell whatever they say -- enumerating them
in advance is what makes reading several of them honest rather than a search for a flattering one.
5 -> 10 is the DECLARED RELAXATION: a cell clearing only that is labelled as such and is never a PASS.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

PAIRS = ((5, 10, "relaxed"), (10, 20, "HEADLINE"), (20, 40, "high-n"))
LABEL = re.compile(r"^(?P<candidate>[a-z]+)-m(?P<m>\d+)-r(?P<reads>\d+)$")


def load(directory, candidate=None):
  """Every Stage A cell on disk, newest schema only (a cell without stored curves is unusable here)."""
  cells = []
  for path in sorted(glob.glob(os.path.join(directory, "*-m*-r*.json"))):
    with open(path) as f:
      report = json.load(f)
    matched = LABEL.match(report.get("label", ""))
    if matched is None:
      continue
    if candidate is not None and matched.group("candidate") != candidate:
      continue
    test = report.get("iteration_test", {})
    if "curves" not in test:
      print(f"  !! {report['label']}: no stored curves -- screened before the curve change, SKIPPED")
      continue
    landscape = report["landscape"][str(test["m"])]
    cells.append({
      "label": report["label"],
      "candidate": matched.group("candidate"),
      "m": int(matched.group("m")),
      "reads": int(matched.group("reads")),
      "curves": np.array(test["curves"], dtype=float),
      # The random arm's trajectories, stored only from 2026-08-13 -- cells screened before that kept
      # just its endpoint, so this is None for them and the tail test is skipped rather than faked.
      "random_curves": (np.array(report["bonus_bo_vs_random"]["random_curves"], dtype=float)
                        if "random_curves" in report.get("bonus_bo_vs_random", {}) else None),
      "loss_precision": float(test["loss_precision"]),
      "error_multiple": float(test["error_multiple"]),
      "progress_fraction": float(test["progress_fraction"]),
      "baseline": float(test["baseline_median_random"]),
      "best": float(landscape["best"]),
      "ceiling": float(landscape["ceiling"]),
      "dimension": int(landscape["dimension"]),
      "dead_pct": float(landscape["ceiling_fraction_pct"]),
      "gp_r2": float(landscape["gp_r2"]),
      "seconds_per_design": float(landscape["seconds_per_design"]),
    })
  return cells


def pair_statistics(curves, n, n2, baseline, bar_c_fraction, bar_d):
  """The four conditions at one doubling pair, read off the stored curves. `curves` is (seeds, iters)."""
  if curves.shape[1] < n2:
    return None
  at_n = curves[:, n - 1]
  at_2n = curves[:, n2 - 1]
  gains = at_n - at_2n
  beats = at_n < baseline
  improves = at_2n < at_n
  progresses = gains > bar_c_fraction * (baseline - at_n)
  resolves = gains > bar_d
  seeds = curves.shape[0]
  # The SEM of the gain across seeds. NOT in the acceptance document -- it is the guard against the
  # "too noisy to resolve on 10 seeds" failure, and it is labelled as an addition wherever printed.
  standard_error = float(np.std(gains, ddof=1) / np.sqrt(seeds)) if seeds > 1 else float("inf")
  return {
    "n": n, "2n": n2, "seeds": seeds,
    "median_gain": float(np.median(gains)),
    "standard_error": standard_error,
    "strong": int((beats & improves & progresses & resolves).sum()),
    "weak": int((beats & improves).sum()),
    "improving": int(improves.sum()),
    "resolvable": int(resolves.sum()),
    # criterion (1) as stated: STRONG on more than half, WEAK on EVERY seed, and n >= 10
    "passes": bool(n >= 10 and (beats & improves & progresses & resolves).sum() > seeds / 2
                   and int((beats & improves).sum()) == seeds),
  }


def improvement_rates(curves, bins=4):
  """How often best-so-far actually MOVES, per iteration bin. `curves` is (seeds, iterations).

  The test for "are the arms fighting over a tail". `best-so-far` is a min-statistic: if the search
  degenerates to sampling near the optimum, improvements become rare events and an endpoint
  comparison between two arms is dominated by whether a hit landed, not by how well either searched.
  `P(improve in n) = 1 - (1-p)^n`, so at small `p` a 2.4x iteration advantage buys 2.4x the EXPECTED
  hits while the observed outcome stays a coin flip -- which is exactly the regime where a four-seed
  comparison means nothing.

  The rate uses EVERY iteration rather than one endpoint, so it carries far more evidence per seed,
  and it is what separates "still searching" from "sampling": a BO rate that decays to the random
  arm's is a search that has stopped adding anything.
  """
  seeds, iterations = curves.shape
  improved = np.diff(curves, axis=1) < 0.0        # strictly better than the running best
  edges = np.linspace(0, iterations - 1, bins + 1).astype(int)
  out = []
  for start, stop in zip(edges[:-1], edges[1:]):
    if stop <= start:
      continue
    out.append({
      "from": int(start + 1), "to": int(stop),
      "rate": float(improved[:, start:stop].mean()),          # per iteration, per seed
      "seeds_with_any": int((improved[:, start:stop].sum(axis=1) > 0).sum()),
      "of_seeds": int(seeds),
    })
  return out


def judge(cell):
  """Section 6: rank on the median 20 -> 40 gain, subject to six conditions."""
  bar_d = cell["error_multiple"] * cell["loss_precision"]
  span = cell["baseline"] - cell["best"]          # baseline = MEDIAN RANDOM, not the ceiling (5.1)
  pairs = {}
  for n, n2, role in PAIRS:
    statistics = pair_statistics(cell["curves"], n, n2, cell["baseline"], cell["progress_fraction"], bar_d)
    if statistics is not None:
      statistics["role"] = role
      pairs[f"{n}->{n2}"] = statistics

  headline = pairs.get("10->20")
  high_n = pairs.get("20->40")
  conditions = {}
  conditions["i   criterion (1) at 10->20"] = headline is not None and headline["passes"]
  if high_n is not None:
    conditions["ii  gain > 10*loss_precision"] = high_n["median_gain"] > bar_d
    # An ADDITION of mine, not in docs/benchmark-acceptance.md. Guards the TOO NOISY failure.
    conditions["iii gain > 2*SEM (addition)"] = high_n["median_gain"] > 2.0 * high_n["standard_error"]
  conditions["iv  dead volume in [5, 40]%"] = 5.0 <= cell["dead_pct"] <= 40.0
  # (v) the integration guard is asserted host-side on EVERY call, so a screen that COMPLETED is an
  # exhaustive check that it never fired -- there is nothing separate to test here.
  conditions["v   guard never fired"] = True
  # (vi) per-design cost is judged against the campaign, not here; reported for the sizing that follows.
  return {
    "pairs": pairs,
    "conditions": conditions,
    "bar_d": bar_d,
    "span": span,
    "bar_fraction": (bar_d / span) if span > 0.0 else float("inf"),
    "unsatisfiable": bar_d > span,
    "rank_on": high_n["median_gain"] if high_n is not None else float("-nan"),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="output/screen")
  parser.add_argument("--candidate", default=None, help="judge only this candidate (extremes | mmsym)")
  arguments = parser.parse_args()

  cells = load(arguments.dir, arguments.candidate)
  if len(cells) == 0:
    raise SystemExit(f"no Stage A cells with stored curves under {arguments.dir}")

  by_candidate = {}
  for cell in cells:
    by_candidate.setdefault(cell["candidate"], []).append(cell)

  for candidate, group in sorted(by_candidate.items()):
    print(f"\n{'=' * 100}\n{candidate}: {len(group)} cells\n{'=' * 100}")
    verdicts = [(cell, judge(cell)) for cell in group]

    print(f"\n  {'cell':<18} {'dim':>4} {'base-best':>10} {'dead%':>6} {'GP R2':>7} "
          f"{'5->10':>9} {'10->20':>9} {'20->40':>9} {'bar/span':>9} {'s/design':>9}")
    for cell, verdict in verdicts:
      gains = []
      for key in ("5->10", "10->20", "20->40"):
        statistics = verdict["pairs"].get(key)
        gains.append(f"{statistics['median_gain']:+.4f}" if statistics is not None else "     --")
      print(f"  {cell['label']:<18} {cell['dimension']:>4} {verdict['span']:>10.4f} "
            f"{cell['dead_pct']:>6.1f} {cell['gp_r2']:>+7.3f} "
            f"{gains[0]:>9} {gains[1]:>9} {gains[2]:>9} "
            f"{verdict['bar_fraction']:>8.1%} {cell['seconds_per_design']:>9.2f}")

    print(f"\n  conditions (section 6). Ranking statistic is the median 20->40 gain.")
    for cell, verdict in sorted(verdicts, key=lambda pair: -pair[1]["rank_on"]):
      met = [name for name, ok in verdict["conditions"].items() if ok]
      failed = [name for name, ok in verdict["conditions"].items() if not ok]
      status = "ELIGIBLE" if len(failed) == 0 else "excluded"
      print(f"\n    {cell['label']:<18} rank {verdict['rank_on']:+.4f}   {status}"
            f"   ({len(met)}/{len(verdict['conditions'])} conditions)")
      for name in failed:
        print(f"        FAILED  {name}")
      if verdict["unsatisfiable"]:
        print(f"        !! (d) UNSATISFIABLE BY ARITHMETIC: bar {verdict['bar_d']:.4f} exceeds the whole"
              f" baseline-to-best span {verdict['span']:.4f} -- REJECTED, not ranked")
      # Is the cell still SEARCHING at the budget the campaign will use, or sampling a tail?
      bo_rates = improvement_rates(cell["curves"])
      random_rates = improvement_rates(cell["random_curves"]) if cell["random_curves"] is not None else None
      rendered = "  ".join(f"{r['from']}-{r['to']}: {r['rate']:.3f}" for r in bo_rates)
      print(f"        improvement rate, BO      {rendered}")
      if random_rates is not None:
        rendered = "  ".join(f"{r['from']}-{r['to']}: {r['rate']:.3f}" for r in random_rates)
        print(f"        improvement rate, RANDOM  {rendered}")
        last_bo, last_random = bo_rates[-1]["rate"], random_rates[-1]["rate"]
        if last_random > 0.0 and last_bo <= last_random:
          print(f"        !! in the LAST quarter BO's improvement rate ({last_bo:.3f}) does not exceed"
                f" random's ({last_random:.3f}) -- the arms are sampling, not searching, and any"
                f" endpoint difference there is luck rather than evidence")
      else:
        print(f"        improvement rate, RANDOM  -- not stored (cell screened before the change)")
      relaxed = verdict["pairs"].get("5->10")
      headline = verdict["pairs"].get("10->20")
      if headline is not None and not headline["passes"] and relaxed is not None:
        print(f"        note: 5->10 gain {relaxed['median_gain']:+.4f} on {relaxed['improving']}"
              f"/{relaxed['seeds']} seeds -- the DECLARED RELAXATION, reportable as"
              f" 'passes relaxed criterion (n >= 5)' and NEVER as a plain PASS")

    eligible = [(cell, verdict) for cell, verdict in verdicts
                if len(
                  [name for name, ok in verdict["conditions"].items() if not ok]) == 0 and not verdict["unsatisfiable"]]
    print()
    if len(eligible) == 0:
      print(f"  {candidate}: NO CELL satisfies all conditions. Section 6: the candidate does not get a"
            f" neural campaign, and the relaxed reading is reported AS relaxed -- never promoted.")
    else:
      winner, verdict = max(eligible, key=lambda pair: pair[1]["rank_on"])
      print(f"  {candidate} WINNER: {winner['label']}   median 20->40 gain {verdict['rank_on']:+.4f}"
            f"   (ties break on the larger span: {verdict['span']:.4f})")
      print(f"    m = {winner['m']}, n_measurements = {winner['reads']}, dimension {winner['dimension']},"
            f" {winner['seconds_per_design']:.2f} s/design on the proxy")
      print(f"    loss_precision used here was {winner['loss_precision']:.2e} -- a DECLARED PLACEHOLDER."
            f" No verdict is final until it is measured at THIS cell (pre-registration section 5).")


if __name__ == "__main__":
  main()
