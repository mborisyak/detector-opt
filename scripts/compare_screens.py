#!/usr/bin/env python3
"""Read every screen result and report each candidate against ITS OWN criterion.

    python scripts/compare_screens.py                       # everything in output/screen
    python scripts/compare_screens.py --only inhibitor mm

The verdict is criterion (1) of docs/benchmark-acceptance.md, computed within a task against its own
baseline and its own resolution. THE DIAGNOSTICS DO NOT RANK CANDIDATES AGAINST EACH OTHER: two
candidates are two different problems with different targets and different loss distributions, so a
wider top decile on one says nothing about the other. They are printed only to explain a verdict.

What this adds over reading the JSONs by hand:

* it breaks the verdict into the FOUR conditions separately, with seed counts, so a failure names
  itself rather than arriving as a bare FAIL;
* it prints the two bars a task had to clear -- `0.2 * (baseline - loss@n)` and `10 * loss_precision`
  -- next to the gains, because a task can fail (d) for a reason that has nothing to do with its
  physics: if `10 * loss_precision` exceeds the whole baseline-to-best range, (d) is unsatisfiable by
  construction and the candidate should not be blamed for it;
* it projects the 5-seed neural campaign against the ~8 h figure from the measured per-design cost;
* it reports whether the provenance file exists, since a candidate without one is not considered at
  all (the gate, section 1.1).
"""
import argparse
import glob
import json
import os

import numpy as np


def line(width=100):
  print("-" * width)


def report(path):
  with open(path) as f:
    data = json.load(f)
  label = data.get("label", os.path.basename(path))
  it = data.get("iteration_test")
  print(f"\n=== {label.upper()}  ({os.path.basename(path)})")

  provenance = f"output/screen/provenance-{label}.md"
  print(f"  gate: provenance {'present' if os.path.isfile(provenance) else 'MISSING -- not considered'}"
        f" ({provenance})")

  for m, entry in sorted(data.get("landscape", {}).items()):
    print(f"  m={m} dim={entry['dimension']:2d} | baseline {entry.get('baseline_median_random', float('nan')):.4f}"
          f" best {entry['best']:.4f} ceiling {entry['ceiling']:.4f}"
          f" | at-ceiling {entry['ceiling_fraction_pct']:5.1f}%"
          f" top-decile {entry['top_decile_spread_pct']:5.1f}% of range"
          f" | GP R2 {entry['gp_r2']:+.3f} | {entry['seconds_per_design']:.2f} s/design")

  if it is None:
    print("  iteration test: NOT PRESENT (screen incomplete)")
    return None

  gains = np.array(it["gains"])
  at_n, at_2n = np.array(it["best_at_n"]), np.array(it["best_at_2n"])
  baseline, seeds = it["baseline_median_random"], it["seeds"]
  bar_c, bar_d = np.array(it["bar_c_per_seed"]), it["bar_d"]
  span = baseline - float(np.min(at_2n))

  print(f"\n  criterion at m={it['m']}, n={it['n']} -> {it['2n']}, {seeds} seeds"
        f"   (baseline {baseline:.4f})")
  print(f"    (a) loss@n < baseline .................. {it['seeds_beating_baseline']:2d}/{seeds}")
  print(f"    (b) loss@2n < loss@n ................... {it['seeds_improving']:2d}/{seeds}")
  print(f"    (c) gain > 0.2*(baseline - loss@n) ..... {it['seeds_enough_progress']:2d}/{seeds}"
        f"   median bar {np.median(bar_c):.4f}")
  print(f"    (d) gain > 10*loss_precision = {bar_d:<8.4f} {it['seeds_resolvable']:2d}/{seeds}")
  print(f"    STRONG (all four) ...................... {it['seeds_strong']:2d}/{seeds}  (need > {seeds/2:g})")
  print(f"    WEAK (a)+(b), EVERY seed ............... {it['seeds_weak']:2d}/{seeds}  (need {seeds})")
  print(f"    -> {'PASS' if it['PASS'] else 'FAIL'}")

  # Is (d) even reachable? If the whole useful range is smaller than the bar, the task cannot satisfy
  # it however good its physics -- that is a fact about loss_precision, not about the candidate.
  if bar_d > span:
    print(f"    !! (d) IS UNSATISFIABLE BY CONSTRUCTION: its bar {bar_d:.4f} exceeds the entire"
          f" baseline-to-best range {span:.4f}.")
    print(f"       Either loss_precision is too coarse for this task or the criterion needs the"
          f" task's own scale. Not a defect of the candidate.")
  elif bar_d > 0.5 * span:
    print(f"    !  (d)'s bar {bar_d:.4f} is over half the useful range {span:.4f} -- very demanding.")

  print(f"    gains   {np.array2string(gains, precision=4, max_line_width=90)}")

  bonus = data.get("bonus_bo_vs_random")
  if bonus is not None:
    print(f"    bonus: BO {bonus['bo_median_final']:.4f} vs random {bonus['random_median_final']:.4f},"
          f" BO better on {bonus['bo_better_on']}/{bonus['seeds']}")

  # Campaign cost: 5 seeds x 2 arms over 2 shards = 5 runs per shard.
  seconds = data["landscape"][str(it["m"])]["seconds_per_design"]
  print(f"    proxy cost {seconds:.2f} s/design. The neural objective is 300-1000x that per design,"
          f" so MEASURE it with a short bo.py run before sizing the campaign (docs 2.3).")
  return {"label": label, "pass": it["PASS"], "strong": it["seeds_strong"], "weak": it["seeds_weak"],
          "seeds": seeds, "unsatisfiable_d": bool(bar_d > span)}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default="output/screen")
  parser.add_argument("--only", nargs="*", default=None)
  arguments = parser.parse_args()

  paths = sorted(p for p in glob.glob(os.path.join(arguments.dir, "*.json"))
                 if arguments.only is None or any(k in os.path.basename(p) for k in arguments.only))
  if len(paths) == 0:
    raise SystemExit(f"no screen results in {arguments.dir}")

  summaries = [s for s in (report(p) for p in paths) if s is not None]
  line()
  print("SUMMARY -- each judged against its own criterion; these are NOT ranked against each other")
  for s in summaries:
    note = "   [(d) unsatisfiable by construction]" if s["unsatisfiable_d"] else ""
    print(f"  {s['label']:12s} {'PASS' if s['pass'] else 'FAIL'}   strong {s['strong']}/{s['seeds']}"
          f"   weak {s['weak']}/{s['seeds']}{note}")
  passing = [s["label"] for s in summaries if s["pass"]]
  print(f"\n  passing: {', '.join(passing) if len(passing) > 0 else 'none'}")
  if len(passing) > 1:
    print("  Several passed. They are NOT ranked by the diagnostics above -- choose on the"
          " qualitative criteria: realistic parameters, bounds stateable from the prior alone,"
          " low dimensionality, and campaign cost.")


if __name__ == "__main__":
  main()
