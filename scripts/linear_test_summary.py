#!/usr/bin/env python3
"""Per-rung summary of the linear ladder's TEST stage (``output/linear/<rung>/test/<seed>/<arm>``): seeds present
per arm, final best-so-far loss (mean +- SE over seeds), paired differences against ``meta`` on the seeds both arms
share, and the budget-weighted integral rank of the arms (select_retention's construction, ranked within each seed
among the arms present, averaged over seeds), on the REPORTED curves (results.json) and on the VERIFIED curves
(verification.json of cells carrying verified.txt). Cells without results.json or without verified.txt are listed
as missing, so a partial tree reads as partial.

    python scripts/linear_test_summary.py [output/linear] [--rungs d1n2,d4n5]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_retention import best_so_far, left_constant, ranks, verified_curve

ARMS = ("from_scratch", "continue", "meta")


def mean_se(values):
  n = len(values)
  if n == 0:
    return float("nan"), float("nan")
  m = sum(values) / n
  if n == 1:
    return m, float("nan")
  return m, (sum((v - m)**2 for v in values) / (n - 1) / n)**0.5


def integral_ranks(curves, budget):
  arms = [a for a in ARMS if curves.get(a) is not None]
  if len(arms) < 2:
    return {}
  grid = sorted({v for a in arms for v in curves[a][0]})
  sampled = {a: left_constant(grid, *curves[a]) for a in arms}
  widths = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
  total = {a: 0.0 for a in arms}
  for i, w in enumerate(widths):
    for a, r in zip(arms, ranks([sampled[a][i] for a in arms])):
      total[a] += r * w
  return {a: total[a] / budget for a in arms}


def load_rung(root, rung):
  seeds = sorted(os.listdir(os.path.join(root, rung, "test")))
  reported, verified, missing = {}, {}, []
  budget = None
  for seed in seeds:
    for arm in ARMS:
      cell = os.path.join(root, rung, "test", seed, arm)
      rp = os.path.join(cell, "results.json")
      if not os.path.exists(rp):
        missing.append(f"{seed}/{arm}: no results")
        continue
      payload = json.load(open(rp))
      budget = int(payload["config"]["training"]["budget"])
      if not payload.get("completed", False):
        missing.append(f"{seed}/{arm}: results incomplete ({len(payload['results'])} designs)")
        continue
      reported.setdefault(seed, {})[arm] = best_so_far(payload["results"], budget)
      vp = os.path.join(cell, "verification.json")
      if not os.path.exists(os.path.join(cell, "verified.txt")) or not os.path.exists(vp):
        missing.append(f"{seed}/{arm}: not verified")
        continue
      verified.setdefault(seed, {})[arm] = verified_curve(json.load(open(vp)), budget)
  return seeds, reported, verified, missing, budget


def summarise(label, curves_by_seed, budget):
  finals = {a: {s: c[a][1][-1] for s, c in curves_by_seed.items() if a in c} for a in ARMS}
  print(f"  {label}:")
  print(f"    {'arm':<13}{'seeds':>6}{'final best':>20}    {'paired vs meta (meta - arm)':<34}")
  for a in ARMS:
    m, se = mean_se(list(finals[a].values()))
    line = f"    {a:<13}{len(finals[a]):>6}{m:>12.4f} +- {se:.4f}"
    if a != "meta":
      shared = [s for s in finals[a] if s in finals["meta"]]
      d = [finals["meta"][s] - finals[a][s] for s in shared]
      dm, dse = mean_se(d)
      wins = sum(1 for v in d if v < 0)
      line += f"    {dm:+.4f} +- {dse:.4f}  meta better {wins}/{len(d)}"
    print(line)
  per_seed = [integral_ranks(c, budget) for c in curves_by_seed.values()]
  full = [r for r in per_seed if len(r) == len(ARMS)]
  print(f"    integral rank ({len(full)} seeds with all {len(ARMS)} arms):", "  ".join(
      f"{a} {mean_se([r[a] for r in full])[0]:.2f} +- {mean_se([r[a] for r in full])[1]:.2f}" for a in ARMS) if len(full) > 0 else "n/a")


def main():
  root = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "output/linear"
  rungs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "test")))
  if "--rungs" in sys.argv:
    rungs = sys.argv[sys.argv.index("--rungs") + 1].split(",")
  for rung in rungs:
    seeds, reported, verified, missing, budget = load_rung(root, rung)
    print(f"{rung}: {len(seeds)} seeds, budget {budget}; missing/partial cells: {len(missing)}")
    for m in missing:
      print(f"    - {m}")
    summarise("reported", reported, budget)
    summarise("verified", verified, budget)


if __name__ == "__main__":
  main()
