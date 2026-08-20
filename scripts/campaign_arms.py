#!/usr/bin/env python3
"""Arm comparison for a multi-seed campaign, at MATCHED design count.

    python scripts/campaign_arms.py --campaign output/campaign-emnist

Reads ``<campaign>/<seed>/<arm>/results.json`` and answers the two questions a fixed-budget arm
comparison actually poses, which the runs' own `best_loss` does not:

  COST     how many designs each arm bought with the shared budget, and what it spent per design.
           The arms differ here, so the budget is not a level playing field.
  LOSS     best-so-far at a design count every arm reached. `best_loss` is a minimum over draws, so
           an arm that scored more designs is flattered by the comparison it appears to win.

The initial ``n_init`` designs are proposed before the surrogate has any data, so where the arms
share them identically they form a PAIRED sample: a difference there is the arm's, not the design's.
That pairing is detected rather than assumed, and reported separately from the diverged tail.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ARM_ORDER = ("from_scratch", "continue", "closest", "meta")


def load(campaign):
  """``{arm: {seed: [rows]}}`` over every ``results.json`` under the campaign root."""
  runs = {}
  for path in sorted(glob.glob(os.path.join(campaign, "*", "*", "results.json"))):
    seed, arm = path.split(os.sep)[-3:-1]
    rows = json.load(open(path)).get("results", [])
    if len(rows) > 0:
      runs.setdefault(arm, {})[seed] = rows
  return runs


def paired_prefix(runs, seeds):
  """Number of leading designs every arm scored identically, per seed."""
  shared = {}
  for seed in seeds:
    present = [runs[a][seed] for a in runs if seed in runs[a]]
    if len(present) < 2:
      continue
    n, k = min(len(r) for r in present), 0
    while k < n and all(np.allclose(present[0][k]["x_scaled"], r[k]["x_scaled"], atol=1e-6) for r in present[1:]):
      k += 1
    shared[seed] = k
  return shared


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--campaign", required=True)
  parser.add_argument("--at", type=int, default=None, help="design count to compare at; default the common minimum")
  arguments = parser.parse_args()

  runs = load(arguments.campaign)
  if len(runs) == 0:
    raise SystemExit(f"campaign_arms: no results.json under {arguments.campaign}")
  arms = [a for a in ARM_ORDER if a in runs] + sorted(set(runs) - set(ARM_ORDER))
  seeds = sorted({s for a in runs for s in runs[a]})
  complete = [s for s in seeds if all(s in runs[a] for a in arms)]
  print(f"{arguments.campaign}: {len(arms)} arms, {len(seeds)} seeds, {len(complete)} with every arm present\n")

  print("COST")
  print(f"{'arm':>14}{'runs':>6}{'designs':>10}{'calls/design':>14}{'total calls':>13}")
  for arm in arms:
    designs = [len(r) for r in runs[arm].values()]
    spent = [x["spent"] for r in runs[arm].values() for x in r]
    total = [sum(x["spent"] for x in r) for r in runs[arm].values()]
    print(f"{arm:>14}{len(designs):>6}{np.mean(designs):>10.1f}{np.mean(spent):>14.0f}{np.mean(total):>13.0f}")

  if len(complete) == 0:
    print("\nno seed has every arm yet -- loss comparison needs one")
    return

  at = arguments.at if arguments.at is not None else min(len(runs[a][s]) for a in arms for s in complete)
  print(f"\nBEST-SO-FAR AT {at} DESIGNS, per seed (the only comparison the budget makes fair)")
  print(f"{'arm':>14}" + "".join(f"{s[:8]:>11}" for s in complete) + f"{'median':>11}")
  for arm in arms:
    best = [min(x["loss"] for x in runs[arm][s][:at]) for s in complete]
    print(f"{arm:>14}" + "".join(f"{b:>11.4f}" for b in best) + f"{np.median(best):>11.4f}")

  shared = paired_prefix(runs, complete)
  k = min(shared.values()) if len(shared) > 0 else 0
  if k > 0:
    print(f"\nPAIRED on the {k} identical leading designs, difference from {arms[0]} in trained loss")
    print(f"{'arm':>14}{'n':>5}{'mean diff':>12}{'sd':>10}{'signs +/-':>12}")
    base = [x["trained_loss"] for s in complete for x in runs[arms[0]][s][:k]]
    for arm in arms[1:]:
      d = np.asarray([x["trained_loss"] for s in complete for x in runs[arm][s][:k]]) - np.asarray(base)
      sd = d.std(ddof=1) if d.size > 1 else float("nan")
      print(f"{arm:>14}{d.size:>5}{d.mean():>+12.4f}{sd:>10.4f}{f'{int((d > 0).sum())}/{int((d < 0).sum())}':>12}")


if __name__ == "__main__":
  main()
