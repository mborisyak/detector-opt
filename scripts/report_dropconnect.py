#!/usr/bin/env python3
"""H9: does dropconnect add BIAS, and can the criterion see it?

    python scripts/report_dropconnect.py output/dropconnect/*.json

`linear` is the only task with a closed-form floor, so BIAS is `val - bayes_risk` MEASURED, not
inferred. The criterion's own signals -- `gap` and the settled test -- are exactly what the hypothesis
says are blind to bias, so they are reported alongside rather than relied on.

WHAT EACH COLUMN ANSWERS:

    excess_reported   val - floor. The bias the OPTIMISER sees. Rising with dropconnect = H9's claim.
    excess_test       held-out - floor. The bias the NETWORK actually has. If it tracks
                      `excess_reported`, the effect is a real biased network, not a reporting artefact.
    gap               |val - train|, the criterion's overfitting signal. If this does NOT rise while
                      the excesses do, the procedure cannot see what is happening -- H9 confirmed.
    window            what the procedure BOUGHT in response. If the error grows faster than the
                      window, the compensation is inadequate.

Everything is paired within (design, seed): a difference at fixed design and seed is the knob's.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np


def load(paths):
  rows = []
  for path in paths:
    with open(path) as f:
      for row in json.load(f).get("rows", []):
        if row.get("status") in (None, "converged"):
          rows.append(row)
  return rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+")
  arguments = parser.parse_args()

  rows = load(arguments.inputs)
  if len(rows) == 0:
    raise SystemExit("report_dropconnect: no converged rows")
  arms = sorted({r["arm"] for r in rows})
  dcs = sorted({r["dropconnect"] for r in rows})
  print(f"{len(rows)} converged cells | arms {arms} | dropconnect {dcs} | seeds {sorted({r.get('seed') for r in rows})}\n")

  print(f"{'arm':>14}{'dc':>7}{'n':>4}{'window':>9}{'gap':>10}{'err':>10}{'exc_rep':>11}{'exc_test':>11}{'val-test':>11}")
  table = {}
  for arm in arms:
    for dc in dcs:
      sub = [r for r in rows if r["arm"] == arm and r["dropconnect"] == dc]
      if len(sub) == 0:
        continue
      med = lambda k: float(np.median([r[k] for r in sub if r.get(k) is not None]))
      vt = float(np.median([abs(r["val"] - r["test"]) for r in sub if r.get("test") is not None]))
      table[(arm, dc)] = (med("window"), med("diff"), med("err"), med("excess_reported"), med("excess_test"))
      print(
        f"{arm:>14}{dc:>7}{len(sub):>4}{med('window'):>9.0f}{med('diff'):>10.5f}{med('err'):>10.5f}"
        f"{med('excess_reported'):>+11.5f}{med('excess_test'):>+11.5f}{vt:>11.5f}"
      )

  print("\nPAIRED AGAINST dropconnect = 0, within (arm, design, seed)")
  index = {(r["arm"], r["dropconnect"], r["design_index"], r.get("seed")): r for r in rows}
  print(f"{'arm':>14}{'dc':>7}{'pairs':>7}{'window x':>10}{'d gap':>10}{'d exc_rep':>12}{'d exc_test':>12}")
  for arm in arms:
    for dc in dcs:
      if dc == 0.0:
        continue
      pairs = [(index[(arm, dc, d, s)], index[(arm, 0.0, d, s)]) for (a, c, d, s) in index
               if a == arm and c == dc and (arm, 0.0, d, s) in index]
      if len(pairs) == 0:
        continue
      wx = np.median([a["window"] / b["window"] for a, b in pairs])
      dg = np.median([a["diff"] - b["diff"] for a, b in pairs])
      dr = np.median([a["excess_reported"] - b["excess_reported"] for a, b in pairs])
      dt = np.median([
        a["excess_test"] - b["excess_test"] for a, b in pairs
        if a.get("excess_test") is not None and b.get("excess_test") is not None
      ])
      print(f"{arm:>14}{dc:>7}{len(pairs):>7}{wx:>10.2f}{dg:>+10.5f}{dr:>+12.5f}{dt:>+12.5f}")

  print("\nH9 READING")
  print("  CONFIRMED if  d exc_test rises with dc  AND  d gap stays flat  AND  window x does not keep pace")
  print("  REFUTED   if  d gap rises with the excess (the criterion sees it) or the excesses stay flat")


if __name__ == "__main__":
  main()
