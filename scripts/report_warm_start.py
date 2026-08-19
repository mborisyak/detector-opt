#!/usr/bin/env python3
"""Read the warm-start probe cells and decide between the two hypotheses for why the warm arms spend more.

    python scripts/report_warm_start.py output/warmstart/*.json

THE QUESTION. On `extremes`, `meta` spends 24-37% FEWER detector calls per design than `from_scratch`
while `continue` and `closest` -- which also warm-start -- spend 7-44% MORE. Warm-starting should be
at worst neutral for sample efficiency, so spending more is the anomaly.

  A  GAP INFLATION     the warm network memorises a small window fast, `gap` opens, the gate buys data.
                       Signature: warm arms have LOWER early train loss and a LARGER gap at exit.
  B  NEGATIVE TRANSFER the warm start comes from a mismatched design and must be unlearned.
                       Signature: warm arms have HIGHER early train loss and MORE epochs before exit.

The discriminating column is the EARLY train loss, printed at epoch 1 and at the end of the first
growth round. Both hypotheses predict a larger window; only A predicts the warm arms start LOWER.

Everything is paired within (design, seed): the arms score the identical designs in the identical
order from the identical history, so a difference between arms at one (design, seed) is the arm's.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

ARM_ORDER = ("from_scratch", "continue", "closest", "meta")


def load(paths):
  rows = []
  for path in paths:
    with open(path) as f:
      for row in json.load(f)["rows"]:
        rows.append(row)
  return rows


def early(row, key):
  """`key` at epoch 1 and at the end of the first growth round."""
  series = np.asarray(row["per_epoch"][key], np.float64)
  window = np.asarray(row["per_epoch"]["window"], np.int64)
  first = int(np.argmax(window != window[0])) if np.any(window != window[0]) else series.size
  return float(series[0]), float(series[max(first - 1, 0)])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+")
  arguments = parser.parse_args()

  rows = [r for r in load(arguments.inputs) if r.get("status") == "converged"]
  if len(rows) == 0:
    raise SystemExit("report_warm_start: no converged rows")
  arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]
  print(f"{len(rows)} converged designs, arms {arms}, seeds {sorted({r['seed'] for r in rows})}\n")

  print("PER ARM, median over all (design, seed)")
  header = (
    f"{'arm':>14}{'n':>4}{'window':>9}{'epochs':>8}{'rounds':>8}{'gap':>9}{'err':>9}"
    f"{'train ep1':>11}{'train r1':>10}{'train':>9}{'val':>9}{'test':>9}"
  )
  print(header)
  for arm in arms:
    sub = [r for r in rows if r["arm"] == arm]
    e1 = np.median([early(r, "train")[0] for r in sub])
    er = np.median([early(r, "train")[1] for r in sub])
    med = lambda k: np.median([r[k] for r in sub if r[k] is not None])
    print(
      f"{arm:>14}{len(sub):>4}{med('window'):>9.0f}{med('n_epochs'):>8.0f}{med('n_rounds'):>8.0f}"
      f"{med('gap'):>9.5f}{med('err'):>9.5f}{e1:>11.4f}{er:>10.4f}{med('train'):>9.4f}"
      f"{med('val'):>9.4f}{med('test'):>9.4f}"
    )

  base = "from_scratch"
  if base not in arms:
    return
  print(f"\nPAIRED AGAINST {base}, within (design, seed)")
  index = {(r["arm"], r["design_index"], r["seed"]): r for r in rows}
  print(f"{'arm':>14}{'pairs':>7}{'d window':>11}{'window x':>10}{'d gap':>10}{'d err':>10}{'d train ep1':>13}{'d test':>10}")
  for arm in arms:
    if arm == base:
      continue
    pairs = [(index[(arm, d, s)], index[(base, d, s)]) for (a, d, s) in index if a == arm and (base, d, s) in index]
    if len(pairs) == 0:
      continue
    dw = np.array([a["window"] - b["window"] for a, b in pairs], float)
    wx = np.array([a["window"] / b["window"] for a, b in pairs], float)
    dg = np.array([a["gap"] - b["gap"] for a, b in pairs], float)
    de = np.array([a["err"] - b["err"] for a, b in pairs], float)
    d1 = np.array([early(a, "train")[0] - early(b, "train")[0] for a, b in pairs], float)
    dt = np.array([a["test"] - b["test"] for a, b in pairs if a["test"] is not None and b["test"] is not None], float)
    print(
      f"{arm:>14}{len(pairs):>7}{np.median(dw):>+11.0f}{np.median(wx):>10.2f}{np.median(dg):>+10.5f}"
      f"{np.median(de):>+10.5f}{np.median(d1):>+13.4f}{(np.median(dt) if dt.size else float('nan')):>+10.5f}"
    )

  print("\nREADING IT")
  print("  A (gap inflation)     expects  d window > 0,  d gap > 0,  d train ep1 < 0")
  print("  B (negative transfer) expects  d window > 0,  d gap ~ 0,  d train ep1 > 0")


if __name__ == "__main__":
  main()
