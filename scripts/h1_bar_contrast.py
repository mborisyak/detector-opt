#!/usr/bin/env python3
"""H1 part 2: does the reported loss's error scale with `loss_precision`?

    python scripts/h1_bar_contrast.py output/biascross/linear_precision.json --task linear
    python scripts/h1_bar_contrast.py output/biascross/mm_precision.json --task mm

THE QUANTITY. Every cell is PAIRED with the same cell at a different bar -- same design, same arm,
same seed, same patience, same dropconnect, everything but `loss_precision`. Within such a pair the
design's true difficulty is identical by construction, so a change in what the criterion reports is
the criterion's, not the design's.

TWO ERROR MEASURES, because only one task has a floor:

  deviation   `|val - test|`, the reported value against a large held-out sample at the SAME design.
              Available on every task. ⚠️ Carries the held-out sample's OWN error, so on tasks where
              that error is comparable to the deviation the comparison cannot resolve anything and is
              excluded rather than averaged in.
  excess      `val - bayes_risk`, EXACT on `linear` and undefined elsewhere. Strictly better where it
              exists: the floor is closed-form, so there is no reference-sample error at all. Reported
              alongside `err` (the criterion's own standard-error term), because the ratio
              `excess / err` is the sharpest statement of the mechanism -- if the reported loss sits
              one `err` above the floor, then since `err = c / sqrt(window)` and the gate accepts at
              `gap + err <= bar`, the excess MUST scale with the bar.

H1 PREDICTS `deviation` falls with the bar, roughly in proportion, because the exit gate
`gap + err <= loss_precision` is satisfied at `err ~ c / sqrt(window)` and so buys precision in the
estimate rather than quality in the network. The falsifier is a `deviation` that is flat in the bar:
that would mean the bar buys data without buying accuracy, and the reported error lives somewhere
the stopping rule does not control.

⚠️ The window ratio is reported alongside because it is the mechanism's own check: halving the bar
should roughly quadruple the window, subject to the growth ladder's quantisation. A pair whose
window did NOT move is not evidence about the bar at all and is flagged.
"""

from __future__ import annotations

import argparse
import collections
import json
import math


def load(paths):
  """Converged rows carrying a bar, from any number of probe outputs.

  ⚠️ Rows whose `status` is not `converged` are DROPPED and counted, never analysed. A design that hit
  the per-design window cap did not reach the bar, so its window and excess are properties of the cap,
  not of the criterion. Reporting one as a measurement is the failure this project names as "errors are
  not results"."""
  rows, dropped = [], []
  for path in paths:
    with open(path) as f:
      payload = json.load(f)
    for row in payload.get("rows", []):
      if row.get("loss_precision") is None:
        continue
      if row.get("status") not in (None, "converged"):
        dropped.append(row)
        continue
      rows.append(row)
  if len(dropped) > 0:
    print(f"⚠️ {len(dropped)} row(s) DROPPED as not converged -- excluded from every table below:")
    for row in dropped:
      print(
        f"     design {row.get('design_index')} bar {float(row['loss_precision']):g} {row.get('arm')} "
        f"seed {row.get('seed')} window {row.get('window')} status={row.get('status')}"
      )
    print()
  return rows


def key_of(row):
  """Everything that must match within a pair EXCEPT the bar."""
  return (
    row.get("design_index"), row.get("arm"), row.get("seed"), row.get("patience"), row.get("dropconnect"),
    row.get("weight_decay")
  )


def deviation(row):
  """`|val - test|` -- the reported value against an independent sample at the same design."""
  val, test = row.get("val"), row.get("test")
  if val is None or test is None:
    return None
  return abs(float(val) - float(test))


def resolvable(row, sigmas=2.0):
  """Whether `|val - test|` exceeds `sigmas` times the held-out sample's OWN standard error.

  A deviation smaller than the error of the thing measuring it carries no information, and a RATIO of
  two such deviations carries less than none -- it is noise over noise. Pairs failing this are printed
  and excluded from the summary rather than silently averaged in."""
  dev, sem = deviation(row), row.get("test_sem")
  if dev is None or sem in (None, 0):
    return None
  return dev >= sigmas * float(sem)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+")
  parser.add_argument("--task", required=True)
  arguments = parser.parse_args()

  rows = load(arguments.inputs)
  if len(rows) == 0:
    raise SystemExit("h1_bar_contrast: no rows carry `loss_precision`")

  bars = sorted({float(r["loss_precision"]) for r in rows}, reverse=True)
  groups = collections.defaultdict(dict)
  for row in rows:
    groups[key_of(row)][float(row["loss_precision"])] = row

  print(f"{arguments.task}: {len(rows)} cells, bars {bars}, {len(groups)} distinct cells\n")
  print("cells present per bar:")
  for bar in bars:
    print(f"  {bar:<10g} {sum(1 for g in groups.values() if bar in g)}")

  pairs = []
  for key, by_bar in groups.items():
    present = sorted(by_bar, reverse=True)
    for loose, tight in zip(present, present[1:]):
      pairs.append((key, loose, tight, by_bar[loose], by_bar[tight]))

  if len(pairs) == 0:
    print("\nNO PAIRED CELLS YET -- a contrast needs the same design/arm/seed at two bars.")
    return

  floor_rows = [r for r in rows if r.get("bayes_risk") is not None and r.get("err") not in (None, 0)]
  if len(floor_rows) > 0:
    print(f"\nEXACT-FLOOR VIEW ({len(floor_rows)} cells carry `bayes_risk`; no reference-sample error)")
    print(f"{'design':>7}{'bar':>9}{'arm':>13}{'seed':>5}{'window':>9}{'excess':>11}{'err':>10}{'excess/err':>12}")
    for r in sorted(floor_rows, key=lambda r: (r["design_index"], -float(r["loss_precision"]), str(r["arm"]))):
      ratio = float(r["excess_reported"]) / float(r["err"])
      print(
        f"{r['design_index']:>7}{float(r['loss_precision']):>9g}{str(r['arm']):>13}{str(r.get('seed')):>5}"
        f"{r['window']:>9}{float(r['excess_reported']):>+11.5f}{float(r['err']):>10.5f}{ratio:>12.2f}"
      )

  print(f"\n{len(pairs)} paired contrasts (same design, arm, seed; only the bar differs)\n")
  header = (
    f"{'design':>7}{'arm':>13}{'seed':>5}{'loose':>9}{'tight':>9}{'bar x':>7}"
    f"{'win loose':>11}{'win tight':>11}{'win x':>7}{'dev loose':>11}{'dev tight':>11}{'dev x':>7}"
  )
  print(header)
  ratios, unresolved = [], []
  for key, loose, tight, a, b in sorted(pairs, key=lambda p: (p[1], p[0][0], str(p[0][1]))):
    dev_a, dev_b = deviation(a), deviation(b)
    win_x = b["window"] / a["window"] if a["window"] > 0 else float("nan")
    dev_x = dev_b / dev_a if dev_a not in (None, 0) and dev_b is not None else float("nan")
    flag = "  <-- window unchanged" if a["window"] == b["window"] else ""
    print(
      f"{key[0]:>7}{str(key[1]):>13}{str(key[2]):>5}{loose:>9g}{tight:>9g}{loose / tight:>7.1f}"
      f"{a['window']:>11}{b['window']:>11}{win_x:>7.2f}"
      f"{(dev_a if dev_a is not None else float('nan')):>11.5f}"
      f"{(dev_b if dev_b is not None else float('nan')):>11.5f}{dev_x:>7.2f}{flag}"
    )
    ok_a, ok_b = resolvable(a), resolvable(b)
    if ok_a is False or ok_b is False:
      unresolved.append((key[0], key[1], loose, tight))
      continue
    if a["window"] != b["window"] and not math.isnan(dev_x):
      ratios.append((loose / tight, win_x, dev_x))

  if len(unresolved) > 0:
    print(f"\n⚠️ {len(unresolved)} pair(s) EXCLUDED: |val - test| below 2x its own test_sem, so the ratio")
    print("   would be noise over noise. Excluded, not averaged in:")
    for design, arm, loose, tight in unresolved:
      print(f"     design {design} {arm} {loose:g} -> {tight:g}")

  if len(ratios) == 0:
    print("\nNO RESOLVABLE PAIRS. Nothing here bears on the bar -- every deviation is inside the")
    print("held-out estimator's own error. More held-out rows, not more cells, is what would fix it.")
    return

  bar_x = sorted(r[0] for r in ratios)
  win_x = sorted(r[1] for r in ratios)
  dev_x = sorted(r[2] for r in ratios)
  median = lambda v: v[len(v) // 2] if len(v) % 2 == 1 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
  print(f"\nover {len(ratios)} pairs whose window moved:")
  print(f"  bar tightened by      x{median(bar_x):.2f}   (median)")
  print(f"  window grew by        x{median(win_x):.2f}   (H1 mechanism predicts ~x{median(bar_x)**2:.1f})")
  print(f"  deviation changed by  x{median(dev_x):.2f}   (H1 predicts ~x{1.0 / median(bar_x):.2f}; flat = x1.00 = REFUTED)")
  fell = sum(1 for r in ratios if r[2] < 1.0)
  print(f"  deviation FELL in {fell} of {len(ratios)} pairs")


if __name__ == "__main__":
  main()
