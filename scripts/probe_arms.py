#!/usr/bin/env python3
"""The PAIRED table across probe arms -- `scripts/probe_precision.py` outputs, one JSON per arm.

    python scripts/probe_arms.py --baseline grow=output/screen/s12-grow.json \
        --reference fixed=output/screen/s12-fixed.json \
        --arm prior=output/screen/s12-prior.json --arm mix=output/screen/s12-mix05.json

WHAT IT IS FOR. Every arm scores the SAME (design, seed) cells from the SAME initial network, so the
only honest read is a PAIRED one: differences within a cell, then a mean over cells with the spread
beside it. Run-to-run drift on this task is 0.003-0.009 at nominally identical settings -- the size of
the effects being reported -- and pairing is what removes it. An arm's own mean slack is not a result.

WHAT IT REPORTS, per arm and then per pair against `--baseline`:

* `status`      -- CONVERGED cells are CENSORED at the run's own `loss_precision` (they prove
                   `slack <= precision`, not where the gap actually sits); CAPPED cells are
                   uncensored. `bo.py` RAISES on a cap, so the capped COUNT is the operationally
                   binding number, not the mean gap.
* `slack`       -- `|val - train| + hypot(train_sem, val_sem)` at the stopping epoch, the quantity
                   `loss_precision` is compared against.
* `level`       -- `(train + val) / 2`, the number BO consumes. Reported for every cell, including
                   capped ones (where the trainer returns no objective), so the two statuses are
                   comparable.
* `CEILING`     -- the failure mode a regulariser produces and the convergence procedure CANNOT catch:
                   a network pinned near its prior has a settled training loss and a ~zero gap, which
                   is exactly what the procedure returns on. Flagged when a cell's level is within
                   `--ceiling-margin` of the no-information level while the REFERENCE arm at the same
                   design sits below `--ceiling-reference`.
* `init`        -- the initial-parameter checksum. Two arms that do not agree cell-for-cell are not
                   paired, and their difference carries initialisation noise; this is checked, not
                   assumed.

DETECTOR CALLS. A capped cell records `spent = -1` (the trainer raises before returning), so calls are
reconstructed from the window it reached: `window * (1 + val_fraction / (1 - val_fraction))`. Both
forms are printed; a reconstructed one is marked with `~`.

This script SETS NOTHING and decides nothing. The verdict rule it feeds is written down in
`docs/decision-log.md` (D72) BEFORE the numbers existed.
"""

import argparse
import json
import math

import numpy as np


def load_arm(spec):
  """``name=path`` -> ``(name, {(design, seed): row})``, keeping the LAST row written for a cell."""
  name, _, path = spec.partition("=")
  if len(path) == 0:
    raise SystemExit(f"probe_arms: --arm wants `name=path`, got {spec!r}")
  with open(path) as f:
    payload = json.load(f)
  cells = {}
  for row in payload["rows"]:
    cells[(row["design"], int(row["seed"]))] = row
  return name, cells, path


def level(row):
  """``(train + val) / 2`` from the row's own last window, defined on capped cells too."""
  if "train" not in row or "val" not in row:
    return float("nan")
  return 0.5 * (float(row["train"]) + float(row["val"]))


def calls(row, val_fraction):
  """Detector calls the cell spent, and whether it had to be reconstructed."""
  spent = int(row.get("spent", -1))
  if spent > 0:
    return float(spent), False
  window = int(row.get("window", -1))
  if window <= 0:
    return float("nan"), True
  return window * (1.0 + val_fraction / (1.0 - val_fraction)), True


def summarise(name, cells, val_fraction):
  converged = [c for c in cells.values() if c["status"] == "converged"]
  capped = [c for c in cells.values() if c["status"] == "capped"]
  other = [c for c in cells.values() if c["status"] not in ("converged", "capped")]
  slacks = [float(c["slack"]) for c in cells.values() if "slack" in c]
  spend = [calls(c, val_fraction)[0] for c in cells.values()]
  print(
    f"\n{name}: {len(cells)} cells | converged {len(converged)} | CAPPED {len(capped)}" +
    (f" | OTHER {len(other)} ({sorted({c['status'] for c in other})})" if len(other) > 0 else "")
  )
  if len(slacks) > 0:
    print(f"  slack   max {max(slacks):.5f}  mean {np.mean(slacks):.5f}")
  if len(capped) > 0:
    print(f"  capped at: {sorted((c['design'], int(c['seed'])) for c in capped)}")
  if len(spend) > 0 and not all(math.isnan(s) for s in spend):
    print(f"  calls/design  mean {np.nanmean(spend):.0f}  max {np.nanmax(spend):.0f}")


def paired(baseline_name, baseline, arm_name, arm, val_fraction):
  """One arm against the baseline, cell by cell, then the mean and the ACROSS-SEED spread."""
  shared = sorted(set(baseline) & set(arm))
  if len(shared) == 0:
    print(f"\n{arm_name} - {baseline_name}: NO SHARED CELLS")
    return
  print(
    f"\n{arm_name} - {baseline_name}, paired over {len(shared)} cells "
    f"(negative slack = the arm closes the gap; negative level = the arm scores LOWER, i.e. better)"
  )
  print(
    f"  {'design':>14} {'seed':>10}  {'status':>19}  {'d slack':>9} {'d level':>9} "
    f"{'d train':>9} {'d val':>9} {'d epochs':>9} {'d calls':>10}"
  )
  rows = []
  for key in shared:
    base, other = baseline[key], arm[key]
    if abs(float(base.get("init_checksum", 0.0)) - float(other.get("init_checksum", 0.0))) > 1e-6:
      print(
        f"  ⚠️  {key[0]:>12} s{key[1]:<9} INIT MISMATCH "
        f"({base.get('init_checksum')} vs {other.get('init_checksum')}) -- NOT PAIRED"
      )
      continue
    base_calls, base_derived = calls(base, val_fraction)
    other_calls, other_derived = calls(other, val_fraction)
    delta = {
      "slack": float(other.get("slack", float("nan"))) - float(base.get("slack", float("nan"))),
      "level": level(other) - level(base),
      "train": float(other.get("train", float("nan"))) - float(base.get("train", float("nan"))),
      "val": float(other.get("val", float("nan"))) - float(base.get("val", float("nan"))),
      "epochs": float(other.get("n_epochs", -1)) - float(base.get("n_epochs", -1)),
      "calls": other_calls - base_calls,
    }
    rows.append((key, delta))
    marker = "~" if base_derived or other_derived else " "
    print(
      f"  {key[0]:>14} {key[1]:>10}  {base['status']:>9}->{other['status']:<9}  "
      f"{delta['slack']:>9.5f} {delta['level']:>9.5f} {delta['train']:>9.5f} {delta['val']:>9.5f} "
      f"{delta['epochs']:>9.0f} {delta['calls']:>9.0f}{marker}"
    )
  if len(rows) == 0:
    return
  print(
    f"  {'MEAN':>14} {'':>10}  {'':>19}  " + " ".join(
      f"{np.nanmean([d[k] for _, d in rows]):>9.5f}" if k not in ("epochs",
                                                                  "calls") else f"{np.nanmean([d[k] for _, d in rows]):>9.0f}"
      for k in ("slack", "level", "train", "val", "epochs", "calls")
    )
  )
  print(
    f"  {'SD':>14} {'':>10}  {'':>19}  " + " ".join(
      f"{np.nanstd([d[k] for _, d in rows], ddof=1):>9.5f}" if k not in
      ("epochs", "calls") else f"{np.nanstd([d[k] for _, d in rows], ddof=1):>9.0f}"
      for k in ("slack", "level", "train", "val", "epochs", "calls")
    )
  )
  # UNCENSORED subset: a pair where both sides converged is two values pressed against the same bar,
  # so its slack difference understates the effect. The capped-in-either subset is where the gap is
  # measured rather than bounded.
  uncensored = [(key, d) for key, d in rows if baseline[key]["status"] == "capped" or arm[key]["status"] == "capped"]
  if 0 < len(uncensored) < len(rows):
    values = [d["slack"] for _, d in uncensored]
    # A spread needs two values; with one uncensored pair the mean IS the whole evidence and saying so
    # is more honest than printing a nan beside it.
    spread = f"sd {np.nanstd(values, ddof=1):>9.5f}" if len(values) > 1 else "(one pair, no spread)"
    print(
      f"  on the {len(uncensored)} pairs CAPPED on at least one side: d slack mean "
      f"{np.nanmean(values):>9.5f} {spread}"
    )


def ceilings(arms, reference_name, reference, no_information, margin, reference_below):
  """Cells that converged AT the no-information level while the reference arm did not."""
  print(
    f"\nCEILING CHECK against the no-information level {no_information:.3f} "
    f"(flag: level > {no_information - margin:.3f} where {reference_name} < {reference_below:.3f})"
  )
  flagged = 0
  for name, cells, _ in arms:
    for key, row in sorted(cells.items()):
      if key not in reference:
        continue
      if level(row) > no_information - margin and level(reference[key]) < reference_below:
        flagged += 1
        print(
          f"  ⚠️  {name:>8} {key[0]:>14} s{key[1]:<10} level {level(row):.4f} "
          f"({row['status']}, window {row.get('window', -1)}) against {reference_name} "
          f"{level(reference[key]):.4f}"
        )
  if flagged == 0:
    print("  none")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--baseline", required=True, metavar="NAME=PATH",
    help="the arm every other one is differenced against (the growth baseline)"
  )
  parser.add_argument(
    "--reference", default=None, metavar="NAME=PATH",
    help="the arm the LEVEL is judged against (the fixed window). Also differenced "
    "against the baseline, and used by the ceiling check"
  )
  parser.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
  parser.add_argument(
    "--val-fraction", type=float, default=0.25, help="only used to reconstruct detector calls for CAPPED cells, which record "
    "spent = -1; must match the run config's own value"
  )
  parser.add_argument(
    "--no-information", type=float, default=1.0, help="the task's no-information loss -- EXACTLY 1.0 for the binary extremes "
    "task (cross-entropy / ln 2), the target variance for a regression task"
  )
  parser.add_argument(
    "--ceiling-margin", type=float, default=0.03, help="how close to the no-information level counts as collapsed"
  )
  parser.add_argument(
    "--ceiling-reference", type=float, default=0.93,
    help="the reference arm must be BELOW this at the same design for the flag to "
    "mean the arm collapsed rather than the design being uninformative"
  )
  arguments = parser.parse_args()

  baseline_name, baseline, baseline_path = load_arm(arguments.baseline)
  arms = [(baseline_name, baseline, baseline_path)]
  reference_name, reference = None, {}
  if arguments.reference is not None:
    reference_name, reference, reference_path = load_arm(arguments.reference)
    arms.append((reference_name, reference, reference_path))
  arms.extend(load_arm(spec) for spec in arguments.arm)

  print("ARMS")
  for name, cells, path in arms:
    print(f"  {name:>10}  {len(cells):>3} cells  {path}")
    summarise(name, cells, arguments.val_fraction)

  for name, cells, _ in arms[1:]:
    paired(baseline_name, baseline, name, cells, arguments.val_fraction)

  if reference_name is not None:
    ceilings(arms, reference_name, reference, arguments.no_information, arguments.ceiling_margin, arguments.ceiling_reference)

  print(
    "\nREAD IT AS: a CONVERGED cell is censored at its own `loss_precision`; only a CAPPED cell "
    "reports where the gap actually sits. The capped COUNT is what decides whether a campaign can "
    "run at all, and the LEVEL is what decides whether the arm is admissible."
  )


if __name__ == "__main__":
  main()
