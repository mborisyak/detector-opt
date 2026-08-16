#!/usr/bin/env python3
"""Report for :mod:`probe_trajectory_midpoint`: `meta` against `from_scratch` at one mid-trajectory
design, paired on seed.

    python scripts/probe_trajectory_midpoint_report.py output/midpoint/mid_*.json

WHAT IS PAIRED AND WHAT IS NOT. Detector calls and validation loss are paired on SEED -- same design,
same event slice, same validation split, same network initialisation -- so the per-seed ratio is the
unit of measurement and the summary is the mean of its LOGARITHM, which is what a ratio's spread is
symmetric in. Test loss is additionally paired PER ROW over the identical test pool, so its difference
carries a per-row standard error and is reported as a difference rather than a ratio.

THE RATIO IS `from_scratch` / `meta` in calls, so ABOVE 1 MEANS `meta` IS CHEAPER, matching the sign
convention of every earlier report in this series.

The resolution stated before the runs: one growth round is ~10% of `meta`'s level, so a single pair
resolves no better than that, and three repetitions give SE = sd/sqrt(3) on the log ratio. This script
prints that SE from the data rather than assuming the pre-registered value, and says plainly when the
spread is too large for the number of repetitions to separate the campaign's 1.3-1.7x claim.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load(paths):
  rows = []
  for path in paths:
    with open(path) as f:
      rows.extend(json.load(f)["rows"])
  return rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+", help="one or more probe_trajectory_midpoint JSON outputs")
  parser.add_argument("--claim-low", type=float, default=1.3, help="low end of the campaign's reported advantage")
  parser.add_argument("--claim-high", type=float, default=1.7, help="high end of the campaign's reported advantage")
  arguments = parser.parse_args()

  rows = load(arguments.inputs)
  if len(rows) == 0:
    raise SystemExit("probe_trajectory_midpoint_report: no rows")
  by_cell = {(r["arm"], r["seed"]): r for r in rows}
  seeds = sorted({r["seed"] for r in rows})
  reference = rows[0]

  print(
    f"trajectory {reference['trajectory']} | scored design index {reference['design_index']} "
    f"(loss on that trajectory {reference['trajectory_loss']:.4f})"
  )
  print(f"history {reference['history_rows']} train rows, sampled from the designs before it\n")

  header = (
    f"{'seed':>10} {'arm':>13} {'calls':>9} {'window':>8} {'rounds':>7} "
    f"{'val':>8} {'test':>8} {'diff':>8} {'err':>8} {'slack':>8} {'status':>14}"
  )
  print(header)
  print("-" * len(header))
  for seed in seeds:
    for arm in ("from_scratch", "meta"):
      row = by_cell.get((arm, seed))
      if row is None:
        print(f"{seed:>10} {arm:>13} {'MISSING':>9}")
        continue
      print(
        f"{seed:>10} {arm:>13} {row['calls_train_val']:>9d} {row['window']:>8d} {row['n_rounds']:>7d} "
        f"{row['val']:>8.4f} {row['test']:>8.4f} {row['diff']:>8.5f} {row['err']:>8.5f} "
        f"{row['slack']:>8.5f} {row['status']:>14}"
      )

  paired = [s for s in seeds if ("meta", s) in by_cell and ("from_scratch", s) in by_cell]
  if len(paired) == 0:
    print("\nno complete pair yet -- nothing to compare")
    return

  print(f"\nPAIRED, {len(paired)} repetition(s). ratio = from_scratch / meta, so > 1 means meta is cheaper.")
  print(f"{'seed':>10} {'calls ratio':>12} {'window ratio':>13} {'val diff':>10} {'test diff':>10}")
  ratios, val_differences, test_differences = [], [], []
  for seed in paired:
    meta, scratch = by_cell[("meta", seed)], by_cell[("from_scratch", seed)]
    ratio = scratch["calls_train_val"] / meta["calls_train_val"]
    window_ratio = scratch["window"] / meta["window"]
    val_difference = meta["val"] - scratch["val"]
    test_difference = meta["test"] - scratch["test"]
    ratios.append(ratio)
    val_differences.append(val_difference)
    test_differences.append(test_difference)
    print(f"{seed:>10} {ratio:>12.3f} {window_ratio:>13.3f} {val_difference:>+10.4f} {test_difference:>+10.4f}")

  log_ratio = np.log(np.asarray(ratios, np.float64))
  mean_ratio = float(np.exp(np.mean(log_ratio)))
  if len(paired) > 1:
    sd = float(np.std(log_ratio, ddof=1))
    sem = sd / np.sqrt(len(paired))
    low, high = np.exp(np.mean(log_ratio) - 2 * sem), np.exp(np.mean(log_ratio) + 2 * sem)
    print(f"\ngeometric mean ratio {mean_ratio:.3f}, log sd {sd:.4f}, SE {sem:.4f} -> 2-sigma [{low:.3f}, {high:.3f}]")
    separates = (low > arguments.claim_high) or (high < arguments.claim_low)
    inside = arguments.claim_low <= mean_ratio <= arguments.claim_high
    verdict = "CONSISTENT with" if inside else ("BELOW" if mean_ratio < arguments.claim_low else "ABOVE")
    print(
      f"campaign claim {arguments.claim_low}-{arguments.claim_high}x: measured mean is {verdict} it; "
      f"{'the interval EXCLUDES the claimed band' if separates else 'the interval OVERLAPS the claimed band'}"
    )
    if not separates and not inside:
      print("  -> the spread is too large at this many repetitions to call it either way")
  else:
    print(f"\nratio {mean_ratio:.3f} from ONE pair -- no spread, and one growth round is ~10% of the level")

  test_sem = np.mean([by_cell[("meta", s)]["test_sem"] for s in paired])
  mean_test = float(np.mean(test_differences))
  print(
    f"\ntest loss, meta - from_scratch: mean {mean_test:+.4f} over {len(paired)} pair(s) "
    f"(per-row SEM ~{test_sem:.4f}); negative means meta scores the design BETTER"
  )
  print(f"validation loss, meta - from_scratch: mean {float(np.mean(val_differences)):+.4f}")

  charged = [by_cell[("meta", s)]["history_calls_charged"] for s in paired]
  if len(charged) > 0 and charged[0] > 0:
    saving = [by_cell[("from_scratch", s)]["calls_train_val"] - by_cell[("meta", s)]["calls_train_val"] for s in paired]
    mean_saving = float(np.mean(saving))
    if mean_saving > 0:
      print(
        f"\nECONOMICS: meta saves {mean_saving:.0f} calls per design here, against {charged[0]} calls of "
        f"history charged, so that history repays over {charged[0] / mean_saving:.1f} designs"
      )
    else:
      print(
        f"\nECONOMICS: meta saves nothing here ({mean_saving:.0f} calls), so the {charged[0]} calls of "
        f"history never repay"
      )
    print(
      "  (a campaign accumulates its history as a by-product of designs it wanted scored anyway, so "
      "this charge is what the history would cost if bought on purpose, not what the campaign paid)"
    )


if __name__ == "__main__":
  main()
