#!/usr/bin/env python3
"""Paired report for `scripts/probe_exact_perturbed_mix.py`.

    python scripts/probe_exact_perturbed_mix_report.py output/mixprobe

READS the `control_s*.json` / `dual_s*.json` written by the probe (plus their `_test_rows.npz`) and
pairs them on `(design, seed)`. NOTHING here re-runs anything and nothing is fitted: every number is a
difference between two measured runs of one pair, or a mean of such differences.

WHAT IS REPORTED

  * PER PAIR: detector calls to convergence, growth rounds, the final window, `diff` and `err` at
    stopping, converged-or-capped, the COMBINED current loss (train and validation), the EXACT-ONLY and
    PERTURBED-ONLY current losses, and the held-out test loss at the exact design.
  * THE PAIRED MEAN DIFFERENCE with its standard error, for calls (absolute and as a log ratio), growth
    rounds, and the held-out test loss. The log ratio is the primary statistic because the level varies
    several-fold across designs and a plain difference of calls would be dominated by whichever design
    happened to be dearest.
  * THE HELD-OUT TEST DIFFERENCE PER ROW. Both arms score the SAME 32768 rows, so the difference is
    paired row by row and its error is far smaller than either arm's own SEM.
  * THE EXACT-ONLY COUNTERFACTUAL. The probe's convergence criterion pools the current exact and
    perturbed rows; the alternative reading is the criterion on the exact rows alone. This replays the
    SACRED procedure -- unchanged, from `detopt.utils.training` -- over the per-epoch series the run
    actually recorded, using only the exact-only means and SEMs. THE REPLAY IS FAITHFUL ONLY WHILE ITS
    GROWTH DECISIONS COINCIDE WITH THE ONES THE RUN REALLY TOOK: the moment the replayed rule would have
    grown where the run trained on (or the reverse), the data after that epoch is no longer the data
    that rule would have seen, and the replay is reported as DIVERGED rather than extrapolated. Replayed
    against the COMBINED series it must reproduce the run's own stopping epoch exactly, and that check is
    printed as a control.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from detopt.utils.training import (bayesian_trend, probability_above, probability_change_below)


def load(root, arm):
  """Every measured row of one arm, keyed by `(design, seed)`, plus its per-row held-out losses."""
  rows, per_row = {}, {}
  for path in sorted(glob.glob(os.path.join(root, f"{arm}_s*.json"))):
    with open(path) as f:
      payload = json.load(f)
    for row in payload["rows"]:
      rows[(row["design"], int(row["seed"]))] = row
    npz_path = os.path.splitext(path)[0] + "_test_rows.npz"
    if os.path.isfile(npz_path):
      with np.load(npz_path) as data:
        for key in data.files:
          design, seed = key.split("|")[0], int(key.split("|")[1])
          per_row[(design, seed)] = np.asarray(data[key])
  return rows, per_row


def mean_sem(values):
  values = np.asarray(values, np.float64)
  n = values.shape[0]
  if n == 0:
    return float("nan"), float("nan")
  if n == 1:
    return float(values[0]), float("nan")
  return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n))


def replay(row, part, warmup, patience, precision):
  """Replay the convergence procedure over a run's recorded per-epoch series.

  `part` selects the series: None takes the pooled numbers the run actually gated on, an int takes that
  current region alone (0 = exact, 1 = perturbed). Returns
  `(outcome, epoch, window, diff, err)` with outcome one of

    stopped   the rule returned at `epoch`, and every growth decision up to it matched the run's own;
    diverged  the rule would have grown where the run did not (or the reverse) -- everything after
              `epoch` would have been different data, so nothing is claimed past it;
    no-stop   the rule never returned within the epochs the run recorded.
  """
  per_epoch = row["per_epoch"]
  if part is None:
    train = np.asarray(per_epoch["train"], np.float64)
    val = np.asarray(per_epoch["val"], np.float64)
    train_sem = np.asarray(per_epoch["train_sem"], np.float64)
    val_sem = np.asarray(per_epoch["val_sem"], np.float64)
  else:
    train = np.asarray([p[part] for p in per_epoch["train_parts"]], np.float64)
    val = np.asarray([p[part] for p in per_epoch["val_parts"]], np.float64)
    train_sem = np.asarray([p[part] for p in per_epoch["train_sem_parts"]], np.float64)
    val_sem = np.asarray([p[part] for p in per_epoch["val_sem_parts"]], np.float64)
  window = per_epoch["window"]
  actual_round_start = per_epoch["round_start"]

  round_start, epoch_in_round = 0, 0
  for i in range(train.shape[0]):
    epoch_in_round += 1
    err = float(np.hypot(train_sem[i], val_sem[i]))
    diff = float(abs(val[i] - train[i]))
    if epoch_in_round <= warmup:
      continue
    first = round_start + warmup
    tr, va = train[first:i + 1], val[first:i + 1]
    tr_s, va_s = train_sem[first:i + 1], val_sem[first:i + 1]
    if tr.shape[0] < 3:
      continue
    prior_sigma = max(float(tr[0]), float(va[0])) / 3.0
    gap_sem = np.hypot(tr_s, va_s)
    gap_series = np.abs(va - tr) + gap_sem
    tr_mean, tr_cov = bayesian_trend(tr, tr_s, prior_sigma)
    gap_mean, gap_cov = bayesian_trend(gap_series, gap_sem, prior_sigma)
    if probability_above(gap_mean, gap_cov, patience, precision, gap_series.shape[0]) > 0.9:
      decision = "grow"
    elif probability_change_below(tr_mean, tr_cov, patience, 0.5 * precision) > 0.9:
      decision = "grow" if diff + err > precision else "return"
    else:
      continue
    if decision == "return":
      return "stopped", i, int(window[i]), diff, err
    grew = i + 1 < train.shape[0] and int(actual_round_start[i + 1]) == i + 1
    if not grew:
      return "diverged", i, int(window[i]), diff, err
    round_start, epoch_in_round = i + 1, 0
  return "no-stop", train.shape[0] - 1, int(window[-1]), diff, err


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("root", nargs="?", default="output/mixprobe", help="directory holding control_s*.json / dual_s*.json")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--patience", type=int, default=16)
  parser.add_argument("--precision", type=float, default=1.0e-2)
  parser.add_argument("--n-increment", type=int, default=4096)
  arguments = parser.parse_args()

  control, control_rows = load(arguments.root, "control")
  dual, dual_rows = load(arguments.root, "dual")
  keys = sorted(set(control) & set(dual), key=lambda k: (k[1], k[0]))
  print(f"{len(control)} control cells, {len(dual)} dual cells, {len(keys)} PAIRS\n")
  if len(keys) == 0:
    return

  header = (
    f"{'design':>7} {'seed':>4} | {'calls':>7} {'rnd':>4} {'window':>7} {'diff':>7} {'err':>7} {'status':>9} "
    f"{'train':>7} {'val':>7} {'test':>7} | {'calls':>7} {'rnd':>4} {'window':>7} {'diff':>7} {'err':>7} "
    f"{'status':>9} {'train':>7} {'val':>7} {'trainE':>7} {'valE':>7} {'trainP':>7} {'valP':>7} {'test':>7} | "
    f"{'dcalls':>7} {'lnratio':>8} {'dtest':>8}"
  )
  print("                | " + "CONTROL".center(78) + " | " + "DUAL".center(110) + " | PAIRED")
  print(header)

  ratios, deltas, round_deltas, test_deltas, row_deltas = [], [], [], [], []
  for design, seed in keys:
    a, b = control[(design, seed)], dual[(design, seed)]
    ratio = math.log(b["calls_train_val"] / a["calls_train_val"])
    ratios.append(ratio)
    deltas.append(b["calls_train_val"] - a["calls_train_val"])
    round_deltas.append(b["n_rounds"] - a["n_rounds"])
    test_deltas.append(b["test"] - a["test"])
    if (design, seed) in control_rows and (design, seed) in dual_rows:
      row_deltas.append(float(np.mean(dual_rows[(design, seed)] - control_rows[(design, seed)])))
    print(
      f"{design:>7} {seed:>4} | {a['calls_train_val']:>7} {a['n_rounds']:>4} {a['window']:>7} {a['diff']:>7.4f} "
      f"{a['err']:>7.4f} {a['status']:>9} {a['train']:>7.4f} {a['val']:>7.4f} {a['test']:>7.4f} | "
      f"{b['calls_train_val']:>7} {b['n_rounds']:>4} {b['window']:>7} {b['diff']:>7.4f} {b['err']:>7.4f} "
      f"{b['status']:>9} {b['train']:>7.4f} {b['val']:>7.4f} {b['train_parts'][0]:>7.4f} {b['val_parts'][0]:>7.4f} "
      f"{b['train_parts'][1]:>7.4f} {b['val_parts'][1]:>7.4f} {b['test']:>7.4f} | "
      f"{b['calls_train_val'] - a['calls_train_val']:>7} {ratio:>8.4f} {b['test'] - a['test']:>8.4f}"
    )

  print("\nPAIRED DIFFERENCES (dual - control), mean +/- SEM over pairs")
  for label, values, unit in (("detector calls", deltas, ""), ("ln(calls ratio)", ratios, ""),
                              ("growth rounds", round_deltas, ""), ("held-out test loss", test_deltas,
                                                                    ""), ("held-out test loss, per row", row_deltas, "")):
    mean, sem = mean_sem(values)
    z = float("nan") if not sem > 0 else mean / sem
    print(f"  {label:<28} {mean:>10.5f} +/- {sem:<10.5f} ({len(values)} pairs, z = {z:.2f}){unit}")
  mean, sem = mean_sem(ratios)
  if sem > 0:
    print(
      f"  ln-ratio resolvable at 2 sigma: {100 * (math.exp(2 * sem) - 1):.2f}% ; "
      f"3 sigma: {100 * (math.exp(3 * sem) - 1):.2f}% ; measured {100 * (math.exp(mean) - 1):+.2f}%"
    )

  print("\nEXACT-ONLY COUNTERFACTUAL (the criterion on the exact current rows alone)")
  print(f"{'design':>7} {'seed':>4} | {'control combined':>22} | {'dual combined':>22} | {'dual exact-only':>22}")
  for design, seed in keys:
    a, b = control[(design, seed)], dual[(design, seed)]
    fields = []
    for row, part in ((a, None), (b, None), (b, 0)):
      outcome, epoch, window, diff, err = replay(row, part, arguments.warmup, arguments.patience, arguments.precision)
      fields.append(f"{outcome:>8}@e{epoch:<4} w{window:<7}")
    print(f"{design:>7} {seed:>4} | " + " | ".join(fields))
  print(
    "  the two `combined` columns are the CONTROL: they must reproduce each run's own stopping epoch and "
    "window. `diverged` means the replayed rule's growth decisions parted from the run's, so nothing is "
    "claimed past that epoch."
  )


if __name__ == "__main__":
  main()
