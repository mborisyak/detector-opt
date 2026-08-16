#!/usr/bin/env python3
"""Pair the two arms of `scripts/probe_replay_mix.py` and price the trade.

    python scripts/probe_replay_mix_report.py output/replaymix

Reads every `mix_A_s*.json` / `mix_B_s*.json` under the given directory, pairs them on
`(11th design, seed)` -- the two arms differ ONLY in the composition of the history and the batch
geometry, so every other source of variation cancels within a pair -- and reports:

  * per pair: detector calls to convergence, growth rounds, the final window, `diff` and `err` at
    stopping, whether the run converged or hit the cap, and the test loss at the 11th design;
  * the paired mean difference in calls with its standard error, and the paired mean LOG RATIO,
    which is the statistic that survives the 5x seed-to-seed spread in the absolute level;
  * the paired test-loss difference, computed PER ROW over the identical test pool (`*_test_rows.npz`)
    so its error is the error of a difference and not of two independent means;
  * THE TRADE. Arm B's whole-space half buys rows at designs nobody wanted scored:
    `--whole-space-calls` detector calls, which at `meta`'s measured 55976 calls per design is 4.88
    scored designs forgone. A saving of `s` calls per design repays that in `whole_space / s`
    designs. This script prints that number when the saving is positive and says the trade is settled
    when it is not. ONE design is measured, so the payback figure ASSUMES the saving recurs; nothing
    here can show that it does.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

MEDIAN_PER_DESIGN_CALLS = 55976


def load(directory, arm):
  rows, per_row = {}, {}
  for path in sorted(glob.glob(os.path.join(directory, f"mix_{arm}_s*.json"))):
    with open(path) as f:
      for row in json.load(f)["rows"]:
        rows[(row["design"], int(row["seed"]))] = row
    npz = os.path.splitext(path)[0] + "_test_rows.npz"
    if os.path.isfile(npz):
      with np.load(npz) as data:
        for key in data.files:
          per_row[key] = np.asarray(data[key])
  return rows, per_row


def mean_sem(values):
  values = np.asarray(values, np.float64)
  n = values.size
  if n < 2:
    return float(values.mean()) if n == 1 else float("nan"), float("nan")
  return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("directory", nargs="?", default="output/replaymix")
  parser.add_argument(
    "--whole-space-calls", type=int, default=273067,
    help="the calls arm B spends on designs nobody wanted scored: its 204800 whole-space TRAIN rows "
    "charged at the campaign's 3:1 split"
  )
  arguments = parser.parse_args()

  a_rows, a_per_row = load(arguments.directory, "A")
  b_rows, b_per_row = load(arguments.directory, "B")
  keys = sorted(set(a_rows) & set(b_rows), key=lambda k: (k[1], k[0]))
  if len(keys) == 0:
    raise SystemExit(f"probe_replay_mix_report: no pairs in {arguments.directory}")
  unpaired = sorted(set(a_rows) ^ set(b_rows))
  if len(unpaired) > 0:
    print(f"UNPAIRED (excluded): {unpaired}")

  header = (
    f"{'design':8s} {'seed':>4s} | {'A calls':>8s} {'B calls':>8s} {'delta':>8s} {'ratio':>6s} | "
    f"{'A rnd':>5s} {'B rnd':>5s} | {'A win':>7s} {'B win':>7s} | {'A diff':>7s} {'B diff':>7s} | "
    f"{'A err':>6s} {'B err':>6s} | {'A test':>7s} {'B test':>7s} {'dtest':>8s} | {'A status':>9s} {'B status':>9s}"
  )
  print(header)
  print("-" * len(header))

  deltas, ratios, test_deltas, test_delta_sems = [], [], [], []
  for design, seed in keys:
    a, b = a_rows[(design, seed)], b_rows[(design, seed)]
    delta = a["calls_train_val"] - b["calls_train_val"]
    ratio = b["calls_train_val"] / a["calls_train_val"]
    deltas.append(delta)
    ratios.append(math.log(ratio))
    key = f"{design}|{seed}"
    if key in a_per_row and key in b_per_row and a_per_row[key].shape == b_per_row[key].shape:
      difference = np.asarray(b_per_row[key], np.float64) - np.asarray(a_per_row[key], np.float64)
      test_deltas.append(float(difference.mean()))
      test_delta_sems.append(float(difference.std(ddof=1) / math.sqrt(difference.size)))
      shown = f"{difference.mean():+8.4f}"
    else:
      shown = f"{b['test'] - a['test']:+8.4f}"
    print(
      f"{design:8s} {seed:4d} | {a['calls_train_val']:8d} {b['calls_train_val']:8d} {delta:+8d} "
      f"{ratio:6.3f} | {a['n_rounds']:5d} {b['n_rounds']:5d} | {a['window']:7d} {b['window']:7d} | "
      f"{a['diff']:7.4f} {b['diff']:7.4f} | {a['err']:6.4f} {b['err']:6.4f} | "
      f"{a['test']:7.4f} {b['test']:7.4f} {shown} | {a['status']:>9s} {b['status']:>9s}"
    )

  print()
  mean_delta, sem_delta = mean_sem(deltas)
  mean_log, sem_log = mean_sem(ratios)
  print(f"pairs                        : {len(keys)}")
  print(f"paired saving A-B, calls     : {mean_delta:+.0f} +/- {sem_delta:.0f}  (t = {mean_delta / sem_delta:+.2f})")
  print(
    f"paired log ratio ln(B/A)     : {mean_log:+.4f} +/- {sem_log:.4f}  "
    f"(B/A = {math.exp(mean_log):.3f}, t = {mean_log / sem_log:+.2f})"
  )
  for label, arm in (("A", a_rows), ("B", b_rows)):
    calls = [arm[k]["calls_train_val"] for k in keys]
    gaps = [arm[k]["diff"] for k in keys]
    slack = [arm[k]["slack"] for k in keys]
    tests = [arm[k]["test"] for k in keys]
    print(
      f"arm {label}: calls {np.mean(calls):8.0f} (median {np.median(calls):8.0f}) | gap |val-train| "
      f"{np.mean(gaps):.5f} | slack {np.mean(slack):.5f} | test {np.mean(tests):.4f} | "
      f"capped {sum(1 for k in keys if arm[k]['status'] != 'converged')}/{len(keys)}"
    )
  if len(test_deltas) > 0:
    mean_test, sem_test = mean_sem(test_deltas)
    within = math.sqrt(sum(s * s for s in test_delta_sems)) / len(test_delta_sems)
    print(
      f"paired test loss B-A         : {mean_test:+.5f} +/- {sem_test:.5f} across pairs "
      f"(mean within-pair per-row SEM {within:.5f})"
    )

  print()
  if mean_delta > 0:
    designs = arguments.whole_space_calls / mean_delta
    print(
      f"THE TRADE. Arm B saves {mean_delta:.0f} calls per design. Its whole-space half costs "
      f"{arguments.whole_space_calls} calls = {arguments.whole_space_calls / MEDIAN_PER_DESIGN_CALLS:.2f} "
      f"scored designs, so it repays in {designs:.1f} designs IF the saving recurs at every later "
      f"design. This probe measures ONE design and cannot show that it does."
    )
  else:
    print(
      f"THE TRADE IS SETTLED AGAINST ARM B: it costs {-mean_delta:.0f} MORE calls per design, so the "
      f"{arguments.whole_space_calls} calls of whole-space history "
      f"({arguments.whole_space_calls / MEDIAN_PER_DESIGN_CALLS:.2f} scored designs) buy a per-design "
      f"LOSS and never repay. No payback arithmetic is needed."
    )


if __name__ == "__main__":
  main()
