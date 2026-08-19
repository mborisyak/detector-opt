#!/usr/bin/env python3
"""Read the concurrency profile and decide between the three scaling regimes.

    python scripts/report_concurrency.py output/concurrency

Let t(K) be the MEDIAN per-process seconds/epoch at concurrency K, and t(1) the solo cost. The
aggregate rate is K / t(K), normalised here to the solo rate so 1.00 means "K processes deliver
exactly what one did".

    t(K) ~ t(1)        latency-bound        aggregate rises ~K
    t(K) ~ K * t(1)    exact equal sharing  aggregate flat at 1.00
    t(K) >  K * t(1)   WORSE than sharing   aggregate falls below 1.00 -- concurrency HURTS

`share` below is t(K) / (K * t(1)): 1.00 is exact equal sharing, < 1 is better, > 1 is worse. The
per-replica spread at fixed K is reported so a degraded level cannot be pinned on one slow copy.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("directory")
  arguments = parser.parse_args()

  by_k = collections.defaultdict(list)
  for path in sorted(glob.glob(os.path.join(arguments.directory, "k*_r*.json"))):
    k = int(os.path.basename(path).split("_")[0][1:])
    with open(path) as f:
      by_k[k].append(json.load(f))
  if len(by_k) == 0:
    raise SystemExit(f"report_concurrency: no k*_r*.json under {arguments.directory}")

  levels = sorted(by_k)
  missing = {k: k - len(by_k[k]) for k in levels if len(by_k[k]) != k}
  if len(missing) > 0:
    print(f"⚠️ INCOMPLETE LEVELS (replicas that produced no json): {missing}")
    print("   A level missing replicas is not a measurement of that K -- read it as a lower bound.\n")

  solo = float(np.median([r["s_per_epoch"] for r in by_k[levels[0]]])) if levels[0] == 1 else None
  print(f"{'K':>4}{'n':>4}{'s/epoch med':>13}{'min':>9}{'max':>9}{'spread':>9}"
        f"{'aggregate':>11}{'share':>8}{'eval s':>9}")
  for k in levels:
    t = np.asarray([r["s_per_epoch"] for r in by_k[k]], float)
    ev = np.median([r["eval_s_per_pass"] for r in by_k[k]])
    med = float(np.median(t))
    aggregate = (k / med) / (1.0 / solo) if solo is not None else float("nan")
    share = med / (k * solo) if solo is not None else float("nan")
    print(f"{k:>4}{len(t):>4}{med:>13.3f}{t.min():>9.3f}{t.max():>9.3f}{t.max()/t.min():>9.2f}"
          f"{aggregate:>11.2f}{share:>8.2f}{ev:>9.3f}")

  if solo is None:
    print("\nNo K=1 level -- every ratio above needs it. Rerun with 1 in LEVELS.")
    return
  top = levels[-1]
  med_top = float(np.median([r["s_per_epoch"] for r in by_k[top]]))
  share_top = med_top / (top * solo)
  print(f"\nVERDICT at K={top}: t({top})/t(1) = {med_top/solo:.2f} against K = {top}")
  if share_top > 1.15:
    print(f"  WORSE THAN EQUAL SHARING (share {share_top:.2f}). Concurrency destroys throughput:")
    print(f"  {top} processes deliver {(top/med_top)/(1/solo):.2f}x what ONE delivers. Run fewer.")
  elif share_top < 0.85:
    print(f"  BETTER than equal sharing (share {share_top:.2f}) -- there is real headroom at K={top}.")
  else:
    print(f"  EXACT EQUAL SHARING (share {share_top:.2f}). Aggregate is flat: K is a scheduling")
    print(f"  choice, not a speed one. ⚠️ THIS DOES NOT MEAN THE CARD IS SATURATED. Two different")
    print(f"  causes give the same signature and this probe cannot tell them apart:")
    print(f"    (a) one process genuinely fills the device, or")
    print(f"    (b) the processes are TIME-SLICING because they hold separate CUDA contexts --")
    print(f"        which is what happens in Default compute mode when clients are NOT connected")
    print(f"        to an MPS daemon. Check CUDA_MPS_PIPE_DIRECTORY in the JOB's environment, not")
    print(f"        just that `nvidia-cuda-mps-control` is alive: a daemon serving a non-default")
    print(f"        pipe directory serves nobody who does not name it.")
    print(f"  Separate them by measuring achieved FLOP/s against the device peak, or by rerunning")
    print(f"  this profile with and without the MPS environment (scripts/mps_ab.sh).")


if __name__ == "__main__":
  main()
