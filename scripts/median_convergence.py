#!/usr/bin/env python3
"""Median-across-seeds convergence of a multi-seed strategy comparison.

Reads, per seed run root (``output/enzyme-42 output/enzyme-43 ...``) and per strategy,
``<run>/<strategy>/results.json`` (the BO run) and, when present, ``<run>/<strategy>/verification.json``
(``scripts/verify_trajectory.py``), and writes a two-panel figure: per strategy the pointwise MEDIAN
over seeds of the best-so-far (cummin) loss vs cumulative detector calls -- the left panel over the
runs' own self-evaluated reported losses, the right panel over the independently verified held-out
test losses. The plotted values go next to the figure as JSON, so it regenerates from JSON alone.

    python scripts/median_convergence.py --runs output/enzyme-42 output/enzyme-43 --output output/enzyme-median.png
"""


import argparse
import json
import os

import matplotlib

matplotlib.use("AGG")

from detopt.utils import io
from detopt.utils.viz.bo import plot_median_convergence

STRATEGIES = ["from_scratch", "continue", "closest", "meta"]


def _load(path):
  if not os.path.exists(path):
    return None
  with open(path) as f:
    return json.load(f)


def main(run_roots, output, strategies, statistic):
  runs = {}
  for strategy in strategies:
    seeds = {}
    for root in run_roots:
      run_dir = os.path.join(root, strategy)
      results = _load(os.path.join(run_dir, "results.json"))
      if results is None:
        print(f"  [skip] no results at {run_dir}/results.json")
        continue
      verification = _load(os.path.join(run_dir, "verification.json"))
      if verification is None:
        print(f"  [warn] {run_dir}: no verification.json -- this seed enters the self-evaluated panel only")
      # Only the SCORED rows -- a finished run ends with a null-loss `incomplete` row.
      seeds[os.path.basename(os.path.normpath(root))] = {
        "results": io.complete_results(results["results"]), "verification": verification}
    if len(seeds) > 0:
      runs[strategy] = seeds
  if len(runs) == 0:
    raise SystemExit(f"no results.json found under any of {run_roots}")

  plot_median_convergence(runs, output, json_path=os.path.splitext(output)[0] + ".json", statistic=statistic)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--runs", nargs="+", required=True, help="per-seed run roots, each holding <strategy>/results.json")
  p.add_argument("--output", required=True, help="destination PNG; the plotted values go next to it as .json")
  p.add_argument("--statistic", choices=("median", "mean"), default="median",
                 help="pointwise reduction across seeds; median is typical-seed, mean is tail-sensitive")
  p.add_argument("--strategies", nargs="*", default=STRATEGIES)
  a = p.parse_args()
  main(a.runs, a.output, a.strategies, a.statistic)
