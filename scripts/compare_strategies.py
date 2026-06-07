#!/usr/bin/env python3
"""Overlay the convergence of the four BO training-strategy runs on one plot.

Reads ``<output>/<strategy>/results.json`` for each strategy and writes
``<output>/convergence_all.png`` (best-so-far loss vs cumulative detector calls).
The runs themselves are produced by ``make.sh`` (one ``scripts/bo.py`` run per
strategy); this script only plots.

    python scripts/compare_strategies.py --output output/compare
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("AGG")

from detopt.utils.viz.bo import plot_convergence_comparison

STRATEGIES = ["from_scratch", "continue", "closest", "meta"]


def main(out_root, strategies):
    runs = {}
    for strategy in strategies:
        path = os.path.join(out_root, strategy, "results.json")
        if os.path.exists(path):
            runs[strategy] = json.load(open(path))["results"]
        else:
            print(f"  [skip] no results at {path}")
    if not runs:
        raise SystemExit(f"no results.json found under {out_root}/<strategy>/")
    plot_convergence_comparison(runs, os.path.join(out_root, "convergence_all.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="output/compare")
    p.add_argument("--strategies", nargs="*", default=STRATEGIES)
    a = p.parse_args()
    main(a.output, a.strategies)
