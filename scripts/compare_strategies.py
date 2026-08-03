#!/usr/bin/env python3
"""Compare the four BO training-strategy runs: convergence overlay + verification table.

Reads ``<output>/<strategy>/results.json`` for each strategy and writes
``<output>/convergence_all.png`` (best-so-far loss vs cumulative detector calls). For strategies
that also have a ``verification.json`` (``scripts/verify_trajectory.py``), writes
``<output>/comparison.txt``: total BO steps, detector calls per step (avg+-std), and the FINAL
design's reported loss vs the proper held-out test estimate. The runs themselves are produced by
``make.sh`` (one ``scripts/bo.py`` run per strategy).

    python scripts/compare_strategies.py --output output/compare
"""

import argparse
import json
import os
from statistics import mean, pstdev

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
    comparison_table(out_root, runs)


def comparison_table(out_root, runs):
    """``comparison.txt``: per strategy the BO step statistics (total steps, detector calls per
    step avg+-std over the per-iteration ``spent``) and the FINAL design's reported loss vs the
    verified held-out test estimate. Strategies without a ``verification.json`` are left out; the
    file is only written once at least one strategy is verified."""
    lines = [
        f"{'strategy':<14}{'steps':>6}{'calls/step':>16}{'calls':>9}{'reported':>10}"
        f"{'val':>8}{'test':>8}{'sem':>8}{'delta':>9}",
        "-" * 88,
    ]
    for strategy, results in runs.items():
        vpath = os.path.join(out_root, strategy, "verification.json")
        if not os.path.exists(vpath):
            continue
        p = json.load(open(vpath))["points"][-1]  # sorted by iteration; the final design is always verified
        spent = [int(r["spent"]) for r in results]
        cps = f"{mean(spent):.0f}+-{pstdev(spent):.0f}"
        lines.append(
            f"{strategy:<14}{len(spent):>6}{cps:>16}{int(p['detector_calls']):>9}{p['reported_loss']:>10.4f}"
            f"{p['val_loss']:>8.4f}{p['test_loss']:>8.4f}{p['test_sem']:>8.4f}"
            f"{p['test_loss'] - p['reported_loss']:>+9.4f}"
        )
    text = "\n".join(lines) + "\n"
    path = os.path.join(out_root, "comparison.txt")
    with open(path, "w") as f:
        f.write(text)
    print(text, end="")
    print(f"comparison -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="output/compare")
    p.add_argument("--strategies", nargs="*", default=STRATEGIES)
    a = p.parse_args()
    main(a.output, a.strategies)
