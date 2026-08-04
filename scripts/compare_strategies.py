#!/usr/bin/env python3
"""Final plot of a strategy comparison: self-evaluated vs independently verified.

Reads, per strategy, ``<output>/<strategy>/results.json`` (the BO run) and, when present,
``<output>/<strategy>/verification.json`` (``scripts/verify_trajectory.py``), and writes
``<output>/convergence_all.png`` plus ``<output>/convergence_all.json`` -- the latter holding every
plotted number, so the figure regenerates from JSON alone.

Each strategy gets one colour. Its **dashed** step is the run's own best-so-far loss, which is
self-evaluated: BO's convergence procedure stopped on the very losses it reports. **Open markers**
are the raw reported loss of exactly the designs that were verified, and the **solid** line is those
same designs re-scored on a held-out test split. The gap between open and filled markers is that
strategy's optimism, and it differs between strategies -- which is why the dashed curves alone would
rank them wrongly.

    python scripts/compare_strategies.py --output output/enzyme
    python scripts/compare_strategies.py --output output/enzyme --strategies meta closest
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("AGG")

from detopt.utils.viz.bo import plot_strategy_verification

STRATEGIES = ["from_scratch", "continue", "closest", "meta"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(out_root, strategies):
    runs = {}
    for strategy in strategies:
        run_dir = os.path.join(out_root, strategy)
        results = _load(os.path.join(run_dir, "results.json"))
        if results is None:
            print(f"  [skip] no results at {run_dir}/results.json")
            continue
        verification = _load(os.path.join(run_dir, "verification.json"))
        if verification is None:
            print(f"  [warn] {strategy}: no verification.json -- plotting the self-evaluated curve only")
        runs[strategy] = {"results": results["results"], "verification": verification}
    if len(runs) == 0:
        raise SystemExit(f"no results.json found under {out_root}/<strategy>/")

    plot_strategy_verification(
        runs,
        os.path.join(out_root, "convergence_all.png"),
        json_path=os.path.join(out_root, "convergence_all.json"),
    )

    print(f"\n{'strategy':<14} {'self-evaluated':>15} {'verified (test)':>22} {'optimism':>10}")
    for label, run in runs.items():
        best = min(r["loss"] for r in run["results"])
        line = f"{label:<14} {best:>15.5f}"
        points = (run["verification"] or {}).get("points") or []
        if len(points) > 0:
            last = points[-1]
            line += (f"   {last['test_loss']:>8.5f}±{last['test_sem']:<8.5f}"
                     f" {last['test_loss'] - last['reported_loss']:>+10.5f}")
        print(line)
    print("\n'verified' is the FINAL trajectory point's held-out test loss; optimism is "
          "verified - self-evaluated for that same design.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="output/compare")
    p.add_argument("--strategies", nargs="*", default=STRATEGIES)
    a = p.parse_args()
    main(a.output, a.strategies)
