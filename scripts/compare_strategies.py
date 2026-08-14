#!/usr/bin/env python3
"""Final plot of a strategy comparison: self-evaluated vs independently verified.

Reads, per strategy, ``<output>/<strategy>/results.json`` (the BO run) and, when present,
``<output>/<strategy>/verification.json`` (``scripts/verify_trajectory.py``), and writes
``<output>/convergence_all.png`` plus ``<output>/convergence_all.json`` -- the latter holding every
plotted number, so the figure regenerates from JSON alone -- and ``<output>/comparison.txt``, the
same comparison as a table: BO steps, detector calls per step (avg+-std), and the last verified
design's reported loss against its held-out test estimate.

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
from statistics import mean, pstdev

import matplotlib

matplotlib.use("AGG")

from detopt.utils import io
from detopt.utils.viz.bo import plot_convergence_two_panel

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
        # Only the SCORED rows: every finished run ends with an `incomplete` row (the design its
        # budget could not pay for), carrying null loss/spent.
        runs[strategy] = {"results": io.complete_results(results["results"]), "verification": verification}
    if len(runs) == 0:
        raise SystemExit(f"no results.json found under {out_root}/<strategy>/")

    # One figure, two stacked step-panels: top = self-evaluated best-so-far, bottom = verified
    # (held-out) best-so-far. When no strategy has a verification.json, a single self-evaluated panel
    # is drawn instead. Both panels share the same step style.
    plot_convergence_two_panel(
        runs,
        os.path.join(out_root, "convergence_all.png"),
        json_path=os.path.join(out_root, "convergence_all.json"),
    )
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
    for strategy, run in runs.items():
        if run["verification"] is None:
            continue
        p = run["verification"]["points"][-1]  # sorted by iteration; the last verified design
        spent = [int(r["spent"]) for r in run["results"]]
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
