#!/usr/bin/env python3
"""Score a `linear` run against the ANSWER -- the check the debug task exists to make possible.

    python scripts/score_linear.py output/linear

`LinearDetector.bayes_risk(design)` is the smallest loss ANY estimator can reach at that design: the
posterior of (w, b) under the standard-normal prior and Gaussian read-out noise is Gaussian with
covariance (X^T X / sigma^2 + I)^-1, and the loss is the mean squared error over the two components,
so half that matrix's trace is a floor and not an approximation to one. Every other task in this repo
can only be judged against its own output; here each number has a right answer sitting beside it.

THREE THINGS ARE READ OFF, in increasing order of what they would catch:

* **Nothing may sit below the floor BY MORE THAN ITS OWN ERROR.** `bayes_risk` is an EXPECTATION and
  the reported loss is a finite-sample estimate of it, so individual designs fall below it as often
  as sampling noise says they should -- measured here, 22 of 86, worst 0.45 sigma, none past 2. What
  would falsify the pipeline is a deficit LARGE compared with the design's own reported uncertainty:
  an optimistic objective, a leak between the train and validation windows, a target the network can
  see. The bar is therefore `loss < floor - sigma_tolerance * loss_std`, not `loss < floor`.
* **How far above the floor the trainer lands**, per design and in aggregate. This is the honest cost
  of a finite network and a finite window, and it is the number to watch when a trainer changes.
* **Where the search ended up.** At ONE dimension and TWO probes the optimum is the pair of box
  corners, in either order, so the best design's distance to it is a direct statement about the
  optimiser -- not a curve that only goes down. THAT COLUMN EXISTS ONLY THERE. For a wider rung the
  optimum is not a corner pair in closed form (`n = d + 1` probes in `[-1, 1]^d` admits no orthogonal
  design in general), so rather than assert one this prints `--` and reports the floor checks alone,
  which are the verification proper and need no optimum.

Exit status is 1 if anything landed below its floor, so this can gate a pipeline change.
"""

import argparse
import glob
import json
import os

import numpy as np

import detopt.detector
import detopt.utils.io


def runs_under(root):
    """Every `linear` run below ``root``, finished or in flight, as (label, rows).

    A run in progress writes `partial.json` and a finished one `results.json`; both are read, because
    a campaign is worth looking at before it ends and the rows mean the same thing in either file.
    """
    found = []
    for name in ("results.json", "partial.json"):
        for path in sorted(glob.glob(os.path.join(root, "**", name), recursive=True)):
            directory = os.path.dirname(path)
            if name == "partial.json" and os.path.exists(os.path.join(directory, "results.json")):
                continue  # the finished file supersedes its own partial
            with open(path) as f:
                data = json.load(f)
            label = os.path.relpath(directory, root)
            found.append((label if label != "." else os.path.basename(root), detopt.utils.io.complete_results(
                data.get("results", [])), bool(data.get("completed"))))
    return found


def score(detector, rows):
    """Reported loss, its floor and the excess, per row."""
    designs = [np.asarray(r["design"], np.float32) for r in rows]
    reported = np.asarray([float(r["loss"]) for r in rows])
    error = np.asarray([float(r.get("loss_std") or 0.0) for r in rows])
    floor = np.asarray([detector.bayes_risk(x) for x in designs])
    return designs, reported, floor, reported - floor, error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="a run directory, or a campaign root holding many")
    parser.add_argument("--detector", default="linear", help="detector config under config/detector/")
    parser.add_argument("--per-design", action="store_true", help="print every design, not just the summary")
    parser.add_argument("--sigma-tolerance", type=float, default=3.0,
                        help="how many of a design's OWN reported standard deviations below its floor "
                             "counts as impossible rather than as sampling noise")
    arguments = parser.parse_args()

    import yaml

    with open(f"config/detector/{arguments.detector}.yaml") as f:
        detector = detopt.detector.from_config(yaml.safe_load(f))
    # The corner pair is the optimum ONLY for the 1-dimension / 2-probe rung; see the module docstring.
    corner = np.array([-1.0, 1.0]) if detector.design_dim() == 2 else None
    optimum = float(detector.bayes_risk(corner.astype(np.float32))) if corner is not None else None

    found = runs_under(arguments.root)
    if len(found) == 0:
        raise SystemExit(f"score_linear: no results.json or partial.json under {arguments.root}")

    if optimum is not None:
      print(f"analytic optimum {optimum:.6f} at the box corners (either order); "
            f"the no-information level is 1.0\n")
    else:
      print(f"design_dim {detector.design_dim()}: no closed-form optimum, so |d-corner| is omitted; "
            f"the floor check below is the verification. The no-information level is 1.0\n")
    print(f"{'run':<28} {'n':>4} {'best':>9} {'floor':>9} {'excess':>9} {'|d-corner|':>10} "
          f"{'med excess':>10} {'min excess':>10}")
    below = []
    for label, rows, completed in found:
        if len(rows) == 0:
            print(f"{label:<28} {0:>4}  (no scored designs yet)")
            continue
        designs, reported, floor, excess, error = score(detector, rows)
        best = int(np.argmin(reported))
        # The design is a SET, so the corner is matched in either order.
        if corner is None:
          distance = None
        else:
          distance = min(float(np.max(np.abs(np.sort(designs[best]) - np.sort(corner)))), float(
              np.max(np.abs(np.sort(designs[best])[::-1] - np.sort(corner)))))
        mark = "" if completed else "  (in flight)"
        print(f"{label:<28} {len(rows):>4} {reported[best]:>9.6f} {floor[best]:>9.6f} "
              f"{excess[best]:>+9.6f} {'--' if distance is None else format(distance, '.4f'):>10} "
              f"{np.median(excess):>10.6f} "
              f"{np.min(excess):>+10.6f}{mark}")
        sigma = excess / np.maximum(error, 1e-12)
        below.extend((label, d, r, f, z) for d, r, f, z in zip(designs, reported, floor, sigma)
                     if z < -arguments.sigma_tolerance)
        if arguments.per_design:
            for d, r, f, e, z in zip(designs, reported, floor, excess, sigma):
                flag = "  BELOW THE FLOOR" if z < -arguments.sigma_tolerance else ""
                print(f"      ({', '.join(format(v, '+.3f') for v in np.ravel(d))})  reported {r:.6f}  floor {f:.6f}  "
                      f"excess {e:+.6f} ({z:+.1f} sigma){flag}")

    print()
    if len(below) > 0:
        print(f"{len(below)} DESIGN(S) BELOW THEIR BAYES RISK BY MORE THAN {arguments.sigma_tolerance} SIGMA "
              f"-- too far to be sampling noise, so it is a defect in the pipeline, not a result:")
        for label, d, r, f, z in below[:10]:
            print(f"  {label}  ({', '.join(format(v, '+.3f') for v in np.ravel(d))})  "
                  f"reported {r:.6f} vs floor {f:.6f}  ({z:+.1f} sigma)")
        raise SystemExit(1)
    print(f"no design sits more than {arguments.sigma_tolerance} sigma below its Bayes risk")


if __name__ == "__main__":
    main()
