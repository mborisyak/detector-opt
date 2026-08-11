#!/usr/bin/env python3
"""Is the design SIGNAL bigger than the trainer ARTEFACT, and where does that stop being true?

    python scripts/signal_vs_artefact.py --artefact 0.0103

`scripts/capacity_probe.py` measured a trainer artefact: on identical designs `meta` reports
+0.0103 (median over 4 repeats) more than `from_scratch`. A number is only alarming next to the
quantity it corrupts, so this prints the three regimes an optimiser passes through and the ratio
artefact / signal in each:

  * WHOLE SPACE -- spread of the loss over designs drawn at random (a BO run's `n_init` prefix, plus
    any explicitly random run). This is what the optimiser sees at the start.
  * TOP 20%     -- spread within the best fifth of the designs actually visited. This is what it
    sees once it has found the good region.
  * LATE RUN    -- consecutive gaps among the best 10 designs of a run. This is the quantity a
    proposal is ranked by at the end, and it is the smallest of the three.

A ratio below 1 means the design difference dominates and the artefact is a nuisance; above 1 means
the optimiser is choosing between designs whose true difference is smaller than the bias between two
ways of measuring them. The peer session `enzyme-measurements` measured 0.5x / 2.4x / 19.3x on its
own detector (a different artefact -- swapping the regressor -- on a different detector), i.e. the
ratio INVERTS as BO converges. This asks the same question here.

Only runs at the CURRENT settings are pooled by default (`loss_precision` 1.0e-2, box [25, 80],
budget 524288), because the artefact was measured there; older runs carry a different objective_std
definition and a different box, and mixing them would compare spreads that are not the same
quantity. `--runs` overrides the list.
"""
import argparse
import glob
import json

import numpy as np

# Runs at the current settings. A run is used for the whole-space row only through its random
# prefix (`n_init` designs, which BO does not choose) and for the other two rows through everything.
CURRENT = ("output/neural-confirm/*/results.json", "output/neural-confirm-*/*/results.json")
N_INIT = 5  # scripts/bo.py defaults bo.n_init to gp.n_folds = 5


def spread(values):
  """Population sd -- the peer's `sd` rows, so the two detectors' numbers are the same statistic."""
  return float(np.std(values)) if len(values) > 1 else float("nan")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--artefact", type=float, default=0.0103,
                      help="the trainer artefact to compare against (default: capacity_probe's median)")
  parser.add_argument("--runs", nargs="*", default=None, help="explicit results.json paths")
  parser.add_argument("--n-best", type=int, default=10)
  parser.add_argument("--probe", default="output/capacity/probe.json",
                      help="capacity_probe output; used to split the artefact into a constant "
                           "offset and the design-dependent part that actually reorders designs")
  arguments = parser.parse_args()

  paths = arguments.runs
  if paths is None:
    paths = sorted({p for pattern in CURRENT for p in glob.glob(pattern)})

  random_losses, all_losses, per_run = [], [], []
  for path in paths:
    payload = json.load(open(path))
    losses = [float(r["loss"]) for r in payload["results"]]
    if len(losses) == 0:
      continue
    random_losses.extend(losses[:N_INIT])
    all_losses.extend(losses)
    per_run.append((path, losses))
    print(f"{path}: {len(losses)} designs, best {min(losses):.4f}")

  if len(all_losses) == 0:
    raise SystemExit("no runs found")

  all_losses = np.array(all_losses)
  top = np.sort(all_losses)[:max(2, int(round(0.2 * len(all_losses))))]

  # Late run: consecutive gaps among each run's best `n_best`, pooled. Per run, because the gap is
  # between designs the SAME optimiser was ranking; pooling raw losses across runs would measure
  # between-run offsets instead.
  gaps = []
  for _, losses in per_run:
    best = np.sort(np.array(losses))[:arguments.n_best]
    if len(best) > 1:
      gaps.extend(np.diff(best).tolist())

  rows = [
      ("whole space (sd, random prefix)", spread(random_losses), len(random_losses)),
      ("all visited  (sd)", spread(all_losses), len(all_losses)),
      ("top 20%      (sd)", spread(top), len(top)),
      (f"late run     (median gap, best {arguments.n_best})",
       float(np.median(gaps)) if len(gaps) > 0 else float("nan"), len(gaps)),
  ]
  print(f"\nartefact = {arguments.artefact:.4f} (meta - from_scratch on identical designs)\n")
  print(f"{'where BO is working':44s} {'design signal':>14s} {'n':>4s} {'artefact/signal':>16s}")
  for label, value, n in rows:
    ratio = arguments.artefact / value if value > 0 else float("inf")
    print(f"{label:44s} {value:14.5f} {n:4d} {ratio:15.1f}x")

  # A CONSTANT offset between strategies does not reorder designs within an arm -- only the
  # design-DEPENDENT part does, and that is the part a ranking-based optimiser actually suffers.
  # Split them, because the median above conflates the two.
  try:
    probe = json.load(open(arguments.probe))
  except OSError:
    return
  pairs = [(r["meta"], r["from_scratch"]) for r in probe["rows"]
           if r["meta"] is not None and r["from_scratch"] is not None]
  difference = np.array([m["loss"] - s["loss"] for m, s in pairs])
  late = rows[-1][1]
  print(f"\nSPLIT of the artefact over {len(difference)} probe repeats "
        f"(a constant offset cannot reorder; the spread can)")
  print(f"  constant part (median)                     {np.median(difference):+.5f}")
  print(f"  design-dependent part (sd)                 {np.std(difference):.5f}"
        f"   = {np.std(difference) / late:.1f}x the late-run gap")
  print(f"  design-dependent part (sd, outlier removed) "
        f"{np.std(np.sort(difference)[:-1]):.5f}   = {np.std(np.sort(difference)[:-1]) / late:.1f}x")
  print(f"  per repeat: {', '.join(f'{d:+.4f}' for d in difference)}")


if __name__ == "__main__":
  main()
