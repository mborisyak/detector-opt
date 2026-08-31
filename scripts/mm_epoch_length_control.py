#!/usr/bin/env python3
"""Is `meta`'s step-count advantage EPOCH-LENGTH driven? A control that is already on disk.

    python scripts/mm_epoch_length_control.py

THE FREE CONTROL. The two MM campaigns run the SAME detector, regressor, BO block, `patience` 16,
`warmup_epochs` 2 and `rewind` 0.25, and differ in the growth schedule and the EPOCH LENGTH:

    campaign-mm        loss_precision 2.0e-3  n0 8192  n_increment 4096  batch 256  ->  2048 steps/epoch
    campaign-mm-p5e3   loss_precision 5.0e-3  n0 2048  n_increment 1024  batch 128  ->   512 steps/epoch

`patience` is written in EPOCHS, so at 512 steps an epoch it spans a QUARTER of the optimisation it
spans at 2048. The p5e3 config's own header states the risk and does not act on it: "WATCH FOR
premature plateau calls -- designs converging in very few rounds at a suspiciously small window."

THE PREDICTION UNDER THE EPOCH-LENGTH HYPOTHESIS. If `meta` converges at a smaller window than
`from_scratch` BECAUSE `patience` spans too little optimisation to notice the descent it is still on,
then shortening the epoch 4x must make the asymmetry WORSE: the p5e3 campaigns should show a larger
`meta`-to-`from_scratch` spend ratio than the 2048-step campaigns. If the ratio is unchanged, epoch
length is not the mechanism and the asymmetry has to be explained by something else -- replay closing
the train/validation gap, which is measured elsewhere.

⚠️ THIS IS NOT A CLEAN SINGLE-VARIABLE COMPARISON and no claim here rests on it alone. The two
campaigns also differ in `loss_precision` (2.5x), in the growth quanta, and therefore in the WINDOW a
design needs, so the absolute spend is not comparable across them. The RATIO between arms within a
campaign is what is compared, and it is compared at matched seeds where they exist.

⚠️ `results.json` STORES NO CONFIG. The directory name is the only link between a trajectory and its
settings, which is exactly how the two were confused in the first place; the schedule constants below
are therefore read from the config files, and the reconstruction of the round count is CHECKED against
each design's recorded `spent`.

THE EXTREMES CAMPAIGN IS INCLUDED AS THE REFERENCE ASYMMETRY. It is the task on which `meta` was
found to exit at a 26-38% smaller window while `continue` and `closest` -- which also warm-start --
spent the same as `from_scratch`, so all four arms are listed. It runs at 2048 steps an epoch and has
no short-epoch sibling, so it contributes the size of the effect, not a second point on the
epoch-length axis.

THE SPEND RATIO BY DESIGN POSITION is reported as well. A cloud-side analysis found it is NOT a
constant offset on the extremes families -- pooled, it runs 0.82 at design 0, peaks at 2.23 over
designs 3-5 and decays to 1.00 by design 21 -- which does not match "a larger replay pool dilutes
more". Whether the same shape appears here is a clue in its own right. ⚠️ The two arms visit DIFFERENT
designs from their common Sobol prefix onward, so this is matched by POSITION IN THE TRAJECTORY and
by nothing else.

THIS SCRIPT MEASURES NOTHING. It reads finished trajectories.
"""

from __future__ import annotations

import collections
import json
import os

import numpy as np
import scipy.stats

CAMPAIGNS = {
  "campaign-mm (2048 steps/epoch, precision 2.0e-3)": {
    "root": "output/campaign-mm",
    "n0": 8192,
    "n_increment": 4096,
    "steps_per_epoch": 2048,
  },
  "campaign-mm-p5e3 (512 steps/epoch, precision 5.0e-3)": {
    "root": "output/campaign-mm-p5e3",
    "n0": 2048,
    "n_increment": 1024,
    "steps_per_epoch": 512,
  },
  "enzyme_extremes (2048 steps/epoch, precision 1.0e-2)": {
    "root": "output/enzyme_extremes",
    "n0": 8192,
    "n_increment": 4096,
    "steps_per_epoch": 2048,
    "arms": ("from_scratch", "meta", "continue", "closest"),
  },
}
ARMS = ("from_scratch", "meta")


def designs(path, n0, n_increment):
  """`[(window, rounds, loss, seconds), ...]` for every scored design of one trajectory.

  `spent` is train + validation rows at a 3:1 split, so the train window is three quarters of it and
  the round count follows from the growth quanta. The reconstruction is checked: a window that is not
  `n0 + k * n_increment` means the schedule constants do not belong to this trajectory.
  """
  with open(path) as f:
    rows = [r for r in json.load(f)["results"] if r.get("loss") is not None]
  out = []
  for row in rows:
    spent = int(row["spent"])
    base = n0 + round(n0 / 3.0)
    step = n_increment + round(n_increment / 3.0)
    rounds = int(round((spent - base) / step)) + 1
    window = n0 + (rounds - 1) * n_increment
    if rounds < 1 or base + (rounds - 1) * step != spent:
      raise SystemExit(f"{path}: spent {spent} is not {base} + k * {step} -- wrong config for this trajectory")
    out.append((window, rounds, float(row["loss"]), float(row["time_s"])))
  return out


def main():
  for label, schedule in CAMPAIGNS.items():
    root = schedule["root"]
    arms = schedule.get("arms", ARMS)
    seeds = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"\n=== {label}")
    print(
      f"{'seed':>12}{'arm':>14}{'n':>4}{'median window':>15}{'median rounds':>15}"
      f"{'median epochs*':>16}{'median loss':>13}{'median s':>10}"
    )
    ratios = []
    for seed in seeds:
      per_arm = {}
      for arm in arms:
        path = os.path.join(root, seed, arm, "results.json")
        if not os.path.isfile(path):
          continue
        per_arm[arm] = designs(path, schedule["n0"], schedule["n_increment"])
      for arm in arms:
        if arm not in per_arm:
          continue
        values = per_arm[arm]
        windows = [v[0] for v in values]
        rounds = [v[1] for v in values]
        seconds = [v[3] for v in values]
        print(
          f"{seed:>12}{arm:>14}{len(values):>4}{np.median(windows):>15.0f}{np.median(rounds):>15.1f}"
          f"{'n/a':>16}{np.median([v[2] for v in values]):>13.5f}{np.median(seconds):>10.0f}"
        )
      if all(arm in per_arm for arm in ("from_scratch", "meta")):
        scratch_window = np.median([v[0] for v in per_arm["from_scratch"]])
        parts = []
        for arm in arms:
          if arm == "from_scratch" or arm not in per_arm:
            continue
          parts.append(f"{arm} {np.median([v[0] for v in per_arm[arm]]) / scratch_window:.3f}")
        ratios.append(np.median([v[0] for v in per_arm["meta"]]) / scratch_window)
        print(f"{'':>12}  window ratio against from_scratch: " + "  ".join(parts))
    if len(ratios) > 0:
      print(f"  POOLED median window ratio meta / from_scratch: {np.median(ratios):.3f}  over {len(ratios)} seed(s)")

  print("\n=== pooled across seeds, per campaign (every scored design, unpaired)")
  print(f"{'campaign':>52}{'arm':>14}{'n':>5}{'median window':>15}{'median rounds':>15}{'Mann-Whitney p':>16}")
  for label, schedule in CAMPAIGNS.items():
    root = schedule["root"]
    arms = schedule.get("arms", ARMS)
    pooled = {arm: [] for arm in arms}
    for seed in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
      for arm in arms:
        path = os.path.join(root, seed, arm, "results.json")
        if os.path.isfile(path):
          pooled[arm].extend(designs(path, schedule["n0"], schedule["n_increment"]))
    windows = {arm: [v[0] for v in pooled[arm]] for arm in arms}
    p = float(scipy.stats.mannwhitneyu(windows["meta"], windows["from_scratch"]).pvalue
              ) if all(len(windows[arm]) > 0 for arm in ("from_scratch", "meta")) else float("nan")
    for arm in arms:
      if len(pooled[arm]) == 0:
        continue
      print(
        f"{label:>52}{arm:>14}{len(pooled[arm]):>5}{np.median(windows[arm]):>15.0f}"
        f"{np.median([v[1] for v in pooled[arm]]):>15.1f}{p if arm == 'meta' else float('nan'):>16.4f}"
      )
    print(
      f"{'':>52}{'ratio':>14}{'':>5}{np.median(windows['meta']) / np.median(windows['from_scratch']):>15.3f}"
      f"{np.median([v[1] for v in pooled['meta']]) / np.median([v[1] for v in pooled['from_scratch']]):>15.3f}"
    )


def spend_by_position(campaigns):
  """Median window per arm at each POSITION in the trajectory, and their ratio."""
  print("\n=== spend ratio by design POSITION (medians over seeds; matched by position only)")
  for label, schedule in campaigns.items():
    root = schedule["root"]
    per_position = {arm: collections.defaultdict(list) for arm in ARMS}
    for seed in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
      for arm in ARMS:
        path = os.path.join(root, seed, arm, "results.json")
        if not os.path.isfile(path):
          continue
        for position, value in enumerate(designs(path, schedule["n0"], schedule["n_increment"])):
          per_position[arm][position].append(value[0])
    positions = sorted(set(per_position["meta"]) & set(per_position["from_scratch"]))
    if len(positions) == 0:
      continue
    buckets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
    print(f"  {label}")
    print("    designs   " + "".join(f"{f'{low}-{high}' if low != high else str(low):>10}" for low, high in buckets))
    for arm in ARMS:
      cells = [[v for position in positions if low <= position <= high for v in per_position[arm][position]]
               for low, high in buckets]
      print(f"    {arm:<10}" + "".join(f"{np.median(c):>10.0f}" if len(c) > 0 else f"{'-':>10}" for c in cells))
    ratios = []
    for low, high in buckets:
      meta = [v for position in positions if low <= position <= high for v in per_position["meta"][position]]
      scratch = [v for position in positions if low <= position <= high for v in per_position["from_scratch"][position]]
      ratios.append(np.median(meta) / np.median(scratch) if len(meta) > 0 and len(scratch) > 0 else float("nan"))
    print("    ratio     " + "".join(f"{r:>10.3f}" for r in ratios))


if __name__ == "__main__":
  main()
  spend_by_position(CAMPAIGNS)
