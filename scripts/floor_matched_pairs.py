#!/usr/bin/env python3
"""FLOOR-MATCHED DESIGN PAIRS on `linear`: designs that are exactly as good, to test whether the
criterion scores them the same.

    srun --cpus-per-task=2 --mem=4000 --time=00:20:00 python scripts/floor_matched_pairs.py \
        =linear_d2n3_growth --n-pairs 6 --output output/criterion-bench/floor_pairs.json

WHY. Slot 4 of `docs/criterion-investigation.md` found that trajectory designs 9 and 11, which share
the exact floor 0.37656, are scored 0.008 apart -- 36/38 matched cells, p = 5.4e-09, and 15/15 of the
cells where the exit window was identical too. That is ONE pair. It says nothing about how common
such pairs are, or whether the gap grows with the floor. `LinearDetector.bayes_risk` is closed form
and costs a 3x3 inverse, so the pairs can be found offline for nothing and the question becomes
answerable on several pairs spanning the floor range.

WHAT A PAIR IS, and the two conditions that make it a fair test:

  EQUAL FLOOR    `|bayes_risk(a) - bayes_risk(b)| <= --tolerance`. The default 2.0e-4 is 40x below
                 the 0.008 effect and 2% of the 1.0e-2 bar, so a floor difference cannot account for
                 anything the scoring finds.
  FAR APART      `||a - b|| >= --min-distance` in the SCALED cube. Without this the "pair" can be two
                 nearly identical points, which would agree by construction and prove nothing. The
                 default 0.6 is well above zero for the 6-dimensional unit cube, whose mean
                 inter-point distance is about 1.0.

THE PAIRS SPAN THE FLOOR RANGE: the candidate pairs are bucketed into `--n-pairs` equal-count bins of
their own floor, and the most widely separated pair in each bin is taken, so the set covers cheap and
expensive designs rather than clustering wherever pairs happen to be dense.

THE OUTPUT IS SHAPED LIKE A `results.json`, with `loss` set to the floor, so
`probe_bias_cross.scored_designs` reads it UNCHANGED and the scoring job needs no new code path.
Since that helper takes ORDER STATISTICS of `loss`, and the file contains only the selected designs
with pairs well separated in floor, rank `2k` and rank `2k + 1` are the two members of pair `k`.

THIS SCRIPT SIMULATES NOTHING and calls no detector: it evaluates a closed form. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import detopt
import detopt.detector


def floors(detector, designs):
  """`bayes_risk` for each row of `designs`, which are SCALED. It takes a NOMINAL design; feeding it
  the scaled vector silently returns a larger, wrong floor."""
  values = np.empty(designs.shape[0], np.float64)
  for index, scaled in enumerate(designs):
    nominal = detector.to_nominal(np.asarray(scaled, np.float32)[None, :])
    values[index] = float(detector.bayes_risk(np.asarray(detector.flatten_design(nominal), np.float32)[0]))
  return values


def candidate_pairs(designs, values, tolerance, min_distance):
  """Every `(i, j, distance)` whose floors agree to `tolerance` and whose designs are at least
  `min_distance` apart, found by walking the floor-sorted order so the scan is linear in the number of
  qualifying neighbours rather than quadratic in the sample."""
  order = np.argsort(values)
  pairs = []
  for position, i in enumerate(order):
    for j in order[position + 1:]:
      if values[j] - values[i] > tolerance:
        break
      distance = float(np.linalg.norm(designs[i] - designs[j]))
      if distance >= min_distance:
        pairs.append((int(i), int(j), distance))
  return pairs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a run config, e.g. =linear_d2n3_growth")
  parser.add_argument("--n-samples", type=int, default=20000, help="designs drawn UNIFORMLY in the scaled cube")
  parser.add_argument("--n-pairs", type=int, default=6)
  parser.add_argument("--tolerance", type=float, default=2.0e-4, help="max |difference| in bayes_risk within a pair")
  parser.add_argument("--min-distance", type=float, default=0.6, help="min Euclidean separation in the SCALED cube")
  parser.add_argument("--seed", type=int, default=20260818)
  parser.add_argument("--output", default="output/criterion-bench/floor_pairs.json")
  arguments = parser.parse_args()

  import yaml

  name = arguments.config.lstrip("=")
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  detector = detopt.detector.from_config(detector_config)
  if not hasattr(detector, "bayes_risk"):
    raise SystemExit(f"floor_matched_pairs: {name} has no closed-form floor; this study is `linear` only")

  rng = np.random.default_rng(arguments.seed)
  designs = np.asarray(rng.random((arguments.n_samples, int(detector.design_dim()))), np.float32)
  values = floors(detector, designs)
  print(
    f"{arguments.n_samples} designs | floor min {values.min():.5f} median {np.median(values):.5f} "
    f"max {values.max():.5f}", flush=True
  )

  pairs = candidate_pairs(designs, values, arguments.tolerance, arguments.min_distance)
  if len(pairs) < arguments.n_pairs:
    raise SystemExit(
      f"floor_matched_pairs: only {len(pairs)} candidate pairs at tolerance {arguments.tolerance:g} and "
      f"min distance {arguments.min_distance:g}; raise --n-samples or relax one of them"
    )
  print(f"{len(pairs)} candidate pairs", flush=True)

  pair_floor = np.asarray([0.5 * (values[i] + values[j]) for i, j, _ in pairs])
  edges = np.quantile(pair_floor, np.linspace(0.0, 1.0, arguments.n_pairs + 1))
  chosen = []
  for bucket in range(arguments.n_pairs):
    low, high = edges[bucket], edges[bucket + 1]
    inside = [
      index for index in range(len(pairs))
      if (pair_floor[index] >= low and pair_floor[index] <= high) and index not in {c
                                                                                    for c, _ in chosen}
    ]
    if len(inside) == 0:
      continue
    best = max(inside, key=lambda index: pairs[index][2])
    chosen.append((best, bucket))
  if len(chosen) < arguments.n_pairs:
    raise SystemExit(f"floor_matched_pairs: filled only {len(chosen)} of {arguments.n_pairs} floor buckets")

  results, manifest = [], []
  for pair_index, (index, _bucket) in enumerate(sorted(chosen, key=lambda c: pair_floor[c[0]])):
    i, j, distance = pairs[index]
    for member, design_index in enumerate((i, j)):
      results.append({
        "iteration": len(results),
        "loss": float(values[design_index]),
        "x_scaled": [float(v) for v in designs[design_index]],
        "pair": int(pair_index),
        "member": int(member),
      })
    manifest.append({
      "pair": int(pair_index),
      "floor_a": float(values[i]),
      "floor_b": float(values[j]),
      "floor_difference": float(abs(values[i] - values[j])),
      "design_distance": distance,
      "ranks": [2 * pair_index, 2 * pair_index + 1],
    })
    print(
      f"  pair {pair_index}: floors {values[i]:.6f} / {values[j]:.6f} "
      f"(difference {abs(values[i] - values[j]):.2e}) separation {distance:.3f} -> ranks "
      f"{2 * pair_index}, {2 * pair_index + 1}", flush=True
    )

  ordered = sorted(range(len(results)), key=lambda k: results[k]["loss"])
  for rank, position in enumerate(ordered):
    if results[position]["pair"] != rank // 2:
      raise SystemExit(
        "floor_matched_pairs: sorting by loss does not put pair members adjacent -- two pairs are "
        "closer in floor than a pair is wide; raise --n-samples or lower --tolerance"
      )

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump({"results": results, "pairs": manifest, "tolerance": arguments.tolerance}, f, indent=1)
  print(f"-> {arguments.output}  ({len(results)} designs, --design-ranks {' '.join(str(k) for k in range(len(results)))})")


if __name__ == "__main__":
  main()
