#!/usr/bin/env python3
"""Score a campaign's incumbent designs with the detector's OWN analytic estimator -- no network.

    python scripts/score_incumbents_analytic.py --detector-config config/detector/mm_hk_m2.yaml \\
        --campaign control=output/mm-campaign-hk-m2 --campaign scale10=output/mm-noise10-hk-m2 \\
        --output output/mm-noise10-hk-m2/analytic.json

WHY NOT THE REPORTED LOSS. A neural run reports `design quality + a residual that depends on the arm
and on how that design's network happened to train`, and on tasks where the residual is comparable
to the spread between designs the reported number cannot order two campaigns. The instrument here
has no network in it: each cell's incumbent `x_scaled` is handed to the detector's grid posterior.

ONE BLOCK FOR EVERYTHING. Every cell of every campaign is scored on the SAME held-out event indices,
so the comparison is paired in the design; the block is placed far from any index a run trains on.
It is scored in equal chunks whose means are averaged, so the value is the block mean whatever the
memory limit. `--n-random` scores uniform designs on that same block as the no-search reference.

An INCOMPLETE cell is scored and flagged, never silently dropped: a campaign short of a cell is a
different measurement from one whose cell converged.
"""
import argparse
import glob
import json
import os

import numpy as np

import detopt.detector
import detopt.utils.config


def build_detector(path, overrides):
  import yaml

  with open(path) as handle:
    config = yaml.safe_load(handle)
  if len(overrides) > 0:
    config = detopt.utils.config.override(config, overrides)
  return detopt.detector.from_config(config)


def score(detector, scaled, event_index, chunk):
  """Mean analytic loss of one SCALED design over `event_index`, in chunks of `chunk` events."""
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(scaled, np.float32))
  blocks = [event_index[start:start + chunk] for start in range(0, event_index.shape[0], chunk)]
  total, seen = 0.0, 0
  for block in blocks:
    _, event, _, target = detector(design, block)
    predicted = detector.estimate(design, event)
    losses = detector.loss(predicted, detector.normalize_target(target))
    total += float(jnp.sum(losses))
    seen += int(block.shape[0])
  return total / seen


def incumbent(rows):
  """The best-reported scored row of a trajectory, and the reported summary beside it."""
  scored = [r for r in rows if r.get("loss") is not None]
  if len(scored) == 0:
    return None
  best = min(scored, key=lambda r: float(r["loss"]))
  return {
    "x_scaled": [float(v) for v in best["x_scaled"]],
    "design": [float(v) for v in best["design"]],
    "reported_loss": float(best["loss"]),
    "reported_iteration": int(best["iteration"]),
    "n_designs": len(scored),
    "calls_per_design": float(np.mean([int(r["spent"]) for r in scored])),
    "mean_loss_std": float(np.mean([float(r["loss_std"]) for r in scored])),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--detector-config", required=True)
  parser.add_argument("--campaign", action="append", required=True, help="NAME=ROOT, repeatable")
  parser.add_argument("--override", action="append", default=[], help="detector-config override, repeatable")
  parser.add_argument("--event-start", type=int, default=50_000_000)
  parser.add_argument("--n-events", type=int, default=32768)
  parser.add_argument("--chunk", type=int, default=4096)
  parser.add_argument("--n-random", type=int, default=0)
  parser.add_argument("--random-seed", type=int, default=0)
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  detector = build_detector(arguments.detector_config, arguments.override)
  event_index = np.arange(arguments.event_start, arguments.event_start + arguments.n_events, dtype=np.int64)

  cells = []
  for entry in arguments.campaign:
    name, _, root = entry.partition("=")
    if len(root) == 0:
      raise SystemExit(f"--campaign expects NAME=ROOT, got {entry!r}")
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "results.json"))):
      with open(path) as handle:
        trajectory = json.load(handle)
      record = incumbent(trajectory.get("results", []))
      if record is None:
        print(f"[skip] {path}: no scored design")
        continue
      record.update(
        campaign=name, seed=os.path.basename(os.path.dirname(os.path.dirname(path))),
        arm=os.path.basename(os.path.dirname(path)), completed=trajectory.get("completed") is True,
        detector_calls_used=int(trajectory.get("detector_calls_used", 0)),
        noise_scale=float(trajectory.get("config", {}).get("bo", {}).get("observation_noise_scale", 1.0))
      )
      record["analytic"] = score(detector, record["x_scaled"], event_index, arguments.chunk)
      cells.append(record)
      flag = "" if record["completed"] else "  INCOMPLETE"
      print(
        f"{name:>10} {record['seed']:>11} {record['arm']:>12} n={record['n_designs']:>3} "
        f"reported={record['reported_loss']:.5f} analytic={record['analytic']:.6f} "
        f"scale={record['noise_scale']:g}{flag}", flush=True
      )

  random_scores = []
  if arguments.n_random > 0:
    rng = np.random.default_rng(arguments.random_seed)
    for _ in range(arguments.n_random):
      random_scores.append(score(detector, rng.random(detector.design_dim()), event_index, arguments.chunk))
    print(f"[random] n={arguments.n_random} mean={np.mean(random_scores):.6f} median={np.median(random_scores):.6f}")

  os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
  with open(arguments.output, "w") as handle:
    json.dump({
      "detector_config": arguments.detector_config,
      "overrides": arguments.override,
      "event_start": arguments.event_start,
      "n_events": arguments.n_events,
      "cells": cells,
      "random": random_scores
    }, handle, indent=1)
  print(f"[write] {arguments.output}")


if __name__ == "__main__":
  main()
