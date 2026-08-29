#!/usr/bin/env python3
"""Mid-trajectory probe of the GP's observation-noise scale, at DESIGN level and with no training.

    python scripts/probe_gp_noise_scale.py --detector-config config/detector/mm_hk_m2.yaml \\
        --runs output/mm-campaign-hk-m2/9577242/*/results.json --prefixes 10 20 30 \\
        --scales 1 10 --repeats 3 --output output/mm-noise-probe/9577242.json

WHAT THIS SEPARATES, AND WHY IT IS NOT A FIXED-DESIGN PROBE. `bo.observation_noise_scale` multiplies
the noise each observation is handed to the GP; it touches nothing else. At a FIXED design there is
no GP and the knob has no channel at all, so the usual per-design probe is blind to it by
construction. The smallest thing the knob can move is ONE PROPOSAL, so that is what is measured:
a finished run's own observations are replayed into a fresh optimiser at each scale, the state is
identical up to the multiplier, and the readout is the ANALYTIC quality of the design each one then
proposes -- the detector's grid posterior on a held-out event block, no network anywhere.

THE FORM. Three mid-trajectory prefixes (never index 0, where there is nothing to fit), every scale
on the IDENTICAL prefix, and `--repeats` proposal seeds per cell so the spread of the acquisition's
own restarts is visible rather than assumed. The GP's fitted log-lengthscale and log-amplitude are
recorded beside each proposal, because a scale that does not move them has not been tested.
"""
import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import detopt.bo
import detopt.detector
import detopt.utils.config
from detopt.bo import BayesianOptimizer

from score_incumbents_analytic import build_detector, score


def optimiser(detector, bo_config):
  gp = dict(bo_config["gp"])
  kernel = detopt.bo.kernel_from_config(gp.pop("kernel"), detector, gp)
  return BayesianOptimizer(
    int(detector.design_dim()), gp=gp, ei=dict(bo_config["ei"]), kernel=kernel,
    n_init=int(bo_config.get("n_init", gp["n_folds"]))
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--detector-config", required=True)
  parser.add_argument("--runs", nargs="+", required=True)
  parser.add_argument("--prefixes", nargs="+", type=int, default=[10, 20, 30])
  parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 10.0])
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--event-start", type=int, default=50_000_000)
  parser.add_argument("--n-events", type=int, default=32768)
  parser.add_argument("--chunk", type=int, default=2048)
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  detector = build_detector(arguments.detector_config, [])
  event_index = np.arange(arguments.event_start, arguments.event_start + arguments.n_events, dtype=np.int64)

  records = []
  for path in arguments.runs:
    with open(path) as handle:
      trajectory = json.load(handle)
    rows = [r for r in trajectory.get("results", []) if r.get("loss") is not None]
    bo_config = trajectory["config"]["bo"]
    seed, arm = path.split(os.sep)[-3], path.split(os.sep)[-2]
    for prefix in arguments.prefixes:
      if len(rows) < prefix:
        print(f"[skip] {seed}/{arm} prefix {prefix}: only {len(rows)} scored designs", flush=True)
        continue
      history = rows[:prefix]
      for repeat in range(arguments.repeats):
        propose_seed = zlib.crc32(f"{seed}/{arm}/{prefix}/{repeat}".encode()) % (2**31 - 1)
        for scale in arguments.scales:
          book = optimiser(detector, bo_config)
          for row in history:
            book.append(np.asarray(row["x_scaled"], np.float32), float(row["loss"]), noise=float(row["loss_std"]) * scale)
          proposal = book.propose(propose_seed)
          record = {
            "seed": seed,
            "arm": arm,
            "prefix": prefix,
            "repeat": repeat,
            "scale": scale,
            "x_scaled": [float(v) for v in np.asarray(proposal).ravel()],
            "analytic": score(detector, proposal, event_index, arguments.chunk),
            "best_observed": float(min(r["loss"] for r in history)),
          }
          record.update({k: float(v) for k, v in (book.last_info or {}).items()})
          records.append(record)
          print(
            f"{seed:>11} {arm:>12} prefix={prefix:>3} rep={repeat} scale={scale:>5g} "
            f"analytic={record['analytic']:.6f} ei={record.get('ei', float('nan')):.3e} "
            f"log_l={record.get('log_lengthscale_mean', float('nan')):+.3f} "
            f"log_a={record.get('log_amplitude', float('nan')):+.3f}", flush=True
          )

  os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
  with open(arguments.output, "w") as handle:
    json.dump({
      "detector_config": arguments.detector_config,
      "event_start": arguments.event_start,
      "n_events": arguments.n_events,
      "records": records
    }, handle, indent=1)
  print(f"[write] {arguments.output}")


if __name__ == "__main__":
  main()
