#!/usr/bin/env python3
"""Sobol-sample the enzyme objective at a given batch size and store (design, loss).

    JAX_PLATFORMS=cpu python scripts/sample_landscape.py --n-experiments 3 --n-designs 1024 \
        --output output/landscape/m3.npz

Runs on CPU so it can share a machine whose GPU is busy. The samples are the input to
`scripts/fit_landscape.py`, which asks whether the objective has a simple analytic form -- if a
saturating bowl in a few symmetric features reproduces it, the benchmark's whole difficulty is
characterised in closed form and BO behaviour can be studied without the simulator.

Sobol rather than uniform: a low-discrepancy set covers a 2m-dimensional cube far more evenly at the
sample sizes affordable here, which matters because the fit has to resolve the shape of the basin
rather than just its depth.
"""
import argparse
import os

import numpy as np
from scipy.stats import qmc

import detopt.detector
import detopt.utils.config
from detopt.bo.gbdt import score_design


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-experiments", type=int, required=True)
  parser.add_argument("--n-designs", type=int, default=1024)
  parser.add_argument("--n-events", type=int, default=8192)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  overrides = [f"enzyme.n_experiments={arguments.n_experiments}"]
  config = detopt.utils.config.override(
      detopt.utils.config.load_config("config/detector/enzyme.yaml"), overrides)
  detector = detopt.detector.from_config(config)
  dimension = int(detector.design_dim())

  # Sobol wants a power of two for its balance properties; `random_base2` refuses anything else.
  points = int(np.ceil(np.log2(max(arguments.n_designs, 2))))
  scaled = qmc.Sobol(d=dimension, scramble=True, seed=arguments.seed).random_base2(points)

  losses, trains, vals = [], [], []
  for i, x in enumerate(scaled):
    design = np.asarray(detector.flatten_design(detector.to_nominal(x.astype(np.float32))),
                        dtype=np.float32)
    score = score_design(detector, design, n_events=arguments.n_events, event_offset=0, seed=0)
    losses.append(score.loss); trains.append(score.train); vals.append(score.val)
    if (i + 1) % 64 == 0:
      print(f"  {i + 1}/{len(scaled)}  median loss {np.median(losses):.4f}", flush=True)

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  np.savez(arguments.output, scaled=scaled.astype(np.float32),
           design=np.array([detector.flatten_design(detector.to_nominal(x.astype(np.float32)))
                            for x in scaled], dtype=np.float32),
           loss=np.array(losses), train=np.array(trains), val=np.array(vals),
           n_experiments=arguments.n_experiments, dimension=dimension,
           temperature_bounds=np.asarray(detector.temperature_bounds),
           enzyme_fraction_bounds=np.asarray(detector.enzyme_fraction_bounds),
           melting_bounds=np.asarray(detector.melting_bounds))
  print(f"wrote {arguments.output}: {len(scaled)} designs, {dimension} dimensions, "
        f"loss {np.min(losses):.4f} .. {np.max(losses):.4f}")


if __name__ == "__main__":
  main()
