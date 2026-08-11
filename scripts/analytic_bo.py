#!/usr/bin/env python3
"""BO on the ANALYTIC surrogate of the enzyme objective -- many seeds, no simulator.

    python scripts/analytic_bo.py --n-experiments 4 --n-seeds 64 --n-iterations 60 \
        --output output/analytic/m4.json

The point is statistics. On the real objective a run costs ~45 min, so a single seed decides nothing:
`from_scratch` reaching 0.0701 in 12 designs may be the design lottery rather than evidence that 12
designs suffice. Here an evaluation is microseconds, so the same question can be asked over hundreds
of seeds and every batch size at once.

THE OBJECTIVE, fitted to 128 Sobol designs per batch size (output/landscape) and independently
re-checked:

    loss(design) = y0 * prod_i ( 1 - A(f_i) * exp(-((T_i - c)/w)^2 / 2) )
    A(f) = A0 + A1 * f

`y0` is the no-information level (1/3), the bump is the informative window around the melting prior,
and `A(f)` is the basin DEPTH -- the fraction changes how much an experiment can learn, not the
overall scale. The depth coupling is the correction the verification found: fitted as a scale factor
the fraction looks like a 5% effect, fitted as depth it is worth +0.482 R2 at m = 1.

Held-out R2 of this form against the simulator: 0.71 (m=1) to 0.57 (m=6). It is a caricature -- it
cannot show anything the simulator does not do, and a result here is a hypothesis about the
benchmark, not a measurement of it.
"""
import argparse
import json
import os

import numpy as np

import detopt.bo
from detopt.bo import BayesianOptimizer

# Fitted at m = 1 (output/landscape/m1.npz), depth-coupled form.
Y0, A0, A1, CENTRE, WIDTH = 0.3152, 0.818, -0.497, 53.83, 3.29
TEMPERATURE_BOUNDS = (25.0, 80.0)
FRACTION_BOUNDS = (0.0, 1.0)


def objective(x, m, noise=0.0, rng=None):
  """`x` is (2m,) in the unit cube, laid out [fractions, temperatures] like the detector's design."""
  fraction = FRACTION_BOUNDS[0] + x[:m] * (FRACTION_BOUNDS[1] - FRACTION_BOUNDS[0])
  temperature = TEMPERATURE_BOUNDS[0] + x[m:] * (TEMPERATURE_BOUNDS[1] - TEMPERATURE_BOUNDS[0])
  depth = A0 + A1 * fraction
  value = Y0 * np.prod(1.0 - depth * np.exp(-0.5 * ((temperature - CENTRE) / WIDTH) ** 2))
  if noise > 0.0:
    value = float(value + noise * rng.standard_normal())
  return float(max(value, 1e-6))


def run(mode, m, seed, n_iterations, kernel_name, noise):
  dimension = 2 * m
  rng = np.random.default_rng(seed)
  gp = {"n_folds": 5, "n_restarts": 5, "n_steps": 40,
        "log_lengthscale_prior_bounds": [-2.0, 1.0], "log_amplitude_prior_bounds": [-6.0, 1.5]}
  blocks = (tuple(range(m)), tuple(range(m, 2 * m)))
  kernel = {
      "normalised-invariant-rbf": detopt.bo.NormalisedInvariantRBF,
      "sorting-rbf": detopt.bo.SortingRBF,
      "ard-rbf": detopt.bo.ARDRBF,
  }[kernel_name]
  layout = {} if kernel_name == "ard-rbf" else (
      {"sort_blocks": blocks} if kernel_name == "sorting-rbf" else {"blocks": blocks})
  built = kernel(d=dimension, **layout,
                 constant_value=float(np.exp(sum(gp["log_amplitude_prior_bounds"]))),
                 constant_value_bounds=(float(np.exp(2 * gp["log_amplitude_prior_bounds"][0])),
                                        float(np.exp(2 * gp["log_amplitude_prior_bounds"][1]))),
                 length_scale=float(np.exp(0.5 * sum(gp["log_lengthscale_prior_bounds"]))),
                 length_scale_bounds=(float(np.exp(gp["log_lengthscale_prior_bounds"][0])),
                                      float(np.exp(gp["log_lengthscale_prior_bounds"][1]))))
  optimiser = BayesianOptimizer(dimension, gp=gp, ei={"n_restarts": 16, "n_steps": 60},
                                kernel=built, n_init=5, seed=seed)
  losses = []
  for _ in range(n_iterations):
    x = rng.random(dimension) if mode == "random" else np.asarray(optimiser.propose(), dtype=float)
    value = objective(x, m, noise, rng)
    optimiser.append(x, value, noise=max(noise, 1e-6))
    losses.append(value)
  return losses


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-experiments", type=int, required=True)
  parser.add_argument("--n-seeds", type=int, default=64)
  parser.add_argument("--n-iterations", type=int, default=60)
  parser.add_argument("--kernel", default="normalised-invariant-rbf")
  parser.add_argument("--noise", type=float, default=0.0,
                      help="observation noise sd; 0 = the deterministic surrogate")
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  payload = {"n_experiments": arguments.n_experiments, "kernel": arguments.kernel,
             "noise": arguments.noise, "n_iterations": arguments.n_iterations, "curves": {}}
  for mode in ("bo", "random"):
    curves = []
    for seed in range(arguments.n_seeds):
      losses = run(mode, arguments.n_experiments, seed, arguments.n_iterations,
                   arguments.kernel, arguments.noise)
      curves.append(np.minimum.accumulate(losses).tolist())
      if (seed + 1) % 16 == 0:
        print(f"  {mode} {seed + 1}/{arguments.n_seeds}", flush=True)
    payload["curves"][mode] = curves
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump(payload, f)
  best = {k: float(np.median([c[-1] for c in v])) for k, v in payload["curves"].items()}
  print(f"wrote {arguments.output}: median final BO {best['bo']:.4f}, random {best['random']:.4f}")


if __name__ == "__main__":
  main()
