#!/usr/bin/env python3
"""Do invariant GP kernels pay on an ANALYTIC permutation-invariant objective, and how does that
scale with the number of interchangeable elements?

The enzyme benchmark answers this question expensively (~3 s per evaluation) and noisily, so it can
only afford ~12 seeds -- not enough to resolve the kernels (measured: 22% power). Here the objective
is closed-form, so 100 seeds cost less than one enzyme run and the comparison is actually powered.

The design is ``m`` interchangeable SCALARS in ``[0, 1]^m``. That is deliberate: for a scalar
element, sorting realises the EXACT quotient metric (``||sort a - sort b|| = min_pi ||a - pi b||``,
the rearrangement inequality), so this measures the invariance itself and not the keyed-invariant
approximation the enzyme's (fraction, temperature) pairs force.

TWO OBJECTIVES, because where the optimum sits relative to the tie locus is the whole argument:

``sphere``      f(x) = sum_i (x_i - 1/2)^2.  Permutation-invariant, and its minimiser is the
                ALL-EQUAL point -- it sits exactly ON the tie locus. The group average's prior
                variance peaks there, so its "defect" points straight at the answer.
``assignment``  f(x) = sum_i (sort(x)_i - a_i)^2 with ``a`` evenly spread over [0, 1]. Also exactly
                invariant (it depends only on the multiset), but its minimiser is a SPREAD set, as
                far from the tie locus as the box allows. This is the honest case, and it is the one
                the enzyme problem resembles -- a good batch probes several distinct temperatures.

Running both separates "the invariant kernel is better" from "the invariant kernel happens to put
its prior mass where this particular optimum is".

    python scripts/benchmark_invariant.py --dimensions 2,3,4,6 --seeds 40 --iterations 60
"""

import argparse
import itertools
import json
import time

import numpy as np

import detopt.bo
from detopt.bo import BayesianOptimizer

GP = {
  "n_folds": 5, "n_restarts": 5, "n_steps": 40,
  "log_lengthscale_prior_bounds": [-6.0, 4.0], "log_amplitude_prior_bounds": [-6.0, 1.5]
}
EI = {"n_restarts": 32, "n_steps": 100}

# The four surrogates, plus the null. `ard` is the same class with no exchangeable block, which is
# how the package expresses "model no symmetry" -- so the comparison isolates the group, not the code.
KERNELS = ("ard-rbf", "permutation-invariant-rbf", "normalised-invariant-rbf", "sorting-rbf")


def objective(name, x):
  """Exactly permutation-invariant in every case: it reads x only through a symmetric function.

  All four are defined on the SCALED cube ``[0, 1]^m`` that BO searches, and all four put their
  minimiser in the interior, so no arm can win by hugging a bound. ``u = 2x - 1`` is the cube
  recentred on ``[-1, 1]^m``, which is where the two saturating objectives are natural."""
  x = np.asarray(x, dtype=float)
  u = 2.0 * x - 1.0
  if name == "sphere":
    return float(((x - 0.5) ** 2).sum())
  if name == "assignment":
    targets = (np.arange(x.size) + 0.5) / x.size
    return float(((np.sort(x) - targets) ** 2).sum())
  # The two SATURATING objectives, added because the three above are polynomial and the enzyme
  # objective is not: its read-out is a threshold, so most of the design space is a plateau where the
  # loss is exactly the target's own variance and the gradient carries no information. A GP on a
  # function that is constant over most of its domain is the actual difficulty of that benchmark,
  # and neither quadratic reproduces it.
  if name == "saturating":
    # tanh(3 sum_i u_i^2): ONE saturating envelope around a shared radius -- a QUADRATIC bottom that
    # flattens into a plateau away from the optimum. At m = 4, 76% of a uniform draw already scores
    # above 0.99, so most of the space carries no gradient: that is the enzyme benchmark's actual
    # difficulty (65.6% of random designs there score the target's own variance exactly).
    #
    # NOT tanh(3 (sum_i u_i^2)^2). The outer square makes the objective QUARTIC at the optimum --
    # measured, it scores 0.077 where this one scores 0.446 a fifth of the way to the corner -- i.e.
    # a wide flat basin with a sharp rim. That adds a second pathology the enzyme does not have: its
    # plateau lies OUTSIDE the informative window, while the optimum inside that window has real
    # curvature. A quartic floor would confound "did the surrogate find the basin" with "can it
    # resolve the basin's floor", which are different questions.
    return float(np.tanh(3.0 * (u**2).sum()))
  if name == "sum-tanh":
    # sum_i tanh(3 |u_i|): SEPARABLE saturation -- each experiment saturates on its own rather than
    # the batch saturating collectively. Same symmetry, same plateau value, but the plateau is
    # reached coordinate by coordinate, so partial information survives where `saturating` has none.
    return float(np.tanh(3.0 * np.abs(u)).sum())
  raise SystemExit(f"unknown objective {name!r}")


def build(name, m):
  ls_low, ls_high = GP["log_lengthscale_prior_bounds"]
  amp_low, amp_high = GP["log_amplitude_prior_bounds"]
  blocks = () if name == "ard-rbf" else (tuple(range(m)),)
  layout = {"sort_blocks": blocks} if name == "sorting-rbf" else {"blocks": blocks}
  return detopt.bo.__kernels__[name](
    d=m, **layout,
    constant_value=float(np.exp(amp_low + amp_high)),
    constant_value_bounds=(float(np.exp(2.0 * amp_low)), float(np.exp(2.0 * amp_high))),
    length_scale=float(np.exp(0.5 * (ls_low + ls_high))),
    length_scale_bounds=(float(np.exp(ls_low)), float(np.exp(ls_high)))
  )


def run(arm, objective_name, m, seed, iterations):
  """One trajectory. ``arm='random'`` draws uniformly from the SAME box BO searches, so the two
  differ only in how the next point is chosen."""
  rng = np.random.default_rng(seed)
  optimiser = None
  if arm != "random":
    optimiser = BayesianOptimizer(m, gp=GP, ei=EI, kernel=build(arm, m), n_init=GP["n_folds"])
  losses = []
  for _ in range(iterations):
    x = rng.uniform(0.0, 1.0, size=m) if arm == "random" else np.asarray(optimiser.propose(int(seed) + step), dtype=float)
    value = objective(objective_name, x)
    if optimiser is not None:
      optimiser.append(x, value, noise=1e-8)  # the function is exact; this is a conditioning jitter
    losses.append(value)
  return np.minimum.accumulate(losses)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--dimensions", default="2,3,4,6", help="comma-separated element counts m")
  parser.add_argument("--objectives", default="sphere,assignment")
  parser.add_argument("--seeds", type=int, default=40)
  parser.add_argument(
    "--kernels", default=",".join(KERNELS + ("random",)),
    help="which surrogates to run; one per shard lets a driver parallelise across cores. The group "
    "average is O(m!) in the EI gradient (k_and_grad_x loops over the whole group in Python) and "
    "becomes unaffordable past m ~ 5, so high-m cells are run with the cheap arms only -- stated "
    "here rather than silently omitted."
  )
  parser.add_argument("--iterations", type=int, default=60)
  parser.add_argument("--output", default="output/invariant/results.json")
  arguments = parser.parse_args()

  dimensions = [int(v) for v in arguments.dimensions.split(",")]
  objectives = arguments.objectives.split(",")
  results = {}
  for objective_name in objectives:
    for m in dimensions:
      for arm in arguments.kernels.split(","):
        start = time.time()
        curves = np.array([run(arm, objective_name, m, seed, arguments.iterations)
                           for seed in range(arguments.seeds)])
        results[f"{objective_name}|{m}|{arm}"] = curves.tolist()
        print(f"{objective_name:11s} m={m} {arm:26s} best@{arguments.iterations}="
              f"{np.median(curves[:, -1]):.5f}  ({time.time() - start:.0f}s)", flush=True)

  import os
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump({"dimensions": dimensions, "objectives": objectives, "seeds": arguments.seeds,
               "iterations": arguments.iterations, "curves": results}, f)
  print(f"wrote {arguments.output}")


if __name__ == "__main__":
  main()
