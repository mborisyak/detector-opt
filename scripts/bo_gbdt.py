#!/usr/bin/env python3
"""Bayesian optimisation of the enzyme design against the GBDT objective, on an ITERATION axis.

This is deliberately NOT ``scripts/bo.py``. There is no detector-call budget, no event pool, no
warm start and no meta-training: every iteration samples generously at the proposed design, fits a
GBDT, reads off the converged loss (:mod:`detopt.bo.gbdt`) and throws the samples away. The only
question it answers is the one that decides whether the benchmark is worth running at all --

    does the best-so-far loss keep improving as iterations accumulate?

If it does, a strategy that fits twice as many iterations into the same wall clock wins by
arithmetic, which is exactly the claim the four-strategy campaign is supposed to demonstrate.

Every design is scored TWICE: on a fixed event block (common random numbers -- the same enzymes
under every design, which makes the objective a deterministic function of the design and removes
evaluation noise from the comparison) and again on a disjoint block BO never sees. The first is
what BO optimises, the second is the held-out check that BO is not just exploiting its own sample.

    python scripts/bo_gbdt.py --mode bo --seed 0 --n-iterations 60 --output output/gbdt/base-bo-0.json
"""

import argparse
import json
import os
import time

import numpy as np

import detopt
import detopt.bo
import detopt.utils.config
from detopt.bo import BayesianOptimizer
from detopt.bo.gbdt import score_design

DETECTOR_CONFIG = "config/detector/enzyme.yaml"
# The held-out block, far from the [0, n_events) block BO optimises against. Event indices are cast
# to int32 by the detector, so this stays well inside that range.
VERIFY_OFFSET = 1 << 24

# The GP/EI settings of config/enzyme.yaml, which this driver does not otherwise read. The
# lengthscale prior is overridable from the command line (`--log-lengthscale-bounds LO HI`) because
# it is the subject of a study: on the recorded designs of finished runs, narrowing it from [-6, 4]
# to [-1.5, 0.5] raises the spread of the GP's predicted means from 0.105 to 0.251 of the data's own
# spread, i.e. it stops the surrogate predicting nearly the same value everywhere. The width is a
# modelling CHOICE about how quickly the objective may vary, so it belongs in the arms file next to
# the kernel rather than baked in here.
GP = {
  "n_folds": 5,
  "n_restarts": 5,
  "n_steps": 40,
  "log_lengthscale_prior_bounds": [-6.0, 4.0],
  "log_amplitude_prior_bounds": [-6.0, 1.5]
}
EI = {"n_restarts": 32, "n_steps": 100}


def run(mode, seed, n_iterations, n_events, overrides, output, verify, fixed_fraction=None,
        n_init=None, kernel_name=None, shared_fraction=False, log_lengthscale_bounds=None,
        log_amplitude_bounds=None):
  # A per-run copy, so a swept prior never leaks into the module-level default.
  gp_settings = dict(GP)
  if log_lengthscale_bounds is not None:
    gp_settings["log_lengthscale_prior_bounds"] = [float(v) for v in log_lengthscale_bounds]
  if log_amplitude_bounds is not None:
    gp_settings["log_amplitude_prior_bounds"] = [float(v) for v in log_amplitude_bounds]
  config = detopt.utils.config.override(detopt.utils.config.load_config(DETECTOR_CONFIG), overrides)
  detector = detopt.detector.from_config(config)
  n = detector.n_experiments

  # With `fixed_fraction` the optimiser searches the TEMPERATURES only and every experiment gets the
  # same, fixed enzyme stock fraction. On the real campaign the four fraction coordinates correlate
  # with the loss at |rho| <= 0.018 and are uniformly spread even among the best 1% of designs, i.e.
  # they are flat directions -- which is also what the GP said by pinning their ARD lengthscales at
  # the ceiling. Dropping them halves the search space, and BO degrades with dimension.
  # Three parameterisations of the same design space:
  #
  #   full            2n coordinates -- every experiment its own (fraction, temperature).
  #   fixed_fraction  n  coordinates -- temperatures only, fraction pinned at a given value.
  #   shared_fraction 1+n            -- ONE fraction for the whole batch, plus n temperatures.
  #
  # `shared_fraction` is the diagonal of the full space where all fractions are equal, so its losses
  # are the SAME function restricted to a subspace and are directly comparable to the full run's.
  # It is also how the assay is actually run: one master mix, split across a temperature gradient.
  if fixed_fraction is not None and shared_fraction:
    raise SystemExit("--fix-fraction pins every experiment's fraction and --shared-fraction searches "
                     "one shared value; they are different parameterisations, pick one")
  searched = int(detector.design_dim())
  if fixed_fraction is not None:
    searched = n
  elif shared_fraction:
    searched = 1 + n
  scaled_fraction = None
  if fixed_fraction is not None:
    # The scaled value the fixed physical fraction corresponds to, so `to_nominal` reproduces it.
    probe = np.concatenate([np.full(n, float(fixed_fraction)), np.full(n, detector.temperature_bounds[0])])
    scaled_fraction = np.asarray(detector.to_scaled(probe), dtype=np.float32)[:n]

  def to_design(x):
    """SCALED coordinates -> the flat NOMINAL design the detector is called with."""
    if scaled_fraction is not None:
      full = np.concatenate([scaled_fraction, x])
    elif shared_fraction:
      full = np.concatenate([np.full(n, x[0], dtype=np.float32), x[1:]])
    else:
      full = x
    return np.asarray(detector.flatten_design(detector.to_nominal(full)), dtype=np.float32)

  # The surrogate kernel is an ARGUMENT, not a config entry: it is the thing this driver exists to
  # compare, and `kernel` is not an EnzymeDetector parameter -- the detector config is validated
  # against __init__, so a `kernel:` key under `enzyme:` raises rather than being read.
  #
  # `--fix-fraction` searches the temperatures alone, so the vector the kernel sees is not the
  # detector's full design and the exchangeable blocks (read off `design_spec`) would name the wrong
  # coordinates. That combination takes plain ARD, and says so rather than silently downgrading a
  # kernel the caller asked for by name.
  if fixed_fraction is not None:
    if kernel_name not in (None, "ard-rbf"):
      raise SystemExit(f"--fix-fraction searches the temperatures alone; the exchangeable blocks are "
                       f"read off the FULL design, so {kernel_name!r} cannot be declared over it")
    kernel_name = "ard-rbf"
  # Tracks config/enzyme.yaml, which selects the normalised group average on ROBUSTNESS: on the
  # analytic benchmark it is the only invariant kernel with no objective where it falls an order of
  # magnitude behind the best (worst case 8.9x, against sorting 14.9x and the bare group average
  # 20.5x), while mean rank ties all three. It is NOT the kernel that won on the enzyme design --
  # sorting did, 1.150x vs 1.053x at 12 seeds, with nothing surviving multiplicity correction. See
  # the comment on `kernel:` in config/enzyme.yaml for what is and is not established.
  kernel_name = kernel_name if kernel_name is not None else "normalised-invariant-rbf"
  kernel_config = {kernel_name: {} if kernel_name == "ard-rbf" else {"exchangeable": n}}
  if searched != int(detector.design_dim()):
    # The COMPOSED kernel: symmetric over the temperatures, ordinary RBF over the single shared
    # fraction, multiplied. Written out here rather than through `kernel_from_config` because that
    # reads its blocks off `design_spec()` -- the FULL 2n design -- and this run searches 1 + n
    # coordinates laid out as [fraction, T_1 .. T_n]. Two lengthscales: one tied across the
    # temperatures (a symmetric function cannot tell experiment 1 from experiment 3), one for the
    # fraction, which is now a single genuinely distinguishable coordinate.
    ls_low, ls_high = gp_settings["log_lengthscale_prior_bounds"]
    amp_low, amp_high = gp_settings["log_amplitude_prior_bounds"]
    # `kernel_from_config` sizes the kernel from `design_spec()` -- the FULL 2n design -- so it
    # cannot be used for any reduced search. Both reduced parameterisations build it here at the
    # dimension actually searched: `--shared-fraction` declares the temperature block exchangeable
    # (layout [fraction, T_1..T_n]), `--fix-fraction` searches temperatures alone and declares no
    # symmetry, because with the fraction gone the kernel would otherwise inherit block indices
    # naming coordinates that are not in the vector.
    blocks = (tuple(range(1, 1 + n)),) if shared_fraction and kernel_name != "ard-rbf" else ()
    # Looked up in the registry rather than branched on two names: an `else` that returns
    # PermutationInvariantRBF silently DOWNGRADES every kernel added later (the normalised variant
    # would have run as the plain group average here, under its own name in the results).
    clazz = detopt.bo.__kernels__[kernel_name]
    # `ard-rbf` is the plain sklearn product and takes no group argument at all; the others differ
    # only in whether their index tuples are averaged over or sorted by.
    layout = {} if kernel_name == "ard-rbf" else (
        {"sort_blocks": blocks} if kernel_name == "sorting-rbf" else {"blocks": blocks})
    kernel = clazz(
      d=searched, **layout,
      constant_value=float(np.exp(amp_low + amp_high)),
      constant_value_bounds=(float(np.exp(2.0 * amp_low)), float(np.exp(2.0 * amp_high))),
      length_scale=float(np.exp(0.5 * (ls_low + ls_high))),
      length_scale_bounds=(float(np.exp(ls_low)), float(np.exp(ls_high)))
    )
    # Record a config that `kernel_from_config` would actually accept -- this field is provenance,
    # not a label. The shared-fraction layout is not expressible through that function (it sizes
    # itself from the full design), which `searched_dimensions` and `shared_fraction` record instead.
    kernel_config = {kernel_name: ({"exchangeable": n} if len(blocks) > 0 else {})}
  else:
    kernel = detopt.bo.kernel_from_config(kernel_config, detector, gp_settings)
  optimiser = BayesianOptimizer(
    searched, gp=gp_settings, ei=EI, kernel=kernel,
    n_init=int(n_init) if n_init is not None else gp_settings["n_folds"], seed=seed
  )
  rng = np.random.default_rng(seed)
  # Normalised MSE -> C: the target is scaled to [-1, 1] over the T_melting prior, so one unit of
  # normalised RMS is the prior's HALF-range in degrees.
  celsius_per_unit = 0.5 * (detector.melting_bounds[1] - detector.melting_bounds[0])

  results, best_loss, best_design = [], np.inf, None
  for iteration in range(n_iterations):
    start = time.time()
    # `random` draws uniformly from the SAME scaled box BO searches, so the two differ only in how
    # the next point is chosen. The scaled cube is each coordinate affinely on its own design range,
    # so a uniform draw here IS a uniform DESIGN -- the null is not warped, which is exactly what
    # the quantile encoding used to get wrong (it crippled BO and its null identically).
    x = (
      rng.uniform(0.0, 1.0, size=searched).astype(np.float32)
      if mode == "random" else np.asarray(optimiser.propose(), dtype=np.float32)
    )
    design = np.asarray(to_design(x), dtype=np.float32)

    score = score_design(detector, design, n_events=n_events, event_offset=0, seed=0)
    # The GP's observation noise, taken from the regressor's own generalisation gap.
    #
    # The reported objective is (train + val) / 2, so the natural uncertainty on it is the SPREAD of
    # the two numbers it averages: how far this design's estimate could be from its converged value.
    # For the GBDT proxy that is |val - train| (the /2 is dropped here deliberately -- the neural
    # driver, whose val is not selected by a stage-wise minimum, uses |val - train| / 2).
    #
    # What this replaces, and why it matters. `score.sem` is the SAMPLING error of the estimate --
    # measured median 0.0026 -- and handing it to the GP as alpha = sem^2 = 6.5e-06 against a fitted
    # prior variance of ~0.008 forces the surrogate to interpolate every point (a ratio of 1:1200).
    # A step-like objective then has to be explained by shortening the lengthscale, which is what
    # collapses it to ~0.24 and makes it swing 2.4x across the early iterations that set the whole
    # trajectory. The generalisation gap is both larger and better founded: measured median
    # |val - train| = 0.050, i.e. alpha = 0.0025, about 31% of the prior variance.
    #
    # Cross-check on the same runs: single-evaluation reproducibility, from scoring every design on
    # a disjoint block too, is sd(loss - verified_loss)/sqrt(2) = 0.00329 -- an order of magnitude
    # above `sem` and an order below this gap. The gap is therefore a generous nugget rather than a
    # noise estimate, which is the intent: it stops the GP interpolating.
    optimiser.append(x, score.loss, noise=max(abs(score.val - score.train), 1e-6))

    entry = {
      "iteration": iteration,
      "design": design.tolist(),
      "x_scaled": x.tolist(),
      "loss": score.loss,
      "loss_sem": score.sem,
      "gp_noise": float(max(abs(score.val - score.train), 1e-6)),
      # The same loss in DEGREES CELSIUS of target resolution. `loss` is a mean squared error on a
      # target scaled by the T_melting prior half-range, so it is only comparable between runs that
      # share that prior; the physical RMSE is comparable across every variant of the benchmark,
      # including ones that change the prior, the population or the objective. Report both, so a
      # variant cannot look better merely by dividing by a bigger number.
      "rmse_c": float(np.sqrt(max(score.loss, 0.0)) * celsius_per_unit),
      "n_learners": score.n_learners
    }
    # The GP's own diagnostics for the proposal it just made (absent while BO is still drawing its
    # initial random points, and for the `random` arm). An ARD lengthscale pinned at the prior
    # ceiling means the GP has declared that design dimension irrelevant, so a run where most of
    # them pin has a nearly flat posterior -- EI is then almost uniform and "BO" is guessing. That
    # is invisible in the loss curve alone, which is why it is recorded per iteration.
    if mode == "bo" and optimiser.last_info is not None:
      entry["gp"] = {k: float(v) for k, v in optimiser.last_info.items()}
    if verify:
      held_out = score_design(detector, design, n_events=n_events, event_offset=VERIFY_OFFSET, seed=0)
      entry["verified_loss"], entry["verified_sem"] = held_out.loss, held_out.sem
    entry["time_s"] = time.time() - start
    results.append(entry)

    if score.loss < best_loss:
      best_loss, best_design = score.loss, design.tolist()
    print(
      f"[{mode} seed {seed}] iter {iteration + 1:3d}/{n_iterations} loss={score.loss:.5f} "
      f"best={best_loss:.5f} learners={score.n_learners:3d} {entry['time_s']:.1f}s", flush=True
    )

  payload = {
    "mode": mode,
    "seed": seed,
    "n_iterations": n_iterations,
    "n_events": n_events,
    "searched_dimensions": searched,
    "fixed_fraction": fixed_fraction,
    "shared_fraction": bool(shared_fraction),
    "overrides": list(overrides),
    # What this run actually ran, resolved AFTER the overrides. Two runs are the same experiment
    # only if these match: the admissible set, the batch size, the enzyme population, the read-out
    # noise and the normaliser all move the objective, so a number carried across a difference in
    # any of them is a number about a different function. Recorded per run rather than assumed from
    # a config file, which changes under the results that were measured with it.
    "settings": {
      "kernel": kernel_config,
      "n_experiments": int(detector.n_experiments),
      "n_measurements": int(detector.n_measurements),
      "temperature_bounds": [float(v) for v in detector.temperature_bounds],
      "enzyme_fraction_bounds": [float(v) for v in detector.enzyme_fraction_bounds],
      "melting_bounds": [float(v) for v in detector.melting_bounds],
      "measurement_noise": float(detector.measurement_noise),
      "duration": float(detector.duration),
      "concentration_E": float(detector.concentration_E),
      "objective_fraction": float(detector.objective_fraction),
      "half_time_bounds": [float(v) for v in detector.half_time_bounds],
      "celsius_per_unit": float(celsius_per_unit),
      "log_lengthscale_prior_bounds": [float(v) for v in gp_settings["log_lengthscale_prior_bounds"]],
      "log_amplitude_prior_bounds": [float(v) for v in gp_settings["log_amplitude_prior_bounds"]]
    },
    "results": results,
    "best_loss": float(best_loss),
    "best_rmse_c": float(np.sqrt(max(best_loss, 0.0)) * celsius_per_unit),
    "best_design": best_design
  }
  if output is not None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
      json.dump(payload, f, indent=2, default=float)
    print(f"wrote {output}")
  return payload


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--mode", choices=["bo", "random"], default="bo")
  parser.add_argument("--seed", type=int, default=0)
  # Half of what `meta` managed in the finished campaign (124), which is the count this has to
  # sustain improvement over for the strategy comparison to mean anything.
  parser.add_argument("--n-iterations", type=int, default=60)
  parser.add_argument("--n-events", type=int, default=16384)
  parser.add_argument("--no-verify", dest="verify", action="store_false", help="skip the held-out re-score")
  parser.add_argument(
    "--fix-fraction", type=float, default=None,
    help="hold every experiment's enzyme fraction at this value and search the temperatures only "
    "(halves the dimension; the fraction coordinates are flat directions -- see docs/enzyme-gbdt-log.md)"
  )
  parser.add_argument("--n-init", type=int, default=None, help="initial Sobol design size (default: gp.n_folds)")
  parser.add_argument(
    "--shared-fraction", action="store_true",
    help="ONE enzyme fraction for the whole batch plus n temperatures (1 + n coordinates instead of "
    "2n): the diagonal of the full design space, and how the assay is actually run -- one master mix "
    "split across a temperature gradient. With the invariant kernel this composes as "
    "symmetricRBF(temperatures) * RBF(fraction)."
  )
  parser.add_argument(
    "--kernel", default=None, choices=sorted(detopt.bo.__kernels__),
    help="GP kernel (default: sorting-rbf -- the batch's symmetry by sorting, which measured best; "
    "permutation-invariant-rbf averages over the group instead; ard-rbf models no symmetry)"
  )
  parser.add_argument(
    "--log-lengthscale-bounds", nargs=2, type=float, default=None, metavar=("LO", "HI"),
    help="prior bounds on log lengthscale (default -6 4). Narrowing this is how the surrogate is "
         "stopped from going flat: it forbids the marginal likelihood from explaining a nearly "
         "constant objective with an effectively infinite lengthscale."
  )
  parser.add_argument(
    "--log-amplitude-bounds", nargs=2, type=float, default=None, metavar=("LO", "HI"),
    help="prior bounds on log amplitude, i.e. constant_value in (exp(2 LO), exp(2 HI)) (default "
         "-6 1.5, six orders of magnitude). The objective is a normalised MSE with known support -- "
         "measured min 0.032, p95 0.325, no-information level 1/3 -- so its variance is bracketed a "
         "priori and the prior variance need not be free over six decades. Measured: the fit lands "
         "at 1.08-1.19x the data variance and never touches these bounds, so this is hygiene "
         "rather than a live defect."
  )
  parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
  parser.add_argument("--output", default=None)
  arguments = parser.parse_args()
  run(
    arguments.mode, arguments.seed, arguments.n_iterations, arguments.n_events, arguments.overrides, arguments.output,
    arguments.verify, arguments.fix_fraction, arguments.n_init, arguments.kernel, arguments.shared_fraction,
    arguments.log_lengthscale_bounds, arguments.log_amplitude_bounds
  )


if __name__ == "__main__":
  main()
