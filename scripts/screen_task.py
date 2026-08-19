#!/usr/bin/env python3
"""Screen a CANDIDATE TASK against the benchmark acceptance criteria -- one instrument for all.

    python scripts/screen_task.py --config config/detector/enzyme.yaml --label melt \
        --m 2 3 4 --n-designs 512 --seeds 10 --iters 30 60 --output output/screen/melt.json

`docs/benchmark-acceptance.md` is the specification; this is its implementation. It exists so that
several candidate tasks are compared by IDENTICAL code rather than by several readings of the same
document -- the failure mode being that three implementations of "top-decile spread" are three
different numbers.

WHAT IT REPORTS. The VERDICT is criterion (1), computed WITHIN the task against the task's own
resolution. Everything else is a diagnostic that explains the verdict. Two candidates are two
different problems -- different targets, different design spaces, different loss distributions -- so
these diagnostics are NOT a ranking scale across tasks, however dimensionless they look.

    ceiling   the no-information loss: what an estimator scores by predicting the prior mean.
              Computed from the data (the variance of the normalised target), not assumed, so it
              is right for any regression target. Pass --ceiling for a target whose ceiling is
              known analytically (e.g. 1.0 for cross-entropy normalised by ln K).
    best      the best design in ITS OWN Sobol sample, at the same sample size for every candidate.
    range     ceiling - best. Every metric below is a fraction of it.

    iteration_test       best@n - best@2n over `seeds` proxy runs: the fraction of seeds whose gain
                         exceeds the task's own resolution, and the median gain. THIS IS THE
                         VERDICT -- criterion (1) of docs/benchmark-acceptance.md.
    ceiling_fraction     DIAGNOSTIC: designs within 5% of the ceiling -- how much of the box is dead.
    top_decile_spread    DIAGNOSTIC: spread inside the best 10%, as % of the task's own range.
    gp_r2                DIAGNOSTIC: held-out R2 of a GP on the landscape -- is it modelable at all.
    seconds_per_design   measured, and the projected hours for a 5-seed neural campaign.

THE PROXY. Scoring a design on the neural objective takes 27-137 s; on the gradient-boosted proxy
~0.7 s, at Spearman 0.940 against it (`detopt/bo/gbdt.py`). Every number here is therefore a PROXY
number: it decides which candidate earns a neural campaign, it does not replace one.

CALIBRATION. Before trusting it on a new task, run it on the melt task (config/detector/enzyme.yaml,
m=4) and check it reproduces what is already known there: ~4.7% of designs at the ceiling, a top
decile spanning ~2.9% of the range, GP R2 collapsing to about -0.05 at 8 dimensions, and criterion
(1) FAILING. That is a check on the INSTRUMENT, not a bar for other tasks to clear -- the melt is
the worked example of failure, and no other task's numbers are compared with its.
"""
import argparse
import json
import math
import os
import time

# BLAS/OpenMP default to every core on the machine, which is wrong under a scheduler: SLURM says
# which cores this job may use, not how many threads it should start. Several jobs each spawning a
# dozen BLAS threads onto four allocated cores is how this box reached load 44. Set before numpy is
# imported -- the thread pools are sized at import time.
#
# THE BLAS VARIABLES ARE NOT ENOUGH, AND THAT IS THE BUG THIS BLOCK NOW FIXES. MEASURED: with all
# four of them set to 2, each screen still ran 25 THREADS inside a `cpu=2` allocation and achieved
# only ~1.1 cores of real work with the node at CPULoad 10.97. The extra threads are XLA's own CPU
# pools, which `detopt.bo` starts on import and which size themselves from the machine's core count,
# not from the allocation -- so they are invisible to OMP_NUM_THREADS and must be capped in
# XLA_FLAGS, before jax is imported. XGBoost is capped separately, per call, through `n_threads`.
_allocated = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, _allocated)
os.environ.setdefault(
  "XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={_allocated}"
)

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold

import detopt.bo
import detopt.detector
import detopt.utils.config
from detopt.bo import BayesianOptimizer
from detopt.bo.gbdt import score_design


def build(config_path, n_experiments=None, overrides=()):
  """The detector under test. `n_experiments` overrides the config's batch size where the detector
  has one, so a single config screens several batch sizes.

  `overrides` are the repo's usual dotted ``key=value`` strings, applied by
  `detopt.utils.config.override` exactly as `scripts/enzyme_landscape.py` and `scripts/bo_gbdt.py`
  apply them (and as `scripts/screen_sweep.sh` already sweeps `n_measurements`). A lever is therefore
  swept from the command line rather than by one near-duplicate config per value -- and the override
  lands in the report's `settings`, so what was actually built stays on the record. The values swept
  must be pre-registered in the candidate's own gate document; this is a mechanism for varying a
  DECLARED value, not for inventing one."""
  config = detopt.utils.config.override(detopt.utils.config.load_config(config_path), list(overrides))
  (name, ) = [k for k in config if isinstance(config[k], dict)]
  if n_experiments is not None:
    config[name] = dict(config[name], n_experiments=int(n_experiments))
  return detopt.detector.from_config(config)


def no_information_level(detector, n_events, seed):
  """What an estimator scores by predicting the prior mean -- MEASURED, not assumed.

  This is the variance of the normalised target over the population, which is exactly the loss of
  the best design-independent predictor. For a target scaled to [-1, 1] with a uniform prior it is
  1/3; computing it rather than hard-coding 1/3 keeps the screener honest for targets whose prior is
  not uniform, and catches a normalisation mistake immediately.
  """
  rng = np.random.default_rng(seed)
  design = rng.random(int(detector.design_dim())).astype(np.float32)
  _, _, _, target = detector(design, np.arange(n_events, dtype=np.int64))
  normalised = np.asarray(detector.normalize_target(target), np.float32)
  return float(np.mean(np.square(normalised - normalised.mean(axis=0))))


def sobol_landscape(detector, n_designs, n_events, seed, n_threads=None):
  """Sobol designs and their proxy scores. Low-discrepancy rather than uniform: at these sample
  sizes it covers a 2m-dimensional cube far more evenly, which the top-decile statistic needs.

  Alongside the loss, each design's ATTAINABLE RESOLUTION is kept: ``|val - train| + hypot(sems)``
  at the validation minimum, the same expression `detopt/nn/trainer/design.py` stops on. It is a
  distribution over designs, never one number -- a campaign fails at its WORST design, so the
  maximum is what binds and the median only says what a typical design costs."""
  dimension = int(detector.design_dim())
  designs = qmc.Sobol(d=dimension, scramble=True, seed=seed).random(n_designs).astype(np.float32)
  losses, elapsed = np.empty(len(designs)), []
  slack = np.empty(len(designs))
  gaps = np.empty(len(designs))
  errors = np.empty(len(designs))
  for i, scaled in enumerate(designs):
    # score_design takes a NOMINAL (physical) design; the Sobol point is in the scaled [0,1] box.
    # Passing the scaled vector straight through makes "temperature" ~0.5 C -- a dead enzyme, every
    # design at the ceiling, and a landscape that looks flat for a reason that is entirely an
    # encoding mistake. (Caught by the melt calibration: best 0.2714 where 0.0477 was known.)
    design = np.asarray(detector.flatten_design(detector.to_nominal(scaled)), dtype=np.float32)
    started = time.time()
    # A per-design event offset: every design sees its OWN draw, so the landscape is not a single
    # population's quirks. (Common random numbers is the right choice INSIDE an optimiser run,
    # where designs are being compared; here the spread across designs is the quantity.)
    score = score_design(
      detector, design, n_events=n_events, event_offset=i * n_events, seed=seed, n_threads=n_threads
    )
    elapsed.append(time.time() - started)
    losses[i] = score.loss
    gaps[i] = abs(score.val - score.train)
    errors[i] = math.hypot(score.train_sem, score.val_sem)
    slack[i] = gaps[i] + errors[i]
  return designs, losses, float(np.median(elapsed)), {"gap": gaps, "err": errors, "slack": slack}


def gp_r2(designs, losses, n_folds=5, seed=0):
  """Held-out R2 of a GP on the landscape: can a surrogate model this task at all.

  Not the optimiser's own GP -- a plain ARD RBF with a learned noise term, fitted on the raw sample,
  so the number says something about the LANDSCAPE rather than about the acquisition function.
  """
  centred = losses - losses.mean()
  scale = centred.std()
  if scale <= 0:
    return float("nan")
  kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.full(designs.shape[1], 0.3), (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-8, 1e0))
  )
  predictions = np.empty_like(centred)
  for train, test in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(designs):
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=1, random_state=seed)
    model.fit(designs[train], centred[train] / scale)
    predictions[test] = model.predict(designs[test]) * scale
  return float(1.0 - np.sum(np.square(centred - predictions)) / np.sum(np.square(centred)))


def proxy_runs(
  detector, n_iterations, seeds, n_events, mode, kernel_name, exchangeable, seed_offset=0, log_lengthscale_bounds=(-2.0, 1.0),
  n_threads=None
):
  """`seeds` BO (or random) runs on the proxy, returning each run's best-so-far curve.

  `seed_offset` shifts the block of run seeds so several jobs can each produce a DISJOINT block of
  runs that pool into one sample -- the criterion compares curves from DIFFERENT runs, so it needs
  far more of them than a per-seed verdict does, and one process is not the unit of the sample.

  `log_lengthscale_bounds` is the GP hyperprior on the RBF lengthscale, in NATURAL log, over designs
  living in the scaled cube [0, 1]^d. It is a modelling choice about how fast the objective may vary,
  not a bound on the search, and it is passed in so a sweep of it is visible in the report rather
  than buried here."""
  dimension = int(detector.design_dim())
  gp = {
    "n_folds": 5,
    "n_restarts": 5,
    "n_steps": 40,
    "log_lengthscale_prior_bounds": [float(v) for v in log_lengthscale_bounds],
    "log_amplitude_prior_bounds": [-6.0, 1.5]
  }
  # BUILD THE KERNEL THE WAY scripts/bo.py DOES, i.e. through kernel_from_config with the DETECTOR.
  # An earlier version constructed it with `d` alone and no `blocks`, which silently made the
  # permutation group the IDENTITY: `normalised-invariant-rbf` then degrades to a plain ARD RBF and
  # the screen measures BO deprived of the one symmetry the design actually has. Every candidate here
  # is a batch of interchangeable experiments, so that understates all of them -- equally, but the
  # question is whether the TASK converts iterations, not whether a handicapped optimiser does.
  # `exchangeable` = the number of interchangeable elements; the blocks come off the detector's
  # design_spec, so a detector with a different layout cannot be mis-wired here.
  kernel_config = ({kernel_name: {}} if kernel_name == "ard-rbf" else {kernel_name: {"exchangeable": int(exchangeable)}})
  curves = []
  for seed in range(int(seed_offset), int(seed_offset) + int(seeds)):
    rng = np.random.default_rng(1000 + seed)
    kernel = detopt.bo.kernel_from_config(dict(kernel_config), detector, gp)
    optimiser = BayesianOptimizer(dimension, gp=gp, ei={"n_restarts": 16, "n_steps": 60}, kernel=kernel, n_init=5)
    # THE SEED CONTRACT OF `scripts/bo.py`, reproduced here rather than approximated: one
    # SeedSequence per run, one spawn per iteration, and that integer seeds BOTH the proposal (the
    # initial Sobol block on the first call, the GP fit and the EI restarts after it) and the
    # scoring. Two runs at different `seed` are therefore independent draws of a whole trajectory,
    # which is exactly the unit the between-run criterion compares.
    iteration_seq = np.random.SeedSequence(int(seed))
    losses, sems = [], []
    for iteration in range(n_iterations):
      iteration_seed = int(iteration_seq.spawn(1)[0].generate_state(1)[0])
      # `x` is the SCALED design the optimiser works in; the detector needs it in nominal units.
      x = (rng.random(dimension) if mode == "random" else np.asarray(optimiser.propose(iteration_seed), dtype=float))
      nominal = np.asarray(detector.flatten_design(detector.to_nominal(np.asarray(x, np.float32))), dtype=np.float32)
      score = score_design(
        detector, nominal, n_events=n_events, event_offset=(seed * n_iterations + iteration) * n_events, seed=seed,
        n_threads=n_threads
      )
      # The GP is told the observation's own SEM, matching what scripts/bo.py does with the neural
      # objective, so proxy and neural runs are driven by the same noise convention.
      optimiser.append(x, score.loss, noise=max(score.sem, 1e-6))
      losses.append(score.loss)
      sems.append(score.sem)
    losses = np.asarray(losses)
    # The INCUMBENT's own uncertainty at each iteration -- the error on the number the criterion
    # compares, not the error of whatever design happened to be evaluated last.
    incumbent = np.minimum.accumulate(losses)
    incumbent_sem = [sems[int(np.argmin(losses[:i + 1]))] for i in range(len(losses))]
    curves.append({"best": incumbent.tolist(), "best_sem": incumbent_sem})
  return curves


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, help="detector config, e.g. config/detector/enzyme.yaml")
  parser.add_argument("--label", required=True, help="candidate name, used in the output")
  parser.add_argument(
    "--provenance", required=True, help="markdown file declaring, per PARAMETER, its literature source or "
    "ESTIMATE+reasoning, and per BOUND, the rule that produced it using only "
    "the prior and the noise floor. This is the GATE of "
    "docs/benchmark-acceptance.md 1.1 -- realistic parameters and defensible "
    "ranges -- and it is required: numbers from a task whose provenance is "
    "not declared are not considered."
  )
  parser.add_argument("--m", type=int, nargs="*", default=[2, 3, 4], help="batch sizes to screen")
  parser.add_argument(
    "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
    help="dotted config override, the repo's usual form, e.g. "
    "--set enzyme_inhib.n_measurements=16. The values swept must be "
    "pre-registered in the candidate's gate document."
  )
  parser.add_argument("--n-designs", type=int, default=512, help="Sobol designs per batch size")
  parser.add_argument("--n-events", type=int, default=4096, help="events per design -- BE GENEROUS")
  parser.add_argument("--seeds", type=int, default=10, help="proxy BO runs for the iteration test")
  parser.add_argument(
    "--iters", type=int, nargs=2, default=[15, 30], metavar=("N", "2N"),
    help="the doubling; N must be >= 10 so it starts from a real run, not the "
    "n_init=5 random block"
  )
  parser.add_argument("--kernel", default="normalised-invariant-rbf")
  parser.add_argument(
    "--ceiling", type=float, default=None, help="override the measured no-information level (e.g. 1.0 for CE/lnK)"
  )
  parser.add_argument(
    "--loss-precision", type=float, default=1.0e-2, help="the task's `error`: the slack its convergence criterion allows a "
    "reported loss to carry. Criterion (d) is gain > 10 * this."
  )
  parser.add_argument(
    "--progress-fraction", type=float, default=0.2, metavar="F",
    help="criterion (c): the doubling must add at least F of the progress already "
    "made, gain > F * (baseline - loss@n). The user's 0.2."
  )
  parser.add_argument(
    "--error-multiple", type=float, default=10.0, metavar="K", help="criterion (d): gain > K * loss_precision. The user's 10."
  )
  parser.add_argument("--screen-m", type=int, default=None, help="batch size for the iteration test")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--seed-offset", type=int, default=0, help="first run seed of the iteration test (default 0). Disjoint offsets let "
    "several jobs build ONE pooled sample of independent runs, which the "
    "between-run form of the criterion needs and a 10-seed cell cannot give."
  )
  parser.add_argument(
    "--log-lengthscale-bounds", type=float, nargs=2, default=[-2.0, 1.0], metavar=("LO", "HI"),
    help="GP hyperprior on the RBF lengthscale in NATURAL log, over the scaled "
    "cube [0, 1]^d. A modelling choice about how fast the objective may vary, "
    "NOT a bound on the search; keep it wide enough that the GP can still fit "
    "the landscape rather than be told the answer."
  )
  parser.add_argument(
    "--skip-random-arm", action="store_true", help="omit the BO-vs-random control, which costs as much as the test itself. "
    "For pooling jobs only -- the control belongs in the cell's own screen."
  )
  parser.add_argument(
    "--n-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
    help="threads XGBoost may start, defaulting to the SLURM allocation. Left to its own default it "
    "takes every VISIBLE core, which under a scheduler is every core on the box rather than every "
    "core this job was given."
  )
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  if not os.path.isfile(arguments.provenance):
    raise SystemExit(
      f"screen_task: --provenance {arguments.provenance} does not exist. The gate "
      f"(realistic parameters, defensible bounds) is declared, not inferred -- write "
      f"it before screening, not after seeing the numbers."
    )
  with open(arguments.provenance) as f:
    provenance = f.read()

  report = {
    "label": arguments.label,
    "config": arguments.config,
    "settings": vars(arguments),
    "provenance": provenance,
    "landscape": {}
  }
  landscape_losses = {}

  # ---------------------------------------------------------------- landscape, per batch size
  for m in (arguments.m if arguments.n_designs > 0 else []):
    detector = build(arguments.config, m, arguments.overrides)
    ceiling = arguments.ceiling if arguments.ceiling is not None else \
        no_information_level(detector, arguments.n_events, arguments.seed)
    designs, losses, seconds, resolution = sobol_landscape(
      detector, arguments.n_designs, arguments.n_events, arguments.seed, n_threads=arguments.n_threads
    )
    order = np.sort(losses)
    best = float(order[0])
    span = ceiling - best
    top = order[:max(3, len(order) // 10)]
    entry = {
      "dimension": int(detector.design_dim()),
      "ceiling": float(ceiling),
      "best": best,
      "range": float(span),
      # the criterion's anchor: the median random design, not the no-information ceiling
      "baseline_median_random": float(np.median(losses)),
      "ceiling_fraction_pct": float((losses >= 0.95 * ceiling).mean() * 100.0),
      "top_decile_spread_pct": float(top.std() / span * 100.0),
      "random_spread_pct": float(losses.std() / span * 100.0),
      "gp_r2": gp_r2(designs, losses, seed=arguments.seed),
      "seconds_per_design": seconds,
      "flat_columns": int(detector.n_experiments * detector.n_measurements),
      "slack_median": float(np.median(resolution["slack"])),
      "slack_max": float(np.max(resolution["slack"])),
      "gap_median": float(np.median(resolution["gap"])),
      "gap_max": float(np.max(resolution["gap"])),
      "err_median": float(np.median(resolution["err"])),
      "err_max": float(np.max(resolution["err"])),
    }
    report["landscape"][str(m)] = entry
    landscape_losses[m] = losses
    print(
      f"m={m} dim={entry['dimension']:2d} | ceiling {ceiling:.4f} best {best:.4f} | "
      f"at-ceiling {entry['ceiling_fraction_pct']:5.1f}% | top-decile {entry['top_decile_spread_pct']:5.1f}% "
      f"| GP R2 {entry['gp_r2']:+.3f} | {seconds:.2f} s/design", flush=True
    )
    np.savez(
      os.path.splitext(arguments.output)[0] + f"_m{m}.npz", designs=designs, losses=losses, gap=resolution["gap"],
      err=resolution["err"], slack=resolution["slack"]
    )
    print(
      f"      |train-val|+err over {arguments.n_designs} designs: median {entry['slack_median']:.4f} "
      f"max {entry['slack_max']:.4f}   (gap median {entry['gap_median']:.4f} max {entry['gap_max']:.4f}; "
      f"err median {entry['err_median']:.5f} max {entry['err_max']:.5f})", flush=True
    )

  # ---------------------------------------------------------------- criterion (1): n -> 2n
  m = arguments.screen_m if arguments.screen_m is not None else arguments.m[-1]
  detector = build(arguments.config, m, arguments.overrides)
  n, n2 = arguments.iters
  print(
    f"\niteration test at m={m}: {arguments.seeds} proxy runs from seed {arguments.seed_offset}, "
    f"{n} -> {n2}", flush=True
  )
  curves = proxy_runs(
    detector, n2, arguments.seeds, arguments.n_events, "bo", arguments.kernel, m, seed_offset=arguments.seed_offset,
    log_lengthscale_bounds=arguments.log_lengthscale_bounds, n_threads=arguments.n_threads
  )
  at_n = np.array([c["best"][n - 1] for c in curves])
  at_2n = np.array([c["best"][n2 - 1] for c in curves])
  gains = at_n - at_2n
  # baseline := the MEDIAN loss of RANDOM designs. Taken from this task's own Sobol landscape at the
  # screened batch size -- those ARE random designs, scored by the same instrument as the BO runs.
  # WITH NO LANDSCAPE (`--n-designs 0`) there is nothing to anchor it to and the four-condition
  # verdict is not computed; such a job exists only to add curves to a pooled sample, and it says so
  # rather than inventing a baseline out of the runs it is judging.
  baseline = float(np.median(landscape_losses[m])) if m in landscape_losses else float("nan")
  progress = baseline - at_n  # what BO has achieved by n
  bar_c = arguments.progress_fraction * progress
  bar_d = arguments.error_multiple * arguments.loss_precision

  # docs/benchmark-acceptance.md 2.1 -- the user's criterion.
  beat_baseline = at_n < baseline  # (a)
  improved = at_2n < at_n  # (b)
  enough_progress = gains > bar_c  # (c)
  resolvable = gains > bar_d  # (d)
  strong = beat_baseline & improved & enough_progress & resolvable
  weak = beat_baseline & improved  # required of EVERY seed, passing or not
  passing = int(strong.sum())

  median_gain = float(np.median(gains))
  report["iteration_test"] = {
    "m": m,
    "n": n,
    "2n": n2,
    "seeds": arguments.seeds,
    "baseline_median_random": baseline,
    "progress_fraction": arguments.progress_fraction,
    "error_multiple": arguments.error_multiple,
    "loss_precision": arguments.loss_precision,
    "best_at_n": at_n.tolist(),
    "best_at_2n": at_2n.tolist(),
    "gains": gains.tolist(),
    # THE WHOLE best-so-far CURVE per seed, so any doubling pair below `2n` is recoverable from this
    # one run rather than costing another. `at_n`/`at_2n` above are just two columns of it, and the
    # pairs that will be read off it are pre-registered in docs/tuning-preregistration.md -- keeping
    # every curve is what stops a pair being chosen after the numbers are in.
    "curves": [[float(v) for v in curve["best"]] for curve in curves],
    "bar_c_per_seed": bar_c.tolist(),
    "bar_d": bar_d,
    "seeds_beating_baseline": int(beat_baseline.sum()),
    "seeds_improving": int(improved.sum()),
    "seeds_enough_progress": int(enough_progress.sum()),
    "seeds_resolvable": int(resolvable.sum()),
    "seeds_strong": passing,
    "seeds_weak": int(weak.sum()),
    "median_gain": median_gain,  # reported, not a condition
    "n_at_least_10": bool(n >= 10),
    # 50+% of seeds meet all four, AND every remaining seed still beats the baseline and still
    # improves -- so the failures are smaller gains, never absent ones.
    "PASS": bool(n >= 10 and arguments.seeds >= 10 and passing > arguments.seeds / 2 and int(weak.sum()) == arguments.seeds),
  }
  it = report["iteration_test"]
  if not np.isfinite(baseline):
    it["PASS"] = None
    print(
      "  VERDICT NOT COMPUTED: no landscape in this job (--n-designs 0), so there is no baseline. "
      "The curves are written and are meant to be pooled with other jobs.", flush=True
    )
  print(f"  baseline (median random design): {baseline:.4f}")
  print(f"  best@{n}:  {np.array2string(at_n, precision=4)}")
  print(f"  best@{n2}: {np.array2string(at_2n, precision=4)}")
  print(f"  gains:   {np.array2string(gains, precision=4)}")
  print(f"  (a) beat baseline      {it['seeds_beating_baseline']}/{arguments.seeds}")
  print(f"  (b) improved n->2n     {it['seeds_improving']}/{arguments.seeds}")
  print(f"  (c) > {arguments.progress_fraction:g}*(baseline-loss@n)  {it['seeds_enough_progress']}/{arguments.seeds}")
  print(f"  (d) > {arguments.error_multiple:g}*loss_precision = {bar_d:g}   {it['seeds_resolvable']}/{arguments.seeds}")
  print(f"  STRONG (all four): {passing}/{arguments.seeds} seeds   (need > {arguments.seeds / 2:g})")
  print(f"  WEAK  (a)+(b) on EVERY seed: {it['seeds_weak']}/{arguments.seeds}   (need all)")
  print(f"  median gain {median_gain:.4f} (reported, not a condition) -> "
        f"{'PASS' if it['PASS'] else 'FAIL'}", flush=True)
  if n < 10 or arguments.seeds < 10:
    print(f"  !! n={n} (need >= 10), seeds={arguments.seeds} (need >= 10)", flush=True)

  # ---------------------------------------------------------------- bonus: BO vs random
  if arguments.skip_random_arm:
    with open(arguments.output, "w") as f:
      json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {arguments.output} (random arm skipped)", flush=True)
    return
  random_curves = proxy_runs(
    detector, n2, arguments.seeds, arguments.n_events, "random", arguments.kernel, m, seed_offset=arguments.seed_offset,
    log_lengthscale_bounds=arguments.log_lengthscale_bounds, n_threads=arguments.n_threads
  )
  bo_final = np.array([c["best"][-1] for c in curves])
  random_final = np.array([c["best"][-1] for c in random_curves])
  report["bonus_bo_vs_random"] = {
    "bo_median_final": float(np.median(bo_final)),
    "random_median_final": float(np.median(random_final)),
    "bo_better_on": int((bo_final < random_final).sum()),
    "seeds": arguments.seeds,
    # THE RANDOM ARM'S WHOLE CURVES, not just its endpoint. This arm already costs `seeds * 2n` design
    # scorings -- 30% of a cell -- and until now everything but the last value was thrown away.
    #
    # WHY THEY ARE NEEDED. `best-so-far` is a MIN-STATISTIC, so if BO degenerates to random sampling
    # near the optimum, improvements become rare events with P(improve in n) = 1 - (1-p)^n ~ np, and
    # an endpoint comparison between two arms is dominated by whether a hit happened at all -- very
    # noisy at 10 seeds, and hopeless at the 4 seeds an earlier campaign comparison had. The
    # IMPROVEMENT RATE along the trajectory uses every iteration instead of one endpoint, and it is
    # what distinguishes "BO is searching" from "BO is sampling": if BO's rate decays to the random
    # arm's late in the run, the arms are fighting over a tail and any endpoint difference between
    # them is luck.
    #
    # Paired by construction: both arms use the same seeds and the same event offsets per iteration.
    "random_curves": [[float(v) for v in curve["best"]] for curve in random_curves],
  }
  b = report["bonus_bo_vs_random"]
  print(
    f"  bonus: BO {b['bo_median_final']:.4f} vs random {b['random_median_final']:.4f}, "
    f"BO better on {b['bo_better_on']}/{arguments.seeds}", flush=True
  )

  # ---------------------------------------------------------------- cost of a neural campaign
  # The proxy is ~0.7 s a design; the neural objective was measured at 250-380 s. Report the
  # PROXY cost and the scaling assumption separately -- never present a projection as a measurement.
  # WITH NO LANDSCAPE (`--n-designs 0`) there is no measured per-design cost, and this block used to
  # index straight into `report["landscape"]` and raise KeyError -- AFTER every run had been scored.
  # The whole job's work was then lost to a report line, which is the "errors are not results" failure
  # in its most expensive form: two CPU-hours of finished curves thrown away because the summary could
  # not be printed. A cost this job did not measure is reported as None, not invented.
  seconds = report["landscape"].get(str(m), {}).get("seconds_per_design")
  report["cost"] = {
    "proxy_seconds_per_design":
    seconds,
    "note":
    "neural cost must be measured with a short bo.py run; the melt benchmark ran 250-380 s "
    "a design and 900 s for an uninformative one. 5 seeds x 2 arms x 20 designs / 2 shards "
    "= 100 designs per shard, so ~290 s a design fits 8 h.",
  }

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump(report, f, indent=2)
  print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
  main()
