#!/usr/bin/env python3
"""Does training on SCATTERED designs cost anything AT THE DESIGN YOU CARE ABOUT?

    python scripts/probe_scatter.py =enzyme_extremes --seeds 1 2 3 4 5 \
        --designs best median poor --sigma 0.1 --output output/scatter/scatter.json

THE QUESTION. The network is DESIGN-CONDITIONED: every row of the pool carries its own physical
design and `combine_scaled` merges each event with its own, so ONE network can be trained on rows
belonging to many different designs. The `meta` arm (`ContinualTrainer`) exploits that -- its batch is
half current-design rows and half REPLAY rows from earlier designs -- while convergence is judged on
the current design's window alone. If rows at scattered designs make the current design harder to fit,
that is a mechanism for `meta`'s per-design cost. If they do not, replay is free and the explanation
lies elsewhere. This script measures the scattered-design cost DIRECTLY, without replay, without a
second strategy, and without any change to the convergence procedure.

THE CONSTRUCTION. Five designs the campaign actually visited, spread over its loss range, all
resimulated FRESHLY (new event indices; nothing is reused from the campaign's pools). Each takes a
turn as the CURRENT design, and the other four are its HISTORY -- LEAVE-ONE-OUT, so every design is
current exactly once and historical four times, and every design in the study is one the campaign
really scored. Each historical design contributes `--history-rows` TRAIN rows, laid down before
training starts, so the current design's window opens at `w0_train > 0` exactly as a mid-campaign
design's does. The default 36864 models `meta`'s MEASURED pooled median per-design cost of 55976
calls (49150 at seed 1244111331, 62802 at seed 126382657), whose train share at this study's 2:1
split is 37317, rounded to 9 * `n_increment`.

THE TWO ARMS, paired on (current design, seed) and differing in ONE thing.

  A (perturbed) EVERY design perturbed -- the historical rows AND the current design's rows alike,
                one independent draw PER ROW, at every data request;
  B (precise)   nothing perturbed anywhere; every row simulated at its exact design.

It is all-or-nothing by instruction (user, 2026-08-16): there is no "history scattered, current design
exact" variant.

THE TEST POOL IS NEVER PERTURBED, IN EITHER ARM. It is held at the exact current design and is the
identical set of rows in every run, so the paired difference is two networks measured against one
fixed target rather than a difference between two targets. Both arms also use the same event indices,
the same pool sizes, the same network initialisation, the same growth schedule and the same stopping
rule, so the only difference within a pair is whether the data was scattered.

THE PERTURBATION, applied PER ROW at the moment the row is added to the pool:

  1. the base design in SCALED space, `u` in [0, 1]^n;
  2. `z = uniform_to_normal(u, 0, 1)` -- the inverse normal CDF, `erfinv(2u - 1) * sqrt(2)`;
  3. `z' = z + sigma * N(0, I_n)`, sigma = 0.05;
  4. `u' = normal_to_uniform(z', 0, 1)`.

`detopt/utils/encoding.py` owns both maps; nothing here reimplements them. The pair keeps every
perturbed design inside the box BY CONSTRUCTION, so nothing is clipped and nothing piles up on a
bound, and it perturbs the interior more than the edges. `uniform_to_normal` already clips its
rescaled input to `[eps - 1, 1 - eps]` in the array's own dtype before `erfinv`, so a coordinate at
exactly 0 or 1 maps to a finite +/- 5.2947 in float32 rather than to an infinity; no separate guard is
needed or written.

THE ORDER IS LOAD-BEARING: perturb, THEN simulate at the perturbed design, THEN store the event
TOGETHER WITH the design it was simulated at. Simulating at the base design and storing a perturbed
one would be a MISLABELLED dataset, and would measure the network's response to wrong labels rather
than to scattered designs.

THIS IS NOT AUGMENTATION. A row is perturbed exactly ONCE, when it is added. Growth appends new rows
with their own fresh draws; rows already in the pool are never re-perturbed and never re-simulated. The
dataset grows, but no row ever changes -- so the effective sample size is the same in both arms.

THE PROCEDURE IS THE CAMPAIGN'S OWN, unchanged. The loop below is `detopt/nn/trainer/design.py`'s
`train`, copied because this study needs a THIRD pool that the trainer does not have and a per-row
design that `Trainer._fill_pool` cannot express. Everything it does is that file's: warmup, the
bayesian gap/plateau tests in their fixed order, `n_increment` growth on (1) or (2.1), the `param_mix`
rewind toward the run's initial network with the optimiser reset, and the exit on
`|val - train| + hypot(sems) <= loss_precision`. The kernels are the trainer's own -- the scan-folded
jitted epoch (`Trainer._build_train_epoch`) and its eval passes -- never a python-driven step loop.
`design.py` is NOT modified and NOT subclassed.

WHAT IS REPORTED, primary first:

  * DETECTOR CALLS TO CONVERGENCE (train + validation), the growth-round count and the final window.
    This is the quantity `meta`'s per-design cost is denominated in: if scattered rows make a design
    harder to fit, the perturbed arm asks for more data before the stopping rule fires.
  * TEST LOSS at the unperturbed base design, plus the paired per-row difference between the arms.
    The two arms score on the SAME test rows, so their difference is a paired per-row statistic and
    its error is much smaller than either arm's own SEM.
  * epochs, the train/validation gap at stopping, and whether the run converged or hit the cap.

POOL SIZES. train : validation : test = 2 : 1 : 1, i.e. `val_fraction = 1/3`. Train and validation
GROW together under the procedure; the test pool is filled ONCE, before training, at a fixed size
(`--n-test`, default 32768) which sits inside the range a final validation window reaches on this task
(the campaign's per-design spend of 55000-95000 calls at its own 3:1 split is a 14000-24000-row
validation window; at 2:1 it would be 18000-32000). Holding it fixed and identical across every run is
what makes the test comparison exact row by row.

`iteration_limit` STAYS AT THE CONFIG'S 524288 because it is also the epoch length
(`steps_per_epoch = iteration_limit // batch`); moving it would silently retune warmup, patience and
the convergence posterior, which are all written in epoch units. The window is instead bounded by
`training.budget`, which sizes the pools.

GROWTH CURVES. `--plots-dir` renders `detopt.utils.viz.bo.plot_iteration` -- the figure `scripts/bo.py`
draws -- one per run under `<plots-dir>/<design>/s<seed>/<arm>/`, so a perturbed run and its control sit
in sibling directories. Rendered on a DETACHED DAEMON THREAD, every 16th epoch and once at the end,
NOT through `design.py`'s `ThreadPoolExecutor`: that is used as a context manager, so its `__exit__`
joins the queue and the design blocks at the end until every figure has rendered (~36% of a design's
wall clock elsewhere). A detached thread has no tail; the cost is that a figure still rendering at
process exit may be lost, which is why the end of every design is rendered too. Growth rounds are tens
of epochs apart, so every 16th epoch still shows where each addition falls.

THIS SCRIPT SETS NOTHING. It writes a JSON of measurements plus an NPZ of per-row test losses.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import threading
import time

import matplotlib

matplotlib.use("AGG")

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

import detopt
import detopt.detector
import detopt.utils.config
from detopt.nn import from_config as regressor_from_config
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.nn.trainer.common import fresh_design_network, regressor_rngs
from detopt.utils.encoding import normal_to_uniform, uniform_to_normal
from detopt.utils.events import shuffled_event_index
from detopt.utils.pools import Pool
from detopt.utils.training import (bayesian_trend, masked_mean_sem, probability_above, probability_change_below)
from detopt.utils.viz.bo import plot_iteration

CAMPAIGN = "output/enzyme_extremes"


def campaign_rows(root=CAMPAIGN):
  """Every scored design of the finished campaign, as `(loss, x_scaled, provenance)` ascending by loss.

  Both seeds and both arms are pooled: a base design is a POINT OF THE LANDSCAPE here, not a claim
  about which arm found it, and pooling is what makes the ranks below span the range the campaign
  actually visited.
  """
  rows = []
  for seed in sorted(os.listdir(root)):
    for arm in ("meta", "from_scratch", "continue", "closest"):
      path = os.path.join(root, seed, arm, "results.json")
      if not os.path.isfile(path):
        continue
      with open(path) as f:
        recorded = json.load(f)["results"]
      for entry in recorded:
        if "x_scaled" not in entry or "loss" not in entry:
          continue
        rows.append(
          (float(entry["loss"]), np.asarray(entry["x_scaled"], np.float32), f"{seed}/{arm}#{int(entry['iteration'])}")
        )
  rows.sort(key=lambda r: r[0])
  return rows


def design_set(rows, ceiling=0.95):
  """The FIVE base designs to study, keyed by name, from the campaign's own scored designs.

  A SPREAD is required because an effect may exist only where the landscape is informative, so the set
  is fixed order statistics of the campaign's own scored losses: best, 25th, 50th, 75th, 85th.

  THE CEILING DESIGNS ARE EXCLUDED ON PURPOSE, and this is the "state what the probe must separate"
  clause applied to the design choice. The task is binary, scored by cross-entropy over ln 2, so an
  uninformative design leaves train and validation pinned at the no-information value 1.0: nothing the
  perturbation does can move a number that is already at its ceiling, in either arm. A pair run there
  would return "no difference" for a reason that has nothing to do with the question. `ceiling` drops
  them.

  `poor` is the 85th percentile of what remains rather than the very worst, for the same reason in
  weaker form: the worst informative design of this campaign sits at 0.9348, i.e. 0.065 of headroom
  below the ceiling, which is thin ground on which to resolve anything.
  """
  informative = [r for r in rows if r[0] < ceiling]
  if len(informative) < 5:
    raise SystemExit(f"probe_scatter: only {len(informative)} campaign designs below the {ceiling} ceiling")
  n = len(informative)
  return {
    "best": informative[0],
    "p25": informative[int(0.25 * n)],
    "median": informative[n // 2],
    "p75": informative[int(0.75 * n)],
    "poor": informative[int(0.85 * n)],
  }


def perturb(base_scaled, n, sigma, rng):
  """`n` independent perturbed copies of `base_scaled`, in SCALED space, one draw per row.

  `sigma = 0` returns the base design EXACTLY (broadcast), rather than round-tripping it through
  `erfinv`/`erf` for a no-op that would differ from it in the last float32 bit -- the control arm must
  be the campaign's own design, not a numerically jittered copy of it.
  """
  base_scaled = np.asarray(base_scaled, np.float32)
  if sigma <= 0.0:
    return np.broadcast_to(base_scaled[None, :], (n, base_scaled.size)).copy()
  z = uniform_to_normal(base_scaled, 0.0, 1.0)
  noise = rng.standard_normal((n, base_scaled.size)).astype(np.float32)
  return np.asarray(normal_to_uniform((z[None, :] + sigma * noise).astype(np.float32), 0.0, 1.0), np.float32)


def fill(detector, pool, index_array, designs_scaled, chunk=1024):
  """Append `len(designs_scaled)` rows, each SIMULATED AT ITS OWN design and STORED WITH IT.

  The event indices are the next slice of `index_array` taken at the pool's CURRENT fill, exactly as
  `Trainer._fill_pool` does, so the detector call stays deterministic in `(design, event_index)` and
  the two arms of a pair consume the identical events. The design handed to the detector and the
  design written into the pool are the SAME array: the physics and the stored conditioning cannot
  disagree.
  """
  n = int(designs_scaled.shape[0])
  added = 0
  while added < n:
    k = min(chunk, n - added)
    start = pool.current
    event_index = index_array[start:start + k]
    record = detector.to_nominal(designs_scaled[added:added + k])
    physical = detector.flatten_design(record)
    _ground_truth, event, mask, target = detector(physical, event_index)
    pool.append(event, mask, target, record)
    added += k


def make_trainer(detector, run, seed, sampler):
  """The campaign's trainer, built from the run config -- its pools, its kernels, its optimiser.

  Used AS IS: this script drives the loop itself (it needs a third pool and a per-row design), but
  every kernel it steps is the trainer's own, so the epoch is the scan-folded jitted one and the
  evaluation passes are the ones the campaign reports from.

  HOW THE HISTORY ENTERS TRAINING is decided HERE, by which trainer is built, because the choice IS
  the trainer's `_sample_indices` and the kernels are compiled around it:

  * `window` -- `DesignTrainer`, i.e. `window_sample_indices`: minibatches are drawn uniformly over
    the CURRENT design's window `[start, start+count)` alone. Historical rows sit in the pool and are
    never sampled, so this is the no-replay reference.
  * `replay` -- `ContinualTrainer`, i.e. half of each member's batch from the current window and half
    from `[0, start)`, the accumulated history, with `_sample_weights` normalised to mean 1. This is
    `meta`'s batch geometry exactly, and the historical fraction is fixed at one half however large
    the pool grows.

  WHAT IS AND IS NOT TAKEN FROM `ContinualTrainer`: its SAMPLING, and nothing else. The network is
  built by `fresh_design_network` in `measure`, NOT by the continual trainer's persistent-net hook, so
  both arms of a pair start from byte-identical parameters and the only difference between them is
  whether the data was scattered. `meta`'s network persistence is therefore deliberately absent, and
  the result speaks to the batch geometry rather than to `meta` entire.
  """
  factory = ContinualTrainer if sampler == "replay" else DesignTrainer
  return factory.from_config(detector, run, checkpoint_dir=None, seed=seed)


def build_test_eval(trainer, detector, window):
  """An eval pass over exactly `window` test rows, built from the trainer's own `_build_eval`.

  Sized to the test pool rather than reusing `_eval_val`, whose window is the validation cap: reusing
  it would scan 262144 clipped indices to read 32768 real ones.
  """
  regressor = regressor_from_config(detector, config=trainer.regressor_config, rngs=regressor_rngs(trainer.seed))
  definition = nnx.split(regressor, nnx.Param, nnx.Variable)[0]
  return trainer._build_eval(definition, int(window))


def measure(trainer, detector, test_pool, eval_test, n_test, base_scaled, sigma, seed, plots_dir, history=(), history_rows=0):
  """ONE run: lay down the HISTORY, then grow, train and stop under the campaign's procedure on the
  CURRENT design, and score on a test pool held at that design's PRECISE value.

  The loop is `detopt/nn/trainer/design.py::_DesignBase.train`, clause for clause and in its order.
  What differs, and only this: the data is drawn at PER-ROW designs (`sigma > 0`) or at the exact
  design (`sigma = 0`); the train pool is pre-loaded with `history_rows` rows at each design in
  `history`, so the current design's window opens at `w0_train > 0` exactly as a mid-campaign design's
  does; and a third, non-growing pool at the PRECISE current design is evaluated alongside the two the
  procedure reads.

  EVERY DESIGN IS PERTURBED OR NONE IS (user, 2026-08-16). In the perturbed arm the historical rows AND
  the current design's rows are both drawn per row; in the precise arm neither is. There is no
  "history scattered, current exact" variant.

  THE TEST POOL IS NEVER PERTURBED, IN EITHER ARM. It is filled by the caller at the exact current
  design and is the identical set of rows in both arms, so the paired difference is a difference of
  two networks on one fixed target rather than a difference of targets.

  THE HISTORY GOES INTO THE TRAIN POOL ONLY, and `w0_val` stays 0. In `meta` the validation pool
  accumulates historical rows too, but NOTHING ever reads them -- `_eval_val` evaluates from `w0_val`
  forward (the current design's window alone) and the replay draw in
  `ContinualTrainer._sample_indices` reads the TRAIN pool's `[0, start)`. Simulating them would spend
  a third of the history's budget on rows no measured quantity depends on, so they are skipped and the
  saving is stated rather than hidden.
  """
  train_pool, val_pool = trainer.train_pool, trainer.val_pool
  train_pool.current, val_pool.current = 0, 0
  val_ratio = trainer._val_ratio

  design_rng = np.random.default_rng([int(seed), int(round(1e6 * float(sigma)))])
  init_seq, training_seq = np.random.SeedSequence(int(seed)).spawn(2)
  params, state, opt_state = fresh_design_network(trainer, init_seq, None)
  initial_params = params
  key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))

  for historical in history:
    fill(detector, train_pool, trainer._train_index, perturb(historical, int(history_rows), sigma, design_rng))
  w0_train, w0_val = train_pool.current, 0
  w0_train_j, w0_val_j = jnp.int32(w0_train), jnp.int32(0)
  history_calls = int(w0_train)

  def sample_round(n_requested):
    """One growth round: perturb -> simulate on the perturbed designs -> store both. Returns the
    number of TRAIN rows added, 0 if this design's window is full, or None if a pool is full."""
    n_train = min(int(n_requested), trainer.iteration_limit - (train_pool.current - w0_train))
    if n_train <= 0:
      return 0
    n_val = round(n_train * val_ratio)
    n_val = max(0, min(n_val, trainer.val_iteration_limit - (val_pool.current - w0_val)))
    if n_train > train_pool.capacity - train_pool.current or n_val > val_pool.capacity - val_pool.current:
      return None
    fill(detector, train_pool, trainer._train_index, perturb(base_scaled, n_train, sigma, design_rng))
    if n_val > 0:
      fill(detector, val_pool, trainer._val_index, perturb(base_scaled, n_val, sigma, design_rng))
    return n_train

  train_history, val_history, test_history = [], [], []
  train_sem_history, val_sem_history, test_sem_history = [], [], []
  pool_history = []
  started = time.time()
  design_list = np.asarray(base_scaled).tolist()

  def render(train_count, val_count, val_mean):
    """Queue one growth-curve figure on a DETACHED daemon thread -- started, never joined.

    NOT `design.py`'s `ThreadPoolExecutor` context manager: that is a deferred join (`__exit__` calls
    `shutdown(wait=True)`), so with one worker the queue grows all through the design and the design
    blocks at the end until every figure has rendered -- measured elsewhere at ~36% of a design's wall
    clock. A detached thread has no tail at all. The snapshot helper builds FRESH arrays, so the thread
    never touches a buffer this loop goes on to mutate, and `plot_iteration` uses matplotlib's OO
    `Figure` API (never pyplot, whose global state is not thread-safe).

    The accepted trade: nothing joins these threads, so a figure still rendering at process exit may be
    lost. That is why the END of every design is rendered as well as every 16th epoch -- the final
    state is the one most likely to survive, and a missing intermediate figure is not a failure.
    """
    snapshot = trainer._snapshot(
      train_history, val_history, train_sem_history, val_sem_history, pool_history, train_count, val_count
    )
    threading.Thread(target=plot_iteration, args=(snapshot, 0, design_list, val_mean, plots_dir), daemon=True).start()

  if sample_round(trainer.n0) is None:
    return None

  round_start, epoch_in_round = 0, 0
  objective, status = None, "converged"
  test_per_row = np.zeros(n_test, np.float32)
  train_mean, val_mean, diff, err = float("nan"), float("nan"), float("nan"), float("nan")

  while True:
    train_count = train_pool.current - w0_train
    val_count = val_pool.current - w0_val
    key, subkey = jax.random.split(key)
    params, state, opt_state, _ = trainer._train_epoch(
      params, state, opt_state, subkey, w0_train_j, jnp.int32(train_count), train_pool.buffers()
    )

    train_eval = trainer._eval_train(params, state, train_pool.buffers(), w0_train_j)
    train_mean, train_sem = masked_mean_sem(train_eval, train_count)
    val_eval = trainer._eval_val(params, state, val_pool.buffers(), w0_val_j)
    val_mean, val_sem = masked_mean_sem(val_eval, val_count)
    test_eval = eval_test(params, state, test_pool.buffers(), jnp.int32(0))
    test_mean, test_sem = masked_mean_sem(test_eval, n_test)
    train_mean, train_sem = float(train_mean), float(train_sem)
    val_mean, val_sem = float(val_mean), float(val_sem)
    test_mean, test_sem = float(test_mean), float(test_sem)

    train_history.append(train_mean)
    val_history.append(val_mean)
    test_history.append(test_mean)
    train_sem_history.append(train_sem)
    val_sem_history.append(val_sem)
    test_sem_history.append(test_sem)
    pool_history.append(train_count)
    epoch_in_round += 1

    if plots_dir is not None and len(train_history) % 16 == 0:
      render(train_count, val_count, val_mean)

    gap_signed = val_mean - train_mean
    err = float(np.hypot(train_sem, val_sem))
    diff = abs(gap_signed)

    if epoch_in_round <= trainer.warmup_epochs:
      continue

    first = round_start + trainer.warmup_epochs
    tr = np.asarray(train_history[first:], dtype=np.float64)
    va = np.asarray(val_history[first:], dtype=np.float64)
    tr_s = np.asarray(train_sem_history[first:], dtype=np.float64)
    va_s = np.asarray(val_sem_history[first:], dtype=np.float64)
    if tr.shape[0] < 3:
      continue
    prior_sigma = max(float(tr[0]), float(va[0])) / 3.0
    gap_sem = np.hypot(tr_s, va_s)
    gap_series = np.abs(va - tr) + gap_sem
    tr_mean, tr_cov = bayesian_trend(tr, tr_s, prior_sigma)
    gap_mean, gap_cov = bayesian_trend(gap_series, gap_sem, prior_sigma)

    p_gap_exceeds = probability_above(gap_mean, gap_cov, trainer.patience, trainer.loss_precision, gap_series.shape[0])
    if p_gap_exceeds > 0.9:
      pass
    else:
      p_train_settled = probability_change_below(tr_mean, tr_cov, trainer.patience, 0.5 * trainer.loss_precision)
      if p_train_settled > 0.9:
        if diff + err > trainer.loss_precision:
          pass
        else:
          objective = (val_mean, diff + err)
          test_per_row = np.asarray(test_eval, np.float32)[:n_test]
          print(
            f"  [converged/bayes] train={train_mean:.4f} val={val_mean:.4f} test={test_mean:.4f} "
            f"diff={diff:.4f} err={err:.4f} | window={train_count}", flush=True
          )
          break
      else:
        continue
    n_added = sample_round(trainer.n_increment)
    if n_added is None:
      status = "pool exhausted"
      test_per_row = np.asarray(test_eval, np.float32)[:n_test]
      break
    if n_added == 0:
      status = "capped"
      test_per_row = np.asarray(test_eval, np.float32)[:n_test]
      break
    if trainer.param_mix > 0.0:
      mix = trainer.param_mix
      params = jax.tree.map(lambda p, q: q + (1.0 - mix) * (p - q), params, initial_params)
      opt_state = trainer.optimizer.init(params)
    round_start = len(train_history)
    epoch_in_round = 0
    print(f"  [grow] window -> {train_pool.current - w0_train}, pool {train_pool.current}/{train_pool.capacity}", flush=True)

  wall = time.time() - started
  train_count = train_pool.current - w0_train
  val_count = val_pool.current - w0_val
  if plots_dir is not None:
    render(train_count, val_count, val_history[-1])
  n_rounds = int(1 + max(0, (train_count - trainer.n0 + trainer.n_increment - 1) // trainer.n_increment))
  row = {
    "status": status,
    "wall_s": wall,
    "sigma": float(sigma),
    "seed": int(seed),
    "calls_train_val": int(train_count + val_count),
    "history_calls": history_calls,
    "calls_total": int(train_count + val_count + n_test + history_calls),
    "window": int(train_count),
    "val_window": int(val_count),
    "n_rounds": n_rounds,
    "n_epochs": len(train_history),
    "train": train_history[-1],
    "val": val_history[-1],
    "test": test_history[-1],
    "test_sem": test_sem_history[-1],
    "diff": diff,
    "err": err,
    "slack": diff + err,
    "objective": float("nan") if objective is None else float(objective[0]),
    "objective_std": float("nan") if objective is None else float(objective[1]),
    "per_epoch": {
      "train": [round(float(v), 6) for v in train_history],
      "val": [round(float(v), 6) for v in val_history],
      "test": [round(float(v), 6) for v in test_history],
      "train_sem": [round(float(v), 6) for v in train_sem_history],
      "val_sem": [round(float(v), 6) for v in val_sem_history],
      "window": [int(v) for v in pool_history],
    },
  }
  return row, test_per_row


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a RUN config, e.g. =enzyme_extremes")
  parser.add_argument("--designs", nargs="*", default=["best", "p25", "median", "p75", "poor"], help="current designs by name")
  parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="repetitions; the arms are PAIRED on these")
  parser.add_argument(
    "--sigma", type=float, nargs="+", default=[0.05],
    help="perturbed-arm scales in the normal space; the precise arm (0.0) is always run"
  )
  parser.add_argument("--n-test", type=int, default=32768, help="test rows at the PRECISE current design, filled once")
  parser.add_argument(
    "--history-rows", type=int, default=36864,
    help="TRAIN rows contributed by EACH historical design (leave-one-out: every other design in the "
    "set). 0 disables the history entirely and reproduces the sigma=0.1 pilot. The default models "
    "`meta`'s measured pooled median per-design cost of 55976 calls, whose train share at this "
    "study's 2:1 split is 37317, rounded to 36864 = 9 * n_increment"
  )
  parser.add_argument(
    "--sampler", choices=("window", "replay"), default="replay",
    help="how the history enters training: `window` samples the current design's window alone "
    "(`DesignTrainer`, history never sampled); `replay` takes half of every batch from the "
    "accumulated history (`ContinualTrainer`, `meta`'s own batch geometry)"
  )
  parser.add_argument(
    "--budget", type=int, default=1 << 20, help="`training.budget`, which sizes the pools and so caps the window"
  )
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument("--n0", type=int, default=None, help="override `training.n0` (SMOKE TEST ONLY)")
  parser.add_argument("--n-increment", type=int, default=None, help="override `training.n_increment` (SMOKE TEST ONLY)")
  parser.add_argument("--iteration-limit", type=int, default=None, help="override `training.iteration_limit` (SMOKE TEST ONLY)")
  parser.add_argument("--plots-dir", default="output/scatter/plots", help="growth curves; '' turns the callback off")
  parser.add_argument("--output", default="output/scatter/scatter.json")
  parser.add_argument("--resume", action="store_true", help="keep cells already in --output and skip them")
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

  run = json.loads(json.dumps(config))
  run["training"]["val_fraction"] = 1.0 / 3.0
  run["training"]["budget"] = int(arguments.budget)
  if arguments.n0 is not None:
    run["training"]["n0"] = int(arguments.n0)
  if arguments.n_increment is not None:
    run["training"]["n_increment"] = int(arguments.n_increment)
  if arguments.iteration_limit is not None:
    run["training"]["iteration_limit"] = int(arguments.iteration_limit)
  if arguments.device is not None:
    run["device"] = arguments.device
  run["plot_per_epoch"] = False

  designs = design_set(campaign_rows())
  missing = [d for d in arguments.designs if d not in designs]
  if len(missing) > 0:
    raise SystemExit(f"probe_scatter: unknown design(s) {missing}; have {sorted(designs)}")

  rows, per_row = [], {}
  npz_path = os.path.splitext(arguments.output)[0] + "_test_rows.npz"
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)["rows"]
    if os.path.isfile(npz_path):
      with np.load(npz_path) as data:
        per_row = {k: np.asarray(data[k]) for k in data.files}
  done = {(r["design"], r["seed"], r["sigma"]) for r in rows}
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)

  print(f"design set: {[(k, round(designs[k][0], 4), designs[k][2]) for k in arguments.designs]}", flush=True)
  print(
    f"seeds {arguments.seeds} | sigmas {[0.0] + list(arguments.sigma)} | n_test {arguments.n_test} "
    f"| sampler {arguments.sampler} | history {arguments.history_rows} rows per historical design", flush=True
  )

  for seed in arguments.seeds:
    trainer = make_trainer(detector, run, seed, arguments.sampler)
    eval_test = build_test_eval(trainer, detector, arguments.n_test)
    test_index = shuffled_event_index(detector.size(), arguments.n_test, int(seed) + 987654321, name="test events")
    test_pool = Pool(arguments.n_test, trainer.train_pool.specs, trainer.device)
    print(
      f"seed {seed}: pools train {trainer.train_pool.capacity} val {trainer.val_pool.capacity} "
      f"test {test_pool.capacity} | iteration_limit {trainer.iteration_limit} "
      f"val cap {trainer.val_iteration_limit} | steps/epoch {trainer.steps_per_epoch}", flush=True
    )
    for design_name in arguments.designs:
      base_loss, base_scaled, provenance = designs[design_name]
      test_pool.current = 0
      fill(detector, test_pool, test_index, perturb(base_scaled, arguments.n_test, 0.0, None))
      for sigma in [0.0] + list(arguments.sigma):
        if (design_name, int(seed), float(sigma)) in done:
          print(f"skip {design_name} s{seed} sigma={sigma} (already measured)", flush=True)
          continue
        arm = "control" if sigma == 0.0 else f"scatter{sigma:g}"
        plots_dir = None
        if len(arguments.plots_dir) > 0:
          plots_dir = os.path.join(arguments.plots_dir, design_name, f"s{seed}", arm)
        history = [designs[k][1] for k in arguments.designs if k != design_name] if arguments.history_rows > 0 else []
        print(
          f"=== {design_name} (campaign loss {base_loss:.4f}) seed {seed} arm {arm} "
          f"| history {len(history)} designs x {arguments.history_rows} rows", flush=True
        )
        outcome = measure(
          trainer, detector, test_pool, eval_test, arguments.n_test, base_scaled, float(sigma), int(seed), plots_dir,
          history=history, history_rows=arguments.history_rows
        )
        if outcome is None:
          print("  pool exhausted before the first round -- not a measurement", flush=True)
          continue
        row, test_losses = outcome
        row.update({
          "design": design_name,
          "arm": arm,
          "provenance": provenance,
          "campaign_loss": base_loss,
          "x_scaled": [float(v) for v in np.asarray(base_scaled)],
        })
        row.update({"sampler": arguments.sampler, "history_rows": int(arguments.history_rows), "n_history": len(history)})
        rows.append(row)
        per_row[f"{design_name}|{seed}|{sigma:g}"] = test_losses
        with open(arguments.output, "w") as f:
          json.dump({"rows": rows, "n_test": int(arguments.n_test), "budget": int(arguments.budget)}, f, indent=1)
        np.savez_compressed(npz_path, **per_row)
        print(
          f"  -> calls {row['calls_train_val']} rounds {row['n_rounds']} epochs {row['n_epochs']} "
          f"window {row['window']} test {row['test']:.4f} val {row['val']:.4f} slack {row['slack']:.4f} "
          f"[{row['status']}] {row['wall_s']:.0f} s (history {row['history_calls']})", flush=True
        )
    for buffer in jax.tree.leaves(test_pool.buffers()):
      buffer.delete()
    for pool in (trainer.train_pool, trainer.val_pool):
      for buffer in jax.tree.leaves(pool.buffers()):
        buffer.delete()
    del trainer, eval_test, test_pool
    gc.collect()


if __name__ == "__main__":
  main()
