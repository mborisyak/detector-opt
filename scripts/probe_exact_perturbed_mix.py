#!/usr/bin/env python3
"""Does `meta` benefit from training on a MIXTURE of exact and perturbed designs?

    python scripts/probe_exact_perturbed_mix.py =enzyme_extremes --arm dual --seeds 1 \
        --output output/mixprobe/dual_s1.json

THE QUESTION. `meta` (`ContinualTrainer`) trains ONE persistent network on rows that each carry their
own design: half of every minibatch is the current design's window, half is replay from the designs
already visited. Every one of those rows sits at a design the campaign actually asked for. This probe
asks whether REPLACING HALF of that data -- half the replay and half the current window -- with rows
simulated at PERTURBED designs makes the current design cheaper or dearer to fit. A parallel study
found that `meta`'s cost is gated by the train/validation GAP, and a prior scattering study found that
perturbing EVERY row at sigma = 0.1 cost +36% to +44% more detector calls (and +4 to +6 growth rounds)
across three designs with the loss at the base design essentially unmoved -- the cost appeared in the
gap. If a HALF dose at HALF the sigma regularises instead, the mixture converges cheaper; if it is
simply dilution, it converges dearer. Neither answer is favoured here.

THE TWO ARMS, paired on (current design, seed), at the SAME TOTAL VOLUME.

  control  Standard `meta`. ONE exact pool: `--history-designs` past designs of exact history, then the
           current design's window, exact. The batch is 1:1 current : historical, i.e.
           `ContinualTrainer._sample_indices` EXACTLY AS SHIPPED (128 + 128 of a 256 batch per ensemble
           member). This arm is the shipped code path, not a re-implementation of it.
  dual     TWO pools, 1:1, each split into a historical and a current part -- FOUR components:
               exact-current | exact-historical | perturbed-current | perturbed-historical
           The batch takes ONE QUARTER (64 per ensemble member) from each, so current : historical is
           still 1:1 and only the COMPOSITION inside each half changes. Every count is halved, never
           doubled: the same number of historical rows, the same window at every growth round, the same
           validation rows. The two arms differ in composition and in nothing else.

SCORING: ONLY THE CURRENT ROWS ARE SCORED -- current-exact and current-perturbed. Both historical parts
are training-only, exactly as `meta`'s replay half is training-only today (`_eval_train` reads from
`w0_train` forward, so replay rows contribute to the update and to nothing that is measured).

⚠️ THE ONE CHOICE MADE FOR THIS PROBE, STATED SO IT CAN BE CORRECTED (user, 2026-08-16). The
convergence criterion `diff + err <= loss_precision`, with `diff = |val - train|`, is computed over the
current rows TAKEN TOGETHER -- exact and perturbed as ONE scored set, each with its own train/validation
split contributing. So in the `dual` arm the number the procedure gates on, and the number it would hand
BO, is a loss over a NEIGHBOURHOOD of the design rather than over the design itself. The alternative
reading is the criterion on the exact rows alone. To keep that recoverable WITHOUT A RE-RUN, this script
records the exact-only and the perturbed-only train/validation means and SEMs AT EVERY EPOCH beside the
combined ones, so the exact-only criterion can be replayed after the fact over the epochs that were
actually run.

THE PERTURBATION, applied PER ROW at the moment the row is added, `--sigma` default 0.05:

  1. the base design in SCALED space, `u` in [0, 1]^16;
  2. `z = uniform_to_normal(u, 0, 1)`, i.e. `erfinv(2u - 1) * sqrt(2)`;
  3. `z' = z + sigma * N(0, I)`;
  4. `u' = normal_to_uniform(z', 0, 1)`.

`detopt/utils/encoding.py` owns both maps and `probe_scatter.perturb` is the one implementation; nothing
here reimplements either. The pair keeps every perturbed design inside the box BY CONSTRUCTION, so
nothing is clipped and nothing piles up on a bound.

THE ORDER IS LOAD-BEARING: perturb, THEN simulate at the perturbed design, THEN store the event TOGETHER
WITH the design it was simulated at. THIS IS NOT AUGMENTATION: a row is perturbed exactly ONCE, when it
is added; growth appends new rows with their own fresh draws and never redraws an old one.

THE HELD-OUT TEST POOL is filled once per (design, seed) at the EXACT current design, is never perturbed
in either arm, and is the identical set of rows in both arms -- so the two arms are scored against ONE
common target and the test difference is paired row by row.

WHAT IS SHARED WITHIN A PAIR, and it is nearly everything: the current design, the seed, the ten past
designs, the network initialisation, the growth schedule, the stopping rule, the test rows, the total
historical volume, the total window at every round, and THE EVENT INDICES. The last is exact, not
approximate: the control's historical block `i` consumes `_train_index[i*B : (i+1)*B]`, and the dual arm
consumes the FIRST HALF of that same slice for its exact history and the SECOND HALF for its perturbed
history; the control's current window consumes `_train_index[H : H+w]`, and the dual arm takes the EVEN
offsets of that slice for its exact half and the ODD offsets for its perturbed half. After the same
number of growth rounds the two arms have consumed the identical multiset of events, so no part of the
paired difference is a difference of which events were drawn.

`training.budget` IS FIXED AT 2097152 IN EVERY CELL AND IS NOT A FREE PARAMETER. It SELECTS THE
VALIDATION SET: `common.py` draws one index of length `budget` and splits it at `train_budget`, so
different budgets give DISJOINT validation sets -- measured, one design cost 49150 calls at budget
2097152 and 595250 at 786432, a 12x swing from the draw alone. It is common-mode within a budget (both
arms of a cell share it, so the arm RATIO is robust) and decisive across budgets, so cells at different
budgets are never comparable and none is run.

`iteration_limit` STAYS AT THE CONFIG'S 524288 because it is also the epoch length
(`steps_per_epoch = iteration_limit // batch`); moving it would silently retune warmup, patience and the
convergence posterior, which are all written in epoch units. In the `dual` arm the current window is
capped at 524288 rows IN TOTAL, 262144 per side, so the cap is the same quantity in both arms.

THE HISTORY VOLUME. `meta`'s MEASURED per-design cost, campaign medians, is 49150 (seed 1244111331) and
62802 (seed 126382657), pooled median 55976 detector calls. At the campaign's own 3:1 train:validation
split that is 41982 train rows, rounded DOWN to 40960 = 10 * `n_increment` so a historical block is a
whole number of growth quanta. Ten past designs is therefore 409600 train rows standing for 546133
charged calls -- a mid-campaign `meta` history. THE HISTORICAL VALIDATION ROWS ARE NOT SIMULATED,
because nothing reads them: `_eval_val` evaluates from `w0_val` forward (the current design's window
alone) and every replay draw reads the TRAIN pool. The saving is stated rather than hidden.

THE PAST DESIGNS are drawn from the campaign's OWN scored designs -- `meta`'s history is a trajectory,
not a space-filling sample -- excluding the uninformative ceiling and excluding the five designs that
take a turn as the current one. They are drawn once per seed from the study seed alone, so both arms of
a pair get the identical ten, and the history is laid down ONCE per seed and reused by all five current
designs.

THE CURRENT DESIGNS are the campaign's own scored designs at fixed order statistics of its loss range
(best / p25 / median / p75 / poor, `probe_scatter.design_set`), excluding the uninformative ceiling: a
design pinned at the no-information value 1.0 cannot move in either arm and would return "no difference"
for a reason that has nothing to do with the question.

WHAT THE PROBE MUST SEPARATE, stated before it was run. The primary quantity is DETECTOR CALLS TO
CONVERGENCE at the current design, and it moves in QUANTA of one growth round -- 4096 train + 1365
validation = 5461 calls. `meta` converges near 55976 calls, so one quantum is ~10% of the level and NO
single pair can resolve anything finer than that. Across pairs the paired log-ratio has an sd of about
0.03 where an effect is large (measured over the three pairs of the sigma = 0.1 scattering study) and,
pessimistically, about 0.10 near zero, where the quantisation alone can move a pair by one round. With
5 designs x 4 seeds = 20 pairs the standard error of the mean paired log-ratio is then 0.007 to 0.022,
i.e. a resolvable effect of 1.4% to 4.5% at 2 sigma and 2.1% to 6.7% at 3 sigma. THE SCALE THAT MATTERS:
perturbing EVERY row at sigma = 0.1 cost +36% to +44%; this probe perturbs HALF the rows at HALF the
sigma, so if the cost follows the injected perturbation variance the expectation is 0.5 * (0.05/0.1)^2 =
1/8 of that, about +5%. That sits AT the pessimistic 2-sigma bar, which is why 20 pairs are run rather
than the 15 a three-seed design would give, and why a null result here is reported as "not resolved
below X%" rather than as an absence. The secondary quantities are the held-out test loss at the exact
design, paired over 32768 identical rows where the per-row SEM is ~0.003 against a `loss_precision` of
0.01, and the exact-vs-perturbed current losses, which say whether the perturbed half is a different
target at all.

THE PROCEDURE IS THE CAMPAIGN'S OWN, unchanged. The loop in :func:`measure` is
`detopt/nn/trainer/design.py::_DesignBase.train`, clause for clause and in its order -- warmup, the
bayesian gap/plateau tests, `n_increment` growth on (1) or (2.1), the `param_mix` rewind with the
optimiser reset, and the exit on `|val - train| + hypot(sems) <= loss_precision`. Every kernel it steps
is the trainer's own. `design.py` is NOT modified and NOT subclassed by this file's arm classes beyond
the abstract hooks. The loop is copied rather than called because this study needs a held-out pool the
trainer does not have, a history built ONCE per seed, and a per-row design that `Trainer._fill_pool`
cannot express. `scripts/probe_scatter.py` supplies the pieces that ARE reusable -- `campaign_rows`,
`design_set`, `perturb`, `fill`, `build_test_eval` -- and is left UNEDITED because queued jobs run it.

BOTH ARMS RUN THE SAME LOOP. The only things that differ are the trainer's `_sample_indices` and the
REGION TABLE: the control has one train region and one validation region, the dual arm has two of each.
The combined mean and SEM over the regions are pooled exactly (sum of squares about the pooled mean),
so at one region they reduce to `masked_mean_sem` itself and the control arm's numbers are the shipped
ones.

THIS SCRIPT SETS NOTHING. It writes a JSON of measurements plus an NPZ of per-row held-out losses.
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

import detopt
import detopt.detector
from detopt.nn.trainer import ContinualTrainer
from detopt.nn.trainer.common import _round_down, fresh_design_network
from detopt.nn.trainer.design import _DesignBase
from detopt.utils.events import shuffled_event_index
from detopt.utils.pools import Pool
from detopt.utils.training import (bayesian_trend, masked_mean_sem, probability_above, probability_change_below)
from detopt.utils.viz.bo import plot_iteration

from probe_scatter import build_test_eval, campaign_rows, design_set, fill, perturb

MEDIAN_PER_DESIGN_CALLS = 55976


class QuarterMixTrainer(_DesignBase):
  """`meta`'s replay over FOUR components instead of two.

  The batch is split into four EQUAL parts per ensemble member and concatenated in the trainer's
  `(members, batch)` layout, so the loss's reshape still recovers one minibatch per member:

      current-exact | current-perturbed | historical-exact | historical-perturbed

  `start` and `count` are the CURRENT-EXACT region's offset and filled length, supplied dynamically by
  the loop; the current-perturbed region sits `current_offset` rows further on and always holds the same
  number of filled rows, so one `count` addresses both. The two historical ranges are PYTHON INTS baked
  into the kernel at construction -- the history layout is fixed for a whole job, and a value read at
  trace time must not be allowed to change under a cached kernel.

  Current : historical stays 1:1, exactly as in `ContinualTrainer`; what changes is the composition
  inside each half. This IMPLEMENTS the abstract hooks and overrides no concrete method.
  """

  def __init__(self, *args, history_exact=(0, 0), history_perturbed=(0, 0), current_offset=0, **kwargs):
    self.history_exact = (int(history_exact[0]), int(history_exact[1]))
    self.history_perturbed = (int(history_perturbed[0]), int(history_perturbed[1]))
    self.current_offset = int(current_offset)
    if self.history_exact[1] <= self.history_exact[0] or self.history_perturbed[1] <= self.history_perturbed[0]:
      raise ValueError("QuarterMixTrainer needs a non-empty exact and perturbed history range")
    super().__init__(*args, **kwargs)

  def _sample_indices(self, key, start, count):
    members = self.n_ensemble or 1
    per = self.batch // 4
    n_current = self.batch - 3 * per
    keys = jax.random.split(key, 4)
    high = jnp.maximum(count, 1)
    current_exact = start + jax.random.randint(keys[0], (members, n_current), 0, high)
    current_perturbed = start + self.current_offset + jax.random.randint(keys[1], (members, per), 0, high)
    historical_exact = jax.random.randint(keys[2], (members, per), self.history_exact[0], self.history_exact[1])
    historical_perturbed = jax.random.randint(keys[3], (members, per), self.history_perturbed[0], self.history_perturbed[1])
    columns = [current_exact, current_perturbed, historical_exact, historical_perturbed]
    return jnp.concatenate(columns, axis=1).reshape(-1)

  def _sample_weights(self):
    """Uniform: every row of the batch counts the same, as it does in `ContinualTrainer` at its default
    `replay_weight` of 1.0 (whose weight vector is all ones after its mean-1 normalisation). The
    composition of the data is what is under test, not the price of a row."""
    return None

  def _init_design_network(self, init_seq, init_params):
    return fresh_design_network(self, init_seq, init_params)

  def _carried_state(self):
    return {}

  def _load_carried_state(self, data):
    """Nothing crosses a design boundary here: :func:`measure` builds the network itself."""


def pooled_mean_sem(means, sems, counts):
  """Mean and standard error over the UNION of several disjoint groups, from each group's own
  `masked_mean_sem` output.

  `masked_mean_sem` returns `mean` and `sqrt(var / (n - 1))` where `var` is the POPULATION variance, so a
  group's sum of squares about its own mean is `n * (n - 1) * sem^2`. The union's sum of squares adds the
  between-group term `n_g * (mean_g - mean)^2`, and its standard error divides the union's population
  variance by `n - 1`, which is exactly the convention the trainer's own numbers are quoted in. At one
  group this returns that group's `masked_mean_sem` unchanged.
  """
  counts = [int(c) for c in counts]
  total = sum(counts)
  if total <= 0:
    return float("nan"), float("nan")
  mean = sum(float(m) * c for m, c in zip(means, counts)) / total
  squares = 0.0
  for m, s, c in zip(means, sems, counts):
    squares += c * (c - 1) * float(s)**2 + c * (float(m) - mean)**2
  variance = squares / total
  return mean, math.sqrt(variance / max(total - 1, 1))


def region_index_array(events, offset, length):
  """An event-index array addressable AT POOL POSITIONS: position `offset + i` holds `events[i]`.

  `probe_scatter.fill` takes the next slice of an index array at the pool's CURRENT fill, which is what
  keeps the detector call deterministic in `(design, event_index)`. A region that does not start at pool
  position 0 therefore needs its events shifted to sit under its own offsets; the prefix is never read.
  """
  events = np.asarray(events, np.int64)
  return np.concatenate([np.zeros(int(offset), np.int64), events[:int(length)]])


def past_design_set(rows, exclude, n_past, seed, ceiling=0.95):
  """`n_past` past designs drawn from the campaign's OWN scored designs, excluding the ceiling and the
  designs that take a turn as the current one.

  `meta`'s replay history is the trajectory an optimiser walked, so the historical designs are drawn from
  the points the campaign really scored rather than uniformly over the cube. The draw is seeded on the
  study seed alone, so both arms of a pair get the identical set and the history can be laid down once
  per seed.
  """
  excluded = {provenance for provenance in exclude}
  pool = [r for r in rows if r[0] < ceiling and r[2] not in excluded]
  if len(pool) < n_past:
    raise SystemExit(f"probe_exact_perturbed_mix: only {len(pool)} campaign designs available for a history of {n_past}")
  chosen = np.random.default_rng([int(seed), 20260816]).choice(len(pool), size=int(n_past), replace=False)
  return [pool[int(i)] for i in sorted(chosen)]


def measure(trainer, detector, arm, test_pool, eval_test, n_test, base_scaled, sigma, seed, regions, plots_dir):
  """ONE run: grow, train and stop under the campaign's procedure on the CURRENT design, with the history
  already in the train pool below `history_end`, and score on a held-out pool at the EXACT design.

  The loop is `detopt/nn/trainer/design.py::_DesignBase.train`, clause for clause and in its order. What
  differs, and only this: the history was laid down by :func:`build_history` once per seed rather than by
  earlier `train` calls; the current design's rows live in one region (`control`) or two (`dual`, exact
  and perturbed), which the growth round fills in equal parts and the epoch evaluates separately before
  pooling; and a third, non-growing pool at the EXACT current design is evaluated alongside.

  `regions` is `(train_regions, val_regions, evals)`: each region is `(pool, start, window, sigma,
  index_array)` and `evals` maps a window length to the eval kernel built for it. THE SCORED SET IS THE
  UNION OF THE CURRENT REGIONS -- the history is training-only in both arms.

  The `count` handed to the train kernel is the SMALLEST current train region's fill, because one count
  addresses every current region and reading past a region's fill would sample a zeroed row. The
  cumulative split keeps the regions within one row of each other and every train total here is even, so
  in practice they are equal and nothing is left out; the `min` is the guard, not the mechanism. The
  SCORING uses each region's own fill, so it is exact regardless.
  """
  train_regions, val_regions, evals = regions
  train_pool, val_pool = trainer.train_pool, trainer.val_pool
  val_ratio = trainer._val_ratio
  train_counts = [0 for _ in train_regions]
  val_counts = [0 for _ in val_regions]

  design_rng = np.random.default_rng([int(seed), int(round(1e6 * float(sigma))), 7717])
  init_seq, training_seq = np.random.SeedSequence(int(seed)).spawn(2)
  params, state, opt_state = fresh_design_network(trainer, init_seq, None)
  initial_params = params
  key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))

  w0_train_j = jnp.int32(train_regions[0][1])
  region_cap_train = sum(window for _pool, _start, window, _sigma, _index in train_regions)
  region_cap_val = sum(window for _pool, _start, window, _sigma, _index in val_regions)

  def fill_regions(regions_, counts, n_total):
    """Add `n_total` rows split as evenly as possible over the regions, and return the number added, or
    None if a pool is full.

    THE SPLIT IS CUMULATIVE, not per round: each region's share is the difference between its target
    count AFTER this round and what it already holds, so the regions stay within one row of each other
    however many rounds are laid. Splitting each ROUND with a ceiling instead lets the first region
    collect one extra row EVERY round and it reaches its own cap early -- measured on the smoke test,
    the validation regions drifted to 301/295 and the run died as `pool exhausted` two rounds before the
    window cap.
    """
    total_after = sum(counts) + int(n_total)
    n_regions = len(regions_)
    targets = [total_after // n_regions + (1 if i < total_after % n_regions else 0) for i in range(n_regions)]
    shares = [target - count for target, count in zip(targets, counts)]
    for (pool, start, window, _sigma, _index), count, share in zip(regions_, counts, shares):
      if count + share > window:
        raise RuntimeError(f"region of window {window} asked to hold {count + share} rows -- sample_round's cap is wrong")
      if pool.current + share > pool.capacity:
        return None
    added = 0
    for i, ((pool, start, window, region_sigma, index_array), share) in enumerate(zip(regions_, shares)):
      if share <= 0:
        continue
      pool.current = start + counts[i]
      fill(detector, pool, index_array, perturb(base_scaled, share, region_sigma, design_rng))
      counts[i] += share
      added += share
    return added

  def sample_round(n_requested):
    """One growth round. Returns the number of TRAIN rows added, 0 if the window is full, or None if a
    pool is full. The train and validation TOTALS are the shipped ones -- the split into regions is
    what differs between the arms, never the amount."""
    n_train = min(int(n_requested), trainer.iteration_limit - sum(train_counts), region_cap_train - sum(train_counts))
    if n_train <= 0:
      return 0
    n_val = round(n_train * val_ratio)
    n_val = max(0, min(n_val, trainer.val_iteration_limit - sum(val_counts), region_cap_val - sum(val_counts)))
    added = fill_regions(train_regions, train_counts, n_train)
    if added is None:
      return None
    if n_val > 0 and fill_regions(val_regions, val_counts, n_val) is None:
      return None
    return added

  def evaluate(pools_regions, counts):
    """Per-region and pooled (mean, sem) over the CURRENT rows of one side (train or validation)."""
    means, sems = [], []
    for (pool, start, window, _sigma, _index), count in zip(pools_regions, counts):
      values = evals[window](params, state, pool.buffers(), jnp.int32(start))
      mean, sem = masked_mean_sem(values, count)
      means.append(float(mean))
      sems.append(float(sem))
    pooled = pooled_mean_sem(means, sems, counts)
    return pooled[0], pooled[1], means, sems

  history = {
    "train": [],
    "val": [],
    "test": [],
    "train_sem": [],
    "val_sem": [],
    "test_sem": [],
    "window": [],
    "train_parts": [],
    "val_parts": [],
    "train_sem_parts": [],
    "val_sem_parts": [],
    "round_start": []
  }
  started = time.time()
  design_list = np.asarray(base_scaled).tolist()

  def render(train_count, val_count, val_mean):
    """One growth-curve figure on a DETACHED daemon thread -- started, never joined, exactly as
    `probe_scatter` does and for the reason given there: `design.py`'s `ThreadPoolExecutor` is a deferred
    join and blocks the end of every run until its queue drains."""
    snapshot = trainer._snapshot(
      history["train"], history["val"], history["train_sem"], history["val_sem"], history["window"], train_count, val_count
    )
    threading.Thread(target=plot_iteration, args=(snapshot, 0, design_list, val_mean, plots_dir), daemon=True).start()

  if sample_round(trainer.n0) is None:
    return None

  round_start, epoch_in_round = 0, 0
  objective, status = None, "converged"
  test_per_row = np.zeros(n_test, np.float32)
  train_mean, val_mean, diff, err = float("nan"), float("nan"), float("nan"), float("nan")

  while True:
    train_count = sum(train_counts)
    val_count = sum(val_counts)
    key, subkey = jax.random.split(key)
    params, state, opt_state, _ = trainer._train_epoch(
      params, state, opt_state, subkey, w0_train_j, jnp.int32(min(train_counts)), train_pool.buffers()
    )

    train_mean, train_sem, train_parts, train_sem_parts = evaluate(train_regions, train_counts)
    val_mean, val_sem, val_parts, val_sem_parts = evaluate(val_regions, val_counts)
    test_eval = eval_test(params, state, test_pool.buffers(), jnp.int32(0))
    test_mean, test_sem = masked_mean_sem(test_eval, n_test)
    test_mean, test_sem = float(test_mean), float(test_sem)

    history["train"].append(train_mean)
    history["val"].append(val_mean)
    history["test"].append(test_mean)
    history["train_sem"].append(train_sem)
    history["val_sem"].append(val_sem)
    history["test_sem"].append(test_sem)
    history["window"].append(train_count)
    history["train_parts"].append(train_parts)
    history["val_parts"].append(val_parts)
    history["train_sem_parts"].append(train_sem_parts)
    history["val_sem_parts"].append(val_sem_parts)
    history["round_start"].append(round_start)
    epoch_in_round += 1

    if plots_dir is not None and len(history["train"]) % 16 == 0:
      render(train_count, val_count, val_mean)

    err = float(np.hypot(train_sem, val_sem))
    diff = abs(val_mean - train_mean)

    if epoch_in_round <= trainer.warmup_epochs:
      continue

    first = round_start + trainer.warmup_epochs
    tr = np.asarray(history["train"][first:], dtype=np.float64)
    va = np.asarray(history["val"][first:], dtype=np.float64)
    tr_s = np.asarray(history["train_sem"][first:], dtype=np.float64)
    va_s = np.asarray(history["val_sem"][first:], dtype=np.float64)
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
            f"diff={diff:.4f} err={err:.4f} | parts train={['%.4f' % v for v in train_parts]} "
            f"val={['%.4f' % v for v in val_parts]} | window={train_count}", flush=True
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
    round_start = len(history["train"])
    epoch_in_round = 0
    print(f"  [grow] window -> {sum(train_counts)} ({train_counts}), val {sum(val_counts)} ({val_counts})", flush=True)

  wall = time.time() - started
  train_count = sum(train_counts)
  val_count = sum(val_counts)
  if plots_dir is not None:
    render(train_count, val_count, history["val"][-1])
  n_rounds = int(1 + max(0, (train_count - trainer.n0 + trainer.n_increment - 1) // trainer.n_increment))
  row = {
    "arm": arm,
    "status": status,
    "wall_s": wall,
    "seed": int(seed),
    "sigma": float(sigma),
    "calls_train_val": int(train_count + val_count),
    "window": int(train_count),
    "val_window": int(val_count),
    "train_counts": [int(c) for c in train_counts],
    "val_counts": [int(c) for c in val_counts],
    "n_rounds": n_rounds,
    "n_epochs": len(history["train"]),
    "train": history["train"][-1],
    "val": history["val"][-1],
    "train_parts": history["train_parts"][-1],
    "val_parts": history["val_parts"][-1],
    "train_sem_parts": history["train_sem_parts"][-1],
    "val_sem_parts": history["val_sem_parts"][-1],
    "test": history["test"][-1],
    "test_sem": history["test_sem"][-1],
    "diff": diff,
    "err": err,
    "slack": diff + err,
    "objective": float("nan") if objective is None else float(objective[0]),
    "objective_std": float("nan") if objective is None else float(objective[1]),
    "per_epoch": {
      "train": [round(float(v), 6) for v in history["train"]],
      "val": [round(float(v), 6) for v in history["val"]],
      "test": [round(float(v), 6) for v in history["test"]],
      "train_sem": [round(float(v), 6) for v in history["train_sem"]],
      "val_sem": [round(float(v), 6) for v in history["val_sem"]],
      "train_parts": [[round(float(v), 6) for v in p] for p in history["train_parts"]],
      "val_parts": [[round(float(v), 6) for v in p] for p in history["val_parts"]],
      "train_sem_parts": [[round(float(v), 6) for v in p] for p in history["train_sem_parts"]],
      "val_sem_parts": [[round(float(v), 6) for v in p] for p in history["val_sem_parts"]],
      "window": [int(v) for v in history["window"]],
      "round_start": [int(v) for v in history["round_start"]],
    },
  }
  return row, test_per_row


def build_history(detector, trainer, arm, past_designs, history_rows, sigma, seed, train_index):
  """Lay the WHOLE history into the train pool, once per seed, and return `(history_end, ranges)`.

  control: `len(past_designs)` blocks of `history_rows` EXACT rows, block `i` consuming
           `train_index[i*B : (i+1)*B]`.
  dual:    `[0, H/2)` holds `len(past_designs)` blocks of `B/2` EXACT rows, block `i` consuming the
           FIRST HALF of the control's block `i` events; `[H/2, H)` holds the same blocks PERTURBED,
           consuming the SECOND HALF. Same designs, same events, same total volume -- half the rows
           moved onto perturbed designs.
  """
  train_pool = trainer.train_pool
  half = history_rows // 2
  design_rng = np.random.default_rng([int(seed), int(round(1e6 * float(sigma))), 4242])
  train_pool.current = 0
  if arm == "control":
    for design in past_designs:
      fill(detector, train_pool, train_index, perturb(design, history_rows, 0.0, design_rng))
    return train_pool.current, [("exact", 0, train_pool.current)]
  exact_events = np.concatenate([train_index[i * history_rows:i * history_rows + half] for i in range(len(past_designs))])
  perturbed_events = np.concatenate([
    train_index[i * history_rows + half:(i + 1) * history_rows] for i in range(len(past_designs))
  ])
  exact_index = region_index_array(exact_events, 0, len(exact_events))
  for design in past_designs:
    fill(detector, train_pool, exact_index, perturb(design, half, 0.0, design_rng))
  exact_end = train_pool.current
  perturbed_index = region_index_array(perturbed_events, exact_end, len(perturbed_events))
  for design in past_designs:
    fill(detector, train_pool, perturbed_index, perturb(design, half, sigma, design_rng))
  return train_pool.current, [("exact", 0, exact_end), ("perturbed", exact_end, train_pool.current)]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a RUN config, e.g. =enzyme_extremes")
  parser.add_argument(
    "--arm", choices=("control", "dual"), required=True,
    help="control = standard `meta`, one exact pool, ContinualTrainer as shipped; dual = exact + "
    "perturbed pools in 1:1, four-quarter batch"
  )
  parser.add_argument("--designs", nargs="*", default=["best", "p25", "median", "p75", "poor"], help="current designs, by name")
  parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="repetitions; the arms are PAIRED on these")
  parser.add_argument("--sigma", type=float, default=0.05, help="perturbation scale in the NORMAL space")
  parser.add_argument("--n-past", type=int, default=10, help="past designs in the replay history")
  parser.add_argument(
    "--history-rows", type=int, default=40960,
    help="TRAIN rows per past design, the same TOTAL in both arms (the dual arm lays half of them at "
    "the design and half at perturbed copies of it). The default is the train share (3:1) of `meta`'s "
    "measured pooled median per-design cost of 55976 calls, rounded down to 10 * n_increment"
  )
  parser.add_argument("--n-test", type=int, default=32768, help="held-out rows at the EXACT current design")
  parser.add_argument(
    "--budget", type=int, default=2097152,
    help="`training.budget`. FIXED: it selects the validation set, so cells at different budgets are "
    "not comparable and the campaign's own value is the only one used"
  )
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument("--n0", type=int, default=None, help="override `training.n0` (SMOKE TEST ONLY)")
  parser.add_argument("--n-increment", type=int, default=None, help="override `training.n_increment` (SMOKE TEST ONLY)")
  parser.add_argument("--iteration-limit", type=int, default=None, help="override `training.iteration_limit` (SMOKE TEST ONLY)")
  parser.add_argument("--plots-dir", default="output/mixprobe/plots", help="growth curves; '' turns the callback off")
  parser.add_argument("--output", default="output/mixprobe/mix.json")
  parser.add_argument("--resume", action="store_true", help="keep cells already in --output and skip them")
  arguments = parser.parse_args()

  import yaml

  from detopt.utils.config import optimizer as make_optimizer, resolve_device

  name = arguments.config.lstrip("=")
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  detector = detopt.detector.from_config(detector_config)

  run = json.loads(json.dumps(config))
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

  rows_all = campaign_rows()
  designs = design_set(rows_all)
  missing = [d for d in arguments.designs if d not in designs]
  if len(missing) > 0:
    raise SystemExit(f"probe_exact_perturbed_mix: unknown design(s) {missing}; have {sorted(designs)}")

  history_rows = int(arguments.history_rows)
  history_total = int(arguments.n_past) * history_rows
  charged = round(history_total / (1.0 - config["training"]["val_fraction"]))

  rows, per_row = [], {}
  npz_path = os.path.splitext(arguments.output)[0] + "_test_rows.npz"
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)["rows"]
    if os.path.isfile(npz_path):
      with np.load(npz_path) as data:
        per_row = {k: np.asarray(data[k]) for k in data.files}
  done = {(r["design"], r["seed"]) for r in rows}
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)

  print(
    f"arm {arguments.arm} | sigma {arguments.sigma} | budget {arguments.budget} | "
    f"current designs: {[(k, round(designs[k][0], 4), designs[k][2]) for k in arguments.designs]}", flush=True
  )
  print(
    f"history: {arguments.n_past} past designs x {history_rows} train rows = {history_total} rows "
    f"(charged {charged} calls at the 3:1 split; the validation share is NOT simulated because nothing "
    f"reads it) | n_test {arguments.n_test}", flush=True
  )

  for seed in arguments.seeds:
    if arguments.arm == "control":
      trainer = ContinualTrainer.from_config(detector, run, checkpoint_dir=None, seed=seed)
    else:
      iteration_limit = _round_down(run["training"]["n0"], run["training"]["n_increment"], run["training"]["iteration_limit"])
      trainer = QuarterMixTrainer(
        detector, regressor_config=run["regressor"], optimizer=make_optimizer(run["training"]["optimizer"]),
        device=resolve_device(run.get("device")), checkpoint_dir=None, seed=seed, history_exact=(0, history_total // 2),
        history_perturbed=(history_total // 2, history_total), current_offset=iteration_limit // 2, **{
          k: v
          for k, v in run["training"].items() if k != "optimizer"
        }
      )

    eval_test = build_test_eval(trainer, detector, arguments.n_test)
    test_index = shuffled_event_index(detector.size(), arguments.n_test, int(seed) + 987654321, name="test events")
    test_pool = Pool(arguments.n_test, trainer.train_pool.specs, trainer.device)
    print(
      f"seed {seed}: pools train {trainer.train_pool.capacity} val {trainer.val_pool.capacity} "
      f"test {test_pool.capacity} | iteration_limit {trainer.iteration_limit} val cap "
      f"{trainer.val_iteration_limit} | steps/epoch {trainer.steps_per_epoch}", flush=True
    )

    past = past_design_set(rows_all, [designs[k][2] for k in designs], arguments.n_past, seed)
    past_designs = [np.asarray(entry[1], np.float32) for entry in past]
    started = time.time()
    history_end, ranges = build_history(
      detector, trainer, arguments.arm, past_designs, history_rows, float(arguments.sigma), int(seed), trainer._train_index
    )
    print(
      f"  history: {history_end} train rows in ranges {ranges} from designs "
      f"{[(round(e[0], 4), e[2]) for e in past]} in {time.time() - started:.0f} s "
      f"({history_end / max(1e-9, time.time() - started):.0f} events/s)", flush=True
    )
    if history_end != history_total:
      raise SystemExit(f"probe_exact_perturbed_mix: laid {history_end} historical rows, expected {history_total}")

    train_window = trainer.iteration_limit
    val_window = trainer.val_iteration_limit
    if arguments.arm == "control":
      train_regions_spec = [(history_end, train_window, 0.0, trainer._train_index)]
      val_regions_spec = [(0, val_window, 0.0, trainer._val_index)]
    else:
      half_train, half_val = train_window // 2, -(-val_window // 2)
      train_regions_spec = [
        (history_end, half_train, 0.0, region_index_array(trainer._train_index[history_end::2], history_end, half_train)),
        (
          history_end + half_train, half_train, float(arguments.sigma),
          region_index_array(trainer._train_index[history_end + 1::2], history_end + half_train, half_train)
        ),
      ]
      val_regions_spec = [
        (0, half_val, 0.0, region_index_array(trainer._val_index[0::2], 0, half_val)),
        (half_val, half_val, float(arguments.sigma), region_index_array(trainer._val_index[1::2], half_val, half_val)),
      ]
    evals = {}
    for _start, window, _sigma, _index in train_regions_spec + val_regions_spec:
      if window not in evals:
        evals[window] = build_test_eval(trainer, detector, window)
    train_regions = [(trainer.train_pool, s, w, g, i) for s, w, g, i in train_regions_spec]
    val_regions = [(trainer.val_pool, s, w, g, i) for s, w, g, i in val_regions_spec]
    print(
      f"  regions: train {[(s, w, g) for s, w, g, _ in train_regions_spec]} "
      f"val {[(s, w, g) for s, w, g, _ in val_regions_spec]}", flush=True
    )

    for design_name in arguments.designs:
      base_loss, base_scaled, provenance = designs[design_name]
      base_scaled = np.asarray(base_scaled, np.float32)
      if (design_name, int(seed)) in done:
        print(f"skip {design_name} s{seed} (already measured)", flush=True)
        continue
      test_pool.current = 0
      fill(detector, test_pool, test_index, perturb(base_scaled, arguments.n_test, 0.0, None))
      plots_dir = None
      if len(arguments.plots_dir) > 0:
        plots_dir = os.path.join(arguments.plots_dir, design_name, f"s{seed}", arguments.arm)
      print(f"=== {design_name} (campaign loss {base_loss:.4f}) seed {seed} arm {arguments.arm}", flush=True)
      outcome = measure(
        trainer, detector, arguments.arm, test_pool, eval_test, arguments.n_test, base_scaled, float(arguments.sigma),
        int(seed), (train_regions, val_regions, evals), plots_dir
      )
      if outcome is None:
        print("  pool exhausted before the first round -- not a measurement", flush=True)
        continue
      row, test_losses = outcome
      row.update({
        "design": design_name,
        "provenance": provenance,
        "campaign_loss": base_loss,
        "x_scaled": [float(v) for v in base_scaled],
        "history_rows": history_total,
        "history_calls_charged": int(charged),
        "history_ranges": [[n, int(a), int(b)] for n, a, b in ranges],
        "n_past": int(arguments.n_past),
        "past_provenance": [e[2] for e in past],
        "budget": int(arguments.budget),
        "calls_total": int(row["calls_train_val"] + arguments.n_test + history_total),
      })
      rows.append(row)
      per_row[f"{design_name}|{seed}"] = test_losses
      with open(arguments.output, "w") as f:
        json.dump({
          "rows": rows,
          "n_test": int(arguments.n_test),
          "budget": int(arguments.budget),
          "arm": arguments.arm,
          "sigma": float(arguments.sigma)
        }, f, indent=1)
      np.savez_compressed(npz_path, **per_row)
      print(
        f"  -> calls {row['calls_train_val']} rounds {row['n_rounds']} epochs {row['n_epochs']} "
        f"window {row['window']} test {row['test']:.4f} val {row['val']:.4f} "
        f"val_parts {['%.4f' % v for v in row['val_parts']]} slack {row['slack']:.4f} "
        f"[{row['status']}] {row['wall_s']:.0f} s", flush=True
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
