#!/usr/bin/env python3
"""Does the COMPOSITION of `meta`'s replay history change the cost of fitting the NEXT design?

    python scripts/probe_replay_mix.py =enzyme_extremes --arm A --seeds 1 \
        --output output/replaymix/mix_A_s1.json

THE QUESTION. `meta` replays the designs the optimiser happened to visit -- a TRAJECTORY-CONCENTRATED
sample: every historical row sits at one of a handful of points. Would a replay pool that COVERS the
design space fit the next design more cheaply? A parallel study established that `meta`'s advantage
runs entirely through the train/validation gap (`meta` 0.00200 against `from_scratch` 0.00420) and the
stopping rule is `diff + err <= loss_precision`, so a smaller gap converges at a smaller window and
costs fewer detector calls. If broad coverage regularises better, the spread arm converges cheaper; if
replay works by being NEAR the current design, it converges dearer. Both answers are interesting and
neither is favoured here.

THE TWO ARMS, paired on (11th design, seed), at the SAME total historical volume.

  A (concentrated)  History = 10 random designs, `--history-rows` TRAIN rows each, laid down before
                    training starts. Batch = 1:1 current-design : history, i.e.
                    `ContinualTrainer._sample_indices` EXACTLY AS SHIPPED (128 + 128 of a 256 batch,
                    per ensemble member).
  B (mixed)         The same total rows, composed HALF concentrated / HALF whole-space, and a
                    FOUR-QUARTER batch (64 each, per ensemble member) drawn from four SEPARATE index
                    ranges:
                      * the current design's window,
                      * a WHOLE-SPACE pool (fresh coverage; each row its own uniform design),
                      * the 10 past designs,
                      * a RANDOM history (again one uniform design per row).
                    The two whole-space ranges are drawn SEPARATELY and never merged: one stands for
                    coverage bought on purpose, the other for coverage that accumulated.

⚠️ ARM B CHANGES TWO THINGS AT ONCE -- the history's composition AND the batch geometry (the current
design falls from half the batch to a quarter). This is what was asked for; it is NOT a
single-variable comparison, and the report must say so. A third cell holding the batch geometry fixed
(1:1 current : whole-space-only history) would separate them and is NOT run here.

THE HISTORY VOLUME. `meta`'s MEASURED per-design cost, campaign medians, is 49150 (seed 1244111331)
and 62802 (seed 126382657), pooled median 55976 detector calls, so ten designs of history is 559760
calls. At the campaign's own 3:1 train:validation split that is 419820 train rows, ROUNDED DOWN to
409600 = 10 x 40960 = 100 x `n_increment` (2.4% below the nominal figure) so every historical block is
a whole number of growth quanta. THE HISTORICAL VALIDATION ROWS ARE NOT SIMULATED: nothing reads them
-- `_eval_val` evaluates from `w0_val` forward (the current design's window alone) and every replay
draw reads the TRAIN pool -- so simulating them would spend 136533 calls per run on rows no measured
quantity depends on. The saving is stated rather than hidden, and the reported `history_calls_charged`
charges the full 546133 calls the 409600 train rows stand for.

WHAT IS AND IS NOT SHARED WITHIN A PAIR. Same 11th design, same seed, same network initialisation,
same growth schedule, same stopping rule, same test rows, same total historical rows, and the SAME 10
past designs. The window's event indices are identical too: both arms lay down exactly 409600
historical train rows, so the current design's window opens at the same pool cursor and consumes the
same slice of the run's event index. Only the composition of the history and the batch geometry
differ.

THE CONVERGENCE PROBE RUNS ON THE 11TH DESIGN ALONE, in both arms -- its own window, from `w0_train`
forward, exactly as `ContinualTrainer` does. Both arms are scored on one fixed test pool of
`--n-test` rows simulated at the EXACT 11th design, never perturbed and identical row for row across
arms and seeds, so the test comparison is paired per row.

THE 11TH DESIGNS are the campaign's own scored designs at fixed order statistics of its loss range
(best / p25 / median / p75 / poor, `probe_scatter.design_set`), excluding the uninformative ceiling:
a design pinned at the no-information value 1.0 cannot move in either arm and would return "no
difference" for a reason that has nothing to do with the question. The HISTORICAL designs are drawn
UNIFORMLY in the scaled cube, in both arms, so the two arms' histories have the same design
DISTRIBUTION and differ only in how CONCENTRATED it is -- 10 points against ~200k.

WHAT THE PROBE MUST SEPARATE, stated before it was run. The primary quantity is detector calls to
convergence on the 11th design, which moves in quanta of one growth round (4096 train + 1365
validation rows). Arm A is expected near `meta`'s own per-design cost, i.e. a window of ~40960 train
rows and ~9 rounds, so ONE round is ~10% of the level and a single pair cannot resolve better than
that. The decision this feeds is economic: arm B's whole-space half buys 204800 train rows (273067
charged calls) of designs nobody wanted scored, which is 4.88 scored designs at 55976 calls each, so
the saving must reach 273067 / K calls per design to repay within K designs -- 24% of the per-design
cost to repay over 20 designs, 49% to repay over 10. With 5 designs x 3 seeds = 15 pairs, a per-pair
sd of the log ratio of 0.15 (5x the 0.03 measured across the three pairs of the scattering study,
deliberately pessimistic because arm B also changes the batch geometry) gives SE 0.039, i.e. 8% at
2 sigma and 12% at 3 sigma. That resolves the 24% that decides the trade with room to spare; it does
NOT resolve a few per cent, and no such claim is made. The secondary quantity, test loss at the 11th
design, is paired over 32768 identical rows, where the per-row SEM is ~0.003 against a
`loss_precision` of 0.01.

WHAT ONE DESIGN CANNOT SAY. The 11th design is measured once, against a history of ten. If a saving
recurs at every later design the payback is `273067 / saving_per_design` designs; if it decays as the
real history grows, a single design cannot show that, and this script does not extrapolate.

THE PROCEDURE IS THE CAMPAIGN'S OWN. The loop in :func:`measure` is
`detopt/nn/trainer/design.py::_DesignBase.train`, clause for clause and in its order -- warmup, the
bayesian gap/plateau tests, `n_increment` growth on (1) or (2.1), the `param_mix` rewind with the
optimiser reset, and the exit on `|val - train| + hypot(sems) <= loss_precision`. Every kernel it
steps is the trainer's own. `design.py` is NOT modified and NOT subclassed. The loop is copied rather
than called because this study needs a third pool the trainer does not have and a history built ONCE
per seed and reused across the five 11th designs (`_DesignBase.train` resets nothing and would
re-simulate 409600 rows per design). `scripts/probe_scatter.py` supplies the pieces that ARE reusable
-- `campaign_rows`, `design_set`, `fill`, `build_test_eval` -- and is deliberately left UNEDITED,
because queued jobs run it.

THIS SCRIPT SETS NOTHING. It writes a JSON of measurements plus an NPZ of per-row test losses.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from detopt.nn.trainer.common import fresh_design_network
from detopt.nn.trainer.design import _DesignBase
from detopt.utils.events import shuffled_event_index
from detopt.utils.pools import Pool
from detopt.utils.training import (bayesian_trend, masked_mean_sem, probability_above, probability_change_below)
from detopt.utils.viz.bo import plot_iteration

from probe_scatter import build_test_eval, campaign_rows, design_set, fill

MEDIAN_PER_DESIGN_CALLS = 55976


class BlockReplayTrainer(_DesignBase):
  """`meta`'s replay, drawn from SEVERAL NAMED BLOCKS of the pool instead of from one.

  `replay_blocks` is a tuple of `(low, high)` row ranges. The batch is split into `1 + len(blocks)`
  EQUAL parts per ensemble member: the first from the current design's window `[start, start+count)`,
  then one from each block, concatenated in the trainer's `(members, batch)` layout so the loss's
  reshape still recovers one minibatch per member. With a single block `(0, start)` this is
  `ContinualTrainer._sample_indices`; ARM A DOES NOT USE THIS CLASS, it uses `ContinualTrainer`
  itself, so the concentrated arm is the shipped code path and not a re-implementation of it.

  The ranges are PYTHON INTS baked into the kernel at construction, which is why an arm gets its own
  trainer instance: the history layout is fixed for a whole job, and a value read at trace time must
  not be allowed to change under a cached kernel.

  This IMPLEMENTS the abstract hooks (`_sample_indices`, `_sample_weights`, `_init_design_network`,
  `_carried_state`, `_load_carried_state`); it overrides no concrete method.
  """

  def __init__(self, *args, replay_blocks=(), **kwargs):
    self.replay_blocks = tuple((int(low), int(high)) for low, high in replay_blocks)
    if len(self.replay_blocks) == 0:
      raise ValueError("BlockReplayTrainer needs at least one replay block")
    super().__init__(*args, **kwargs)

  def _sample_indices(self, key, start, count):
    members = self.n_ensemble or 1
    per = self.batch // (len(self.replay_blocks) + 1)
    n_current = self.batch - per * len(self.replay_blocks)
    keys = jax.random.split(key, len(self.replay_blocks) + 1)
    columns = [start + jax.random.randint(keys[0], (members, n_current), 0, jnp.maximum(count, 1))]
    for subkey, (low, high) in zip(keys[1:], self.replay_blocks):
      columns.append(jax.random.randint(subkey, (members, per), low, high))
    return jnp.concatenate(columns, axis=1).reshape(-1)

  def _sample_weights(self):
    """Uniform: every row of the batch counts the same, as it does in `ContinualTrainer` at its
    default `replay_weight` of 1.0. Nothing here reweights the quarters -- the composition of the
    history is what is under test, not the price of a replay row."""
    return None

  def _init_design_network(self, init_seq, init_params):
    return fresh_design_network(self, init_seq, init_params)

  def _carried_state(self):
    return {}

  def _load_carried_state(self, data):
    """Nothing crosses a design boundary here: :func:`measure` builds the network itself."""


def uniform_designs(n, n_dimensions, rng):
  """`n` designs drawn UNIFORMLY in the scaled cube [0, 1]^n_dimensions, one row each.

  The scaled cube is the space BO searches and the space the invariant kernel is written on, so a
  uniform draw there is what "covers the design space" means for this task. Nothing is clipped and
  nothing piles up on a bound.
  """
  return np.asarray(rng.random((int(n), int(n_dimensions))), np.float32)


def build_history(detector, trainer, arm, past_designs, history_rows, rng):
  """Lay the WHOLE history into the train pool, once, and return its block boundaries.

  ARM A: `len(past_designs)` blocks of `history_rows`, all rows of a block at that block's design.
  ARM B: the same total, as
      [0, 5H)      the same past designs at HALF the rows each      (concentrated)
      [5H, 7.5H)   one uniform design PER ROW                       (whole-space, fresh coverage)
      [7.5H, 10H)  one uniform design PER ROW                       (whole-space, accumulated)
  The two whole-space blocks are drawn from independent rows and kept apart because the batch draws
  them separately.

  The PAST DESIGNS ARE DRAWN BEFORE ANYTHING ELSE by the caller, from an rng seeded on the study seed
  alone, so both arms of a pair get the identical ten.
  """
  n_past = len(past_designs)
  train_pool = trainer.train_pool
  blocks = []
  if arm == "A":
    for design in past_designs:
      start = train_pool.current
      fill(detector, train_pool, trainer._train_index, np.broadcast_to(design[None, :], (history_rows, design.size)).copy())
      blocks.append(("past", start, train_pool.current))
    return [("past_designs", 0, train_pool.current)], blocks
  half = history_rows // 2
  start = train_pool.current
  for design in past_designs:
    block_start = train_pool.current
    fill(detector, train_pool, trainer._train_index, np.broadcast_to(design[None, :], (half, design.size)).copy())
    blocks.append(("past", block_start, train_pool.current))
  concentrated = (start, train_pool.current)
  spread_rows = (n_past * history_rows - (train_pool.current - start)) // 2
  ranges = []
  for name in ("whole_space", "random_history"):
    block_start = train_pool.current
    fill(detector, train_pool, trainer._train_index, uniform_designs(spread_rows, past_designs.shape[1], rng))
    ranges.append((name, block_start, train_pool.current))
    blocks.append((name, block_start, train_pool.current))
  named = [("past_designs", concentrated[0], concentrated[1])] + ranges
  return named, blocks


def measure(trainer, detector, test_pool, eval_test, n_test, base_scaled, seed, history_end, plots_dir):
  """ONE run: grow, train and stop under the campaign's procedure on the 11TH DESIGN, with the
  history already in the pool below `history_end`, and score on a test pool held at that design.

  The loop is `detopt/nn/trainer/design.py::_DesignBase.train`, clause for clause and in its order.
  What differs, and only this: the history was laid down by :func:`build_history` (once per seed, so
  five 11th designs share it) rather than by earlier `train` calls, and a third, non-growing pool at
  the 11th design is evaluated alongside the two the procedure reads.

  THE HISTORY IS IN THE TRAIN POOL ONLY and `w0_val` is 0 -- see the module docstring for why the
  historical validation rows are not simulated.
  """
  train_pool, val_pool = trainer.train_pool, trainer.val_pool
  train_pool.current, val_pool.current = int(history_end), 0
  val_ratio = trainer._val_ratio

  init_seq, training_seq = np.random.SeedSequence(int(seed)).spawn(2)
  params, state, opt_state = fresh_design_network(trainer, init_seq, None)
  initial_params = params
  key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))

  w0_train, w0_val = train_pool.current, 0
  w0_train_j, w0_val_j = jnp.int32(w0_train), jnp.int32(0)

  def sample_round(n_requested):
    """One growth round at the 11th design. Returns the number of TRAIN rows added, 0 if this
    design's window is full, or None if a pool is full."""
    n_train = min(int(n_requested), trainer.iteration_limit - (train_pool.current - w0_train))
    if n_train <= 0:
      return 0
    n_val = round(n_train * val_ratio)
    n_val = max(0, min(n_val, trainer.val_iteration_limit - (val_pool.current - w0_val)))
    if n_train > train_pool.capacity - train_pool.current or n_val > val_pool.capacity - val_pool.current:
      return None
    fill(detector, train_pool, trainer._train_index, np.broadcast_to(base_scaled[None, :], (n_train, base_scaled.size)).copy())
    if n_val > 0:
      fill(detector, val_pool, trainer._val_index, np.broadcast_to(base_scaled[None, :], (n_val, base_scaled.size)).copy())
    return n_train

  train_history, val_history, test_history = [], [], []
  train_sem_history, val_sem_history, test_sem_history = [], [], []
  pool_history = []
  started = time.time()
  design_list = np.asarray(base_scaled).tolist()

  def render(train_count, val_count, val_mean):
    """One growth-curve figure on a DETACHED daemon thread -- started, never joined, exactly as
    `probe_scatter` does and for the reason given there: `design.py`'s `ThreadPoolExecutor` is a
    deferred join and blocks the end of every run until its queue drains."""
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
    "seed": int(seed),
    "calls_train_val": int(train_count + val_count),
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
  parser.add_argument(
    "--arm", choices=("A", "B"), required=True,
    help="A = concentrated (ContinualTrainer as shipped); B = mixed, four-quarter batch"
  )
  parser.add_argument(
    "--designs", nargs="*", default=["best", "p25", "median", "p75", "poor"], help="the 11th designs, by name"
  )
  parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="repetitions; the arms are PAIRED on these")
  parser.add_argument("--n-past", type=int, default=10, help="historical designs")
  parser.add_argument(
    "--history-rows", type=int, default=40960,
    help="TRAIN rows per historical design in arm A (arm B lays HALF of these at each past design and "
    "spends the other half on whole-space rows). The default is the train share (3:1) of `meta`'s "
    "measured pooled median per-design cost of 55976 calls, rounded down to 10 * n_increment"
  )
  parser.add_argument("--n-test", type=int, default=32768, help="test rows at the EXACT 11th design, filled once per design")
  parser.add_argument(
    "--budget", type=int, default=1310720,
    help="`training.budget`, which sizes the pools: the train share must hold the whole history plus a "
    "full `iteration_limit` window, so no run can end as `pool exhausted`"
  )
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument("--n0", type=int, default=None, help="override `training.n0` (SMOKE TEST ONLY)")
  parser.add_argument("--n-increment", type=int, default=None, help="override `training.n_increment` (SMOKE TEST ONLY)")
  parser.add_argument("--iteration-limit", type=int, default=None, help="override `training.iteration_limit` (SMOKE TEST ONLY)")
  parser.add_argument("--plots-dir", default="output/replaymix/plots", help="growth curves; '' turns the callback off")
  parser.add_argument("--output", default="output/replaymix/mix.json")
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

  designs = design_set(campaign_rows())
  missing = [d for d in arguments.designs if d not in designs]
  if len(missing) > 0:
    raise SystemExit(f"probe_replay_mix: unknown design(s) {missing}; have {sorted(designs)}")

  n_dimensions = int(detector.design_shape()[0])
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
    f"arm {arguments.arm} | design set: {[(k, round(designs[k][0], 4), designs[k][2]) for k in arguments.designs]}", flush=True
  )
  print(
    f"history: {arguments.n_past} past designs x {history_rows} train rows = {history_total} rows "
    f"(charged {charged} calls at the 3:1 split; the validation share is NOT simulated because "
    f"nothing reads it) | n_test {arguments.n_test}", flush=True
  )

  for seed in arguments.seeds:
    design_rng = np.random.default_rng([int(seed), 20260816])
    past_designs = uniform_designs(arguments.n_past, n_dimensions, design_rng)

    if arguments.arm == "A":
      trainer = ContinualTrainer.from_config(detector, run, checkpoint_dir=None, seed=seed)
    else:
      concentrated = arguments.n_past * (history_rows // 2)
      spread = (history_total - concentrated) // 2
      trainer = BlockReplayTrainer(
        detector, regressor_config=run["regressor"], optimizer=make_optimizer(run["training"]["optimizer"]),
        device=resolve_device(run.get("device")), checkpoint_dir=None, seed=seed,
        replay_blocks=((0, concentrated), (concentrated, concentrated + spread),
                       (concentrated + spread, concentrated + 2 * spread)), **{
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

    started = time.time()
    trainer.train_pool.current = 0
    named, blocks = build_history(detector, trainer, arguments.arm, past_designs, history_rows, design_rng)
    history_end = trainer.train_pool.current
    print(
      f"  history laid down: {history_end} train rows in {len(blocks)} blocks, replay ranges "
      f"{[(n, a, b) for n, a, b in named]} in {time.time() - started:.0f} s "
      f"({history_end / max(1e-9, time.time() - started):.0f} events/s)", flush=True
    )
    if history_end != history_total:
      raise SystemExit(f"probe_replay_mix: laid {history_end} historical rows, expected {history_total}")

    for design_name in arguments.designs:
      base_loss, base_scaled, provenance = designs[design_name]
      base_scaled = np.asarray(base_scaled, np.float32)
      if (design_name, int(seed)) in done:
        print(f"skip {design_name} s{seed} (already measured)", flush=True)
        continue
      test_pool.current = 0
      fill(detector, test_pool, test_index, np.broadcast_to(base_scaled[None, :], (arguments.n_test, base_scaled.size)).copy())
      plots_dir = None
      if len(arguments.plots_dir) > 0:
        plots_dir = os.path.join(arguments.plots_dir, design_name, f"s{seed}", arguments.arm)
      print(f"=== {design_name} (campaign loss {base_loss:.4f}) seed {seed} arm {arguments.arm}", flush=True)
      outcome = measure(
        trainer, detector, test_pool, eval_test, arguments.n_test, base_scaled, int(seed), history_end, plots_dir
      )
      if outcome is None:
        print("  pool exhausted before the first round -- not a measurement", flush=True)
        continue
      row, test_losses = outcome
      row.update({
        "design": design_name,
        "arm": arguments.arm,
        "provenance": provenance,
        "campaign_loss": base_loss,
        "x_scaled": [float(v) for v in base_scaled],
        "history_rows": history_total,
        "history_calls_charged": int(charged),
        "n_past": int(arguments.n_past),
        "replay_blocks": [[n, int(a), int(b)] for n, a, b in named],
        "calls_total": int(row["calls_train_val"] + arguments.n_test + history_total),
      })
      rows.append(row)
      per_row[f"{design_name}|{seed}"] = test_losses
      with open(arguments.output, "w") as f:
        json.dump({
          "rows": rows,
          "n_test": int(arguments.n_test),
          "budget": int(arguments.budget),
          "arm": arguments.arm
        }, f, indent=1)
      np.savez_compressed(npz_path, **per_row)
      print(
        f"  -> calls {row['calls_train_val']} rounds {row['n_rounds']} epochs {row['n_epochs']} "
        f"window {row['window']} test {row['test']:.4f} val {row['val']:.4f} slack {row['slack']:.4f} "
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
