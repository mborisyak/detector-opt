#!/usr/bin/env python3
"""`meta` against `from_scratch` at ONE mid-trajectory design, with `meta`'s history SAMPLED rather
than accumulated.

    python scripts/probe_trajectory_midpoint.py =enzyme_extremes --seeds 1 2 3 \
        --output output/midpoint/mid_s1.json

THE QUESTION, and why the campaign cannot answer it. A per-design arm ratio taken from the campaign is
a property of `(design, draw, trajectory)`, not of the design. `from_scratch` is MEMORYLESS, so a
validation draw touches only the design being scored -- its design-3 gap was 0.00560 under two
different draws, identical to five digits. `meta` is PATH-DEPENDENT: it arrives at design k carrying
whatever pool designs 1..k-1 happened to cost. Measured: under `training.budget` 786432 an expensive
design 2 left `meta` at design 3 with 600712 calls of history against 60073 under the campaign's own
2097152, and its gap fell 0.00540 -> 0.00400, its window 81920 -> 36864, and the arm ratio moved
1.10 -> 2.89. The advantage mechanism and the confound are THE SAME THING, which is why no amount of
re-reading campaign results separates them.

THE CONSTRUCTION removes the path by FIXING the history. One trajectory is chosen, its MIDDLE design
is scored, and `meta`'s pool is filled from that trajectory's EARLIER designs at a volume set here
rather than by its own spending. Both arms then meet at the same design, on the same draw, with the
same event slice, and the ratio is a property of the design alone.

THE TRAJECTORY IS `continue`, DELIBERATELY -- a third arm, which neither of the two under test
produced. Scoring `meta` on a path `meta` chose (or `from_scratch` on one `from_scratch` chose) hands
one arm designs its own acquisition preferred; `continue` is neutral to both. The trajectory is the
finished campaign's, seed 1244111331, 25 designs.

THE SCORED DESIGN IS CHOSEN UNDER TWO CONSTRAINTS, BOTH OF THEM VALIDITY REQUIREMENTS.

  POSITION -- the MIDDLE THIRD of the trajectory, so `meta` arrives with a realistic pool. Early
  designs do not test the arms against each other at all: at design 1 `_sample_indices` takes its
  `start == 0` branch and BOTH halves of the batch come from the current window, so `meta` IS
  `from_scratch`, identically; at design 2 the entire history is 8192 rows at one design. The
  campaign's own prefix shows the consequence -- arm ratios 1.00, 1.00, 1.10 over designs 1-3, rising
  only as the pool fills. Measuring the advantage before the mechanism has anything to work with
  measures nothing.

  INFORMATIVENESS -- among those, the design with the LOWEST recorded loss. A design at the
  no-information value 1.0 makes the optimal response a CONSTANT, and convergence there measures how
  fast a network learns to emit a constant, which both arms do at the same speed. This is not
  hypothetical: the campaign's first five designs sit at 0.999, 0.646, 0.830, 0.914 and 0.932 against
  a ceiling of 1.0 and a best of 0.514, so four of the five are 80% of the way to carrying no signal.

  NEITHER CONSTRAINT SELECTS ON THE OUTCOME. Both are fixed before any run, both are read off the
  trajectory's OWN recorded loss rather than off anything either arm did, and both apply to the two
  arms identically -- the design is the same design in both. What is excluded is a design at which the
  question cannot be asked, not a design at which the answer is inconvenient.

At the default trajectory that selects index 15, loss 0.5138, with fifteen designs of history behind
it.

THE HISTORY. `--history-rows` TRAIN rows, default 589824 = 144 * `n_increment`, split across the
fifteen earlier designs IN PROPORTION to what the trajectory spent on each (largest-remainder rounding
to whole growth quanta, so every block is a whole number of increments and the fifteen sum exactly).
The default models `meta`'s OWN accumulated volume at that point -- 791860 calls over its first fifteen
designs, whose train share at the campaign's 3:1 split is 593895, rounded down to a whole 144 quanta
(0.7% below nominal). THE HISTORICAL VALIDATION ROWS ARE NOT SIMULATED: `_eval_val` reads from
`w0_val` forward, i.e. the current design's window alone, and every replay draw reads the TRAIN pool,
so those rows would cost 196608 calls that no measured quantity depends on. The saving is stated, not
hidden, and `history_calls_charged` charges the full 786432 the train rows stand for.

THE TWO ARMS, paired on seed, SHIPPED CODE in both cases -- no subclass, no re-implementation.

  meta          `ContinualTrainer`. History in `[0, history_end)`; batch half current-design window,
                half replay, exactly `ContinualTrainer._sample_indices`.
  from_scratch  `DesignTrainer`. Batch entirely from the current window, exactly
                `window_sample_indices`.

⚠️ THE NETWORK IS FRESH IN BOTH ARMS. What is under test is `meta`'s HISTORY, not its carried weights.
A parallel decomposition measured the weight channel alone at 0.91-1.11x, i.e. nothing, so this
isolates the channel that carries the effect; it is NOT the full `meta` arm and the report must not
call it one. Reconstructing the carried network would mean training it on the sampled history first,
which is a different and more expensive study.

THE EVENT SLICE IS SHARED. `from_scratch` never reads a row below `start` -- `window_sample_indices`
draws from `[start, start + count)` -- so its history is NOT simulated; its pool cursor is simply
ADVANCED to `history_end` and the rows below stay at the pool's zeros. That costs nothing and buys the
pairing that matters: both arms open the current design's window at the same pool cursor and therefore
consume the IDENTICAL slice of the run's event index. Filling those rows for an arm that cannot see
them would spend 589824 calls per seed to change no measured quantity.

WHAT ELSE IS SHARED WITHIN A PAIR: the scored design, the seed, the network initialisation, the
validation split (`training.budget` is left at the campaign's own 2097152, and the budget IS what
selects the split), the growth schedule, the stopping rule, and the test rows. Only the history the
batch can reach differs.

THE TEST POOL is `--n-test` rows at the EXACT scored design, on a FIXED event index independent of the
study seed, so all six runs are scored against the identical rows and every comparison -- across arms
and across repetitions -- is paired per row.

WHAT THE PROBE MUST SEPARATE, stated before it was run. The primary quantity is detector calls to
convergence, which moves in quanta of one growth round (4096 train + 1365 validation), so at `meta`'s
measured level of ~49150 calls ONE round is ~10% and no single pair resolves better than that. With
three repetitions the SE of the mean log ratio is sd/sqrt(3); at the 0.03 per-pair sd measured across
the scattering study's three pairs that is 2%, and at a deliberately pessimistic 0.15 it is 9%. The
campaign's own claim is 1.3-1.7x, i.e. 26-53% in log terms, so three repetitions resolve it with room;
they do NOT resolve a few per cent and no such claim is made. The secondary quantity, test loss, is
paired over 32768 identical rows at a per-row SEM of ~0.003 against a `loss_precision` of 0.01 -- it
can separate a tenth of the bar, not less.

WHAT THREE REPETITIONS CANNOT SAY. One design, one trajectory, one history volume. Whether the ratio
holds at other designs, other trajectories, or other history volumes is not measured here and is not
extrapolated.

THE PROCEDURE IS THE CAMPAIGN'S OWN, driven by `probe_replay_mix.measure`, which is
`detopt/nn/trainer/design.py::_DesignBase.train` clause for clause. `design.py` is NOT modified and
NOT subclassed. `probe_replay_mix` and `probe_scatter` are IMPORTED and left UNEDITED, because queued
jobs run them.

THIS SCRIPT SETS NOTHING. It writes a JSON of measurements plus an NPZ of per-row test losses.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("AGG")

import numpy as np

import detopt
import detopt.detector
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.utils.events import shuffled_event_index
from detopt.utils.pools import Pool

from probe_replay_mix import measure
from probe_scatter import build_test_eval, fill

TRAJECTORY = "output/enzyme_extremes/1244111331/continue/results.json"
TEST_INDEX_SEED = 20260817


def trajectory_designs(path, midpoint=None):
  """`(scored_scaled, past_scaled, past_spent, index, loss)` for the trajectory at `path`.

  The scored design is the LOWEST-LOSS design of the trajectory's MIDDLE THIRD, or `midpoint` if the
  caller names an index outright. The history is every design BEFORE it, in the order the run visited
  them, each with the number of detector calls the run spent on it. See the module docstring for why
  both constraints are validity requirements rather than outcome selection.
  """
  with open(path) as f:
    recorded = json.load(f)["results"]
  rows = [r for r in recorded if "x_scaled" in r and r.get("loss") is not None]
  if midpoint is None:
    low, high = len(rows) // 3, 2 * len(rows) // 3 + 1
    index = min(range(low, high), key=lambda i: rows[i]["loss"])
  else:
    index = int(midpoint)
  if index <= 0 or index >= len(rows):
    raise SystemExit(f"probe_trajectory_midpoint: midpoint {index} outside 1..{len(rows) - 1}")
  scored = np.asarray(rows[index]["x_scaled"], np.float32)
  past = np.asarray([r["x_scaled"] for r in rows[:index]], np.float32)
  spent = np.asarray([r["spent"] for r in rows[:index]], np.int64)
  return scored, past, spent, index, float(rows[index]["loss"])


def history_blocks(spent, total_rows, quantum):
  """`spent` split into whole `quantum`s summing EXACTLY to `total_rows`, by largest remainder.

  Proportional to what the trajectory spent on each design, so a design the run laboured over is
  proportionally represented in the pool -- which is what `meta`'s own accumulation does. Every block
  is a whole number of growth quanta, and a design whose share rounds to zero is given one quantum so
  no historical design vanishes from the pool entirely.
  """
  n_quanta = int(total_rows) // int(quantum)
  if n_quanta < len(spent):
    raise SystemExit(f"probe_trajectory_midpoint: {n_quanta} quanta cannot cover {len(spent)} designs")
  share = np.asarray(spent, np.float64) / float(np.sum(spent)) * n_quanta
  counts = np.maximum(1, np.floor(share).astype(np.int64))
  while int(np.sum(counts)) > n_quanta:
    counts[int(np.argmax(counts))] -= 1
  for position in np.argsort(-(share - np.floor(share))):
    if int(np.sum(counts)) >= n_quanta:
      break
    counts[position] += 1
  return counts * int(quantum)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a RUN config, e.g. =enzyme_extremes")
  parser.add_argument("--arms", nargs="+", default=["meta", "from_scratch"], choices=("meta", "from_scratch"))
  parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="repetitions; the arms are PAIRED on these")
  parser.add_argument("--trajectory", default=TRAJECTORY, help="results.json of the run whose middle design is scored")
  parser.add_argument("--midpoint", type=int, default=None, help="scored design INDEX (default: the middle)")
  parser.add_argument(
    "--history-rows", type=int, default=589824,
    help="TOTAL train rows of history, split across the earlier designs in proportion to their spend"
  )
  parser.add_argument("--n-test", type=int, default=32768, help="test rows at the EXACT scored design")
  parser.add_argument(
    "--budget", type=int, default=2097152,
    help="`training.budget`. It sizes the pools AND selects the validation split, so the campaign's own "
    "value keeps this study on the campaign's draw"
  )
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument(
    "--features", default=None,
    help="override the regressor's `features` as JSON, e.g. '[[36,24],[24,36]]' for the 1.5x width. "
    "BOTH ARMS ALWAYS SHARE ONE ARCHITECTURE -- this sets the whole cell, never one arm"
  )
  parser.add_argument("--n0", type=int, default=None, help="override `training.n0` (SMOKE TEST ONLY)")
  parser.add_argument("--n-increment", type=int, default=None, help="override `training.n_increment` (SMOKE TEST ONLY)")
  parser.add_argument("--iteration-limit", type=int, default=None, help="override `training.iteration_limit` (SMOKE TEST ONLY)")
  parser.add_argument("--plots-dir", default="output/midpoint/plots", help="growth curves; '' turns the callback off")
  parser.add_argument("--output", default="output/midpoint/mid.json")
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
  run["training"]["budget"] = int(arguments.budget)
  for key, value in (("n0", arguments.n0), ("n_increment", arguments.n_increment), ("iteration_limit",
                                                                                    arguments.iteration_limit)):
    if value is not None:
      run["training"][key] = int(value)
  if arguments.device is not None:
    run["device"] = arguments.device
  if arguments.features is not None:
    key = next(iter(run["regressor"]))
    run["regressor"][key]["features"] = json.loads(arguments.features)
    print(f"regressor features overridden -> {run['regressor'][key]['features']}", flush=True)
  run["plot_per_epoch"] = False

  scored, past, spent, index, recorded_loss = trajectory_designs(arguments.trajectory, arguments.midpoint)
  blocks = history_blocks(spent, arguments.history_rows, run["training"]["n_increment"])
  history_total = int(np.sum(blocks))
  charged = round(history_total / (1.0 - config["training"]["val_fraction"]))

  print(
    f"trajectory {arguments.trajectory}\n"
    f"  scored design: index {index} of {index + len(past) + 1 - len(past)}..; recorded loss "
    f"{recorded_loss:.4f}; history = the {len(past)} designs before it", flush=True
  )
  print(
    f"  history {history_total} train rows in {len(blocks)} blocks (charged {charged} calls at the "
    f"{1 - config['training']['val_fraction']:.2f}:{config['training']['val_fraction']:.2f} split; the "
    f"validation share is NOT simulated because nothing reads it)", flush=True
  )
  print(f"  blocks: {[int(b) for b in blocks]}", flush=True)

  rows = []
  npz_path = os.path.splitext(arguments.output)[0] + "_test_rows.npz"
  per_row = {}
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)["rows"]
    if os.path.isfile(npz_path):
      with np.load(npz_path) as data:
        per_row = {k: np.asarray(data[k]) for k in data.files}
  done = {(r["arm"], r["seed"]) for r in rows}
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)

  test_index = shuffled_event_index(detector.size(), arguments.n_test, TEST_INDEX_SEED, name="test events")

  for seed in arguments.seeds:
    for arm in arguments.arms:
      if (arm, int(seed)) in done:
        print(f"skip {arm} s{seed} (already measured)", flush=True)
        continue
      trainer = (ContinualTrainer
                 if arm == "meta" else DesignTrainer).from_config(detector, run, checkpoint_dir=None, seed=seed)
      required = history_total + trainer.iteration_limit
      if trainer.train_pool.capacity < required:
        raise SystemExit(
          f"probe_trajectory_midpoint: train pool {trainer.train_pool.capacity} < history {history_total} + "
          f"iteration_limit {trainer.iteration_limit}; raise --budget"
        )
      eval_test = build_test_eval(trainer, detector, arguments.n_test)
      test_pool = Pool(arguments.n_test, trainer.train_pool.specs, trainer.device)
      test_pool.current = 0
      fill(detector, test_pool, test_index, np.broadcast_to(scored[None, :], (arguments.n_test, scored.size)).copy())

      trainer.train_pool.current = 0
      started = time.time()
      if arm == "meta":
        for design, count in zip(past, blocks):
          fill(
            detector, trainer.train_pool, trainer._train_index,
            np.broadcast_to(design[None, :], (int(count), design.size)).copy()
          )
        if trainer.train_pool.current != history_total:
          raise SystemExit(f"probe_trajectory_midpoint: laid {trainer.train_pool.current} rows, expected {history_total}")
        print(
          f"  history laid down in {time.time() - started:.0f} s "
          f"({history_total / max(1e-9, time.time() - started):.0f} events/s)", flush=True
        )
      else:
        trainer.train_pool.current = history_total
        print(
          "  history NOT simulated: `window_sample_indices` cannot reach it; cursor advanced so the "
          "current design's window consumes the same event slice", flush=True
        )

      plots_dir = None
      if len(arguments.plots_dir) > 0:
        plots_dir = os.path.join(arguments.plots_dir, f"s{seed}", arm)
      print(f"=== arm {arm} seed {seed} | design index {index} (trajectory loss {recorded_loss:.4f})", flush=True)
      outcome = measure(trainer, detector, test_pool, eval_test, arguments.n_test, scored, int(seed), history_total, plots_dir)
      if outcome is None:
        print("  pool exhausted before the first round -- not a measurement", flush=True)
        continue
      row, test_per_row = outcome
      row["arm"] = arm
      row["design_index"] = int(index)
      row["trajectory"] = arguments.trajectory
      row["trajectory_loss"] = recorded_loss
      row["history_rows"] = history_total
      row["history_calls_charged"] = int(charged) if arm == "meta" else 0
      rows.append(row)
      per_row[f"{arm}_s{seed}"] = test_per_row
      print(
        f"  -> {arm} s{seed}: window {row['window']} calls {row['calls_train_val']} "
        f"val {row['val']:.4f} test {row['test']:.4f} diff {row['diff']:.4f} err {row['err']:.4f} "
        f"[{row['status']}] {row['wall_s']:.0f} s", flush=True
      )
      with open(arguments.output, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
      np.savez_compressed(npz_path, **per_row)

  print(f"-> {arguments.output}", flush=True)


if __name__ == "__main__":
  main()
