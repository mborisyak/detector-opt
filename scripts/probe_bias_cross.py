#!/usr/bin/env python3
"""The Bayesian convergence rule's BIAS, over the FULL cross `arm x dropconnect x patience`.

    python scripts/probe_bias_cross.py =linear_d2n3_growth --seeds 1 2 \
        --patience 4 8 16 32 64 --dropconnect 0.0 0.1 --output output/biascross/linear.json

WHAT THIS ADDS TO `probe_cost_bias.py`, which it imports rather than repeats. That study moved ONE
knob at a time from a common baseline and swept `meta` over `patience` alone, so it can say whether
several levers trace one cost-versus-bias curve but it cannot say whether `dropconnect` acts
DIFFERENTLY on the two arms -- the arm x dropconnect cells were never run. This script runs the cross,
and it runs on tasks that have no closed-form floor.

THE PRECISION AXIS (H1/H2 of `docs/criterion-investigation.md`). `--loss-precision` takes a LIST and
crosses it with the arm axis. Two things move with it and both are forced, not chosen:

  `iteration_limit` is HELD FIXED at `--iteration-limit` across the whole sweep. `steps_per_epoch =
  iteration_limit // batch` does not depend on the window, so a fixed `iteration_limit` keeps one
  epoch worth a constant number of GRADIENT STEPS -- which is what makes `patience`, written in
  epochs, mean the same thing at every bar. It must also exceed the window the TIGHTEST bar needs, so
  it is normally raised above the run config's own value; the passes-over-window this implies FALL as
  the window grows and that is reported, not hidden.

  `n0` and `n_increment` are SCALED by `(config_precision / precision)^2`, the same law the window
  follows, so the number of growth rounds stays comparable across bars instead of exploding at the
  tight end. `n0 == n_increment == iteration_limit` disables growth entirely and voided a previous
  `linear` arm comparison; the scaling never approaches it here and the values are printed.

THE FLOOR, and what is and is not measurable.

  `linear`   `LinearDetector.bayes_risk` is the EXACT per-design floor,
             `tr[(X^T X / noise^2 + I)^-1] / (d + 1)`, so `excess = loss - floor` IS the bias. It takes
             a NOMINAL design; the scaled vector silently returns a larger, wrong floor. The paired
             control variate of `probe_cost_bias.optimal_row_losses` is the primary read-out.
  `extremes`, `MM`
             NO CLOSED FORM EXISTS and none is invented here. `floor` is None, no `excess` is written,
             and what is reported instead is the held-out `test` loss on a pool of `--n-test` rows
             simulated at the EXACT scored design, identical row for row across every cell. Its offset
             from the unknown floor is a property of the DESIGN, not of the cell, so DIFFERENCES
             between cells at the same design are bias differences and the absolute level is not a
             bias. The report must say so.

⚠️ THE NETWORK IS FRESH IN BOTH ARMS. What separates `meta` here is its REPLAY POOL and nothing else,
which is `probe_cost_bias`'s scope and keeps every row of this study comparable with the rows already
in `output/costbias/`. It is NOT the full `meta` arm and must not be reported as one.

WITHOUT A HISTORY `meta` IS `from_scratch`, IDENTICALLY -- at `start == 0`
`ContinualTrainer._sample_indices` draws both halves of the batch from the current window. A history
of `--n-past` designs x `--history-rows` train rows is therefore laid down before the scored design's
window opens, drawn UNIFORMLY in the scaled cube from an rng seeded on the study seed alone, so both
arms and every setting of one seed share the identical history. `from_scratch`'s cursor is simply
ADVANCED past it, which costs no simulation and makes both arms open the scored window at the same
pool cursor.

THE PROCEDURE IS THE CAMPAIGN'S OWN, driven by `probe_replay_mix.measure`, which is
`detopt/nn/trainer/design.py::_DesignBase.train` clause for clause. `design.py` is NOT modified and
NOT subclassed.

THIS SCRIPT SETS NOTHING. It writes a JSON of measurements.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time

import matplotlib

matplotlib.use("AGG")

import numpy as np

import jax

import detopt
import detopt.detector
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.utils.events import shuffled_event_index
from detopt.utils.pools import Pool

from probe_cost_bias import TEST_INDEX_SEED, optimal_row_losses, scored_designs, settled_probability
from probe_replay_mix import measure, uniform_designs
from probe_scatter import build_test_eval, fill


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a GROWTH run config, e.g. =linear_d2n3_growth")
  parser.add_argument("--trajectory", required=True, help="results.json whose designs are scored")
  parser.add_argument(
    "--design-ranks", type=int, nargs="+", default=[0, 5, 10, 15, 19],
    help="ORDER STATISTICS of the trajectory's recorded loss, fixed before anything runs"
  )
  parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2], help="repetitions; every cell is PAIRED on these")
  parser.add_argument("--patience", type=int, nargs="+", default=[4, 8, 16, 32, 64])
  parser.add_argument(
    "--loss-precision", type=float, nargs="+", default=None,
    help="the bar, crossed with every other axis; `n0` and `n_increment` scale as 1/precision^2 from "
    "the config's own values. Default: the config's own single value"
  )
  parser.add_argument(
    "--iteration-limit", type=int, default=None,
    help="HELD FIXED across the precision sweep; it is the epoch length and must exceed the tightest "
    "bar's window"
  )
  parser.add_argument("--dropconnect", type=float, nargs="+", default=[0.0, 0.1])
  parser.add_argument("--arms", nargs="+", default=["from_scratch", "meta"], choices=("from_scratch", "meta"))
  parser.add_argument("--n-past", type=int, default=9, help="historical designs, drawn UNIFORMLY in the scaled cube")
  parser.add_argument("--history-rows", type=int, default=6144, help="TRAIN rows per historical design")
  parser.add_argument("--n-test", type=int, default=32768, help="held-out rows at the EXACT scored design")
  parser.add_argument("--budget", type=int, default=None, help="override `training.budget` (it sizes the pools)")
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument("--output", default="output/biascross/cross.json")
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
  has_floor = hasattr(detector, "bayes_risk")

  run = json.loads(json.dumps(config))
  if arguments.budget is not None:
    run["training"]["budget"] = int(arguments.budget)
  if arguments.device is not None:
    run["device"] = arguments.device
  run["plot_per_epoch"] = False
  if arguments.iteration_limit is not None:
    run["training"]["iteration_limit"] = int(arguments.iteration_limit)
  regressor_key = next(iter(run["regressor"]))
  optimizer_key = next(iter(run["training"]["optimizer"]))

  config_precision = float(run["training"]["loss_precision"])
  precisions = [config_precision] if arguments.loss_precision is None else [float(v) for v in arguments.loss_precision]
  schedules = {}
  for precision in precisions:
    scale = (config_precision / precision)**2
    n0 = int(round(int(run["training"]["n0"]) * scale))
    n_increment = int(round(int(run["training"]["n_increment"]) * scale))
    if n_increment >= n0 or n0 >= int(run["training"]["iteration_limit"]):
      raise SystemExit(
        f"probe_bias_cross: precision {precision:g} gives n0 {n0}, n_increment {n_increment} against "
        f"iteration_limit {run['training']['iteration_limit']} -- growth would be degenerate"
      )
    schedules[precision] = (n0, n_increment)

  designs = scored_designs(arguments.trajectory, arguments.design_ranks)
  test_index = shuffled_event_index(detector.size(), arguments.n_test, TEST_INDEX_SEED, name="test events")
  floors, optimal_rows = {}, {}
  for rank, index, recorded, x_scaled in designs:
    if has_floor:
      floors[rank] = float(detector.bayes_risk(detector.flatten_design(detector.to_nominal(x_scaled[None, :]))[0]))
      optimal_rows[rank] = optimal_row_losses(detector, x_scaled, test_index)
    else:
      floors[rank] = None

  history_total = int(arguments.n_past) * int(arguments.history_rows)
  charged = round(history_total / (1.0 - float(run["training"]["val_fraction"])))

  print(f"config {name} | trajectory {arguments.trajectory} | closed-form floor {has_floor}", flush=True)
  for rank, index, recorded, _ in designs:
    floor_text = "NOT MEASURABLE" if floors[rank] is None else f"{floors[rank]:.5f}"
    print(f"  rank {rank:>3} -> trajectory index {index:>2}  recorded {recorded:.5f}  floor {floor_text}", flush=True)
  print(
    f"cross: {len(arguments.arms)} arms x {len(precisions)} precision x {len(arguments.dropconnect)} "
    f"dropconnect x {len(arguments.patience)} patience x {len(designs)} designs x "
    f"{len(arguments.seeds)} seeds | history {history_total} train rows (charged {charged})", flush=True
  )
  steps_per_epoch = int(run["training"]["iteration_limit"]) // int(run["training"]["batch"])
  for precision in precisions:
    n0, n_increment = schedules[precision]
    print(
      f"  precision {precision:g}: n0 {n0} n_increment {n_increment} iteration_limit "
      f"{run['training']['iteration_limit']} ({steps_per_epoch} steps/epoch)", flush=True
    )

  rows = []
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)["rows"]
  done = {(r["arm"], r["seed"], r.get("loss_precision", config_precision), r["dropconnect"], r["patience"], r["rank"])
          for r in rows}
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)

  for seed in arguments.seeds:
    history_rng = np.random.default_rng([int(seed), 20260818])
    past_designs = uniform_designs(arguments.n_past, int(detector.design_dim()), history_rng)
    for precision in precisions:
      for dropconnect in arguments.dropconnect:
        for patience in arguments.patience:
          for arm in arguments.arms:
            key = (arm, int(seed), float(precision), float(dropconnect), int(patience))
            pending = [d for d in designs if key + (d[0], ) not in done]
            if len(pending) == 0:
              print(
                f"skip {arm} s{seed} precision {precision:g} dropconnect {dropconnect:g} "
                f"patience {patience} (all designs measured)", flush=True
              )
              continue

            cell_run = copy.deepcopy(run)
            cell_run["training"]["patience"] = int(patience)
            cell_run["training"]["loss_precision"] = float(precision)
            cell_run["training"]["n0"], cell_run["training"]["n_increment"] = schedules[precision]
            if dropconnect > 0.0:
              cell_run["regressor"][regressor_key]["dropconnect"] = float(dropconnect)
            else:
              cell_run["regressor"][regressor_key].pop("dropconnect", None)

            factory = ContinualTrainer if arm == "meta" else DesignTrainer
            trainer = factory(
              detector, regressor_config=cell_run["regressor"], optimizer=make_optimizer(cell_run["training"]["optimizer"]),
              device=resolve_device(cell_run.get("device")), checkpoint_dir=None, seed=seed, **{
                k: v
                for k, v in cell_run["training"].items() if k != "optimizer"
              }
            )
            required = history_total + trainer.iteration_limit
            if trainer.train_pool.capacity < required:
              raise SystemExit(
                f"probe_bias_cross: train pool {trainer.train_pool.capacity} < history {history_total} + "
                f"iteration_limit {trainer.iteration_limit}; raise --budget"
              )
            eval_test = build_test_eval(trainer, detector, arguments.n_test)
            test_pool = Pool(arguments.n_test, trainer.train_pool.specs, trainer.device)

            started = time.time()
            trainer.train_pool.current = 0
            if arm == "meta":
              for design in past_designs:
                fill(
                  detector, trainer.train_pool, trainer._train_index,
                  np.broadcast_to(design[None, :], (int(arguments.history_rows), design.size)).copy()
                )
              if trainer.train_pool.current != history_total:
                raise SystemExit(f"probe_bias_cross: laid {trainer.train_pool.current} rows, expected {history_total}")
            else:
              trainer.train_pool.current = history_total
            print(
              f"=== s{seed} precision {precision:g} dropconnect {dropconnect:g} patience {patience} arm {arm} | history "
              f"{trainer.train_pool.current} in {time.time() - started:.1f} s | steps/epoch {trainer.steps_per_epoch}",
              flush=True
            )

            for rank, index, recorded, x_scaled in pending:
              test_pool.current = 0
              fill(
                detector, test_pool, test_index,
                np.broadcast_to(x_scaled[None, :], (arguments.n_test, x_scaled.size)).copy()
              )
              outcome = measure(
                trainer, detector, test_pool, eval_test, arguments.n_test, x_scaled, int(seed), history_total, None
              )
              if outcome is None:
                print(f"  rank {rank}: pool exhausted before the first round -- NOT a measurement", flush=True)
                continue
              row, network_rows = outcome
              probability, slope, round_start = settled_probability(
                row, trainer.warmup_epochs, trainer.patience, trainer.loss_precision
              )
              row.update({
                "arm": arm,
                "setting": f"precision={precision:g},dropconnect={dropconnect:g},patience={patience}",
                "loss_precision": float(precision),
                "n0": int(cell_run["training"]["n0"]),
                "n_increment": int(cell_run["training"]["n_increment"]),
                "steps_per_epoch": int(trainer.steps_per_epoch),
                "patience": int(patience),
                "weight_decay": float(cell_run["training"]["optimizer"][optimizer_key]["weight_decay"]),
                "dropconnect": float(dropconnect),
                "rank": int(rank),
                "design_index": int(index),
                "trajectory_loss": float(recorded),
                "x_scaled": [float(v) for v in x_scaled],
                "bayes_risk": floors[rank],
                "p_settled_at_stop": probability,
                "train_slope_at_stop": slope,
                "final_round_start": int(round_start),
                "history_rows": history_total,
                "history_calls_charged": int(charged) if arm == "meta" else 0,
              })
              if floors[rank] is not None:
                paired = np.asarray(network_rows, np.float64) - optimal_rows[rank]
                row.update({
                  "excess_paired": float(np.mean(paired)),
                  "excess_paired_sem": float(np.std(paired, ddof=1) / np.sqrt(paired.size)),
                  "optimal_test": float(np.mean(optimal_rows[rank])),
                  "excess_reported": float(row["val"]) - floors[rank],
                  "excess_test": float(row["test"]) - floors[rank],
                })
              rows.append(row)
              with open(arguments.output, "w") as f:
                json.dump({"rows": rows, "n_test": int(arguments.n_test), "config": name}, f, indent=1)
              settled = "n/a" if probability is None else f"{probability:.3f}"
              excess = "" if floors[rank] is None else f"excess {row['excess_paired']:+.5f} "
              print(
                f"  rank {rank:>3}: window {row['window']:>6} calls {row['calls_train_val']:>6} "
                f"epochs {row['n_epochs']:>4} val {row['val']:.4f} test {row['test']:.4f} {excess}"
                f"diff {row['diff']:.4f} err {row['err']:.4f} P(settled) {settled} "
                f"[{row['status']}] {row['wall_s']:.0f} s", flush=True
              )

            for buffer in jax.tree.leaves(test_pool.buffers()):
              buffer.delete()
            for pool in (trainer.train_pool, trainer.val_pool):
              for buffer in jax.tree.leaves(pool.buffers()):
                buffer.delete()
            del trainer, eval_test, test_pool
            gc.collect()

  print(f"-> {arguments.output}", flush=True)


if __name__ == "__main__":
  main()
