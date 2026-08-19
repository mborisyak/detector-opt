#!/usr/bin/env python3
"""Is `meta`'s "fewer detector calls" a LEARNING gain, or is it UNDERFITTING bought by a regulariser?

    python scripts/probe_cost_bias.py =linear_d2n3_growth --seeds 1 2 \
        --output output/costbias/costbias.json

THE HYPOTHESIS UNDER TEST (user, 2026-08-18). The growth procedure exits when
`diff + err <= loss_precision` with `diff = |val - train|`, so it REWARDS A SMALL TRAIN/VALIDATION
GAP -- and the cheapest way to have a small gap is not to fit the data. Anything that prevents
fitting shrinks `diff`, converges at a smaller window, and therefore "wins" on detector calls while
reporting a BIASED loss. `meta`'s batch is half off-design replay, which is a regulariser in all but
name (`ContinualTrainer`'s own docstring calls it "a fixed 2x dilution of the current design's
signal" and records +0.0184 / +0.0214 / +0.0326 of reported loss against a fresh network on three
enzyme designs). So `meta`'s cost advantage may be regularisation strength and nothing else.

THE DECISIVE CONTROL. Sweep REGULARISATION on `from_scratch` alone and see whether it reproduces
`meta`'s position. Every cell reports two numbers and they are plotted against each other:

  COST  detector calls (and the train window) at convergence -- the claimed advantage;
  BIAS  `excess = loss - bayes_risk(design)` -- the price.

`from_scratch` swept over its knobs traces a COST-VERSUS-BIAS CURVE. `meta` is then one more point on
those axes. If `meta` lands ON the curve, its advantage IS regularisation. If it lands BELOW it --
less cost at equal bias, or less bias at equal cost -- replay is doing something a regulariser cannot.

⚠️ `linear` IS THE ONLY TASK HERE WHERE THIS IS MEASURABLE. `LinearDetector.bayes_risk` is the EXACT
per-design floor, `tr[(X^T X / noise^2 + I)^-1] / (d + 1)`, a posterior covariance trace rather than
an approximation, so `excess` is a bias measurement and not a proxy for one. On the enzyme tasks one
can see that a curve is still descending but not how far it had to go. It takes a NOMINAL design.

THREE EXCESSES ARE REPORTED, and the distinction matters.
  `excess_reported = val - bayes_risk`  what BO actually consumes. Its own sampling error is the
      validation SEM, ~0.013 at a 1365-row validation window at level 0.38, which is comparable to
      the effects looked for -- so it is reported but it is NOT the primary number.
  `excess_test = test - bayes_risk`     the same network scored on a FIXED test pool of `--n-test`
      rows at the exact design, identical row for row across every cell of the study (the test event
      index is seeded independently of the study seed). Better, but its own sampling error is still
      ~0.0036 at 32768 rows and a level of 0.5.
  `excess_paired`                       the same quantity with the BAYES-OPTIMAL predictor as a
      CONTROL VARIATE: `mean(network_row_loss - optimal_row_loss)` over those same rows, the optimal
      predictor being the closed-form posterior mean (:func:`optimal_row_losses`). Same expectation,
      far smaller variance, because both losses are computed on the identical draw of `(w, b)` and
      the identical readings. THIS IS THE PRIMARY BIAS READ-OUT, and `optimal_test` against
      `bayes_risk` is the check that it is measuring what it claims.

THE SETTINGS SWEPT, each varying ONE knob from a common baseline (`patience` 16, `weight_decay`
1.0e-3, no dropconnect -- the run config's own values), so the curve is not an artefact of one lever:

  patience       4, 8, 16, 32, 64     the stopping horizon. The settled test is
                                      `P(|slope| * patience < 0.5 * loss_precision) > 0.9`, so the
                                      flatness bar in per-STEP terms is
                                      `0.5 * loss_precision / (patience * steps_per_epoch)` and this
                                      sweep moves it 16-fold. It is a stopping knob, not a
                                      regulariser, and it belongs on the same axes for exactly that
                                      reason: it trades the same two quantities.
  weight_decay   1e-5 ... 1e-1        the optimiser's own regularisation, 4 decades.
  dropconnect    off, 0.05, 0.1, 0.2  weight-level noise. ⚠️ PRIOR: on `linear` at the NARROW width
                                      it was measured to do nothing (slack 0.00879 with against
                                      0.00883 without, decision log D96) -- capacity, not
                                      regularisation, is the lever on this task. The network here is
                                      the wider [[32, 16], [16, 32]] of the d2n3 configs, so there
                                      may be room; the prior is stated so a flat result is a
                                      confirmation rather than a surprise.

THE ARMS, paired on (design, seed) and, within a setting, on the event slice.

  from_scratch  `DesignTrainer` -- `window_sample_indices`, batch entirely from the current design's
                window. Swept over all of the above.
  meta          `ContinualTrainer` -- `_sample_indices` as shipped: half the batch from the current
                window, half from `[0, history_end)`. Swept over `patience` only, which places it on
                the axes at five points rather than one.

⚠️ THE NETWORK IS FRESH IN BOTH ARMS, so what is measured is `meta`'s REPLAY, not its carried
weights. This is `probe_trajectory_midpoint`'s scope and its reason: a parallel decomposition put the
weight channel at 0.91-1.11x, i.e. nothing, and the replay channel is the one the hypothesis is
about. The report must not call this the full `meta` arm.

WITHOUT A HISTORY `meta` IS `from_scratch`, IDENTICALLY. At `start == 0` `ContinualTrainer._sample_indices`
takes its no-history branch and draws both halves of the batch from the current window, so a
single-design comparison is vacuous. A history is therefore laid down before the scored design's
window opens.

THE HISTORY IS `--n-past` designs drawn UNIFORMLY in the scaled cube, `--history-rows` train rows
each, from an rng seeded on the STUDY SEED alone -- so both arms, all settings and all scored designs
of one seed share the identical history. IT IS DELIBERATELY NOT THE TRAJECTORY'S OWN PREFIX. The
scored designs are chosen by RANK from that trajectory, and its first nine designs contain two of
them outright (ranks 15 and 19 are trajectory indices 3 and 1) plus a BIT-EXACT DUPLICATE of the
median design (index 8 equals index 9); a prefix history would therefore put three of the five scored
designs into `meta`'s own replay pool and confound precisely the cross-design comparison this study
makes. A uniform history is disjoint from every scored design by construction and is the distribution
BO's own `n_init` block draws from.

THE HISTORICAL VALIDATION ROWS ARE NOT SIMULATED and `from_scratch`'s historical TRAIN rows are not
either: `_eval_val` reads from `w0_val` forward (the current design's window alone), every replay
draw reads the train pool, and `window_sample_indices` cannot reach below `start`. `from_scratch`'s
pool cursor is simply ADVANCED to `history_end`, which costs nothing and buys the pairing that
matters -- both arms open the scored design's window at the same cursor and consume the IDENTICAL
slice of the run's event index. `history_calls_charged` charges `meta` the full figure the train rows
stand for.

THE SCORED DESIGNS come from a finished FIXED-procedure trajectory at fixed ORDER STATISTICS of its
own recorded loss (`--design-ranks`, default 0 / 5 / 10 / 15 / 19 of 20). Rank, never outcome: the
rule is fixed before any run and is read off the trajectory's recorded loss, not off anything either
arm did here. Rank 10 is the median design the study was originally briefed on.

P(settled) AT THE FIRING EPOCH is recomputed post hoc from the per-epoch history the measurement
already stores -- same arrays, same `bayesian_trend` / `probability_change_below`, and the round
boundary recovered from the window history -- so "the test crossed a bar" and "the test detected a
plateau" can be told apart without touching the loop.

THE PROCEDURE IS THE CAMPAIGN'S OWN, driven by `probe_replay_mix.measure`, which is
`detopt/nn/trainer/design.py::_DesignBase.train` clause for clause. `design.py` is NOT modified and
NOT subclassed. `probe_replay_mix` and `probe_scatter` are IMPORTED and left UNEDITED, because queued
jobs run them.

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
from detopt.utils.training import bayesian_trend, probability_change_below

from probe_replay_mix import measure, uniform_designs
from probe_scatter import build_test_eval, fill

TRAJECTORY = "output/linear-d2n3-fixed/1244111331/from_scratch/results.json"
TEST_INDEX_SEED = 20260818


def scored_designs(path, ranks):
  """`[(rank, index, recorded_loss, x_scaled), ...]` at fixed ORDER STATISTICS of the trajectory's own
  recorded loss.

  Rank, not outcome: the ranks are given on the command line before anything runs and the ordering is
  the trajectory's recorded loss, which no arm of this study produced.
  """
  with open(path) as f:
    recorded = json.load(f)["results"]
  rows = [r for r in recorded if "x_scaled" in r and r.get("loss") is not None]
  order = sorted(range(len(rows)), key=lambda i: float(rows[i]["loss"]))
  chosen = []
  for rank in ranks:
    if rank < 0 or rank >= len(order):
      raise SystemExit(f"probe_cost_bias: rank {rank} outside 0..{len(order) - 1}")
    index = order[int(rank)]
    chosen.append((int(rank), int(index), float(rows[index]["loss"]), np.asarray(rows[index]["x_scaled"], np.float32)))
  return chosen


def optimal_row_losses(detector, x_scaled, test_index):
  """The BAYES-OPTIMAL predictor's loss on each test row, in closed form.

  The posterior of `(w, b)` given the readings is Gaussian with mean `Sigma X^T y / noise^2` and
  covariance `Sigma = (X^T X / noise^2 + I)^-1`, and `LinearDetector.loss` is the mean squared error
  over the `d + 1` components, so this is the per-row loss of the estimator whose EXPECTATION is
  `bayes_risk`. Two uses, and the second is the reason it is here:

  * a CHECK -- its mean over the test pool must agree with `bayes_risk` to within its own standard
    error, which validates the design encoding, the pool contents and the floor together;
  * a CONTROL VARIATE. `excess = test_loss - bayes_risk` measured directly carries the test pool's own
    sampling error, ~0.0036 at 32768 rows and a level of 0.5, which is the size of the effects looked
    for. `mean(network_row - optimal_row)` has the SAME expectation and a far smaller variance,
    because the two losses are computed on the identical draw of `(w, b)` and the identical readings
    and are therefore strongly correlated. It is the same quantity measured better, not a different
    quantity.
  """
  physical = detector.flatten_design(detector.to_nominal(np.asarray(x_scaled, np.float32)[None, :]))
  _ground_truth, event, _mask, target = detector(physical, test_index)
  readings = np.asarray(event.response, np.float64)
  truth = np.asarray(target.coefficients, np.float64)
  flat = np.asarray(physical, np.float64).reshape(-1)
  probe = flat.reshape(detector.n_dimensions, detector.n_probes).T
  rows = np.concatenate([probe, np.ones((detector.n_probes, 1), np.float64)], axis=-1)
  covariance = np.linalg.inv(rows.T @ rows / detector.noise**2 + np.eye(detector.n_dimensions + 1))
  posterior_mean = (readings @ rows / detector.noise**2) @ covariance
  return np.mean(np.square(posterior_mean - truth), axis=-1)


def settings_grid(patience_values, weight_decay_values, dropconnect_values, baseline):
  """One knob moved at a time from `baseline`, deduplicated, in sweep order.

  A grid over all three would be 100 cells of which most vary two knobs at once; the question is
  whether SEVERAL DIFFERENT levers trace the SAME cost-versus-bias curve, and that needs one axis at
  a time.
  """
  grid = []
  for knob, values in (("patience", patience_values), ("weight_decay", weight_decay_values), ("dropconnect",
                                                                                              dropconnect_values)):
    for value in values:
      entry = dict(baseline)
      entry[knob] = value
      entry["knob"] = knob
      if any(all(entry[k] == other[k] for k in ("patience", "weight_decay", "dropconnect")) for other in grid):
        continue
      grid.append(entry)
  return grid


def settled_probability(row, warmup_epochs, patience, loss_precision):
  """`P(train change over +patience < loss_precision / 2)` at the FIRING epoch, recomputed post hoc.

  The measurement stores every epoch's train/validation mean and SEM and the window at that epoch, so
  the final round is the trailing block whose window does not change, and the fit is the same
  `bayesian_trend` over its post-warmup epochs that the loop ran. Returns `None` when the round holds
  fewer than the three post-warmup epochs the posterior needs (a `capped` cell can end that way).
  """
  history = row["per_epoch"]
  window = np.asarray(history["window"], np.int64)
  changes = np.nonzero(np.diff(window) != 0)[0]
  round_start = int(changes[-1] + 1) if changes.size > 0 else 0
  first = round_start + int(warmup_epochs)
  train = np.asarray(history["train"][first:], np.float64)
  train_sem = np.asarray(history["train_sem"][first:], np.float64)
  validation = np.asarray(history["val"][first:], np.float64)
  if train.shape[0] < 3:
    return None, None, round_start
  prior_sigma = max(float(train[0]), float(validation[0])) / 3.0
  train_mean, train_cov = bayesian_trend(train, train_sem, prior_sigma)
  probability = float(probability_change_below(train_mean, train_cov, int(patience), 0.5 * float(loss_precision)))
  slope_per_epoch = float(train_mean[1])
  return probability, slope_per_epoch, round_start


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a GROWTH run config, e.g. =linear_d2n3_growth")
  parser.add_argument("--trajectory", default=TRAJECTORY, help="results.json whose designs are scored")
  parser.add_argument(
    "--design-ranks", type=int, nargs="+", default=[0, 5, 10, 15, 19],
    help="ORDER STATISTICS of the trajectory's recorded loss; rank 10 of 20 is the median design"
  )
  parser.add_argument(
    "--seeds", type=int, nargs="+", default=[1, 2], help="repetitions; every arm and setting is PAIRED on these"
  )
  parser.add_argument("--patience", type=int, nargs="+", default=[4, 8, 16, 32, 64])
  parser.add_argument("--weight-decay", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
  parser.add_argument(
    "--dropconnect", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2],
    help="0 means the layer draws no mask at all (a strict no-op), which is the config's own state"
  )
  parser.add_argument(
    "--meta-knobs", nargs="+", default=["patience"], choices=("patience", "weight_decay", "dropconnect"),
    help="which sweeps `meta` is run over; `from_scratch` is always run over all of them"
  )
  parser.add_argument("--n-past", type=int, default=9, help="historical designs, drawn UNIFORMLY in the scaled cube")
  parser.add_argument("--history-rows", type=int, default=6144, help="TRAIN rows per historical design")
  parser.add_argument("--n-test", type=int, default=32768, help="test rows at the EXACT scored design")
  parser.add_argument("--budget", type=int, default=None, help="override `training.budget` (it sizes the pools)")
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument("--n0", type=int, default=None, help="override `training.n0` (SMOKE TEST ONLY)")
  parser.add_argument("--n-increment", type=int, default=None, help="override `training.n_increment` (SMOKE TEST ONLY)")
  parser.add_argument("--iteration-limit", type=int, default=None, help="override `training.iteration_limit` (SMOKE TEST ONLY)")
  parser.add_argument("--output", default="output/costbias/costbias.json")
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
  for key, value in (("n0", arguments.n0), ("n_increment", arguments.n_increment),
                     ("iteration_limit", arguments.iteration_limit), ("budget", arguments.budget)):
    if value is not None:
      run["training"][key] = int(value)
  if arguments.device is not None:
    run["device"] = arguments.device
  run["plot_per_epoch"] = False
  regressor_key = next(iter(run["regressor"]))
  optimizer_key = next(iter(run["training"]["optimizer"]))

  baseline = {
    "patience": int(run["training"]["patience"]),
    "weight_decay": float(run["training"]["optimizer"][optimizer_key]["weight_decay"]),
    "dropconnect": float(run["regressor"][regressor_key].get("dropconnect") or 0.0),
  }
  grid = settings_grid(arguments.patience, arguments.weight_decay, arguments.dropconnect, baseline)

  designs = scored_designs(arguments.trajectory, arguments.design_ranks)
  test_index = shuffled_event_index(detector.size(), arguments.n_test, TEST_INDEX_SEED, name="test events")
  floors, optimal_rows = {}, {}
  for rank, index, recorded, x_scaled in designs:
    floors[rank] = float(detector.bayes_risk(detector.flatten_design(detector.to_nominal(x_scaled[None, :]))[0]))
    optimal_rows[rank] = optimal_row_losses(detector, x_scaled, test_index)

  history_total = int(arguments.n_past) * int(arguments.history_rows)
  charged = round(history_total / (1.0 - float(run["training"]["val_fraction"])))

  print(f"config {name} | trajectory {arguments.trajectory}", flush=True)
  for rank, index, recorded, _ in designs:
    empirical = optimal_rows[rank]
    sem = float(np.std(empirical, ddof=1) / np.sqrt(empirical.size))
    print(
      f"  rank {rank:>3} -> trajectory index {index:>2}  recorded {recorded:.5f}  bayes_risk {floors[rank]:.5f}  "
      f"optimal predictor on the test pool {float(np.mean(empirical)):.5f} +- {sem:.5f}", flush=True
    )
  print(f"baseline {baseline} | {len(grid)} settings | history {history_total} train rows (charged {charged})", flush=True)

  rows = []
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)["rows"]
  done = {(r["arm"], r["seed"], r["setting"], r["rank"]) for r in rows}
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)

  for seed in arguments.seeds:
    history_rng = np.random.default_rng([int(seed), 20260818])
    past_designs = uniform_designs(arguments.n_past, int(detector.design_dim()), history_rng)
    for entry in grid:
      label = f"{entry['knob']}={entry[entry['knob']]}"
      arms = ["from_scratch"]
      if entry["knob"] in arguments.meta_knobs:
        arms.append("meta")
      for arm in arms:
        pending = [d for d in designs if (arm, int(seed), label, d[0]) not in done]
        if len(pending) == 0:
          print(f"skip {arm} s{seed} {label} (all designs measured)", flush=True)
          continue

        cell_run = copy.deepcopy(run)
        cell_run["training"]["patience"] = int(entry["patience"])
        cell_run["training"]["optimizer"][optimizer_key]["weight_decay"] = float(entry["weight_decay"])
        if entry["dropconnect"] > 0.0:
          cell_run["regressor"][regressor_key]["dropconnect"] = float(entry["dropconnect"])
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
            f"probe_cost_bias: train pool {trainer.train_pool.capacity} < history {history_total} + "
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
            raise SystemExit(f"probe_cost_bias: laid {trainer.train_pool.current} rows, expected {history_total}")
        else:
          trainer.train_pool.current = history_total
        print(
          f"=== s{seed} {label} arm {arm} | patience {trainer.patience} weight_decay {entry['weight_decay']:g} "
          f"dropconnect {entry['dropconnect']:g} | history {trainer.train_pool.current} in "
          f"{time.time() - started:.1f} s | steps/epoch {trainer.steps_per_epoch}", flush=True
        )

        for rank, index, recorded, x_scaled in pending:
          test_pool.current = 0
          fill(detector, test_pool, test_index, np.broadcast_to(x_scaled[None, :], (arguments.n_test, x_scaled.size)).copy())
          outcome = measure(trainer, detector, test_pool, eval_test, arguments.n_test, x_scaled, int(seed), history_total, None)
          if outcome is None:
            print(f"  rank {rank}: pool exhausted before the first round -- NOT a measurement", flush=True)
            continue
          row, network_rows = outcome
          probability, slope, round_start = settled_probability(
            row, trainer.warmup_epochs, trainer.patience, trainer.loss_precision
          )
          floor = floors[rank]
          paired = np.asarray(network_rows, np.float64) - optimal_rows[rank]
          excess_paired = float(np.mean(paired))
          excess_paired_sem = float(np.std(paired, ddof=1) / np.sqrt(paired.size))
          row.update({
            "excess_paired": excess_paired,
            "excess_paired_sem": excess_paired_sem,
            "optimal_test": float(np.mean(optimal_rows[rank])),
            "arm": arm,
            "setting": label,
            "knob": entry["knob"],
            "patience": int(entry["patience"]),
            "weight_decay": float(entry["weight_decay"]),
            "dropconnect": float(entry["dropconnect"]),
            "rank": int(rank),
            "design_index": int(index),
            "trajectory_loss": float(recorded),
            "x_scaled": [float(v) for v in x_scaled],
            "bayes_risk": floor,
            "excess_reported": float(row["val"]) - floor,
            "excess_test": float(row["test"]) - floor,
            "p_settled_at_stop": probability,
            "train_slope_at_stop": slope,
            "final_round_start": int(round_start),
            "history_rows": history_total,
            "history_calls_charged": int(charged) if arm == "meta" else 0,
          })
          row.pop("per_epoch_kept", None)
          rows.append(row)
          with open(arguments.output, "w") as f:
            json.dump({"rows": rows, "baseline": baseline, "n_test": int(arguments.n_test)}, f, indent=1)
          settled = "n/a" if probability is None else f"{probability:.3f}"
          print(
            f"  rank {rank:>3} (bayes {floor:.4f}): window {row['window']:>6} calls {row['calls_train_val']:>6} "
            f"epochs {row['n_epochs']:>4} val {row['val']:.4f} test {row['test']:.4f} "
            f"excess {row['excess_paired']:+.5f} +- {row['excess_paired_sem']:.5f} "
            f"diff {row['diff']:.4f} err {row['err']:.4f} "
            f"P(settled) {settled} [{row['status']}] {row['wall_s']:.0f} s", flush=True
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
