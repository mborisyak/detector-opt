#!/usr/bin/env python3
"""H7/H7a: does the restart machinery do anything for `meta`, or is it inert?

    python scripts/probe_rewind.py =linear_d2n3_growth --seeds 1 2 \
        --param-mix 0.0 0.25 1.0 --reinit-on-grow --output output/criterion-bench/rewind.json

THE RULE IS UNIFORM; ONLY ITS DISPLACEMENT DIFFERS (`docs/criterion-investigation.md`, H7/H7a). At a
data addition `design.py` sets `params = q + (1 - mix) * (p - q)` with `q = initial_params`, the
network `_init_design_network` returned at the top of `train`: the network is pulled a fraction of the
way back toward ITS OWN starting point for the current design. The starting point is what DEFINES an
arm, so the rule is the same in both and nothing here is written wrong. What differs is the SIZE of
what it acts on:

  `from_scratch`  `fresh_design_network` draws a NEW random network per design, so `q` is a fresh
                  draw and `p - q` is everything the design learned. LARGE.
  `meta`          `ContinualTrainer._init_design_network` returns `self._running`, the carried
                  already-trained network, so `p - q` is only what THIS design added on top of an
                  existing fit. SMALLER, by an amount this probe measures rather than assumes.

This is a measurement of CONSEQUENCE, not a criticism: the same rule does different amounts of work in
each arm because the arms differ in exactly the quantity it is proportional to.

WHY THE CONSEQUENCE COULD MATTER. A network that moves little within a design has a flat train curve
AND a flat validation curve, so a small `gap`, so it clears `gap + err <= loss_precision` at a SMALLER
window -- the anchor observation with the opposite meaning. Whether that is what happens is what is
measured here; for `meta` a restart also never reaches back past the CURRENT design's start, so its
starting point ratchets forward across a trajectory.

H7a, AND THE SHARP PREDICTION IT GIVES. BOTH restart knobs are expressed in terms of
`_init_design_network`, and the continual strategy defines a "new network" as the one it already
carries -- its docstring says so. So `reinit_on_grow`, which calls
`_init_design_network(init_seq.spawn(1)[0], None)` at every addition, returns the SAME carried params
and the SAME buffer state for `meta`, and its only surviving effect there is the optimiser reset. For
`meta`, `reinit_on_grow: true` and `rewind: 0.0` are therefore almost the same computation and must
give near-identical window, epochs and excess; for `from_scratch` they must differ substantially. That
is a factual consequence of what each arm means by "the network this design started from", not a
defect in either knob.

⚠️ `design.py` RAISES `ValueError` IF BOTH are set, so the `reinit_on_grow` cell carries
`rewind: 0.0`. `rewind: 0.25` is a settled default; this measures whether it FUNCTIONS and
changes no shipped config.

THE SPREAD ACROSS THE FOUR CELLS WITHIN AN ARM is the measure of how much the restart machinery does
for that arm, and it is reported as a range per arm rather than as four separate numbers.

⚠️ WHY THIS PROBE EXISTS SEPARATELY FROM `probe_bias_cross.py`. That study drives
`probe_replay_mix.measure`, which builds a FRESH network in both arms by construction, so its `meta`
differs from `from_scratch` in REPLAY ALONE and its `initial_params` is a fresh draw in both arms. It
therefore cannot see H7 at all. This probe calls `trainer.train` -- `_DesignBase.train` itself -- so
`_init_design_network` resolves per arm exactly as it does in a campaign, and `meta` carries one
network across the whole design sequence.

THE SEQUENCE IS THE CONTROL. Each cell scores `--n-designs` CONSECUTIVE mid-trajectory designs in
order. At the FIRST design `meta`'s carried network has never been trained, so its `q` is as fresh as
`from_scratch`'s and the two arms should agree; from the second design on `q` is a trained network. A
collapse in `meta`'s displacement between position 1 and position 2, with `from_scratch` flat across
positions, is the effect H7 predicts, measured against the arm's own first design rather than against
the other arm.

WHAT IS MEASURED PER DESIGN. `||p - q|| / ||p||`, the RELATIVE within-design displacement, with `q`
reconstructed the way `train` builds it -- `trainer._running` for `meta`, and for `from_scratch` a
re-draw of `fresh_design_network` under the SAME seed split `train` uses, which is deterministic and
therefore exact. `p` is `TrainResult.params`. ⚠️ It is measured at the design's END; the rewind acts
on the same quantity at each ADDITION, where it is smaller still, so this is an UPPER BOUND on what
the rewind has to work with.

`design.py` IS NOT MODIFIED AND NOT SUBCLASSED. Nothing here instruments the loop; the seed split is
reproduced from the outside.

THE HISTORY. `meta` without a replay pool takes `_sample_indices`' `start == 0` branch and IS
`from_scratch`, so every design BEFORE the scored block is laid into the train pool first, exactly as
`probe_criterion.py` does it -- the trajectory's own prefix, at `--history-rows` train rows each.

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
import jax.numpy as jnp

import detopt
import detopt.detector
from detopt.nn.trainer import ContinualTrainer, DesignTrainer, FixedWindowTrainer
from detopt.nn.trainer.common import fresh_design_network
from detopt.nn.trainer.fixed_window import PLATEAU_MESSAGE

from detopt.utils.events import shuffled_event_index
from detopt.utils.pools import Pool

from probe_criterion import fill, floor_at, trajectory_designs
from probe_scatter import build_test_eval
from probe_scatter import fill as fill_rows

ARMS = {"from_scratch": DesignTrainer, "continue": DesignTrainer, "closest": DesignTrainer, "meta": ContinualTrainer}

# `continue` and `closest` are NOT separate trainers -- both are `DesignTrainer` plus an `init_params`
# warm start that `bo.py` supplies from an EARLIER design's trained network. A mid-BO probe has to
# produce that network itself, which is what `warm_start_params` does.

# THE NO-GROWTH CONTROL (user, 2026-08-30). `FixedWindowTrainer` presents the SAME EVENTS the growth
# cell ended with, all at once: same window, same validation window, and `train_offset`/`val_offset`
# set to the pool cursors the scored design opened at, so it draws that design's own slice of the
# run's event index rather than a different sample. The only difference is the PATH to the dataset.
#
# THE GROWTH TRAINER CANNOT SERVE AS THIS CONTROL -- its procedure is entitled to ask for more data,
# and at a fixed window that request is a hard error. `FixedWindowTrainer` has no data-addition
# branch, so the training-loss plateau is its only exit and the gap rule never fires.
#
# It is run PER GROWTH CELL, at that cell's own final window, so every `rewind` setting gets its
# own matched control rather than sharing one.
CONTROL_ARM = "fixed_window"

# TRAINING KNOBS ONLY SOME TRAINERS ACCEPT, and which arms own them -- the same table `scripts/bo.py`
# keeps, for the same reason: a config written for a multi-arm campaign carries EVERY arm's knobs, and
# a trainer handed one it does not take dies with `TypeError: __init__() got an unexpected keyword
# argument` before the first design. `config/ship_intersect_prec1e2_w2x_2m.yaml` carries
# `replay_weight`, which `DesignTrainer` does not accept, so the `from_scratch` arm cannot be built
# from it unfiltered. Dropped LOUDLY: a setting that vanishes without a line in the log is how a probe
# ends up measuring something else.
STRATEGY_KNOBS = {"replay_weight": ("meta", ), "current_replay_ratio": (), "alpha": (), "random_weight": (), }


def training_for(arm, training):
  """``training`` with the knobs ``arm``'s trainer does not accept removed."""
  out = dict(training)
  for knob, owners in STRATEGY_KNOBS.items():
    if knob in out and arm not in owners:
      print(
        f"[config] dropped `training.{knob}` for arm {arm}: it applies to "
        f"{'/'.join(owners) if owners else 'no arm here'}", flush=True
      )
      out.pop(knob)
  return out


TEST_INDEX_SEED = 20260818


def norm(tree):
  """Euclidean norm of a parameter pytree."""
  return float(np.sqrt(sum(float(jnp.sum(jnp.square(leaf))) for leaf in jax.tree.leaves(tree))))


def relative_displacement(after, before):
  """`||p - q|| / ||p||` -- how much of the network at the design's end was learned inside it."""
  difference = jax.tree.map(lambda a, b: a - b, after, before)
  scale = norm(after)
  return float(norm(difference) / scale) if scale > 0.0 else float("nan")


def start_params(trainer, arm, seed):
  """The `q` that `_DesignBase.train` will use, WITHOUT calling it.

  `train` splits `SeedSequence(seed)` into `(init_seq, training_seq)` and passes `init_seq` to
  `_init_design_network`. For `meta` that hook ignores the sequence and returns the carried network;
  for `from_scratch` it is `fresh_design_network`, which is deterministic in the sequence, so
  re-running it here reproduces the same draw exactly.
  """
  init_seq, _training_seq = np.random.SeedSequence(int(seed)).spawn(2)
  if arm == "meta":
    return trainer._running[0]
  return fresh_design_network(trainer, init_seq, None)[0]


def warm_start_params(arm, cell_run, detector, history, history_rows, design_scaled, seed):
  """The `init_params` an earlier design would have handed this one, for `continue` / `closest`.

  ⚠️ TRAINED ON A SEPARATE TRAINER, deliberately. Training the donor on the SCORED trainer would
  consume its pools and shift the scored design's window offsets, making the arm incomparable to
  `from_scratch` and `meta`, which consume nothing extra. A second instance with the same config and
  the same history leaves the scored trainer's pool untouched.

  `continue` warm-starts from the IMMEDIATELY PRECEDING design; `closest` from the history design
  nearest the scored one in scaled space, mirroring `bo.py`'s argmin over earlier proposals.
  """
  from detopt.utils.config import optimizer as make_optimizer, resolve_device

  if arm == "continue":
    donor, note = history[-1], "preceding"
  else:
    distances = [float(np.linalg.norm(np.asarray(h, np.float64) - np.asarray(design_scaled, np.float64))) for h in history]
    pick = int(np.argmin(distances))
    donor, note = history[pick], f"closest history {pick} (dist={distances[pick]:.3f})"
  print(f"  [warm-start] {arm}: donor = {note}", flush=True)
  donor_trainer = DesignTrainer(
    detector, regressor_config=cell_run["regressor"], optimizer=make_optimizer(cell_run["training"]["optimizer"]),
    device=resolve_device(cell_run.get("device")), checkpoint_dir=None, seed=seed, **{
      k: v
      for k, v in training_for("from_scratch", cell_run["training"]).items() if k != "optimizer"
    }
  )
  donor_trainer.train_pool.current = 0
  for design in history:
    fill(detector, donor_trainer.train_pool, donor_trainer._train_index, design, int(history_rows))
  result = donor_trainer.train(donor, seed * 100003 - 1)
  if result is None:
    raise SystemExit("probe_rewind: the donor design exhausted its pool; raise --budget")
  return result.params


def per_epoch(snapshot):
  """The trainer's snapshot in the layout `report_bias_cross.py` reads."""
  return {
    "train": [round(float(v), 6) for v in snapshot["train_loss_per_epoch"]],
    "val": [round(float(v), 6) for v in snapshot["val_loss_per_epoch"]],
    "train_sem": [round(float(v), 6) for v in snapshot["train_sem_per_epoch"]],
    "val_sem": [round(float(v), 6) for v in snapshot["val_sem_per_epoch"]],
    "window": [int(v) for v in snapshot["train_budget_per_epoch"]],
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a GROWTH run config, e.g. =linear_d2n3_growth")
  parser.add_argument("--trajectory", default="output/linear-d2n3-fixed/1244111331/from_scratch/results.json")
  parser.add_argument("--n-designs", type=int, default=5, help="CONSECUTIVE designs scored per cell, in order")
  parser.add_argument("--middle", type=int, default=None, help="explicit start index (default: the positional middle)")
  parser.add_argument("--history-rows", type=int, default=6144, help="TRAIN rows per historical design")
  parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
  parser.add_argument("--param-mix", type=float, nargs="+", default=[0.0, 0.25, 1.0])
  parser.add_argument(
    "--reinit-on-grow", action="store_true",
    help="add one more cell per arm with `reinit_on_grow: true` and `rewind: 0.0` (design.py "
    "rejects the two together)"
  )
  parser.add_argument("--n-test", type=int, default=32768, help="HELD-OUT rows at the EXACT scored design")
  parser.add_argument("--arms", nargs="+", default=["from_scratch", "meta"], choices=sorted(ARMS))
  parser.add_argument("--budget", type=int, default=None, help="override `training.budget` (it sizes the pools)")
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument(
    "--shrink-perturb", nargs=2, type=float, default=None, metavar=("SHRINK", "PARAM_NOISE"),
    help="add one more cell per arm running SHRINK-AND-PERTURB instead of the rewind: at each data "
    "addition `params <- shrink * params + param_noise * fresh_draw` with `rewind = 0`. Ash & "
    "Adams use 0.6 0.01 for online learning; note they apply it ONCE PER ROUND while this "
    "trainer fires at EVERY data addition, so the AR(1) memory horizon is 1/(1-shrink) additions"
  )
  parser.add_argument("--output", default="output/criterion-bench/rewind.json")
  parser.add_argument("--resume", action="store_true", help="keep cells already in --output and skip them")
  parser.add_argument(
    "--fixed-window-control", action="store_true",
    help="after each growth cell, train a FixedWindowTrainer on THAT cell's final window with no growth, "
    "at the same design and the same event slice, scored on the same held-out test pool"
  )
  parser.add_argument("--control-max-epochs", type=int, default=512, help="epoch cap for the no-growth control")
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

  cells = [{
    "label": f"rewind={value:g}",
    "rewind": float(value),
    "reinit_on_grow": False
  } for value in arguments.rewind]
  if arguments.reinit_on_grow:
    cells.append({"label": "reinit_on_grow", "rewind": 0.0, "reinit_on_grow": True})
  if arguments.shrink_perturb is not None:
    shrink, noise = arguments.shrink_perturb
    cells.append({
      "label": f"shrink={shrink:g},param_noise={noise:g}",
      "rewind": 0.0,
      "shrink": float(shrink),
      "param_noise": float(noise),
    })

  scored, history, window = trajectory_designs(arguments.trajectory, arguments.n_designs, arguments.middle)
  test_index = shuffled_event_index(detector.size(), arguments.n_test, TEST_INDEX_SEED, name="test events")
  history_total = len(history) * int(arguments.history_rows)
  print(
    f"config {name} | trajectory {arguments.trajectory}: designs {window[0]}..{window[1] - 1} scored "
    f"in order, {len(history)} designs of history x {arguments.history_rows} rows = {history_total}", flush=True
  )
  print(
    f"cross: {len(arguments.arms)} arms x {len(cells)} restart cells "
    f"({', '.join(c['label'] for c in cells)}) x {len(scored)} designs x {len(arguments.seeds)} seeds "
    f"| held-out {arguments.n_test} rows | closed-form floor {has_floor}", flush=True
  )

  rows = []
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)["rows"]
  done = {(r["arm"], r["seed"], r["cell"]) for r in rows}
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)

  for seed in arguments.seeds:
    for cell in cells:
      for arm in arguments.arms:
        if (arm, int(seed), cell["label"]) in done:
          print(f"skip {arm} s{seed} {cell['label']} (already measured)", flush=True)
          continue

        cell_run = copy.deepcopy(run)
        cell_run["training"]["rewind"] = float(cell["rewind"])
        for knob in ("shrink", "param_noise"):
          if knob in cell:
            cell_run["training"][knob] = float(cell[knob])
        cell_run["training"]["reinit_on_grow"] = bool(cell["reinit_on_grow"])
        trainer = ARMS[arm](
          detector, regressor_config=cell_run["regressor"], optimizer=make_optimizer(cell_run["training"]["optimizer"]),
          device=resolve_device(cell_run.get("device")), checkpoint_dir=None, seed=seed, **{
            k: v
            for k, v in training_for(arm, cell_run["training"]).items() if k != "optimizer"
          }
        )
        eval_test = build_test_eval(trainer, detector, arguments.n_test)
        test_pool = Pool(arguments.n_test, trainer.train_pool.specs, trainer.device)
        _definition, _params, buffer_state = trainer._build_regressor(int(seed))
        buffer_state = jax.device_put(buffer_state, trainer.device)
        started = time.time()
        trainer.train_pool.current = 0
        for design in history:
          fill(detector, trainer.train_pool, trainer._train_index, design, int(arguments.history_rows))
        if trainer.train_pool.current != history_total:
          raise SystemExit(f"probe_rewind: laid {trainer.train_pool.current} rows, expected {history_total}")
        print(
          f"=== s{seed} {cell['label']} arm {arm} | history {trainer.train_pool.current} rows in "
          f"{time.time() - started:.1f} s | steps/epoch {trainer.steps_per_epoch}", flush=True
        )
        first_network = trainer._running[0] if arm == "meta" else None

        for position, (index, design_scaled, recorded) in enumerate(scored, start=1):
          design_seed = int(seed) * 100003 + index
          before = start_params(trainer, arm, design_seed)
          latest = {}

          def on_epoch(snapshot, _latest=latest):
            if snapshot["train_loss_per_epoch"].size >= _latest.get("n", 0):
              _latest["n"] = snapshot["train_loss_per_epoch"].size
              _latest["snapshot"] = snapshot

          # The pool cursors this design OPENS at, and (after training) what it added. The control
          # needs both: the offsets pick that design's own slice of the event index, the counts size
          # its window to the growth run's own final spend.
          w0_counts = {"train_start": trainer.train_pool.current, "val_start": trainer.val_pool.current}
          began = time.time()
          capped = False
          try:
            warm = None
            if arm in ("continue", "closest"):
              warm = warm_start_params(arm, cell_run, detector, history, arguments.history_rows, design_scaled, seed)
            result = trainer.train(design_scaled, design_seed, init_params=warm, on_epoch=on_epoch)
          except RuntimeError as error:
            # THE ONE expected failure: the window cap. `design.py` raises it with this exact opening
            # when a design reaches `iteration_limit` without meeting the bar, and `rewind = 0` --
            # pure carry -- is documented to hit it (that file records carry capping 6/9). It is an
            # OUTCOME of the cell, recorded as `capped` with its per-epoch history intact, and it is
            # NOT a converged measurement: `val`, the held-out loss and every excess stay None. Any
            # other RuntimeError is re-raised untouched.
            if not str(error).startswith("design did not reach precision within iteration_limit"):
              raise
            capped = True
            result = None
            print(f"  position {position} design {index}: CAPPED at iteration_limit -- {error}", flush=True)
          if result is None and not capped:
            print(f"  position {position} design {index}: POOL EXHAUSTED -- not a measurement", flush=True)
            continue
          w0_counts["train_added"] = trainer.train_pool.current - w0_counts["train_start"]
          w0_counts["val_added"] = trainer.val_pool.current - w0_counts["val_start"]
          if "snapshot" not in latest:
            raise SystemExit("probe_rewind: no per-epoch snapshot was captured")
          if not capped:
            test_pool.current = 0
            fill_rows(
              detector, test_pool, test_index,
              np.broadcast_to(design_scaled[None, :], (arguments.n_test, design_scaled.size)).copy()
            )
            test_eval = eval_test(result.params, buffer_state, test_pool.buffers(), jnp.int32(0))
            held_out = float(np.mean(np.asarray(test_eval, np.float64)[:arguments.n_test]))
          else:
            held_out = None
          history_arrays = per_epoch(latest["snapshot"])
          gap = abs(history_arrays["val"][-1] - history_arrays["train"][-1])
          err = float(np.hypot(history_arrays["train_sem"][-1], history_arrays["val_sem"][-1]))
          floor = floor_at(detector, design_scaled) if has_floor else None
          row = {
            "arm": arm,
            "seed": int(seed),
            "cell": cell["label"],
            "rewind": float(cell["rewind"]),
            "reinit_on_grow": bool(cell["reinit_on_grow"]),
            "position": int(position),
            "design_index": int(index),
            "recorded_loss": recorded,
            "bayes_risk": floor,
            "val": None if capped else float(result.objective_loss),
            "objective_std": None if capped else float(result.objective_std),
            "test": held_out,
            "excess_reported": None if (floor is None or capped) else float(result.objective_loss) - floor,
            "excess_test": None if (floor is None or capped) else held_out - floor,
            "calls_train_val": None if capped else int(result.spent),
            "window": int(history_arrays["window"][-1]),
            "n_epochs": len(history_arrays["train"]),
            "n_rounds": int(len(set(history_arrays["window"]))),
            "train": history_arrays["train"][-1],
            "diff": gap,
            "err": err,
            "slack": gap + err,
            "displacement": None if capped else relative_displacement(result.params, before),
            "norm_before": norm(before),
            "norm_after": None if capped else norm(result.params),
            "drift_from_first": (relative_displacement(result.params, first_network) if arm == "meta" and not capped else None),
            "patience": int(cell_run["training"]["patience"]),
            "dropconnect": float(cell_run["regressor"][next(iter(cell_run["regressor"]))].get("dropconnect") or 0.0),
            "loss_precision": float(cell_run["training"]["loss_precision"]),
            "status": "capped" if capped else "converged",
            "wall_s": time.time() - began,
            "per_epoch": history_arrays,
          }
          rows.append(row)

          if arguments.fixed_window_control and not capped:
            control_began = time.time()
            control = FixedWindowTrainer.from_config(
              detector, cell_run, window=int(w0_counts["train_added"]), val_window=int(w0_counts["val_added"]),
              max_epochs=int(arguments.control_max_epochs), seed=design_seed, train_offset=int(w0_counts["train_start"]),
              val_offset=int(w0_counts["val_start"]),
            )
            control_latest = {}

            def control_on_epoch(snapshot, _l=control_latest):
              if snapshot["train_loss_per_epoch"].size >= _l.get("n", 0):
                _l["n"] = snapshot["train_loss_per_epoch"].size
                _l["snapshot"] = snapshot

            control_capped = False
            try:
              control_result = control.train(design_scaled, design_seed, on_epoch=control_on_epoch)
            except RuntimeError as error:
              # Its ONE expected failure is the plateau cap, spelled by `fixed_window.PLATEAU_MESSAGE`.
              if PLATEAU_MESSAGE not in str(error):
                raise
              control_capped = True
              control_result = None
            if control_result is not None:
              test_pool.current = 0
              fill_rows(
                detector, test_pool, test_index,
                np.broadcast_to(design_scaled[None, :], (arguments.n_test, design_scaled.size)).copy()
              )
              control_test = float(
                np.mean(
                  np.asarray(eval_test(control_result.params, buffer_state, test_pool.buffers(), jnp.int32(0)),
                             np.float64)[:arguments.n_test]
                )
              )
              control_hist = per_epoch(control_latest["snapshot"]) if "snapshot" in control_latest else None
              rows.append({
                **{
                  k: row[k]
                  for k in (
                    "seed", "cell", "rewind", "reinit_on_grow", "position", "design_index", "recorded_loss", "bayes_risk", "loss_precision"
                  )
                }, "arm": f"{arm}:{CONTROL_ARM}",
                "control_for": arm,
                "val": float(control_result.objective_loss),
                "objective_std": float(control_result.objective_std),
                "test": control_test,
                "excess_test": None if floor is None else control_test - floor,
                "calls_train_val": int(control_result.spent),
                "window": int(w0_counts["train_added"]),
                "n_epochs": None if control_hist is None else len(control_hist["train"]),
                "n_rounds": 1,
                "growth_window": row["window"],
                "growth_test": row["test"],
                "status": "converged",
                "wall_s": time.time() - control_began,
                "per_epoch": control_hist,
              })
              print(
                f"    control (no growth) at window {w0_counts['train_added']}: test {control_test:.4f} "
                f"vs growth {row['test']:.4f}  ({control_test - row['test']:+.4f})  "
                f"epochs {rows[-1]['n_epochs']} in {rows[-1]['wall_s']:.0f} s", flush=True
              )
            else:
              print(
                f"    control (no growth): PLATEAU CAP at window {w0_counts['train_added']} -- not a measurement", flush=True
              )
            for pool in (control.train_pool, control.val_pool):
              for buffer in jax.tree.leaves(pool.buffers()):
                buffer.delete()
            del control
            gc.collect()

          with open(arguments.output, "w") as f:
            json.dump({"rows": rows, "config": name, "trajectory": arguments.trajectory}, f, indent=1)
          if capped:
            print(
              f"  position {position} design {index}: [capped] window {row['window']:>6} "
              f"epochs {row['n_epochs']:>4} gap {gap:.4f} err {err:.4f} in {row['wall_s']:.0f} s", flush=True
            )
          else:
            excess = "" if floor is None else f"excess {row['excess_reported']:+.5f} "
            print(
              f"  position {position} design {index}: window {row['window']:>6} calls {row['calls_train_val']:>6} "
              f"epochs {row['n_epochs']:>4} val {row['val']:.4f} test {held_out:.4f} {excess}gap {gap:.4f} "
              f"err {err:.4f} ||p-q||/||p|| {row['displacement']:.4f} in {row['wall_s']:.0f} s", flush=True
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
