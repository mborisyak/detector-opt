"""What does the REWIND cost and buy? One mid-BO design trained from scratch at several ``rewind``.

    python scripts/probe_rewind.py --run <cell> --design 4 --param-mix 0.1 --seed 0 --output <dir>

``rewind`` fires at every data addition: ``params <- (1 - lambda) * current + lambda * INITIAL``,
where INITIAL is the network the run started from, and the optimiser moments are reset with it. The
two ENDPOINTS are measured (pure carry capped 6/9 designs, rebuild 0/9); what this probe records is
the INTERIOR, at one fixed design, so lambda is the only systematic difference between cells.

ONE DESIGN, ONE ARM. It trains the ``per_design`` (from_scratch) trainer on a single design taken from
a completed run's trajectory. No BO, no warm start, no second design -- the question is about the
growth procedure, not about transfer.

WHAT IT RECORDS. The whole curve, not a summary: ``train_loss_per_epoch``, ``val_loss_per_epoch``,
their standard errors, and ``train_budget_per_epoch`` -- the window after every DATA INJECTION, which
is what says how much data the exit test demanded and when. Plus the final objective, the spend split
train/validation, and whether the round converged or CAPPED.

THE CONTROL (``--control``). After the growth run, the SAME design is trained on the SAME dataset with
no growth at all: :class:`FixedWindowTrainer` draws the growth run's own final spend up front and trains
until the TRAINING LOSS PLATEAUS, by the growth trainer's own test -- ``P(train change over +patience <
loss_precision / 2) > 0.9`` on the Bayesian trend of the post-warmup training history. The two runs then
differ only in the path taken to that dataset. Every cell carries its own control, so the comparison is
within a cell: same design, same trial seed, same events.

WHY NOT THE GROWTH TRAINER AT A FIXED WINDOW. That was the first construction (``n0 = n_increment =
iteration_limit = W``, so nothing can be added) and it is unsound. The procedure is entitled to ask for
more data, and where there is none that request is a hard error: on 2026-08-22 it fired in 2 of 9 cells
at epoch 123, with the validation loss still falling at 1.3e-3 per epoch, and a mid-descent number was
recorded as the control. The fixed-window trainer has no data-addition branch, so the plateau is its
only exit.

IT IS THE SAME EVENTS, not merely the same count. The event index is
``shuffled_event_index(detector size, seed, generations)``, the budget is the run's own so the shuffle is
unchanged, and the growth run fills sequentially from position 0 -- so the control draws exactly
``index[:spent_train]`` and ``val_index[:spent_val]``, the growth run's own spend.

THE STEP COUNTS ARE NOT EQUAL, AND ARE NOT MATCHED. An epoch is ``iteration_limit // batch`` steps, so a
growth run's epoch is its whole per-design limit while the control's is its window. Both runs report
``epochs`` and ``steps``; the difference is a property of the growth procedure and is reported, not
normalised away.

⚠️ ``--param-mix 0.0`` IS A VALID SETTING HERE and must be passed explicitly, because 0.0 is also the
code default -- the no-rewind condition is the thing under test, not an omission. The override is
applied on ``is not None``, never on truthiness, so 0.0 survives it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detopt.detector  # noqa: E402
from detopt.nn.trainer.design import DesignTrainer  # noqa: E402
from detopt.nn.trainer.fixed_window import PLATEAU_MESSAGE, FixedWindowTrainer  # noqa: E402

CAP_MESSAGE = "did not reach precision within iteration_limit"


def control_key(arguments):
  """`control` for the same-events 1x control, `control_x<N>` / `..._independent` for the variants, so one
    file can hold several controls without one overwriting another."""
  key = "control" if arguments.control_scale == 1 else f"control_x{arguments.control_scale}"
  return key + ("_independent" if arguments.control_independent else "")


def run_control(detector, config, design_scaled, *, window, val_window, seed, max_epochs, step, scale=1,
                independent=False):
  """The growth run's own dataset, trained with no growth until the TRAINING LOSS PLATEAUS.

    Same events (``index[:window]`` / ``val_index[:val_window]``, the run's own budget so the shuffle is
    unchanged), same design, same trial seed. Reaching ``max_epochs`` is reported as ``settled: false``
    with its curve, never as a converged number."""
  train_offset, val_offset = (window, val_window) if independent else (0, 0)
  window, val_window = window * int(scale), val_window * int(scale)
  print(
    f"[control] fixed window {window}+{val_window}, no growth, from scratch, max_epochs {max_epochs}, "
    f"offset {train_offset}+{val_offset} "
    f"({'independent of' if independent else 'the same events as'} the growth run)", flush=True
  )
  trainer = FixedWindowTrainer.from_config(
    detector, config, window=window, val_window=val_window, max_epochs=max_epochs, seed=seed,
    train_offset=train_offset, val_offset=val_offset
  )
  latest, began, settled = {}, time.time(), True
  try:
    result = trainer.train(design_scaled, int(seed), on_epoch=lambda s: (latest.clear(), latest.update(s)),
                           step=int(step))
  except RuntimeError as error:
    if PLATEAU_MESSAGE not in str(error):
      raise
    settled, result = False, None
  if len(latest) == 0:
    raise RuntimeError("control: no epoch completed; nothing to record")

  train = np.asarray(latest["train_loss_per_epoch"], np.float64)
  validation = np.asarray(latest["val_loss_per_epoch"], np.float64)
  train_sem = np.asarray(latest["train_sem_per_epoch"], np.float64)
  val_sem = np.asarray(latest["val_sem_per_epoch"], np.float64)
  control = {
    "fixed_window": int(window),
    "fixed_val_window": int(val_window),
    "scale": int(scale),
    "independent": bool(independent),
    "train_offset": int(train_offset),
    "val_offset": int(val_offset),
    "settled": settled,
    "epochs": int(train.shape[0]),
    "steps": int(train.shape[0]) * int(trainer.steps_per_epoch),
    "steps_per_epoch": int(trainer.steps_per_epoch),
    "train": float(train[-1]),
    "validation": float(validation[-1]),
    "gap": abs(float(validation[-1]) - float(train[-1])),
    "err": float(np.hypot(float(train_sem[-1]), float(val_sem[-1]))),
    "objective": (float(result.objective_loss) if settled else 0.5 * (float(train[-1]) + float(validation[-1]))),
    "spent_train": int(trainer.train_pool.current),
    "spent_val": int(trainer.val_pool.current),
    "seconds": time.time() - began,
    "train_loss_per_epoch": train.tolist(),
    "val_loss_per_epoch": validation.tolist(),
    "train_sem_per_epoch": train_sem.tolist(),
    "val_sem_per_epoch": val_sem.tolist(),
    "window_per_epoch": np.asarray(latest["train_budget_per_epoch"], np.int64).tolist(),
  }
  print(
    f"[control] settled={control['settled']}  train={control['train']:.5f}  "
    f"validation={control['validation']:.5f}  objective={control['objective']:.5f}  "
    f"epochs={control['epochs']}  steps={control['steps']}", flush=True
  )
  return control


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, help="a completed cell; supplies the config and the design")
  parser.add_argument("--design", type=int, required=True, help="index of the design to train (mid-BO)")
  parser.add_argument("--param-mix", type=float, required=True, help="the rewind; 0.0 is a valid setting")
  parser.add_argument("--seed", type=int, default=0, help="the trial: network init and data order")
  parser.add_argument("--device", default=None)
  parser.add_argument("--n0", type=int, default=None)
  parser.add_argument("--n-increment", type=int, default=None)
  parser.add_argument("--iteration-limit", type=int, default=None)
  parser.add_argument("--loss-precision", type=float, default=None)
  parser.add_argument("--budget", type=int, default=None,
                      help="override training.budget; a tighter loss_precision needs a pool the run config "
                           "was never sized for (the error falls as 1/sqrt(N), so 10x the precision is 100x "
                           "the data)")
  parser.add_argument(
    "--control", action="store_true", help="after the growth run, ALSO train from scratch on its dataset with no growth"
  )
  parser.add_argument(
    "--control-only", action="store_true",
    help="skip the growth run: read its spend from an existing rewind.json and run the "
    "control alone, rewriting that file"
  )
  parser.add_argument(
    "--learning-rate", type=float, default=None, help="override the optimizer's learning rate"
  )
  parser.add_argument(
    "--features", default=None,
    help="override the set-regressor's features as JSON, e.g. '[[16,16],[16,16]]'; the network's SHAPE is "
    "the only thing this changes"
  )
  parser.add_argument(
    "--control-scale", type=int, default=1,
    help="multiply the control's window by this; 2 asks whether the growth run's margin is the path or "
    "merely more data"
  )
  parser.add_argument(
    "--control-independent", action="store_true",
    help="draw the control's events from a DISJOINT slice of the run's event index instead of the growth "
    "run's own events; the index is a permutation, so a disjoint slice is an independent sample of the "
    "same population"
  )
  parser.add_argument(
    "--control-max-epochs", type=int, default=20000,
    help="guard on the control; reaching it is reported as settled=false, never as a result"
  )
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  with open(os.path.join(arguments.run, "results.json")) as handle:
    payload = json.load(handle)
  results = payload["results"]
  if arguments.design >= len(results):
    sys.exit(f"{arguments.run}: design {arguments.design} needs {arguments.design + 1} rows, run has {len(results)}")

  config = {**payload["config"], "training": {**payload["config"]["training"], "rewind": float(arguments.rewind)}}
  if arguments.device is not None:
    config["device"] = arguments.device
  for key, value in (("n0", arguments.n0), ("n_increment", arguments.n_increment),
                     ("iteration_limit", arguments.iteration_limit), ("loss_precision", arguments.loss_precision),
                     ("budget", arguments.budget)):
    if value is not None:
      print(f"[config] training.{key} {config['training'].get(key)} -> {value}", flush=True)
      config["training"] = {**config["training"], key: value}

  if arguments.features is not None:
    regressor_name = list(config["regressor"])[0]
    features = json.loads(arguments.features)
    print(f"[config] {regressor_name}.features {config['regressor'][regressor_name].get('features')} -> {features}",
          flush=True)
    config = {
      **config,
      "regressor": {regressor_name: {**config["regressor"][regressor_name], "features": features}}
    }

  if arguments.learning_rate is not None:
    optimizer_name = list(config["training"]["optimizer"])[0]
    optimizer_block = config["training"]["optimizer"][optimizer_name]
    print(f"[config] optimizer.{optimizer_name}.learning_rate {optimizer_block['learning_rate']} -> "
          f"{arguments.learning_rate}", flush=True)
    config["training"] = {
      **config["training"],
      "optimizer": {optimizer_name: {**optimizer_block, "learning_rate": arguments.learning_rate}}
    }

  detector = detopt.detector.from_config(config["detector"])
  design_scaled = np.asarray(results[arguments.design]["x_scaled"], np.float32)
  print(
    f"[probe] design {arguments.design} of {len(results)}  rewind={arguments.rewind}  "
    f"trial seed={arguments.seed}  reveal={config['training'].get('reveal')}", flush=True
  )
  print(f"[probe] features {detector.combined_event_shape(config['training'].get('reveal') != 'none')}", flush=True)

  os.makedirs(arguments.output, exist_ok=True)
  target = os.path.join(arguments.output, "rewind.json")

  if arguments.control_only:
    with open(target) as handle:
      report = json.load(handle)
    report[control_key(arguments)] = run_control(
      detector, config, design_scaled, window=int(report["spent_train"]), val_window=int(report["spent_val"]),
      seed=arguments.seed, max_epochs=arguments.control_max_epochs, step=arguments.design, scale=arguments.control_scale,
      independent=arguments.control_independent
    )
    with open(target, "w") as handle:
      json.dump(report, handle, indent=1)
    print(f"wrote {target}", flush=True)
    return

  trainer = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)

  latest, began, capped = {}, time.time(), False
  start = (trainer.train_pool.current, trainer.val_pool.current)
  try:
    result = trainer.train(
      design_scaled, int(arguments.seed), init_params=None, on_epoch=lambda s: (latest.clear(), latest.update(s)),
      step=int(arguments.design)
    )
  except RuntimeError as error:
    if CAP_MESSAGE not in str(error):
      raise
    capped, result = True, None
  if len(latest) == 0:
    raise RuntimeError("no epoch completed; nothing to record")

  train = np.asarray(latest["train_loss_per_epoch"], np.float64)
  validation = np.asarray(latest["val_loss_per_epoch"], np.float64)
  window = np.asarray(latest["train_budget_per_epoch"], np.int64)
  injections = np.flatnonzero(np.diff(window, prepend=window[0] - 1)) if window.size > 0 else np.asarray([])

  report = {
    "run": arguments.run,
    "design": int(arguments.design),
    "rewind": float(arguments.rewind),
    "trial_seed": int(arguments.seed),
    "reveal": config["training"].get("reveal"),
    "loss_precision": config["training"].get("loss_precision"),
    "converged": not capped,
    "objective": (0.5 * (float(train[-1]) + float(validation[-1])) if capped else float(result.objective_loss)),
    "final_window": int(window[-1]) if window.size > 0 else None,
    "epochs": int(train.shape[0]),
    "steps": int(train.shape[0]) * int(trainer.steps_per_epoch),
    "steps_per_epoch": int(trainer.steps_per_epoch),
    "train": float(train[-1]),
    "validation": float(validation[-1]),
    "n_injections": int(injections.size),
    "spent_train": int(trainer.train_pool.current - start[0]),
    "spent_val": int(trainer.val_pool.current - start[1]),
    "seconds": time.time() - began,
    "train_loss_per_epoch": train.tolist(),
    "val_loss_per_epoch": validation.tolist(),
    "train_sem_per_epoch": np.asarray(latest["train_sem_per_epoch"], np.float64).tolist(),
    "val_sem_per_epoch": np.asarray(latest["val_sem_per_epoch"], np.float64).tolist(),
    "window_per_epoch": window.tolist(),
    "injection_epochs": injections.tolist(),
  }

  def flush():
    with open(target, "w") as handle:
      json.dump(report, handle, indent=1)

  flush()
  print(
    f"[done] converged={report['converged']}  objective={report['objective']:.5f}  "
    f"window={report['final_window']}  epochs={report['epochs']}  injections={report['n_injections']}  "
    f"spent={report['spent_train']}+{report['spent_val']}", flush=True
  )
  if arguments.control and report["final_window"] is not None:
    report[control_key(arguments)] = run_control(
      detector, config, design_scaled, window=int(report["spent_train"]), val_window=int(report["spent_val"]),
      seed=arguments.seed, max_epochs=arguments.control_max_epochs, step=arguments.design, scale=arguments.control_scale,
      independent=arguments.control_independent
    )
    flush()

  print(f"wrote {target}", flush=True)


if __name__ == "__main__":
  main()
