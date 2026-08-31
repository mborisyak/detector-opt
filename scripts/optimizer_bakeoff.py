#!/usr/bin/env python3
"""Five optimisers x two learning-rate schedules on ONE design, from a campaign's own state.

WHAT IT IS. A matched bake-off: every arm trains the SAME network on the SAME design against the SAME
event pool, and differs only in the optax transform. The starting point is a real campaign's state
rather than a fresh init -- `meta`'s persistent network and both budget pools are restored from that
run's `trainer.npz`, which is the network at the end of its last completed design and the pools with
their fill cursors -- so what is measured is what a NEXT design would cost from where the campaign
actually stood, not what a toy problem costs from scratch.

THE DESIGN IS THE RUN'S OWN BEST, taken as the `x_scaled` of the minimum-loss row in its trajectory
file. It is trained with the iteration seed the run would have used for its next design (the seed
branch is replayed `n_iterations_completed` times), so the events this bake-off draws are the events
the campaign would have drawn.

THE DATA SCHEDULE IS THIS STUDY'S, not the campaign's: `n0` 8192 and `n_increment` 4096, twice
`enzyme_extremes`' own. The campaign measured 47 growth rounds on this design, and a round costs at
least `warmup_epochs + patience` epochs of demonstrating stillness no matter what the optimiser does --
so at the campaign's schedule most of the wall clock is the schedule, not the thing under test. Halving
the number of rounds is what makes the arms about optimisers. The campaign config is NOT edited; the
values live here (`N0` / `N_INCREMENT`) so this study is reproducible from this file.

WHAT IS HELD FIXED, and why it has to be. `weight_decay` is 1.0e-3 in every arm -- the settled default
-- because an arm without it would differ in REGULARISATION as well as in optimiser and the comparison
would not be about optimisers at all. `optax.nadam` and `optax.amsgrad` carry no decoupled decay, so
they are built as `nadamw` and as an explicit `scale_by_amsgrad -> add_decayed_weights ->
scale_by_learning_rate` chain, which is exactly how optax composes `adamw`. Everything else except the
data schedule above -- the architecture, `batch`, `iteration_limit`, `patience`, `warmup_epochs`,
`rewind`, `loss_precision` and the whole convergence procedure -- comes from the campaign config
untouched.

THE SCHEDULES. `constant` is the config's own 2.5e-4. `hyperbolic` is `alpha / (t / K + 1)`,
implemented here because optax has no such schedule, with `t` the optimiser's step count and
**K = 2 * steps_per_epoch** -- i.e. K is DERIVED from `iteration_limit // batch` rather than fixed, so
the decay is stated in epochs and not in raw steps: the rate halves after two epochs and is at alpha/6
by the tenth, whatever the epoch length. A fixed K would mean something different in every config,
since the same step count is a different fraction of an epoch each time.

EACH ARM IS ITS OWN TRAINER, restored from the same snapshot, so no arm can see another's data: a
restore rewrites both pools AND their cursors, and the appended window is a function of the design and
the seed alone. Arms are written to disk one at a time and an arm whose file exists is skipped, so a
killed job keeps everything it finished and rerunning resumes.

An arm that fails to converge within `iteration_limit` is RECORDED as such and the sweep continues --
that is a result about the optimiser, not an error to abort on.

    srun --gres=shard:1 --cpus-per-task=4 --mem=12000 -u python scripts/optimizer_bakeoff.py \
         =enzyme_extremes output=output/optimizer-bakeoff run=output/enzyme_extremes/1244111331/meta \
         run_seed=1244111331
    ... arms=adan/hyperbolic          # one cell; comma-separate for several
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import optax

import detopt.detector
from detopt.nn.trainer import ContinualTrainer
from detopt.utils.config import resolve_device

LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 1.0e-3
K_EPOCHS = 2
N0 = 8192
N_INCREMENT = 4096
OPTIMIZERS = ("adamw", "adamaxw", "adan", "nadamw", "amsgrad")
SCHEDULES = ("constant", "hyperbolic")


def steps_per_epoch(config):
  """``iteration_limit // batch``, the trainer's own epoch length. It is read from the config rather
    than from a trainer because the schedule has to exist BEFORE the trainer it is passed to."""
  return int(config["training"]["iteration_limit"]) // int(config["training"]["batch"])


def hyperbolic_schedule(alpha, k):
  """``alpha / (t / k + 1)``, the schedule this study adds.

    optax has no inverse-time schedule, so it is written here as the plain callable optax accepts as a
    learning rate: it is handed the optimiser's own step count and returns that step's rate. ``k`` is
    ``K_EPOCHS * steps_per_epoch``, so the decay reads in EPOCHS -- alpha/2 after two of them -- and
    means the same thing under any epoch length."""

  def schedule(count):
    return alpha / (count / float(k) + 1.0)

  return schedule


def learning_rate(schedule, k):
  return LEARNING_RATE if schedule == "constant" else hyperbolic_schedule(LEARNING_RATE, k)


def transform(name, rate):
  """The optax transform for one arm, every one carrying the same decoupled `weight_decay`.

    `nadam` and `amsgrad` have no decoupled-decay form in optax: the first is `nadamw` (which is
    `adamw` with Nesterov momentum), and the second is assembled the way optax assembles `adamw`, so
    the only difference from the `adamw` arm is amsgrad's running MAXIMUM of the second moment."""
  if name == "adamw":
    return optax.adamw(rate, b1=0.9, b2=0.999, eps=1.0e-8, weight_decay=WEIGHT_DECAY)
  if name == "adamaxw":
    return optax.adamaxw(rate, b1=0.9, b2=0.999, eps=1.0e-8, weight_decay=WEIGHT_DECAY)
  if name == "adan":
    return optax.adan(rate, b1=0.98, b2=0.92, b3=0.99, eps=1.0e-8, weight_decay=WEIGHT_DECAY)
  if name == "nadamw":
    return optax.nadamw(rate, b1=0.9, b2=0.999, eps=1.0e-8, weight_decay=WEIGHT_DECAY)
  if name == "amsgrad":
    return optax.chain(
      optax.scale_by_amsgrad(b1=0.9, b2=0.999, eps=1.0e-8),
      optax.add_decayed_weights(WEIGHT_DECAY),
      optax.scale_by_learning_rate(rate),
    )
  raise ValueError(f"unknown optimizer {name!r}, not one of {OPTIMIZERS}")


def trajectory(run_dir):
  """The run's rows, best row and completed-iteration count, from whichever file it left behind."""
  for name in ("results.json", "partial.json"):
    path = os.path.join(run_dir, name)
    if os.path.exists(path):
      with open(path) as f:
        report = json.load(f)
      rows = [r for r in report.get("results", []) if r.get("loss") is not None]
      if len(rows) == 0:
        raise ValueError(f"{path} holds no scored design")
      best = min(rows, key=lambda r: float(r["loss"]))
      return report, rows, best, int(report.get("n_iterations_completed", len(rows)))
  raise FileNotFoundError(f"{run_dir}: neither results.json nor partial.json")


def seeds(run_seed, n_completed):
  """The run's own two seeds for the design that follows ``n_completed``.

    `scripts/bo.py` splits ONE sequence into a network branch and an iteration branch and takes the
    k-th iteration seed by spawning k times, so replaying that here puts this study on exactly the
    stream the campaign was on -- the network seed the trainer must be rebuilt with (`restore` refuses
    any other, since it fixes the train/val split) and the iteration seed the next design would use."""
  network_seq, iteration_seq = np.random.SeedSequence(int(run_seed)).spawn(2)
  iteration_seq.spawn(n_completed)
  return int(network_seq.generate_state(1)[0]), int(iteration_seq.spawn(1)[0].generate_state(1)[0])


def device_memory(device):
  """``(in use, peak)`` bytes on ``device``, or ``(None, None)`` where the backend reports nothing.

    The point of recording it is that an arm's footprint is NOT the network -- the regressor is ~12k
    parameters -- it is the budget-sized event pools plus whatever the peak of restoring them and
    filling them costs. That peak is what decides how many arms fit on one card, and it was guessed at
    once already; it is measured here so it does not have to be guessed at again."""
  if device is None or not hasattr(device, "memory_stats"):
    return None, None
  stats = device.memory_stats()
  if stats is None:
    return None, None
  return int(stats.get("bytes_in_use", 0)), int(stats.get("peak_bytes_in_use", 0))


def run_arm(config, detector, run_dir, network_seed, iteration_seed, design, optimizer, schedule, step):
  """One cell: a trainer restored from the campaign snapshot, trained on ``design`` with this arm."""
  k = K_EPOCHS * steps_per_epoch(config)
  training = {k_: v for k_, v in config["training"].items() if k_ != "optimizer"}
  trainer = ContinualTrainer(
    detector,
    regressor_config=config["regressor"],
    optimizer=transform(optimizer, learning_rate(schedule, k)),
    device=resolve_device(config.get("device")),
    checkpoint_dir=None,
    seed=network_seed,
    **training,
  )
  built = device_memory(trainer.device)
  trainer.restore(os.path.join(run_dir, "trainer.npz"))
  restored = device_memory(trainer.device)
  spent_before = trainer.train_pool.current + trainer.val_pool.current

  history = {}

  def on_epoch(snapshot):
    """Keep the LONGEST snapshot seen. The trainer submits these to a thread pool, so they can land
        out of order, but each one carries the whole history so far -- the longest IS the latest."""
    if len(snapshot["val_loss_per_epoch"]) >= len(history.get("val_loss_per_epoch", ())):
      history.update(snapshot)

  started = time.time()
  status, message, result = "converged", "", None
  try:
    result = trainer.train(design, iteration_seed, on_epoch=on_epoch, step=step)
    if result is None:
      status, message = "budget", "pool exhausted before the design finished"
  except RuntimeError as error:
    # ONLY the precision failure is a RESULT; everything else is an error and must surface as one.
    # This caught a GPU OOM once and wrote it to disk as `unconverged`, which reads as a finding about
    # the optimiser and is not one. `bo.py` makes the same distinction on the same string.
    text = str(error).replace("\n", " ")
    if "did not reach precision within iteration_limit" not in text:
      raise
    status, message = "unconverged", text
  wall = time.time() - started

  in_use, peak = device_memory(trainer.device)
  record = {
    "optimizer": optimizer,
    "device_bytes_after_build": built[0],
    "device_bytes_after_restore": restored[0],
    "device_bytes_in_use": in_use,
    "device_bytes_peak": peak,
    "schedule": schedule,
    "status": status,
    "message": message,
    "wall_s": wall,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "k": k,
    "steps_per_epoch": trainer.iteration_limit // trainer.batch,
    "n0": trainer.n0,
    "n_increment": trainer.n_increment,
    "spent_before": int(spent_before),
    "spent": None if result is None else int(result.spent),
    "objective": None if result is None else float(result.objective_loss),
    "objective_std": None if result is None else float(result.objective_std),
    "n_epochs": int(len(history.get("val_loss_per_epoch", ()))),
    "train_loss_per_epoch": [float(v) for v in history.get("train_loss_per_epoch", ())],
    "val_loss_per_epoch": [float(v) for v in history.get("val_loss_per_epoch", ())],
    "train_sem_per_epoch": [float(v) for v in history.get("train_sem_per_epoch", ())],
    "val_sem_per_epoch": [float(v) for v in history.get("val_sem_per_epoch", ())],
    "train_budget_per_epoch": [int(v) for v in history.get("train_budget_per_epoch", ())],
  }
  return record


def bakeoff(output, run, run_seed: int, arms=None, force: bool = False, dry: bool = False, n0=None,
            n_increment=None, **config):
  """Every requested cell, written one file at a time. Config comes from gearup, as in `bo.py`.

    ``dry`` stops after the header: it resolves the config, reads the trajectory, replays the seeds and
    builds every arm's transform, which is everything that can be wrong without a GPU.

    ``n0`` / ``n_increment`` default to this study's own data schedule (``N0`` / ``N_INCREMENT``), NOT
    to the campaign config's -- see the module docstring. Passing them overrides that; any sweep at a
    different schedule belongs in its OWN output directory, since cells are keyed by optimiser and
    schedule alone and would otherwise be read as this one's arms."""
  config["training"]["n0"] = int(N0 if n0 is None else n0)
  config["training"]["n_increment"] = int(N_INCREMENT if n_increment is None else n_increment)
  detector = detopt.detector.from_config(config["detector"])
  _report, rows, best, n_completed = trajectory(run)
  network_seed, iteration_seed = seeds(int(run_seed), n_completed)
  design = np.asarray(best["x_scaled"], dtype=np.float32)

  os.makedirs(output, exist_ok=True)
  cells = [f"{o}/{s}" for s in SCHEDULES for o in OPTIMIZERS] if arms is None else str(arms).split(",")

  print(f"run           {run}")
  print(f"designs       {n_completed} completed, {len(rows)} scored")
  print(f"best design   loss {float(best['loss']):.4f} at iteration {rows.index(best)}")
  print(f"pool          restored from {os.path.join(run, 'trainer.npz')}")
  print(f"seeds         network {network_seed}, iteration {iteration_seed}")
  print(f"cells         {len(cells)}: {', '.join(cells)}", flush=True)

  with open(os.path.join(output, "meta.json"), "w") as f:
    json.dump({
      "run": run,
      "run_seed": int(run_seed),
      "n_completed": n_completed,
      "best_loss_reported": float(best["loss"]),
      "best_iteration": rows.index(best),
      "design_scaled": [float(v) for v in design],
      "design": best.get("design"),
      "network_seed": network_seed,
      "iteration_seed": iteration_seed,
      "learning_rate": LEARNING_RATE,
      "weight_decay": WEIGHT_DECAY,
      "k": K_EPOCHS * steps_per_epoch(config),
      "k_epochs": K_EPOCHS,
      "steps_per_epoch": steps_per_epoch(config),
      "n0": config["training"]["n0"],
      "n_increment": config["training"]["n_increment"],
    }, f, indent=2)

  for cell in cells:
    optimizer, schedule = cell.split("/")
    transform(optimizer, learning_rate(schedule, K_EPOCHS * steps_per_epoch(config)))
  if dry:
    print("\n[dry] config, trajectory, seeds and all transforms build; stopping before training")
    return

  for cell in cells:
    optimizer, schedule = cell.split("/")
    path = os.path.join(output, f"{optimizer}-{schedule}.json")
    if os.path.exists(path) and not force:
      print(f"[skip] {cell}: {path} exists", flush=True)
      continue
    print(f"\n=== {cell} ===", flush=True)
    record = run_arm(config, detector, run, network_seed, iteration_seed, design, optimizer, schedule,
                     step=n_completed)
    with open(path, "w") as f:
      json.dump(record, f, indent=2)
    objective = "--" if record["objective"] is None else f"{record['objective']:.4f}"
    print(f"[{record['status']}] {cell}: objective {objective} after {record['n_epochs']} epochs, "
          f"{record['wall_s'] / 60.0:.1f} min", flush=True)

  print("\ndone", flush=True)


if __name__ == "__main__":
  import sys

  import gearup

  gearup.gearup(bakeoff).with_config("config/root.yaml")(sys.argv[1:])
