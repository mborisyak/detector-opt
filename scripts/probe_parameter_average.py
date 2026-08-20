#!/usr/bin/env python3
"""Does scoring an AVERAGE of the iterates lower the ``|val - train|`` floor?

ONE design, trained repeatedly with ``training.param_average_epochs`` off and on. The design and the
state it is reached from come from a finished run: the recorded ``x_scaled`` sequence in its
``results.json`` is REPLAYED into a fresh trainer, design by design, so at design ``index`` the event
pools hold the same designs' events in the same order and the continual arm holds the network its
replay history produced. Replaying rather than restoring a checkpoint reconstructs the pool as well as
the parameters, and the pool is what the continual arm's minibatches draw half of themselves from.

The two conditions share that reconstructed state exactly: the replay runs once, is written out with
``Trainer.persist`` (pools, cursors, and whatever the strategy carries across designs), and the
averaged trainer is restored from it. Within a condition the repeats differ only in the training key --
the network draw is ``design_init_sequence(trainer_seed, index)``, which does not move with the seed
passed to ``train`` -- so the spread across repeats is the optimiser's own randomness and nothing else.

The measurement is the trainer's ``[average-vs-raw]`` trace: with averaging on, every epoch scores BOTH
parameter points on the SAME two windows, so the averaged and unaveraged ``gap`` differ by the
evaluation point alone.

Usage:
    python scripts/probe_parameter_average.py =<config> run=<finished run dir> run_seed=<seed> \\
        index=<design> arm=<meta|from_scratch> repeats=3 horizon=1.0 output=<dir>
"""

import json
import os
import sys

import gearup
import numpy as np

import detopt
import detopt.detector
import detopt.utils.io
from detopt.nn.trainer import ContinualRatioTrainer, ContinualTrainer, DesignTrainer

TRAINERS = {"per_design": DesignTrainer, "meta": ContinualTrainer, "meta_ratio": ContinualRatioTrainer}
STRATEGY_KNOBS = {"replay_weight": ("meta", "meta_ratio"), "current_replay_ratio": ("meta_ratio", )}


def _config_for_arm(config, arm):
  """``config`` with the knobs this arm's trainer does not accept removed -- bo.py's own rule."""
  for knob, owners in STRATEGY_KNOBS.items():
    if arm not in owners and knob in config.get("training", {}):
      config = {**config, "training": {k: v for k, v in config["training"].items() if k != knob}}
  return config


def _build(config, arm, horizon, seed, checkpoint_dir):
  training = dict(config["training"])
  training["param_average_epochs"] = float(horizon)
  config = _config_for_arm({**config, "training": training}, arm)
  detector = detopt.detector.from_config(config["detector"])
  trainer_cls = TRAINERS.get(arm, TRAINERS["per_design"])
  return trainer_cls.from_config(detector, config, checkpoint_dir=checkpoint_dir, seed=seed)


def _train_window(spent, val_fraction):
  """The TRAIN half of a recorded ``spent``. ``_sample_round`` adds ``round(n * f / (1 - f))``
    validation events for ``n`` training events, so the recorded total fixes both."""
  ratio = val_fraction / (1.0 - val_fraction)
  for n_train in range(1, int(spent) + 1):
    if n_train + round(n_train * ratio) == int(spent):
      return n_train
  raise ValueError(f"no train/val split of {spent} events at val_fraction {val_fraction}")


def _seeds(run_seed, count):
  """bo.py's own seed derivation: ONE sequence per run, split into a network branch (which seeds the
    trainer, and through it every design's network draw) and an iteration branch that yields one seed
    per iteration. Replaying it here puts the replay on the stream the recorded run was on."""
  network_seq, iteration_seq = np.random.SeedSequence(int(run_seed)).spawn(2)
  trainer_seed = int(network_seq.generate_state(1)[0])
  return trainer_seed, [int(iteration_seq.spawn(1)[0].generate_state(1)[0]) for _ in range(count)]


def _replay(trainer, rows, index, iteration_seeds):
  """Fill the pools with designs ``0 .. index - 1``, training each one on its recorded design at its
    recorded seed, so a continual arm reaches design ``index`` with the network its history produced."""
  for row in rows[:index]:
    step = int(row["iteration"])
    scaled = np.asarray(row["x_scaled"], dtype=np.float32)
    target = _train_window(int(row["spent"]), trainer.val_fraction)
    result = trainer.train(scaled, iteration_seeds[step], step=step)
    if result is None:
      raise RuntimeError(f"budget pool exhausted while replaying design {step}")
    print(
      f"  [replay] design {step:3d} spent={result.spent} (recorded {row['spent']}, "
      f"train target {target}) loss={result.objective_loss:.5f} (recorded {row['trained_loss']:.5f})", flush=True
    )


class _History:
  """Keeps the LONGEST per-epoch snapshot the trainer offered, which is the complete one: the trainer
    hands a fresh snapshot of the whole history every epoch."""

  def __init__(self):
    self.snapshot = None

  def __call__(self, snapshot):
    if self.snapshot is None or snapshot["train_loss_per_epoch"].shape[0] > self.snapshot["train_loss_per_epoch"].shape[0]:
      self.snapshot = snapshot

  def trace(self):
    if self.snapshot is None:
      return {}
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self.snapshot.items()}


def _snapshot(trainer):
  return (trainer.train_pool.current, trainer.val_pool.current, getattr(trainer, "_running", None))


def _rewind(trainer, snapshot):
  """Put the pools' cursors and the carried network back. The rows above a cursor are overwritten by
    the next fill, and the detector is a function of (design, event index), so the next repeat
    regenerates exactly the events this one did."""
  train_current, val_current, running = snapshot
  trainer.train_pool.current = train_current
  trainer.val_pool.current = val_current
  if running is not None:
    trainer._running = running


def probe(output, run, run_seed: int, index: int, arm: str, repeats: int = 3, horizon: float = 1.0, **config):
  os.makedirs(output, exist_ok=True)
  index, repeats = int(index), int(repeats)
  horizons = [0.0, float(horizon)]
  rows = detopt.utils.io.complete_results(json.load(open(os.path.join(run, "results.json")))["results"])
  if index < 1 or index >= len(rows):
    raise ValueError(f"index must name a MID-trajectory design in [1, {len(rows)}), got {index}")
  trainer_seed, iteration_seeds = _seeds(run_seed, index + 1)
  design_scaled = np.asarray(rows[index]["x_scaled"], dtype=np.float32)
  print(
    f"[probe] {run} arm={arm} design {index} of {len(rows)}: x_scaled={design_scaled.tolist()} "
    f"reported={rows[index]['trained_loss']:.5f} spent={rows[index]['spent']}", flush=True
  )

  state_path = os.path.join(output, "replayed.npz")
  record = []
  for position, horizon in enumerate(horizons):
    trainer = _build(config, arm, horizon, trainer_seed, checkpoint_dir=None)
    if position == 0:
      print(f"[replay] {index} designs into a fresh trainer (arm={arm})", flush=True)
      _replay(trainer, rows, index, iteration_seeds)
      trainer.persist(state_path)
      detopt.utils.io.commit([state_path])
    else:
      trainer.restore(state_path)
    print(
      f"[state] pools at {trainer.train_pool.current}+{trainer.val_pool.current}; "
      f"param_average_epochs={horizon}", flush=True
    )
    snapshot = _snapshot(trainer)
    for repeat in range(repeats):
      _rewind(trainer, snapshot)
      print(f"\n[run] horizon={horizon} repeat={repeat}", flush=True)
      history = _History()
      result = trainer.train(design_scaled, iteration_seeds[index] + repeat, step=index, on_epoch=history)
      if result is None:
        raise RuntimeError("budget pool exhausted on the probe design")
      trace = history.trace()
      epochs = len(trace.get("train_loss_per_epoch", []))
      record.append({
        "horizon": horizon,
        "repeat": repeat,
        "objective_loss": float(result.objective_loss),
        "objective_std": float(result.objective_std),
        "spent": int(result.spent),
        "epochs": epochs,
        "trace": trace,
      })
      print(
        f"[run] horizon={horizon} repeat={repeat} loss={result.objective_loss:.6f} "
        f"std={result.objective_std:.6f} spent={result.spent} epochs={epochs}", flush=True
      )
  with open(os.path.join(output, "probe.json"), "w") as f:
    json.dump({"run": run, "arm": arm, "index": index, "runs": record}, f, indent=2)
  print(f"\n[probe] wrote {os.path.join(output, 'probe.json')}", flush=True)


if __name__ == "__main__":
  gearup.gearup(probe).with_config("config/bo.yaml")(sys.argv[1:])
