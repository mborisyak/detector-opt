#!/usr/bin/env python3
"""Score ONE banked design under a retention setting and record what the training COST, epoch by epoch.

    python scripts/probe_cosine.py =linear_d3n4 strategy=linear-from_scratch-sp-l03-s1e2 \
        trajectory=output/linear/select/d3n4/282522616/from_scratch/norewind/results.json \
        seed=282522616 design_index=4 training.cosine_epochs=16 training.cosine_peak=4.0 \
        output=output/probe-cosine/d3n4/282522616/cosine-on.json

The state is reconstructed exactly as `probe_retention.py` does it (pools replayed from the trajectory, the
design's own iteration seed, warm start for the arms that take one) and the call is `trainer.train()`, so
the growth procedure, the exit test and the retention rule all run. What this adds is a per-epoch record
of the things an integral rank against detector calls cannot see: wall time, the schedule multiplier the
epoch started at, and the parameter norm by leaf kind (kernel / bias / activation gain) at the start and
end of every epoch. The norm at the start of the epoch after a data addition against the norm at the end
of the epoch before it is the retention rule's own arithmetic, measured in the loop.
"""
import json
import os
import shutil
import time

import jax
import numpy as np

import detopt
import detopt.detector
from detopt.nn.trainer.schedule import schedule_count

from bo import TRAINERS, drop_foreign_knobs, _resolve_strategy

NORM_GROUPS = {"kernel": ("kernel", ), "bias": ("bias", ), "gain": ("positive", "negative")}


def parameter_norms(params):
  """L2 norm of the whole parameter tree and of each leaf kind in ``NORM_GROUPS``."""
  squares = {name: 0.0 for name in NORM_GROUPS}
  squares["other"] = 0.0
  for path, leaf in jax.tree_util.tree_leaves_with_path(params):
    key = jax.tree_util.keystr(path)
    group = "other"
    for name, markers in NORM_GROUPS.items():
      if any(marker in key for marker in markers):
        group = name
    squares[group] += float(np.sum(np.square(np.asarray(leaf, dtype=np.float64))))
  norms = {name: float(np.sqrt(value)) for name, value in squares.items()}
  norms["total"] = float(np.sqrt(sum(squares.values())))
  return norms


def splice_epoch_recorder(trainer, records):
  """Wrap ``trainer._train_epoch`` so every epoch appends its window, multiplier, wall time and norms."""
  original = trainer._train_epoch

  def recorded(params, state, opt_state, key, start, count, buffers):
    started = time.time()
    norm_in = parameter_norms(params)
    multiplier = 1.0 if trainer.cosine_schedule is None else float(trainer.cosine_schedule(schedule_count(opt_state)))
    params, state, opt_state, losses = original(params, state, opt_state, key, start, count, buffers)
    jax.block_until_ready(params)
    records.append({
      "window": int(count),
      "multiplier": multiplier,
      "wall_s": time.time() - started,
      "norm_in": norm_in,
      "norm_out": parameter_norms(params),
    })
    return params, state, opt_state, losses

  trainer._train_epoch = recorded


def probe_cosine(output, trajectory, seed: int, design_index: int = 4, replay: bool = True, **config):
  config = _resolve_strategy(config)
  arm = config["nn_init_strategy"]
  config = drop_foreign_knobs(config, arm)
  training = config["training"]

  with open(trajectory) as handle:
    rows = json.load(handle)["results"]
  index = int(design_index)
  if not 0 < index < len(rows):
    raise SystemExit(f"probe_cosine: design index {index} outside 1..{len(rows) - 1} of {trajectory}")

  source_checkpoints = os.path.join(os.path.dirname(trajectory), "checkpoints")
  if not os.path.isdir(source_checkpoints):
    raise SystemExit(f"probe_cosine: no checkpoints beside {trajectory}")
  work = os.path.splitext(output)[0] + ".checkpoints"
  if os.path.isdir(work):
    shutil.rmtree(work)
  os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
  shutil.copytree(source_checkpoints, work)

  detector = detopt.detector.from_config(config["detector"])
  network_seq, iteration_seq = np.random.SeedSequence(int(seed)).spawn(2)
  trainer = TRAINERS.get(arm, TRAINERS["per_design"]).from_config(
    detector, config, checkpoint_dir=work, seed=int(network_seq.generate_state(1)[0]),
  )
  if replay:
    trainer.replay(rows[:index])
  else:
    print(f"[probe] NO REPLAY: pools start empty, the window opens at stream position 0", flush=True)
  iteration_seq.spawn(index)
  iteration_seed = int(iteration_seq.spawn(1)[0].generate_state(1)[0])

  init_params = None
  if arm in ("continue", "closest"):
    if arm == "continue":
      donor = index - 1
    else:
      history = np.asarray([row["x_scaled"] for row in rows[:index]], dtype=np.float64)
      donor = int(np.argmin(np.linalg.norm(history - np.asarray(rows[index]["x_scaled"], np.float64)[None, :], axis=1)))
    print(f"[warm-start] {arm}: design {donor}", flush=True)
    init_params = trainer.restore_design_parameters(donor)

  design_scaled = np.asarray(rows[index]["x_scaled"], dtype=np.float32)
  print(f"[probe] {arm} @ design {index}/{len(rows)} of {trajectory}; detopt from {detopt.__file__}", flush=True)
  print(
    f"[probe] shrink={training.get('shrink', 1.0)} param_noise={training.get('param_noise')} "
    f"rewind={training.get('rewind', 0.0)} cosine_epochs={training.get('cosine_epochs', 0)} "
    f"cosine_peak={training.get('cosine_peak', 4.0)} optimizer={training['optimizer']}", flush=True
  )

  longest = {}

  def on_epoch(snapshot, _longest=longest):
    if len(snapshot["train_loss_per_epoch"]) >= len(_longest.get("train_loss_per_epoch", ())):
      _longest.clear()
      _longest.update(snapshot)

  epochs = []
  splice_epoch_recorder(trainer, epochs)
  started = time.time()
  result = trainer.train(design_scaled, iteration_seed, init_params=init_params, on_epoch=on_epoch, step=index)
  wall = time.time() - started
  if result is None:
    raise SystemExit("probe_cosine: the shared budget pool was exhausted before the design converged")

  windows = [record["window"] for record in epochs]
  additions = [i for i in range(1, len(windows)) if windows[i] != windows[i - 1]]
  record = {
    "arm": arm,
    "seed": int(seed),
    "trajectory": trajectory,
    "design_index": index,
    "replayed_prefix": bool(replay),
    "n_designs": len(rows),
    "x_scaled": [float(v) for v in design_scaled],
    "training": {
      k: v
      for k, v in training.items()
    },
    "detopt_file": detopt.__file__,
    "loss": float(result.objective_loss),
    "loss_std": float(result.objective_std),
    "reference_loss": rows[index].get("loss"),
    "spent": int(result.spent),
    "epochs": len(epochs),
    "additions": additions,
    "wall_s": wall,
    "train_per_epoch": [round(float(v), 8) for v in longest["train_loss_per_epoch"]],
    "val_per_epoch": [round(float(v), 8) for v in longest["val_loss_per_epoch"]],
    "window_per_epoch": [int(v) for v in longest["train_budget_per_epoch"]],
    "epoch_records": epochs,
  }
  with open(output, "w") as handle:
    json.dump(record, handle, indent=1)
  shutil.rmtree(work, ignore_errors=True)
  print(
    f"[probe] loss {record['loss']:.6f}+-{record['loss_std']:.4f} on {record['spent']} calls, "
    f"{record['epochs']} epochs, {len(additions)} additions, {wall / 60:.1f} min -> {output}", flush=True
  )
  return record


if __name__ == "__main__":
  import sys

  import gearup

  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  gearup.gearup(probe_cosine).with_config("config/root.yaml")(sys.argv[1:])
