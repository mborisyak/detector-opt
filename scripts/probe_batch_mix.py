#!/usr/bin/env python3
"""ONE-DESIGN probe of the continual strategy's current:replay BATCH COMPOSITION.

    python scripts/probe_batch_mix.py --run output/enzyme_extremes/1244111331/meta --design 17 \
        --repeats 3 --current-replay-ratio 3.0 --output output/probe-batchmix/extremes-d17

WHAT IT MEASURES. One MID-TRAJECTORY design of an existing run, trained three ways -- `continue`
(fresh per-design network warm-started from the previous design), `meta` (the persistent network at
the 50/50 batch) and `meta_ratio` (the same at a configurable ratio) -- and repeated, so a difference
between conditions can be read against the scatter WITHIN one. Design 0 is refused: with ``start ==
0`` there is no replay history and the composition has nothing to act on.

WHY IT IS CHEAP. The replay half needs the POOL populated with the earlier designs' events; it does
not need those designs TRAINED. The earlier designs are therefore replayed by SAMPLING alone -- the
recorded ``spent`` of each is reproduced round by round through the trainer's own ``_sample_round``,
so the pool ends up holding the same events, at the same positions, that the original run had when it
opened the probed design's window. The network that enters the probed design is read from the
previous design's checkpoint, which is what both the persistent and the warm-started arms would have
carried into it.

THE READOUT IS THE FULL PER-EPOCH HISTORY, not just the exit values. "Samples needed" must be samples
to reach a FIXED LOSS LEVEL: the convergence rule is a slope test that never looks at a level, so
samples-until-it-fires would re-import the very artefact this probe exists to avoid. Every repeat
therefore stores its train/validation curves, their standard errors and the window at every epoch, and
the level crossings are computed afterwards from those.

WHAT VARIES BETWEEN REPEATS is the training seed alone: the trainer's own seed (which fixes the event
stream, the pool contents and the network the buffer state is built from) and the restored parameters
are held. Between conditions, the pool is REWOUND to the probed design's window start, so every
repeat of every arm trains on the identical events -- and the FILLED POOL ITSELF is handed from one
arm to the next, which is what makes the replay cost be paid once rather than once per arm.

SIZE THE DEVICE FOR TWO POOL PAIRS, not one. ``adopt_pools`` releases a later arm's own pools only
AFTER ``build_trainer`` has constructed that arm's trainer, and ``Buffered.__init__`` allocates
``jnp.zeros`` at FULL capacity the moment it is handed no ``buffers`` -- so two pairs are resident at
once, and (its own docstring) "XLA keeps that high-water mark for the process's life". Deleting the
spare buffers frees the arrays, it does not lower the mark. At the SHiP budget of 2**20 that is
3.28 GB per pair and a 6.6 GB peak, which is what a GPU has to be chosen against. The API already
admits the real fix -- ``Buffered(..., buffers=...)`` takes the first arm's slots instead of
allocating new ones -- and taking it would make the peak one pair again.

PROVENANCE. The trajectory is used only as a SEQUENCE OF DESIGNS to populate a pool with, which is
independent of the code that produced it; all conditions are trained by today's code inside one
process.

THE FRESH-NETWORK MODE (``--fresh-network``), and the overrides that exist only to serve it.
A checkpoint records the regressor config it was trained under and ``checkpoint_regressor_config``
treats that as authoritative, so an ARCHITECTURE the run never used -- a different activation, say --
cannot be probed by restoring the run's network: it would either fail to pour into the new structure
or silently run the old one. ``--fresh-network`` builds ONE network from the trainer's own
``design_init_sequence`` and gives the SAME parameters to every arm (as ``init_params`` for the
per-design arms, as ``_running`` for the continual ones), reads no checkpoint at all, and therefore
also skips the checkpoint-derived regressor config and the checkpoint-vs-results design check.

WHAT SURVIVES THAT, AND WHAT DOES NOT. The carried NETWORK is gone by construction, so `continue` and
the continual arms no longer differ in their starting weights. What still separates them is the only
other thing that differs: ``_sample_indices``. The continual trainers draw half of every minibatch
from the replayed history ``[0, w0)`` at ``replay_weight``, the per-design trainer draws all of it
from the current design's window -- and both are SCORED on the current window alone. The pool history
is what ``replay_pool`` puts there without training anything, so the comparison is the REPLAY channel
in isolation. It is not the full strategy difference and must not be reported as one.

``--loss-precision``, ``--activation`` and ``--features`` override the trajectory's own saved config,
which is otherwise authoritative here (``--config`` is only a fallback for trajectories that predate the saved
field, so it CANNOT be used to change a setting). ``--allow-cap`` records a cell that reaches
``iteration_limit`` without converging as an unconverged row -- objective, gap, err and window at the
last epoch, by the trainer's own midpoint/uniform formula -- instead of letting the trainer's
RuntimeError abort every arm still to come. That is a censoring difference worth keeping: a converged
cell only proves ``gap + err <= precision``, a capped one reports where the gap actually sits. Every
one of these flags is off by default, so a caller that does not pass them gets the previous behaviour
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import numpy as np
from flax import nnx

import detopt
import detopt.detector
import detopt.utils.config
import detopt.utils.io
from detopt.nn.trainer import ContinualRatioTrainer, ContinualTrainer, DesignTrainer
from detopt.nn.trainer.common import design_init_sequence

ARMS = ("continue", "meta", "meta_ratio")
CAP_MESSAGE = "did not reach precision within iteration_limit"
TRAINERS = {"continue": DesignTrainer, "meta": ContinualTrainer, "meta_ratio": ContinualRatioTrainer}
STRATEGY_KNOBS = {"replay_weight": ("meta", "meta_ratio"), "current_replay_ratio": ("meta_ratio", )}


def load_run_config(trajectory, name):
  """The run config, and its detector block resolved the way gearup resolves them.

  A trajectory written by a current ``bo.py`` carries its own config and that one is used; an older
  one does not, and ``--config`` supplies the run config it was launched with."""
  config = trajectory.get("config")
  if config is None:
    if name is None:
      raise SystemExit("this results.json predates the saved-config field, so --config <run config name> is required")
    config = detopt.utils.config.load_config(f"config/{str(name).lstrip('=')}.yaml")
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    detector_config = detopt.utils.config.load_config(f"config/detector/{detector_config}.yaml")
  return {**config, "detector": detector_config}


def checkpoint_regressor_config(run_dir, iteration):
  """The REGRESSOR config the run itself trained with, read from the design's own checkpoint.

  The config FILE is edited between campaigns -- this run's checkpoints record ``dropconnect: 0.1``
  where the file now says ``0.05`` -- and a network restored under a regulariser it was not trained
  with is a different network. The checkpoint's copy is therefore authoritative here, and any
  difference from the file is announced rather than absorbed."""
  path = os.path.join(run_dir, "checkpoints", f"design_{iteration:04d}")
  manager = detopt.utils.io.get_checkpointer(path)
  stored = detopt.utils.io.restore_config(manager)
  manager.close()
  return stored["regressor"]


def network_seed(run_seed):
  """The seed ``scripts/bo.py`` builds its trainer with, so the probe draws the run's OWN event
  stream: one sequence per run, split once, the network branch's first state."""
  network_sequence, _ = np.random.SeedSequence(int(run_seed)).spawn(2)
  return int(network_sequence.generate_state(1)[0])


def build_trainer(arm, detector, config, seed, current_replay_ratio):
  """A trainer for ``arm`` with NO checkpoint directory -- the probe reads checkpoints, never writes
  them. Knobs only some trainers accept are dropped for the arms that do not own them, as
  ``scripts/bo.py`` does, so one config serves every arm."""
  training = {k: v for k, v in config["training"].items() if arm in STRATEGY_KNOBS.get(k, (arm, ))}
  if arm == "meta_ratio":
    training["current_replay_ratio"] = float(current_replay_ratio)
  return TRAINERS[arm].from_config(detector, {**config, "training": training}, checkpoint_dir=None, seed=seed)


def run_iteration_seed(run_seed, iteration):
  """The training seed ``scripts/bo.py`` used for ``iteration``: the iteration branch's ``k``-th spawn,
  which is what a resumed run replays to land on the same stream."""
  _, iteration_sequence = np.random.SeedSequence(int(run_seed)).spawn(2)
  return int(iteration_sequence.spawn(int(iteration) + 1)[int(iteration)].generate_state(1)[0])


def replay_pool(trainer, detector, designs_scaled, spends):
  """Fill the pools with designs ``0 .. len(spends) - 1`` by SAMPLING ONLY, reproducing each design's
  recorded ``spent`` round by round. Raises if a design's spend is not reachable as ``n0 + k *
  n_increment`` worth of rounds, which would mean the pool no longer matches the run's."""
  train_pool, val_pool = trainer.train_pool, trainer.val_pool
  for index, (scaled, spent) in enumerate(zip(designs_scaled, spends)):
    design = detector.to_nominal(np.asarray(scaled, dtype=np.float32))
    w0_train, w0_val = train_pool.current, val_pool.current
    requested = trainer.n0
    while (train_pool.current - w0_train) + (val_pool.current - w0_val) < spent:
      added = trainer._sample_round(design, w0_train, w0_val, requested)
      if added is None:
        raise RuntimeError(f"replay of design {index}: the budget pool filled before its {spent} events did")
      if added == 0:
        raise RuntimeError(f"replay of design {index}: the window cap was reached before its {spent} events were added")
      requested = trainer.n_increment
    got = (train_pool.current - w0_train) + (val_pool.current - w0_val)
    if got != spent:
      raise ValueError(
        f"replay of design {index} added {got} events, the run recorded {spent} -- the pool would "
        f"not hold what the run's own window held"
      )
  return train_pool.current, val_pool.current


def restore_previous_network(trainer, run_dir, iteration):
  """``(parameters, state)`` of the network the run reported for ``iteration``, poured into the
  structure of a regressor this trainer builds. That is the network BOTH continual arms carry into
  the next design and the one the warm-started arm reads."""
  path = os.path.join(run_dir, "checkpoints", f"design_{iteration:04d}")
  if not os.path.isdir(path):
    raise FileNotFoundError(
      f"no checkpoint at {path} -- the probe continues the network the run reported for "
      f"design {iteration}, so that design's checkpoint must be kept"
    )
  manager = detopt.utils.io.get_checkpointer(path)
  if manager.latest_step() is None:
    raise ValueError(f"{path} holds no saved epoch")
  _, parameters, state = trainer._build_regressor(trainer.seed)
  pure_parameters, pure_state, design, _aux = detopt.utils.io.restore_training_checkpoint(
    manager, regressor=(parameters, state)
  )
  manager.close()
  nnx.replace_by_pure_dict(parameters, pure_parameters)
  nnx.replace_by_pure_dict(state, pure_state)
  device = trainer.device
  return jax.device_put(parameters, device), jax.device_put(state, device), design


def fresh_network(trainer, design_index):
  """``(parameters, state)`` of an UNTRAINED network of this trainer's own architecture, drawn at the
  seed a per-design arm would use for ``design_index``.

  Every arm calls this with the same trainer seed and the same index, so every arm starts from the
  same weights -- which is the whole point of it, and the reason it is derived rather than random."""
  _, parameters, state = trainer._build_regressor(
    int(design_init_sequence(trainer.seed, int(design_index)).generate_state(1)[0])
  )
  device = trainer.device
  return jax.device_put(parameters, device), jax.device_put(state, device)


def adopt_pools(trainer, pools):
  """Give ``trainer`` an ALREADY-FILLED pool pair, releasing the empty one it allocated for itself."""
  for attribute, pool in zip(("train_pool", "val_pool"), pools):
    for buffer in jax.tree.leaves(getattr(trainer, attribute).buffers()):
      buffer.delete()
    setattr(trainer, attribute, pool)


def run_condition(
  arm, trainer, run_dir, design_index, designs_scaled, window_start, repeats, seed_base, save, fidelity_seed=None, fresh=False,
  allow_cap=False
):
  """Every repeat of one arm on one design, on a pool that is rewound between them. ``save`` is called
  with the rows so far after every repeat, so a job killed at its wall clock still leaves what it measured."""
  w0_train, w0_val = window_start
  if fresh:
    parameters, state = fresh_network(trainer, design_index)
  else:
    parameters, state, checkpoint_design = restore_previous_network(trainer, run_dir, design_index - 1)
    previous_scaled = np.asarray(designs_scaled[design_index - 1], dtype=np.float32)
    if not np.allclose(np.asarray(checkpoint_design["scaled"], np.float32), previous_scaled):
      raise ValueError(f"the checkpoint for design {design_index - 1} holds a different design than results.json")

  rows = []
  # The FIDELITY run comes first and is not a repeat: it trains this design at the seed the ORIGINAL
  # run used for it, so its objective and `spent` are directly comparable to the recorded ones and the
  # reconstruction is checked rather than assumed.
  schedule = [] if fidelity_seed is None else [(-1, int(fidelity_seed))]
  schedule += [(repeat, seed_base + repeat) for repeat in range(repeats)]
  for repeat, training_seed in schedule:
    trainer.train_pool.current, trainer.val_pool.current = w0_train, w0_val
    if arm == "continue":
      initial = parameters
    else:
      trainer._running = (parameters, state)
      initial = None
    latest = {}

    def on_epoch(snapshot, sink=latest):
      sink.clear()
      sink.update(snapshot)

    started = time.time()
    capped = False
    try:
      result = trainer.train(
        np.asarray(designs_scaled[design_index], dtype=np.float32), training_seed, init_params=initial, on_epoch=on_epoch,
        step=design_index
      )
    except RuntimeError as error:
      if not allow_cap or CAP_MESSAGE not in str(error):
        raise
      capped, result = True, None
    elapsed = time.time() - started
    if result is None and not capped:
      raise RuntimeError(f"[{arm}] repeat {repeat}: the budget pool filled -- the probe cannot score this design")
    train = float(latest["train_loss_per_epoch"][-1])
    validation = float(latest["val_loss_per_epoch"][-1])
    train_sem = float(latest["train_sem_per_epoch"][-1])
    validation_sem = float(latest["val_sem_per_epoch"][-1])
    gap = abs(validation - train)
    err = float(np.hypot(train_sem, validation_sem))
    if capped:
      objective = 0.5 * (train + validation)
      objective_std = float(np.hypot(gap / np.sqrt(12.0), 0.5 * err))
      spent = (trainer.train_pool.current - w0_train) + (trainer.val_pool.current - w0_val)
    else:
      objective, objective_std, spent = float(result.objective_loss), float(result.objective_std), int(result.spent)
    row = {
      "arm": arm,
      "repeat": repeat,
      "seed": training_seed,
      "train": train,
      "validation": validation,
      "gap": gap,
      "err": err,
      "objective": objective,
      "objective_std": objective_std,
      "converged": not capped,
      "epochs": int(latest["train_loss_per_epoch"].shape[0]),
      "window": int(latest["final_train_budget"]),
      "spent": int(spent),
      "seconds": elapsed,
      "train_per_epoch": [float(v) for v in latest["train_loss_per_epoch"]],
      "validation_per_epoch": [float(v) for v in latest["val_loss_per_epoch"]],
      "train_sem_per_epoch": [float(v) for v in latest["train_sem_per_epoch"]],
      "validation_sem_per_epoch": [float(v) for v in latest["val_sem_per_epoch"]],
      "window_per_epoch": [int(v) for v in latest["train_budget_per_epoch"]],
    }
    rows.append(row)
    save(rows)
    print(
      f"[{arm}] {'fidelity' if repeat < 0 else f'repeat {repeat}'}: objective={row['objective']:.4f} gap={row['gap']:.4f} err={row['err']:.4f} "
      f"epochs={row['epochs']} window={row['window']} spent={row['spent']} "
      f"{'CONVERGED' if not capped else 'CAPPED'} ({elapsed:.0f}s)", flush=True
    )
  return rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True, help="a finished bo.py run directory (results.json + checkpoints)")
  parser.add_argument("--design", type=int, required=True, help="the design index to probe; 0 is refused")
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--current-replay-ratio", type=float, default=3.0)
  parser.add_argument("--seed-base", type=int, default=1000)
  parser.add_argument("--arms", nargs="+", default=list(ARMS))
  parser.add_argument("--config", default=None, help="run config name, for a trajectory that predates the saved one")
  parser.add_argument("--device", default=None, help="override the config's device (a CPU dry run needs it)")
  parser.add_argument(
    "--fidelity-arm", default="meta", help="the arm that also trains at the run's OWN seed, "
    "so the reconstruction is checked against the recorded loss; empty to skip"
  )
  parser.add_argument(
    "--fresh-network", action="store_true", help="give every arm the SAME untrained network instead of "
    "the previous design's checkpoint, so an architecture the run never trained can be probed"
  )
  parser.add_argument("--loss-precision", type=float, default=None, help="override the saved config's training.loss_precision")
  parser.add_argument("--activation", default=None, help="override the saved config's regressor activation")
  parser.add_argument(
    "--features", default=None, help="override the saved config's regressor features, as JSON: '[[16,24],[24,16]]'"
  )
  parser.add_argument(
    "--allow-cap", action="store_true", help="record a cell that reaches iteration_limit unconverged as an "
    "unconverged row instead of letting it abort the probe"
  )
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  if arguments.design <= 0:
    raise SystemExit("--design 0 has no replay history (start == 0), so the composition cannot act on it")
  for arm in arguments.arms:
    if arm not in TRAINERS:
      raise SystemExit(f"unknown arm {arm!r}; known: {sorted(TRAINERS)}")

  trajectory = json.load(open(os.path.join(arguments.run, "results.json")))
  config = load_run_config(trajectory, arguments.config)
  if arguments.device is not None:
    config = {**config, "device": arguments.device}
  results = trajectory["results"]
  if arguments.design >= len(results):
    raise SystemExit(f"--design {arguments.design} is past the run's {len(results)} designs")
  designs_scaled = [row["x_scaled"] for row in results]
  spends = [int(row["spent"]) for row in results]
  run_seed = int(os.path.basename(os.path.dirname(os.path.normpath(arguments.run))))
  seed = network_seed(run_seed)

  os.makedirs(arguments.output, exist_ok=True)
  if not arguments.fresh_network:
    stored_regressor = checkpoint_regressor_config(arguments.run, arguments.design - 1)
    if stored_regressor != config["regressor"]:
      print(
        f"[config] the run trained with regressor {json.dumps(stored_regressor)}, the config file now says "
        f"{json.dumps(config['regressor'])}; the CHECKPOINT's is used, because the restored network is that "
        f"network", flush=True
      )
      config = {**config, "regressor": stored_regressor}
  if arguments.loss_precision is not None:
    print(
      f"[config] loss_precision {config['training']['loss_precision']} -> {arguments.loss_precision} (--loss-precision)",
      flush=True
    )
    config = {**config, "training": {**config["training"], "loss_precision": float(arguments.loss_precision)}}
  overrides = {}
  if arguments.activation is not None:
    overrides["activation"] = arguments.activation
  if arguments.features is not None:
    overrides["features"] = [[int(width) for width in block] for block in json.loads(arguments.features)]
  if len(overrides) > 0:
    regressor_name, regressor_arguments = detopt.utils.config.split(config["regressor"])
    for key, value in overrides.items():
      print(f"[config] regressor {key} {json.dumps(regressor_arguments.get(key))} -> {json.dumps(value)} (--{key})", flush=True)
    config = {**config, "regressor": {regressor_name: {**regressor_arguments, **overrides}}}
  detector = detopt.detector.from_config(config["detector"])
  print(
    f"probe: {arguments.run} design {arguments.design} ({len(spends[:arguments.design])} designs replayed, "
    f"{sum(spends[:arguments.design])} events), run seed {run_seed}, trainer seed {seed}, "
    f"loss_precision {config['training']['loss_precision']}", flush=True
  )
  print(
    f"  the run itself reported {results[arguments.design]['loss']:.4f} +- "
    f"{results[arguments.design]['loss_std']:.4f} for this design at {spends[arguments.design]} events", flush=True
  )

  path = os.path.join(arguments.output, "probe.json")
  rows, pools, window_start = [], None, None

  def save(current):
    json.dump({
      "run": arguments.run,
      "design": arguments.design,
      "current_replay_ratio": arguments.current_replay_ratio,
      "loss_precision": float(config["training"]["loss_precision"]),
      "regressor": config["regressor"],
      "fresh_network": bool(arguments.fresh_network),
      "reported_loss": float(results[arguments.design]["loss"]),
      "reported_loss_std": float(results[arguments.design]["loss_std"]),
      "rows": list(rows) + list(current),
    }, open(path, "w"), indent=1)

  for arm in arguments.arms:
    trainer = build_trainer(arm, detector, config, seed, arguments.current_replay_ratio)
    if pools is None:
      print(f"[{arm}] filling the pool with designs 0..{arguments.design - 1}", flush=True)
      started = time.time()
      window_start = replay_pool(trainer, detector, designs_scaled[:arguments.design], spends[:arguments.design])
      print(f"[{arm}] pool at {window_start[0]} train / {window_start[1]} val after {time.time() - started:.0f}s", flush=True)
    else:
      adopt_pools(trainer, pools)
    rows.extend(
      run_condition(
        arm, trainer, arguments.run, arguments.design, designs_scaled, window_start, arguments.repeats, arguments.seed_base,
        save, fidelity_seed=run_iteration_seed(run_seed, arguments.design) if arm == arguments.fidelity_arm else None,
        fresh=arguments.fresh_network, allow_cap=arguments.allow_cap
      )
    )
    pools = (trainer.train_pool, trainer.val_pool)
    del trainer
    save([])
  print(f"\nwrote {path}")


if __name__ == "__main__":
  main()
