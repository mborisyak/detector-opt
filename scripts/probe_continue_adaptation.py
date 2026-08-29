#!/usr/bin/env python3
"""SEQUENTIAL-PAIR probe of what makes the `continue` arm adapt: the ACTIVATION or the DESIGN INPUT.

    python scripts/probe_continue_adaptation.py --run output/ship-tied-rw025/750143450/continue \
        --design 9 --phase first  --activation celu --constant-design mid --output output/probe-adapt/celu-const
    python scripts/probe_continue_adaptation.py --run output/ship-tied-rw025/750143450/continue \
        --design 9 --phase second --activation celu --constant-design mid --output output/probe-adapt/celu-const

WHAT IT MEASURES. Two SEQUENTIAL designs ``n`` and ``n + 1`` of a recorded `continue` trajectory.
Phase ``first`` trains ``n`` from an untrained network and writes its parameters; phase ``second``
restores them and trains ``n + 1``, which is what the `continue` arm does at that step.
``--no-restore`` trains ``n + 1`` from an untrained network instead, and that is the CONTROL:
"condition X adapts faster" is not separable from "condition X is better at this task" unless the
same condition's cold start is known, so the reportable quantity is (cold - warm) at a fixed loss
level, within a condition.

TWO INVOCATIONS, ONE TRAINER PER PROCESS. A trainer allocates its own full-capacity pools at
construction, and ``Buffered.__init__`` warns that allocating and then replacing holds two pairs at
once and that XLA keeps that high-water mark for the life of the process. Holding both designs'
trainers in one process would therefore peak at two pool pairs -- 6.6 GB at the SHiP budget of
``2 ** 20`` against 3.28 GB for one -- so the phases are split across processes and the job's
GPU-memory floor is sized to one pair.

THE TWO CAUSES THIS SEPARATES. ``--activation`` swaps the run's own learnable ``leaky-tanh`` (two
gains per unit, a fast re-gating channel) for a parameter-free one, so an arm difference that is
re-gating rather than carried history shows up here as an activation difference.
``--constant-design`` tells the network a FIXED geometry while the events keep being simulated at the
TRUE one, so the design-as-input channel is removed without touching the physics.

THE OVERRIDE, AND WHY IT IS ONLY A COMBINE. The design reaches the features ONLY through
``_scaled_to_layer_geometry`` -- layer ``positions`` become ``norm_z``, ``angles`` become
``wire_y_left`` / ``wire_y_right``, and ``TDC`` is design-independent -- so substituting a constant
scaled design inside ``combine_scaled`` is the whole of it. It is a DIRTY FEATURE-LEVEL OVERRIDE
applied by wrapping the detector INSTANCE here; nothing under ``detopt/detector/`` is touched, event
simulation is untouched, and ``--phase verify`` reports the numbers that show it bites.

WHY MID-BOX AND NOT ZEROS. ``stereo_bound`` starts at 0.0, so a scaled-zero design has
``view_angle`` exactly 0 and ``wire_y_left == wire_y_right`` for every hit: the stereo channel would
be destroyed and "design withheld" would be confounded with "two features collapsed into one". A
mid-box constant is a valid geometry with a non-zero angle, so within one design's window the
constant-design features stay an invertible relabelling of the true ones and the only thing removed
is the design signal itself.

THE READOUT IS THE FULL PER-EPOCH HISTORY. "Samples needed" must be samples to a FIXED LOSS LEVEL;
the convergence rule is a slope test that never looks at a level, so samples-until-it-fires would
re-import the artefact this probe exists to avoid. Train and validation loss, their standard errors
and the window are stored at every epoch and the crossings are computed afterwards from those. The
file is rewritten every epoch with ``complete: false``, so a running cell can be read live and a
killed one still leaves its curve.

THE POOL is filled by SAMPLING ALONE, reproducing each earlier design's recorded ``spent`` round by
round (``probe_batch_mix.replay_pool``), so both phases open their window where the run's own did.
Phase ``second`` replays THROUGH design ``n`` at its RECORDED spend rather than at whatever phase
``first`` spent, so every condition trains design ``n + 1`` on identical events.

PROVENANCE. The trajectory supplies designs, spends and seeds only; no checkpoint of it is read, so
an architecture the run never trained is probed without pouring its network into a different
structure. Every condition is trained by today's code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import detopt
import detopt.detector
import detopt.utils.config
import detopt.utils.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_batch_mix import build_trainer, load_run_config, network_seed, replay_pool, run_iteration_seed

CAP_MESSAGE = "did not reach precision within iteration_limit"
GEOMETRY_COLUMNS = (1, 2, 3)
TDC_COLUMN = 0


def constant_scaled_design(detector, choice):
  """The SCALED design the override substitutes: ``mid`` (0.5 everywhere), ``zeros`` (the lower
  corner of the design box) or an explicit JSON vector."""
  dimension = detector.design_dim()
  if choice == "mid":
    return np.full((dimension, ), 0.5, dtype=np.float32)
  if choice == "zeros":
    return np.zeros((dimension, ), dtype=np.float32)
  values = np.asarray(json.loads(choice), dtype=np.float32)
  if values.shape != (dimension, ):
    raise SystemExit(f"--constant-design must hold {dimension} values, got {tuple(values.shape)}")
  return values


def constant_design_combine(detector, constant_scaled):
  """A ``combine_scaled`` that IGNORES the design it is handed and uses ``constant_scaled`` instead.

  Installed on the detector INSTANCE by the caller, so ``combine`` -- which is final and calls
  ``self.combine_scaled`` -- routes through it while ``detector(design, event_index)`` keeps
  simulating at the true design."""
  original = detector.combine_scaled
  fixed = jnp.asarray(constant_scaled, dtype=jnp.float32)

  def combine_scaled(event, design_scaled, mask=None):
    given = jnp.asarray(design_scaled, dtype=jnp.float32)
    return original(event, jnp.broadcast_to(fixed, given.shape), mask=mask)

  return combine_scaled


def simulate(detector, design_scaled, n_events):
  """``(event, mask)`` for the first ``n_events`` event indices simulated at ``design_scaled``."""
  index = np.arange(int(n_events), dtype=np.int32)
  design = detector.to_nominal(np.asarray(design_scaled, dtype=np.float32))
  batched = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a)[None], (index.shape[0], ) + jnp.asarray(a).shape), design)
  _truth, event, mask, _target = detector(batched, index)
  return event, mask


def column_spread(left, right, valid):
  """Per-feature-column ``max |left - right|`` over the VALID hits only."""
  difference = np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64))
  return [float(difference[..., column][valid].max()) for column in range(difference.shape[-1])]


def verify_override(detector, constant_scaled, design_a, design_b, n_events):
  """Check that the constant-design override BITES, on events simulated at ``design_a``.

  Three things must hold, and each is reported as a number rather than asserted silently: under the
  override two DIFFERENT designs give identical geometry columns; the override's features DIFFER from
  the true-design path; and the TDC column is untouched. Raises if any of them fails, because a probe
  whose treatment does nothing cannot separate anything."""
  event, mask = simulate(detector, design_a, n_events)
  valid = np.asarray(mask) > 0
  dimension = detector.design_dim()
  batch = valid.shape[0]
  plain = detector.combine_scaled
  override = constant_design_combine(detector, constant_scaled)
  scaled_a = jnp.broadcast_to(jnp.asarray(design_a, jnp.float32), (batch, dimension))
  scaled_b = jnp.broadcast_to(jnp.asarray(design_b, jnp.float32), (batch, dimension))
  true_a, true_b = plain(event, scaled_a, mask=mask), plain(event, scaled_b, mask=mask)
  fixed_a, fixed_b = override(event, scaled_a, mask=mask), override(event, scaled_b, mask=mask)

  report = {
    "n_events": int(batch),
    "n_valid_hits": int(valid.sum()),
    "constant_scaled": [float(v) for v in np.asarray(constant_scaled)],
    "true_two_designs": column_spread(true_a, true_b, valid),
    "override_two_designs": column_spread(fixed_a, fixed_b, valid),
    "override_vs_true": column_spread(fixed_a, true_a, valid),
    "override_stereo_span": float(np.abs(np.asarray(fixed_a)[..., 3] - np.asarray(fixed_a)[..., 2])[valid].max()),
    "true_stereo_span": float(np.abs(np.asarray(true_a)[..., 3] - np.asarray(true_a)[..., 2])[valid].max()),
  }
  columns = ["tdc", "norm_z", "wire_y_left", "wire_y_right"]
  print(f"[verify] {report['n_valid_hits']} valid hits over {batch} events; columns {columns}", flush=True)
  print(f"[verify] constant scaled design {report['constant_scaled']}", flush=True)
  for name in ("true_two_designs", "override_two_designs", "override_vs_true"):
    print(f"[verify] max|d| {name:22s} " + "  ".join(f"{c}={v:.6g}" for c, v in zip(columns, report[name])), flush=True)
  print(
    f"[verify] max|wire_y_right - wire_y_left|: true={report['true_stereo_span']:.6g} "
    f"override={report['override_stereo_span']:.6g}", flush=True
  )

  geometry_moves = max(report["override_two_designs"][column] for column in GEOMETRY_COLUMNS)
  if geometry_moves > 0.0:
    raise SystemExit(f"the override does NOT hold the geometry constant: two designs still differ by {geometry_moves:.6g}")
  bites = max(report["override_vs_true"][column] for column in GEOMETRY_COLUMNS)
  if not bites > 0.0:
    raise SystemExit("the override changes NOTHING against the true-design path; the experiment would be void")
  if report["override_vs_true"][TDC_COLUMN] > 0.0:
    raise SystemExit(f"the override moved the TDC column by {report['override_vs_true'][TDC_COLUMN]:.6g}; it must not")
  if not report["override_stereo_span"] > 0.0:
    raise SystemExit("the constant design has zero stereo angle, so wire_y_left == wire_y_right; pick a non-degenerate one")
  print("[verify] the override bites: geometry frozen, features moved, TDC untouched, stereo alive", flush=True)
  return report


def save_parameters(path, parameters):
  """Write the trained parameters as a FLAT list of leaves.

  Flat is the project's checkpoint form (``detopt.utils.io._leaves``): a flat list has no structure
  to reconstruct, and phase ``second`` recovers the structure from the regressor it builds itself."""
  leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(parameters)]
  detopt.utils.io.atomic_save({path: {f"leaf_{index:04d}": leaf for index, leaf in enumerate(leaves)}})
  return len(leaves), float(sum(np.abs(leaf).sum() for leaf in leaves))


def restore_parameters(trainer, path):
  """The flat leaves at ``path`` poured into a freshly built regressor's parameter structure."""
  source = detopt.utils.io.restore_path(path)
  if source is None:
    raise FileNotFoundError(f"no phase-first parameters at {path}; run --phase first for this condition first")
  _graphdef, parameters, _state = trainer._build_regressor(trainer.seed)
  reference = jax.tree.leaves(parameters)
  with np.load(source) as data:
    leaves = [data[key] for key in sorted(data.files)]
  if len(leaves) != len(reference):
    raise ValueError(f"{source} holds {len(leaves)} leaves, this architecture has {len(reference)}")
  for index, (stored, live) in enumerate(zip(leaves, reference)):
    if stored.shape != live.shape:
      raise ValueError(f"{source} leaf {index} is {tuple(stored.shape)}, this architecture wants {tuple(live.shape)}")
  restored = jax.tree.unflatten(jax.tree.structure(parameters), [jnp.asarray(leaf) for leaf in leaves])
  checksum = float(sum(np.abs(leaf).sum() for leaf in leaves))
  return jax.device_put(restored, trainer.device), len(leaves), checksum


def curve_row(snapshot):
  """The per-epoch history of one training run, plus the values at its last epoch."""
  train = snapshot["train_loss_per_epoch"]
  validation = snapshot["val_loss_per_epoch"]
  train_sem = snapshot["train_sem_per_epoch"]
  validation_sem = snapshot["val_sem_per_epoch"]
  return {
    "epochs": int(train.shape[0]),
    "window": int(snapshot["final_train_budget"]),
    "val_window": int(snapshot["final_val_pool_size"]),
    "train": float(train[-1]),
    "validation": float(validation[-1]),
    "train_sem": float(train_sem[-1]),
    "validation_sem": float(validation_sem[-1]),
    "gap": abs(float(validation[-1]) - float(train[-1])),
    "err": float(np.hypot(train_sem[-1], validation_sem[-1])),
    "train_per_epoch": [float(v) for v in train],
    "validation_per_epoch": [float(v) for v in validation],
    "train_sem_per_epoch": [float(v) for v in train_sem],
    "validation_sem_per_epoch": [float(v) for v in validation_sem],
    "window_per_epoch": [int(v) for v in snapshot["train_budget_per_epoch"]],
  }


def train_design(trainer, design_scaled, design_index, training_seed, init_params, save, allow_cap):
  """Train one design and return its row. ``save`` is called with the row so far after every epoch,
  so a cell can be read live and a killed one still leaves its curve."""
  window_start = (trainer.train_pool.current, trainer.val_pool.current)
  latest = {}

  def on_epoch(snapshot, sink=latest):
    sink.clear()
    sink.update(snapshot)
    save(curve_row(sink), False)

  started = time.time()
  capped = False
  try:
    result = trainer.train(
      np.asarray(design_scaled, dtype=np.float32), int(training_seed), init_params=init_params, on_epoch=on_epoch,
      step=int(design_index)
    )
  except RuntimeError as error:
    if not allow_cap or CAP_MESSAGE not in str(error):
      raise
    capped, result = True, None
  elapsed = time.time() - started
  if result is None and not capped:
    raise RuntimeError(f"design {design_index}: the budget pool filled, so this design could not be scored")
  if len(latest) == 0:
    raise RuntimeError(f"design {design_index}: no epoch completed, so there is no curve to report")

  row = curve_row(latest)
  if capped:
    row["objective"] = 0.5 * (row["train"] + row["validation"])
    row["objective_std"] = float(np.hypot(row["gap"] / np.sqrt(12.0), 0.5 * row["err"]))
    row["spent"] = (trainer.train_pool.current - window_start[0]) + (trainer.val_pool.current - window_start[1])
  else:
    row["objective"] = float(result.objective_loss)
    row["objective_std"] = float(result.objective_std)
    row["spent"] = int(result.spent)
  row["converged"] = not capped
  row["seconds"] = elapsed
  row["seed"] = int(training_seed)
  row["window_start"] = [int(window_start[0]), int(window_start[1])]
  return row, (None if capped else result.params)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True, help="a finished bo.py run directory (results.json is enough)")
  parser.add_argument(
    "--design", type=int, required=True, help="the FIRST design of the pair; --phase second trains --design + 1"
  )
  parser.add_argument("--phase", required=True, choices=("first", "second", "verify"))
  parser.add_argument(
    "--no-restore", action="store_true", help="phase second: train from an untrained network -- the COLD-START "
    "control that makes the warm start's benefit a difference rather than a level"
  )
  parser.add_argument("--constant-design", default="none", help="'none', 'mid', 'zeros', or a JSON scaled vector")
  parser.add_argument("--detector", default=None, help="override the saved config's detector, by registry name")
  parser.add_argument("--learning-rate", type=float, default=None, help="override the saved config's optimizer learning rate")
  parser.add_argument(
    "--reveal", default=None, choices=("design", "none", "zeros"),
    help="what the network is TOLD (Trainer.reveal). 'none' gives the detector's design-free features -- for the "
    "straw family the hit ADDRESS [TDC, station, view, layer, straw] instead of resolved geometry. This is the "
    "principled replacement for --constant-design, which hacked the same effect by wrapping combine_scaled."
  )
  parser.add_argument("--activation", default=None, help="override the saved config's regressor activation")
  parser.add_argument("--features", default=None, help="override the saved config's regressor features, as JSON")
  parser.add_argument("--loss-precision", type=float, default=None, help="override the saved config's training.loss_precision")
  parser.add_argument("--allow-cap", action="store_true", help="record a cell that reaches iteration_limit as unconverged")
  parser.add_argument("--device", default=None, help="override the config's device (a CPU dry run needs it)")
  parser.add_argument("--config", default=None, help="run config name, for a trajectory that predates the saved one")
  parser.add_argument("--parameters", default=None, help="where phase first writes and phase second reads the network")
  parser.add_argument("--verify-events", type=int, default=256, help="events the override check is measured on")
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  trajectory = json.load(open(os.path.join(arguments.run, "results.json")))
  config = load_run_config(trajectory, arguments.config)
  if arguments.device is not None:
    config = {**config, "device": arguments.device}
  results = trajectory["results"]
  designs_scaled = [row["x_scaled"] for row in results]
  spends = [int(row["spent"]) for row in results]
  design_index = arguments.design + (1 if arguments.phase == "second" else 0)
  if arguments.design < 1:
    raise SystemExit("--design must be >= 1: the pair is taken mid-trajectory, past the initial draw")
  if design_index >= len(results):
    raise SystemExit(f"design {design_index} is past the run's {len(results)} designs")
  bo_config = config["bo"]
  n_init = int(bo_config.get("n_init", bo_config["gp"]["n_folds"]))
  print(
    f"[trajectory] n_init = {n_init} ({'bo.n_init' if 'n_init' in bo_config else 'bo.gp.n_folds, bo.n_init absent'}), "
    f"so designs 0..{n_init - 1} are the Sobol draw and the pair {arguments.design}->{arguments.design + 1} is past it",
    flush=True
  )

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

  if arguments.detector is not None:
    print(f"[config] detector {list(config['detector'])[0]!r} -> {arguments.detector!r}", flush=True)
    config = {**config, "detector": {arguments.detector: list(config["detector"].values())[0]}}

  if arguments.learning_rate is not None:
    name, block = detopt.utils.config.split(config["training"]["optimizer"])
    print(f"[config] optimizer {name} learning_rate {block.get('learning_rate')} -> {arguments.learning_rate}", flush=True)
    optimizer = {name: {**block, "learning_rate": float(arguments.learning_rate)}}
    config = {**config, "training": {**config["training"], "optimizer": optimizer}}

  if arguments.reveal is not None:
    if arguments.constant_design != "none":
      raise SystemExit("--reveal and --constant-design both remove the design; pass one, not both")
    config = {**config, "training": {**config["training"], "reveal": arguments.reveal}}
    print(f"[config] training.reveal -> {arguments.reveal!r}", flush=True)

  detector = detopt.detector.from_config(config["detector"])
  constant_scaled = None
  if arguments.constant_design != "none":
    constant_scaled = constant_scaled_design(detector, arguments.constant_design)

  os.makedirs(arguments.output, exist_ok=True)
  verification = None
  if constant_scaled is not None or arguments.phase == "verify":
    check_constant = constant_scaled if constant_scaled is not None else constant_scaled_design(detector, "mid")
    verification = verify_override(
      detector, check_constant, designs_scaled[design_index], designs_scaled[design_index - 1], arguments.verify_events
    )
    json.dump(verification, open(os.path.join(arguments.output, "override_check.json"), "w"), indent=1)
  if arguments.phase == "verify":
    print(f"\nwrote {os.path.join(arguments.output, 'override_check.json')}")
    return
  if constant_scaled is not None:
    detector.combine_scaled = constant_design_combine(detector, constant_scaled)
    print(f"[override] the network is told the CONSTANT scaled design {list(map(float, constant_scaled))}", flush=True)

  run_seed = int(os.path.basename(os.path.dirname(os.path.normpath(arguments.run))))
  seed = network_seed(run_seed)
  training_seed = run_iteration_seed(run_seed, design_index)
  parameters_path = arguments.parameters
  if parameters_path is None:
    parameters_path = os.path.join(arguments.output, "first_parameters.npz")

  print(
    f"probe: {arguments.run} phase {arguments.phase} design {design_index} "
    f"({design_index} designs replayed, {sum(spends[:design_index])} events), run seed {run_seed}, "
    f"trainer seed {seed}, training seed {training_seed}, loss_precision {config['training']['loss_precision']}", flush=True
  )
  print(
    f"  the run itself reported {results[design_index]['loss']:.4f} +- {results[design_index]['loss_std']:.4f} "
    f"for this design at spent={spends[design_index]}", flush=True
  )

  trainer = build_trainer("continue", detector, config, seed, 0.0)
  print(f"[pool] filling with designs 0..{design_index - 1}", flush=True)
  started = time.time()
  window_start = replay_pool(trainer, detector, designs_scaled[:design_index], spends[:design_index])
  print(f"[pool] at {window_start[0]} train / {window_start[1]} val after {time.time() - started:.0f}s", flush=True)

  init_params, restored = None, None
  if arguments.phase == "second" and not arguments.no_restore:
    init_params, n_leaves, checksum = restore_parameters(trainer, parameters_path)
    restored = {"path": parameters_path, "leaves": n_leaves, "abs_sum": checksum}
    print(f"[restore] {n_leaves} leaves from {parameters_path}, sum|leaf| = {checksum:.6f}", flush=True)
  elif arguments.phase == "second":
    print("[restore] SKIPPED (--no-restore): design n+1 is trained from an untrained network", flush=True)

  path = os.path.join(arguments.output, f"phase_{arguments.phase}.json")

  def save(row, complete):
    payload = {
      "run": arguments.run,
      "phase": arguments.phase,
      "pair": [arguments.design, arguments.design + 1],
      "design": design_index,
      "restore": bool(arguments.phase == "second" and not arguments.no_restore),
      "restored": restored,
      "activation": arguments.activation,
      "constant_design": arguments.constant_design,
      "constant_scaled": None if constant_scaled is None else [float(v) for v in constant_scaled],
      "override_check": verification,
      "n_init": n_init,
      "loss_precision": float(config["training"]["loss_precision"]),
      "training": config["training"],
      "detector": config["detector"],
      "regressor": config["regressor"],
      "trainer_seed": seed,
      "training_seed": int(training_seed),
      "reported_loss": float(results[design_index]["loss"]),
      "reported_loss_std": float(results[design_index]["loss_std"]),
      "reported_spent": int(spends[design_index]),
      "complete": bool(complete),
      "row": row,
    }
    with open(path + ".tmp", "w") as handle:
      json.dump(payload, handle, indent=1)
    os.replace(path + ".tmp", path)

  row, parameters = train_design(
    trainer, designs_scaled[design_index], design_index, training_seed, init_params, save, arguments.allow_cap
  )
  save(row, True)
  print(
    f"[{arguments.phase}] objective={row['objective']:.4f} +- {row['objective_std']:.4f} spent={row['spent']} "
    f"epochs={row['epochs']} window={row['window']} gap={row['gap']:.4f} err={row['err']:.4f} "
    f"{'CONVERGED' if row['converged'] else 'CAPPED'} ({row['seconds']:.0f}s)", flush=True
  )
  if arguments.phase == "first":
    if parameters is None:
      raise SystemExit("phase first did not converge, so there is no trained network for phase second to restore")
    n_leaves, checksum = save_parameters(parameters_path, parameters)
    print(f"[save] {n_leaves} leaves to {parameters_path}, sum|leaf| = {checksum:.6f}", flush=True)
  print(f"\nwrote {path}")


if __name__ == "__main__":
  main()
