#!/usr/bin/env python3
"""Blind transfer between consecutive BO designs, under one task REPRESENTATION.

    python scripts/probe_strip_transfer.py --run <cell> --pair 7 --representation strip \
        --reveal none --output <dir>

FOUR MEASUREMENTS PER PAIR ``(i, i+1)``, all under the growth procedure:

    cold_i       train from scratch at design i                       -> the reference
    blind_i1     that SAME network evaluated at design i+1, NO TRAINING -> the transfer
    warm_i1      that same network trained on to design i+1            -> what `continue` does
    warm_head_i1 that same network onto i+1 with the BACKBONE FROZEN  -> is only the read-out stale?
    cold_i1      train from scratch at design i+1                      -> the control warm is judged against
    meta_i1      fresh at i+1 but with the design REVEALED              -> does knowing the design help?

``blind_i1`` is the quantity the campaigns never measure: `continue` carries a network across the
design boundary and immediately trains it, so how much of the previous fit still applied is never
seen. Here it is, and ``cold_i1`` is what says whether the carry was worth anything.

⚠️ THE WARM START IS RE-DERIVED, NOT RESTORED. It continues the network THIS probe trained at design
i, not the campaign's checkpoint. Two reasons, and the first is decisive: 64 of the 105 per-design
checkpoints in `ship-addr-prec2e2`'s `continue` arms were never committed (no `commit_success.txt`,
no `_CHECKPOINT_METADATA`), so restoring is impossible for most pairs. The second is that re-deriving
makes cold and warm share a byte-identical starting point, which the campaign's checkpoint could not
guarantee -- it was trained under a different representation.

THE DESIGNS COME FROM A COMPLETED RUN AND NOTHING ELSE DOES. ``--run`` supplies ``x_scaled`` per
iteration; its detector, regressor and losses are all replaced. So the designs are a realistic BO
trajectory while the task representation under test is whatever ``--representation`` names.
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import optax
import matplotlib

matplotlib.use("AGG")

import jax
import jax.numpy as jnp
from flax import nnx

import detopt
import detopt.detector
import detopt.utils.tensor as tensor
from detopt.nn.trainer import ContinualTrainer, DesignTrainer

CAP_MESSAGE = "did not reach precision within iteration_limit"

REPRESENTATION = {
  "strip": ("stereo_strip", {
    "strip-regressor": {}
  }),
  "image": ("stereo_image", {
    "alpha-conv-regressor": {
      "channels": [32, 24, 16, 12],
      "blocks": 3,
      "kernel_size": 3
    }
  }),
  "address": ("stereo_address_design", {
    "set-regressor": {
      "features": [[16, 24], [24, 16]],
      "n_models": None
    }
  }),
  "layerset": ("stereo_layer_set", {
    "set-regressor": {
      "features": [[24, 16], [16, 24]],
      "n_models": None
    }
  }),
}


def curve_row(snapshot):
  train, validation = snapshot["train_loss_per_epoch"], snapshot["val_loss_per_epoch"]
  train_sem, validation_sem = snapshot["train_sem_per_epoch"], snapshot["val_sem_per_epoch"]
  return {
    "epochs": int(train.shape[0]),
    "window": int(snapshot["final_train_budget"]),
    "train": float(train[-1]),
    "validation": float(validation[-1]),
    "gap": abs(float(validation[-1]) - float(train[-1])),
    "err": float(np.hypot(train_sem[-1], validation_sem[-1])),
    "train_loss_per_epoch": np.asarray(train, np.float64).tolist(),
    "val_loss_per_epoch": np.asarray(validation, np.float64).tolist(),
    "train_sem_per_epoch": np.asarray(train_sem, np.float64).tolist(),
    "val_sem_per_epoch": np.asarray(validation_sem, np.float64).tolist(),
    "window_per_epoch": np.asarray(snapshot["train_budget_per_epoch"], np.int64).tolist(),
  }


def write_curves(report, output):
  """One ``bo.py``-style convergence figure per trained arm, drawn from the curves in ``report``.

    Reuses :func:`detopt.utils.viz.bo.plot_iteration` so the probe's figures and the driver's are the
    same plot. It names its output by iteration index, so each arm is drawn under its own index and
    the file is then renamed to the arm; the design sidecar it writes is dropped, since every number
    the probe knows is already in its own JSON.
    """
  from detopt.utils.viz.bo import plot_iteration

  written = []
  arms = [name for name, row in report.items() if isinstance(row, dict) and "train_loss_per_epoch" in row]
  for index, name in enumerate(arms):
    row = report[name]
    history = {
      "train_loss_per_epoch": np.asarray(row["train_loss_per_epoch"], np.float64),
      "val_loss_per_epoch": np.asarray(row["val_loss_per_epoch"], np.float64),
      "train_sem_per_epoch": np.asarray(row["train_sem_per_epoch"], np.float64),
      "val_sem_per_epoch": np.asarray(row["val_sem_per_epoch"], np.float64),
      "train_budget_per_epoch": np.asarray(row["window_per_epoch"], np.int64),
      "final_train_budget": int(row["window"]),
    }
    drawn = plot_iteration(history, index, {"arm": name}, float(row["validation"]), output)
    target = os.path.join(output, f"curve_{name}.png")
    os.replace(drawn, target)
    sidecar = drawn.replace(".png", "_design.json")
    if os.path.exists(sidecar):
      os.remove(sidecar)
    written.append(target)
  return written


HEAD = "output"


def head_only(optimizer, params):
  """``optimizer`` on the final layer, ``optax.set_to_zero`` on every other leaf.

    Returns ``(transform, trainable, frozen)``; the counts are for the log only.

    ⚠️ THE SELECTOR IS A CALLABLE, NOT A PRECOMPUTED TREE. A tree built here from the CARRIED
    parameters need not match the tree the trainer actually optimises, and when it does not,
    `multi_transform` partitions wrongly and `optax.chain` raises "The number of updates and states has
    to be the same in chain!" at the first update -- which killed 9 cells on 2026-08-22. A callable is
    evaluated on whatever tree is passed, at BOTH init and update, so the two cannot disagree.

    It is a multi_transform LABEL TREE, not a mask: here a mask is the per-hit element mask."""

  def labels_of(tree):

    def label(path, _leaf):
      root = getattr(path[0], 'key', None) if len(path) > 0 else None
      return 'head' if str(root) == HEAD else 'frozen'

    return jax.tree_util.tree_map_with_path(label, tree)

  counts = {'head': 0, 'frozen': 0}
  for value in jax.tree.leaves(labels_of(params), is_leaf=lambda x: isinstance(x, str)):
    counts[value] = counts.get(value, 0) + 1
  transform = optax.multi_transform({'head': optimizer, 'frozen': optax.set_to_zero()}, labels_of)
  return transform, counts['head'], counts['frozen']


def head_only_trainer(detector, config, seed):
  """A ``DesignTrainer`` built with a HEAD-ONLY optimiser -- composed BEFORE construction, not swapped
    onto a built trainer.

    ⚠️ DO NOT go back to assigning ``trainer.optimizer`` after ``from_config``. ``Trainer.__init__``
    eagerly builds ``_train_epoch``, which CLOSES OVER the optimiser (``common.py:287``), so a later
    swap leaves the epoch calling the ORIGINAL transform against a state the NEW one produced --
    "The number of updates and states has to be the same in chain!". That killed 9 cells on
    2026-08-22. Passing ``optimizer=`` to the constructor means there is only ever one.

    Mirrors :meth:`DesignTrainer.from_config`; the ONLY difference is the composed optimiser."""
  from detopt.utils.config import optimizer as make_optimizer, resolve_device

  training = {k: v for k, v in config["training"].items() if k != "optimizer"}
  base = make_optimizer(config["training"]["optimizer"])
  probe = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=seed)
  _graphdef, params, _state = probe._build_regressor(probe.seed)
  frozen_optimizer, trainable, frozen = head_only(base, params)
  trainer = DesignTrainer(
    detector, regressor_config=config["regressor"], optimizer=frozen_optimizer, device=resolve_device(config.get("device")),
    checkpoint_dir=None, seed=seed, **training
  )
  return trainer, trainable, frozen


def train_design(trainer, design_scaled, index, training_seed, init_params, *, output=None, name=None, stride=8):
  """One design trained to the exit test. Returns ``(row, params)``; ``params`` is None if capped.

    With ``output`` and ``name``, the curve and its figure are rewritten every ``stride`` epochs
    DURING training, not only once the design exits. A probe that runs for hours and writes only at
    the end has nothing to show while it runs and loses everything if it is cancelled -- which is
    how three coarse cells finished with neither plot nor data.
    """
  start = (trainer.train_pool.current, trainer.val_pool.current)
  latest = {}
  began = time.time()
  capped = False

  def _write_partial(partial):
    """Render and dump one partial curve. Runs OFF the training loop."""
    write_curves({name: partial}, output)
    with open(os.path.join(output, f"partial_{name}.json"), "w") as handle:
      json.dump(partial, handle, indent=1)

  def _observe(snapshot):
    latest.clear()
    latest.update(snapshot)
    if output is None or name is None:
      return
    epoch = int(snapshot["val_loss_per_epoch"].size)
    if epoch != 1 and epoch % stride != 0:
      return
    # `curve_row` COPIES the arrays into lists here, in the training thread, so the writer cannot
    # race the trainer's next epoch. Only the render -- the expensive half -- is handed off, and it
    # is fire-and-forget: a dropped partial costs one stride, never an epoch of training.
    partial = curve_row(snapshot)
    partial["epochs_so_far"] = epoch
    partial["complete"] = False
    partial["seconds"] = time.time() - began
    threading.Thread(target=_write_partial, args=(partial, ), daemon=True).start()

  try:
    result = trainer.train(
      np.asarray(design_scaled, np.float32), int(training_seed), init_params=init_params, on_epoch=_observe, step=int(index)
    )
  except RuntimeError as error:
    if CAP_MESSAGE not in str(error):
      raise
    capped, result = True, None
  if len(latest) == 0:
    raise RuntimeError(f"design {index}: no epoch completed")
  row = curve_row(latest)
  row["objective"] = 0.5 * (row["train"] + row["validation"]) if capped else float(result.objective_loss)
  row["spent"] = ((trainer.train_pool.current - start[0]) +
                  (trainer.val_pool.current - start[1]) if capped else int(result.spent))
  row["converged"] = not capped
  row["seconds"] = time.time() - began
  return row, (None if capped else result.params)


def evaluate(trainer, detector, params, design_scaled, indices, batch):
  """Mean loss of ``params`` at ``design_scaled`` on fresh events. No gradient, no training."""
  graphdef, _p, state = trainer._build_regressor(trainer.seed)
  scaled = jnp.asarray(design_scaled, jnp.float32)
  total, count = 0.0, 0
  for start in range(0, len(indices), batch):
    chunk = indices[start:start + batch]
    physical = jax.tree.map(
      lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (len(chunk), ) + jnp.asarray(x).shape), detector.to_nominal(scaled)
    )
    _truth, event, mask, target = detector(physical, chunk)
    features = trainer._combine(event, scaled, mask)
    model = nnx.merge(graphdef, params, state)
    predicted = model(features, detector.element_mask(event, mask), deterministic=True)
    loss = detector.loss(predicted, detector.normalize_target(target))
    total += float(jnp.sum(loss)) if jnp.ndim(loss) else float(loss) * len(chunk)
    count += len(chunk)
  return total / count


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, help="a completed cell; supplies the design trajectory only")
  parser.add_argument("--pair", type=int, required=True, help="the source iteration i; the pair is (i, i+1)")
  parser.add_argument("--representation", choices=sorted(REPRESENTATION), default="strip")
  parser.add_argument("--reveal", choices=("none", "design", "zeros"), default="none")
  parser.add_argument("--n-events", type=int, default=128 * 1024, help="events for the no-training evaluation")
  parser.add_argument("--batch", type=int, default=4096, help="batch for the NO-TRAINING evaluation, not for training")
  parser.add_argument("--train-batch", type=int, default=None, help="override training.batch (also sets steps_per_epoch)")
  parser.add_argument("--eval-batch", type=int, default=None, help="override training.eval_batch")
  parser.add_argument("--event-offset", type=int, default=3_000_000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default=None)
  parser.add_argument("--n0", type=int, default=None, help="override training.n0 (a smoke test wants this small)")
  parser.add_argument("--n-increment", type=int, default=None, help="override training.n_increment")
  parser.add_argument("--iteration-limit", type=int, default=None, help="override training.iteration_limit")
  parser.add_argument("--loss-precision", type=float, default=None, help="override training.loss_precision")
  parser.add_argument("--budget", type=int, default=None, help="override training.budget (the pool capacity)")
  parser.add_argument(
    "--channels", default=None, help="comma-separated block widths for the regressor, e.g. 16,12,8,6; the network's WIDTH "
    "is the only thing this changes"
  )
  parser.add_argument("--learning-rate", type=float, default=None, help="override the optimizer's learning rate")
  parser.add_argument(
    "--only-meta", action="store_true", help="run ONLY cold_i then meta_i1: sample spent(cold_i)/2 detector calls at every "
    "design before i into the pool, then train i+1 from cold_i's network with replay"
  )
  parser.add_argument(
    "--only-head", action="store_true", help="run ONLY cold_i then warm_head_i1 (the backbone-frozen warm start)"
  )
  parser.add_argument(
    "--only-control", action="store_true",
    help="run ONLY cold_i1, the fresh-network control at design i+1; it needs no carried "
    "network, so it can complete a run whose earlier rounds already finished"
  )
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  with open(os.path.join(arguments.run, "results.json")) as handle:
    payload = json.load(handle)
  results = payload["results"]
  i = arguments.pair
  if i + 1 >= len(results):
    raise SystemExit(f"pair {i} needs design {i+1} but the run has {len(results)}")

  name, regressor = REPRESENTATION[arguments.representation]
  block = list(payload["config"]["detector"].values())[0]
  config = {
    **payload["config"], "detector": {
      name: block
    },
    "regressor": regressor,
    "training": {
      **payload["config"]["training"], "reveal": arguments.reveal
    },
  }
  if arguments.device is not None:
    config["device"] = arguments.device
  for key, value in (("n0", arguments.n0), ("n_increment", arguments.n_increment),
                     ("iteration_limit", arguments.iteration_limit), ("loss_precision", arguments.loss_precision),
                     ("budget", arguments.budget), ("batch", arguments.train_batch), ("eval_batch", arguments.eval_batch)):
    if value is not None:
      print(f"[config] training.{key} {config['training'][key]} -> {value}", flush=True)
      config["training"] = {**config["training"], key: value}
  if arguments.learning_rate is not None:
    optimizer_name = list(config["training"]["optimizer"])[0]
    optimizer_block = config["training"]["optimizer"][optimizer_name]
    print(
      f"[config] optimizer.{optimizer_name}.learning_rate {optimizer_block['learning_rate']} -> "
      f"{arguments.learning_rate}", flush=True
    )
    config["training"] = {
      **config["training"], "optimizer": {
        optimizer_name: {
          **optimizer_block, "learning_rate": arguments.learning_rate
        }
      }
    }
  if arguments.channels is not None:
    widths = [int(c) for c in arguments.channels.split(",")]
    regressor_name = list(regressor)[0]
    print(f"[config] {regressor_name}.channels -> {widths}", flush=True)
    config = {**config, "regressor": {regressor_name: {**regressor[regressor_name], "channels": widths}}}
  detector = detopt.detector.from_config(config["detector"])
  print(
    f"[repr] {arguments.representation} -> {name}, reveal={arguments.reveal}, "
    f"features {detector.combined_event_shape(arguments.reveal != 'none')}", flush=True
  )

  design_i = np.asarray(results[i]["x_scaled"], np.float32)
  design_j = np.asarray(results[i + 1]["x_scaled"], np.float32)
  indices = np.arange(arguments.event_offset, arguments.event_offset + arguments.n_events, dtype=np.int32)
  plot_stride = int(config.get("plot_per_epoch", 8))
  os.makedirs(arguments.output, exist_ok=True)
  report = {
    "run": arguments.run,
    "pair": [i, i + 1],
    "representation": arguments.representation,
    "reveal": arguments.reveal,
    "n_events": int(arguments.n_events),
    "design_distance": float(np.linalg.norm(design_j - design_i)),
    "reported_at_i": float(results[i]["loss"]),
    "reported_at_i1": float(results[i + 1]["loss"]),
  }

  def flush():
    with open(os.path.join(arguments.output, "strip_transfer.json"), "w") as handle:
      json.dump(report, handle, indent=1)

  if arguments.only_control:
    cold = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
    report["cold_i1"], _ = train_design(
      cold, design_j, i + 1, arguments.seed + 1, None, output=arguments.output, stride=plot_stride, name="cold_i1"
    )
    print(
      f"[cold_i1]  {report['cold_i1']['objective']:.5f}  window {report['cold_i1']['window']}  "
      f"spent {report['cold_i1']['spent']}  converged {report['cold_i1']['converged']}", flush=True
    )
    with open(os.path.join(arguments.output, "control.json"), "w") as handle:
      json.dump(report, handle, indent=1)
    for figure in write_curves(report, arguments.output):
      print(f"wrote {figure}", flush=True)
    print(f"wrote {os.path.join(arguments.output, 'control.json')}", flush=True)
    return

  if arguments.only_meta:
    # META'S START IS ITS OWN COLD RUN, WITH THE DESIGN REVEALED. It cannot be continue's network: a
    # revealed network takes wider per-hit features, so the blind run's parameters do not fit its first
    # layer. CONTEXT IS DATA, NOT TRAINING -- every design before i contributes `n // 2` detector calls
    # to the pool, `n` being what this cold run consumed, and the continual trainer's replay half then
    # draws from exactly that history, the rows in [0, w0).
    revealed = {**config, "training": {**config["training"], "reveal": "design"}}
    report["reveal"] = "design"
    print(f"[meta] start: cold at design {i}, reveal=design, features {detector.combined_event_shape(True)}", flush=True)
    start_trainer = DesignTrainer.from_config(detector, revealed, checkpoint_dir=None, seed=arguments.seed)
    report["cold_i_revealed"], params_i = train_design(
      start_trainer, design_i, i, arguments.seed, None, output=arguments.output, stride=plot_stride, name="cold_i_revealed"
    )
    print(
      f"[cold_i/revealed] {report['cold_i_revealed']['objective']:.5f}  window "
      f"{report['cold_i_revealed']['window']}  spent {report['cold_i_revealed']['spent']}", flush=True
    )
    if params_i is None:
      print("[abort] the revealed cold run CAPPED: no network, so meta has no start", flush=True)
      report["aborted"] = "cold_i_revealed capped"
      with open(os.path.join(arguments.output, "meta.json"), "w") as handle:
        json.dump(report, handle, indent=1)
      return

    context_spend = int(report["cold_i_revealed"]["spent"]) // 2
    val_fraction = float(config["training"].get("val_fraction", 0.25))
    # EVERY CONTEXT CALL GOES TO THE TRAIN POOL, because that is the only pool replay reads: a batch is
    # half the current window and half [0, w0_train), while the validation pool is read from the CURRENT
    # design's offset forward. A validation history would be sampled and never looked at.
    needed = int(np.ceil((i * context_spend + int(config["training"]["iteration_limit"])) / (1.0 - val_fraction) * 1.02))
    budget = max(int(config["training"]["budget"]), needed)
    context_config = {**revealed, "training": {**revealed["training"], "budget": budget}}
    print(
      f"[meta] context: {i} designs x {context_spend} train calls, "
      f"n={report['cold_i_revealed']['spent']}, pool budget {budget}", flush=True
    )

    meta_trainer = ContinualTrainer.from_config(detector, context_config, checkpoint_dir=None, seed=arguments.seed)
    report["context"] = []
    for k in range(i):
      design_k = detector.to_nominal(np.asarray(results[k]["x_scaled"], np.float32))
      meta_trainer._fill_pool(design_k, meta_trainer.train_pool, context_spend, meta_trainer._train_index)
      report["context"].append({"design": k, "train": context_spend})
      print(f"[context]  design {k}: train pool {meta_trainer.train_pool.current}", flush=True)
    report["context_spent_total"] = int(meta_trainer.train_pool.current)

    # The start network, params only -- the buffer state stays fresh, exactly as a warm start does.
    meta_trainer._running = (jax.device_put(params_i, meta_trainer.device), meta_trainer._running[1])
    report["meta_i1"], _ = train_design(
      meta_trainer, design_j, i + 1, arguments.seed + 1, None, output=arguments.output, stride=plot_stride, name="meta_i1"
    )
    print(
      f"[meta_i1]  {report['meta_i1']['objective']:.5f}  window {report['meta_i1']['window']}  "
      f"spent {report['meta_i1']['spent']}  converged {report['meta_i1']['converged']}", flush=True
    )
    with open(os.path.join(arguments.output, "meta.json"), "w") as handle:
      json.dump(report, handle, indent=1)
    for figure in write_curves(report, arguments.output):
      print(f"wrote {figure}", flush=True)
    print(f"wrote {os.path.join(arguments.output, 'meta.json')}", flush=True)
    return

  if arguments.only_head:
    trainer = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
    report["cold_i"], params_i = train_design(
      trainer, design_i, i, arguments.seed, None, output=arguments.output, stride=plot_stride, name="cold_i"
    )
    print(f"[cold_i]   {report['cold_i']['objective']:.5f}  window {report['cold_i']['window']}", flush=True)
    if params_i is None:
      print("[abort] cold_i CAPPED: no carried network, so no head-only start exists", flush=True)
      report["aborted"] = "cold_i capped"
    else:
      warm_head, trainable, frozen = head_only_trainer(detector, config, arguments.seed)
      report["head_leaves"], report["frozen_leaves"] = int(trainable), int(frozen)
      print(f"[freeze]   head leaves {trainable}, frozen leaves {frozen}", flush=True)
      report["warm_head_i1"], _ = train_design(
        warm_head, design_j, i + 1, arguments.seed + 1, params_i, output=arguments.output, stride=plot_stride,
        name="warm_head_i1"
      )
      print(
        f"[warm_head_i1] {report['warm_head_i1']['objective']:.5f}  "
        f"window {report['warm_head_i1']['window']}  spent {report['warm_head_i1']['spent']}  "
        f"converged {report['warm_head_i1']['converged']}", flush=True
      )
    with open(os.path.join(arguments.output, "head.json"), "w") as handle:
      json.dump(report, handle, indent=1)
    for figure in write_curves(report, arguments.output):
      print(f"wrote {figure}", flush=True)
    print(f"wrote {os.path.join(arguments.output, 'head.json')}", flush=True)
    return

  trainer = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
  report["cold_i"], params_i = train_design(
    trainer, design_i, i, arguments.seed, None, output=arguments.output, stride=plot_stride, name="cold_i"
  )
  print(f"[cold_i]   {report['cold_i']['objective']:.5f}  spent {report['cold_i']['spent']}", flush=True)
  flush()

  if params_i is not None:
    report["blind_i1"] = evaluate(trainer, detector, params_i, design_j, indices, arguments.batch)
    report["blind_i"] = evaluate(trainer, detector, params_i, design_i, indices, arguments.batch)
    print(f"[blind]    on i {report['blind_i']:.5f}   on i+1 {report['blind_i1']:.5f}", flush=True)
    flush()

  warm = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
  report["warm_i1"], _ = train_design(
    warm, design_j, i + 1, arguments.seed + 1, params_i, output=arguments.output, stride=plot_stride, name="warm_i1"
  )
  print(f"[warm_i1]  {report['warm_i1']['objective']:.5f}  spent {report['warm_i1']['spent']}", flush=True)
  flush()

  warm_head, trainable, frozen = head_only_trainer(detector, config, arguments.seed)
  report["head_leaves"], report["frozen_leaves"] = int(trainable), int(frozen)
  if trainable == 0 or frozen == 0:
    sys.exit(
      f"the label tree selected {trainable} trainable and {frozen} frozen leaves; "
      f"'{HEAD}' did not name this model's last layer"
    )
  print(f"[freeze]   head leaves {trainable}, frozen leaves {frozen}", flush=True)
  report["warm_head_i1"], _ = train_design(
    warm_head, design_j, i + 1, arguments.seed + 1, params_i, output=arguments.output, stride=plot_stride, name="warm_head_i1"
  )
  print(f"[warm_head_i1] {report['warm_head_i1']['objective']:.5f}  spent {report['warm_head_i1']['spent']}", flush=True)
  flush()

  cold = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
  report["cold_i1"], _ = train_design(
    cold, design_j, i + 1, arguments.seed + 1, None, output=arguments.output, stride=plot_stride, name="cold_i1"
  )
  print(f"[cold_i1]  {report['cold_i1']['objective']:.5f}  spent {report['cold_i1']['spent']}", flush=True)
  flush()
  print(f"wrote {os.path.join(arguments.output, 'strip_transfer.json')}", flush=True)


if __name__ == "__main__":
  main()
