#!/usr/bin/env python3
"""Is the META HANDICAP a BIAS of carrying a network, or is the network simply UNDER CAPACITY?

    python scripts/probe_meta_capacity.py =enzyme_extremes --width narrow \
        --features '[[24,16],[16,24]]' --eval-index 0 --seed 1 \
        --output output/screen/metacap-narrow-e0.json

THE QUESTION. A campaign scores each candidate design either with a FRESH network per design
(`from_scratch`, :class:`DesignTrainer`) or with ONE PERSISTENT network carried across designs
(`meta`, :class:`ContinualTrainer` -- carried parameters AND experience replay). `meta` reports a
WORSE loss at the same design. Two explanations compete:

  (a) BIAS -- carrying a network across designs costs something intrinsically (replay dilutes the
      gradient on the design being scored, the network is pulled toward the design-MARGINAL answer);
  (b) CAPACITY -- one network of this size cannot hold many designs at once.

They are separated by running the arms at TWO WIDTHS. (a) predicts the handicap survives widening;
(b) predicts it shrinks.

THE CONSTRUCTION. Eleven designs from the task's own Sobol landscape at evenly spaced QUANTILES of
the proxy loss over a stated band (default 0.50 to 0.75) -- the MID-RANGE, where a search actually
sits mid-run. That band is chosen against a measured trap: designs drawn uniformly from the box sit
at the no-information ceiling, where every arm reports the same number and no probe can resolve
anything. ONE is held out as the EVALUATION design; the other TEN are HISTORY.

The history is not walked sequentially -- it is POOLED. Every history design's events go into one
training set and ONE network is trained on all of it JOINTLY (no per-design growth loop, no
convergence procedure, one plain settle criterion), so "a network that already carries ten designs"
is produced in one pass rather than reconstructed from a whole BO path. That pretrained network is
then the initial network of the arms below, and its own loss ON the pooled history is reported --
without it there is no evidence the network ever learned the ten designs it is supposed to carry.

THE TWO ARMS, both scoring the SAME evaluation design through the SAME per-design procedure:

  from_scratch  DesignTrainer, fresh initialisation, the evaluation design's data only.
  meta          ContinualTrainer whose persistent network IS the pretrained one and whose pool is
                PRE-FILLED with the ten history designs, so its replay half draws from them. The
                faithful campaign arm -- carried parameters AND real replay, at the shipped
                ``replay_weight`` of 1.0.

THE HEADLINE IS THE INTERACTION, not either difference: ``meta - from_scratch`` at each width, and
how that CHANGES between the widths. If the handicap shrinks materially when the network is widened,
capacity is the binding constraint. If it does not, the handicap belongs to the continual strategy
and widening is not the lever. Each arm's own improvement with width is reported alongside, because
a `from_scratch` arm that also improves a lot says the network is under capacity for a SINGLE design
and the finding is not specific to `meta`.

WHAT IS HELD IDENTICAL across the arms, because the difference being reported is smaller than the
noise these would otherwise inject: the design; the training rng (one child ``SeedSequence``); and
the EVENTS -- both arms consume the same measurement index stream at the same offset, which for
`meta` means its index array is built as ``[history stream, measurement stream]`` so the evaluation
design's window opens on exactly the events `from_scratch` sees. The history events come from a
SEPARATE stream, so the pretrained network has not already seen the evaluation window's compounds at
another design.

WHAT IS REPORTED per arm: the evaluation design's TRAIN and VALIDATION loss SEPARATELY (validation
is the objective; ``(train + val) / 2`` alone hides which side moved), their gap, the achieved slack
``|val - train| + hypot(train_sem, val_sem)`` that `loss_precision` is compared against, the window
reached, whether it CONVERGED or exhausted its window allowance, epochs, detector calls and wall
clock.

This script SETS NOTHING and decides nothing: it writes a JSON of measurements.
"""

import argparse
import gc
import json
import math
import os
import time

import matplotlib

matplotlib.use("AGG")

import numpy as np

import jax
import jax.numpy as jnp

import detopt
import detopt.detector
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.nn.trainer.common import fresh_design_network
from detopt.utils.events import shuffled_event_index
from detopt.utils.training import masked_mean_sem

# Already written, and each already documents why it is what it is -- reuse rather than keep a
# second copy. `_per_window` collapses the epoch history per data-addition round (and states why the
# stopping epoch's `diff` understates the gap); `fill` is the LARGE-CHUNK pool fill (the detector is
# latency-bound, so the trainer's own 256-event chunk is many times slower for the same events);
# `settled` is the block-mean stopping rule, with the measurement behind its tolerance.
from probe_dropout import _per_window
from probe_meta_cripple import fill, settled

# The HISTORY event stream is drawn from its own seed, offset from the run seed by this constant, so
# the pretrained network has not seen the evaluation window's (enzyme, compound) draws at another
# design. The analytic source draws indices uniformly from [0, 2**31), so two streams of ~5e5 draws
# overlap in ~1e2 events, i.e. ~0.02% of the evaluation window -- stated rather than argued away.
HISTORY_SEED_OFFSET = 10007


def load_config(name):
  """The RUN config and its detector config, resolved the way gearup resolves them."""
  import yaml

  name = name.lstrip("=")
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  return config, detector_config


def design_band(landscape, detector, q_low, q_high, n_designs):
  """``n_designs`` landscape designs at evenly spaced QUANTILES of the proxy loss in ``[q_low, q_high]``.

  The landscape is used ONLY to CHOOSE designs -- every number this script reports is measured fresh
  through the neural trainer -- so a stale proxy costs a less well-placed band and nothing else.
  """
  with np.load(landscape) as data:
    designs, losses = np.asarray(data["designs"], np.float32), np.asarray(data["losses"], np.float64)
  if designs.shape[1] != int(detector.design_dim()):
    raise SystemExit(
      f"probe_meta_capacity: {landscape} holds {designs.shape[1]}-dimensional designs but "
      f"the detector's design is {int(detector.design_dim())}-dimensional"
    )
  order = np.argsort(losses)  # ascending loss: best first
  n = len(order)
  chosen = []
  for position, q in enumerate(np.linspace(float(q_low), float(q_high), int(n_designs))):
    rank = int(np.clip(round(q * (n - 1)), 0, n - 1))
    index = int(order[rank])
    chosen.append({
      "label": f"q{q:.3f}",
      "position": position,  # index into the band, hence into the per-design seed sequences
      "quantile": float(q),
      "rank": rank,
      "index": index,
      "x_scaled": np.asarray(designs[index], np.float32),
      "proxy_loss": float(losses[index]),
    })
  ranks = [c["rank"] for c in chosen]
  if len(set(ranks)) != len(ranks):
    raise SystemExit(f"probe_meta_capacity: the quantile band produced repeated ranks {ranks}")
  return chosen


def run_config_for(
  config, *, features, budget, param_mix, dropconnect, p_dropout, iteration_limit=None, device=None, loss_precision=None
):
  """A deep copy of the run config with this arm's overrides applied (nothing is written to disk).

  Everything else -- optimiser, weight decay, `n_models`, the convergence knobs -- is the shipped
  config's, because the handicap is a property of THAT operating point and a run at different
  settings follows a different trajectory and is not comparable.

  `loss_precision` is the ONE exception, and only when it is passed explicitly: it is the BAR the
  two arms are measured against, and the arm ratio is a function of it (a design stops when its
  train/validation gap plus its statistical error fits under the bar, so an arm whose gap is a large
  share of the bar gains far more from relaxing it). Sweeping it is the point of the sweep; a run at
  one value is still not comparable to a run at another EXCEPT through that ratio, and the value is
  recorded in the output either way. Default None keeps the shipped value.
  """
  run = json.loads(json.dumps(config))  # plain JSON-able yaml
  (regressor_name, ), = (run["regressor"].keys(), )
  regressor = run["regressor"][regressor_name]
  if features is not None:
    regressor["features"] = [[int(w) for w in block] for block in features]
  # `None` REMOVES dropout entirely (no layer is built at all), which is not the same code path as a
  # rate of 0; the brief asks for None, so None is what is stored.
  regressor["p_dropout"] = None if p_dropout is None else float(p_dropout)
  regressor["dropconnect"] = None if dropconnect is None else float(dropconnect)
  run["training"]["param_mix"] = float(param_mix)
  run["training"]["budget"] = int(budget)
  if loss_precision is not None:
    run["training"]["loss_precision"] = float(loss_precision)
  if iteration_limit is not None:
    run["training"]["iteration_limit"] = int(iteration_limit)
  if device is not None:
    run["device"] = device
  return run


def power_of_two_budget(train_rows, val_fraction):
  """The smallest POWER OF TWO whose TRAIN share covers ``train_rows``.

  ``Trainer._make_pools`` splits the budget as ``round(budget * val_fraction)`` validation rows and
  the rest train; every train/validation pairing here keeps that same ratio or less, so covering the
  train share covers the validation share too.
  """
  return 1 << max(0, int(math.ceil(math.log2(int(train_rows) / (1.0 - float(val_fraction))))))


def parameter_report(params):
  """``(|params|_1, n_parameters)`` -- the checksum that identifies an initial network, and its size.

  Counted, not derived: the two blocks have different shapes, every linear carries a bias, the
  learnable activation adds two gains per unit, and ``n_models`` multiplies all of it, so a width
  factor does not map to a parameter factor by any rule worth trusting.
  """
  leaves = jax.tree.leaves(params)
  checksum = float(sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves))
  return checksum, int(sum(int(leaf.size) for leaf in leaves))


def pretrain(detector, run, history, *, n_train, n_val, seed, chunk, block, tolerance, min_epochs, max_epochs):
  """ONE network trained JOINTLY on the ten history designs' pooled events.

  No growth loop and no convergence procedure: the whole pooled set is present from the first step,
  and training stops when the pooled VALIDATION loss stops improving -- the mean over the last
  ``block`` epochs failing to beat the ``block`` before it by ``tolerance`` (:func:`settled`).
  Comparing two block MEANS averages the per-epoch evaluation noise down by ``sqrt(block)``, which a
  slope fit over a short horizon does not.

  One EPOCH is ``steps_per_epoch = iteration_limit // batch`` steps, and ``iteration_limit`` is set
  to the pooled train size, so an epoch draws the pooled set once per ensemble member.

  Returns ``(params, state, report, initial_params)``. ``initial_params`` is the fresh draw the
  pretraining STARTED from, and it is what the `from_scratch` arm is given, so the only difference
  between that arm and `warm-start` is the pretraining itself and not the random draw.
  """
  trainer = DesignTrainer.from_config(detector, run, checkpoint_dir=None, seed=int(seed) + HISTORY_SEED_OFFSET)
  pooled_train, pooled_val = n_train * len(history), n_val * len(history)
  # The history stream, and nothing but it: the measurement stream belongs to the evaluation design.
  trainer._train_index = shuffled_event_index(detector.size(), pooled_train, int(seed) + HISTORY_SEED_OFFSET)
  trainer._val_index = shuffled_event_index(detector.size(), pooled_train + pooled_val,
                                            int(seed) + HISTORY_SEED_OFFSET)[pooled_train:]

  started = time.time()
  for entry in history:
    design = detector.to_nominal(entry["x_scaled"])
    fill(detector, trainer.train_pool, design, n_train, trainer._train_index, chunk)
    fill(detector, trainer.val_pool, design, n_val, trainer._val_index, chunk)
  sampled = time.time() - started
  print(
    f"  pooled {pooled_train} train + {pooled_val} validation events over {len(history)} history "
    f"designs ({n_train} + {n_val} each) in {sampled:.0f}s", flush=True
  )

  reg_def = trainer._build_regressor(int(seed))[0]
  eval_pooled_train = trainer._build_eval(reg_def, pooled_train)
  eval_pooled_val = trainer._build_eval(reg_def, pooled_val)
  eval_design_train = trainer._build_eval(reg_def, n_train)
  eval_design_val = trainer._build_eval(reg_def, n_val)

  # The fresh draw BOTH the pretraining and the `from_scratch` arm start from.
  initial_params = trainer._build_regressor(int(seed))[1]
  params, state, opt_state = fresh_design_network(trainer, np.random.SeedSequence(int(seed)), initial_params)
  key = jax.random.PRNGKey(int(seed))
  train_pool, val_pool = trainer.train_pool, trainer.val_pool
  zero = jnp.int32(0)
  val_history, train_history = [], []
  status = "settled"
  for epoch in range(int(max_epochs)):
    key, subkey = jax.random.split(key)
    params, state, opt_state, _ = trainer._train_epoch(
      params, state, opt_state, subkey, zero, jnp.int32(pooled_train), train_pool.buffers()
    )
    val_mean, val_sem = masked_mean_sem(eval_pooled_val(params, state, val_pool.buffers(), zero), pooled_val)
    val_history.append(float(val_mean))
    print(
      f"  [pretrain] epoch {epoch + 1:>3d}  pooled val {float(val_mean):.4f} +- {float(val_sem):.4f}  "
      f"({time.time() - started:.0f}s)", flush=True
    )
    if epoch + 1 < int(min_epochs):
      continue
    if settled(val_history, int(block), float(tolerance)):
      break
  else:
    status = "epoch cap"

  train_mean, train_sem = masked_mean_sem(eval_pooled_train(params, state, train_pool.buffers(), zero), pooled_train)
  val_mean, val_sem = masked_mean_sem(eval_pooled_val(params, state, val_pool.buffers(), zero), pooled_val)
  per_design = []
  for i, entry in enumerate(history):
    d_train, _ = masked_mean_sem(eval_design_train(params, state, train_pool.buffers(), jnp.int32(i * n_train)), n_train)
    d_val, _ = masked_mean_sem(eval_design_val(params, state, val_pool.buffers(), jnp.int32(i * n_val)), n_val)
    per_design.append({
      "design": entry["label"],
      "proxy_loss": entry["proxy_loss"],
      "train": float(d_train),
      "val": float(d_val)
    })
    print(f"  [pretrain] {entry['label']}: train {float(d_train):.4f}  val {float(d_val):.4f}", flush=True)

  drift = float(np.mean(val_history[-2 * block:-block]) -
                np.mean(val_history[-block:])) if len(val_history) >= 2 * block else float("nan")
  report = {
    "status": status,
    "n_history_designs": len(history),
    "n_train_per_design": int(n_train),
    "n_val_per_design": int(n_val),
    "pooled_train": int(pooled_train),
    "pooled_val": int(pooled_val),
    "calls": int(pooled_train + pooled_val),
    "steps_per_epoch": int(trainer.steps_per_epoch),
    "epochs": len(val_history),
    "block": int(block),
    "tolerance": float(tolerance),
    "val_drift": drift,
    "train": float(train_mean),
    "train_sem": float(train_sem),
    "val": float(val_mean),
    "val_sem": float(val_sem),
    "gap_val_minus_train": float(val_mean) - float(train_mean),
    "per_design": per_design,
    "val_per_epoch": [round(v, 6) for v in val_history],
    "sample_s": sampled,
    "wall_s": time.time() - started,
  }
  print(
    f"  [pretrain] {status.upper()} after {len(val_history)} epochs of {trainer.steps_per_epoch} steps "
    f"({report['wall_s']:.0f}s): pooled train {report['train']:.4f} val {report['val']:.4f} "
    f"gap {report['gap_val_minus_train']:+.4f}, last-block val drift {drift:+.5f}", flush=True
  )
  params = jax.tree.map(lambda a: jax.device_put(a, trainer.device), params)
  initial_params = jax.tree.map(lambda a: jax.device_put(a, trainer.device), initial_params)
  del trainer, train_history
  gc.collect()
  return params, report, initial_params


def measure(trainer, entry, seed, *, init_params=None, val_fraction=0.25):
  """Score ONE design through ``trainer.train`` and return what it achieved.

  The per-epoch history is the primary source, because it is available on BOTH exits: the trainer
  submits the epoch's snapshot to ``on_epoch`` before it decides, so the last row is the converged
  epoch on a convergence and the capped epoch on a ``RuntimeError``. A capped design returns no
  ``spent``, so its detector calls are RECONSTRUCTED from the window it reached and marked as such.
  """
  history = {}
  started = time.time()
  status, message = "converged", ""
  objective, objective_std, spent = float("nan"), float("nan"), -1
  try:
    result = trainer.train(entry["x_scaled"], seed, step=0, on_epoch=history.update, init_params=init_params)
    if result is None:
      status, message = "pool exhausted", "the shared budget pool filled -- this arm is NOT a measurement"
    else:
      objective, objective_std, spent = float(result.objective_loss), float(result.objective_std), int(result.spent)
  except RuntimeError as error:
    # ONLY the trainer's own window-cap error. A JAX runtime failure (an OOM, say) also subclasses
    # RuntimeError, and recording one as "the design could not reach the precision" would invent a
    # measurement out of an infrastructure fault -- so anything else is re-raised.
    text = str(error).replace("\n", " ")
    if "did not reach precision within iteration_limit" not in text:
      raise
    status, message = "capped", text
  wall = time.time() - started

  row = {
    "status": status,
    "message": message,
    "wall_s": wall,
    "spent": spent,
    "objective": objective,
    "objective_std": objective_std
  }
  per_window = _per_window(history)
  if len(per_window) > 0:
    last = per_window[-1]
    calls = float(spent) if spent > 0 else last["window"] * (1.0 + val_fraction / (1.0 - val_fraction))
    row.update({
      "train": last["train"],
      "val": last["val"],
      "gap_val_minus_train": last["val"] - last["train"],
      "diff": last["diff"],
      "err": last["err"],
      "slack": last["diff"] + last["err"],
      "window": last["window"],
      "diff_last_round_mean": last["diff_mean"],
      "diff_last_round_max": last["diff_max"],
      "n_epochs": int(np.asarray(history["train_loss_per_epoch"]).size),
      "n_rounds": len(per_window),
      "calls": calls,
      "calls_reconstructed": spent <= 0,
    })
    row["per_window"] = per_window
    row["per_epoch"] = {
      "train": [round(float(v), 6) for v in np.asarray(history["train_loss_per_epoch"])],
      "val": [round(float(v), 6) for v in np.asarray(history["val_loss_per_epoch"])],
      "window": [int(v) for v in np.asarray(history["train_budget_per_epoch"])],
    }
  return row


def announce(tag, row):
  print(
    f"  [{tag}] {row['status'].upper()}: train={row.get('train', float('nan')):.4f} "
    f"val={row.get('val', float('nan')):.4f} gap={row.get('gap_val_minus_train', float('nan')):+.4f} "
    f"slack={row.get('slack', float('nan')):.5f} window={row.get('window', -1)} "
    f"epochs={row.get('n_epochs', -1)} calls={row.get('calls', float('nan')):.0f}"
    f"{'~' if row.get('calls_reconstructed', False) else ''} {row['wall_s']:.0f}s", flush=True
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("config", help="gearup root token of a RUN config, e.g. =enzyme_extremes")
  parser.add_argument(
    "--landscape", default="output/screen/extremes_m4.npz", help="a screen_task.py `_m<m>.npz` (designs + proxy losses), "
    "used ONLY to pick designs -- never as a reported number"
  )
  parser.add_argument("--q-low", type=float, default=0.50, help="low end of the proxy-loss quantile band")
  parser.add_argument("--q-high", type=float, default=0.75, help="high end of the proxy-loss quantile band")
  parser.add_argument("--n-designs", type=int, default=11, help="10 history + 1 evaluation")
  parser.add_argument(
    "--eval-index", type=int, default=0, metavar="I",
    help="WHICH of the band's designs is the EVALUATION design, by ascending-quantile position"
  )
  parser.add_argument(
    "--events-per-design", type=int, default=32768, metavar="N",
    help="history events POOLED per design (train). The pretraining has no growth loop, so this "
    "is a stated sample size rather than a converged one"
  )
  parser.add_argument(
    "--val-per-design", type=int, default=8192, metavar="N",
    help="held-out history events per design, for the pretraining's stopping rule and for its "
    "reported per-design losses"
  )
  parser.add_argument(
    "--features", default=None, metavar="JSON", help="regressor block widths, e.g. '[[32,24],[24,32]]' "
    "(default: the config's own)"
  )
  parser.add_argument("--width", default="config", help="a NAME for this width, recorded in the output")
  parser.add_argument("--p-dropout", type=float, default=None, help="regressor `p_dropout`; unset means None (no layer built)")
  parser.add_argument("--dropconnect", type=float, default=0.1, help="regressor `dropconnect` (weight dropping)")
  parser.add_argument("--param-mix", type=float, default=0.25, help="`training.param_mix` at every data addition")
  parser.add_argument(
    "--loss-precision", type=float, default=None, metavar="P",
    help="the convergence BAR both arms are measured against; unset means the config's own. Sweeping "
    "it measures how the arm ratio depends on the bar, which is the one thing the arms do NOT share"
  )
  parser.add_argument("--pretrain-block", type=int, default=6, help="epochs per block in the pretraining stopping rule")
  parser.add_argument(
    "--pretrain-tolerance", type=float, default=5.0e-4,
    help="the pretraining stopping rule's improvement threshold; it must sit WELL BELOW the "
    "arm-to-arm difference the study has to resolve"
  )
  parser.add_argument("--pretrain-min-epochs", type=int, default=18)
  parser.add_argument("--pretrain-max-epochs", type=int, default=80)
  parser.add_argument("--chunk", type=int, default=8192, help="detector call chunk (latency-bound: bigger is faster)")
  parser.add_argument("--arms", nargs="+", default=["from_scratch", "meta"], choices=["from_scratch", "meta"])
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--device", default=None)
  parser.add_argument(
    "--dry-run", action="store_true", help="report the parameter count, the initial-parameter checksum and the pool "
    "sizes each arm would use, then exit without sampling or training"
  )
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  config, detector_config = load_config(arguments.config)
  detector = detopt.detector.from_config(detector_config)
  band = design_band(arguments.landscape, detector, arguments.q_low, arguments.q_high, arguments.n_designs)

  eval_index = int(arguments.eval_index)
  if not 0 <= eval_index < len(band):
    raise SystemExit(f"probe_meta_capacity: --eval-index {eval_index} outside [0, {len(band)})")
  evaluation = band[eval_index]
  history = [c for i, c in enumerate(band) if i != eval_index]

  features = json.loads(arguments.features) if arguments.features is not None else None
  val_fraction = float(config["training"]["val_fraction"])
  iteration_limit = int(config["training"]["iteration_limit"])
  n_train, n_val = int(arguments.events_per_design), int(arguments.val_per_design)
  pooled_train, pooled_val = n_train * len(history), n_val * len(history)

  # THREE pools, each a power of two and each sized to what its own arm actually holds.
  pretrain_budget = power_of_two_budget(pooled_train, val_fraction)
  measure_budget = power_of_two_budget(iteration_limit, val_fraction)
  meta_budget = power_of_two_budget(pooled_train + iteration_limit, val_fraction)

  pretrain_run = run_config_for(
    config, features=features, budget=pretrain_budget, param_mix=arguments.param_mix, dropconnect=arguments.dropconnect,
    p_dropout=arguments.p_dropout, loss_precision=arguments.loss_precision, iteration_limit=pooled_train,
    device=arguments.device
  )
  measure_run = run_config_for(
    config, features=features, budget=measure_budget, param_mix=arguments.param_mix, dropconnect=arguments.dropconnect,
    p_dropout=arguments.p_dropout, loss_precision=arguments.loss_precision, device=arguments.device
  )
  meta_run = run_config_for(
    config, features=features, budget=meta_budget, param_mix=arguments.param_mix, dropconnect=arguments.dropconnect,
    p_dropout=arguments.p_dropout, loss_precision=arguments.loss_precision, device=arguments.device
  )
  (regressor_name, ), = (measure_run["regressor"].keys(), )

  print(f"config {arguments.config} | width `{arguments.width}` | seed {arguments.seed}")
  print(f"regressor: {json.dumps(measure_run['regressor'][regressor_name])}")
  print(
    f"training: param_mix={measure_run['training']['param_mix']} weight_decay="
    f"{list(measure_run['training']['optimizer'].values())[0]['weight_decay']} "
    f"iteration_limit={iteration_limit} loss_precision={measure_run['training']['loss_precision']}"
  )
  print(
    f"budgets (powers of two): pretrain 2^{int(math.log2(pretrain_budget))}={pretrain_budget} | "
    f"per-design 2^{int(math.log2(measure_budget))}={measure_budget} | "
    f"meta 2^{int(math.log2(meta_budget))}={meta_budget}"
  )
  print(f"evaluation design: {evaluation['label']} rank {evaluation['rank']} proxy {evaluation['proxy_loss']:.4f}")
  print("history (POOLED, not sequential): " + " ".join(f"{c['label']}({c['proxy_loss']:.3f})" for c in history), flush=True)

  # ONE child sequence PER BAND POSITION, so every arm scores the evaluation design with the SAME
  # training rng.
  eval_seq = int(np.random.SeedSequence(int(arguments.seed)).spawn(len(band))[eval_index].generate_state(1)[0])

  payload = {
    "config": arguments.config.lstrip("="),
    "landscape": arguments.landscape,
    "width": arguments.width,
    "features": measure_run["regressor"][regressor_name].get("features"),
    "n_models": measure_run["regressor"][regressor_name].get("n_models"),
    "p_dropout": measure_run["regressor"][regressor_name].get("p_dropout"),
    "dropconnect": measure_run["regressor"][regressor_name].get("dropconnect"),
    "param_mix": measure_run["training"]["param_mix"],
    "weight_decay": list(measure_run["training"]["optimizer"].values())[0]["weight_decay"],
    "loss_precision": measure_run["training"]["loss_precision"],
    "iteration_limit": iteration_limit,
    "budgets": {
      "pretrain": pretrain_budget,
      "per_design": measure_budget,
      "meta": meta_budget
    },
    "seed": int(arguments.seed),
    "eval_index": eval_index,
    "band": [{
      k: v
      for k, v in c.items() if k != "x_scaled"
    } for c in band],
    "evaluation_design": evaluation["label"],
    "history_designs": [c["label"] for c in history],
    "arms": {},
  }

  def write():
    os.makedirs(os.path.dirname(os.path.abspath(arguments.output)) or ".", exist_ok=True)
    with open(arguments.output, "w") as f:
      json.dump(payload, f, indent=2, default=float)

  if arguments.dry_run:
    probe = DesignTrainer.from_config(detector, measure_run, checkpoint_dir=None, seed=int(arguments.seed))
    checksum, n_parameters = parameter_report(probe._build_regressor(int(arguments.seed))[1])
    print(
      f"\nDRY RUN, width `{arguments.width}`: {n_parameters} parameters over {probe.n_ensemble} ensemble "
      f"members | initial |params|_1 = {checksum:.6f}\n"
      f"  pretrain pool {pooled_train} train + {pooled_val} validation; meta pool holds those plus the "
      f"evaluation window (<= {iteration_limit} train)"
    )
    return

  # ------------------------------------------------------------------ #
  # THE PRETRAINED NETWORK: one network, the ten history designs pooled, trained jointly.
  # ------------------------------------------------------------------ #
  print(f"\n=== PRETRAIN: one network on {pooled_train} pooled events from {len(history)} designs", flush=True)
  pretrained, pretrain_report, initial_params = pretrain(
    detector, pretrain_run, history, n_train=n_train, n_val=n_val, seed=int(arguments.seed), chunk=arguments.chunk,
    block=arguments.pretrain_block, tolerance=arguments.pretrain_tolerance, min_epochs=arguments.pretrain_min_epochs,
    max_epochs=arguments.pretrain_max_epochs
  )
  initial_checksum, n_parameters = parameter_report(initial_params)
  pretrained_checksum, _ = parameter_report(pretrained)
  pretrain_report.update({
    "initial_checksum": initial_checksum,
    "pretrained_checksum": pretrained_checksum,
    "n_parameters": n_parameters,
  })
  payload["pretrain"] = pretrain_report
  print(
    f"  initial |params|_1 = {initial_checksum:.6f} -> pretrained {pretrained_checksum:.6f} "
    f"over {n_parameters} parameters", flush=True
  )
  write()

  # The MEASUREMENT event stream, identical for all three arms. A per-design trainer at
  # `measure_budget` builds exactly this and splits it here, so re-pointing the meta arm's arrays at
  # `[history stream, measurement stream]` makes its evaluation window open on the same events.
  measure_val_budget = round(measure_budget * val_fraction)
  measure_index = shuffled_event_index(detector.size(), measure_budget, int(arguments.seed))
  measure_train_index = measure_index[:measure_budget - measure_val_budget]
  measure_val_index = measure_index[measure_budget - measure_val_budget:]

  def record(name, row):
    payload["arms"][name] = row
    announce(name, row)
    write()

  if "from_scratch" in arguments.arms:
    print(f"\n=== from_scratch: DesignTrainer, FRESH init (|params|_1 {initial_checksum:.6f})", flush=True)
    trainer = DesignTrainer.from_config(detector, measure_run, checkpoint_dir=None, seed=int(arguments.seed))
    row = measure(trainer, evaluation, eval_seq, init_params=initial_params, val_fraction=val_fraction)
    row.update({"kind": "from_scratch", "init_checksum": initial_checksum, "budget": measure_budget})
    record("from_scratch", row)
    del trainer
    gc.collect()

  if "meta" in arguments.arms:
    print("\n=== meta: ContinualTrainer, PRETRAINED persistent network + history in the replay pool", flush=True)
    trainer = ContinualTrainer.from_config(detector, meta_run, checkpoint_dir=None, seed=int(arguments.seed))
    # The persistent network's PARAMS become the pretrained ones; its freshly built buffer state is
    # kept, and carries no information (rng counters, never read during training -- the dropout key is
    # threaded fresh -- nor during evaluation, which is deterministic). `_running` is a 2-tuple:
    # the optimiser is no longer carried on it at all, because `ContinualTrainer` rebuilds it at every
    # design boundary.
    _, state = trainer._running
    trainer._running = (pretrained, state)
    # THE HISTORY FIRST, then the measurement stream: the replay sampler draws from `[0, w0)`, so the
    # ten designs must be in the pool BEFORE the evaluation design's window opens, and the window
    # must then open on the same events the other two arms see.
    trainer._train_index = np.concatenate([
      shuffled_event_index(detector.size(), pooled_train,
                           int(arguments.seed) + HISTORY_SEED_OFFSET), measure_train_index
    ])
    trainer._val_index = np.concatenate([
      shuffled_event_index(detector.size(), pooled_train + pooled_val,
                           int(arguments.seed) + HISTORY_SEED_OFFSET)[pooled_train:], measure_val_index
    ])
    started = time.time()
    for entry in history:
      design = detector.to_nominal(entry["x_scaled"])
      fill(detector, trainer.train_pool, design, n_train, trainer._train_index, arguments.chunk)
      fill(detector, trainer.val_pool, design, n_val, trainer._val_index, arguments.chunk)
    pool_sample_s = time.time() - started
    print(
      f"  replay pool pre-filled: {trainer.train_pool.current} train + {trainer.val_pool.current} validation "
      f"events over {len(history)} designs ({pool_sample_s:.0f}s); replay_weight={trainer.replay_weight:g}", flush=True
    )
    row = measure(trainer, evaluation, eval_seq, val_fraction=val_fraction)
    row.update({
      "kind": "meta",
      "init_checksum": pretrained_checksum,
      "budget": meta_budget,
      "replay_weight": float(trainer.replay_weight),
      "w0_train": int(pooled_train),
      "w0_val": int(pooled_val),
      "pool_sample_s": pool_sample_s,
    })
    record("meta", row)
    del trainer
    gc.collect()

  arms = payload["arms"]
  contrasts = {}
  for later, earlier in (("meta", "from_scratch"), ):
    if later in arms and earlier in arms and "val" in arms[later] and "val" in arms[earlier]:
      contrasts[f"{later} - {earlier}"] = {
        "val":
        float(arms[later]["val"] - arms[earlier]["val"]),
        "train":
        float(arms[later]["train"] - arms[earlier]["train"]),
        "objective":
        float(0.5 * (arms[later]["train"] + arms[later]["val"]) - 0.5 * (arms[earlier]["train"] + arms[earlier]["val"])),
      }
  payload["contrasts"] = contrasts
  write()
  if len(contrasts) > 0:
    print(f"\nCONTRASTS at the evaluation design ({evaluation['label']}), width `{arguments.width}`:")
    for name, value in contrasts.items():
      print(
        f"  {name:<28s} VALIDATION {value['val']:+.4f}   train {value['train']:+.4f}   "
        f"(train+val)/2 {value['objective']:+.4f}"
      )
    print(
      "READ IT AS: replication noise on this system is 0.006 with a worst case of 0.011, so a smaller "
      "difference is not a difference."
    )
  print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
  main()
