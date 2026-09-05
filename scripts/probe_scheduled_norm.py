#!/usr/bin/env python3
"""Does an ANNEALED soft normalisation beat a fixed one, or none, when verification trains from scratch?

    python scripts/probe_scheduled_norm.py \
        --run output/archive-final-cern-20260903T133704/angle/test/1244111331/from_scratch \
        --point 3 --seed 1244111331 --budget 524288 --epochs 64 --arms baseline \
        --output output/probe-scheduled-norm/angle-1244111331-p3

FOUR ARMS on ONE mid-BO design, sharing ONE train/validation/test split and ONE initialisation:

    baseline    no normalisation -- the network as the campaign builds it
    layernorm   (x - mean) / sqrt(var + eps) over the trailing feature axis
    c_const     x / (1 + C * ||x|| / sqrt(dim)) at C = 1 throughout
    c_decay     the same, with C cosine-annealed 1 -> 0, reaching 0 at 0.75 of the training window

``||x|| / sqrt(dim)`` is the root-mean-square of the row, so the map is ``x / (1 + C * rms(x))``: at
C = 0 it is EXACTLY the identity, which is why the annealed arm ends as the baseline architecture and
the schedule is a training device rather than a different network at read-out.

EVERY NORM HERE IS PARAMETER-FREE. None of them contributes a leaf to the parameter pytree, so one
seed gives all four arms a BIT-IDENTICAL initial network -- asserted at run time, not assumed. The
only difference between arms is the forward pass.

WHAT IT DEVIATES FROM `verify_trajectory.py`, deliberately, and nothing else:

* THE NETWORK IS FRESH, not the design's checkpoint. The campaign's verification continues the network
  the run reported the design with; here every arm starts from the same fresh draw, because a
  checkpoint was trained without any of these norms and restoring it into three of the four arms would
  compare a trained network against a partly mismatched one.
* THE WINDOW IS FIXED at ``--epochs`` with no early exit, and both cosines -- the learning rate and C
  -- span that same window. The stopping rule is therefore NOT part of what is measured.

WHAT IT DOES NOT ESTABLISH. One design, one seed, one task. Four arms on one point separate an
architecture effect from nothing at all; the paired split and the shared initialisation are what make
the four numbers comparable to EACH OTHER, not what make one point representative.

THE DETECTOR COMES FROM THE TRAJECTORY'S OWN BANKED CONFIG, never from a file in `config/`. The
archived CERN runs banked ``fixed_stations [8407, 8607, 9307, 9507]`` and every `ship_angle_*` config
on disk now carries ``[8325, 8767, 9600, 9700]``; verifying against the file would silently simulate a
different detector from the one the design was optimised on.
"""

import argparse
import json
import os
import sys
import threading
import time
from functools import partial

import matplotlib

matplotlib.use("AGG")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detopt  # noqa: E402
from detopt.nn.trainer.common import regressor_rngs  # noqa: E402
from detopt.utils.config import optimizer as make_optimizer, resolve_device, split  # noqa: E402
from detopt.utils.events import split_disjoint  # noqa: E402
from detopt.utils.pools import RingBuffer  # noqa: E402

ARMS = ("baseline", "layernorm", "c_const", "c_decay")


class Coefficient:
  """Mutable carrier for C, read at CALL time by every spliced norm in one arm.

    The jitted kernels set ``value`` from a TRACED argument on entry, so C varies per step without
    the norm modules holding any array state: a coefficient stored on the module would become either
    a parameter leaf (changing the initialisation) or static metadata (forcing a recompile per epoch).
    One carrier per arm; the modules hold it as static graph metadata, hashed by identity.
    """

  __slots__ = ("value", )

  def __init__(self, value):
    self.value = value


class ParameterFreeLayerNorm(nnx.Module):
  """``(x - mean) / sqrt(var + eps)`` over the trailing feature axis, no affine.

    Statistics are per row over the last axis, so the hit axis never enters and a masked slot cannot
    reach a live one. Verbatim from ``scripts/probe_layernorm.py`` so the two probes' `layernorm` arms
    are the same architecture.
    """

  def __init__(self, epsilon: float = 1e-6):
    self.epsilon = float(epsilon)

  def __call__(self, x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + self.epsilon)


class ScheduledSoftNorm(nnx.Module):
  """``x / (1 + C * ||x|| / sqrt(dim))`` over the trailing feature axis, C read from the carrier.

    ``||x|| / sqrt(dim)`` is the row's root-mean-square, so C = 0 is exactly the identity and large C
    approaches an RMS normalisation. Parameter-free and state-free.
    """

  def __init__(self, carrier: Coefficient):
    self.carrier = carrier

  def __call__(self, x):
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True))
    return x / (1.0 + self.carrier.value * rms)


def splice_norm(model, factory):
  """Insert ``factory()`` after every hidden ``EnsembleLinear`` in every block's shared MLP.

    The insertion point is the one ``scripts/probe_layernorm.py`` uses -- linear -> norm -> activation,
    hidden layers only, never a block's output map (which emits the aggregation gate). Returns the
    number inserted; mutates ``model`` in place.
    """
  from detopt.nn.set_regressor import EnsembleLinear

  inserted = 0
  for block in model.blocks:
    rebuilt = []
    for layer in block.shared:
      rebuilt.append(layer)
      if isinstance(layer, EnsembleLinear):
        rebuilt.append(factory())
        inserted += 1
    block.shared = nnx.List(rebuilt)
  return inserted


def coefficient_schedule(arm, decay_steps):
  """C as a function of the global step, for one arm.

    ``c_decay`` is a single cosine from 1 to 0 over ``decay_steps`` and stays at 0 after; ``c_const``
    holds 1; the arms with no soft norm return 0, which their (absent) carrier never reads.
    """
  if arm == "c_const":
    return lambda step: jnp.float32(1.0)
  if arm == "c_decay":
    return lambda step: 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.minimum(step / jnp.float32(decay_steps), 1.0)))
  return lambda step: jnp.float32(0.0)


def build_arm(arm, detector, config, seed, reveals, carrier):
  """A fresh regressor for ``arm``, built from ONE seed and then spliced.

    The splice happens after construction, so no arm perturbs the rng streams the parameters are drawn
    from and all four draw the identical network.
    """
  model = detopt.nn.from_config(detector, config=config["regressor"], rngs=regressor_rngs(int(seed)), design=reveals)
  if arm == "layernorm":
    inserted = splice_norm(model, ParameterFreeLayerNorm)
  elif arm in ("c_const", "c_decay"):
    inserted = splice_norm(model, lambda: ScheduledSoftNorm(carrier))
  else:
    inserted = 0
  return model, inserted


def describe_architecture(model):
  """The spliced forward, block by block: every layer of each block's shared MLP in call order, then
    the block's output map, then the read-out. What the norms actually sit between."""
  from detopt.nn.set_regressor import EnsembleLinear

  def name(layer):
    if isinstance(layer, EnsembleLinear):
      return f"EnsembleLinear({layer.kernel.shape[-2]}->{layer.kernel.shape[-1]})"
    if isinstance(layer, ParameterFreeLayerNorm):
      return "ParameterFreeLayerNorm"
    if isinstance(layer, ScheduledSoftNorm):
      return "ScheduledSoftNorm[x/(1+C*rms(x))]"
    return type(layer).__name__

  lines = []
  for i, block in enumerate(model.blocks):
    chain = " -> ".join(name(layer) for layer in block.shared)
    lines.append(f"  block {i}: {chain} -> {name(block.output)} -> masked_weighted_aggregate")
  lines.append(f"  readout: {name(model.output)}")
  return "\n".join(lines)


def parameter_checksum(params):
  """A value that changes if any initial weight does -- the identical-initialisation assertion."""
  leaves = jax.tree_util.tree_leaves(params)
  return float(sum(float(np.sum(np.asarray(leaf, np.float64) * (i + 1))) for i, leaf in enumerate(leaves)))


def _forward_loss(reg, loss_fn, feats, mask, target, members, batch, *, deterministic, rngs=None):
  """TRAIN loss path (mirrors ``scripts/verify_trajectory.py``): a ``(members*batch, ...)`` minibatch
    -> per-sample loss; each member gets its OWN slice."""
  if members is None:
    return reg.loss(loss_fn, feats, mask, target, deterministic=deterministic, rngs=rngs)
  feats_e = feats.reshape((members, batch) + feats.shape[1:])
  mask_e = mask.reshape((members, batch) + mask.shape[1:])
  target_e = target.reshape((members, batch) + target.shape[1:])
  loss = reg.loss(loss_fn, feats_e, mask_e, target_e, deterministic=deterministic, rngs=rngs)
  return loss.reshape((members * batch, ) + loss.shape[2:])


def _predict_shared(reg, feats, mask, members):
  """EVAL path: ONE batch fed to every member (broadcast), member predictions AVERAGED."""
  if members is None:
    return reg(feats, mask, deterministic=True)
  fe = jnp.broadcast_to(feats[None], (members, ) + feats.shape)
  me = jnp.broadcast_to(mask[None], (members, ) + mask.shape)
  return reg(fe, me, deterministic=True).mean(axis=0)


def _split_indices(size, budget, seed):
  """Disjoint 6:2:2 train/validation/test event indices, the split every arm shares (paired scores).
    Mirrors ``verify_trajectory._split_indices``: for a finite detector the split is applied to the
    UNIQUE event universe first, since a repeated index replays the identical row."""
  rng = np.random.default_rng(seed)
  n_train, n_validation = round(0.6 * budget), round(0.2 * budget)
  n_test = budget - n_train - n_validation
  if size is None:
    index = rng.integers(0, 2**31 - 1, size=budget, dtype=np.int64)
    return index[:n_train], index[n_train:n_train + n_validation], index[n_train + n_validation:]
  universes = split_disjoint(rng.permutation(int(size)), 0.6, 0.2)
  return tuple(u[:n] for u, n in zip(universes, (n_train, n_validation, n_test)))


_PLOT_LOCK = threading.Lock()


def _plot_arm(history, arm, path):
  """One arm's learning curves: deterministic train and validation per epoch, with C on a twin axis."""
  from matplotlib.figure import Figure

  h = np.asarray(history, np.float64)
  with _PLOT_LOCK:
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots(1, 1)
    ax.plot(h[:, 0], h[:, 2], "-", color="tab:blue", label="train (deterministic)")
    ax.plot(h[:, 0], h[:, 4], "-", color="tab:orange", label="validation")
    ax.set(title=f"arm: {arm}", xlabel="epoch", ylabel="loss (normalized)", yscale="log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    twin = ax.twinx()
    twin.plot(h[:, 0], h[:, 6], ":", color="0.45", lw=1.4)
    twin.set_ylabel("C (dotted)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def _plot_comparison(record, path):
  """All four arms on one pair of axes: validation curves, and the test scores as a bar with SEM."""
  from matplotlib.figure import Figure

  colors = {"baseline": "0.35", "layernorm": "tab:green", "c_const": "tab:red", "c_decay": "tab:blue"}
  fig = Figure(figsize=(13, 5.5))
  axes = fig.subplots(1, 2)
  names, tests, sems = [], [], []
  for arm, entry in record["arms"].items():
    h = np.asarray(entry["history"], np.float64)
    axes[0].plot(h[:, 0], h[:, 4], "-", lw=1.6, color=colors.get(arm), label=f"{arm} (test {entry['test_loss']:.4f})")
    names.append(arm)
    tests.append(entry["test_loss"])
    sems.append(entry["test_sem"])
  axes[0].set(title="validation loss per epoch", xlabel="epoch", ylabel="loss (normalized)", yscale="log")
  axes[0].legend(fontsize=8)
  axes[1].bar(names, tests, yerr=sems, capsize=4, color=[colors.get(n) for n in names])
  axes[1].set(title="held-out test loss at best validation", ylabel="loss (normalized)")
  axes[1].set_ylim(min(tests) - 3 * max(sems), max(tests) + 3 * max(sems))
  for ax in axes:
    ax.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(path, dpi=140)


SHARED_SETTINGS = (
  "run", "point", "seed", "budget", "split", "epochs", "steps_per_epoch", "total_steps", "batch", "init_seed", "design_scaled",
  "decay_fraction", "c_zero_step", "peak_learning_rate", "reveal"
)


def report(directory):
  """Merge the per-arm runs under ``directory`` into one comparison.

    With one arm per job, "the same split and the same initialisation" is a claim ACROSS PROCESSES.
    It is CHECKED here, not assumed: every shared setting must agree between the parts, and every
    arm's parameter checksum must be the identical value -- the same assertion the all-arms-in-one
    path makes in memory. A disagreement means the arms are not paired and the comparison is void.
    """
  import glob

  paths = sorted(glob.glob(os.path.join(directory, "*", "scheduled_norm.json")))
  if len(paths) == 0:
    sys.exit(f"{directory}: no <arm>/scheduled_norm.json to merge")
  merged, checksums, sources = None, {}, {}
  for path in paths:
    with open(path) as handle:
      part = json.load(handle)
    if merged is None:
      merged = {k: v for k, v in part.items() if k != "arms"}
      merged["arms"] = {}
    else:
      disagree = [k for k in SHARED_SETTINGS if part.get(k) != merged.get(k)]
      if len(disagree) > 0:
        raise RuntimeError(
          f"{path} disagrees with the other arms on {', '.join(disagree)} -- the arms are NOT paired and "
          f"the comparison between them means nothing"
        )
    for arm, entry in part["arms"].items():
      merged["arms"][arm] = entry
      checksums[arm] = entry["parameter_checksum"]
      sources[arm] = path

  distinct = {round(v, 6) for v in checksums.values()}
  if len(distinct) > 1:
    raise RuntimeError(
      f"the arms did NOT start from the same network: parameter checksums {checksums}. Every norm here is "
      f"parameter-free, so one seed must give a bit-identical initialisation across jobs"
    )
  merged["arms"] = {arm: merged["arms"][arm] for arm in ARMS if arm in merged["arms"]}
  print(f"[report] merged {len(merged['arms'])} arms from {directory}")
  print(f"[report] identical initialisation confirmed across jobs (checksum {distinct.pop():.6f})")

  json_path = os.path.join(directory, "scheduled_norm.json")
  with open(json_path, "w") as handle:
    json.dump(merged, handle, indent=2)
  if len(merged["arms"]) > 1:
    _plot_comparison(merged, os.path.join(directory, "comparison.png"))
  baseline = merged["arms"].get("baseline")
  print(f"\n{'arm':<12} {'best_ep':>8} {'validation':>11} {'test':>18} {'C@best':>8} {'vs baseline':>22}")
  for arm, entry in merged["arms"].items():
    if baseline is None or arm == "baseline":
      against = ""
    else:
      delta = entry["test_loss"] - baseline["test_loss"]
      pooled = float(np.hypot(entry["test_sem"], baseline["test_sem"]))
      against = f"{delta:+.4f}+/-{pooled:.4f} ({abs(delta) / pooled:.1f}s)"
    print(
      f"{arm:<12} {entry['best_epoch']:>8} {entry['validation_loss']:>11.4f} "
      f"{entry['test_loss']:>11.4f}+/-{entry['test_sem']:.4f} {entry['best_c']:>8.4f} {against:>22}"
    )
  print(f"saved -> {json_path}")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--run", help="a finished cell; supplies the BANKED config and the design")
  parser.add_argument("--point", type=int, help="trajectory index of the mid-BO design to verify")
  parser.add_argument("--seed", type=int, help="the split, the initialisation and the data order")
  parser.add_argument("--output", help="where this arm's scheduled_norm.json and curves go")
  parser.add_argument(
    "--report", default=None,
    help="merge mode: read <DIR>/<arm>/scheduled_norm.json for every arm, check they are paired, and write the "
    "comparison. Takes no other argument"
  )
  parser.add_argument("--epochs", type=int, default=128, help="the FIXED verification window; no early exit")
  parser.add_argument("--decay-fraction", type=float, default=0.75, help="fraction of the window at which C reaches 0")
  parser.add_argument(
    "--budget", type=int, default=None, help="verification detector calls; default = the run's training.budget"
  )
  parser.add_argument("--batch", type=int, default=None, help="default = the run's training.batch")
  parser.add_argument("--eval-batch", type=int, default=None, help="default = the run's verify.eval_batch")
  parser.add_argument("--sample-batch", type=int, default=None, help="events simulated per detector call while filling")
  parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of " + ",".join(ARMS))
  parser.add_argument("--device", default=None)
  parser.add_argument("--progress", default="plain", choices=("bar", "plain"))
  arguments = parser.parse_args()

  if arguments.report is not None:
    report(arguments.report)
    return
  missing = [n for n in ("run", "point", "seed", "output") if getattr(arguments, n) is None]
  if len(missing) > 0:
    sys.exit(f"missing required argument(s): {', '.join('--' + n for n in missing)} (or use --report)")

  arms = tuple(a.strip() for a in arguments.arms.split(",") if len(a.strip()) > 0)
  unknown = [a for a in arms if a not in ARMS]
  if len(unknown) > 0:
    sys.exit(f"unknown arm(s) {unknown}; choose from {ARMS}")

  run_path = arguments.run if arguments.run.endswith(".json") else os.path.join(arguments.run, "results.json")
  with open(run_path) as handle:
    payload = json.load(handle)
  config = payload["config"]
  if isinstance(config, str):
    config = json.loads(config)
  results = payload["results"]
  if arguments.point >= len(results):
    sys.exit(f"{run_path}: point {arguments.point} needs {arguments.point + 1} rows, run has {len(results)}")
  row = results[arguments.point]
  if row.get("loss") is None:
    sys.exit(f"{run_path}: point {arguments.point} is an incomplete row -- it has no loss and was never scored")

  training = config["training"]
  budget = int(arguments.budget if arguments.budget is not None else training["budget"])
  batch = int(arguments.batch if arguments.batch is not None else training["batch"])
  verify_block = config.get("verify") or {}
  eval_batch = int(arguments.eval_batch if arguments.eval_batch is not None else verify_block.get("eval_batch", 1024))
  sample_batch = int(arguments.sample_batch if arguments.sample_batch is not None else verify_block.get("sample_batch", 1024))
  epochs = int(arguments.epochs)
  device = resolve_device(arguments.device if arguments.device is not None else config.get("device"))

  detector = detopt.detector.from_config(config["detector"])
  design_dim = int(detector.design_dim())
  labels = tuple(detector.metric_labels())
  reveal = training.get("reveal")
  if reveal is None:
    strategy = payload.get("nn_init_strategy") or config.get("nn_init_strategy") or ""
    reveal = "design" if strategy.startswith("meta") else "none"
  reveals = reveal != "none"

  theta = jnp.asarray(row["x_scaled"], jnp.float32)
  physical = np.asarray(detector.flatten_design(detector.to_nominal(theta)), np.float32)
  reported = float(row["loss"])

  master = np.random.SeedSequence(int(arguments.seed))
  index_seq, init_seq, order_seq = master.spawn(3)
  init_seed = int(init_seq.generate_state(1)[0])
  order_seed = int(order_seq.generate_state(1)[0])
  train_index, validation_index, test_index = _split_indices(detector.size(), budget, index_seq)

  M = int(jax.tree.leaves(detector.event_spec())[0].shape[0])
  specs = (detector.event_spec(), jax.ShapeDtypeStruct((M, ), jnp.int32), detector.target_spec())
  train_buffer = RingBuffer(len(train_index), specs, device=device)
  validation_buffer = RingBuffer(len(validation_index), specs, device=device)
  test_buffer = RingBuffer(len(test_index), specs, device=device)
  steps_per_epoch = max(1, len(train_index) // batch)
  total_steps = epochs * steps_per_epoch
  decay_steps = arguments.decay_fraction * total_steps

  os.makedirs(arguments.output, exist_ok=True)
  print(
    f"[probe] run {run_path}\n"
    f"[probe] point {arguments.point}/{len(results)} reported={reported:.4f} design={np.round(physical, 4).tolist()} "
    f"x_scaled={np.round(np.asarray(theta), 6).tolist()}\n"
    f"[probe] reveal={reveal!r} features={detector.combined_event_shape(reveals)} seed={arguments.seed}\n"
    f"[probe] budget={budget} -> train/validation/test = {len(train_index)}/{len(validation_index)}/{len(test_index)}\n"
    f"[probe] epochs={epochs} x {steps_per_epoch} steps = {total_steps} steps | batch={batch} | "
    f"C reaches 0 at step {decay_steps:.0f} (epoch {decay_steps / steps_per_epoch:.1f})\n"
    f"[probe] arms: {', '.join(arms)}", flush=True
  )

  template, _ = build_arm("baseline", detector, config, init_seed, reveals, None)
  members = template.ensemble()
  optimizer_name, optimizer_arguments = split(training["optimizer"])
  peak_learning_rate = float(optimizer_arguments["learning_rate"])
  draw = (members or 1) * batch

  def fill(buffer, event_index, description):
    """(Re)fill ``buffer`` with the events at ``event_index`` simulated at the FIXED design. STRICTLY
        SERIAL: the propagation engine fills detector-owned buffers in place, so concurrent calls on one
        detector instance corrupt each other."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    physical_full = detector.to_nominal(jnp.broadcast_to(theta[None, :], (sample_batch, design_dim)))
    bar = tqdm(total=n, desc=description, disable=arguments.progress != "bar")
    for offset in range(0, n, sample_batch):
      idx = event_index[offset:offset + sample_batch]
      k = idx.shape[0]
      phys = physical_full if k == sample_batch else detector.to_nominal(jnp.broadcast_to(theta[None, :], (k, design_dim)))
      _ground_truth, event, mask, target = detector(phys, idx)
      buffer.push(event, mask, target)
      bar.update(k)
    bar.close()

  began = time.time()
  fill(train_buffer, train_index, "sample train")
  fill(validation_buffer, validation_index, "sample validation")
  fill(test_buffer, test_index, "sample test")
  print(f"[probe] buffers filled in {(time.time() - began) / 60:.1f} min", flush=True)

  record = {
    "run": os.path.abspath(run_path),
    "point": int(arguments.point),
    "seed": int(arguments.seed),
    "reported_loss": reported,
    "design_physical": physical.tolist(),
    "design_scaled": np.asarray(theta).tolist(),
    "budget": budget,
    "split": [len(train_index), len(validation_index), len(test_index)],
    "epochs": epochs,
    "steps_per_epoch": steps_per_epoch,
    "total_steps": total_steps,
    "batch": batch,
    "members": members,
    "reveal": reveal,
    "init": "fresh",
    "init_seed": init_seed,
    "lr_schedule": "cosine",
    "peak_learning_rate": peak_learning_rate,
    "decay_fraction": float(arguments.decay_fraction),
    "c_zero_step": float(decay_steps),
    "history_columns": ["epoch", "train_running", "train", "train_sem", "validation", "validation_sem", "c", "learning_rate"],
    "arms": {},
  }
  json_path = os.path.join(arguments.output, "scheduled_norm.json")
  checksums = {}

  for arm in arms:
    carrier = Coefficient(jnp.float32(0.0))
    model, inserted = build_arm(arm, detector, config, init_seed, reveals, carrier)
    regressor_definition = nnx.split(model, nnx.Param, nnx.Variable)[0]
    _, params, state = nnx.split(model, nnx.Param, nnx.Variable)
    params, state = jax.device_put(params, device), jax.device_put(state, device)
    checksums[arm] = parameter_checksum(params)

    schedule = coefficient_schedule(arm, decay_steps)
    learning_rate = optax.cosine_decay_schedule(init_value=peak_learning_rate, decay_steps=total_steps)
    opt = make_optimizer({optimizer_name: {**optimizer_arguments, "learning_rate": learning_rate}})
    opt_state = opt.init(params)

    def _net_loss(params, state, drop_key, event_b, mask_b, target_b):
      reg = nnx.merge(regressor_definition, params, state)
      feats = detector.combine_scaled(event_b, theta, mask=mask_b, reveal_design=reveals)
      element_mask = detector.element_mask(event_b, mask_b)
      loss = jnp.mean(
        _forward_loss(
          reg, detector.loss, feats, element_mask, detector.normalize_target(target_b), members, batch, deterministic=False,
          rngs=nnx.Rngs(dropout=jax.random.fold_in(drop_key, 0), dropconnect=jax.random.fold_in(drop_key, 1))
        )
      )
      _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
      return loss, new_state

    @jax.jit
    def train_epoch(params, state, opt_state, key, step0, event_buffer, mask_buffer, target_buffer, n):
      """One epoch of scan-folded SGD. C is set per step from the traced global step, so the carrier
            never holds a concrete value and the kernel compiles once for the whole run."""

      def step(carry, scanned):
        offset, k = scanned
        params, state, opt_state = carry
        carrier.value = schedule(step0 + offset)
        k_index, k_drop = jax.random.split(k)
        idx = jax.random.randint(k_index, (draw, ), 0, n)
        event_b = jax.tree.map(lambda a: a[idx], event_buffer)
        target_b = jax.tree.map(lambda a: a[idx], target_buffer)
        (loss, state), grads = jax.value_and_grad(_net_loss,
                                                  has_aux=True)(params, state, k_drop, event_b, mask_buffer[idx], target_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, state, opt_state), loss

      scanned = (jnp.arange(steps_per_epoch, dtype=jnp.float32), jax.random.split(key, steps_per_epoch))
      (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), scanned)
      return params, state, opt_state, losses

    @partial(jax.jit, static_argnames="rows")
    def evaluate(params, state, coefficient, event_buffer, mask_buffer, target_buffer, rows):
      """One sequential pass over the WHOLE buffer on the ensemble-mean prediction, at a FIXED C."""
      carrier.value = coefficient
      reg = nnx.merge(regressor_definition, params, state)
      n_batches = -(-rows // eval_batch)
      pool_rows = jax.tree.leaves(event_buffer)[0].shape[0]

      def step(_, c):
        idx = jnp.clip(c * eval_batch + jnp.arange(eval_batch, dtype=jnp.int32), 0, pool_rows - 1)
        ev = jax.tree.map(lambda a: a[idx], event_buffer)
        m = mask_buffer[idx]
        feats = detector.combine_scaled(ev, theta, mask=m, reveal_design=reveals)
        element_mask = detector.element_mask(ev, m)
        normalized = detector.normalize_target(jax.tree.map(lambda a: a[idx], target_buffer))
        pred = _predict_shared(reg, feats, element_mask, members)
        return None, (detector.loss(pred, normalized), detector.metric(pred, normalized))

      _, (losses, metrics) = jax.lax.scan(step, None, jnp.arange(n_batches))
      losses = losses.reshape(-1)[:rows]
      out = {k: jnp.mean(metrics[k].reshape(-1)[:rows]) for k in labels}
      out.update(loss=jnp.mean(losses), loss_sem=jnp.std(losses) / jnp.sqrt(rows))
      return out

    print(
      f"\n=== arm {arm} === ({inserted} norms spliced, parameter checksum {checksums[arm]:.6f}, "
      f"{sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))} parameters)\n"
      f"{describe_architecture(model)}", flush=True
    )
    arm_key = jax.random.PRNGKey(order_seed)
    n_train = jnp.int32(len(train_buffer))
    history, best, arm_began = [], None, time.time()
    plot_path = os.path.join(arguments.output, f"curve_{arm}.png")
    for epoch in range(1, epochs + 1):
      arm_key, subkey = jax.random.split(arm_key)
      step0 = jnp.float32((epoch - 1) * steps_per_epoch)
      params, state, opt_state, losses = train_epoch(params, state, opt_state, subkey, step0, *train_buffer.buffers(), n_train)
      last_step = epoch * steps_per_epoch - 1
      c_now = jnp.float32(schedule(jnp.float32(last_step)))
      train = evaluate(params, state, c_now, *train_buffer.buffers(), rows=len(train_buffer))
      validation = evaluate(params, state, c_now, *validation_buffer.buffers(), rows=len(validation_buffer))
      entry = [
        epoch,
        float(jnp.mean(losses)),
        float(train["loss"]),
        float(train["loss_sem"]),
        float(validation["loss"]),
        float(validation["loss_sem"]),
        float(c_now),
        float(peak_learning_rate * 0.5 * (1.0 + np.cos(np.pi * min(last_step / total_steps, 1.0)))),
      ]
      history.append(entry)
      if best is None or entry[4] < best[0]:
        best = (entry[4], epoch, params, state, float(c_now))
      threading.Thread(target=_plot_arm, args=(list(history), arm, plot_path), daemon=True).start()
      print(
        f"  [{arm}] epoch {epoch}/{epochs}  train={entry[2]:.4f}  validation={entry[4]:.4f}  "
        f"best={best[0]:.4f}@{best[1]}  C={entry[6]:.4f}  lr={entry[7]:.2e}", flush=True
      )

    best_validation, best_epoch, best_params, best_state, best_c = best
    test = {
      k: float(x)
      for k, x in evaluate(best_params, best_state, jnp.float32(best_c), *test_buffer.buffers(), rows=len(test_buffer)).items()
    }
    seconds = time.time() - arm_began
    print(
      f"  -> [{arm}] best validation={best_validation:.4f} (epoch {best_epoch}, C={best_c:.4f})  "
      f"TEST={test['loss']:.4f}+/-{test['loss_sem']:.4f}  reported={reported:.4f}  wall={seconds / 60:.1f} min", flush=True
    )
    _plot_arm(history, arm, plot_path)
    record["arms"][arm] = {
      "norms_spliced": inserted,
      "parameter_checksum": checksums[arm],
      "best_epoch": int(best_epoch),
      "best_c": best_c,
      "validation_loss": float(best_validation),
      "test_loss": test["loss"],
      "test_sem": test["loss_sem"],
      "test_metric": {
        k: x
        for k, x in test.items() if k != "loss_sem"
      },
      "final_train": history[-1][2],
      "final_validation": history[-1][4],
      "seconds": seconds,
      "history": history,
    }
    with open(json_path, "w") as handle:
      json.dump(record, handle, indent=2)

  distinct = {round(v, 6) for v in checksums.values()}
  if len(distinct) > 1:
    raise RuntimeError(
      f"the arms did NOT start from the same network: parameter checksums {checksums}. Every norm here is "
      f"parameter-free, so one seed must give a bit-identical initialisation; a difference means the splice "
      f"reached the parameter pytree and the comparison is between initialisations, not architectures"
    )
  print(f"\n[probe] identical initialisation confirmed across {len(checksums)} arms (checksum {distinct.pop():.6f})")

  if len(record["arms"]) > 1:
    _plot_comparison(record, os.path.join(arguments.output, "comparison.png"))
  print(f"\n{'arm':<12} {'best_ep':>8} {'validation':>11} {'test':>18} {'C@best':>8} {'wall_min':>9}")
  for arm, entry in record["arms"].items():
    print(
      f"{arm:<12} {entry['best_epoch']:>8} {entry['validation_loss']:>11.4f} "
      f"{entry['test_loss']:>11.4f}+/-{entry['test_sem']:.4f} {entry['best_c']:>8.4f} {entry['seconds'] / 60:>9.1f}"
    )
  print(f"saved -> {json_path}")


if __name__ == "__main__":
  main()
