#!/usr/bin/env python3
"""GPU throughput against process count, on a PRE-SAMPLED dataset.

Two modes, deliberately separated so the measurement is not contaminated by data generation:

    python scripts/profile_gpu_throughput.py =enzyme_extremes make output=<path.npz> gigabytes=0.5
    python scripts/profile_gpu_throughput.py =enzyme_extremes run dataset=<path.npz> seconds=90

`make` simulates events once at a fixed design and writes raw (event, mask, target) plus the scaled
design to an npz. `run` loads that file, builds the same regressor and optimiser
`scripts/verify_trajectory.py` uses, and trains for a wall-clock budget, reporting steps/s and
events/s. The training step is imported from the verification script rather than re-implemented, so
the probe measures the same computation the campaign does.

Why pre-sampled: a verification run spends most of its time simulating events, so N concurrent
verifications measure the detector's ODE integration, not the network's throughput, and the dataset
would be regenerated identically N times. Loading a fixture makes every process do the same GPU work
from the first second, and makes the dataset size an independent variable rather than a side effect.

`run` is TIME-BOUNDED, not step-bounded: it executes chunks until `seconds` elapse, so every process
takes the same wall clock whatever the hardware and the comparison across N is fair.

TWO EXECUTION MODES, `mode=scan` (default) and `mode=step`:

* `scan` -- the whole dataset lives on the DEVICE and 64 SGD steps are folded into one dispatch with
  `lax.scan`. One kernel launch amortises python and launch overhead over 64 steps, and no host/device
  traffic happens during training.
* `step` -- the dataset stays in HOST RAM, each minibatch is gathered with numpy and transferred, and
  ONE step is `jax.jit`-ed with `donate_argnums` on the parameters, the network state and the optimiser
  state so XLA updates them in place instead of allocating a new copy per step.

They trade the same two things against each other: `scan` pays device memory to avoid per-step launch
and transfer cost; `step` pays transfer and launch cost to keep device memory to the working set. Which
wins is a measurement, and on a card where memory caps the process count it is the interesting one.
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detopt.detector
import detopt.nn
from detopt.nn.trainer.common import regressor_rngs
from detopt.utils.config import resolve_device, split
from verify_trajectory import _forward_loss


def _bytes_per_event(detector):
  """Raw bytes one stored row costs: event leaves + the per-element mask + the target."""
  total = 0
  for spec in (detector.event_spec(), detector.target_spec()):
    for leaf in jax.tree.leaves(spec):
      total += int(np.prod(leaf.shape)) * np.dtype(leaf.dtype).itemsize
  total += int(detector.combined_event_shape()[0]) * 4
  return total


def make(output, gigabytes: float = 0.5, seed: int = 0, sample_batch: int = 1024, **config):
  """Simulate a fixture of ~`gigabytes` at one fixed design and save it."""
  detector = detopt.detector.from_config(config["detector"])
  per_event = _bytes_per_event(detector)
  n_events = int(float(gigabytes) * 1e9 / per_event)
  rng = np.random.default_rng(int(seed))
  theta = jnp.asarray(rng.uniform(0.0, 1.0, detector.design_dim()), jnp.float32)

  print(f"{per_event} B/event -> {n_events} events for {gigabytes} GB", flush=True)
  events, masks, targets = [], [], []
  design_dim = detector.design_dim()
  for offset in range(0, n_events, sample_batch):
    count = min(sample_batch, n_events - offset)
    physical = detector.to_nominal(jnp.broadcast_to(theta[None, :], (count, design_dim)))
    _truth, event, mask, target = detector(physical, np.arange(offset, offset + count, dtype=np.int64))
    events.append(jax.tree.map(np.asarray, event))
    masks.append(np.asarray(mask))
    targets.append(jax.tree.map(np.asarray, target))
    if offset % (sample_batch * 64) == 0:
      print(f"  {offset}/{n_events}", flush=True)

  payload = {"theta": np.asarray(theta), "mask": np.concatenate(masks)}
  for name, chunks in (("event", events), ("target", targets)):
    for index, leaf in enumerate(jax.tree.leaves(jax.tree.map(lambda *a: np.concatenate(a), *chunks))):
      payload[f"{name}_{index}"] = np.asarray(leaf)
  os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
  np.savez(output, **payload)
  size = os.path.getsize(output if output.endswith(".npz") else output + ".npz")
  print(f"saved {n_events} events -> {output} ({size / 1e9:.3f} GB on disk)", flush=True)


def run(dataset, seconds: float = 90.0, seed: int = 0, warmup_chunks: int = 1, mode: str = "scan",
        data: str = "auto", donate: str = "yes", output=None, **config):
  """Load the fixture and train against it for `seconds`, reporting throughput. See the module docstring
  for `mode`."""
  if mode not in ("scan", "step"):
    raise ValueError(f"mode must be 'scan' or 'step', got {mode!r}")
  # `data` and `donate` are independent of `mode` so the three effects -- folding, residency, buffer
  # reuse -- can be separated instead of moving together.
  residency = ("device" if mode == "scan" else "host") if data == "auto" else data
  if residency not in ("device", "host"):
    raise ValueError(f"data must be 'auto', 'device' or 'host', got {data!r}")
  donating = str(donate).lower() in ("yes", "true", "1")
  if mode == "scan" and residency != "device":
    raise ValueError("mode=scan folds over device-resident buffers; use mode=step for host residency")
  device = resolve_device(config.get("device"))
  detector = detopt.detector.from_config(config["detector"])
  batch = int(config["training"]["batch"])

  path = dataset if dataset.endswith(".npz") else dataset + ".npz"
  with np.load(path) as data:
    theta = jnp.asarray(data["theta"], jnp.float32)
    mask_buf = jax.device_put(jnp.asarray(data["mask"], jnp.int32), device)
    event_leaves = [jnp.asarray(data[f"event_{i}"]) for i in range(len([k for k in data.files if k.startswith("event_")]))]
    target_leaves = [jnp.asarray(data[f"target_{i}"]) for i in range(len([k for k in data.files if k.startswith("target_")]))]
  event_tree = jax.tree.unflatten(jax.tree.structure(detector.event_spec()), event_leaves)
  target_tree = jax.tree.unflatten(jax.tree.structure(detector.target_spec()), target_leaves)
  if residency == "device":
    event_buf = jax.device_put(event_tree, device)
    target_buf = jax.device_put(target_tree, device)
    mask_host = None
  else:
    # HOST-resident: numpy gathers the minibatch and only the batch crosses the bus.
    event_buf = jax.tree.map(np.asarray, event_tree)
    target_buf = jax.tree.map(np.asarray, target_tree)
    mask_host = np.asarray(mask_buf)
  n_rows = int(mask_buf.shape[0])

  model = detopt.nn.from_config(detector, config=config["regressor"], rngs=regressor_rngs(int(seed)))
  reg_def, params, state = nnx.split(model, nnx.Param, nnx.Variable)
  members = model.ensemble()
  draw = (members or 1) * batch
  opt_name, opt_args = split(config["training"]["optimizer"])
  optimiser = getattr(optax, opt_name)(**dict(opt_args))
  opt_state = optimiser.init(params)

  def loss_of(params, state, drop_key, event_b, mask_b, target_b):
    reg = nnx.merge(reg_def, params, state)
    feats = detector.combine_scaled(event_b, theta, mask=mask_b)
    element_mask = detector.element_mask(event_b, mask_b)
    loss = jnp.mean(
      _forward_loss(reg, detector.loss, feats, element_mask, detector.normalize_target(target_b), members, batch,
                    deterministic=False,
                    rngs=nnx.Rngs(dropout=jax.random.fold_in(drop_key, 0), dropconnect=jax.random.fold_in(drop_key, 1)))
    )
    _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
    return loss, new_state

  chunk = 64  # SGD steps folded into one dispatch, so python overhead is not what is measured

  @jax.jit
  def train_chunk(params, state, opt_state, key):
    def step(carry, k):
      params, state, opt_state = carry
      k_index, k_drop = jax.random.split(k)
      index = jax.random.randint(k_index, (draw, ), 0, n_rows)
      event_b = jax.tree.map(lambda a: a[index], event_buf)
      target_b = jax.tree.map(lambda a: a[index], target_buf)
      (loss, state), grads = jax.value_and_grad(loss_of, has_aux=True)(params, state, k_drop, event_b, mask_buf[index],
                                                                      target_b)
      updates, opt_state = optimiser.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return (params, state, opt_state), loss

    (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), jax.random.split(key, chunk))
    return params, state, opt_state, jnp.mean(losses)

  def _step(params, state, opt_state, key, event_b, mask_b, target_b):
    """ONE step. Donation is applied at the jit below, not here."""
    (loss, state), grads = jax.value_and_grad(loss_of, has_aux=True)(params, state, key, event_b, mask_b, target_b)
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, state, opt_state, loss

  def _step_device(params, state, opt_state, key, index):
    """Device-resident variant: the minibatch is gathered INSIDE the jit, so nothing crosses the bus."""
    return _step(params, state, opt_state, key, jax.tree.map(lambda a: a[index], event_buf), mask_buf[index],
                 jax.tree.map(lambda a: a[index], target_buf))

  # DONATED (when `donate=yes`): parameters, network state and optimiser state are passed AND returned,
  # so XLA may write the updates into the same device buffers rather than allocating a fresh copy.
  donated = ("params", "state", "opt_state") if donating else ()
  train_step = jax.jit(_step, donate_argnames=donated)
  train_step_device = jax.jit(_step_device, donate_argnames=donated)

  def one_chunk(params, state, opt_state, key, generator):
    """`chunk` per-step calls. HOST residency gathers with numpy and transfers the batch; DEVICE
        residency passes only the index and gathers inside the kernel."""
    for _ in range(chunk):
      key, k_index, sub = (lambda a, b, c: (a, b, c))(*jax.random.split(key, 3))
      if residency == "host":
        index = generator.integers(0, n_rows, draw)
        event_b = jax.tree.map(lambda a: jnp.asarray(a[index]), event_buf)
        target_b = jax.tree.map(lambda a: jnp.asarray(a[index]), target_buf)
        params, state, opt_state, loss = train_step(params, state, opt_state, sub, event_b,
                                                    jnp.asarray(mask_host[index]), target_b)
      else:
        index = jax.random.randint(k_index, (draw, ), 0, n_rows)
        params, state, opt_state, loss = train_step_device(params, state, opt_state, sub, index)
    return params, state, opt_state, loss, key

  key = jax.random.PRNGKey(int(seed))
  generator = np.random.default_rng(int(seed))
  print(f"loaded {n_rows} rows from {path} | mode={mode} data={residency} donate={donating} | batch {batch} x {members or 1} members | "
        f"chunk {chunk} steps", flush=True)

  # Warm-up is EXCLUDED from the measurement: the first chunk pays for XLA compilation, which is a
  # fixed cost per process and would otherwise be charged to whichever N ran first.
  def advance(params, state, opt_state, key):
    if mode == "scan":
      key, sub = jax.random.split(key)
      params, state, opt_state, loss = train_chunk(params, state, opt_state, sub)
      return params, state, opt_state, loss, key
    return one_chunk(params, state, opt_state, key, generator)

  for _ in range(int(warmup_chunks)):
    params, state, opt_state, loss, key = advance(params, state, opt_state, key)
    loss.block_until_ready()

  steps, started = 0, time.time()
  while time.time() - started < float(seconds):
    params, state, opt_state, loss, key = advance(params, state, opt_state, key)
    loss.block_until_ready()
    steps += chunk
  elapsed = time.time() - started

  record = {
    "mode": mode,
    "residency": residency,
    "donate": donating,
    "dataset": path,
    "rows": n_rows,
    "batch": batch,
    "members": members or 1,
    "steps": steps,
    "elapsed_s": elapsed,
    "steps_per_s": steps / elapsed,
    "samples_per_s": steps * draw / elapsed,
    "final_loss": float(loss),
    "pid": os.getpid(),
  }
  print(
    f"steps {steps} in {elapsed:.1f}s -> {record['steps_per_s']:.1f} steps/s, "
    f"{record['samples_per_s']:.0f} samples/s, loss {record['final_loss']:.4f}", flush=True
  )
  if output is not None:
    with open(output, "w") as f:
      json.dump(record, f, indent=2)


if __name__ == "__main__":
  import gearup

  gearup.gearup(make=make, run=run).with_config("config/root.yaml")(sys.argv[1:])
