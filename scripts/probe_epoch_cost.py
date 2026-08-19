#!/usr/bin/env python3
"""STATIC cost model for one training epoch: is the loop latency-bound or arithmetic-bound?

    python scripts/probe_epoch_cost.py =bo --n-models 4 --epochs 100 --output output/epochcost/m4.json

ONE FIXED SAMPLE, NO GROWTH, NO STOPPING RULE. The pool is filled once with `iteration_limit` events at
a single design and then trained for exactly `--epochs` epochs. Nothing here decides anything; the only
output is wall time. That is deliberate -- every cost number measured so far came out of a run whose
epoch count was chosen by the convergence criterion, so cost and stopping behaviour were confounded.

WHAT IT MUST SEPARATE, and the check that it can. The question is whether the GPU is doing arithmetic
or waiting on kernel launches. Those predict OPPOSITE things for the ensemble size:

    arithmetic-bound   time per epoch proportional to n_models (1 -> 2 -> 4 doubles then doubles)
    latency-bound      time per epoch roughly FLAT in n_models -- the members are vmapped into one
                       kernel, so 4 members is a wider kernel, not 4x the launches

The two are distinguishable here because `n_models` changes the arithmetic by 4x while leaving the
number of `lax.scan` steps per epoch exactly unchanged at `iteration_limit // batch`. Any other knob
(batch, iteration_limit) moves both at once and cannot separate them.

TIMED SEPARATELY, because they are different machines:
    sample    the detector filling the pool -- CPU, and expected to be seconds
    compile   the first epoch, which pays XLA compilation
    train     epochs 2..N, the steady-state number this probe exists to produce
    eval      one train+val eval pass, which the real loop runs EVERY epoch on top of training

RUN ONE AT A TIME. Two of these on one card time-share it and both report inflated, meaningless
numbers. The launcher chains them with SLURM dependencies for exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

import detopt
import detopt.detector
import detopt.utils.config
from detopt.nn.trainer import DesignTrainer
from detopt.utils.config import optimizer as make_optimizer, resolve_device


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a run config, e.g. =bo")
  parser.add_argument("--n-models", type=int, required=True, help="ensemble members; the knob under test")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--iteration-limit", type=int, default=None, help="override training.iteration_limit")
  parser.add_argument("--budget", type=int, default=None,
                      help="override training.budget. PURE MEMORY KNOB FOR THIS PROBE: `budget` sizes the "
                           "event POOLS, but the probe fills and evaluates exactly `iteration_limit` rows, and "
                           "`steps_per_epoch = iteration_limit // batch`. So lowering it to `iteration_limit` "
                           "shrinks the allocation without changing a single measured quantity -- which is what "
                           "lets several processes share a small card for a concurrency sweep.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--design-seed", type=int, default=20260819,
                      help="only used when the config has no nominal_design; see the module docstring")
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  name = arguments.config.lstrip("=")
  run = detopt.utils.config.load_config(f"config/{name}.yaml")
  inner = next(iter(run["regressor"].values()))
  inner["n_models"] = int(arguments.n_models)
  if arguments.iteration_limit is not None:
    run["training"]["iteration_limit"] = int(arguments.iteration_limit)
  if arguments.budget is not None:
    if int(arguments.budget) < int(run["training"]["iteration_limit"]):
      raise SystemExit(
        f"probe_epoch_cost: budget {arguments.budget} < iteration_limit {run['training']['iteration_limit']}; "
        "the pool could not hold one window and the probe would measure a truncated epoch"
      )
    run["training"]["budget"] = int(arguments.budget)

  detector_config = run["detector"]
  if isinstance(detector_config, str):
    detector_config = detopt.utils.config.load_config(f"config/detector/{detector_config}.yaml")
  detector = detopt.detector.from_config(detector_config)

  training = {k: v for k, v in run["training"].items() if k != "optimizer"}
  trainer = DesignTrainer(
    detector, regressor_config=run["regressor"], optimizer=make_optimizer(run["training"]["optimizer"]),
    device=resolve_device(run.get("device")), checkpoint_dir=None, seed=arguments.seed, **training
  )
  window = int(trainer.iteration_limit)
  steps = int(trainer.steps_per_epoch)

  nominal = run.get("nominal_design")
  if nominal is not None:
    design_scaled = np.asarray(detector.to_scaled(nominal), np.float32).reshape(-1)
    design_source = "config nominal_design"
  else:
    # TIMING ONLY. Most task configs carry no `nominal_design` (BO seeds from proposals), but a cost
    # probe still has to sample SOMEWHERE. This draws one design uniformly in the SCALED cube from a
    # fixed seed and PRINTS it, so the run is reproducible and the choice is visible. It is not a
    # detector-side default: the detector is never asked for a design, and nothing here is reported as
    # a loss. ⚠️ Absolute cost can depend on the design (an ODE detector's stiffness does), so numbers
    # from different `--design-seed` values are not comparable -- but every cell of a concurrency or
    # ensemble sweep uses the SAME design, which is what those comparisons require.
    probe_rng = np.random.default_rng(int(arguments.design_seed))
    spec = detector.design_spec()
    n_design = sum(int(np.prod(v.shape)) for v in (spec if isinstance(spec, (list, tuple)) else spec._asdict().values()))
    design_scaled = probe_rng.random(n_design, dtype=np.float32)
    design_source = f"RANDOM in the scaled cube, design_seed={arguments.design_seed}"
  print(f"  design : {design_source} -> {np.round(design_scaled, 4).tolist()}", flush=True)

  print(
    f"=== {name} | n_models {arguments.n_models} | iteration_limit {window} | batch {trainer.batch} | "
    f"steps/epoch {steps} | epochs {arguments.epochs} | device {trainer.device}", flush=True
  )

  began = time.time()
  trainer.train_pool.current = 0
  added = 0
  while added < window:
    k = min(4096, window - added)
    start = trainer.train_pool.current
    record = detector.to_nominal(np.broadcast_to(design_scaled[None, :], (k, design_scaled.size)).copy())
    _gt, event, mask, target = detector(detector.flatten_design(record), trainer._train_index[start:start + k])
    trainer.train_pool.append(event, mask, target, record)
    added += k
  sample_s = time.time() - began
  print(f"  sample : {window} events in {sample_s:.1f} s -> {window / sample_s:.0f} events/s", flush=True)

  reg_def, params, state = trainer._build_regressor(int(arguments.seed))
  opt_state = trainer.optimizer.init(params)
  params, state, opt_state = jax.device_put((params, state, opt_state), trainer.device)

  train_epoch = trainer._build_train_epoch(reg_def)
  eval_pass = trainer._build_eval(reg_def, window)
  buffers = trainer.train_pool.buffers()
  key = jax.random.PRNGKey(arguments.seed)

  began = time.time()
  key, sub = jax.random.split(key)
  params, state, opt_state, losses = train_epoch(params, state, opt_state, sub, jnp.int32(0), jnp.int32(window), buffers)
  jax.block_until_ready(losses)
  compile_s = time.time() - began
  print(f"  compile: first epoch (incl. XLA compile) {compile_s:.1f} s", flush=True)

  began = time.time()
  for _ in range(arguments.epochs):
    key, sub = jax.random.split(key)
    params, state, opt_state, losses = train_epoch(params, state, opt_state, sub, jnp.int32(0), jnp.int32(window), buffers)
  jax.block_until_ready(losses)
  train_s = time.time() - began

  began = time.time()
  out = eval_pass(params, state, buffers, jnp.int32(0))
  jax.block_until_ready(out)
  eval_compile_s = time.time() - began
  began = time.time()
  for _ in range(5):
    out = eval_pass(params, state, buffers, jnp.int32(0))
  jax.block_until_ready(out)
  eval_s = (time.time() - began) / 5.0

  row = {
    "config": name,
    "n_models": int(arguments.n_models),
    "iteration_limit": window,
    "batch": int(trainer.batch),
    "steps_per_epoch": steps,
    "epochs": int(arguments.epochs),
    "sample_s": sample_s,
    "sample_events_per_s": window / sample_s,
    "compile_s": compile_s,
    "train_s_total": train_s,
    "s_per_epoch": train_s / arguments.epochs,
    "ms_per_step": 1000.0 * train_s / (arguments.epochs * steps),
    "eval_compile_s": eval_compile_s,
    "eval_s_per_pass": eval_s,
  }
  print(
    f"  train  : {arguments.epochs} epochs in {train_s:.1f} s -> {row['s_per_epoch']:.3f} s/epoch, "
    f"{row['ms_per_step']:.2f} ms/step", flush=True
  )
  print(f"  eval   : {eval_s:.3f} s per full-window pass (the real loop runs one EVERY epoch)", flush=True)
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump(row, f, indent=1)
  print(f"-> {arguments.output}", flush=True)


if __name__ == "__main__":
  main()
