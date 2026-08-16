#!/usr/bin/env python3
"""Why is one training epoch four orders of magnitude slower than its own arithmetic?

    python scripts/probe_epoch_cost.py "=enzyme_extremes" output=<dir> batches=64,256,1024

An epoch is ONE dispatch: `Trainer._build_train_epoch` wraps `jax.lax.scan` over
`steps_per_epoch = iteration_limit // batch` in a single `jax.jit`, so host launch overhead is
amortised over thousands of steps and is NOT why the card sits at a fraction of its power cap.

WHAT THIS MUST SEPARATE, and the two hypotheses predict OPPOSITE things, so it can fail:

  * DEPENDENCY / LATENCY BOUND -- each scan iteration is a strict chain (gather minibatch -> forward
    -> backward -> optimiser update -> apply), every link is microscopic, and the device spends its
    time waiting rather than computing. Then per-step wall clock is roughly FLAT as the batch grows:
    a wider minibatch rides along inside the same latency.
  * BANDWIDTH / COMPUTE BOUND -- the step is limited by moving or multiplying its data. Then
    per-step wall clock RISES roughly in proportion to the batch.

`steps_per_epoch` is held FIXED across cells (via `iteration_limit = batch * steps`) so every cell
scans the same number of iterations and only the WIDTH changes. Otherwise batch and step count would
move together and neither could be blamed. The pool is filled once per cell with `fill` events and
the window `count` is passed at run time, so the detector cost does not scale with the batch either.

THIS IS A DIAGNOSTIC, NOT A TUNING RUN. `batch` is a settled campaign default and nothing measured
here is a reason to change it; the point is to know which resource is actually scarce.
"""

import json
import os
import time

import matplotlib

matplotlib.use("AGG")

import jax
import jax.numpy as jnp
import numpy as np

import detopt
import detopt.detector
from detopt.nn.trainer import DesignTrainer

PEAK_FLOPS_ESTIMATE = 96.0e12


def measure(detector, config, batch, steps, fill, repeats):
  """Best-of-`repeats` wall clock for one scan of `steps` train steps at this batch."""
  local = json.loads(json.dumps(config))
  local["training"]["batch"] = batch
  local["training"]["iteration_limit"] = batch * steps
  local["training"]["budget"] = fill * 4
  local["training"]["n0"] = fill
  local["training"]["n_increment"] = fill

  trainer = DesignTrainer.from_config(detector, local, checkpoint_dir=None, seed=0)
  design_scaled = np.full(int(detector.design_dim()), 0.5, dtype=np.float32)
  design = detector.to_nominal(design_scaled)
  init_seq, _ = np.random.SeedSequence(0).spawn(2)
  params, state, opt_state = trainer._init_design_network(init_seq, None)
  trainer._fill_pool(design, trainer.train_pool, fill, trainer._train_index)

  buffers = trainer.train_pool.buffers()
  key = jax.random.key(0)
  start, count = jnp.int32(0), jnp.int32(fill)

  outputs = trainer._train_epoch(params, state, opt_state, key, start, count, buffers)
  jax.block_until_ready(outputs)

  best = float("inf")
  for _ in range(repeats):
    at = time.time()
    outputs = trainer._train_epoch(params, state, opt_state, key, start, count, buffers)
    jax.block_until_ready(outputs)
    best = min(best, time.time() - at)
  ensemble = trainer.n_ensemble or 1
  del trainer
  return best, ensemble


def probe(output, batches: str = "64,256,1024", steps: int = 64, fill: int = 16384, repeats: int = 3, **config):
  os.makedirs(output, exist_ok=True)
  sizes = [int(token) for token in str(batches).replace("[", "").replace("]", "").split(",")]
  detector = detopt.detector.from_config(config["detector"])

  n_experiments = int(getattr(detector, "n_experiments", 1))
  features = config["regressor"]["set-regressor"]["features"]
  per_element_macs = sum(int(a) * int(b) for a, b in features)

  print(f"device {jax.devices()[0].device_kind}, {getattr(jax.devices()[0], 'core_count', '?')} SMs")
  print(f"steps per scan {steps}, pool fill {fill}, per-element MACs {per_element_macs}, "
        f"n_experiments {n_experiments}\n")

  rows = []
  for batch in sizes:
    seconds, ensemble = measure(detector, config, batch, steps, fill, repeats)
    per_step = seconds / steps
    macs = batch * ensemble * n_experiments * per_element_macs
    flop = 2.0 * macs * 3.0
    rows.append({
      "batch": batch, "ensemble": ensemble, "steps": steps, "scan_s": seconds, "per_step_s": per_step,
      "flop_per_step": flop, "achieved_flops": flop / per_step,
      "fraction_of_peak": flop / per_step / PEAK_FLOPS_ESTIMATE,
    })
    print(f"batch {batch:>5} (x{ensemble} members): {per_step * 1e6:>9.1f} us/step  "
          f"{flop / per_step / 1e9:>8.2f} GFLOP/s  {flop / per_step / PEAK_FLOPS_ESTIMATE * 100:>8.4f}% of peak",
          flush=True)

  print("\nper-step time against the smallest batch "
        "(FLAT => latency/dependency bound; PROPORTIONAL => bandwidth/compute bound):")
  for row in rows:
    print(f"  batch x{row['batch'] / rows[0]['batch']:>6.1f}  ->  time x{row['per_step_s'] / rows[0]['per_step_s']:>6.2f}")

  with open(os.path.join(output, "epoch_cost.json"), "w") as handle:
    json.dump(rows, handle, indent=2)
  return rows


if __name__ == "__main__":
  import sys

  import gearup

  gearup.gearup(probe).with_config("config/bo.yaml")(sys.argv[1:])
