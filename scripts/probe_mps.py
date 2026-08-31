#!/usr/bin/env python3
"""ONE campaign-shaped unit of work, timed. The worker of the MPS / concurrency probe.

    python scripts/probe_mps.py =enzyme_extremes output=<dir> seed=0 designs=1

WHAT THIS MUST SEPARATE, stated before it is run. Choosing how many campaign jobs share one GPU is a
question about which ceiling binds first, and the three candidates have different fixes, so a probe
that cannot tell them apart is worthless:

  1. SM OCCUPANCY -- without CUDA MPS, N processes TIME-SLICE the device rather than co-running, so
     aggregate throughput is flat in N. MPS is the fix, and it is the thing under test.
  2. HOST CPU -- this box has 24 cores and a campaign job is scheduled at 4, i.e. exactly 6 jobs.
     If 6 is where throughput saturates, the GPU was never the constraint and MPS settings are the
     wrong knob.
  3. DEVICE MEMORY -- ~2.2 GiB per job against 96 GiB is ~40 jobs, so this is expected NOT to bind.
     It is measured anyway, because the expectation rests on a figure taken on a different card.

The unit is a REAL design through the REAL trainer, not a synthetic kernel: the campaign's cost is an
interleaving of detector calls (RKC2 integration, latency-bound) with training steps, and measuring
either half alone would give two different answers with no way to combine them. It is fixed work in
the sense that matters here -- the same (design, seed, config) does the same computation no matter
how many copies run beside it -- so wall clock is directly comparable across concurrency levels.

`training.n0` and `training.n_increment` may be scaled DOWN on the command line to make the sweep
affordable. That keeps the interleaving and the kernel mix while shrinking the window, which is a
valid proxy for a SCHEDULING question and is not a physics measurement. The chosen concurrency is
then confirmed at full campaign settings before anything is launched.

Reports per design: wall clock, detector calls spent, the objective and its uncertainty; and for the
process: peak and live device bytes from the device's own allocator.
"""

import json
import os
import time

import matplotlib

matplotlib.use("AGG")

import jax
import numpy as np

import detopt
import detopt.detector
import detopt.utils.io
from detopt.nn.trainer import ContinualTrainer, DesignTrainer


def memory_stats():
  """Peak and live device bytes as the device's own allocator reports them, or an empty dict on CPU."""
  device = jax.devices()[0]
  stats = getattr(device, "memory_stats", lambda: None)()
  if stats is None:
    return {}
  return {
    "bytes_in_use": int(stats.get("bytes_in_use", 0)),
    "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
    "bytes_limit": int(stats.get("bytes_limit", 0)),
  }


def probe(output, seed: int = 0, designs: int = 1, strategy: str = "from_scratch", **config):
  seed = int(seed)
  designs = int(designs)
  os.makedirs(output, exist_ok=True)

  started = time.time()
  detector = detopt.detector.from_config(config["detector"])
  dimension = int(detector.design_dim())
  network_seq, iteration_seq = np.random.SeedSequence(seed).spawn(2)
  trainer_cls = ContinualTrainer if strategy == "meta" else DesignTrainer
  trainer = trainer_cls.from_config(
    detector, config, checkpoint_dir=os.path.join(output, "checkpoints"),
    seed=int(network_seq.generate_state(1)[0])
  )
  build_seconds = time.time() - started

  rows = []
  proposal_rng = np.random.default_rng(seed)
  for index in range(designs):
    design_scaled = proposal_rng.random(dimension).astype(np.float32)
    design_seed = int(iteration_seq.generate_state(1)[0])
    at = time.time()
    result = trainer.train(design_scaled, design_seed, step=index)
    elapsed = time.time() - at
    if result is None:
      rows.append({"design": index, "time_s": elapsed, "exhausted": True})
      break
    rows.append({
      "design": index,
      "time_s": elapsed,
      "spent": int(result.spent),
      "objective_loss": float(result.objective_loss),
      "objective_std": float(result.objective_std),
      "exhausted": False,
    })
    print(f"[probe] design {index}: {elapsed:.1f} s, {int(result.spent)} calls", flush=True)

  total = time.time() - started
  record = {
    "pid": os.getpid(),
    "seed": seed,
    "strategy": strategy,
    "designs_requested": designs,
    "designs_completed": sum(1 for row in rows if row.get("exhausted") is False),
    "build_s": build_seconds,
    "total_s": total,
    "train_s": sum(row["time_s"] for row in rows),
    "spent": sum(row.get("spent", 0) for row in rows),
    "backend": jax.default_backend(),
    "memory": memory_stats(),
    "mps_pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY", ""),
    "active_thread_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", ""),
    "n0": int(config["training"]["n0"]),
    "n_increment": int(config["training"]["n_increment"]),
    "iteration_limit": int(config["training"]["iteration_limit"]),
    "rows": rows,
  }
  with open(os.path.join(output, "probe.json"), "w") as handle:
    json.dump(record, handle, indent=2)
  print(f"[probe] total {total:.1f} s ({build_seconds:.1f} s build), {record['spent']} calls, "
        f"peak {record['memory'].get('peak_bytes_in_use', 0) / 2**30:.2f} GiB", flush=True)
  return record


if __name__ == "__main__":
  import sys

  import gearup

  gearup.gearup(probe).with_config("config/root.yaml")(sys.argv[1:])
