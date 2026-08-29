#!/usr/bin/env python3
"""Which REGRESSOR ARCHITECTURE can converge at a tighter `loss_precision`, and what does it cost?

    python scripts/probe_architecture.py =enzyme_inhib \
        --results output/campaign-inhib/126382657/from_scratch/results.json \
        --rank 0 --arch baseline shallow-24-16 wd-1e-2 --precision 3.2e-3 \
        --output output/screen/arch-rank0-a.json

THE PROBLEM. `detopt/nn/trainer/design.py` stops a design when `diff + err < loss_precision`, where
`err` is the loss estimate's standard error (falls like `1/sqrt(window)`, so data always fixes it) and
`diff = |train - val|` is the finite-sample overfitting gap, and it falls with data too -- both
splits are IID from one stream and the network is fixed-capacity, so both empirical risks converge
to the same population risk. What bites is the PRE-ASYMPTOTIC regime at an affordable window. Measured there, on the campaign's
own worst design, `diff` alone is 0.0055 with the shipped set regressor, which is why `loss_precision`
had to be raised to 8.0e-3 -- and that puts criterion (d)'s bar (`10 x loss_precision`) at 0.080
against a baseline-to-best span of ~0.24. The benchmark's resolution is then set by the REGRESSOR's
overfitting, not by the search.

⚠️ NOT MONOTONE, and the sign is design-dependent. Uniform convergence gives `diff -> 0`
ASYMPTOTICALLY; it says nothing about the path. At small `n` an underfitting network holds train
and val both high and close, so `diff` is small; as `n` grows and the network starts to fit, train
falls faster than val and `diff` RISES; only later does val catch up and `diff` fall. Measured on
SHiP: median slope `dlog(diff)/dlog(window) = -1.38` over 9 designs, but `+0.428` on one design
that was still rising at window 262,144 (`err` on that same design fell at -0.495, matching
`n^-1/2` to three digits, so the measurement is sound and it is `diff` alone that is unruly).
Do NOT assume a sign for the affordable range.

WHAT THIS PROBE MUST SEPARATE, stated before running it. A `diff` floor of ~0.005 (the shipped
architecture, which cannot converge at a tighter precision) from a floor of ~0.001-0.0015 (which
would permit `loss_precision = 3.2e-3`, the 2.5x reduction that matters). It can: `err` at the window
cap (`iteration_limit = 131072` train + 43691 val) is ~0.0013 measured, so the slack `diff + err` is
resolved to about a thousandth -- five times finer than the difference asked about. The LEVEL
differences that decide whether an architecture is merely too weak (dropout 0.2 cost +0.034 of level)
are similarly resolved: the level's own standard error is `err/2` ~ 0.001.

THE PROTOCOL. Every setting is run AT the precision being asked about (`--precision`, default the 2.5x
target 3.2e-3) rather than at the campaign's 8.0e-3, so the run's dynamics are exactly a campaign's at
that precision:

* it CONVERGES -> the architecture permits that `loss_precision`, and the row's `spend` is what a
  design would then cost (the campaign's budget is 2 097 152 calls and criterion (b)/(c) needs 30
  designs, i.e. <= 69 905 calls a design -- a converged row that spends more is still a failure);
* it hits the window cap -> it does NOT permit it, and the row's `slack` is the FLOOR of `diff + err`
  the architecture reaches on its full window, i.e. the tightest precision it could ever permit.

TWO DESIGNS, and why both are needed. `--rank 0` is the WORST design of a finished BO run (the most
uninformative one the search actually visited) -- where the gap bites and where campaigns crash.
`--rank <last>` is the BEST. An architecture that closes the gap by being too weak to fit anything
shows up as a raised level on the GOOD design, and the decision quantity is not the gap but the gap
RELATIVE TO THE SPAN `level(worst) - level(good)` measured under the same architecture: shrinking the
gap 3x while compressing the span 3x buys the benchmark nothing.

The full per-epoch history (train/val loss, their SEMs, the window) is written next to the json as an
npz per (architecture, rank), so `diff` vs window can be read afterwards without re-running.
"""

import argparse
import gc
import json
import os
import time

import matplotlib

matplotlib.use("AGG")

import jax
import numpy as np
import yaml

import detopt
import detopt.detector
from detopt.nn.trainer import DesignTrainer

# Each entry patches the run config: a whole `regressor` block, plus optionally the AdamW
# `weight_decay` (the config ships 1.0e-6, i.e. off). The hypothesis each one tests is stated -- the
# comparison is only interpretable if the reason for including a setting was fixed in advance.
ARCHITECTURES = {
  # The shipped architecture. Two blocks, each a 2-weight-layer MLP per element (hidden -> output),
  # so four weight layers per element plus the head. Measured floor: diff 0.0055 at the worst design.
  "baseline": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16], [16, 24]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
  },
  # ONE block: input -> hidden -> output per element, one aggregation, linear head. Strictly shallower
  # than the baseline (half the weight layers, and the per-element net never sees the aggregated event
  # representation). Fewer parameters should fit less read-out noise; the good design says whether it
  # can still fit the signal.
  "shallow-24-16": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
  },
  "shallow-16-8": {
    "regressor": {
      "set-regressor": {
        "features": [[16, 8]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
  },
  # Same depth as `shallow-24-16`, more width: separates "depth costs gap" from "capacity costs gap".
  "shallow-32-24": {
    "regressor": {
      "set-regressor": {
        "features": [[32, 24]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
  },
  # Capacity FLOOR control: two blocks with no hidden layer at all, so each block is a single affine
  # map and the only nonlinearity left is the aggregation gate. If the gap survives here it is not a
  # capacity problem; if the level collapses here the good-design column has something to measure
  # against.
  "linear-16": {
    "regressor": {
      "set-regressor": {
        "features": [[16], [16]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
  },
  # The residual variant: per-element stack of `depth` units `h <- h + alpha * f(h)` with alpha
  # zero-initialised, so the network starts as the identity in its residual part and the optimiser
  # decides how much capacity to switch on -- capacity that is present but not used cannot overfit.
  "alpha-w32-d2": {
    "regressor": {
      "alpha-set-regressor": {
        "features": [16, 24],
        "width": 32,
        "depth": 2,
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
  },
  "alpha-w64-d4": {
    "regressor": {
      "alpha-set-regressor": {
        "features": [16, 24],
        "width": 64,
        "depth": 4,
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
  },
  # Decoupled weight decay on the shipped architecture. Unlike dropout it does not inject noise into
  # the forward pass, so it should not move the converged level the way p_dropout 0.2 did (+0.034).
  # At lr 2.5e-4 the per-step shrinkage is lr*wd, so 1e-2 and 1e-1 bracket "mild" and "strong" over
  # the ~10^5 steps a design takes; 1e-3 would be a 6% total shrinkage, i.e. still off.
  "wd-1e-2": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16], [16, 24]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 1.0e-2,
  },
  "wd-1e-1": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16], [16, 24]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 1.0e-1,
  },
  "wd-1e0": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16], [16, 24]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 1.0,
  },
  # MEASURED bracket: 1e-2 still leaves diff 0.0031 on the informative design, 1e0 closes the gap but
  # costs 0.118 of level there (it underfits the signal). These two say how sharp the optimum between
  # them is -- a recommendation that only works at exactly 1e-1 is a worse recommendation.
  "wd-3e-2": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16], [16, 24]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 3.0e-2,
  },
  "wd-3e-1": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16], [16, 24]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 3.0e-1,
  },
  # MEASURED motivation: at the informative design the UNREGULARISED high-capacity nets reach a
  # validation loss of 0.705 (`linear-16`) and 0.713 (`alpha-w32-d2`) against the shipped net's 0.778,
  # i.e. there is signal the shipped architecture never extracts -- but they pay a gap of 0.004-0.068
  # that makes them unusable. Weight decay 1e-1 closed the gap on the shipped net at no cost in level;
  # these two ask whether it also closes it on a net with the capacity to reach the lower level, which
  # would WIDEN the benchmark's span instead of merely tightening its convergence.
  "wide-wd-1e-1": {
    "regressor": {
      "set-regressor": {
        "features": [[48, 32], [32, 48]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 1.0e-1,
  },
  "alpha-w32-d2-wd-1e-1": {
    "regressor": {
      "alpha-set-regressor": {
        "features": [16, 24],
        "width": 32,
        "depth": 2,
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 1.0e-1,
  },
  # The two levers that each helped on their own, together.
  "shallow-24-16-wd-1e-1": {
    "regressor": {
      "set-regressor": {
        "features": [[24, 16]],
        "n_models": 4,
        "p_dropout": 0.1
      }
    },
    "weight_decay": 1.0e-1,
  },
}

def load_run_config(name):
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)
  # A RUN config names its detector by string and gearup resolves it against config/detector/<name>.yaml;
  # doing the same here keeps the probe on exactly the detector the campaign ran.
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  return config, detector_config

def count_parameters(detector, regressor_config):
  from flax import nnx

  from detopt.nn import from_config as regressor_from_config

  model = regressor_from_config(detector, config=regressor_config, rngs=nnx.Rngs(0))
  _, params, _ = nnx.split(model, nnx.Param, nnx.Variable)
  return int(sum(int(np.asarray(leaf).size) for leaf in jax.tree.leaves(params)))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token, e.g. =enzyme_inhib")
  parser.add_argument("--results", required=True, help="a finished results.json to take the design(s) from")
  parser.add_argument("--rank", type=int, nargs="+", default=[0], help="0 = the WORST design; -1 = the BEST")
  parser.add_argument("--arch", nargs="+", default=["baseline"], help=f"keys of {sorted(ARCHITECTURES)}")
  parser.add_argument("--precision", type=float, default=3.2e-3, help="the loss_precision being asked about")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--device", default=None, help="override the config's device, e.g. cpu")
  parser.add_argument("--iteration-limit", type=int, default=None, help="SMOKE ONLY -- it sets the epoch length")
  parser.add_argument("--budget", type=int, default=None, help="SMOKE ONLY -- pool size")
  parser.add_argument("--output", default="output/screen/arch-probe.json")
  arguments = parser.parse_args()

  config, detector_config = load_run_config(arguments.config.lstrip("="))
  detector = detopt.detector.from_config(detector_config)

  with open(arguments.results) as f:
    recorded = json.load(f)["results"]
  order = sorted(recorded, key=lambda r: -r["loss"])  # worst first
  print(f"{len(order)} designs in {arguments.results}: worst {order[0]['loss']:.4f}, best {order[-1]['loss']:.4f}", flush=True)

  rows = []
  history_dir = os.path.splitext(arguments.output)[0]
  os.makedirs(history_dir, exist_ok=True)
  for rank in arguments.rank:
    # CLAMPED, as `probe_dropout.py` does: a reference file may hold only a few designs while the rank
    # naming the BEST one is written as a large number (`--rank 999`). Negative ranks index from the
    # good end unchanged (`min(-1, n-1) = -1`).
    chosen = order[min(rank, len(order) - 1)]
    x_scaled = np.asarray(chosen["x_scaled"], dtype=np.float32)
    print(f"\n=== rank {rank}: recorded loss {chosen['loss']:.4f} (its own std {chosen['loss_std']:.4f})", flush=True)

    for name in arguments.arch:
      patch = ARCHITECTURES[name]
      run_config = json.loads(json.dumps(config))  # deep copy; the config is plain JSON-able yaml
      run_config["regressor"] = json.loads(json.dumps(patch["regressor"]))
      run_config["training"]["loss_precision"] = float(arguments.precision)
      if "weight_decay" in patch:
        run_config["training"]["optimizer"]["adamw"]["weight_decay"] = float(patch["weight_decay"])
      if arguments.device is not None:
        run_config["device"] = arguments.device
      if arguments.iteration_limit is not None:
        run_config["training"]["iteration_limit"] = int(arguments.iteration_limit)
      if arguments.budget is not None:
        run_config["training"]["budget"] = int(arguments.budget)

      n_parameters = count_parameters(detector, run_config["regressor"])
      # Each setting builds its OWN trainer, and a trainer owns event pools sized to the whole budget
      # (~0.5 GB of device memory here). Dropping the previous one BEFORE constructing the next keeps
      # the peak at one trainer instead of two -- the card is 8 GB and is shared with other jobs.
      trainer = None
      gc.collect()
      trainer = DesignTrainer.from_config(detector, run_config, checkpoint_dir=None, seed=arguments.seed)
      sequence = np.random.SeedSequence(arguments.seed)

      # The trainer's per-epoch callback carries the whole history (train/val loss, their SEMs and the
      # window), so `diff` and `err` at the STOPPING epoch are read from it directly instead of being
      # parsed out of a print -- and they are available on the failure path too, where the RuntimeError
      # is the answer rather than an accident.
      history = {}
      started = time.time()
      try:
        result = trainer.train(x_scaled, int(sequence.spawn(1)[0].generate_state(1)[0]), step=0, on_epoch=history.update)
        status = "converged"
        level, spent = float(result.objective_loss), int(result.spent)
      except RuntimeError as error:
        status = "hit the window cap"
        level, spent = float("nan"), -1
        print(f"  {name}: {status} -- {str(error).splitlines()[0]}", flush=True)
      elapsed = time.time() - started

      train_history = history["train_loss_per_epoch"]
      val_history = history["val_loss_per_epoch"]
      diff = float(abs(val_history[-1] - train_history[-1]))
      err = float(np.hypot(history["train_sem_per_epoch"][-1], history["val_sem_per_epoch"][-1]))
      window = int(history["train_budget_per_epoch"][-1])
      row = {
        "arch": name,
        "rank": rank,
        "recorded_loss": float(chosen["loss"]),
        "status": status,
        "level": level,
        "diff": diff,
        "err": err,
        "slack": diff + err,
        "window": window,
        "spend": spent,
        "n_parameters": n_parameters,
        "n_epochs": int(len(train_history)),
        "seconds": elapsed,
        "precision": float(arguments.precision),
      }
      rows.append(row)
      print(
        f"  {name}: {status}  level {level:.4f}  diff {diff:.4g}  err {err:.4g}  "
        f"diff+err {diff + err:.4g}  window {window}  spend {spent}  "
        f"params {n_parameters}  epochs {row['n_epochs']}  {elapsed:.0f} s", flush=True
      )

      np.savez(os.path.join(history_dir, f"{name}_rank{rank}.npz"), **history)
      with open(arguments.output, "w") as f:
        json.dump({
          "results": arguments.results,
          "precision": arguments.precision,
          "seed": arguments.seed,
          "rows": rows
        }, f, indent=2, default=float)

  print(f"\nwrote {arguments.output} ({len(rows)} rows) and per-epoch histories to {history_dir}/", flush=True)
  print("READ IT AS: a row that CONVERGED permits this loss_precision at that spend; a row that hit the")
  print("cap does not, and its diff+err is the tightest precision that architecture could ever permit.")

if __name__ == "__main__":
  main()
