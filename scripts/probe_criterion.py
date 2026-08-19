#!/usr/bin/env python3
"""BAYESIAN vs STRICT convergence criterion, on the SAME mid-trajectory designs.

    python scripts/probe_criterion.py --output output/criterion/probe.json

THE QUESTION. Two stopping rules, everything else identical:

  bayesian  `detopt/nn/trainer/design.py` -- add data while the gap trend says it will still exceed
            the bar; converge when `P(|slope| * patience < loss_precision/2) > 0.9` AND
            `gap + err <= loss_precision`. Warmup 2, patience 16.
  strict    `detopt/nn/trainer/strict.py` -- add data while `gap + err > loss_precision`; converge
            when `patience` epochs pass with NO IMPROVEMENT on the validation loss. Warmup 32,
            patience 10.

The SCORING gate is the same in both (`gap + err <= loss_precision`, report `val` with `gap + err` as
its noise), so a difference between them is a difference between CONVERGENCE TESTS and nothing else.
Each rule runs at ITS OWN specified settings; forcing one procedure's warmup onto the other would
measure something neither rule is.

WHY THE MIDDLE OF A TRAJECTORY, and not the first designs. `bo.n_init = 5`, so a run's first five
designs are a seeded Sobol block with no GP in the loop -- and on this project's tasks they sit near
the no-information ceiling, where a network converges trivially because there is nothing to fit. The
designs BO actually spends its life on are mid-trajectory. This probe takes five CONSECUTIVE designs
from the positional middle of a finished trajectory and lays every design BEFORE them into the pool as
history, so each is scored where a real run would meet it.

WHY `linear`. `LinearDetector.bayes_risk` returns the EXACT best loss achievable at a design, so
`excess = reported - floor` is a direct measurement of each rule's BIAS rather than a proxy for one.
No other task in this project allows it. ⚠️ `bayes_risk` takes a NOMINAL design; feeding it the scaled
vector silently returns a larger, wrong floor.

WHAT IS SHARED WITHIN A PAIR: the design, the seed, the network initialisation, the history, the
event indices, the growth quanta, `loss_precision`, the optimiser and the architecture. Only the
convergence test differs.

THIS SCRIPT SETS NOTHING. It writes a JSON of measurements. `design.py` is used AS IS and never
modified; `strict.py` is the sibling procedure under test.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import detopt
import detopt.detector
import detopt.utils.config
from detopt.utils.viz.bo import plot_iteration
from detopt.nn.trainer import DesignTrainer
from detopt.nn.trainer.strict import StrictDesignTrainer

TRAJECTORY = "output/linear-d2n3-fixed/1244111331/from_scratch/results.json"
CRITERIA = {"bayes": ("linear_d2n3_growth", DesignTrainer), "strict": ("linear_d2n3_strict", StrictDesignTrainer)}


def trajectory_designs(path, n_designs, middle=None):
  """`(scored, history, indices)` -- `n_designs` CONSECUTIVE designs from the positional middle of the
  trajectory, and every design before the first of them as history. Taken by POSITION, never by loss."""
  with open(path) as f:
    rows = [r for r in json.load(f)["results"] if r.get("loss") is not None]
  start = (len(rows) - n_designs) // 2 if middle is None else int(middle)
  if start <= 0 or start + n_designs > len(rows):
    raise SystemExit(f"probe_criterion: window {start}..{start + n_designs} outside 1..{len(rows)}")
  scored = [(index, np.asarray(rows[index]["x_scaled"], np.float32), rows[index]["loss"])
            for index in range(start, start + n_designs)]
  history = [np.asarray(r["x_scaled"], np.float32) for r in rows[:start]]
  return scored, history, (start, start + n_designs)


def fill(detector, pool, index_array, design_scaled, n_rows, chunk=1024):
  """Append `n_rows` events simulated AT `design_scaled` and stored WITH it, exactly as the trainer's
  own `_fill_pool` does -- the event indices are the next slice at the pool's current fill, so the
  detector call stays deterministic in `(design, event_index)`."""
  added = 0
  while added < n_rows:
    k = min(chunk, n_rows - added)
    start = pool.current
    record = detector.to_nominal(np.broadcast_to(design_scaled[None, :], (k, design_scaled.size)).copy())
    _ground_truth, event, mask, target = detector(detector.flatten_design(record), index_array[start:start + k])
    pool.append(event, mask, target, record)
    added += k


def floor_at(detector, design_scaled):
  """The CLOSED-FORM best loss at this design. `bayes_risk` wants a NOMINAL design."""
  nominal = detector.to_nominal(np.asarray(design_scaled, np.float32)[None, :])
  return float(detector.bayes_risk(np.asarray(detector.flatten_design(nominal), np.float32)[0]))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--trajectory", default=TRAJECTORY)
  parser.add_argument("--n-designs", type=int, default=5, help="consecutive designs from the middle")
  parser.add_argument("--middle", type=int, default=None, help="explicit start index (default: the middle)")
  parser.add_argument("--history-rows", type=int, default=4096, help="TRAIN rows laid down per historical design")
  parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
  parser.add_argument("--criteria", nargs="+", default=sorted(CRITERIA), choices=sorted(CRITERIA))
  parser.add_argument("--output", default="output/criterion/probe.json")
  parser.add_argument(
    "--plot-per-epoch", type=int, default=8,
    help="render the per-epoch convergence curve every Nth epoch (0 = off). A STRIDE, not a switch, "
    "exactly as `bo.py` treats `plot_per_epoch` -- the render is the expensive half and runs on a "
    "detached worker, so a stride of 8 shows every design's shape at an eighth of the cost"
  )
  arguments = parser.parse_args()

  scored, history, window = trajectory_designs(arguments.trajectory, arguments.n_designs, arguments.middle)
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  print(
    f"trajectory {arguments.trajectory}: designs {window[0]}..{window[1] - 1} scored, "
    f"{len(history)} designs of history x {arguments.history_rows} rows", flush=True
  )

  rows = []
  for criterion in arguments.criteria:
    config_name, trainer_cls = CRITERIA[criterion]
    config = detopt.utils.config.load_config(f"config/{config_name}.yaml")
    detector_config = config["detector"]
    if isinstance(detector_config, str):
      detector_config = detopt.utils.config.load_config(f"config/detector/{detector_config}.yaml")
    detector = detopt.detector.from_config(detector_config)
    training = config["training"]
    print(
      f"\n=== {criterion}: warmup {training['warmup_epochs']} patience {training['patience']} "
      f"n0 {training['n0']} increment {training['n_increment']} precision {training['loss_precision']}", flush=True
    )

    for seed in arguments.seeds:
      trainer = trainer_cls.from_config(detector, config, checkpoint_dir=None, seed=seed)
      trainer.train_pool.current = 0
      started = time.time()
      for design in history:
        fill(detector, trainer.train_pool, trainer._train_index, design, arguments.history_rows)
      history_end = trainer.train_pool.current
      print(f"  seed {seed}: history {history_end} rows in {time.time() - started:.0f} s", flush=True)

      plots_dir = os.path.join(os.path.dirname(arguments.output) or ".", "plots", criterion, f"seed{seed}")
      if arguments.plot_per_epoch > 0:
        os.makedirs(plots_dir, exist_ok=True)

      for index, design_scaled, recorded in scored:
        floor = floor_at(detector, design_scaled)
        design_phys = np.asarray(detector.flatten_design(detector.to_nominal(design_scaled[None, :])), np.float32)[0].tolist()

        latest = {}

        def on_epoch(snapshot, _index=index, _design=design_phys, _dir=plots_dir, _latest=latest):
          # Keep the LONGEST snapshot seen -- the arrays are cumulative, so the last epoch's is the
          # whole trajectory. Callbacks run on detached threads, so order is not guaranteed and the
          # length is the only reliable discriminator.
          if snapshot["train_loss_per_epoch"].size >= _latest.get("n", 0):
            _latest["n"] = snapshot["train_loss_per_epoch"].size
            _latest["snapshot"] = snapshot
          # The snapshot's length IS the epoch count, so the stride gates here. The FIRST epoch and
          # every Nth are drawn: without the first, a design that converges inside one stride would
          # produce no plot at all.
          per_epoch = snapshot["val_loss_per_epoch"]
          epoch = int(per_epoch.size)
          if epoch != 1 and epoch % arguments.plot_per_epoch != 0:
            return
          live = float(per_epoch[-1]) if per_epoch.size > 0 else float("nan")
          plot_iteration(snapshot, iteration=_index, design=_design, val_loss=live, plots_dir=_dir)

        began = time.time()
        result = trainer.train(
          design_scaled,
          int(seed) * 100003 + index, on_epoch=on_epoch if arguments.plot_per_epoch > 0 else None
        )
        if "snapshot" in latest:
          np.savez(os.path.join(plots_dir, f"iter_{index:03d}_history.npz"), **latest["snapshot"])
        if result is None:
          print(f"    design {index}: POOL EXHAUSTED -- not a measurement", flush=True)
          continue
        rows.append({
          "criterion": criterion,
          "seed": int(seed),
          "design_index": int(index),
          "recorded_loss": recorded,
          "bayes_risk": floor,
          "reported": float(result.objective_loss),
          "reported_std": float(result.objective_std),
          "excess": float(result.objective_loss) - floor,
          "spent": int(result.spent),
          "wall_s": time.time() - began,
          "warmup_epochs": int(training["warmup_epochs"]),
          "patience": int(training["patience"]),
        })
        print(
          f"    design {index}: reported {result.objective_loss:.4f} floor {floor:.4f} "
          f"excess {rows[-1]['excess']:+.5f} spent {result.spent} in {rows[-1]['wall_s']:.0f} s", flush=True
        )
        with open(arguments.output, "w") as f:
          json.dump({"rows": rows, "window": window, "trajectory": arguments.trajectory}, f, indent=2)

  print(f"\n-> {arguments.output}")
  for criterion in arguments.criteria:
    subset = [r for r in rows if r["criterion"] == criterion]
    if len(subset) == 0:
      continue
    print(
      f"  {criterion:<7} n={len(subset):2d}  spent median {np.median([r['spent'] for r in subset]):8.0f}  "
      f"excess median {np.median([r['excess'] for r in subset]):+.5f}  "
      f"wall median {np.median([r['wall_s'] for r in subset]):6.0f} s"
    )


if __name__ == "__main__":
  main()
