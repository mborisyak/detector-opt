#!/usr/bin/env python3
"""WHY DO THE WARM-STARTED ARMS SPEND MORE? `gap` and `err` at exit, per arm.

    python scripts/probe_warm_start.py =enzyme_extremes --arm continue --seed 1 \
        --output output/warmstart/continue_s1.json

THE OBSERVATION THIS EXISTS TO EXPLAIN. On `extremes`, over 5 seeds and 130 completed runs, `meta`
spends 24-37% FEWER detector calls per design than `from_scratch` -- while `continue` and `closest`,
which also warm-start, spend 7-44% MORE. Warm-starting ought to be at worst neutral for sample
efficiency, so spending more is the anomaly.

WHY THE EXISTING DATA CANNOT ANSWER IT. `results.json` records `loss_std`, which IS `gap + err` at
exit -- but the stopping rule pins that sum at the bar (measured: 82% of 468 designs within 10% of
their mean, against a 1.0e-2 bar), so the sum carries no information about the split. Separating them
requires re-scoring with the two terms recorded individually, which is what this script does.

THE TWO HYPOTHESES, which predict opposite things:

  A  GAP INFLATION.    A warm network carries useful features, so it fits -- and therefore MEMORISES
                       -- a small window fast. Train pulls away from validation, `gap` opens, and the
                       gate buys more data. Predicts: warm arms show LOWER train loss and a LARGER
                       gap at a given window. `meta` escapes because replay draws half of every batch
                       from past designs, so the current window cannot be memorised.
  B  NEGATIVE TRANSFER. The warm start comes from a design that may be far away, so the network must
                       unlearn a wrong fit. Predicts: warm arms show HIGHER train loss and a LONGER
                       descent before the gap closes.

`train`, `val`, `gap`, `err` and the full per-epoch curves are recorded at every exit, so the two are
distinguished by inspection rather than by inference.

THE ARMS ARE NOT FOUR TRAINERS. `from_scratch`, `continue` and `closest` are all `DesignTrainer`; they
differ ONLY in what `init_params` is handed to `train`, exactly as `scripts/bo.py` does it:

    from_scratch   None
    continue       the params of the PREVIOUS design in the sequence
    closest        the params of the nearest already-scored design by L2 in the SCALED cube
    meta           `ContinualTrainer`, which ignores `init_params` and carries its own network

`bo.py` reads those params back from each design's checkpoint; this script threads the params that
`TrainResult` returns, which the checkpoint docstring states are the same arrays ("the per-design
checkpoint is written ONCE, at convergence, holding exactly the network whose loss the run reported").

ONE PROCESS = ONE (arm, seed). The designs within a process are scored IN ORDER because `continue` and
`closest` depend on that order. Cells are independent across processes and each writes its own file,
so N processes may run concurrently without a write race.
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
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.utils.config import optimizer as make_optimizer, resolve_device
from detopt.utils.events import shuffled_event_index
from detopt.utils.pools import Pool

from probe_criterion import fill, floor_at, trajectory_designs
from probe_scatter import build_test_eval
from probe_scatter import fill as fill_rows

ARMS = ("from_scratch", "continue", "closest", "meta")
TEST_INDEX_SEED = 20260818


def per_epoch(snapshot):
  """The per-epoch arrays a stopping decision was made from."""
  return {
    "train": np.asarray(snapshot["train_loss_per_epoch"], np.float64),
    "val": np.asarray(snapshot["val_loss_per_epoch"], np.float64),
    "train_sem": np.asarray(snapshot["train_sem_per_epoch"], np.float64),
    "val_sem": np.asarray(snapshot["val_sem_per_epoch"], np.float64),
    "window": np.asarray(snapshot["train_budget_per_epoch"], np.int64),
  }


def warm_start_params(arm, scored_so_far, design_scaled):
  """`init_params` for this design under this arm, following `scripts/bo.py` exactly.

  `scored_so_far` is a list of `(design_scaled, params)` in the order they were scored. Returns
  ``(params, warm_from)``; `warm_from` is the index warm-started from, or None."""
  if arm not in ("continue", "closest") or len(scored_so_far) == 0:
    return None, None
  if arm == "continue":
    index = len(scored_so_far) - 1
  else:
    previous = np.asarray([x for x, _ in scored_so_far], np.float32)
    index = int(np.argmin(np.linalg.norm(previous - design_scaled[None, :], axis=1)))
  return scored_so_far[index][1], index


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a GROWTH run config, e.g. =enzyme_extremes")
  parser.add_argument("--trajectory", required=True, help="results.json whose designs are scored, in order")
  parser.add_argument("--arm", required=True, choices=ARMS)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--n-designs", type=int, default=5, help="CONSECUTIVE designs scored, in order")
  parser.add_argument("--middle", type=int, default=None, help="explicit start index (default: the positional middle)")
  parser.add_argument("--history-rows", type=int, default=6144, help="TRAIN rows per historical design")
  parser.add_argument("--n-test", type=int, default=32768, help="HELD-OUT rows at the EXACT scored design")
  parser.add_argument("--device", default=None, help="override `device`")
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  name = arguments.config.lstrip("=")
  run = detopt.utils.config.load_config(f"config/{name}.yaml")
  if arguments.device is not None:
    run["device"] = arguments.device
  detector_config = run["detector"]
  if isinstance(detector_config, str):
    detector_config = detopt.utils.config.load_config(f"config/detector/{detector_config}.yaml")
  detector = detopt.detector.from_config(detector_config)
  has_floor = hasattr(detector, "bayes_risk")

  scored, history, window = trajectory_designs(arguments.trajectory, arguments.n_designs, arguments.middle)
  test_index = shuffled_event_index(detector.size(), arguments.n_test, TEST_INDEX_SEED, name="test events")
  history_total = len(history) * int(arguments.history_rows)

  cls = ContinualTrainer if arguments.arm == "meta" else DesignTrainer
  trainer = cls(
    detector, regressor_config=run["regressor"], optimizer=make_optimizer(run["training"]["optimizer"]),
    device=resolve_device(run.get("device")), checkpoint_dir=None, seed=arguments.seed, **{
      k: v
      for k, v in run["training"].items() if k != "optimizer"
    }
  )
  eval_test = build_test_eval(trainer, detector, arguments.n_test)
  test_pool = Pool(arguments.n_test, trainer.train_pool.specs, trainer.device)
  _definition, _params, buffer_state = trainer._build_regressor(int(arguments.seed))
  buffer_state = jax.device_put(buffer_state, trainer.device)

  started = time.time()
  trainer.train_pool.current = 0
  for design in history:
    fill(detector, trainer.train_pool, trainer._train_index, design, int(arguments.history_rows))
  if trainer.train_pool.current != history_total:
    raise SystemExit(f"probe_warm_start: laid {trainer.train_pool.current} rows, expected {history_total}")
  print(
    f"=== {name} arm {arguments.arm} seed {arguments.seed} | designs {window[0]}..{window[1] - 1} in order | "
    f"history {trainer.train_pool.current} rows in {time.time() - started:.1f} s | "
    f"steps/epoch {trainer.steps_per_epoch} | closed-form floor {has_floor}", flush=True
  )

  rows, scored_so_far = [], []
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  for position, (index, design_scaled, recorded) in enumerate(scored, start=1):
    init_params, warm_from = warm_start_params(arguments.arm, scored_so_far, design_scaled)
    design_seed = int(arguments.seed) * 100003 + index
    latest = {}

    def on_epoch(snapshot, _latest=latest):
      if snapshot["train_loss_per_epoch"].size >= _latest.get("n", 0):
        _latest["n"] = snapshot["train_loss_per_epoch"].size
        _latest["snapshot"] = snapshot

    began = time.time()
    capped = False
    try:
      result = trainer.train(design_scaled, design_seed, init_params=init_params, on_epoch=on_epoch)
    except RuntimeError as error:
      if not str(error).startswith("design did not reach precision within iteration_limit"):
        raise
      capped, result = True, None
      print(f"  position {position} design {index}: CAPPED at iteration_limit", flush=True)
    if result is None and not capped:
      print(f"  position {position} design {index}: POOL EXHAUSTED -- not a measurement", flush=True)
      break
    if "snapshot" not in latest:
      raise SystemExit("probe_warm_start: no per-epoch snapshot was captured")

    if not capped:
      test_pool.current = 0
      fill_rows(
        detector, test_pool, test_index,
        np.broadcast_to(design_scaled[None, :], (arguments.n_test, design_scaled.size)).copy()
      )
      test_eval = eval_test(result.params, buffer_state, test_pool.buffers(), jnp.int32(0))
      held_out = float(np.mean(np.asarray(test_eval, np.float64)[:arguments.n_test]))
      scored_so_far.append((design_scaled, result.params))
    else:
      held_out = None

    curves = per_epoch(latest["snapshot"])
    gap = float(abs(curves["val"][-1] - curves["train"][-1]))
    err = float(np.hypot(curves["train_sem"][-1], curves["val_sem"][-1]))
    floor = floor_at(detector, design_scaled) if has_floor else None
    rows.append({
      "arm": arguments.arm,
      "seed": int(arguments.seed),
      "design_index": int(index),
      "position": position,
      "warm_from": warm_from,
      "status": "capped" if capped else "converged",
      "window": int(curves["window"][-1]),
      "n_epochs": int(curves["train"].size),
      "n_rounds": int(np.unique(curves["window"]).size),
      "train": float(curves["train"][-1]),
      "val": None if capped else float(result.objective_loss),
      "test": held_out,
      "gap": gap,
      "err": err,
      "bayes_risk": floor,
      "spent": None if capped else int(result.spent),
      "recorded_loss": recorded,
      "wall_s": time.time() - began,
      "per_epoch": {
        k: v.tolist()
        for k, v in curves.items()
      },
    })
    print(
      f"  position {position} design {index}: window {curves['window'][-1]} epochs {curves['train'].size} "
      f"rounds {rows[-1]['n_rounds']} train {curves['train'][-1]:.4f} val {curves['val'][-1]:.4f} "
      f"test {held_out if held_out is None else round(held_out, 4)} gap {gap:.4f} err {err:.4f} "
      f"warm_from {warm_from} in {time.time() - began:.0f} s", flush=True
    )
    with open(arguments.output, "w") as f:
      json.dump({"rows": rows, "arm": arguments.arm, "seed": int(arguments.seed), "config": name}, f)

  print(f"-> {arguments.output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
  main()
