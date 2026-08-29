#!/usr/bin/env python3
"""How far does the `continue` arm's carried network transfer? Evaluate the network trained at design
`i` on designs `i+1`, `i+2`, `i+3` WITHOUT any training.

    python scripts/probe_continue_transfer.py --run output/ship-addr-prec2e2/<seed>/continue \
        --first 0 --last 3 --n-events 131072 --output <dir>

WHAT IT MEASURES, and why it is not the same as the arm comparison. `continue` starts each design from
the network the PREVIOUS design finished on. Whether that is worth anything depends on how much of the
previous fit still applies at the new design -- and the campaign never measures that directly, because
the carried network is immediately trained. Here it is not: the network at design `i` is restored and
evaluated cold on the designs the run went on to try, so the number is the transfer itself with the
adaptation removed.

THE REFERENCE IS THE SAME NETWORK ON ITS OWN DESIGN. Every row also reports `i -> i`, the restored
network evaluated on the design it was trained for, sampled from FRESH events. The rise from `i -> i`
to `i -> i+k` is the transfer penalty; the `i -> i` value against the run's own reported loss is a
consistency check that the restore worked (they should agree to sampling error, not exactly, because
the events differ).

⚠️ EVENTS ARE DRAWN FRESH AND ARE NOT THE RUN'S. The detector draws from `event_index` alone, so a
fixed index range is the same physics under every design (common random numbers) -- but it is NOT the
window the run trained on, so `i -> i` prices the network on unseen events. That is the honest
comparison for transfer and it means `i -> i` may sit slightly above the reported loss.

NO TRAINING HAPPENS. The trainer is built only to supply the regressor architecture, the combine and
the loss; its pools are never grown and no optimizer step is taken.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("AGG")

import jax
import jax.numpy as jnp
from flax import nnx

import detopt
import detopt.detector
import detopt.utils.io as io
from detopt.nn.trainer import DesignTrainer


def restore_design_network(run_dir, iteration, regressor):
  """``(parameters, state, design_scaled)`` of the network the run left at ``iteration``."""
  path = os.path.join(run_dir, "checkpoints", f"design_{iteration:04d}")
  if not os.path.isdir(path):
    raise FileNotFoundError(f"no checkpoint at {path}")
  manager = io.get_checkpointer(path)
  if manager.latest_step() is None:
    raise ValueError(f"{path} holds no saved epoch")
  parameters, state, design, _aux = io.restore_training_checkpoint(manager, regressor=regressor)
  manager.close()
  return parameters, state, design


def evaluate(trainer, detector, graphdef, parameters, state, design_scaled, indices, batch):
  """Mean loss of the restored network on freshly drawn events at ``design_scaled``. No gradient."""
  design_scaled = jnp.asarray(design_scaled, jnp.float32)
  total, count = 0.0, 0
  for start in range(0, len(indices), batch):
    chunk = indices[start:start + batch]
    physical = jax.tree.map(
      lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (len(chunk), ) + jnp.asarray(x).shape),
      detector.to_nominal(design_scaled)
    )
    _truth, event, mask, target = detector(physical, chunk)
    features = trainer._combine(event, design_scaled, mask)
    model = nnx.merge(graphdef, parameters, state)
    predicted = model(features, trainer.detector.element_mask(event, mask), deterministic=True)
    loss = trainer.detector.loss(predicted, trainer.detector.normalize_target(target))
    total += float(jnp.sum(loss)) if loss.ndim else float(loss) * len(chunk)
    count += len(chunk)
  return total / count


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, help="a completed `continue` cell directory")
  parser.add_argument("--first", type=int, required=True, help="first source iteration i (inclusive)")
  parser.add_argument("--last", type=int, required=True, help="last source iteration i (inclusive)")
  parser.add_argument("--horizon", type=int, default=3, help="evaluate i on i+1 .. i+horizon")
  parser.add_argument("--n-events", type=int, default=128 * 1024)
  parser.add_argument("--batch", type=int, default=4096)
  parser.add_argument(
    "--event-offset", type=int, default=3_000_000,
    help="index base. The ship2numpy pool holds ~4.09M events and a cell spends at most 1048576 of "
    "them from index 0, so 3e6 is clear of the run's own window and leaves ~1.09M -- enough for the "
    "131072 drawn here. A larger base runs off the end of the pool and the detector raises."
  )
  parser.add_argument("--device", default=None, help="override the saved config's `device` (cuda at CERN, cpu for a smoke test)")
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  with open(os.path.join(arguments.run, "results.json")) as handle:
    payload = json.load(handle)
  config, results = payload["config"], payload["results"]
  if arguments.device is not None:
    print(f"[config] device {config.get('device')} -> {arguments.device}", flush=True)
    config = {**config, "device": arguments.device}
  detector = detopt.detector.from_config(config["detector"])
  trainer = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=0)
  graphdef, parameters, state = trainer._build_regressor(trainer.seed)

  indices = np.arange(arguments.event_offset, arguments.event_offset + arguments.n_events, dtype=np.int32)
  os.makedirs(arguments.output, exist_ok=True)
  rows = []
  for i in range(arguments.first, arguments.last + 1):
    if i + arguments.horizon >= len(results):
      print(f"[skip] i={i}: needs {i + arguments.horizon} but the run has {len(results)} designs", flush=True)
      continue
    started = time.time()
    restored, restored_state, _design_at_i = restore_design_network(arguments.run, i, (parameters, state))
    row = {
      "i": i,
      "reported_loss_at_i": float(results[i]["loss"]),
      "reported_std_at_i": float(results[i].get("loss_std", 0.0)),
      "n_events": int(arguments.n_events),
      "transfer": {},
    }
    for k in range(0, arguments.horizon + 1):
      target_design = np.asarray(results[i + k]["x_scaled"], np.float32)
      value = evaluate(trainer, detector, graphdef, restored, restored_state, target_design, indices, arguments.batch)
      row["transfer"][str(k)] = value
      print(f"[i={i}] on i+{k} (design {i + k}): {value:.5f}", flush=True)
    row["seconds"] = time.time() - started
    rows.append(row)
    with open(os.path.join(arguments.output, "transfer.json"), "w") as handle:
      json.dump({"run": arguments.run, "horizon": arguments.horizon, "rows": rows}, handle, indent=1)
  print(f"wrote {os.path.join(arguments.output, 'transfer.json')} with {len(rows)} rows", flush=True)


if __name__ == "__main__":
  main()
