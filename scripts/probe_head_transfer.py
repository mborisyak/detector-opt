"""HEAD-ONLY transfer between consecutive BO designs: how much survives if only the last layer moves.

    python scripts/probe_head_transfer.py --run <cell> --pair 9 --representation strip \
        --reveal none --output <dir>

The twin of ``probe_strip_transfer.py``. That one lets the warm start tune EVERYTHING; this one freezes
the backbone and tunes the final layer alone, which separates two things a full warm start confounds:

    is the learned REPRESENTATION still right at the new design, and only its read-out needs moving?
    or does the body itself have to change?

If ``warm_head`` reaches the bar in a window close to a full warm start, the body transferred and the
design only moved the read-out. If it caps while the full warm start converges, the body did not.

MEASUREMENTS, all under the growth procedure, all at one ``(i, i+1)`` pair:

    cold_i        train from scratch at design i                         -> supplies the carried network
    blind_i       that network at design i, NO TRAINING                  -> its own-design reference
    blind_i1      that network at design i+1, NO TRAINING                -> the blind transfer
    warm_head_i1  that network at i+1, BACKBONE FROZEN, head only        -> what this probe is for
    cold_i1       train from scratch at design i+1                       -> the control

⚠️ THE READ-OUT IS THE WINDOW, NOT THE LOSS. Every round runs to the same exit test, so what differs
between them is how many samples that took. A network can score badly at i+1 and still need very few
samples to recover, which is strong transfer, not weak -- the loss alone would call it the opposite.
A round that CAPS has no window, only a floor, and is reported as ``converged: false`` so it is never
read as a measurement.

FREEZING IS DONE BY THE OPTIMISER, not by rebuilding the model: the head keeps the configured
transform and every other leaf is routed to ``optax.set_to_zero``, so the growth procedure, the exit
test and the rewind are the ones the campaign uses. Both regressors end in a layer named ``output``.

The selector is an ``optax.multi_transform`` LABEL TREE -- one label per parameter leaf, ``head`` or
``frozen``. It is NOT a mask: in this repository a mask is the per-hit element mask, and the two must
not be confused.
"""

import argparse
import json
import os
import sys

import jax
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detopt.detector  # noqa: E402
from detopt.nn.trainer.design import DesignTrainer  # noqa: E402
from probe_strip_transfer import REPRESENTATION, evaluate, train_design  # noqa: E402

HEAD = "output"


def head_only(optimizer, params):
  """``optimizer`` on the head, ``set_to_zero`` everywhere else. Returns ``(transform, trainable, frozen)``.

    The selector is a multi_transform LABEL TREE (one label per parameter leaf), not a hit mask."""

  def label(path, _leaf):
    root = getattr(path[0], 'key', None) if len(path) > 0 else None
    return 'head' if str(root) == HEAD else 'frozen'

  labels = jax.tree_util.tree_map_with_path(label, params)
  counts = {'head': 0, 'frozen': 0}
  for value in jax.tree.leaves(labels, is_leaf=lambda x: isinstance(x, str)):
    counts[value] = counts.get(value, 0) + 1
  transform = optax.multi_transform({'head': optimizer, 'frozen': optax.set_to_zero()}, labels)
  return transform, counts['head'], counts['frozen']


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, help="a completed cell; supplies the design trajectory only")
  parser.add_argument("--pair", type=int, required=True, help="the source iteration i; the pair is (i, i+1)")
  parser.add_argument("--representation", choices=sorted(REPRESENTATION), default="strip")
  parser.add_argument("--reveal", choices=("none", "design", "zeros"), default="none")
  parser.add_argument("--n-events", type=int, default=128 * 1024)
  parser.add_argument("--batch", type=int, default=4096)
  parser.add_argument("--event-offset", type=int, default=3_000_000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default=None)
  parser.add_argument("--n0", type=int, default=None)
  parser.add_argument("--n-increment", type=int, default=None, help="override training.n_increment")
  parser.add_argument("--iteration-limit", type=int, default=None)
  parser.add_argument("--loss-precision", type=float, default=None)
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  with open(os.path.join(arguments.run, "results.json")) as handle:
    payload = json.load(handle)
  results = payload["results"]
  i = int(arguments.pair)
  if i + 1 >= len(results):
    sys.exit(f"{arguments.run}: pair ({i}, {i + 1}) needs {i + 2} rows, the run has {len(results)}")

  name, regressor = REPRESENTATION[arguments.representation]
  block = list(payload["config"]["detector"].values())[0]
  config = {
    **payload["config"], "detector": {name: block}, "regressor": regressor,
    "training": {**payload["config"]["training"], "reveal": arguments.reveal},
  }
  if arguments.device is not None:
    config["device"] = arguments.device
  for key, value in (("n0", arguments.n0), ("n_increment", arguments.n_increment),
                     ("iteration_limit", arguments.iteration_limit),
                     ("loss_precision", arguments.loss_precision)):
    if value is not None:
      print(f"[config] training.{key} {config['training'][key]} -> {value}", flush=True)
      config["training"] = {**config["training"], key: value}

  detector = detopt.detector.from_config(config["detector"])
  print(f"[repr] {arguments.representation} -> {name}, reveal={arguments.reveal}, "
        f"features {detector.combined_event_shape(arguments.reveal != 'none')}", flush=True)

  design_i = np.asarray(results[i]["x_scaled"], np.float32)
  design_j = np.asarray(results[i + 1]["x_scaled"], np.float32)
  indices = np.arange(arguments.event_offset, arguments.event_offset + arguments.n_events, dtype=np.int32)
  os.makedirs(arguments.output, exist_ok=True)
  report = {
    "run": arguments.run, "pair": [i, i + 1], "representation": arguments.representation,
    "reveal": arguments.reveal, "warm_mode": "head_only", "n_events": int(arguments.n_events),
    "design_distance": float(np.linalg.norm(design_j - design_i)),
    "reported_at_i": float(results[i]["loss"]), "reported_at_i1": float(results[i + 1]["loss"]),
  }
  target = os.path.join(arguments.output, "head_transfer.json")

  def flush():
    with open(target, "w") as handle:
      json.dump(report, handle, indent=1)

  trainer = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
  report["cold_i"], params_i = train_design(trainer, design_i, i, arguments.seed, None)
  print(f"[cold_i]       {report['cold_i']['objective']:.5f}  window {report['cold_i']['window']}  "
        f"converged {report['cold_i']['converged']}", flush=True)
  flush()

  if params_i is None:
    print("[abort] cold_i CAPPED: there is no carried network, so no transfer can be measured", flush=True)
    report["aborted"] = "cold_i capped"
    flush()
    return

  report["blind_i"] = evaluate(trainer, detector, params_i, design_i, indices, arguments.batch)
  report["blind_i1"] = evaluate(trainer, detector, params_i, design_j, indices, arguments.batch)
  print(f"[blind]        on i {report['blind_i']:.5f}   on i+1 {report['blind_i1']:.5f}", flush=True)
  flush()

  warm = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
  warm.optimizer, trainable, frozen = head_only(warm.optimizer, params_i)
  report["head_leaves"], report["frozen_leaves"] = int(trainable), int(frozen)
  print(f"[freeze]       head leaves {trainable}, frozen leaves {frozen}", flush=True)
  if trainable == 0 or frozen == 0:
    sys.exit(f"the label tree selected {trainable} trainable and {frozen} frozen leaves; "
             f"'{HEAD}' did not name this model's last layer")
  report["warm_head_i1"], _ = train_design(warm, design_j, i + 1, arguments.seed + 1, params_i)
  print(f"[warm_head_i1] {report['warm_head_i1']['objective']:.5f}  window {report['warm_head_i1']['window']}  "
        f"converged {report['warm_head_i1']['converged']}", flush=True)
  flush()

  cold = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed)
  report["cold_i1"], _ = train_design(cold, design_j, i + 1, arguments.seed + 1, None)
  print(f"[cold_i1]      {report['cold_i1']['objective']:.5f}  window {report['cold_i1']['window']}  "
        f"converged {report['cold_i1']['converged']}", flush=True)
  flush()

  if report["warm_head_i1"]["converged"] and report["cold_i1"]["converged"]:
    ratio = report["warm_head_i1"]["window"] / max(report["cold_i1"]["window"], 1)
    report["window_ratio"] = float(ratio)
    print(f"[ratio]        head-only window / cold window = {ratio:.3f}", flush=True)
  else:
    print("[ratio]        NOT REPORTED: a capped round has a floor, not a window", flush=True)
  flush()
  print(f"wrote {target}", flush=True)


if __name__ == "__main__":
  main()
