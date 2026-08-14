#!/usr/bin/env python3
"""Does a LESS EXPRESSIVE ACTIVATION close the train/validation gap that makes uninformative designs
unconvergeable?

A FROZEN SIBLING of scripts/probe_dropout.py -- same measurement protocol, same convergence criterion,
same design selection -- kept separate so the activation study's numbers cannot be changed underneath it
by edits to the shared probe. The knob it sweeps is `--activation` (see below); `--dropout` / `--n-models`
still work and reproduce the dropout/ensemble rows of that probe exactly.

    python scripts/probe_dropout.py =enzyme_inhib --results output/campaign-inhib/126382657/from_scratch/results.json \
        --dropout 0.1 0.2 0.3 --output output/screen/dropout-probe.json

THE PROBLEM THIS MEASURES. `detopt/nn/trainer` calls a design converged when `diff + err` falls under
`loss_precision`, where `err` is the loss estimate's standard error and `diff` is the train/validation
gap. `err` falls like `1/sqrt(window)`, so more data always fixes it. `diff` does NOT: on an
uninformative design the network fits read-out noise, and the gap is an OVERFITTING BIAS that sits
where it is however much data arrives. Two campaign runs died exactly there --

    diff=0.0053   err=0.0022  ->  0.00748  vs precision 0.006
    diff=0.008349 err=0.0028  ->  0.0112   vs precision 0.008

-- and because bo.py is deterministic given a seed, a resubmission reproduces the same design and the
same crash. Raising `loss_precision` past the worst gap is not a fix either: clearing 0.0083 needs
~0.012, which puts criterion (d)'s bar (`10 x loss_precision`) at 0.120 against an achievable span of
0.196, i.e. 61% of the whole range. The benchmark would then be limited by the REGRESSOR's overfitting
rather than by the search.

WHAT IS COMPARED, and why it is not just "does dropout reduce the gap" (it must, trivially). Dropout
trades variance for bias: it shrinks `diff` but RAISES the converged loss level, which compresses the
baseline-to-best span the criterion is measured against. So the quantity that decides the trade is the
RATIO

    gap / span      (both in the same units as the loss)

measured on the SAME design at each setting. A dropout level that halves the gap while costing a tenth
of the span is a win; one that shrinks both equally changes nothing. Both halves are reported, plus
the per-design spend, since a heavier network that converges sooner is also cheaper.

THE DESIGN. Not hand-picked: `--results` reads a finished BO run and takes the design with the WORST
recorded loss, i.e. the most uninformative one the search actually visited -- the regime where the gap
bites. `--rank` selects a different order statistic (0 = worst) to check the effect is not one design's
accident.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("AGG")

import numpy as np

import detopt
import detopt.detector
from detopt.nn.trainer import DesignTrainer


def _per_window(history, tail_epochs=8):
  """Collapse the per-epoch history into one row per WINDOW SIZE (i.e. per data-addition round).

  The trainer returns on the first epoch whose ``diff + err`` dips under the precision, so the
  reported ``diff`` is a stopped -- hence downward-biased -- sample of the gap. Per round this
  reports the gap at the LAST epoch (what the criterion saw), its MAX over the round, and its mean
  over the round's last ``tail`` epochs -- the settled part, after the mandatory warmup, where the
  network has finished fitting the new data. That tail mean is the honest estimate of the bias
  FLOOR, and the floor is what decides the ``loss_precision`` the design can be run at. The full
  per-epoch arrays are returned alongside so any other summary can be recomputed offline.
  """
  if len(history) == 0:
    return []
  train = np.asarray(history["train_loss_per_epoch"], dtype=np.float64)
  val = np.asarray(history["val_loss_per_epoch"], dtype=np.float64)
  train_sem = np.asarray(history["train_sem_per_epoch"], dtype=np.float64)
  val_sem = np.asarray(history["val_sem_per_epoch"], dtype=np.float64)
  window = np.asarray(history["train_budget_per_epoch"], dtype=np.int64)
  diff = np.abs(val - train)
  err = np.hypot(train_sem, val_sem)

  rows = []
  for size in sorted(set(window.tolist())):
    sel = window == size
    round_diff = diff[sel]
    tail = round_diff[-min(tail_epochs, round_diff.shape[0]):]
    rows.append({
      "window": int(size), "n_epochs": int(sel.sum()),
      "train": float(train[sel][-1]), "val": float(val[sel][-1]),
      "diff": float(diff[sel][-1]), "err": float(err[sel][-1]),
      "diff_tail_mean": float(tail.mean()), "diff_max": float(round_diff.max()),
    })
  return rows


def _epoch_history(history):
  """The raw per-epoch arrays, as lists, so the JSON keeps everything the summary throws away."""
  if len(history) == 0:
    return {}
  return {
    key: np.asarray(history[key]).tolist() for key in (
      "train_loss_per_epoch", "val_loss_per_epoch", "train_sem_per_epoch", "val_sem_per_epoch",
      "train_budget_per_epoch"
    )
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token, e.g. =enzyme_inhib")
  parser.add_argument("--results", required=True, help="a finished results.json to take the design from")
  parser.add_argument("--rank", type=int, default=0, help="0 = the WORST (most uninformative) design")
  parser.add_argument("--dropout", type=float, nargs="+", default=[0.1, 0.2, 0.3])
  parser.add_argument("--n-models", type=int, nargs="+", default=None,
                      help="sweep the ENSEMBLE size instead of dropout, at the config's own p_dropout. "
                           "Averaging independent members cuts the variance that drives the gap WITHOUT "
                           "the bias dropout charges, so it is the lever to try when dropout costs more "
                           "level than it buys gap.")
  parser.add_argument("--activation", type=str, nargs="+", default=None,
                      help="sweep the per-unit NONLINEARITY instead, at the config's own dropout/ensemble. "
                           "`leaky-tanh` is the current one and is LEARNABLE (two gains per unit = capacity); "
                           "`fixed-leaky-tanh` is the same shape frozen at its initialisation (tanh(x)+x, no "
                           "parameters) and so isolates the learnable capacity from the shape; `tanh` is the "
                           "bounded, least expressive option; `relu`/`gelu` are unbounded and fixed.")
  parser.add_argument("--precision", type=float, default=None,
                      help="override `training.loss_precision`. ONLY for asking 'does this design still "
                           "so a run at a different precision is NOT comparable to the table.")
  parser.add_argument("--device", default=None,
                      help="override `device`. Use `cpu` (together with JAX_PLATFORMS=cpu) to run this "
                           "probe as a CPU job alongside the GPU jobs; numbers from the two backends are "
                           "internally consistent but should not be mixed in one table.")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--checkpoint-dir", default="output/probe-activation",
                      help="per-setting checkpoints go under here; give CONCURRENT probes different "
                           "roots or two runs of the same setting write the same orbax directory")
  parser.add_argument("--output", default="output/screen/activation-probe.json")
  arguments = parser.parse_args()

  import yaml  # noqa: E402 -- local import keeps the module importable without a config
  name = arguments.config.lstrip("=")
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)

  with open(arguments.results) as f:
    recorded = json.load(f)["results"]
  order = sorted(recorded, key=lambda r: -r["loss"])  # worst first
  chosen = order[min(arguments.rank, len(order) - 1)]
  x_scaled = np.asarray(chosen["x_scaled"], dtype=np.float32)
  print(f"design from {arguments.results} rank {arguments.rank}: recorded loss {chosen['loss']:.4f}"
        f" (of {len(recorded)} designs; worst {order[0]['loss']:.4f}, best {order[-1]['loss']:.4f})")

  # A RUN config names its detector by string (`detector: enzyme_inhib`) and gearup resolves that
  # against config/detector/<name>.yaml; doing the same here keeps the probe on exactly the detector
  # the campaign ran, rather than a second copy of its parameters that could drift.
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  detector = detopt.detector.from_config(detector_config)
  # Sweep ONE knob at a time: the activation or the ensemble size if asked for, otherwise dropout.
  if arguments.activation is not None and arguments.n_models is not None:
    parser.error("sweep ONE knob: --activation or --n-models or --dropout")
  if arguments.activation is not None:
    knob, settings, cast = "activation", arguments.activation, str
  elif arguments.n_models is not None:
    knob, settings, cast = "n_models", arguments.n_models, int
  else:
    knob, settings, cast = "p_dropout", arguments.dropout, float

  rows = []
  for setting in settings:
    # The regressor block is `{<name>: {...hyper-parameters...}}`; set the knob of whichever it is.
    run_config = json.loads(json.dumps(config))  # deep copy, config is plain JSON-able yaml
    (regressor_name, ), = (run_config["regressor"].keys(), )
    run_config["regressor"][regressor_name][knob] = cast(setting)
    if arguments.precision is not None:
      run_config["training"]["loss_precision"] = float(arguments.precision)
    if arguments.device is not None:
      run_config["device"] = arguments.device

    trainer = DesignTrainer.from_config(
      detector, run_config, checkpoint_dir=os.path.join(arguments.checkpoint_dir, f"{knob}{setting}"),
      seed=arguments.seed
    )
    # The per-epoch history: `diff` at CONVERGENCE is a stopped sample (the loop returns on the first
    # epoch that dips under the precision), so it understates the gap. The history carries `diff` at
    # every window, which is what says whether the gap is a bias FLOOR or still falling with data.
    history = {}
    sequence = np.random.SeedSequence(arguments.seed)
    try:
      result = trainer.train(x_scaled, int(sequence.spawn(1)[0].generate_state(1)[0]), step=0, on_epoch=history.update)
      # `objective_std` IS the convergence slack `diff + err` (detopt/nn/trainer/design.py: the
      # objective is set to `(0.5 * (train + val), diff + err)`), so it is directly the quantity that
      # decides whether a design can converge at a given `loss_precision`.
      status = "converged"
      loss, slack, spent = float(result.objective_loss), float(result.objective_std), int(result.spent)
    except RuntimeError as error:
      # Did not converge: the message carries the slack it got stuck at, which is the number wanted.
      text = str(error).replace("\n", " ")
      status = "did not converge"
      loss, spent = float("nan"), -1
      slack = float(text.split("diff+err=")[1].split()[0]) if "diff+err=" in text else float("nan")
      print(f"  {knob} {setting}: {status}, stuck at slack {slack:.4g}")
    per_window = _per_window(history)
    row = {knob: setting, "status": status, "loss": loss, "slack": slack, "spent": spent,
           "per_window": per_window, "epochs": _epoch_history(history)}
    if len(per_window) > 0:
      last = per_window[-1]
      row["diff"], row["err"], row["window"] = last["diff"], last["err"], last["window"]
      # The gap FLOOR at the largest window reached: NOT the lucky epoch the stopping rule picked,
      # but the settled tail of that round. It is what a lower `loss_precision` would have to clear.
      row["diff_last_round_max"] = last["diff_max"]
      row["diff_last_round_tail_mean"] = last["diff_tail_mean"]
    rows.append(row)
    if status == "converged":
      print(f"  {knob} {setting}: converged  level {loss:.4f}  slack {slack:.4g}  spent {spent}")
    for w in per_window[-4:]:
      print(f"      window {w['window']:>7d}  diff {w['diff']:.4f} (round tail mean {w['diff_tail_mean']:.4f}, "
            f"max {w['diff_max']:.4f})  err {w['err']:.4f}  train {w['train']:.4f} val {w['val']:.4f}")

  with open(arguments.output, "w") as f:
    json.dump({"design": chosen, "results": arguments.results, "rows": rows}, f, indent=2, default=float)
  print(f"\nwrote {arguments.output}")
  print("READ IT AS: a higher dropout is worth it only if the converged LEVEL rises by less than the")
  print("gap falls -- the level sets the span the criterion is measured against.")


if __name__ == "__main__":
  main()
