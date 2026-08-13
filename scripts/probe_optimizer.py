#!/usr/bin/env python3
"""Is the OPTIMISER (weight decay / lr / lr-schedule) a lever on the train-validation GAP?

    python scripts/probe_optimizer.py =enzyme_inhib \
        --results output/campaign-inhib/126382657/from_scratch/results.json \
        --rank 0 --setting baseline wd=1.0e-2 'sched=cosine,decay_epochs=400,alpha=0.05' \
        --output output/screen/optim-wd.json

WHAT IS BEING MEASURED, and how it differs from `scripts/benchmark_lr.py`. That earlier study put
three schedules (constant / SGDR seesaw / smooth cyclical) on a DIFFERENT task (the debug detector)
and compared DATA-TO-CONVERGENCE and EPOCHS; it found them identical and concluded "stopping is
data-bound, not optimization-bound". It never separated the two terms of the stopping rule. The
trainer stops when ``diff + err < loss_precision`` with ``err`` the standard error (falls like
1/sqrt(window): data fixes it) and ``diff = |train - val|`` an OVERFITTING BIAS that does not fall
with data on an uninformative design. This probe targets ``diff`` specifically, on the real
``enzyme_inhib`` task, at the designs a real BO run actually visited.

THE CENSORING PROBLEM, and what is therefore reported. The loop EXITS the moment ``diff + err``
first dips below ``loss_precision``, so the ``diff`` of a converged run is truncated by construction
-- every converged setting has ``diff + err < precision`` and comparing those numbers directly says
almost nothing. The uncensored readout is the WINDOW/SPEND at which that dip happened: a setting
that genuinely shrinks the gap converges on a much smaller window (dropout 0.2 exits at spend 8191
against the baseline's 46411). So the primary comparison is (converged LEVEL, SPEND), and
``--loss-precision`` re-runs the winner at a tightened precision to read off the achievable
``diff + err`` FLOOR at the full ``iteration_limit`` -- the number that says what precision the
benchmark could actually be run at.

THE UNDER-FITTING TRAP. Any setting that simply trains less (a decayed lr, a heavy decay) shrinks the
gap for free, and the plateau test rewards it: a frozen network is trivially "flat". That is why
``--rank`` takes a LIST. Rank 0 is the WORST (most uninformative) design a real run visited, where
the gap bites; a high rank is a GOOD low-loss design, where under-fitting shows up immediately as a
raised converged level. A setting is only a win if it cuts spend at rank 0 WITHOUT raising the level
at the good design.

SETTING SPEC: comma-separated ``key=value``, keys ``lr``, ``wd``, ``b1``, ``b2``, ``sched``
(``constant``|``cosine``), ``decay_epochs``, ``alpha``, ``warmup_epochs`` (schedule warmup, in
epochs, NOT the trainer's data warmup), ``precision``. ``baseline`` means the config as-is.
"""

import argparse
import gc
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("AGG")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

import detopt
import detopt.detector
from detopt.nn.trainer import DesignTrainer
from detopt.utils.config import resolve_device, split
from detopt.utils.training import masked_mean_sem


class _Tee:
  """stdout pass-through that also keeps the lines, so the trainer's own ``[converged]`` line (the
  only place ``diff`` and ``err`` are reported SEPARATELY) can be read back out."""

  def __init__(self, stream):
    self.stream = stream
    self.lines = []

  def write(self, text):
    self.stream.write(text)
    self.lines.extend(text.splitlines())

  def flush(self):
    self.stream.flush()


def parse_setting(text):
  """``"lr=1e-4,wd=1.0e-2"`` -> ``{"lr": 1e-4, "wd": 1.0e-2}``; ``"baseline"`` -> ``{}``."""
  if text == "baseline":
    return {}
  keys = {"lr", "wd", "b1", "b2", "sched", "decay_epochs", "alpha", "warmup_epochs", "precision"}
  out = {}
  for item in text.split(","):
    key, separator, value = item.partition("=")
    if separator != "=":
      raise ValueError(f"expected key=value, got {item!r}")
    if key not in keys:
      raise ValueError(f"unknown setting key {key!r}; known: {sorted(keys)}")
    out[key] = value if key == "sched" else float(value)
  return out


def build_optimizer(base_config, setting, steps_per_epoch):
  """The run's optimiser with this setting's overrides applied. Returns ``(transformation, label)``.

  A cosine schedule needs a horizon, and the per-design trainer has no total step count (the window
  grows until convergence), so the horizon is stated in EPOCHS -- ``decay_epochs * steps_per_epoch``
  steps -- and floored at ``alpha * lr`` so a design that outlives the horizon keeps learning
  instead of freezing (a frozen network passes the plateau test trivially, which would be a fake
  convergence at a bad level).
  """
  name, arguments = split(base_config)
  arguments = dict(arguments)
  if "lr" in setting:
    arguments["learning_rate"] = setting["lr"]
  if "wd" in setting:
    arguments["weight_decay"] = setting["wd"]
  if "b1" in setting:
    arguments["b1"] = setting["b1"]
  if "b2" in setting:
    arguments["b2"] = setting["b2"]

  peak = float(arguments["learning_rate"])
  schedule_name = setting.get("sched", "constant")
  if schedule_name == "cosine":
    decay_steps = int(round(setting.get("decay_epochs", 400.0) * steps_per_epoch))
    alpha = float(setting.get("alpha", 0.05))
    schedule = optax.cosine_decay_schedule(init_value=peak, decay_steps=decay_steps, alpha=alpha)
    warmup = int(round(setting.get("warmup_epochs", 0.0) * steps_per_epoch))
    if warmup > 0:
      schedule = optax.join_schedules(
        [optax.linear_schedule(init_value=peak / 100.0, end_value=peak, transition_steps=warmup), schedule],
        boundaries=[warmup],
      )
    arguments["learning_rate"] = schedule
  elif schedule_name != "constant":
    raise ValueError(f"unknown schedule {schedule_name!r}")

  label = ",".join(f"{k}={v}" for k, v in sorted(setting.items())) if len(setting) > 0 else "baseline"
  return getattr(optax, name)(**arguments), label


def parse_converged(lines):
  """``diff`` / ``err`` / ``window`` off the trainer's last ``[converged]`` line."""
  for line in reversed(lines):
    if "[converged]" in line:
      fields = {}
      for token in line.replace("|", " ").split():
        key, separator, value = token.partition("=")
        if separator == "=":
          fields[key] = value
      return {
        "train": float(fields["train"]),
        "val": float(fields["val"]),
        "diff": float(fields["diff"]),
        "err": float(fields["err"]),
        "window": int(fields["window"].split("/")[0]),
      }
  return {}


def parse_failure(message):
  """``diff`` / ``err`` / ``window`` off the ``did not reach precision`` RuntimeError."""
  text = message.replace("\n", " ").replace("(", " ").replace(")", " ").replace(";", " ")
  fields = {}
  for token in text.split():
    key, separator, value = token.partition("=")
    if separator == "=":
      fields[key] = value
  out = {}
  for key, cast in (("diff", float), ("err", float), ("window", int)):
    if key in fields:
      try:
        out[key] = cast(fields[key])
      except ValueError:
        pass
  return out


def run_fixed_window(trainer, x_scaled, seed, window, epochs):
  """Train ONE fixed window for a FIXED number of epochs and return the per-epoch loss curves.

  The convergence protocol CENSORS the gap (it exits the instant ``diff + err`` dips under
  ``loss_precision``), so converged ``diff`` values cannot be compared across settings. This mode
  removes the censoring by fixing the data and the epoch count: whatever ``diff`` a setting reaches
  is the setting's own gap, not the stopping rule's threshold. An epoch is the SAME 512 optimiser
  steps the campaign uses (``iteration_limit // batch``), so the optimisation dynamics per epoch are
  the campaign's; only the growing window and the early exit are removed.
  """
  detector = trainer.detector
  design = detector.to_nominal(np.asarray(x_scaled, dtype=np.float32))
  init_seq, training_seq = np.random.SeedSequence(seed).spawn(2)
  params, state, opt_state = trainer._init_design_network(init_seq, None)

  tp, vp = trainer.train_pool, trainer.val_pool
  w0_train, w0_val = tp.current, vp.current
  if trainer._sample_round(design, w0_train, w0_val, window) is None:
    raise RuntimeError("budget pool too small for the requested fixed window")
  train_count, val_count = tp.current - w0_train, vp.current - w0_val
  w0_train_j, w0_val_j = jnp.int32(w0_train), jnp.int32(w0_val)

  key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))
  curves = {"train": [], "val": [], "train_sem": [], "val_sem": []}
  for _ in range(epochs):
    key, subkey = jax.random.split(key)
    params, state, opt_state, _ = trainer._train_epoch(
      params, state, opt_state, subkey, w0_train_j, jnp.int32(train_count), tp.buffers()
    )
    train_mean, train_sem = masked_mean_sem(trainer._eval_train(params, state, tp.buffers(), w0_train_j), train_count)
    val_mean, val_sem = masked_mean_sem(trainer._eval_val(params, state, vp.buffers(), w0_val_j), val_count)
    curves["train"].append(float(train_mean))
    curves["val"].append(float(val_mean))
    curves["train_sem"].append(float(train_sem))
    curves["val_sem"].append(float(val_sem))
  return curves, train_count, val_count


def summarise_fixed_window(curves, tail=25):
  """Headline numbers from a fixed-window curve. The TAIL means are what to compare: a single final
  epoch is one noisy draw, and the quantity of interest is where the pair of curves settled."""
  train = np.asarray(curves["train"], dtype=np.float64)
  val = np.asarray(curves["val"], dtype=np.float64)
  train_sem = np.asarray(curves["train_sem"], dtype=np.float64)
  val_sem = np.asarray(curves["val_sem"], dtype=np.float64)
  tail = min(tail, train.shape[0])
  err = float(np.hypot(train_sem[-tail:], val_sem[-tail:]).mean())
  diff = float(np.abs(val[-tail:] - train[-tail:]).mean())
  return {
    "train": float(train[-tail:].mean()),
    "val": float(val[-tail:].mean()),
    "level": float(0.5 * (train[-tail:] + val[-tail:]).mean()),
    "diff": diff,
    "err": err,
    "slack": diff + err,
    "val_min": float(val.min()),
    "val_min_epoch": int(val.argmin()),
    "diff_final": float(abs(val[-1] - train[-1])),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token, e.g. =enzyme_inhib")
  parser.add_argument("--results", required=True, help="a finished results.json to take designs from")
  parser.add_argument(
    "--rank", type=int, nargs="+", default=[0],
    help="order statistics of the recorded loss; 0 = the WORST (most uninformative) design"
  )
  parser.add_argument("--setting", nargs="+", default=["baseline"], help="optimiser specs, see the module docstring")
  parser.add_argument("--seed", type=int, nargs="+", default=[7])
  parser.add_argument(
    "--fixed-window", type=int, default=None, help="SCREEN mode: hold the window at this many train events and train for "
    "`--epochs` epochs, reporting the UNCENSORED gap instead of running the "
    "convergence protocol (which truncates `diff` at `loss_precision`)."
  )
  parser.add_argument("--epochs", type=int, default=150, help="fixed-window mode: epochs to train")
  parser.add_argument("--device", default=None, help="override the config's device (e.g. `cpu`)")
  parser.add_argument("--output", default="output/screen/optim-probe.json")
  arguments = parser.parse_args()

  name = arguments.config.lstrip("=")
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)

  with open(arguments.results) as f:
    recorded = json.load(f)["results"]
  order = sorted(recorded, key=lambda r: -r["loss"])  # worst first
  print(f"{len(recorded)} recorded designs; worst {order[0]['loss']:.4f}, best {order[-1]['loss']:.4f}")

  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  detector = detopt.detector.from_config(detector_config)

  training_config = config["training"]
  steps_per_epoch = max(1, int(training_config["iteration_limit"]) // int(training_config["batch"]))
  device = resolve_device(arguments.device if arguments.device is not None else config.get("device"))

  def run_one(x_scaled, setting, seed):
    """One (design, setting, seed) measurement. The trainer -- which owns budget-sized event pools --
    is local, so each measurement's pools are released before the next allocates its own."""
    optimizer, label = build_optimizer(training_config["optimizer"], setting, steps_per_epoch)
    training = {k: v for k, v in training_config.items() if k != "optimizer"}
    if "precision" in setting:
      training["loss_precision"] = setting["precision"]
    trainer = DesignTrainer(
      detector, regressor_config=config["regressor"], optimizer=optimizer, device=device, checkpoint_dir=None, seed=seed,
      **training
    )
    started = time.time()
    if arguments.fixed_window is not None:
      curves, train_count, val_count = run_fixed_window(trainer, x_scaled, seed, arguments.fixed_window, arguments.epochs)
      row = {
        "status": "fixed window",
        "spent": train_count + val_count,
        "window": train_count,
        "epochs": arguments.epochs,
        "curves": curves
      }
      row.update(summarise_fixed_window(curves))
    else:
      tee = _Tee(sys.stdout)
      try:
        sys.stdout = tee
        result = trainer.train(x_scaled, np.random.SeedSequence(seed), step=0)
      except RuntimeError as error:
        sys.stdout = tee.stream
        row = {"status": "did not converge", "loss": float("nan"), "slack": float("nan"), "spent": -1}
        row.update(parse_failure(str(error)))
        if "diff" in row and "err" in row:
          row["slack"] = row["diff"] + row["err"]
      else:
        sys.stdout = tee.stream
        if result is None:
          row = {"status": "budget exhausted", "loss": float("nan"), "slack": float("nan"), "spent": -1}
        else:
          row = {
            "status": "converged",
            "loss": float(result.objective_loss),
            "slack": float(result.objective_std),
            "spent": int(result.spent)
          }
          row.update(parse_converged(tee.lines))
      finally:
        sys.stdout = tee.stream
    row.update({
      "setting": label,
      "seed": seed,
      "loss_precision": training["loss_precision"],
      "time_s": round(time.time() - started, 1)
    })
    return row

  rows = []
  for rank in arguments.rank:
    chosen = order[min(rank, len(order) - 1)]
    x_scaled = np.asarray(chosen["x_scaled"], dtype=np.float32)
    print(
      f"\n=== rank {rank}: recorded loss {chosen['loss']:.4f} (std {chosen['loss_std']:.4f}, "
      f"spent {chosen['spent']}) ==="
    )
    for text in arguments.setting:
      setting = parse_setting(text)
      for seed in arguments.seed:
        row = run_one(x_scaled, setting, seed)
        gc.collect()
        row.update({"rank": rank, "recorded_loss": chosen["loss"]})
        rows.append(row)
        if arguments.fixed_window is not None:
          print(
            f"  [rank {rank}] {row['setting']} seed {seed}: window {row['window']} x {row['epochs']} ep -> "
            f"train {row['train']:.4f} val {row['val']:.4f} level {row['level']:.4f} "
            f"diff {row['diff']:.4f} err {row['err']:.4f} slack {row['slack']:.4g} "
            f"val_min {row['val_min']:.4f}@{row['val_min_epoch']} ({row['time_s']:.0f} s)"
          )
        else:
          print(
            f"  [rank {rank}] {row['setting']} seed {seed}: {row['status']} level {row['loss']:.4f} "
            f"diff {row.get('diff', float('nan')):.4f} err {row.get('err', float('nan')):.4f} "
            f"slack {row['slack']:.4g} spent {row['spent']} ({row['time_s']:.0f} s)"
          )
        with open(arguments.output, "w") as f:
          json.dump({"results": arguments.results, "rows": rows}, f, indent=2, default=float)

  print(f"\nwrote {arguments.output}")
  print(
    "READ IT AS: a converged `diff` is CENSORED by the stopping rule -- compare SPEND at rank 0 "
    "(does the gap resist?) against LEVEL at the good rank (did it just under-fit?)."
  )


if __name__ == "__main__":
  main()
