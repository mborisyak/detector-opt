"""Gradient variance of the structured (current:replay) minibatch against a uniform draw, at one design of a
`meta` trajectory, under the verification training protocol.

    python scripts/probe_gradient_variance.py =angle strategy=angle-meta-rewind-01 \
        trajectory=output/angle/test/1244111331/meta/results.json seed=1244111331 design_index=2 \
        output=output/ablation-gradvar/angle/1244111331/design-2.json [draws=16] [epochs=128] [warmup=16]

THE CONSTRUCTION (user, 2026-09-08: "take some meta checkpoint, sample context, sample whole recorded budget,
train (using verify protocol), record gradient variance"):

  1. The run's own state at design ``design_index`` (0-based; the user's "3rd, 5th, 7th" are 2, 4, 6):
     `trainer.replay(rows[:k])` re-simulates the committed designs 0..k-1 into the pools (the context) and
     reads the carried network from the checkpoint written at design k-1 -- exactly what
     `scripts/probe_retention.py` does.
  2. The current design's data is sampled ONCE, at the size the trajectory recorded for it
     (``spent_train`` / ``spent_val`` of row k). No growth procedure, no exit test.
  3. From that one checkpoint, TWO trainings of the same design, each under the verification protocol
     (`scripts/verify_trajectory.py`: the run's optimiser with a single-cycle cosine over ``verify.epochs``
     (128) epochs, which is also the cap; ``verify.warmup_epochs`` (16) epochs unconditional; then the
     settle test ``P(train change over +patience < loss_precision / 2) > 0.9``, plus ``patience`` epochs
     of overrun): one with the campaign's STRUCTURED batch (half current design, half replay, replay rows
     weighted ``replay_weight``), one with every row a UNIFORM draw from the whole pool at unit weight
     (`ContinualUniformTrainer`'s draw). Same pools, same network, same kernels but for the draw.
  4. At every epoch of BOTH trainings, at the parameters of that moment, the minibatch gradient is drawn
     ``draws`` times under EACH composition (one dropout key per measurement, shared by the draws, so
     the spread is the data sampling alone) and reduced, per epoch, to: the trace of its covariance
     (``variance``) and the AVERAGE per-parameter variance (``mean_variance``, user: "record average
     variance for each epoch"); the squared norm of its mean (``signal``); the parameter averages of the
     first moment's magnitude and of its standard deviation (``mean_abs_moment``, ``mean_std``) and the
     first moment RELATIVE TO THE STD both as the ratio of those averages (``moment_over_std``) and as the
     average of the per-parameter ratio (``mean_moment_over_std``, user: "record average first momentum
     relative to the std" -- "the main figure for the optimizers"), plus the Adam form of the same
     ratio, |mean| over the root mean square of the draws (``mean_moment_over_rms``); the mean minibatch
     loss; and the cosine between the two compositions' mean gradients (``cosine``). Both compositions are
     measured on both trajectories, so every comparison is at identical parameters.

Output: one JSON with the metadata (design, seed, sizes, the trajectory's recorded loss) and, per arm,
the stop epoch, the final train / val / objective / loss (objective + design penalty, the campaign's
scale) and the per-epoch series. The loss is evaluated on the current design's window alone, as the
campaign does. The checkpoint directory is COPIED, never shared.
"""
import json
import os
import shutil
import types

import jax
import jax.numpy as jnp
import numpy as np
import optax

import detopt.detector
from detopt.nn.trainer import ContinualUniformTrainer
from detopt.utils.config import split
from detopt.utils.training import bayesian_trend, masked_mean_sem, probability_change_below
from bo import TRAINERS, drop_foreign_knobs, _resolve_strategy


def build_grad_stats(loss_fn, sample_indices, draws):
  """``stats(params, state, key, start, count, buffers)`` from ``draws`` minibatches drawn by ``sample_indices`` at
  fixed parameters: the mean minibatch loss; ``signal`` (squared norm of the mean gradient); ``variance`` (trace of
  the gradient covariance) and ``mean_variance`` (its average per parameter); ``mean_abs_moment`` and ``mean_std``
  (parameter averages of the mean gradient's magnitude and of its standard deviation); ``moment_over_std`` (the
  ratio of those two averages) and ``mean_moment_over_std`` (the average over parameters of the per-parameter
  ratio); ``mean_moment_over_rms`` (the Adam form, |mean| over the root mean square); and the mean gradient vector."""

  def one(params, state, key_idx, key_drop, start, count, buffers):
    event_buf, mask_buf, target_buf, design_buf = buffers
    idx = sample_indices(key_idx, start, count)
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
      params, state, key_drop, jax.tree.map(lambda a: a[idx], event_buf), mask_buf[idx],
      jax.tree.map(lambda a: a[idx], design_buf), jax.tree.map(lambda a: a[idx], target_buf)
    )
    return loss, jnp.concatenate([jnp.ravel(g).astype(jnp.float32) for g in jax.tree.leaves(grads)])

  @jax.jit
  def stats(params, state, key, start, count, buffers):
    keys = jax.random.split(key, draws + 1)
    size = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))

    def body(carry, k):
      loss, flat = one(params, state, k, keys[0], start, count, buffers)
      total, squares, losses = carry
      return (total + flat, squares + flat * flat, losses + loss), None

    (total, squares, losses), _ = jax.lax.scan(body, (jnp.zeros(size), jnp.zeros(size), jnp.float32(0.0)), keys[1:])
    mean = total / draws
    per_parameter = jnp.maximum(squares / draws - mean * mean, 0.0) * draws / (draws - 1)
    std = jnp.sqrt(per_parameter)
    return {
      "loss": losses / draws,
      "signal": jnp.sum(mean * mean),
      "variance": jnp.sum(per_parameter),
      "mean_variance": jnp.mean(per_parameter),
      "mean_abs_moment": jnp.mean(jnp.abs(mean)),
      "mean_std": jnp.mean(std),
      "moment_over_std": jnp.mean(jnp.abs(mean)) / jnp.maximum(jnp.mean(std), 1e-30),
      "mean_moment_over_std": jnp.mean(jnp.abs(mean) / jnp.maximum(std, 1e-30)),
      "mean_moment_over_rms": jnp.mean(jnp.abs(mean) / jnp.maximum(jnp.sqrt(squares / draws), 1e-30)),
      "mean": mean,
    }

  return stats


def probe_gradient_variance(
  output, trajectory, seed: int, design_index: int, draws: int = 16, epochs: int | None = None, warmup: int | None = None,
  **config
):
  config = _resolve_strategy(config)
  arm = config["nn_init_strategy"]
  if arm != "meta":
    raise SystemExit(f"probe_gradient_variance: the strategy must be meta, got {arm!r}")
  config = drop_foreign_knobs(config, arm)

  with open(trajectory) as handle:
    rows = json.load(handle)["results"]
  index = int(design_index)
  if not 0 < index < len(rows):
    raise SystemExit(f"probe_gradient_variance: design index {index} outside 1..{len(rows) - 1}")
  if int(draws) < 2:
    raise SystemExit("probe_gradient_variance: draws must be at least 2")

  source_checkpoints = os.path.join(os.path.dirname(trajectory), "checkpoints")
  if not os.path.isdir(source_checkpoints):
    raise SystemExit(f"probe_gradient_variance: no checkpoints beside {trajectory}")
  work = os.path.splitext(output)[0] + ".checkpoints"
  if os.path.isdir(work):
    shutil.rmtree(work)
  shutil.copytree(source_checkpoints, work)

  detector = detopt.detector.from_config(config["detector"])
  network_seq, iteration_seq = np.random.SeedSequence(int(seed)).spawn(2)
  trainer = TRAINERS[arm].from_config(detector, config, checkpoint_dir=work, seed=int(network_seq.generate_state(1)[0]))
  trainer.replay(rows[:index])
  iteration_seq.spawn(index)
  probe_seq = iteration_seq.spawn(1)[0]

  design_scaled = np.asarray(rows[index]["x_scaled"], dtype=np.float32)
  design = detector.to_nominal(design_scaled)
  tp, vp = trainer.train_pool, trainer.val_pool
  w0_train, w0_val = tp.current, vp.current
  if trainer._sample_round(design, w0_train, w0_val, int(rows[index]["spent_train"])) is None:
    raise SystemExit("probe_gradient_variance: the pools cannot hold the recorded budget of the design")
  train_count, val_count = tp.current - w0_train, vp.current - w0_val
  w0_train_j, w0_val_j = jnp.int32(w0_train), jnp.int32(w0_val)
  count_j = jnp.int32(train_count)
  print(
    f"[probe] design {index}/{len(rows)} of {trajectory}: context {w0_train}+{w0_val} rows, "
    f"design {train_count}+{val_count} rows (recorded {rows[index]['spent_train']}+{rows[index]['spent_val']})", flush=True,
  )

  training = config["training"]
  verify_cfg = config.get("verify", {})
  max_epochs = int(verify_cfg.get("epochs", 128)) if epochs is None else int(epochs)
  warmup_epochs = int(verify_cfg.get("warmup_epochs", 16)) if warmup is None else int(warmup)
  patience = int(training["patience"])
  loss_precision = float(training["loss_precision"])
  overrun_epochs = patience if bool(verify_cfg.get("overrun", True)) else 0
  opt_name, opt_args = split(training["optimizer"])
  opt_args = dict(opt_args)
  schedule = optax.cosine_decay_schedule(
    init_value=opt_args.pop("learning_rate"), decay_steps=max_epochs * trainer.steps_per_epoch
  )
  trainer.optimizer = getattr(optax, opt_name)(learning_rate=schedule, **opt_args)

  reg_def = trainer._build_regressor(trainer.seed)[0]
  structured_loss = trainer._make_loss_fn(reg_def)
  structured_epoch = trainer._build_train_epoch(reg_def)
  structured_indices = trainer._sample_indices
  trainer._sample_indices = types.MethodType(ContinualUniformTrainer._sample_indices, trainer)
  trainer._sample_weights = types.MethodType(ContinualUniformTrainer._sample_weights, trainer)
  uniform_loss = trainer._make_loss_fn(reg_def)
  uniform_epoch = trainer._build_train_epoch(reg_def)
  uniform_indices = trainer._sample_indices
  del trainer._sample_indices, trainer._sample_weights
  stats = {
    "structured": build_grad_stats(structured_loss, structured_indices, int(draws)),
    "uniform": build_grad_stats(uniform_loss, uniform_indices, int(draws)),
  }
  kernels = {"structured": structured_epoch, "uniform": uniform_epoch}

  carried_params, carried_state = trainer._running
  penalty = detector.design_penalty(design)
  penalty = None if penalty is None else float(penalty)
  record = {
    "trajectory": trajectory,
    "seed": int(seed),
    "design_index": index,
    "n_designs": len(rows),
    "x_scaled": [float(v) for v in design_scaled],
    "context_rows": [int(w0_train), int(w0_val)],
    "design_rows": [int(train_count), int(val_count)],
    "recorded_rows": [int(rows[index]["spent_train"]), int(rows[index]["spent_val"])],
    "reference_loss": rows[index].get("loss"),
    "design_penalty": penalty,
    "replay_weight": float(training.get("replay_weight", 1.0)),
    "batch": int(trainer.batch),
    "members": trainer.n_ensemble,
    "draws": int(draws),
    "protocol": {
      "max_epochs": max_epochs,
      "warmup_epochs": warmup_epochs,
      "patience": patience,
      "loss_precision": loss_precision,
      "overrun": overrun_epochs,
      "steps_per_epoch": int(trainer.steps_per_epoch),
      "optimizer": opt_name,
      "lr_schedule": "cosine",
    },
    "arms": {},
  }

  for arm_name in ("structured", "uniform"):
    params, state = carried_params, carried_state
    opt_state = trainer.optimizer.init(params)
    key = jax.random.PRNGKey(int(probe_seq.generate_state(1)[0]))
    measures = (
      "variance", "mean_variance", "signal", "loss", "mean_abs_moment", "mean_std", "moment_over_std", "mean_moment_over_std",
      "mean_moment_over_rms"
    )
    series = {k: [] for k in ("train", "train_sem", "val", "val_sem", "objective", "cosine")}
    series.update({f"{name}_{m}": [] for name in stats for m in measures})
    trains, train_sems, vals = [], [], []
    epoch, stop_epoch, p_settled = 0, None, float("nan")
    while epoch < max_epochs:
      key, k_train, k_stats = jax.random.split(key, 3)
      params, state, opt_state, _ = kernels[arm_name](params, state, opt_state, k_train, w0_train_j, count_j, tp.buffers())
      epoch += 1
      train_mean, train_sem = masked_mean_sem(trainer._eval_train(params, state, tp.buffers(), w0_train_j), train_count)
      val_mean, val_sem = masked_mean_sem(trainer._eval_val(params, state, vp.buffers(), w0_val_j), val_count)
      train_mean, train_sem, val_mean, val_sem = float(train_mean), float(train_sem), float(val_mean), float(val_sem)
      trains.append(train_mean)
      train_sems.append(train_sem)
      vals.append(val_mean)
      measured = {name: stats[name](params, state, k_stats, w0_train_j, count_j, tp.buffers()) for name in stats}
      ms, mu = measured["structured"]["mean"], measured["uniform"]["mean"]
      cosine = float(jnp.dot(ms, mu) / jnp.maximum(jnp.linalg.norm(ms) * jnp.linalg.norm(mu), 1e-30))
      for name in stats:
        for m in measures:
          series[f"{name}_{m}"].append(float(measured[name][m]))
      series["cosine"].append(cosine)
      series["train"].append(train_mean)
      series["train_sem"].append(train_sem)
      series["val"].append(val_mean)
      series["val_sem"].append(val_sem)
      series["objective"].append(0.5 * (train_mean + val_mean))
      if stop_epoch is None and epoch > warmup_epochs and len(trains) - warmup_epochs >= 3:
        mean, cov = bayesian_trend(
          np.asarray(trains[warmup_epochs:], np.float64), np.asarray(train_sems[warmup_epochs:], np.float64),
          max(trains[warmup_epochs], vals[warmup_epochs]) / 3.0
        )
        p_settled = float(probability_change_below(mean, cov, patience, 0.5 * loss_precision))
        if p_settled > 0.9:
          stop_epoch = epoch
          if overrun_epochs == 0:
            break
      elif stop_epoch is not None and epoch >= stop_epoch + overrun_epochs:
        break
      print(
        f"  [{arm_name}] epoch {epoch}/{max_epochs} train={train_mean:.4f} val={val_mean:.4f} "
        f"var s/u={series['structured_variance'][-1]:.3e}/{series['uniform_variance'][-1]:.3e} "
        f"cos={cosine:.3f} P(settled)={p_settled:.3f}", flush=True,
      )
    objective = 0.5 * (trains[-1] + vals[-1])
    record["arms"][arm_name] = {
      "epochs": epoch,
      "stop_epoch": stop_epoch if stop_epoch is not None else epoch,
      "settled": stop_epoch is not None,
      "train": trains[-1],
      "val": vals[-1],
      "objective": objective,
      "loss": objective if penalty is None else objective + penalty,
      "series": {
        k: [round(v, 8) for v in vs]
        for k, vs in series.items()
      },
    }
    print(
      f"[probe] {arm_name}: {epoch} epochs, objective {objective:.6f}, loss {record['arms'][arm_name]['loss']:.6f}", flush=True
    )

  os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
  with open(output, "w") as handle:
    json.dump(record, handle, indent=2)
  shutil.rmtree(work, ignore_errors=True)
  return record


if __name__ == "__main__":
  import sys

  import gearup

  gearup.gearup(probe_gradient_variance).with_config("config/root.yaml")(sys.argv[1:])
