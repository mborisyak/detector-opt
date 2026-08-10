#!/usr/bin/env python3
"""Why the independent verification does not reproduce the loss the BO run reported.

The experiment, per selected trajectory point: take the ACTUAL network that produced the reported
number -- the last checkpoint of that design, saved by the trainer at the epoch it declared
convergence -- and

  1. re-evaluate it on the design's OWN train/val windows, rebuilt event-index by event-index from
     the run's seed and the recorded ``spent`` (reproduces the reported number, or the checkpoint /
     window reconstruction is wrong and nothing below means anything);
  2. evaluate it, untouched, on the verification's fresh held-out val/test buffers -- the same
     estimator applied to data the design's network never saw. The difference to (1) is the
     reported number's GENERALISATION gap, with no retraining in between;
  3. continue training THAT network on the verification's full-budget train buffer, tracking after
     every epoch: the clean train loss (deterministic, ensemble-averaged -- the same estimator as
     val/test, over a fixed slice of the train buffer), the noisy running train loss (dropout on,
     per member: what the verification's learning curves plot), val, and test.

(3) answers what the reported number would have become with the data the verification gives it:
whether the run stopped early, at a genuinely converged loss, or somewhere the extra data cannot
reach. Writes ``continued.json`` + one plot per point into ``output``.

    python scripts/continue_reported.py =enzyme run=output/enzyme-42/from_scratch iterations=49,120
"""

import json
import os
from functools import partial

import matplotlib

matplotlib.use("AGG")
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import detopt
from detopt.utils import io
from detopt.utils.config import optimizer as make_optimizer, resolve_device
from detopt.utils.events import shuffled_event_index, split_disjoint
from detopt.utils.pools import RingBuffer


def bo_windows(results, n0, n_increment, val_fraction):
  """The per-design ``(train_start, n_train, val_start, n_val)`` windows of a finished run.

  The trainer grows a design's window in rounds (``n0``, then ``n_increment``), each round adding
  ``round(n * val_fraction / (1 - val_fraction))`` val events, and both pools are shared and
  APPENDED across designs -- so the recorded ``spent`` (train + val of one design) determines the
  number of rounds, and the running totals give each design's offsets into the run's event index.
  """
  ratio = val_fraction / (1.0 - val_fraction)
  first, step = n0 + round(n0 * ratio), n_increment + round(n_increment * ratio)
  train_at, val_at, out = 0, 0, []
  for r in results:
    spent = int(r["spent"])
    k, rest = divmod(spent - first, step)
    if rest != 0 or k < 0:
      raise ValueError(f"iteration {r['iteration']}: spent={spent} is not {first} + k*{step} -- window model wrong")
    n_train = n0 + k * n_increment
    n_val = round(n0 * ratio) + k * round(n_increment * ratio)
    out.append((train_at, n_train, val_at, n_val))
    train_at += n_train
    val_at += n_val
  return out


def continued(run, iterations="", seed: int = 0, output=None, epochs=None, progress=True, **config):
  device = resolve_device(config.get("device"))
  training, v = config["training"], config.get("verify", {})
  budget = int(v.get("budget") or training["budget"])
  batch = int(v.get("batch") or training["batch"])
  epochs = int(epochs if epochs is not None else v.get("epochs", 40))
  eval_batch = int(v.get("eval_batch", 2048))
  sample_batch = int(v.get("sample_batch", 1024))

  detector = detopt.detector.from_config(config["detector"])
  design_dim = int(detector.design_dim())
  results_path = os.path.join(run, "results.json")
  results = io.check_bo_results(json.load(open(results_path))["results"], results_path)
  chosen = [int(i) for i in str(iterations).split(",") if len(i) > 0] or [len(results) - 1]
  windows = bo_windows(results, int(training["n0"]), int(training["n_increment"]), float(training["val_fraction"]))
  out_dir = output if output is not None else os.path.join(run, "continued")
  os.makedirs(out_dir, exist_ok=True)

  # The run's own event index: ONE shuffled index over the whole budget, split into disjoint
  # train/val halves at the run's seed -- exactly how Trainer.__init__ builds it.
  run_budget = int(training["budget"])
  run_val_budget = round(run_budget * float(training["val_fraction"]))
  run_index = shuffled_event_index(detector.size(), run_budget, int(seed))
  run_train_index, run_val_index = run_index[:run_budget - run_val_budget], run_index[run_budget - run_val_budget:]

  # The verification's own fresh 6:2:2 split at the same seed (identical to verify_trajectory.py).
  rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(1)[0])
  n_train, n_val = round(0.6 * budget), round(0.2 * budget)
  if detector.size() is None:
    idx = rng.integers(0, 2**31 - 1, size=budget, dtype=np.int64)
    fresh = (idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:])
  else:
    universes = split_disjoint(rng.permutation(int(detector.size())), 0.6, 0.2)
    fresh = tuple(u[:n] for u, n in zip(universes, (n_train, n_val, budget - n_train - n_val)))

  M = int(jax.tree.leaves(detector.event_spec())[0].shape[0])
  specs = (detector.event_spec(), jax.ShapeDtypeStruct((M, ), jnp.int32), detector.target_spec())
  # The three fresh buffers are refilled with exactly ``capacity`` rows per point, so the ring's
  # cursor returns to where it started and row i is always event i of the split. The run's own
  # windows differ in size per design, so those buffers are allocated per point (below) -- a partial
  # push would otherwise land at the previous point's cursor rather than at row 0.
  train_buf, val_buf, test_buf = (RingBuffer(len(i), specs, device=device) for i in fresh)
  steps_per_epoch = max(1, n_train // batch)

  model = detopt.nn.from_config(detector, config=config["regressor"], rngs=nnx.Rngs(0))
  reg_def = nnx.split(model, nnx.Param, nnx.Variable)[0]
  members = model.ensemble()
  opt = make_optimizer(training["optimizer"])  # the run's optimizer, as the run ran it
  draw = (members or 1) * batch

  def fill(buf, theta, event_index, desc):
    event_index = np.asarray(event_index, np.int64)
    phys_full = detector.to_nominal(jnp.broadcast_to(theta[None, :], (sample_batch, design_dim)))
    bar = tqdm(total=event_index.shape[0], desc=desc, disable=not progress)
    for o in range(0, event_index.shape[0], sample_batch):
      idx = event_index[o:o + sample_batch]
      k = idx.shape[0]
      phys = phys_full if k == sample_batch else detector.to_nominal(jnp.broadcast_to(theta[None, :], (k, design_dim)))
      _gt, event, mask, target = detector(phys, idx)
      buf.push(event, mask, target)
      bar.update(k)
    bar.close()

  def _net_loss(params, state, drop_key, theta, event_b, mask_b, target_b):
    """The TRAIN path: dropout on, each ensemble member on its own slice of the drawn minibatch."""
    reg = nnx.merge(reg_def, params, state)
    feats = detector.combine_scaled(event_b, theta, mask=mask_b)
    emask = detector.element_mask(event_b, mask_b)
    target = detector.normalize_target(target_b)
    if members is None:
      per = reg.loss(detector.loss, feats, emask, target, deterministic=False, rngs=nnx.Rngs(drop_key))
    else:
      per = reg.loss(
        detector.loss, feats.reshape((members, batch) + feats.shape[1:]), emask.reshape((members, batch) + emask.shape[1:]),
        target.reshape((members, batch) + target.shape[1:]), deterministic=False, rngs=nnx.Rngs(drop_key)
      )
    _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
    return jnp.mean(per), new_state

  @jax.jit
  def train_epoch(params, state, opt_state, key, theta, event_buf, mask_buf, tgt_buf, n):

    def step(carry, k):
      params, state, opt_state = carry
      k_idx, k_drop = jax.random.split(k)
      idx = jax.random.randint(k_idx, (draw, ), 0, n)
      event_b = jax.tree.map(lambda a: a[idx], event_buf)
      target_b = jax.tree.map(lambda a: a[idx], tgt_buf)
      (loss, state), grads = jax.value_and_grad(_net_loss,
                                                has_aux=True)(params, state, k_drop, theta, event_b, mask_buf[idx], target_b)
      updates, opt_state = opt.update(grads, opt_state, params)
      return (params := optax.apply_updates(params, updates), state, opt_state), loss

    (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), jax.random.split(key, steps_per_epoch))
    return params, state, opt_state, jnp.mean(losses)

  @partial(jax.jit, static_argnames="rows")
  def evaluate(params, state, theta, event_buf, mask_buf, tgt_buf, rows):
    """One sequential pass over ``rows`` of a buffer on the ENSEMBLE-MEAN prediction, deterministic
    -- the estimator the trainers use for train, val and test alike. Returns per-sample losses."""
    reg = nnx.merge(reg_def, params, state)
    pool_rows = jax.tree.leaves(event_buf)[0].shape[0]

    def step(_carry, c):
      idx = jnp.clip(c * eval_batch + jnp.arange(eval_batch, dtype=jnp.int32), 0, pool_rows - 1)
      ev = jax.tree.map(lambda a: a[idx], event_buf)
      m = mask_buf[idx]
      feats = detector.combine_scaled(ev, theta, mask=m)
      emask = detector.element_mask(ev, m)
      tnorm = detector.normalize_target(jax.tree.map(lambda a: a[idx], tgt_buf))
      if members is None:
        pred = reg(feats, emask, deterministic=True)
      else:
        fe = jnp.broadcast_to(feats[None], (members, ) + feats.shape)
        me = jnp.broadcast_to(emask[None], (members, ) + emask.shape)
        pred = reg(fe, me, deterministic=True).mean(axis=0)
      return None, detector.loss(pred, tnorm)

    _, per = jax.lax.scan(step, None, jnp.arange(-(-rows // eval_batch)))
    return per.reshape(-1)[:rows]

  def score(params, state, theta, buf, rows=None):
    """``(mean, SEM)`` of the per-sample loss over the first ``rows`` of ``buf``."""
    rows = len(buf) if rows is None else int(rows)
    per = np.asarray(evaluate(params, state, theta, *buf.buffers(), rows=rows))
    return float(per.mean()), float(per.std() / np.sqrt(per.shape[0]))

  record = {
    "run": os.path.abspath(run),
    "seed": int(seed),
    "budget": budget,
    "batch": batch,
    "epochs": epochs,
    "split": [len(f) for f in fresh],
    "members": members,
    "points": []
  }
  for p in chosen:
    r = results[p]
    theta = jnp.asarray(r["x_scaled"], jnp.float32)
    reported = float(r["loss"])
    train_at, n_train_bo, val_at, n_val_bo = windows[p]

    # The network that reported the number: the design's LAST saved epoch.
    manager = io.get_checkpointer(os.path.join(run, "checkpoints", f"design_{p:04d}"))
    step_saved = manager.latest_step()
    pure_params, pure_state, ckpt_design, aux = io.restore_training_checkpoint(manager)
    manager.close()
    if not np.allclose(np.asarray(ckpt_design["scaled"], np.float32), np.asarray(r["x_scaled"], np.float32)):
      raise ValueError(f"iteration {p}: checkpoint design != results.json design")
    point_model = detopt.nn.from_config(detector, config=config["regressor"], rngs=nnx.Rngs(0))
    _, params, state = nnx.split(point_model, nnx.Param, nnx.Variable)
    nnx.replace_by_pure_dict(params, pure_params)
    nnx.replace_by_pure_dict(state, pure_state)
    params, state = jax.device_put(params, device), jax.device_put(state, device)
    print(
      f"\n[iteration {p}] reported={reported:.4f} (checkpoint epoch {step_saved}, "
      f"aux train={float(aux['train_loss']):.4f} val={float(aux['val_loss']):.4f}) | "
      f"BO window: train {n_train_bo} @ {train_at}, val {n_val_bo} @ {val_at}", flush=True
    )

    # (1) the design's OWN windows, rebuilt from the run's index -> reproduce the reported number.
    bo_train_buf = RingBuffer(n_train_bo, specs, device=device)
    bo_val_buf = RingBuffer(n_val_bo, specs, device=device)
    fill(bo_train_buf, theta, run_train_index[train_at:train_at + n_train_bo], "sample BO train window")
    fill(bo_val_buf, theta, run_val_index[val_at:val_at + n_val_bo], "sample BO val window")
    bo_train, bo_train_sem = score(params, state, theta, bo_train_buf, n_train_bo)
    bo_val, bo_val_sem = score(params, state, theta, bo_val_buf, n_val_bo)
    print(
      f"  own windows: train={bo_train:.4f}±{bo_train_sem:.4f} val={bo_val:.4f}±{bo_val_sem:.4f} "
      f"-> (train+val)/2 = {0.5 * (bo_train + bo_val):.4f} vs reported {reported:.4f}", flush=True
    )

    # (2) the same network on fresh held-out data, untouched.
    fill(train_buf, theta, fresh[0], "sample train")
    fill(val_buf, theta, fresh[1], "sample val")
    fill(test_buf, theta, fresh[2], "sample test")
    restored_val, restored_val_sem = score(params, state, theta, val_buf)
    restored_test, restored_test_sem = score(params, state, theta, test_buf)
    print(
      f"  fresh held-out (no training): val={restored_val:.4f}±{restored_val_sem:.4f} "
      f"test={restored_test:.4f}±{restored_test_sem:.4f} | delta(test-reported)={restored_test - reported:+.4f}", flush=True
    )

    # (3) continue THAT network on the full-budget train buffer -- and, for contrast, train a FRESH
    # network on the very same buffer (what the verification does), so the two curves differ only in
    # where the weights came from.
    n_rows = jnp.int32(len(train_buf))
    train_slice = min(len(train_buf), 16384)  # clean train estimate: a fixed slice, same estimator as val

    def curve(params, state, tag):
      opt_state = opt.init(params)
      key = jax.random.PRNGKey(int(np.random.SeedSequence([int(seed), p]).generate_state(1)[0]))
      rows = []
      for epoch in range(1, epochs + 1):
        key, subkey = jax.random.split(key)
        params, state, opt_state, running = train_epoch(params, state, opt_state, subkey, theta, *train_buf.buffers(), n_rows)
        clean_train, _ = score(params, state, theta, train_buf, train_slice)
        val, val_sem = score(params, state, theta, val_buf)
        test, test_sem = score(params, state, theta, test_buf)
        rows.append([epoch, float(running), clean_train, val, test])
        if progress:
          print(
            f"  [{tag}] epoch {epoch:>3}/{epochs}  running_train={float(running):.4f}  train={clean_train:.4f}  "
            f"val={val:.4f}±{val_sem:.4f}  test={test:.4f}±{test_sem:.4f}", flush=True
          )
      return rows

    history = curve(params, state, "continued")
    scratch_model = detopt.nn.from_config(detector, config=config["regressor"], rngs=nnx.Rngs(int(seed) + p))
    _, scratch_params, scratch_state = nnx.split(scratch_model, nnx.Param, nnx.Variable)
    scratch = curve(jax.device_put(scratch_params, device), jax.device_put(scratch_state, device), "from scratch")

    point = {
      "iteration": p,
      "reported_loss": reported,
      "checkpoint_epoch": int(step_saved),
      "checkpoint_train_loss": float(aux["train_loss"]),
      "checkpoint_val_loss": float(aux["val_loss"]),
      "bo_window": {
        "train": bo_train,
        "train_sem": bo_train_sem,
        "val": bo_val,
        "val_sem": bo_val_sem,
        "n_train": n_train_bo,
        "n_val": n_val_bo
      },
      "restored": {
        "val": restored_val,
        "val_sem": restored_val_sem,
        "test": restored_test,
        "test_sem": restored_test_sem
      },
      "final": {
        "train": history[-1][2],
        "val": history[-1][3],
        "test": history[-1][4]
      },
      "final_from_scratch": {
        "train": scratch[-1][2],
        "val": scratch[-1][3],
        "test": scratch[-1][4]
      },
      "history": history,
      "history_from_scratch": scratch,
    }
    record["points"].append(point)
    with open(os.path.join(out_dir, "continued.json"), "w") as f:
      json.dump(record, f, indent=2)
    _plot(point, os.path.join(out_dir, f"continued_{p:03d}.png"))
    print(
      f"  -> reported {reported:.4f} | restored on fresh test {restored_test:.4f} | "
      f"after {epochs} more epochs on {len(train_buf)} events: test {history[-1][4]:.4f}", flush=True
    )
  print(f"saved -> {out_dir}")
  return record


def _plot(point, path):
  """Left: the reported number (dashed) against what its own network scores on held-out data before
  (star at epoch 0) and during continued training on the full budget, with the from-scratch network
  on the same data for contrast. Right: the two train-loss ESTIMATORS of the continued run -- the
  running per-member dropout loss the learning curves plot, and the clean ensemble loss that is
  comparable with val -- which is why the plotted train can sit above validation."""
  from matplotlib.figure import Figure

  h = np.asarray(point["history"], np.float64)
  s = np.asarray(point["history_from_scratch"], np.float64)
  fig = Figure(figsize=(13, 5.5))
  ax, ax_est = fig.subplots(1, 2)
  for a in (ax, ax_est):
    a.axhline(point["reported_loss"], ls="--", color="0.45", lw=1.5, label=f"reported by BO = {point['reported_loss']:.4f}")
    a.grid(True, alpha=0.25)
  ax.plot(
    0, point["restored"]["test"], "*", ms=14, color="tab:red",
    label=f"reported net, fresh test = {point['restored']['test']:.4f}"
  )
  ax.plot(h[:, 0], h[:, 3], ".-", color="tab:orange", label="continued: validation")
  ax.plot(h[:, 0], h[:, 4], ".-", color="tab:blue", label=f"continued: test -> {h[-1, 4]:.4f}")
  ax.plot(s[:, 0], s[:, 3], ".--", color="tab:orange", alpha=0.55, label="from scratch: validation")
  ax.plot(s[:, 0], s[:, 4], ".--", color="tab:blue", alpha=0.55, label=f"from scratch: test -> {s[-1, 4]:.4f}")
  ax.set(
    title=f"iteration {point['iteration']}: the reported network on the full budget", xlabel="epoch",
    ylabel="loss (normalized)", yscale="log"
  )
  ax.legend(fontsize=8)

  ax_est.plot(h[:, 0], h[:, 1], ".-", color="tab:green", label="train, running (dropout, per member)")
  ax_est.plot(h[:, 0], h[:, 2], ".-", color="tab:olive", label="train, clean (deterministic, ensemble)")
  ax_est.plot(h[:, 0], h[:, 3], ".-", color="tab:orange", label="validation (deterministic, ensemble)")
  ax_est.set(title="the two train-loss estimators (continued)", xlabel="epoch", yscale="log")
  ax_est.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=140)


if __name__ == "__main__":
  import gearup

  gearup.gearup(continued).with_config("config/bo.yaml")()
