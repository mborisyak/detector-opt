#!/usr/bin/env python3
"""Independent verification of a BO design-optimization trajectory.

Reads a finished bo.py run's ``results.json`` (a run directory is searched for it), selects at most
``verify.n_points`` designs along the trajectory (ALWAYS including the final one), spread
more-or-less uniformly in cumulative detector calls, and re-scores every selected design
independently of the run's own training machinery:

  1. sample the verification budget of events at the FIXED design;
  2. split 6:2:2 into train / validation / test buffers (disjoint event sets, drawn once and shared
     by every point, so the scores along the trajectory are paired);
  3. train a fresh (ensemble) regressor on the train buffer for ``verify.epochs`` epochs;
  4. every ``verify.val_every_epochs`` epochs, evaluate the WHOLE validation buffer (sequentially,
     batch-by-batch) and keep the parameters with the best validation loss;
  5. at the end only, evaluate the WHOLE test buffer at those parameters and report the TEST loss
     (+- SEM) -- an independent held-out score the optimizer never saw. Each point also gets a
     learning-curve plot (train/val per epoch, the test score as a dashed line) in ``plots/``.

``verify.budget`` and ``verify.batch`` default to the run's ``training.budget`` / ``training.batch``
(an explicit ``null`` counts as absent); the architecture and optimizer come from the run config
(``regressor`` / ``training.optimizer``). A finite data-backed detector caps each
split at its share of unique events (a repeated index replays the IDENTICAL row -- duplicates would
only burn simulation time), keeping the three sets disjoint. Run with the config of the run that
produced the trajectory, so the detector matches the trajectory's design encoding::

    python scripts/verify_trajectory.py trajectory=output/bo_full/from_scratch

Writes ``verification.json`` + ``verification.png`` next to the trajectory file (or into ``output=``).
Resumable: points already present in ``verification.json`` under identical settings are reused, not
recomputed (a settings mismatch recomputes everything -- the old data answers a different question).
"""

import json
import os
from functools import partial

import matplotlib

matplotlib.use("AGG")  # before any pyplot import (detopt.utils.viz pulls it in)
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import detopt
from detopt.utils.config import optimizer as make_optimizer, resolve_device
from detopt.utils.pools import RingBuffer
from detopt.utils.events import split_disjoint


def _forward_loss(reg, loss_fn, feats, mask, target, members, batch, *, deterministic, rngs=None):
  """TRAIN loss path (mirrors scripts/verify_regressor.py): a ``(members*batch, ...)`` minibatch ->
  per-sample loss ``(members*batch,)``; each member gets its OWN slice."""
  if members is None:
    return reg.loss(loss_fn, feats, mask, target, deterministic=deterministic, rngs=rngs)
  feats_e = feats.reshape((members, batch) + feats.shape[1:])
  mask_e = mask.reshape((members, batch) + mask.shape[1:])
  target_e = target.reshape((members, batch) + target.shape[1:])
  loss = reg.loss(loss_fn, feats_e, mask_e, target_e, deterministic=deterministic, rngs=rngs)
  return loss.reshape((members * batch, ) + loss.shape[2:])


def _predict_shared(reg, feats, mask, members):
  """EVAL path: ONE batch fed to every member (broadcast), member predictions AVERAGED ->
  ``(batch, T)`` -- the ensemble score (matches the trainers' eval kernel)."""
  if members is None:
    return reg(feats, mask, deterministic=True)
  fe = jnp.broadcast_to(feats[None], (members, ) + feats.shape)
  me = jnp.broadcast_to(mask[None], (members, ) + mask.shape)
  return reg(fe, me, deterministic=True).mean(axis=0)


def _key(seq):
  """A jax PRNGKey from the next child of a SeedSequence (advances ``seq`` deterministically)."""
  return jax.random.PRNGKey(int(seq.spawn(1)[0].generate_state(1)[0]))


def _load_trajectory(path):
  """Load a bo.py run's ``results.json`` (given directly, or found in a run directory). Returns a
  dict with the flat PHYSICAL designs ``(n, d)``, the ENCODED designs (exactly what the optimizer
  evaluated), the cumulative detector ``calls`` per iteration (cumsum of the recorded per-design
  ``spent``) and the run's reported objective loss per iteration."""
  if os.path.isdir(path):
    path = os.path.join(path, "results.json")
  with open(path) as f:
    rs = json.load(f)["results"]
  if len(rs) == 0:
    raise ValueError(f"{path}: empty BO results")
  return {
    "path": path,
    "physical": np.asarray([r["design"] for r in rs], np.float32),
    "encoded": np.asarray([r["x_encoded"] for r in rs], np.float32),
    "calls": np.cumsum([int(r["spent"]) for r in rs]).astype(np.float64),
    "reported": np.asarray([r["loss"] for r in rs], np.float64),
  }


def _select_points(calls, n_max):
  """At most ``n_max`` trajectory indices, more-or-less uniformly spread in cumulative detector
  ``calls`` and ALWAYS including the last point: the nearest trajectory point to each of ``n_max``
  uniform call levels between the first and last point (duplicates collapse, so fewer than
  ``n_max`` may come back)."""
  calls = np.asarray(calls, np.float64)
  n = calls.shape[0]
  if n <= n_max:
    return list(range(n))
  if n_max == 1:
    return [n - 1]
  targets = np.linspace(calls[0], calls[-1], n_max)
  chosen = {int(np.argmin(np.abs(calls - t))) for t in targets}
  chosen.add(n - 1)  # the last target lands on the last point already; kept explicit
  return sorted(chosen)


def _split_indices(size, budget, seed):
  """Disjoint 6:2:2 train/val/test event indices over ``budget`` detector calls -- the split every
  trajectory point shares (paired scores). For a finite detector the 6:2:2 split is applied to the
  UNIQUE event universe first and each part is capped at its unique count (the detector is
  deterministic per index, so a wrapped repeat replays the identical row)."""
  rng = np.random.default_rng(seed)
  n_train, n_val = round(0.6 * budget), round(0.2 * budget)
  n_test = budget - n_train - n_val
  if size is None:  # infinite analytic source: any indices, disjoint by construction
    index = rng.integers(0, 2**31 - 1, size=budget, dtype=np.int64)
    return index[:n_train], index[n_train:n_train + n_val], index[n_train + n_val:]
  universes = split_disjoint(rng.permutation(int(size)), 0.6, 0.2)
  return tuple(u[:n] for u, n in zip(universes, (n_train, n_val, n_test)))


def verify(trajectory, seed: int = 0, output=None, progress=True, **config):
  device = resolve_device(config.get("device"))
  v = config.get("verify")
  if v is None:
    v = {}
  training = config.get("training")
  if training is None:
    training = {}
  n_points = int(v.get("n_points", 10))
  budget = v.get("budget")  # None or absent -> the run's training.budget
  if budget is None:
    budget = training.get("budget")
  if budget is None:
    raise ValueError("no verification budget: set verify.budget (or training.budget)")
  budget = int(budget)
  epochs = int(v.get("epochs", 8))  # training epochs per trajectory point (epoch = one pass over the train buffer)
  val_every_epochs = int(v.get("val_every_epochs", 1))  # validate (over the whole val buffer) every this many epochs
  batch = v.get("batch")  # None or absent -> the run's training.batch; minibatch size (per ensemble member)
  if batch is None:
    batch = training.get("batch", 256)
  batch = int(batch)
  sample_batch = int(v.get("sample_batch", 1024))  # events simulated per detector call while filling buffers
  eval_batch = int(v.get("eval_batch", 2048))  # evaluation scans buffers sequentially in chunks of this size

  detector = detopt.detector.from_config(config["detector"])
  design_dim = int(detector.design_dim())
  labels = tuple(detector.metric_labels())

  traj = _load_trajectory(trajectory)
  if traj["physical"].shape[1] != design_dim:
    raise ValueError(
      f"trajectory design dim {traj['physical'].shape[1]} != detector design dim {design_dim} "
      f"-- run with the config of the run that produced the trajectory"
    )
  chosen = _select_points(traj["calls"], n_points)
  out_dir = output if output is not None else os.path.dirname(os.path.abspath(traj["path"]))
  os.makedirs(out_dir, exist_ok=True)

  master = np.random.SeedSequence(int(seed))
  index_seq = master.spawn(1)[0]  # the shared 6:2:2 event split
  template_seq = master.spawn(1)[0]  # the throwaway template regressor (architecture only)
  train_index, val_index, test_index = _split_indices(detector.size(), budget, index_seq)

  # Three disjoint fixed-design buffers of raw (event, mask, target) rows, allocated once and fully
  # overwritten per trajectory point (the design is fixed per point -> no per-event design column;
  # ``combine_encoded`` runs per batch with the point's theta).
  M = int(jax.tree.leaves(detector.event_spec())[0].shape[0])  # per-hit count
  specs = (detector.event_spec(), jax.ShapeDtypeStruct((M, ), jnp.int32), detector.target_spec())
  train_buf = RingBuffer(len(train_index), specs, device=device)
  val_buf = RingBuffer(len(val_index), specs, device=device)
  test_buf = RingBuffer(len(test_index), specs, device=device)
  steps_per_epoch = max(1, len(train_index) // batch)
  scan_steps = val_every_epochs * steps_per_epoch  # SGD steps folded into one train_chunk call

  # Template regressor: the graphdef (architecture) is shared by every point, so the JIT kernels
  # compile once; each point re-initialises fresh params below.
  model = detopt.nn.from_config(detector, config=config["regressor"], rngs=nnx.Rngs(_key(template_seq)))
  reg_def = nnx.split(model, nnx.Param, nnx.Variable)[0]
  members = model.ensemble()
  opt = make_optimizer(config["training"]["optimizer"])
  draw = (members or 1) * batch

  def fill(buf, theta, event_index, desc):
    """(Re)fill ``buf`` with the events at ``event_index`` simulated at the FIXED encoded design
    ``theta``; pushing exactly ``capacity`` rows overwrites the whole ring. STRICTLY SERIAL: the
    propagation engine fills detector-owned buffers in place, so concurrent calls on one detector
    instance corrupt each other (verified: threaded fills produce NaN losses)."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    # One decoded physical design serves every full-size chunk (theta is fixed within a fill).
    phys_full = detector.decode_design(jnp.broadcast_to(theta[None, :], (sample_batch, design_dim)))
    bar = tqdm(total=n, desc=desc, disable=not progress)
    for o in range(0, n, sample_batch):
      idx = event_index[o:o + sample_batch]
      k = idx.shape[0]
      phys = phys_full if k == sample_batch else detector.decode_design(jnp.broadcast_to(theta[None, :], (k, design_dim)))
      _gt, event, mask, target = detector(phys, idx)
      buf.push(event, mask, target)
      bar.update(k)
    bar.close()

  def _net_loss(params, state, drop_key, theta, event_b, mask_b, target_b):
    reg = nnx.merge(reg_def, params, state)
    feats = detector.combine_encoded(event_b, theta, mask=mask_b)  # fixed design (encoded), per hit
    emask = detector.element_mask(event_b, mask_b)  # per-element mask (== hit mask, unless layer-wise)
    loss = jnp.mean(
      _forward_loss(
        reg, detector.loss, feats, emask, detector.normalize_target(target_b), members, batch, deterministic=False,
        rngs=nnx.Rngs(drop_key)
      )
    )
    _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
    return loss, new_state

  @jax.jit
  def train_chunk(params, state, opt_state, key, theta, event_buf, mask_buf, tgt_buf, n):
    """One validation interval -- ``val_every_epochs`` epochs of scan-folded SGD over the train buffer at
    the fixed ``theta``. Returns the mean training loss of the interval (accumulated in the carry)."""

    def step(carry, k):
      params, state, opt_state, loss_sum = carry
      k_idx, k_drop = jax.random.split(k)
      idx = jax.random.randint(k_idx, (draw, ), 0, n)
      event_b = jax.tree.map(lambda a: a[idx], event_buf)
      target_b = jax.tree.map(lambda a: a[idx], tgt_buf)
      (loss, state), grads = jax.value_and_grad(_net_loss,
                                                has_aux=True)(params, state, k_drop, theta, event_b, mask_buf[idx], target_b)
      updates, opt_state = opt.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return (params, state, opt_state, loss_sum + loss), None

    carry0 = (params, state, opt_state, jnp.float32(0.0))
    (params, state, opt_state, loss_sum), _ = jax.lax.scan(step, carry0, jax.random.split(key, scan_steps))
    return params, state, opt_state, loss_sum / scan_steps

  @partial(jax.jit, static_argnames="rows")
  def evaluate(params, state, theta, event_buf, mask_buf, tgt_buf, rows):
    """One sequential batch-by-batch pass over the WHOLE buffer (all ``rows``; the final partial
    chunk is masked, nothing is dropped), on the ENSEMBLE-MEAN prediction. Scalar accumulation
    only: returns the mean per-key metric plus the mean loss and its SEM."""
    reg = nnx.merge(reg_def, params, state)
    n_chunks = -(-rows // eval_batch)  # ceil
    pool_rows = jax.tree.leaves(event_buf)[0].shape[0]

    def step(acc, c):
      idx = c * eval_batch + jnp.arange(eval_batch, dtype=jnp.int32)
      valid = (idx < rows).astype(jnp.float32)  # mask for the final partial chunk
      safe = jnp.clip(idx, 0, pool_rows - 1)
      ev = jax.tree.map(lambda a: a[safe], event_buf)
      m = mask_buf[safe]
      feats = detector.combine_encoded(ev, theta, mask=m)
      emask = detector.element_mask(ev, m)
      tnorm = detector.normalize_target(jax.tree.map(lambda a: a[safe], tgt_buf))
      pred = _predict_shared(reg, feats, emask, members)
      per = detector.loss(pred, tnorm)  # per-sample ensemble loss (eval_batch,)
      md = detector.metric(pred, tnorm)
      loss_sum, sq_sum, metric_sum = acc
      return (
        loss_sum + jnp.sum(per * valid), sq_sum + jnp.sum(jnp.square(per) * valid), {
          k: metric_sum[k] + jnp.sum(md[k] * valid)
          for k in labels
        },
      ), None

    init = (jnp.float32(0.0), jnp.float32(0.0), {name: jnp.float32(0.0) for name in labels})
    (loss_sum, sq_sum, metric_sum), _ = jax.lax.scan(step, init, jnp.arange(n_chunks))
    mean = loss_sum / rows
    sem = jnp.sqrt(jnp.maximum(sq_sum / rows - jnp.square(mean), 0.0) / rows)
    out = {k: metric_sum[k] / rows for k in labels}
    out.update(loss=mean, loss_sem=sem)
    return out

  print(f"trajectory: {traj['path']} ({traj['physical'].shape[0]} points) | verifying {len(chosen)} points: {chosen}")
  print(
    f"budget={budget} -> split train/val/test = {len(train_index)}/{len(val_index)}/{len(test_index)} | "
    f"epochs={epochs} ({steps_per_epoch} steps each) batch={batch} members={members or 1} seed={seed}", flush=True,
  )

  settings = {
    "trajectory": os.path.abspath(traj["path"]),
    "budget": budget,
    "split": [len(train_index), len(val_index), len(test_index)],
    "seed": int(seed),
    "epochs": epochs,
    "val_every_epochs": val_every_epochs,
    "steps_per_epoch": steps_per_epoch,
    "batch": batch,
    "members": members,
  }
  record = {**settings, "reported": {"calls": traj["calls"].tolist(), "loss": traj["reported"].tolist()}, "points": []}
  json_path = os.path.join(out_dir, "verification.json")
  plots_dir = os.path.join(out_dir, "plots")
  os.makedirs(plots_dir, exist_ok=True)

  # Resume: points already verified under IDENTICAL settings are reused, never recomputed; anything
  # else in the file is superseded (the settings drive the result, so a mismatch means stale data).
  if os.path.exists(json_path):
    with open(json_path) as f:
      stored = json.load(f)
    mismatched = [k for k in settings if stored.get(k) != settings[k]]
    if len(mismatched) == 0:
      record["points"] = stored.get("points", [])
      print(f"resume: reusing {len(record['points'])} already-verified points from {json_path}", flush=True)
    else:
      print(f"existing {json_path} was made with different {', '.join(mismatched)}; recomputing every point", flush=True)
  done_points = {pt["point"] for pt in record["points"]}

  for rank, p in enumerate(chosen):
    # One seed child per point, spawned for done points too, so a resumed run hands every remaining
    # point the same keys as an uninterrupted one (subgradient.py's spawn-and-discard idiom).
    point_seq = master.spawn(1)[0]
    if p in done_points:
      pt = next(entry for entry in record["points"] if entry["point"] == p)
      plot_path = os.path.join(plots_dir, f"verification_{p:03d}.png")
      if not os.path.exists(plot_path):  # e.g. produced under an older plot naming
        _plot_learning(pt["history"], pt["test_loss"], pt["test_sem"], p, plot_path)
      print(
        f"[point {rank + 1}/{len(chosen)}] iteration {p} already verified: test={pt['test_loss']:.4f} -- skipped", flush=True
      )
      continue
    theta = jnp.asarray(traj["encoded"][p], jnp.float32)  # exactly what the optimizer evaluated
    phys_flat = np.asarray(detector.flatten_design(detector.decode_design(theta)), np.float32)
    reported = float(traj["reported"][p])
    print(
      f"\n[point {rank + 1}/{len(chosen)}] iteration {p} @ {traj['calls'][p]:.0f} detector calls | "
      f"reported={reported:.4f} | design={np.round(phys_flat, 3).tolist()}", flush=True,
    )

    fill(train_buf, theta, train_index, "sample train")
    fill(val_buf, theta, val_index, "sample val")
    fill(test_buf, theta, test_index, "sample test")

    # Fresh network + optimizer per point (same graphdef -> the kernels stay compiled).
    point_model = detopt.nn.from_config(detector, config=config["regressor"], rngs=nnx.Rngs(_key(point_seq)))
    _, params, state = nnx.split(point_model, nnx.Param, nnx.Variable)
    params, state = jax.device_put(params, device), jax.device_put(state, device)
    opt_state = opt.init(params)
    n_train = jnp.int32(len(train_buf))

    best = None  # (val_loss, epoch, params, state)
    history = []  # [epoch, train_loss (interval mean), val_loss]
    train_loss = float("nan")
    epoch = 0
    while epoch < epochs:
      params, state, opt_state, loss_mean = train_chunk(
        params, state, opt_state, _key(point_seq), theta, *train_buf.buffers(), n_train
      )
      epoch += val_every_epochs
      train_loss = float(loss_mean)
      val_loss = float(evaluate(params, state, theta, *val_buf.buffers(), rows=len(val_buf))["loss"])
      history.append([epoch, train_loss, val_loss])
      if best is None or val_loss < best[0]:
        best = (val_loss, epoch, params, state)
      if progress:
        print(f"  epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  best={best[0]:.4f}@{best[1]}", flush=True)

    best_val, best_epoch, best_params, best_state = best
    test = {k: float(x) for k, x in evaluate(best_params, best_state, theta, *test_buf.buffers(), rows=len(test_buf)).items()}
    print(
      f"  -> best val={best_val:.4f} (epoch {best_epoch})  TEST={test['loss']:.4f}±{test['loss_sem']:.4f}  "
      f"reported={reported:.4f}  delta(test-reported)={test['loss'] - reported:+.4f}", flush=True,
    )
    _plot_learning(history, test["loss"], test["loss_sem"], p, os.path.join(plots_dir, f"verification_{p:03d}.png"))

    record["points"].append({
      "point": int(p),
      "detector_calls": float(traj["calls"][p]),
      "reported_loss": reported,
      "design_physical": phys_flat.tolist(),
      "design_encoded": np.asarray(theta).tolist(),
      "best_epoch": int(best_epoch),
      "train_loss": train_loss,
      "val_loss": float(best_val),
      "test_loss": test["loss"],
      "test_sem": test["loss_sem"],
      "test_metric": {
        k: x
        for k, x in test.items() if k != "loss_sem"
      },
      "history": history,
    })
    record["points"].sort(key=lambda entry: entry["point"])  # trajectory order, whatever the resume order was
    with open(json_path, "w") as f:  # incremental: every finished point is on disk
      json.dump(record, f, indent=2)

  _plot(record, os.path.join(out_dir, "verification.png"))
  _plot_comparison(record, os.path.join(out_dir, "verification_comparison.png"))
  print(f"\n{'index':>6} {'calls':>10} {'reported':>9} {'train':>8} {'val':>8} {'test':>16}")
  for pt in record["points"]:
    print(
      f"{pt['point']:>6} {pt['detector_calls']:>10.0f} {pt['reported_loss']:>9.4f} "
      f"{pt['train_loss']:>8.4f} {pt['val_loss']:>8.4f} {pt['test_loss']:>9.4f}±{pt['test_sem']:.4f}"
    )
  print(f"saved -> {json_path}")
  return record


def _plot_comparison(record, path):
  """The headline figure: the run's own COMPUTED loss trajectory (dashed -- the optimizer's
  possibly-biased objective) vs the PROPER independent estimates (solid -- held-out test ±SEM at the
  verified points), both against cumulative detector calls."""
  from matplotlib.figure import Figure

  points = record["points"]
  calls = np.asarray([pt["detector_calls"] for pt in points], np.float64)
  reported = record["reported"]
  fig = Figure(figsize=(9, 5.5))
  ax = fig.subplots(1, 1)
  ax.plot(
    np.asarray(reported["calls"], np.float64), np.asarray(reported["loss"], np.float64), "--", color="0.45", lw=1.4,
    label="computed (BO objective)"
  )
  ax.errorbar(
    calls, [pt["test_loss"] for pt in points], yerr=[pt["test_sem"] for pt in points], fmt="o-", ms=5, lw=1.8, capsize=3,
    color="tab:blue", label="proper estimate (held-out test)"
  )
  ax.set(title="computed vs proper loss estimates", xlabel="detector calls", ylabel="loss (normalized)", yscale="log")
  ax.grid(True, alpha=0.25)
  ax.legend(fontsize=9)
  fig.tight_layout()
  fig.savefig(path, dpi=140)


def _plot_learning(history, test_loss, test_sem, iteration, path):
  """Per-point learning curves: train + validation loss per epoch, with the final TEST score as a
  horizontal dashed line (value ± SEM in the legend)."""
  from matplotlib.figure import Figure

  h = np.asarray(history, np.float64)  # (n, 3): epoch, train, val
  fig = Figure(figsize=(8, 5))
  ax = fig.subplots(1, 1)
  ax.plot(h[:, 0], h[:, 1], ".-", color="tab:green", label="train")
  ax.plot(h[:, 0], h[:, 2], ".-", color="tab:orange", label="validation")
  ax.axhline(test_loss, ls="--", color="tab:blue", lw=1.5, label=f"test = {test_loss:.4f} ± {test_sem:.4f}")
  ax.set(title=f"iteration {iteration}", xlabel="epoch", ylabel="loss (normalized)", yscale="log")
  ax.grid(True, alpha=0.25)
  ax.legend(fontsize=9)
  fig.tight_layout()
  fig.savefig(path, dpi=140)


def _plot(record, path):
  """Two panels: the run's reported loss curve with the verified train/val/test scores on top, and
  the per-component test metric of the best-val network -- both against detector calls."""
  from matplotlib.figure import Figure

  points = record["points"]
  calls = np.asarray([pt["detector_calls"] for pt in points], np.float64)
  fig = Figure(figsize=(13, 5.5))
  ax_loss, ax_comp = fig.subplots(1, 2)

  reported = record["reported"]
  rc, rl = np.asarray(reported["calls"], np.float64), np.asarray(reported["loss"], np.float64)
  ax_loss.plot(rc, rl, "-", color="0.65", lw=1.2, label="reported (BO objective)")
  ax_loss.plot(calls, [pt["train_loss"] for pt in points], "s--", ms=4, color="tab:green", alpha=0.7, label="verified train")
  ax_loss.plot(calls, [pt["val_loss"] for pt in points], "o--", ms=5, color="tab:orange", alpha=0.8, label="verified val")
  ax_loss.errorbar(
    calls, [pt["test_loss"] for pt in points], yerr=[pt["test_sem"] for pt in points], fmt="D-", ms=6, capsize=3,
    color="tab:blue", label="verified TEST"
  )
  ax_loss.set(title="independent verification", xlabel="detector calls", ylabel="loss (normalized)", yscale="log")
  ax_loss.grid(True, alpha=0.25)
  ax_loss.legend(fontsize=8)

  names = [k for k in points[0]["test_metric"] if k != "loss"]
  for name in names:
    ax_comp.plot(calls, [pt["test_metric"][name] for pt in points], ".-", label=name)
  ax_comp.set(title="test metric per component (best-val network)", xlabel="detector calls", yscale="log")
  ax_comp.grid(True, alpha=0.25)
  if len(names) > 0:
    ax_comp.legend(fontsize=8, ncol=2)
  fig.tight_layout()
  fig.savefig(path, dpi=140)


if __name__ == "__main__":
  import gearup

  gearup.gearup(verify).with_config("config/bo.yaml")()
