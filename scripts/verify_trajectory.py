#!/usr/bin/env python3
"""Independent verification of a BO design-optimization trajectory.

Reads a finished bo.py run's ``results.json`` (a run directory is searched for it), selects at most
``verify.n_points`` INCUMBENTS -- the iterations where the run's best-so-far loss improved, always
including the LAST such iteration, which is the run's answer -- and re-scores each of them on data
the run never saw:

  1. sample the verification budget of events at the FIXED design;
  2. split 6:2:2 into train / validation / test buffers (disjoint event sets, drawn once and shared
     by every point, so the scores along the trajectory are paired);
  3. restore the network the run REPORTED that design with (its last per-design checkpoint) and
     continue training it on the fresh train buffer for ``verify.epochs`` epochs. Same network, all
     new data: what changes between the reported number and this one is only the data it is measured
     on -- and, past epoch 0, the extra training the fresh budget buys;
  4. every ``verify.val_every_epochs`` epochs, evaluate the WHOLE validation buffer (sequentially,
     batch-by-batch) and keep the parameters with the best validation loss -- epoch 0 (the restored
     network, untouched) counts, so a design that only degrades keeps its reported network;
  5. at the end only, evaluate the WHOLE test buffer at those parameters and report the TEST loss
     (+- SEM) -- an independent held-out score the optimizer never saw. Each point also gets a
     learning-curve plot (validation per epoch, reported + test as horizontal lines) in ``plots/``.

``verify.budget`` and ``verify.batch`` default to the run's ``training.budget`` / ``training.batch``
(an explicit ``null`` counts as absent); the architecture and optimizer come from the run config
(``regressor`` / ``training.optimizer``), with the learning rate wrapped in a single-cycle cosine
decay spanning each point's whole training (peak -> ~0, restarted per point). A finite data-backed detector caps each
split at its share of unique events (a repeated index replays the IDENTICAL row -- duplicates would
only burn simulation time), keeping the three sets disjoint. Run with the config of the run that
produced the trajectory, so the detector matches the trajectory's design encoding::

    python scripts/verify_trajectory.py trajectory=output/bo_full/from_scratch

Writes ``verification.json`` + ``verification.png`` next to the trajectory file (or into ``output=``).
Resumable: points already present in ``verification.json`` under identical settings are reused, not
recomputed (a settings mismatch recomputes everything -- the old data answers a different question),
and ``--force`` recomputes every point regardless.
"""

import json
import os
import threading
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
from detopt.utils import io
from detopt.utils.config import resolve_device, split
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
  dict with the flat NOMINAL designs ``(n, d)``, the SCALED designs (exactly what the optimizer
  evaluated), the cumulative detector ``calls`` per iteration (cumsum of the recorded per-design
  ``spent``) and the run's reported objective loss per iteration."""
  if os.path.isdir(path):
    path = os.path.join(path, "results.json")
  with open(path) as f:
    rs = json.load(f)["results"]
  if len(rs) == 0:
    raise ValueError(f"{path}: empty BO results")
  io.check_bo_results(rs, path)
  return {
    "path": path,
    "physical": np.asarray([r["design"] for r in rs], np.float32),
    "scaled": np.asarray([r["x_scaled"] for r in rs], np.float32),
    "calls": np.cumsum([int(r["spent"]) for r in rs]).astype(np.float64),
    "reported": np.asarray([r["loss"] for r in rs], np.float64),
  }


def _restore_design_network(run_dir, iteration):
  """The ``(parameters, state, design)`` of the network the run trained for ``iteration`` -- the last
  epoch checkpointed under ``<run_dir>/checkpoints/design_<iteration>``, which is the one whose loss
  the run reported. Parameters and state come back as pure dicts (load them into a freshly built
  module's abstract state with ``nnx.replace_by_pure_dict``)."""
  path = os.path.join(run_dir, "checkpoints", f"design_{iteration:04d}")
  if not os.path.isdir(path):
    raise FileNotFoundError(
      f"no checkpoint at {path} -- verification continues the network the run reported, so the run's "
      f"per-design checkpoints must be kept"
    )
  manager = io.get_checkpointer(path)
  if manager.latest_step() is None:
    raise ValueError(f"{path} holds no saved epoch")
  parameters, state, design, _aux = io.restore_training_checkpoint(manager)
  manager.close()
  return parameters, state, design


def _select_points(calls, reported, n_max):
  """At most ``n_max`` trajectory indices, taken from the INCUMBENTS -- the iterations where the
  best-so-far loss improves (the first point always is one). Those are the only designs the run
  actually claims anything about: the rest of the trajectory is proposals the optimizer tried and
  discarded, and re-scoring them says nothing about whether the optimization worked. The last
  incumbent is the run's answer, so it is always kept; if there are more than ``n_max``, the ones in
  between are thinned to those nearest to ``n_max`` levels uniform in cumulative detector calls."""
  reported = np.asarray(reported, np.float64)
  improves = np.flatnonzero(reported < np.minimum.accumulate(np.concatenate([[np.inf], reported[:-1]])))
  if improves.shape[0] <= n_max:
    return [int(i) for i in improves]
  if n_max == 1:
    return [int(improves[-1])]
  calls = np.asarray(calls, np.float64)[improves]
  targets = np.linspace(calls[0], calls[-1], n_max)
  chosen = {int(improves[np.argmin(np.abs(calls - t))]) for t in targets}
  chosen.add(int(improves[-1]))  # the run's answer -- the last target lands on it anyway
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


# How the sampling fills report themselves. ``bar`` is tqdm, fine on a terminal; ``plain`` drops the
# bar and leaves only the per-validation-epoch lines, which is what a log file or a snakemake pipe
# wants (tqdm redraws with carriage returns, so a redirected bar is thousands of unreadable lines);
# ``none`` is silent apart from the per-point summaries.
PROGRESS_MODES = ("bar", "plain", "none")


def verify(trajectory, seed: int = 0, output=None, progress: str = "bar", force: bool = False, **config):
  if progress not in PROGRESS_MODES:
    raise ValueError(f"progress must be one of {PROGRESS_MODES}, got {progress!r}")
  device = resolve_device(config.get("device"))
  v = config.get("verify")
  if v is None:
    v = {}
  training = config.get("training")
  if training is None:
    training = {}
  n_points = int(v.get("n_points", 5))
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
  chosen = _select_points(traj["calls"], traj["reported"], n_points)
  run_dir = os.path.dirname(os.path.abspath(traj["path"]))  # the run's own directory: results.json + checkpoints/
  out_dir = output if output is not None else run_dir
  os.makedirs(out_dir, exist_ok=True)

  master = np.random.SeedSequence(int(seed))
  index_seq = master.spawn(1)[0]  # the shared 6:2:2 event split
  template_seq = master.spawn(1)[0]  # the throwaway template regressor (architecture only)
  train_index, val_index, test_index = _split_indices(detector.size(), budget, index_seq)

  # Three disjoint fixed-design buffers of raw (event, mask, target) rows, allocated once and fully
  # overwritten per trajectory point (the design is fixed per point -> no per-event design column;
  # ``combine_scaled`` runs per batch with the point's theta).
  M = int(jax.tree.leaves(detector.event_spec())[0].shape[0])  # per-hit count
  specs = (detector.event_spec(), jax.ShapeDtypeStruct((M, ), jnp.int32), detector.target_spec())
  train_buf = RingBuffer(len(train_index), specs, device=device)
  val_buf = RingBuffer(len(val_index), specs, device=device)
  test_buf = RingBuffer(len(test_index), specs, device=device)
  steps_per_epoch = max(1, len(train_index) // batch)
  scan_steps = val_every_epochs * steps_per_epoch  # SGD steps folded into one train_epoch call

  # Template regressor: the graphdef (architecture) is shared by every point, so the JIT kernels
  # compile once; each point re-initialises fresh params below.
  model = detopt.nn.from_config(detector, config=config["regressor"], rngs=nnx.Rngs(_key(template_seq)))
  reg_def = nnx.split(model, nnx.Param, nnx.Variable)[0]
  members = model.ensemble()
  # Single-cycle cosine-decayed learning rate over the whole per-point training (peak -> ~0),
  # mirroring FullBudgetTrainer; each point's fresh ``opt.init`` restarts the cycle.
  opt_name, opt_args = split(config["training"]["optimizer"])
  opt_args = dict(opt_args)
  schedule = optax.cosine_decay_schedule(init_value=opt_args.pop("learning_rate"), decay_steps=epochs * steps_per_epoch)
  opt = getattr(optax, opt_name)(learning_rate=schedule, **opt_args)
  draw = (members or 1) * batch

  def fill(buf, theta, event_index, desc):
    """(Re)fill ``buf`` with the events at ``event_index`` simulated at the FIXED scaled design
    ``theta``; pushing exactly ``capacity`` rows overwrites the whole ring. STRICTLY SERIAL: the
    propagation engine fills detector-owned buffers in place, so concurrent calls on one detector
    instance corrupt each other (verified: threaded fills produce NaN losses)."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    # One decoded physical design serves every full-size chunk (theta is fixed within a fill).
    phys_full = detector.to_nominal(jnp.broadcast_to(theta[None, :], (sample_batch, design_dim)))
    bar = tqdm(total=n, desc=desc, disable=progress != "bar")
    for o in range(0, n, sample_batch):
      idx = event_index[o:o + sample_batch]
      k = idx.shape[0]
      phys = phys_full if k == sample_batch else detector.to_nominal(jnp.broadcast_to(theta[None, :], (k, design_dim)))
      _gt, event, mask, target = detector(phys, idx)
      buf.push(event, mask, target)
      bar.update(k)
    bar.close()

  def _net_loss(params, state, drop_key, theta, event_b, mask_b, target_b):
    reg = nnx.merge(reg_def, params, state)
    feats = detector.combine_scaled(event_b, theta, mask=mask_b)  # fixed design (scaled), per hit
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
  def train_epoch(params, state, opt_state, key, theta, event_buf, mask_buf, tgt_buf, n):
    """One validation interval -- ``val_every_epochs`` epochs of scan-folded SGD over the train buffer at
    the fixed ``theta``. Returns the per-step batch losses (the caller averages them)."""

    def step(carry, k):
      params, state, opt_state = carry
      k_idx, k_drop = jax.random.split(k)
      idx = jax.random.randint(k_idx, (draw, ), 0, n)
      event_b = jax.tree.map(lambda a: a[idx], event_buf)
      target_b = jax.tree.map(lambda a: a[idx], tgt_buf)
      (loss, state), grads = jax.value_and_grad(_net_loss,
                                                has_aux=True)(params, state, k_drop, theta, event_b, mask_buf[idx], target_b)
      updates, opt_state = opt.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return (params, state, opt_state), loss

    (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), jax.random.split(key, scan_steps))
    return params, state, opt_state, losses

  @partial(jax.jit, static_argnames="rows")
  def evaluate(params, state, theta, event_buf, mask_buf, tgt_buf, rows):
    """One sequential batch-by-batch pass over the WHOLE buffer on the ENSEMBLE-MEAN prediction.
    The scan returns per-sample losses/metrics; the first ``rows`` are averaged, with the standard
    error of the loss mean."""
    reg = nnx.merge(reg_def, params, state)
    n_batches = -(-rows // eval_batch)  # ceil; the clipped tail past ``rows`` is sliced off below
    pool_rows = jax.tree.leaves(event_buf)[0].shape[0]

    def step(_, c):
      idx = jnp.clip(c * eval_batch + jnp.arange(eval_batch, dtype=jnp.int32), 0, pool_rows - 1)
      ev = jax.tree.map(lambda a: a[idx], event_buf)
      m = mask_buf[idx]
      feats = detector.combine_scaled(ev, theta, mask=m)
      emask = detector.element_mask(ev, m)
      tnorm = detector.normalize_target(jax.tree.map(lambda a: a[idx], tgt_buf))
      pred = _predict_shared(reg, feats, emask, members)
      return None, (detector.loss(pred, tnorm), detector.metric(pred, tnorm))

    _, (losses, metrics) = jax.lax.scan(step, None, jnp.arange(n_batches))
    losses = losses.reshape(-1)[:rows]  # per-sample losses over the whole buffer
    out = {k: jnp.mean(metrics[k].reshape(-1)[:rows]) for k in labels}
    out.update(loss=jnp.mean(losses), loss_sem=jnp.std(losses) / jnp.sqrt(rows))
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
    "init": "checkpoint",  # the run's own network for this design, continued on fresh data
    "lr_schedule": "cosine",
  }
  record = {**settings, "reported": {"calls": traj["calls"].tolist(), "loss": traj["reported"].tolist()}, "points": []}
  json_path = os.path.join(out_dir, "verification.json")
  plots_dir = os.path.join(out_dir, "plots")
  os.makedirs(plots_dir, exist_ok=True)

  # Resume: points already verified under IDENTICAL settings are reused, never recomputed; anything
  # else in the file is superseded (the settings drive the result, so a mismatch means stale data).
  # --force skips the reuse entirely: every point is recomputed and the file overwritten.
  if force:
    print("--force: recomputing every point, ignoring anything already in verification.json", flush=True)
  elif os.path.exists(json_path):
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
        _plot_learning(pt["history"], pt["reported_loss"], pt["test_loss"], pt["test_sem"], p, plot_path)
      print(
        f"[point {rank + 1}/{len(chosen)}] iteration {p} already verified: test={pt['test_loss']:.4f} -- skipped", flush=True
      )
      continue
    theta = jnp.asarray(traj["scaled"][p], jnp.float32)  # exactly what the optimizer evaluated
    phys_flat = np.asarray(detector.flatten_design(detector.to_nominal(theta)), np.float32)
    reported = float(traj["reported"][p])
    print(
      f"\n[point {rank + 1}/{len(chosen)}] iteration {p} @ {traj['calls'][p]:.0f} detector calls | "
      f"reported={reported:.4f} | design={np.round(phys_flat, 3).tolist()}", flush=True,
    )

    fill(train_buf, theta, train_index, "sample train")
    fill(val_buf, theta, val_index, "sample val")
    fill(test_buf, theta, test_index, "sample test")

    # The network the run itself reported this design with -- its LAST checkpointed epoch, i.e. the
    # weights at the moment the trainer declared convergence. The verification continues THAT
    # network (the optimiser state is not checkpointed, so only its moments restart), so the two
    # numbers describe the same network and differ only in the data: all of it is fresh here.
    # ``from_config`` only supplies the architecture; the weights are replaced by the checkpoint's.
    point_model = detopt.nn.from_config(detector, config=config["regressor"], rngs=nnx.Rngs(_key(point_seq)))
    _, params, state = nnx.split(point_model, nnx.Param, nnx.Variable)
    pure_params, pure_state, ckpt_design = _restore_design_network(run_dir, p)
    if not np.allclose(np.asarray(ckpt_design["scaled"], np.float32), np.asarray(traj["scaled"][p], np.float32)):
      raise ValueError(f"iteration {p}: the checkpoint's design differs from the one in results.json")
    nnx.replace_by_pure_dict(params, pure_params)
    nnx.replace_by_pure_dict(state, pure_state)
    params, state = jax.device_put(params, device), jax.device_put(state, device)
    opt_state = opt.init(params)
    n_train = jnp.int32(len(train_buf))

    # Epoch 0 IS the restored network: scored on the fresh validation buffer before any training, so
    # the curve starts at what the reported network is worth on data it never saw -- and if training
    # on the fresh budget only makes it worse, best-val keeps the restored network.
    restored_val = float(evaluate(params, state, theta, *val_buf.buffers(), rows=len(val_buf))["loss"])
    best = (restored_val, 0, params, state)  # (val_loss, epoch, params, state)
    history = [[0, float("nan"), restored_val]]  # [epoch, train_loss (interval mean), val_loss]
    train_loss = float("nan")
    print(f"  restored from checkpoint: val={restored_val:.4f} (reported={reported:.4f})", flush=True)
    plot_path = os.path.join(plots_dir, f"verification_{p:03d}.png")
    epoch = 0
    while epoch < epochs:
      params, state, opt_state, losses = train_epoch(
        params, state, opt_state, _key(point_seq), theta, *train_buf.buffers(), n_train
      )
      epoch += val_every_epochs
      train_loss = float(jnp.mean(losses))
      val_loss = float(evaluate(params, state, theta, *val_buf.buffers(), rows=len(val_buf))["loss"])
      history.append([epoch, train_loss, val_loss])
      if val_loss < best[0]:
        best = (val_loss, epoch, params, state)
      # Live learning curves, refreshed after every eval epoch (fire-and-forget daemon render).
      threading.Thread(target=_plot_learning, args=(list(history), reported, None, None, p, plot_path), daemon=True).start()
      if progress != "none":
        print(f"  epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  best={best[0]:.4f}@{best[1]}", flush=True)

    best_val, best_epoch, best_params, best_state = best
    test = {k: float(x) for k, x in evaluate(best_params, best_state, theta, *test_buf.buffers(), rows=len(test_buf)).items()}
    print(
      f"  -> best val={best_val:.4f} (epoch {best_epoch})  TEST={test['loss']:.4f}±{test['loss_sem']:.4f}  "
      f"reported={reported:.4f}  delta(test-reported)={test['loss'] - reported:+.4f}", flush=True,
    )
    _plot_learning(history, reported, test["loss"], test["loss_sem"], p, plot_path)

    record["points"].append({
      "point": int(p),
      "detector_calls": float(traj["calls"][p]),
      "reported_loss": reported,
      "design_physical": phys_flat.tolist(),
      "design_scaled": np.asarray(theta).tolist(),
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


def _best_so_far(record):
  """``(calls, best)`` of the run's COMPUTED convergence curve -- the running minimum of the reported
  loss. What an optimization run claims is this curve, not the per-iteration losses: those are
  proposals, most of them deliberately bad, and plotting them buries every other line in the figure."""
  calls = np.asarray(record["reported"]["calls"], np.float64)
  return calls, np.minimum.accumulate(np.asarray(record["reported"]["loss"], np.float64))


def _plot_comparison(record, path):
  """The headline figure: the run's own COMPUTED convergence curve (dashed -- the running minimum of
  a possibly-biased objective) against the PROPER independent estimates (solid -- held-out test ±SEM
  of each verified incumbent), both against cumulative detector calls. The verified points are NOT
  made monotone: where they rise, the incumbent that the run believed in did not hold up."""
  from matplotlib.figure import Figure

  points = record["points"]
  calls = np.asarray([pt["detector_calls"] for pt in points], np.float64)
  fig = Figure(figsize=(9, 5.5))
  ax = fig.subplots(1, 1)
  ax.step(*_best_so_far(record), where="post", ls="--", color="0.45", lw=1.6, label="computed (BO objective, best so far)")
  ax.errorbar(
    calls, [pt["test_loss"] for pt in points], yerr=[pt["test_sem"] for pt in points], fmt="o-", ms=5, lw=1.8, capsize=3,
    color="tab:blue", label="proper estimate (held-out test)"
  )
  ax.set(title="computed vs proper loss estimates", xlabel="detector calls", ylabel="loss (normalized)", yscale="log")
  ax.grid(True, alpha=0.25)
  ax.legend(fontsize=9)
  fig.tight_layout()
  fig.savefig(path, dpi=140)


# Serialise plotting: matplotlib is not thread-safe, so the per-epoch daemon renders and the final
# synchronous one must not run concurrently (mirrors scripts/subgradient.py).
_PLOT_LOCK = threading.Lock()


def _plot_learning(history, reported, test_loss, test_sem, iteration, path):
  """Per-point learning curve: validation loss per epoch, starting at epoch 0 -- the RESTORED
  network, before any training on the fresh data -- with the run's reported loss as a reference
  line. Refreshed from a daemon thread after every eval epoch (no test line yet); the final
  synchronous render adds the TEST score (value ± SEM in the legend).

  The train loss is deliberately not drawn: it is the running per-member loss with dropout ACTIVE,
  an estimator that sits several SEM off the deterministic ensemble loss val and test are measured
  with, so putting the two curves on one axis compares nothing."""
  from matplotlib.figure import Figure

  h = np.asarray(history, np.float64)  # (n, 3): epoch, train (running, unplotted), val
  with _PLOT_LOCK:
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots(1, 1)
    ax.axhline(reported, ls=":", color="0.45", lw=1.5, label=f"reported by BO = {reported:.4f}")
    ax.plot(h[:, 0], h[:, 2], ".-", color="tab:orange", label="validation (epoch 0 = restored network)")
    if test_loss is not None:  # the per-epoch daemon render has no test score yet
      ax.axhline(test_loss, ls="--", color="tab:blue", lw=1.5, label=f"test = {test_loss:.4f} ± {test_sem:.4f}")
    ax.set(title=f"iteration {iteration}", xlabel="epoch", ylabel="loss (normalized)", yscale="log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def _plot(record, path):
  """The run's computed convergence curve (running minimum) with the verified scores of each
  incumbent on top, and next to it the paired difference verified - reported with its SEM: the
  systematic part of that difference is the objective's bias, the scatter is the run's own estimator
  noise. A third panel breaks the test metric into its components when there is more than one."""
  from matplotlib.figure import Figure

  points = record["points"]
  calls = np.asarray([pt["detector_calls"] for pt in points], np.float64)
  test = np.asarray([pt["test_loss"] for pt in points], np.float64)
  sem = np.asarray([pt["test_sem"] for pt in points], np.float64)
  delta = test - np.asarray([pt["reported_loss"] for pt in points], np.float64)
  names = [k for k in points[0]["test_metric"] if k != "loss"]
  n_panels = 3 if len(names) > 1 else 2
  fig = Figure(figsize=(6.5 * n_panels, 5.5))
  axes = fig.subplots(1, n_panels)

  axes[0].step(*_best_so_far(record), where="post", ls="--", color="0.45", lw=1.6, label="reported (best so far)")
  axes[0].plot(calls, [pt["val_loss"] for pt in points], "o--", ms=5, color="tab:orange", alpha=0.8, label="verified val")
  axes[0].errorbar(calls, test, yerr=sem, fmt="D-", ms=6, capsize=3, color="tab:blue", label="verified TEST")
  axes[0].set(title="independent verification", xlabel="detector calls", ylabel="loss (normalized)", yscale="log")
  axes[0].legend(fontsize=8)

  axes[1].axhline(0.0, color="0.45", lw=1.2)
  axes[1].errorbar(calls, delta, yerr=sem, fmt="o", ms=6, capsize=3, color="tab:red")
  axes[1].set(title=f"verified - reported (mean {delta.mean():+.4f})", xlabel="detector calls", ylabel="difference in loss")

  if n_panels == 3:
    for name in names:
      axes[2].plot(calls, [pt["test_metric"][name] for pt in points], ".-", label=name)
    axes[2].set(title="test metric per component", xlabel="detector calls", yscale="log")
    axes[2].legend(fontsize=8, ncol=2)
  for ax in axes:
    ax.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(path, dpi=140)


if __name__ == "__main__":
  import sys

  import gearup

  # gearup's CLI is ``key=value``; ``--force`` is the conventional spelling, translated here.
  arguments = ["force=yes" if a == "--force" else a for a in sys.argv[1:]]
  gearup.gearup(verify).with_config("config/bo.yaml")(arguments)
