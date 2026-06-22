"""Subgradient design optimization (no BO), modernized to the bo.py standards.

The detector design ``theta`` lives in **encoded** (N(0,1)) space and is optimized by
gradient descent with a small L2 regulariser. Because ``decode_design`` + ``combine`` are
differentiable, the design gradient is one ``jax.grad`` through ``combine`` -- no
finite-differences, no yaml-design layers.

Each design step (the loop):
  1. perturb theta -> ``decode_design`` -> ``detector(seed, phys)`` (sample ``samples`` events);
  2. ``normalize(X)`` -> ``combine(X_norm, theta_pert)`` -> ``normalize_target(targets)``;
  3. append the fresh ``samples`` to a historical replay RingBuffer (filled BEFORE training);
  4. train ``substeps`` ``jax.lax.scan``-folded SGD steps, each minibatch = ``batch`` rows from the
     current fresh buffer + ``batch`` from the historical ring;
  5. on a FRESH batch at the EXACT current theta, compute d(loss)/d(theta) through
     ``combine`` (events held fixed) + L2 reg, and step theta;
  6. periodically validate + checkpoint (network, theta, optimizer states; the ring is NOT saved --
     it regenerates as training resumes).

Config: ``config/subgradient.yaml`` (mirrors bo.yaml minus the ``bo:`` block).
"""

import json
import os
import threading
import time

import matplotlib

matplotlib.use("AGG")
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


import detopt
from detopt.utils.config import optimizer as make_optimizer, resolve_device
from detopt.utils.pools import RingBuffer


# Checkpointing is the uniform machinery in detopt.utils.io (save_checkpoint / restore_config /
# restore_checkpoint): the network + design + their optimizer states + the regressor config (so a
# resume rebuilds the exact architecture). The replay ring is NOT saved -- it regenerates on resume.


def _initial_theta(detector, config):
    """Initial encoded design: the config ``nominal_design`` for the stereo detector, or the
    detector's own nominal design (e.g. DebugDetector) when shapes don't match."""
    nd = config.get("nominal_design")
    if nd is None:
        nd = detector.get_current_design_array()
    return jnp.asarray(detector.encode_design(nd), jnp.float32)


# --------------------------------------------------------------------------- #
# Ensemble-aware forward (mirrors trainer/common.py): a (members*batch, ...) or
# (batch, ...) minibatch -> predictions (.., target_dim).
# --------------------------------------------------------------------------- #
def _forward(reg, feats, mask, members, batch, *, deterministic, rngs=None):
    """TRAIN path: a (members*batch, ...) minibatch -> (members*batch, T); each member
    gets its OWN slice (reshape)."""
    if members is None:
        return reg(feats, mask, deterministic=deterministic, rngs=rngs)
    feats_e = feats.reshape((members, batch) + feats.shape[1:])
    mask_e = mask.reshape((members, batch) + mask.shape[1:])
    pred = reg(feats_e, mask_e, deterministic=deterministic, rngs=rngs)  # (members, batch, T)
    return pred.reshape((members * batch,) + pred.shape[2:])


def _forward_shared(reg, feats, mask, members, *, deterministic):
    """EVAL / DESIGN-GRAD path: feed ONE batch to every member (broadcast). Returns
    ``(members, batch, T)`` for an ensemble, ``(batch, T)`` for a single net. Pair with
    ``_target_for`` to compare against the (broadcast) target."""
    if members is None:
        return reg(feats, mask, deterministic=deterministic)
    fe = jnp.broadcast_to(feats[None], (members,) + feats.shape)
    me = jnp.broadcast_to(mask[None], (members,) + mask.shape)
    return reg(fe, me, deterministic=deterministic)


def _target_for(targets_norm, members):
    """Broadcast a normalized target batch to match :func:`_forward_shared`'s output."""
    return targets_norm if members is None else jnp.broadcast_to(targets_norm[None], (members,) + targets_norm.shape)


def _key(seq):
    """A jax PRNGKey from the next child of a SeedSequence (advances ``seq`` deterministically)."""
    return jax.random.PRNGKey(int(seq.spawn(1)[0].generate_state(1)[0]))


def _jsonable(aux):
    """``aux`` (arrays + {name: array}) -> nested plain lists for ``json.dump``."""
    return {
        "train": np.asarray(aux["train"]).tolist(),
        "val": {name: np.asarray(arr).tolist() for name, arr in aux["val"].items()},
        "design": np.asarray(aux["design"]).tolist(),
        "design_meta": aux.get("design_meta", {}),
        "warmup": {name: np.asarray(arr).tolist() for name, arr in aux.get("warmup", {}).items()},
    }


def optimize(seed, output, progress=True, restore=True, **config):
    os.makedirs(output, exist_ok=True)
    device = resolve_device(config.get("device"))
    sg = config["subgradient"]
    epochs = int(sg["epochs"])
    steps = int(sg["steps"])  # design optimization steps per epoch
    substeps = int(sg["substeps"])  # regressor steps per design step (scan-folded)
    batch = int(sg["batch"])  # minibatch rows drawn PER SOURCE (fresh + ring) -> 2*batch trained on
    samples = int(sg["samples"])  # fresh events sampled into the current-step buffer each design step
    design_batch = int(sg["design_batch"])  # fresh-batch size for the design gradient
    design_eps = float(sg["design_eps"])
    ring_capacity = int(sg["ring_capacity"])  # historical ring buffer capacity (replay)
    val_batch = int(sg.get("validation_batch", design_batch))  # events simulated per validation batch
    val_batches = int(sg["validation_batches"])  # validation batches accumulated into the val buffer
    wu = sg.get("warmup", {})
    warmup_samples = int(wu.get("samples", 0))  # events sampled once into the warmup buffer (0 = no warmup)
    warmup_steps = int(wu.get("steps", 0))  # network-training steps over the warmup buffer (each scans `substeps`)
    warmup_design_eps = float(wu.get("design_eps", design_eps))  # per-event design spread for the warmup sample

    detector = detopt.detector.from_config(config["detector"])
    design_dim = int(detector.design_dim())
    labels = tuple(detector.metric_labels())  # per-axis target component names
    # Design-trajectory plot metadata (per-dof bounds + station/magnet geometry), stored in
    # aux so report() can plot without rebuilding the detector.
    design_meta = detopt.utils.viz.design.design_trajectory_meta(detector)
    # Three disjoint detector event pools: train the network on 'train', take the design gradient on
    # 'design', validate on held-out 'val' (each kept separate so the design step and validation see
    # events the network never trained on). Defaults to the detector's first three pool keys, clamped
    # if fewer are configured (single pool -> no separation).
    pool_keys = list(detector.pool_split)
    train_pool = sg.get("train_pool", pool_keys[0])
    design_pool = sg.get("design_pool", pool_keys[min(1, len(pool_keys) - 1)])
    val_pool = sg.get("val_pool", pool_keys[min(2, len(pool_keys) - 1)])

    # Deterministic seeding: a master SeedSequence -> one child for the network init,
    # then exactly one child per epoch (each epoch derives ALL its randomness from its
    # own child via _key/spawn). On resume we spawn-and-discard the children of already-
    # done epochs, so epoch e always sees the same seed regardless of where we stopped.
    master = np.random.SeedSequence(int(seed))
    init_seq = master.spawn(1)[0]
    warmup_seq = master.spawn(1)[0]  # always spawned (consistent stream); used only on a fresh run
    prefill_seq = master.spawn(1)[0]  # always spawned (consistent stream); fills the ring before every run
    rngs = nnx.Rngs(jax.random.PRNGKey(int(init_seq.generate_state(1)[0])))

    # --- build the (ensemble) regressor + optimizers --------------------------
    # Architecture from the CHECKPOINT on resume (config files drift), from the config on a fresh run
    # (then saved into the checkpoint). The manager is created here so the stored config is available
    # before the model is built.
    manager = detopt.utils.io.get_checkpointer(output)
    resuming = restore and manager.latest_step() is not None
    stored = detopt.utils.io.restore_config(manager) if resuming else None
    if resuming and stored is None:
        print("warning: checkpoint predates config-saving; using the config-file regressor architecture")
    regressor_config = stored["regressor"] if stored is not None else config["regressor"]
    model = detopt.nn.from_config(detector, config=regressor_config, rngs=rngs)
    reg_def, params0, state0 = nnx.split(model, nnx.Param, nnx.Variable)
    members = model.ensemble() if hasattr(model, "ensemble") else None
    reg_opt = make_optimizer(config["training"]["optimizer"])
    design_opt = make_optimizer(config["training"]["design_optimizer"])  # adamw weight_decay = the design L2 reg

    # --- rings of RAW examples (event, mask, target, per-event PHYSICAL design) ----
    # Store raw records; combine per batch in the kernels (the combined features are wide and
    # would bloat the ring). Each row keeps the physical design it was generated under, so a
    # mixed-design replay batch re-combines correctly. The design gradient instead uses
    # combine_encoded with the LIVE theta on a fresh batch (see design_step).
    M = int(jax.tree.leaves(detector.event_spec())[0].shape[0])  # per-hit count
    raw_specs = (
        detector.event_spec(),
        jax.ShapeDtypeStruct((M,), jnp.int32),
        detector.target_spec(),
        detector.design_spec(),  # raw per-event physical Design record
    )
    ring = RingBuffer(ring_capacity, raw_specs, device=device)
    val_ring = RingBuffer(val_batches * val_batch, raw_specs, device=device)

    def _sample_raw(seq, theta_pert, pool):
        """Sample raw events at the (per-event perturbed) encoded design ``theta_pert``; return
        ``(event, mask, target, design)`` -- the per-event physical ``Design`` record completes the row."""
        phys = detector.decode_design(theta_pert)  # Design namedtuple (batched)
        _gt, event, mask, target = detector(seq.spawn(1)[0], phys, pool=pool)
        return event, mask, target, phys

    # --- jitted kernels -------------------------------------------------------
    def _net_loss(params, state, drop_key, event, mask, target, design, count):
        reg = nnx.merge(reg_def, params, state)
        feats = detector.combine(event, design)  # per-event PHYSICAL design -> encode + gather
        pred = _forward(reg, feats, mask, members, count, deterministic=False, rngs=nnx.Rngs(drop_key))
        loss = jnp.mean(detector.loss(pred, detector.normalize_target(target)))
        _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
        return loss, new_state

    members_eff = members or 1
    draw = members_eff * batch  # rows drawn from EACH source (fresh, ring) per member-step

    def _interleave(a, b):
        """Combine per-source gathers ``a``/``b`` (each ``(members_eff*batch, ...)``, member-major)
        into ``(members_eff*2*batch, ...)`` so every member sees ``batch`` fresh + ``batch`` ring
        rows (the layout :func:`_forward` reshapes back to ``(members_eff, 2*batch, ...)``)."""
        a = a.reshape((members_eff, batch) + a.shape[1:])
        b = b.reshape((members_eff, batch) + b.shape[1:])
        return jnp.concatenate([a, b], axis=1).reshape((members_eff * 2 * batch,) + a.shape[2:])

    @jax.jit
    def train_steps(params, state, opt_state, key, fresh, ring, n_ring):
        """``substeps`` scan-folded SGD steps; each minibatch is ``batch`` rows from the current
        ``fresh`` buffer + ``batch`` from the historical ``ring`` (per member). ``fresh``/``ring``
        are each ``(event, mask, target, design)`` raw records; combine runs in the loss."""
        f_event, f_mask, f_tgt, f_des = fresh
        r_event, r_mask, r_tgt, r_des = ring
        n_fresh = jax.tree.leaves(f_event)[0].shape[0]

        def step(carry, k):
            params, state, opt_state = carry
            k_f, k_r, k_drop = jax.random.split(k, 3)
            i_f = jax.random.randint(k_f, (draw,), 0, n_fresh)
            i_r = jax.random.randint(k_r, (draw,), 0, jnp.maximum(n_ring, 1))
            take = lambda buf, idx: jax.tree.map(lambda a: a[idx], buf)
            event = jax.tree.map(_interleave, take(f_event, i_f), take(r_event, i_r))
            mask = _interleave(f_mask[i_f], r_mask[i_r])
            target = jax.tree.map(_interleave, take(f_tgt, i_f), take(r_tgt, i_r))
            design = jax.tree.map(_interleave, take(f_des, i_f), take(r_des, i_r))
            (loss, new_state), grads = jax.value_and_grad(_net_loss, has_aux=True)(
                params, state, k_drop, event, mask, target, design, 2 * batch
            )
            updates, opt_state = reg_opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, state, opt_state), loss

        (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), jax.random.split(key, substeps))
        return params, state, opt_state, losses

    @jax.jit
    def design_step(theta, design_opt_state, event_f, mask_f, target_f, params, state):
        """One analytic design update: d(loss)/d(theta) through ``combine_encoded`` on a FRESH batch
        (events held fixed) at the EXACT current encoded ``theta`` + the design optimizer step (its
        weight_decay is the L2 reg). Returns updated ``(theta, design_opt_state)`` plus ``(loss, grad)``."""

        def design_loss(theta):
            reg = nnx.merge(reg_def, params, state)
            feats = detector.combine_encoded(event_f, theta)  # (design_batch, M, F) -- only theta-path
            tnorm = detector.normalize_target(target_f)  # (design_batch, T)
            pred = _forward_shared(reg, feats, mask_f, members, deterministic=True)
            return jnp.mean(detector.loss(pred, _target_for(tnorm, members)))

        loss, dgrad = jax.value_and_grad(design_loss)(theta)
        updates, design_opt_state = design_opt.update(dgrad, design_opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, design_opt_state, loss, dgrad

    @jax.jit
    def validate(params, state, event_buf, mask_buf, tgt_buf, des_buf):
        """Scan-folded validation metric over the RAW validation buffer (combine runs per chunk).
        The scan chunk is one simulation batch -- ``(val_batches, val_batch, ...)``. Accumulates the
        detector's per-sample ``metric`` dict and averages over the whole buffer. Returns
        ``{metric_key: scalar}`` keyed by ``labels``."""
        reg = nnx.merge(reg_def, params, state)
        reshape = lambda a: a.reshape((val_batches, val_batch) + a.shape[1:])
        event_b = jax.tree.map(reshape, event_buf)
        mask_b = reshape(mask_buf)
        tgt_b = jax.tree.map(reshape, tgt_buf)
        des_b = jax.tree.map(reshape, des_buf)

        def step(acc, chunk):
            ev, m, t, d = chunk
            feats = detector.combine(ev, d)
            pred = _forward_shared(reg, feats, m, members, deterministic=True)  # (members, val_batch, T) or (val_batch, T)
            md = detector.metric(pred, _target_for(detector.normalize_target(t), members))
            return {k: acc[k] + jnp.sum(md[k]) for k in acc}, None

        init = {name: jnp.float32(0.0) for name in labels}
        acc, _ = jax.lax.scan(step, init, (event_b, mask_b, tgt_b, des_b))
        count = val_batches * val_batch * (members or 1)
        return {k: acc[k] / count for k in labels}

    # --- state init / resume --------------------------------------------------
    theta = _initial_theta(detector, config)
    params, state = params0, state0
    reg_opt_state = reg_opt.init(params)
    design_opt_state = design_opt.init(theta)
    starting_epoch = 0
    # Preallocated per-epoch history; only [:epoch+1] is filled/saved/plotted. The validation
    # metric is per-axis -> {component name: (epochs,) array}.
    train_losses = np.full(epochs, np.nan, np.float32)
    val_metrics = {name: np.full(epochs, np.nan, np.float32) for name in labels}
    designs = np.full((epochs, design_dim), np.nan, np.float32)
    # Warmup curve is fixed-length (one value per warmup step), filled once and then
    # carried verbatim through every snapshot (not sliced per epoch).
    warm_losses = np.full(warmup_steps, np.nan, np.float32)

    def snapshot(n):
        """A self-contained (copied) view of the first ``n`` epochs -- safe to hand to a
        plotting thread / checkpoint while the loop keeps writing."""
        return {
            "train": train_losses[:n].copy(),
            "val": {name: val_metrics[name][:n].copy() for name in labels},
            "design": designs[:n].copy(),
            "design_meta": design_meta,
            "warmup": {"regressor": warm_losses.copy()},
        }

    last = manager.latest_step()
    if last is not None and restore:
        restored = detopt.utils.io.restore_checkpoint(
            manager, last, regressor=(params0, state0, reg_opt), design=design_opt, aux=True)
        params, state, reg_opt_state = restored["regressor"]
        theta, design_opt_state = restored["design"]
        aux = restored["aux"]
        # The replay ring is not restored -- it starts empty and refills as training resumes.
        starting_epoch = int(last) + 1
        train_losses[:starting_epoch] = np.asarray(aux["train"])
        for name in labels:
            val_metrics[name][:starting_epoch] = np.asarray(aux["val"][name])
        designs[:starting_epoch] = np.asarray(aux["design"])
        if "warmup" in aux:  # carry the (already-done) warmup curve forward unchanged
            wr = np.asarray(aux["warmup"]["regressor"])
            warm_losses[: len(wr)] = wr
        print(f"resumed from epoch {last}")

    aux = snapshot(starting_epoch)
    # Spawn-and-discard the per-epoch children of already-done epochs so the seed stream
    # is identical to an uninterrupted run (epoch e always gets the same epoch_seq).
    for _ in range(starting_epoch):
        master.spawn(1)

    losses_path = os.path.join(output, "losses.png")

    # --- warmup (epoch 0 only): sample a fixed buffer once, pre-train the network ----
    # Reuses the same per-event design perturbation and the `substeps`-scan train kernel
    # as the main loop, just on a one-shot buffer. Skipped on resume.
    if starting_epoch == 0 and warmup_steps > 0 and warmup_samples > 0:
        warm_ring = RingBuffer(warmup_samples, raw_specs, device=device)
        filled = 0
        while filled < warmup_samples:
            chunk = min(samples, warmup_samples - filled)
            theta_pert = theta[None, :] + warmup_design_eps * jax.random.normal(_key(warmup_seq), (chunk, design_dim))
            warm_ring.push(*_sample_raw(warmup_seq, theta_pert, train_pool))
            filled += chunk

        # Pretrain on the one-shot warm buffer: it is both the "fresh" and the "ring" source.
        for s in tqdm(range(warmup_steps), desc="warmup"):
            params, state, reg_opt_state, wl = train_steps(
                params,
                state,
                reg_opt_state,
                _key(warmup_seq),
                warm_ring.buffers(),
                warm_ring.buffers(),
                jnp.int32(len(warm_ring)),
            )
            warm_losses[s] = float(jnp.mean(wl))

        aux = snapshot(starting_epoch)  # persist the warmup curve into aux (saved at epoch 0)
        _plot_warmup(aux, os.path.join(output, "warmup.png"))

    # --- pre-fill the replay ring to FULL capacity before training ------------
    # The ring is not checkpointed, so this runs every time (fresh start AND resume), sampled at the
    # current design (perturbed per-event). The ring is full from the first design step, then each
    # step overwrites the oldest `samples`.
    filled = 0
    while filled < ring_capacity:
        chunk = min(samples, ring_capacity - filled)
        theta_pert = theta[None, :] + design_eps * jax.random.normal(_key(prefill_seq), (chunk, design_dim))
        ring.push(*_sample_raw(prefill_seq, theta_pert, train_pool))
        filled += chunk

    # --- the optimization loop: `epochs` x (`steps` design steps each) --------
    for epoch in tqdm(range(starting_epoch, epochs)):
        t0 = time.time()
        epoch_seq = master.spawn(1)[0]  # this epoch's single seed; all draws derive from it
        step_losses = np.empty(steps, np.float32)
        for _step in tqdm(range(steps)):
            # (1-3) sample `samples` fresh events into the current-step buffer, then APPEND them to
            # the historical ring BEFORE training (so the ring is never empty and the current samples
            # count toward both sources -- "completely fair"). theta stays a jax array; numpy only at
            # the C-detector boundary (decode -> phys).
            theta_pert = theta[None, :] + design_eps * jax.random.normal(_key(epoch_seq), (samples, design_dim))
            fresh = _sample_raw(epoch_seq, theta_pert, train_pool)  # (event, mask, target, design_phys)
            ring.push(*fresh)

            # (4) network: `substeps` scan-folded SGD steps; each minibatch = `batch` from the
            # current `fresh` buffer + `batch` from the historical ring.
            params, state, reg_opt_state, losses = train_steps(
                params, state, reg_opt_state, _key(epoch_seq), fresh, ring.buffers(), jnp.int32(len(ring))
            )
            step_losses[_step] = float(jnp.mean(losses))

            # (5) design gradient + step on a FRESH batch at the EXACT current theta (jitted),
            # drawn from the separate 'design' pool (held out from network training). The design's
            # own physical-design column is unused -- the gradient path re-combines with live theta.
            theta_cur = jnp.broadcast_to(theta[None, :], (design_batch, design_dim))
            event_f, mask_f, target_f, _des = _sample_raw(epoch_seq, theta_cur, design_pool)
            theta, design_opt_state, _dloss, dgrad = design_step(
                theta, design_opt_state, event_f, mask_f, target_f, params, state
            )

        # (6) validation: refill the val buffer at the current theta (val_batches x val_batch
        # combined examples), then scan-fold per-axis predictions over the whole buffer.
        theta_v = jnp.broadcast_to(theta[None, :], (val_batch, design_dim))  # theta fixed across the fill
        for _ in range(val_batches):
            val_ring.push(*_sample_raw(epoch_seq, theta_v, val_pool))
        val = validate(params, state, *val_ring.buffers())  # {metric_key: scalar}

        train_losses[epoch] = step_losses.mean()
        designs[epoch] = np.asarray(detector.flatten_design(detector.decode_design(theta)))
        for name in labels:
            val_metrics[name][epoch] = float(val[name])
        aux = snapshot(epoch + 1)

        # checkpoint every epoch (network + theta + optimizer states + regressor config; ring NOT saved)
        detopt.utils.io.save_checkpoint(
            manager,
            epoch,
            config={"regressor": regressor_config},
            regressor=(params, state, reg_opt_state),
            design=(theta, design_opt_state),
            aux=aux,
        )
        with open(os.path.join(output, "trajectory.json"), "w") as f:
            json.dump(_jsonable(aux), f)

        plot_async(aux, losses_path)

        if progress:
            vstr = " ".join(f"{name}={float(val[name]):.3f}" for name in labels)
            print(
                f"epoch {epoch + 1}/{epochs}  train={train_losses[epoch]:.5f}  "
                f"|dgrad|={float(jnp.linalg.norm(dgrad)):.2e}  z={np.round(designs[epoch][: detector.n_stations], 1)}\n"
                f"  val[{vstr}]  ({time.time() - t0:.2f}s)"
            )

    if starting_epoch < epochs:
        _plot(aux, losses_path)  # final synchronous plot (the per-epoch ones are daemon threads)
    manager.close()


def plot_async(aux, path):
    thread = threading.Thread(target=_plot, args=(aux, path), daemon=True)
    thread.start()
    return thread


# Serialise plotting: matplotlib (its mathtext/pyparsing layout in particular) is not
# thread-safe, so the per-epoch daemon _plot threads + the final synchronous one must not
# render concurrently. They still fire-and-forget; the lock just queues them.
_PLOT_LOCK = threading.Lock()


def _plot(aux, path):
    from matplotlib.figure import Figure

    with _PLOT_LOCK:
        fig = Figure(figsize=(14, 10))
        top, bottom = fig.subfigures(2, 1)  # top: losses; bottom: one subplot per design key
        ax_train, ax_val = top.subplots(1, 2)

        train = np.asarray(aux["train"])
        ax_train.plot(np.arange(len(train)), train, ".-")
        ax_train.set(title="network training loss", xlabel="epoch", yscale="log", ylabel="MSE")

        for name, arr in aux["val"].items():
            arr = np.asarray(arr)
            ax_val.plot(np.arange(len(arr)), arr, ".-", label=name)
        ax_val.set(title="validation MSE (normalized)", xlabel="epoch", yscale="log")
        ax_val.legend(fontsize=8, ncol=2)

        detopt.utils.viz.design.plot_design_trajectory(bottom, np.asarray(aux["design"]), aux.get("design_meta"))

        fig.savefig(path)


def _plot_warmup(aux, path):
    """Separate one-shot warmup plot: regressor MSE vs warmup step."""
    from matplotlib.figure import Figure

    reg = np.asarray((aux.get("warmup") or {}).get("regressor", []))
    if reg.size == 0:
        return

    with _PLOT_LOCK:
        fig = Figure(figsize=(9, 5))
        ax = fig.subplots(1, 1)
        ax.plot(np.arange(len(reg)), reg, ".-", label="regressor MSE")
        ax.set(title="warmup network training", xlabel="warmup step", yscale="log")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path)


def report(seed, output, report=None, **config):
    out = report or output
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(output, "trajectory.json")) as f:
        aux = json.load(f)
    _plot(aux, os.path.join(out, "losses.png"))
    _plot_warmup(aux, os.path.join(out, "warmup.png"))


if __name__ == "__main__":
    import gearup

    gearup.gearup(optimize=optimize, report=report).with_config("config/subgradient.yaml")()
