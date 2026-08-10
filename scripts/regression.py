"""Fixed-design regressor training + validation, for ONE design.

Train a single regressor on a large pool of (precomputed-MC-backed) events simulated under one fixed
design, then report held-out validation metrics. No discriminator, no design gradient -- just "how
well can the regressor reconstruct the target at THIS design?". Train and validation events come from
disjoint detector pools.

The design is ALWAYS the config ``design`` (the shared physical-design dict, e.g. ``initial_stereo``);
it is never read from a checkpoint. The regressor itself is persistent: pass ``checkpoint=<dir>`` to
save the regressor (parameters + state + optimizer) into that checkpoint every epoch and resume from
it on the next run, so training accumulates across invocations.

To COMPARE two designs, point ``design:`` at two different design configs (same ``seed``, so both
regressors share their init and event RNG stream and the only difference is the design) and give each
its own checkpoint dir:

    python scripts/regression.py seed=0 checkpoint=output/a design=initial_stereo
    python scripts/regression.py seed=0 checkpoint=output/b design=some_other_design

Everything a run produces lands INSIDE its checkpoint dir: the regressor checkpoint, ``losses.png``
(training-loss + validation curves, refreshed every epoch) and a final ``report.yaml`` (per-component
validation MSE + the decoded physical design), so the two runs can be diffed directly.

Config: ``config/regression.yaml`` -- ``detector``/``design`` are shared references (``detector: stereo``
-> ``config/detector/stereo.yaml``; ``design: initial_stereo`` -> ``config/design/initial_stereo.yaml``)
so they never drift; ``regressor``, ``optimizer`` and the ``training:``/``validation:`` blocks
(``batch``/``epochs``/``samples``) are this process's own knobs.
"""

import os

import yaml

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
from detopt.utils.events import shuffled_event_index, split_disjoint


# --------------------------------------------------------------------------- #
# Ensemble-aware forward (mirrors scripts/verify_regressor.py / lfi.py).
# --------------------------------------------------------------------------- #
def _forward(reg, feats, mask, members, batch, *, deterministic, rngs=None):
    """TRAIN path: a ``(members*batch, ...)`` minibatch -> ``(members*batch, T)`` (each member its slice)."""
    if members is None:
        return reg(feats, mask, deterministic=deterministic, rngs=rngs)
    feats_e = feats.reshape((members, batch) + feats.shape[1:])
    mask_e = mask.reshape((members, batch) + mask.shape[1:])
    pred = reg(feats_e, mask_e, deterministic=deterministic, rngs=rngs)
    return pred.reshape((members * batch,) + pred.shape[2:])


def _forward_shared(reg, feats, mask, members, *, deterministic):
    """EVAL path: feed ONE batch to every member (broadcast) -> ``(members, batch, T)`` or ``(batch, T)``."""
    if members is None:
        return reg(feats, mask, deterministic=deterministic)
    fe = jnp.broadcast_to(feats[None], (members,) + feats.shape)
    me = jnp.broadcast_to(mask[None], (members,) + mask.shape)
    return reg(fe, me, deterministic=deterministic)


def _forward_loss(reg, loss_fn, feats, mask, target, members, batch, *, deterministic, rngs=None):
    """TRAIN loss path: ``(members*batch, ...)`` minibatch -> per-sample loss ``(members*batch,)``.
    The MODEL owns the forward (``reg.loss``); each member sees its own slice (mirrors ``_forward``).
    ``mask`` is the per-ELEMENT mask; ``target`` is already normalised."""
    if members is None:
        return reg.loss(loss_fn, feats, mask, target, deterministic=deterministic, rngs=rngs)
    feats_e = feats.reshape((members, batch) + feats.shape[1:])
    mask_e = mask.reshape((members, batch) + mask.shape[1:])
    target_e = target.reshape((members, batch) + target.shape[1:])
    loss = reg.loss(loss_fn, feats_e, mask_e, target_e, deterministic=deterministic, rngs=rngs)
    return loss.reshape((members * batch,) + loss.shape[2:])


def _forward_shared_loss(reg, loss_fn, feats, mask, target, members, *, deterministic):
    """EVAL / DESIGN-GRAD loss path: ONE batch fed to every member (broadcast) -> per-sample loss
    ``(members, batch)`` or ``(batch,)``. The target is broadcast here (replaces ``_target_for``)."""
    if members is None:
        return reg.loss(loss_fn, feats, mask, target, deterministic=deterministic)
    fe = jnp.broadcast_to(feats[None], (members,) + feats.shape)
    me = jnp.broadcast_to(mask[None], (members,) + mask.shape)
    te = jnp.broadcast_to(target[None], (members,) + target.shape)
    return reg.loss(loss_fn, fe, me, te, deterministic=deterministic)


def _target_for(targets_norm, members):
    return targets_norm if members is None else jnp.broadcast_to(targets_norm[None], (members,) + targets_norm.shape)


def _key(seq):
    """A jax PRNGKey from the next child of a SeedSequence (advances ``seq`` deterministically)."""
    return jax.random.PRNGKey(int(seq.spawn(1)[0].generate_state(1)[0]))


def _sample_buffer(detector, theta, event_index, *, sampling_batch, device, progress):
    """Simulate the events at ``event_index`` at the FIXED design ``theta`` into a buffer (raw event,
    mask, raw target). ``combine_scaled`` + ``normalize_target`` run per batch in the train/eval kernels
    -- the buffer never holds the (wide) combined features."""
    design_dim = detector.design_dim()
    M = int(jax.tree.leaves(detector.event_spec())[0].shape[0])  # per-hit count
    specs = (detector.event_spec(), jax.ShapeDtypeStruct((M,), jnp.int32), detector.target_spec())

    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    buf = RingBuffer(n, specs, device=device)
    filled = 0
    bar = tqdm(total=n, desc="sample", disable=not progress)
    while filled < n:
        chunk = min(sampling_batch, n - filled)
        idx = event_index[filled:filled + chunk]
        phys = detector.to_nominal(jnp.broadcast_to(theta[None, :], (chunk, design_dim)))
        _gt, event, mask, target = detector(phys, idx)
        buf.push(event, mask, target)
        filled += chunk
        bar.update(chunk)
    bar.close()
    return buf


def regress(seed, checkpoint=None, restore=True, init_from=None, progress=True, **config):
    device = resolve_device(config.get("device"))
    train_samples = config["training"]["samples"]
    batch = config["training"]["batch"]
    epochs = config["training"]["epochs"]
    val_samples = config["validation"]["samples"]
    eval_batch = config["validation"]["batch"]
    sampling_batch = config["sampling"]["batch"]
    steps_per_epoch = max(1, train_samples // batch)

    detector = detopt.detector.from_config(config["detector"])
    theta = jnp.asarray(detector.to_scaled(config["design"]), jnp.float32)  # FIXED design, always from config
    design_source = "initial design (config)"
    design_dim = detector.design_dim()
    labels = tuple(detector.metric_labels())

    master = np.random.SeedSequence(int(seed))
    rngs = nnx.Rngs(jax.random.PRNGKey(int(master.spawn(1)[0].generate_state(1)[0])))
    train_seq = master.spawn(1)[0]
    # The script owns the train/val split: a shuffled, DISJOINT pair of event-index sets over the budget
    # (warn + oversample by wrapping if train+val exceeds detector.size()).
    train_index, val_index = split_disjoint(
        shuffled_event_index(detector.size(), train_samples + val_samples, master.spawn(1)[0]),
        train_samples / max(train_samples + val_samples, 1),
    )

    # Persistent regressor: when a checkpoint dir is provided, resume the regressor (parameters +
    # state + optimizer) from it and keep appending epochs so training accumulates across runs. The
    # design is NOT restored -- it always comes from the config above. The REGRESSOR ARCHITECTURE,
    # however, is taken from the checkpoint when resuming (config files drift) and from the config
    # only on a fresh run -- where it is then saved into the checkpoint.
    manager = detopt.utils.io.get_checkpointer(checkpoint) if checkpoint else None
    resuming = manager is not None and restore and manager.latest_step() is not None
    # init_from: WARM-START the params from a DIFFERENT checkpoint -- distinct from resume (fresh
    # optimizer, step 0, the new `checkpoint` dir). Ignored when resuming. The architecture comes from
    # the init checkpoint (the warm-started params must match it).
    init_mgr = detopt.utils.io.get_checkpointer(init_from) if (init_from is not None and not resuming) else None
    stored = detopt.utils.io.restore_config(manager) if resuming else None
    if resuming and stored is None:
        print("warning: checkpoint predates config-saving; using the config-file regressor architecture")
    if stored is not None:
        regressor_config = stored["regressor"]
    elif init_mgr is not None:
        regressor_config = detopt.utils.io.restore_config(init_mgr)["regressor"]
    else:
        regressor_config = config["regressor"]

    model = detopt.nn.from_config(detector, config=regressor_config, rngs=rngs)
    reg_def, params, state = nnx.split(model, nnx.Param, nnx.Variable)
    members = model.ensemble() if hasattr(model, "ensemble") else None
    opt = make_optimizer(config["optimizer"], n_total_steps=steps_per_epoch * epochs)
    opt_state = opt.init(params)

    step_offset = 0
    if resuming:
        last = manager.latest_step()
        params, state, opt_state = detopt.utils.io.restore_checkpoint(manager, last, regressor=(params, state, opt))["regressor"]
        step_offset = int(last)
        print(f"resumed persistent regressor from {checkpoint} (step {last})")
    elif init_mgr is not None:
        il = init_mgr.latest_step()
        params, state, _ = detopt.utils.io.restore_checkpoint(init_mgr, il, regressor=(params, state, opt))["regressor"]
        opt_state = opt.init(params)  # FRESH optimizer on the warm-started params (step stays 0)
        init_mgr.close()
        print(f"initialized regressor params from {init_from} (step {il}); fresh optimizer, training from step 0")

    def fill(event_index):
        return _sample_buffer(detector, theta, event_index, sampling_batch=sampling_batch, device=device, progress=progress)

    def _net_loss(params, state, drop_key, event_b, mask_b, target_b, count):
        reg = nnx.merge(reg_def, params, state)
        feats = detector.combine_scaled(event_b, theta, mask=mask_b)  # fixed design (scaled), per hit
        emask = detector.element_mask(event_b, mask_b)  # per-element mask (== hit mask, unless layer-wise)
        loss = jnp.mean(_forward_loss(reg, detector.loss, feats, emask, detector.normalize_target(target_b),
                                      members, count, deterministic=False, rngs=nnx.Rngs(drop_key)))
        _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
        return loss, new_state

    draw = (members or 1) * batch

    @jax.jit
    def train_epoch(params, state, opt_state, key, event_buf, mask_buf, tgt_buf, n):
        """One epoch = ``steps_per_epoch`` scan-folded AdamW gradient steps over the fixed training buffer."""

        def step(carry, k):
            params, state, opt_state = carry
            k_idx, k_drop = jax.random.split(k)
            idx = jax.random.randint(k_idx, (draw,), 0, n)
            event_b = jax.tree.map(lambda a: a[idx], event_buf)  # raw Event minibatch (pytree)
            target_b = jax.tree.map(lambda a: a[idx], tgt_buf)  # raw Target minibatch (pytree)
            (loss, new_state), grads = jax.value_and_grad(_net_loss, has_aux=True)(
                params, state, k_drop, event_b, mask_buf[idx], target_b, batch
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, state, opt_state), loss

        keys = jax.random.split(key, steps_per_epoch)
        (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), keys)
        return params, state, opt_state, losses

    @jax.jit
    def validate(params, state, event_buf, mask_buf, tgt_buf):
        """Per-key validation metric over the whole validation buffer: mean MSE plus the standard
        error of that mean (sqrt(var / N)), accumulated via sum and sum-of-squares."""
        reg = nnx.merge(reg_def, params, state)
        n_chunks = val_samples // eval_batch
        cut = n_chunks * eval_batch
        reshape = lambda a: a[:cut].reshape((n_chunks, eval_batch) + a.shape[1:])
        event_c = jax.tree.map(reshape, event_buf)
        mask_c = reshape(mask_buf)
        tgt_c = jax.tree.map(reshape, tgt_buf)

        def step(acc, chunk):
            s, ss = acc
            ev_chunk, m, t = chunk
            feats = detector.combine_scaled(ev_chunk, theta, mask=m)
            emask = detector.element_mask(ev_chunk, m)
            pred = _forward_shared(reg, feats, emask, members, deterministic=True)
            md = detector.metric(pred, _target_for(detector.normalize_target(t), members))
            s = {k: s[k] + jnp.sum(md[k]) for k in s}
            ss = {k: ss[k] + jnp.sum(jnp.square(md[k])) for k in ss}
            return (s, ss), None

        zero = lambda: {name: jnp.float32(0.0) for name in labels}
        (s, ss), _ = jax.lax.scan(step, (zero(), zero()), (event_c, mask_c, tgt_c))
        count = n_chunks * eval_batch * (members or 1)
        mean = {k: s[k] / count for k in labels}
        var = {k: jnp.maximum(ss[k] / count - jnp.square(mean[k]), 0.0) for k in labels}
        sem = {k: jnp.sqrt(var[k] / count) for k in labels}  # standard error of the mean MSE
        return mean, sem

    # --- sample the buffers, then train + validate ----------------------------
    train_buf = fill(train_index)
    val_buf = fill(val_index)
    n_train = jnp.int32(len(train_buf))

    physical = detector.flatten_design(detector.to_nominal(theta))
    print(f"decoded design: {detector.to_nominal(theta)}")
    print(
        f"train={train_samples} val={val_samples}  epochs={epochs} batch={batch} "
        f"steps/epoch={steps_per_epoch}  members={members}"
    )

    pbar = tqdm if progress else lambda x, **kwargs: x

    # Real-unit RMSE conversion is detector-owned (it knows its own normalization); fall back to {}
    # for detectors that don't expose it.
    real_rmse = getattr(detector, "metric_real_rmse", lambda means, errors=None: {})

    def build_report(final, final_sem, history):
        return {
            "seed": int(seed),
            "design_source": design_source,
            "design_physical": np.asarray(physical).tolist(),
            "design_scaled": np.asarray(theta).tolist(),
            "labels": list(labels),
            "final_validation": final,  # normalized per-component mean MSE
            "final_validation_sem": final_sem,  # standard error of each mean MSE (normalized)
            "final_validation_rmse": real_rmse(final, final_sem),  # real-unit RMSE +/- error (cm / GeV)
            "history": [{"epoch": e, "train": tr, "val": vv} for e, tr, vv in history],
        }

    ## l, _ = _net_loss(params, state, _key(train_seq), *train_buf.buffers(), n_train)

    history = []
    for epoch in pbar(range(1, epochs + 1)):
        params, state, opt_state, losses = train_epoch(params, state, opt_state, _key(train_seq), *train_buf.buffers(), n_train)
        train_loss = float(jnp.mean(losses))
        mean_val, sem_val = validate(params, state, *val_buf.buffers())
        final = {k: float(mean_val[k]) for k in labels}
        final_sem = {k: float(sem_val[k]) for k in labels}
        history.append((epoch, train_loss, final))

        if manager is not None:
            detopt.utils.io.save_checkpoint(manager, step_offset + epoch, config={"regressor": regressor_config},
                                            regressor=(params, state, opt_state))
            _plot(history, labels, os.path.join(checkpoint, "losses.png"))  # refresh the loss curves each epoch
            with open(os.path.join(checkpoint, "report.yaml"), "w") as f:  # refresh the report each epoch
                yaml.safe_dump(build_report(final, final_sem, history), f, sort_keys=False)

    print("\nfinal validation MSE (normalized):")
    for k in labels:
        print(f"  {k:>10s} = {final[k]:.5f} +/- {final_sem[k]:.5f}")
    rmse = real_rmse(final, final_sem)
    if rmse:
        print("\nfinal validation RMSE (real units):")
        for k, v in rmse.items():
            print(f"  {k:>10s} = {v['rmse']:.4g} +/- {v.get('error', float('nan')):.2g} {v['unit']}")

    if manager is not None:
        manager.close()
    return history


def _resolve_design(detector, design):
    """Scaled ``theta`` from a design given on the command line. ``design`` is either an
    already-resolved design dict (gearup expands the top-level ``design`` key through
    ``config/design/``) or a bare name / path, which we load from ``config/design/<name>.yaml``
    (a second comparison design is NOT a config key, so it arrives as a string we resolve here)."""
    if isinstance(design, str):
        path = design if os.path.exists(design) else os.path.join("config", "design", f"{design}.yaml")
        design = detopt.utils.config.load_config(path)
    return jnp.asarray(detector.to_scaled(design), jnp.float32)


def validate(seed, checkpoint, compare=None, design_b=None, progress=True, **config):
    """Load a persistent regressor from ``checkpoint``, sample a validation pool at its design,
    predict, and histogram the per-quantity prediction errors in REAL units (cm / GeV) -- one
    subplot per quantity, written to ``<checkpoint>/errors.png``.

    The design is taken from the ARGUMENTS, never from the checkpoint: the primary design is the
    config ``design:`` (override with ``design=<name>``). Pass ``compare=<dir>`` to overlay a
    SECOND checkpoint on the same axes, with its own design ``design_b=<name>`` (defaults to the
    primary design). Both runs are sampled from the SAME event RNG stream, so the only difference
    between the two curves is the design + the trained regressor:

        python scripts/regression.py validate seed=0 checkpoint=output/a design=initial_stereo \\
            compare=output/b design_b=stereo-lfi-427
    """
    device = resolve_device(config.get("device"))
    val_samples = config["validation"]["samples"]
    eval_batch = config["validation"]["batch"]
    sampling_batch = config["sampling"]["batch"]

    detector = detopt.detector.from_config(config["detector"])
    master = np.random.SeedSequence(int(seed))
    rngs = nnx.Rngs(jax.random.PRNGKey(int(master.spawn(1)[0].generate_state(1)[0])))
    # Validation events: a single shuffled index set, SHARED across checkpoints -> the only difference
    # between the two curves is the design + the trained regressor.
    val_index = shuffled_event_index(detector.size(), val_samples, master.spawn(1)[0])
    n_chunks = val_samples // eval_batch
    pbar = tqdm if progress else lambda x, **kwargs: x

    def errors_for(ckpt, design):
        theta = _resolve_design(detector, design)
        manager = detopt.utils.io.get_checkpointer(ckpt)
        last = manager.latest_step()
        if last is None:
            raise SystemExit(f"no regressor checkpoint to validate in {ckpt}")
        # Architecture from the CHECKPOINT (config files drift); fall back to the config file for
        # checkpoints that predate config-saving.
        stored = detopt.utils.io.restore_config(manager)
        regressor_config = stored["regressor"] if stored is not None else config["regressor"]
        model = detopt.nn.from_config(detector, config=regressor_config, rngs=rngs)
        reg_def, params0, state0 = nnx.split(model, nnx.Param, nnx.Variable)
        members = model.ensemble() if hasattr(model, "ensemble") else None
        opt = make_optimizer(config["optimizer"], n_total_steps=1)  # only to shape the restored optimizer state
        params, state, _ = detopt.utils.io.restore_checkpoint(manager, last, regressor=(params0, state0, opt))["regressor"]
        manager.close()

        @jax.jit
        def predict(params, state, feats, mask):
            """Deterministic point prediction (ensemble members averaged) on a normalized batch."""
            reg = nnx.merge(reg_def, params, state)
            pred = _forward_shared(reg, feats, mask, members, deterministic=True)
            return pred if members is None else jnp.mean(pred, axis=0)

        buf = _sample_buffer(detector, theta, val_index, sampling_batch=sampling_batch, device=device, progress=progress)
        event_buf, mask_buf, tgt_buf = buf.buffers()

        def chunk_pred(i):
            sl = slice(i * eval_batch, (i + 1) * eval_batch)
            ev = jax.tree.map(lambda a: a[sl], event_buf)
            feats = detector.combine_scaled(ev, theta, mask=mask_buf[sl])
            emask = detector.element_mask(ev, mask_buf[sl])
            return np.asarray(predict(params, state, feats, emask))

        preds = [chunk_pred(i) for i in pbar(range(n_chunks), desc=f"predict[{os.path.basename(os.path.normpath(ckpt))}]")]
        cut = n_chunks * eval_batch
        tgt_norm = detector.normalize_target(jax.tree.map(lambda a: a[:cut], tgt_buf))
        errors = detector.prediction_errors(np.concatenate(preds), np.asarray(tgt_norm))
        print(f"\n{ckpt}  (step {last}, design {detector.to_nominal(theta)}):")
        for name, (err, unit) in errors.items():
            print(f"  {name:>10s}  bias={err.mean():+.4g}  std={err.std():.4g}  {unit}")
        return errors

    runs = [(checkpoint, config["design"])]
    if compare is not None:
        runs.append((compare, design_b if design_b is not None else config["design"]))

    # Ordered (label, errors) pairs -- NOT a dict: two runs on the same checkpoint+design share a
    # basename, and a dict would collapse them into one curve. Disambiguate any repeated label.
    bases = [os.path.basename(os.path.normpath(ckpt)) for ckpt, _ in runs]
    labels = [b if bases.count(b) == 1 else f"{b} #{bases[:i].count(b) + 1}" for i, b in enumerate(bases)]
    series = [(label, errors_for(ckpt, design)) for label, (ckpt, design) in zip(labels, runs)]

    path = os.path.join(checkpoint, "errors.png")
    _plot_errors(series, path)
    print(f"\nwrote {path}")
    return series


def _plot_errors(series, path):
    """One signed-residual (predicted - true) histogram per quantity, in real units, OVERLAID for
    every run in ``series`` -- an ORDERED list of ``(label, per-quantity {name: (errors, unit)})``
    pairs (a list, not a dict, so two runs sharing a checkpoint basename still draw as two curves).
    Histograms are drawn as STEP outlines (so overlaid curves stay legible) over common
    per-quantity bins and density-normalized so unequal sample counts compare directly; the legend
    carries each curve's bias (mean) and resolution (std)."""
    from matplotlib.figure import Figure

    names = list(series[0][1].keys())  # quantities (same across runs)
    ncol = 3
    nrow = -(-len(names) // ncol)
    fig = Figure(figsize=(4 * ncol, 3 * nrow))
    axes = fig.subplots(nrow, ncol, squeeze=False).ravel()
    for ax, name in zip(axes, names):
        unit = series[0][1][name][1]
        arrays = [(label, np.asarray(errors[name][0])) for label, errors in series]
        lo = min(a.min() for _, a in arrays)
        hi = max(a.max() for _, a in arrays)
        edges = np.linspace(lo, hi, 81)
        for label, err in arrays:
            ax.hist(
                err,
                bins=edges,
                histtype="step",
                linewidth=1.5,
                density=True,
                label=f"{label}: bias={err.mean():+.3g}, std={err.std():.3g}",
            )
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set(title=f"{name} [{unit}]", xlabel=f"predicted - true [{unit}]")
        ax.legend(fontsize=7)
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def _plot(history, labels, path):
    from matplotlib.figure import Figure

    epochs = [h[0] for h in history]
    train = [h[1] for h in history]
    fig = Figure(figsize=(11, 5))
    ax_train, ax_val = fig.subplots(1, 2)
    ax_train.plot(epochs, train, ".-")
    ax_train.set(title="regressor training loss", xlabel="epoch", ylabel="MSE", yscale="log")
    for name in labels:
        ax_val.plot(epochs, [h[2][name] for h in history], ".-", label=name)
    ax_val.set(title="validation MSE (normalized)", xlabel="epoch", yscale="log")
    ax_val.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    import gearup

    gearup.gearup(regress=regress, validate=validate).with_config("config/regression.yaml")()
