"""Likelihood-free (LFI) design optimization, structurally identical to
``scripts/subgradient.py`` -- the ONLY difference is how the design gradient is
computed.

The subgradient script descends an analytic ``d(loss)/d(theta)`` through
``combine`` on a fresh batch (events held fixed): it sees only how theta reshapes
the per-hit *features*, not how theta reshapes the *distribution* of the simulated
hits X. The LFI design step adds that missing simulator-path term via the
score-function (REINFORCE) estimator:

    d/dtheta E_{X ~ p(X|theta)}[ l(X, theta) ]
        = E[ d l / d theta ]                          (pathwise, through combine)
        + E[ (l - b) * grad_theta log p(X|theta) ]    (score function)

``grad_theta log p(X|theta)`` is estimated by an *unconditional* discriminator D
trained to separate the joint ``(X, theta)`` from the product
``(X, theta_shuffled)``; at the optimum ``D = log p(X|theta) - log p(X)``, so
``grad_theta D`` is the score. The surrogate whose theta-gradient is the estimator
above is

    L(theta) = mean( l(X, theta) ) + mean( D(X, theta) * stopgrad(l - b) ),

with ``b`` the batch-mean loss (a variance-reducing baseline). Everything else --
the per-event design perturbation, the RingBuffer of network-ready examples, the
scan-folded regressor training, the deterministic SeedSequence seeding, the orbax
checkpointing, validation and plotting -- mirrors ``subgradient.py``.

Config: ``config/lfi.yaml`` (mirrors ``subgradient.yaml``; the ``lfi:`` block
replaces ``subgradient:`` and a ``discriminator:`` net + optimizer are added).
"""

import json
import os
import threading
import time

import yaml

import matplotlib

matplotlib.use("AGG")
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import orbax.checkpoint as ocp

import detopt
from detopt.utils.config import optimizer as make_optimizer, resolve_device
from detopt.utils.pools import RingBuffer


# --------------------------------------------------------------------------- #
# Checkpointing (orbax): regressor + discriminator + design + their optimizer
# states. The replay rings are NOT saved -- they regenerate as training resumes.
# --------------------------------------------------------------------------- #
def _save(
    manager,
    step,
    *,
    reg_params,
    reg_state,
    reg_opt_state,
    disc_params,
    disc_state,
    disc_opt_state,
    theta,
    design_opt_state,
    aux,
):
    # The replay rings are NOT checkpointed -- they are regenerated as training resumes (the
    # resulting non-determinism vs an uninterrupted run is not critical).
    manager.save(
        step,
        args=ocp.args.Composite(
            regressor=ocp.args.PyTreeSave(
                {
                    "parameters": nnx.to_pure_dict(reg_params),
                    "state": nnx.to_pure_dict(reg_state),
                    "optimizer_state": jax.tree.leaves(reg_opt_state),
                }
            ),
            discriminator=ocp.args.PyTreeSave(
                {
                    "parameters": nnx.to_pure_dict(disc_params),
                    "state": nnx.to_pure_dict(disc_state),
                    "optimizer_state": jax.tree.leaves(disc_opt_state),
                }
            ),
            design=ocp.args.PyTreeSave({"theta": theta, "optimizer_state": jax.tree.leaves(design_opt_state)}),
            aux=ocp.args.PyTreeSave(aux),
        ),
    )


def _restore(manager, step, *, reg_params0, reg_state0, reg_opt, disc_params0, disc_state0, disc_opt, design_opt):
    data = manager.restore(
        step,
        args=ocp.args.Composite(
            regressor=ocp.args.PyTreeRestore(),
            discriminator=ocp.args.PyTreeRestore(),
            design=ocp.args.PyTreeRestore(),
            aux=ocp.args.PyTreeRestore(),
        ),
    )

    def load_net(params0, state0, opt, blob):
        params = nnx.eval_shape(lambda: params0)
        nnx.replace_by_pure_dict(params, blob["parameters"])
        state = nnx.eval_shape(lambda: state0)
        nnx.replace_by_pure_dict(state, blob["state"])
        opt_state = jax.tree.unflatten(jax.tree.structure(opt.init(params)), blob["optimizer_state"])
        return params, state, opt_state

    reg_params, reg_state, reg_opt_state = load_net(reg_params0, reg_state0, reg_opt, data["regressor"])
    disc_params, disc_state, disc_opt_state = load_net(disc_params0, disc_state0, disc_opt, data["discriminator"])
    theta = jnp.asarray(data["design"]["theta"], jnp.float32)
    design_opt_state = jax.tree.unflatten(jax.tree.structure(design_opt.init(theta)), data["design"]["optimizer_state"])
    return (
        reg_params,
        reg_state,
        reg_opt_state,
        disc_params,
        disc_state,
        disc_opt_state,
        theta,
        design_opt_state,
        data["aux"],
    )


def _initial_theta(detector, config):
    """Initial encoded design: ``encode_design`` of the resolved ``design`` config -- a named
    physical-design dict (e.g. ``config/design/initial_stereo.yaml``, gearup-resolved from
    ``design: initial_stereo``). The initial design is config-supplied, always: the detector
    has no default/current design to fall back to."""
    return jnp.asarray(detector.encode_design(config["design"]), jnp.float32)


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
        "disc": np.asarray(aux["disc"]).tolist(),
        "design_meta": aux.get("design_meta", {}),
        "warmup": {name: np.asarray(arr).tolist() for name, arr in aux.get("warmup", {}).items()},
    }


def _save_design_yaml(output, epoch, detector, theta):
    """Write the current physical design to ``<output>/designs/detector-lfi-<epoch>.yaml`` in the
    same bare-dict format as the ``config/design/*.yaml`` files (e.g. ``initial_stereo.yaml``), so a
    snapshot can be dropped straight back in as a ``design:`` config. Size-1 arrays unwrap to scalars."""
    design = detector.decode_design(jnp.asarray(theta, jnp.float32))  # {name: array}
    physical = {}
    for name, val in design.items():
        flat = np.asarray(val).reshape(-1)
        physical[name] = float(flat[0]) if flat.size == 1 else [float(x) for x in flat]
    designs_dir = os.path.join(output, "designs")
    os.makedirs(designs_dir, exist_ok=True)
    path = os.path.join(designs_dir, f"detector-lfi-{epoch}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(physical, f, sort_keys=False, default_flow_style=None)


def optimize(seed, output, progress=True, restore=True, **config):
    os.makedirs(output, exist_ok=True)
    device = resolve_device(config.get("device"))
    lfi = config["lfi"]
    epochs = int(lfi["epochs"])
    steps = int(lfi["steps"])  # design optimization steps per epoch
    substeps = int(lfi["substeps"])  # regressor (and discriminator) steps per design step (scan-folded)
    batch = int(lfi["batch"])  # minibatch rows drawn PER SOURCE (fresh + ring) -> 2*batch trained on
    samples = int(lfi["samples"])  # fresh events sampled into the current-step buffer each design step
    design_batch = int(lfi["design_batch"])  # fresh-batch size for the design gradient
    design_eps = float(lfi["design_eps"])
    ring_capacity = int(lfi["ring_capacity"])  # historical ring buffer capacity (replay)
    val_batch = int(lfi.get("validation_batch", design_batch))  # events simulated per validation batch
    val_batches = int(lfi["validation_batches"])  # validation batches accumulated into the val buffer
    wu = lfi.get("warmup", {})
    warmup_samples = int(wu.get("samples", 0))  # events sampled once into the warmup buffer (0 = no warmup)
    warmup_steps = int(wu.get("steps", 0))  # network-training steps over the warmup buffer (each scans `substeps`)
    warmup_design_eps = float(wu.get("design_eps", design_eps))  # per-event design spread for the warmup sample

    detector = detopt.detector.from_config(config["detector"])
    design_dim = int(detector.design_dim())
    labels = tuple(detector.metric_labels())  # per-axis target component names
    # Design-trajectory plot metadata (per-dof bounds + station/magnet geometry), stored in
    # aux so report() can plot without rebuilding the detector.
    design_meta = detopt.utils.viz.design.design_trajectory_meta(detector)
    # Three disjoint detector event pools: train the regressor/discriminator on 'train', take the
    # LFI design gradient on 'design', validate on held-out 'val' (each kept separate so the design
    # step and validation see events the nets never trained on). Defaults to the detector's first
    # three pool keys, clamped if fewer are configured (single pool -> no separation).
    pool_keys = list(detector.pool_split)
    train_pool = lfi.get("train_pool", pool_keys[0])
    design_pool = lfi.get("design_pool", pool_keys[min(1, len(pool_keys) - 1)])
    val_pool = lfi.get("val_pool", pool_keys[min(2, len(pool_keys) - 1)])

    # Deterministic seeding: a master SeedSequence -> one child for the network init,
    # then exactly one child per epoch (each epoch derives ALL its randomness from its
    # own child via _key/spawn). On resume we spawn-and-discard the children of already-
    # done epochs, so epoch e always sees the same seed regardless of where we stopped.
    master = np.random.SeedSequence(int(seed))
    init_seq = master.spawn(1)[0]
    warmup_seq = master.spawn(1)[0]  # always spawned (consistent stream); used only on a fresh run
    prefill_seq = master.spawn(1)[0]  # always spawned (consistent stream); fills the rings before every run
    rngs = nnx.Rngs(jax.random.PRNGKey(int(init_seq.generate_state(1)[0])))

    # --- build the (ensemble) regressor + the discriminator + optimizers ------
    model = detopt.nn.from_config(detector, config=config["regressor"], rngs=rngs)
    reg_def, reg_params0, reg_state0 = nnx.split(model, nnx.Param, nnx.Variable)
    members = model.ensemble() if hasattr(model, "ensemble") else None

    disc = detopt.nn.from_config(detector, config=config["discriminator"], rngs=rngs)
    disc_def, disc_params0, disc_state0 = nnx.split(disc, nnx.Param, nnx.Variable)

    reg_opt = make_optimizer(config["training"]["optimizer"])
    disc_opt = make_optimizer(config["training"]["discriminator_optimizer"])
    design_opt = make_optimizer(config["training"]["design_optimizer"])  # adamw weight_decay = the design L2 reg

    # --- rings ----------------------------------------------------------------
    # reg_ring: network-ready (combined features, mask, normalized target) -- filled from the
    # LOCAL design neighbourhood (theta0 +/- design_eps), since the regressor is the surrogate
    # near the current design.
    # disc_ring: raw normalized X + mask + the per-event [encoded theta | normalized
    # conditioning] packed in the third slot -- filled over the WHOLE encoded space
    # (theta ~ N(0,1)), so the discriminator learns the global X-on-theta ratio (its
    # theta-gradient then supplies the score at the current design). The joint/product
    # samples are drawn from this ring (real = matched (X, theta, gt); pseudo = a SEPARATE X'
    # with an INDEPENDENT theta', each X carrying its own conditioning gt).
    M, F = detector.combined_event_shape()
    Mn, Fn = detector.normalized_event_shape()
    target_dim = int(detector.target_shape()[0])
    cond_dim = int(detector.conditioning_dim())
    reg_ring = RingBuffer(ring_capacity, (M, F), (M,), (target_dim,), device=device)
    disc_ring = RingBuffer(ring_capacity, (Mn, Fn), (Mn,), (design_dim + cond_dim,), device=device)
    # A second, validation-only buffer for the regressor (refilled each epoch).
    val_ring = RingBuffer(val_batches * val_batch, (M, F), (M,), (target_dim,), device=device)

    # --- jitted kernels -------------------------------------------------------
    @jax.jit
    def prepare(X, targets, conditioning, theta_b):
        """Raw events + per-event encoded design -> (normalized X, combined features,
        normalized target, normalized conditioning)."""
        X_norm = detector.normalize(X)
        feats = detector.combine(X_norm, theta_b)
        return X_norm, feats, detector.normalize_target(targets), detector.normalize_conditioning(conditioning)

    @jax.jit
    def _prepare_disc(X, conditioning):
        return detector.normalize(X), detector.normalize_conditioning(conditioning)

    def _disc_batch(seq, n):
        """A discriminator-ring chunk sampled over the WHOLE encoded design space
        (theta ~ N(0,1)), not the local design neighbourhood: returns
        ``(X_norm, mask, [theta | cond_norm])`` ready to push into ``disc_ring``."""
        theta_g = jax.random.normal(_key(seq), (n, design_dim))  # whole space
        ev = detector.sample_events(seq.spawn(1)[0], detector.decode_design(theta_g), pool=train_pool)
        X_norm, cnorm = _prepare_disc(ev["X"], ev["conditioning"])
        return X_norm, ev["mask"], jnp.concatenate([theta_g, cnorm], axis=-1)

    def _net_loss(params, state, drop_key, feats, mask, targets_norm, count):
        reg = nnx.merge(reg_def, params, state)
        pred = _forward(reg, feats, mask, members, count, deterministic=False, rngs=nnx.Rngs(drop_key))
        loss = jnp.mean(detector.loss(pred, targets_norm))  # ring targets are already normalized
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
    def train_regressor(params, state, opt_state, key, fresh, ring, n_ring):
        """``substeps`` scan-folded SGD steps; each minibatch is ``batch`` rows from the current
        ``fresh`` buffer + ``batch`` from the historical ``ring`` (per member). ``fresh``/``ring``
        are each ``(feats, mask, tgt)``; the fresh buffer is always full, the ring grows to ``n_ring``."""
        f_feats, f_mask, f_tgt = fresh
        r_feats, r_mask, r_tgt = ring
        n_fresh = f_feats.shape[0]

        def step(carry, k):
            params, state, opt_state = carry
            k_f, k_r, k_drop = jax.random.split(k, 3)
            i_f = jax.random.randint(k_f, (draw,), 0, n_fresh)
            i_r = jax.random.randint(k_r, (draw,), 0, jnp.maximum(n_ring, 1))
            feats = _interleave(f_feats[i_f], r_feats[i_r])
            mask = _interleave(f_mask[i_f], r_mask[i_r])
            tgt = _interleave(f_tgt[i_f], r_tgt[i_r])
            (loss, new_state), grads = jax.value_and_grad(_net_loss, has_aux=True)(
                params, state, k_drop, feats, mask, tgt, 2 * batch
            )
            updates, opt_state = reg_opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, state, opt_state), loss

        (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), jax.random.split(key, substeps))
        return params, state, opt_state, losses

    def _disc_loss(params, state, key, real, pseudo):
        """BCE separating the joint (X, theta) | gt [real, label 1] from the product
        (X', theta') | gt' [pseudo, label 0]. ``real``/``pseudo`` are each
        ``(X_norm, mask, theta, cond)``; for the product, X' and theta' come from
        INDEPENDENT ring draws (a separately-simulated X' paired with an independent
        theta'), and gt' is X''s own conditioning -- NOT the same X recombined."""
        disc_m = nnx.merge(disc_def, params, state)
        k_r, k_p = jax.random.split(key)
        Xr, mr, thr, cr = real
        Xp, mp, thp, cp = pseudo
        logit_real = disc_m(detector.combine(Xr, thr), mr, cr, deterministic=False, rngs=nnx.Rngs(k_r))
        logit_pseudo = disc_m(detector.combine(Xp, thp), mp, cp, deterministic=False, rngs=nnx.Rngs(k_p))
        # Mean BCE over both classes; the 0.5 averages the two per-example terms so a random
        # discriminator reads ~log(2), not ~2 log(2).
        loss = 0.5 * jnp.mean(jax.nn.softplus(-logit_real) + jax.nn.softplus(logit_pseudo))
        _, _, new_state = nnx.split(disc_m, nnx.Param, nnx.Variable)
        return loss, new_state

    @jax.jit
    def train_discriminator(params, state, opt_state, key, fresh, ring, n_ring):
        """``substeps`` scan-folded steps; each draw is ``batch`` rows from the current ``fresh``
        buffer + ``batch`` from the historical ``ring`` (-> 2*batch real and 2*batch pseudo).
        ``fresh``/``ring`` are each ``(Xn, mask, pack)`` with pack row = [encoded theta | cond]."""
        f_X, f_mask, f_pack = fresh
        r_X, r_mask, r_pack = ring
        n_fresh = f_X.shape[0]

        def draw_both(k):
            """``batch`` from fresh + ``batch`` from ring -> concatenated ``(X, mask, theta, cond)``."""
            kf, kr = jax.random.split(k)
            i_f = jax.random.randint(kf, (batch,), 0, n_fresh)
            i_r = jax.random.randint(kr, (batch,), 0, jnp.maximum(n_ring, 1))
            X = jnp.concatenate([f_X[i_f], r_X[i_r]])
            mask = jnp.concatenate([f_mask[i_f], r_mask[i_r]])
            pack = jnp.concatenate([f_pack[i_f], r_pack[i_r]])
            return X, mask, pack[:, :design_dim], pack[:, design_dim:]

        def step(carry, k):
            params, state, opt_state = carry
            k_r, k_px, k_pt, k_loss = jax.random.split(k, 4)
            # real: matched (X, theta, gt). pseudo: X' (+its gt') from one draw, theta' from
            # an INDEPENDENT draw -> a genuine product sample, not a recombination of X.
            real = draw_both(k_r)
            Xp, mp, _thp, cp = draw_both(k_px)
            _Xt, _mt, thp, _ct = draw_both(k_pt)
            (loss, new_state), grads = jax.value_and_grad(_disc_loss, has_aux=True)(
                params, state, k_loss, real, (Xp, mp, thp, cp)
            )
            updates, opt_state = disc_opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, state, opt_state), loss

        (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), jax.random.split(key, substeps))
        return params, state, opt_state, losses

    @jax.jit
    def design_step(theta, design_opt_state, X_f, mask_f, targets_f, cond_f, r_params, r_state, d_params, d_state):
        """One LFI design update on a FRESH batch at the current theta (events held fixed).
        The surrogate's theta-gradient is the pathwise + score-function estimator; both the
        regressor loss and the discriminator logit depend on theta through ``combine``.
        Returns updated ``(theta, design_opt_state)`` plus ``(loss, grad)`` for logging."""

        def design_loss(theta):
            reg = nnx.merge(reg_def, r_params, r_state)
            disc_m = nnx.merge(disc_def, d_params, d_state)
            theta_b = jnp.broadcast_to(theta[None, :], (design_batch, theta.shape[0]))
            feats = detector.combine(detector.normalize(X_f), theta_b)  # (design_batch, M, F) -- only theta-path
            tnorm = detector.normalize_target(targets_f)  # (design_batch, T)
            cnorm = detector.normalize_conditioning(cond_f)  # (design_batch, C)

            pred = _forward_shared(reg, feats, mask_f, members, deterministic=True)
            loss_reg = detector.loss(pred, _target_for(tnorm, members))  # (members, B) or (B,)
            loss_ev = loss_reg if members is None else jnp.mean(loss_reg, axis=0)  # (B,) per-event

            logits = disc_m(feats, mask_f, cnorm, deterministic=True)  # (B,) grad_theta-estimate of log p(X|theta,gt)
            advantage = jax.lax.stop_gradient(loss_ev - jnp.mean(loss_ev))  # baseline = batch-mean loss
            score_term = jnp.mean(logits * advantage)

            return jnp.mean(loss_reg) + score_term

        loss, dgrad = jax.value_and_grad(design_loss)(theta)
        updates, design_opt_state = design_opt.update(dgrad, design_opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, design_opt_state, loss, dgrad

    @jax.jit
    def validate(params, state, feats_buf, mask_buf, tgt_buf):
        """Scan-folded validation metric over the (already-combined) validation buffer. The scan
        chunk is one simulation batch -- ``(val_batches, val_batch, ...)`` -- so the prediction
        granularity matches how the buffer was filled. Accumulates the detector's per-sample
        ``metric`` dict and averages over the whole buffer (mean of per-sample = MSE). Returns
        ``{metric_key: scalar}`` keyed by ``labels``."""
        reg = nnx.merge(reg_def, params, state)
        shape = lambda a: a.reshape((val_batches, val_batch) + a.shape[1:])
        feats_b, mask_b, tgt_b = shape(feats_buf), shape(mask_buf), shape(tgt_buf)

        def step(acc, chunk):
            f, m, t = chunk
            pred = _forward_shared(reg, f, m, members, deterministic=True)  # (members, val_batch, T) or (val_batch, T)
            md = detector.metric(pred, _target_for(t, members))  # {key: (members, val_batch) | (val_batch,)}
            return {k: acc[k] + jnp.sum(md[k]) for k in acc}, None

        init = {name: jnp.float32(0.0) for name in labels}
        acc, _ = jax.lax.scan(step, init, (feats_b, mask_b, tgt_b))
        count = val_batches * val_batch * (members or 1)
        return {k: acc[k] / count for k in labels}

    # --- state init / resume --------------------------------------------------
    manager = detopt.utils.io.get_checkpointer(output)
    theta = _initial_theta(detector, config)
    reg_params, reg_state = reg_params0, reg_state0
    disc_params, disc_state = disc_params0, disc_state0
    reg_opt_state = reg_opt.init(reg_params)
    disc_opt_state = disc_opt.init(disc_params)
    design_opt_state = design_opt.init(theta)
    starting_epoch = 0
    # Preallocated per-epoch history; only [:epoch+1] is filled/saved/plotted. The validation
    # metric is per-axis -> {component name: (epochs,) array}.
    train_losses = np.full(epochs, np.nan, np.float32)
    disc_losses = np.full(epochs, np.nan, np.float32)
    val_metrics = {name: np.full(epochs, np.nan, np.float32) for name in labels}
    designs = np.full((epochs, design_dim), np.nan, np.float32)
    # Warmup curves are fixed-length (one value per warmup step), filled once and then
    # carried verbatim through every snapshot (not sliced per epoch).
    warm_reg_losses = np.full(warmup_steps, np.nan, np.float32)
    warm_disc_losses = np.full(warmup_steps, np.nan, np.float32)

    def snapshot(n):
        """A self-contained (copied) view of the first ``n`` epochs -- safe to hand to a
        plotting thread / checkpoint while the loop keeps writing."""
        return {
            "train": train_losses[:n].copy(),
            "disc": disc_losses[:n].copy(),
            "val": {name: val_metrics[name][:n].copy() for name in labels},
            "design": designs[:n].copy(),
            "design_meta": design_meta,
            "warmup": {"regressor": warm_reg_losses.copy(), "discriminator": warm_disc_losses.copy()},
        }

    last = manager.latest_step()
    if last is not None and restore:
        (
            reg_params,
            reg_state,
            reg_opt_state,
            disc_params,
            disc_state,
            disc_opt_state,
            theta,
            design_opt_state,
            aux,
        ) = _restore(
            manager,
            last,
            reg_params0=reg_params0,
            reg_state0=reg_state0,
            reg_opt=reg_opt,
            disc_params0=disc_params0,
            disc_state0=disc_state0,
            disc_opt=disc_opt,
            design_opt=design_opt,
        )
        # The replay rings are not restored -- they start empty and refill as training resumes.
        starting_epoch = int(last) + 1
        train_losses[:starting_epoch] = np.asarray(aux["train"])
        disc_losses[:starting_epoch] = np.asarray(aux["disc"])
        for name in labels:
            val_metrics[name][:starting_epoch] = np.asarray(aux["val"][name])
        designs[:starting_epoch] = np.asarray(aux["design"])
        if "warmup" in aux:  # carry the (already-done) warmup curves forward unchanged
            wr, wd = np.asarray(aux["warmup"]["regressor"]), np.asarray(aux["warmup"]["discriminator"])
            warm_reg_losses[: len(wr)], warm_disc_losses[: len(wd)] = wr, wd
        print(f"resumed from epoch {last}")

    aux = snapshot(starting_epoch)
    # Spawn-and-discard the per-epoch children of already-done epochs so the seed stream
    # is identical to an uninterrupted run (epoch e always gets the same epoch_seq).
    for _ in range(starting_epoch):
        master.spawn(1)

    losses_path = os.path.join(output, "losses.png")

    # --- warmup (epoch 0 only): sample a fixed buffer once, pre-train both nets -----
    # Reuses the same per-event design perturbation and the `substeps`-scan train kernels
    # as the main loop, just on a one-shot buffer. Skipped on resume.
    if starting_epoch == 0 and warmup_steps > 0 and warmup_samples > 0:
        warm_reg = RingBuffer(warmup_samples, (M, F), (M,), (target_dim,), device=device)
        warm_disc = RingBuffer(warmup_samples, (Mn, Fn), (Mn,), (design_dim + cond_dim,), device=device)
        filled = 0
        while filled < warmup_samples:
            chunk = min(samples, warmup_samples - filled)
            # regressor: LOCAL design neighbourhood; discriminator: WHOLE space (theta ~ N(0,1)).
            theta_pert = theta[None, :] + warmup_design_eps * jax.random.normal(_key(warmup_seq), (chunk, design_dim))
            ev = detector.sample_events(warmup_seq.spawn(1)[0], detector.decode_design(theta_pert), pool=train_pool)
            _Xn, feats, tnorm, _cn = prepare(ev["X"], ev["targets"], ev["conditioning"], theta_pert)
            warm_reg.push(feats, ev["mask"], tnorm)
            warm_disc.push(*_disc_batch(warmup_seq, chunk))
            filled += chunk

        # Pretrain on the one-shot warm buffer: it is both the "fresh" and the "ring" source.
        for s in tqdm(range(warmup_steps), desc="warmup"):
            reg_params, reg_state, reg_opt_state, rl = train_regressor(
                reg_params,
                reg_state,
                reg_opt_state,
                _key(warmup_seq),
                warm_reg.buffers(),
                warm_reg.buffers(),
                jnp.int32(len(warm_reg)),
            )
            disc_params, disc_state, disc_opt_state, dl = train_discriminator(
                disc_params,
                disc_state,
                disc_opt_state,
                _key(warmup_seq),
                warm_disc.buffers(),
                warm_disc.buffers(),
                jnp.int32(len(warm_disc)),
            )
            warm_reg_losses[s] = float(jnp.mean(rl))
            warm_disc_losses[s] = float(jnp.mean(dl))

        aux = snapshot(starting_epoch)  # persist the warmup curves into aux (saved at epoch 0)
        _plot_warmup(aux, os.path.join(output, "warmup.png"))

    # --- pre-fill the replay rings to FULL capacity before training -----------
    # The rings are not checkpointed, so this runs every time (fresh start AND resume), sampled at
    # the current design (regressor: LOCAL perturbation; discriminator: WHOLE space). The rings are
    # full from the first design step, then each step overwrites the oldest `samples`.
    filled = 0
    while filled < ring_capacity:
        chunk = min(samples, ring_capacity - filled)
        theta_pert = theta[None, :] + design_eps * jax.random.normal(_key(prefill_seq), (chunk, design_dim))
        ev = detector.sample_events(prefill_seq.spawn(1)[0], detector.decode_design(theta_pert), pool=train_pool)
        _Xn, feats, tnorm, _cn = prepare(ev["X"], ev["targets"], ev["conditioning"], theta_pert)
        reg_ring.push(feats, ev["mask"], tnorm)
        disc_ring.push(*_disc_batch(prefill_seq, chunk))
        filled += chunk

    # --- the optimization loop: `epochs` x (`steps` design steps each) --------
    for epoch in tqdm(range(starting_epoch, epochs)):
        t0 = time.time()
        epoch_seq = master.spawn(1)[0]  # this epoch's single seed; all draws derive from it
        step_losses = np.empty(steps, np.float32)
        step_disc = np.empty(steps, np.float32)
        for _step in tqdm(range(steps)):
            # (1-3) sample `samples` fresh events into the current-step buffers, then APPEND them
            # to the historical rings BEFORE training (so the ring is never empty and the current
            # samples count toward both sources -- "completely fair"). theta stays a jax array;
            # numpy only at the C-detector boundary (decode -> phys).
            # regressor data: LOCAL perturbation around the current design.
            theta_pert = theta[None, :] + design_eps * jax.random.normal(_key(epoch_seq), (samples, design_dim))
            ev = detector.sample_events(epoch_seq.spawn(1)[0], detector.decode_design(theta_pert), pool=train_pool)
            _Xn, feats, tnorm, _cn = prepare(ev["X"], ev["targets"], ev["conditioning"], theta_pert)
            fresh_reg = (feats, ev["mask"], tnorm)
            # discriminator data: the WHOLE encoded space (theta ~ N(0,1)), independent of the design.
            fresh_disc = _disc_batch(epoch_seq, samples)
            reg_ring.push(*fresh_reg)
            disc_ring.push(*fresh_disc)

            # (4a) regressor: `substeps` scan-folded SGD steps; each minibatch = `batch` from the
            # current `fresh_reg` + `batch` from the historical ring.
            reg_params, reg_state, reg_opt_state, rlosses = train_regressor(
                reg_params, reg_state, reg_opt_state, _key(epoch_seq), fresh_reg, reg_ring.buffers(), jnp.int32(len(reg_ring))
            )
            step_losses[_step] = float(jnp.mean(rlosses))

            # (4b) discriminator: `substeps` scan-folded steps separating joint from product
            disc_params, disc_state, disc_opt_state, dlosses = train_discriminator(
                disc_params,
                disc_state,
                disc_opt_state,
                _key(epoch_seq),
                fresh_disc,
                disc_ring.buffers(),
                jnp.int32(len(disc_ring)),
            )
            step_disc[_step] = float(jnp.mean(dlosses))

            # (5) LFI design gradient + step on a FRESH batch at the EXACT current theta (jitted),
            # drawn from the separate 'design' pool (held out from regressor/discriminator training).
            phys_cur = detector.decode_design(jnp.broadcast_to(theta[None, :], (design_batch, design_dim)))
            ev = detector.sample_events(epoch_seq.spawn(1)[0], phys_cur, pool=design_pool)
            theta, design_opt_state, _dloss, dgrad = design_step(
                theta,
                design_opt_state,
                ev["X"],
                ev["mask"],
                ev["targets"],
                ev["conditioning"],
                reg_params,
                reg_state,
                disc_params,
                disc_state,
            )

        # (6) validation: refill the val buffer at the current theta, then scan-fold per-axis
        # regressor predictions over the whole buffer.
        theta_v = jnp.broadcast_to(theta[None, :], (val_batch, design_dim))
        phys_v = detector.decode_design(theta_v)  # design dict; theta is fixed across the fill
        for _ in range(val_batches):
            ev = detector.sample_events(epoch_seq.spawn(1)[0], phys_v, pool=val_pool)
            _Xn, featsv, tnv, _cnv = prepare(ev["X"], ev["targets"], ev["conditioning"], theta_v)
            val_ring.push(featsv, ev["mask"], tnv)
        val = validate(reg_params, reg_state, *val_ring.buffers())  # {metric_key: scalar}

        train_losses[epoch] = step_losses.mean()
        disc_losses[epoch] = step_disc.mean()
        designs[epoch] = np.asarray(detector.flatten_design(detector.decode_design(theta)))
        for name in labels:
            val_metrics[name][epoch] = float(val[name])
        aux = snapshot(epoch + 1)

        # checkpoint every epoch (both nets + theta + optimizer states; rings are NOT saved)
        _save(
            manager,
            epoch,
            reg_params=reg_params,
            reg_state=reg_state,
            reg_opt_state=reg_opt_state,
            disc_params=disc_params,
            disc_state=disc_state,
            disc_opt_state=disc_opt_state,
            theta=theta,
            design_opt_state=design_opt_state,
            aux=aux,
        )
        with open(os.path.join(output, "trajectory.json"), "w") as f:
            json.dump(_jsonable(aux), f)
        _save_design_yaml(output, epoch, detector, theta)

        plot_async(aux, losses_path)

        if progress:
            vstr = " ".join(f"{name}={float(val[name]):.3f}" for name in labels)
            print(
                f"epoch {epoch + 1}/{epochs}  train={train_losses[epoch]:.5f}  disc={disc_losses[epoch]:.4f}  "
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
        fig = Figure(figsize=(16, 11))
        top, bottom = fig.subfigures(2, 1)  # top: losses; bottom: one subplot per design key
        ax_train, ax_val, ax_disc = top.subplots(1, 3)

        train = np.asarray(aux["train"])
        ax_train.plot(np.arange(len(train)), train, ".-")
        ax_train.set(title="regressor training loss", xlabel="epoch", yscale="log", ylabel="MSE")

        for name, arr in aux["val"].items():
            arr = np.asarray(arr)
            ax_val.plot(np.arange(len(arr)), arr, ".-", label=name)
        ax_val.set(title="validation MSE (normalized)", xlabel="epoch", yscale="log")
        ax_val.legend(fontsize=8, ncol=2)

        disc = np.asarray(aux["disc"])
        ax_disc.plot(np.arange(len(disc)), disc, ".-", color="tab:red")
        ax_disc.set(title="discriminator BCE (joint vs product)", xlabel="epoch", ylabel="BCE")

        detopt.utils.viz.design.plot_design_trajectory(bottom, np.asarray(aux["design"]), aux.get("design_meta"))
        fig.tight_layout()
        fig.savefig(path)


def _plot_warmup(aux, path):
    """Separate one-shot warmup plot: regressor MSE + discriminator BCE vs warmup step."""
    from matplotlib.figure import Figure

    wu = aux.get("warmup") or {}
    reg, disc = np.asarray(wu.get("regressor", [])), np.asarray(wu.get("discriminator", []))
    if reg.size == 0 and disc.size == 0:
        return

    with _PLOT_LOCK:
        fig = Figure(figsize=(9, 5))
        ax = fig.subplots(1, 1)
        if reg.size:
            ax.plot(np.arange(len(reg)), reg, ".-", label="regressor MSE")
        if disc.size:
            ax.plot(np.arange(len(disc)), disc, ".-", color="tab:red", label="discriminator BCE")
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

    gearup.gearup(optimize=optimize, report=report).with_config("config/lfi.yaml")()
