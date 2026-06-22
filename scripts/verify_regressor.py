"""Fixed-design regressor verification.

A sanity check decoupled from design optimization: take ONE design, sample a large training buffer
under it, train ONLY the regressor on that buffer, and report held-out validation metrics. No
discriminator, no design gradient -- just "how well can the regressor reconstruct the target at this
design?". Train and validation events come from disjoint detector pools.

Config: ``config/subgradient.yaml`` (detector + regressor + training.optimizer + nominal_design +
pool_split). The design defaults to ``nominal_design``; override with a ``design:`` config key (a
named physical-design dict). The ``verify:`` block sets the buffer sizes / training budget.

    python scripts/verify_regressor.py seed=0 verify.train_samples=65536 verify.steps=8000
"""

import os

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


# --------------------------------------------------------------------------- #
# Ensemble-aware forward (mirrors scripts/subgradient.py).
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


def _target_for(targets_norm, members):
    return targets_norm if members is None else jnp.broadcast_to(targets_norm[None], (members,) + targets_norm.shape)


def _key(seq):
    """A jax PRNGKey from the next child of a SeedSequence (advances ``seq`` deterministically)."""
    return jax.random.PRNGKey(int(seq.spawn(1)[0].generate_state(1)[0]))


def verify(seed, output=None, progress=True, **config):
    device = resolve_device(config.get("device"))
    v = config.get("verify", {})
    train_samples = int(v.get("train_samples", 32768))  # events in the fixed training buffer
    val_samples = int(v.get("val_samples", 8192))  # events in the held-out validation buffer
    steps = int(v.get("steps", 4000))  # total SGD steps over the training buffer
    batch = int(v.get("batch", 256))  # minibatch size (per member)
    sample_chunk = int(v.get("sample_chunk", 1024))  # events simulated per detector call while filling buffers
    eval_batch = int(v.get("eval_batch", 512))  # validation buffer is scanned in chunks of this size
    val_every = int(v.get("val_every", max(1, steps // 20)))  # validate (+ log) every this many SGD steps

    detector = detopt.detector.from_config(config["detector"])
    design = config.get("design") or config["nominal_design"]
    theta = jnp.asarray(detector.encode_design(design), jnp.float32)  # the FIXED design (encoded), shared by every event
    design_dim = int(detector.design_dim())
    labels = tuple(detector.metric_labels())

    # Disjoint pools: train the regressor on 'train', validate on a held-out pool (the last key,
    # so a {train, design, val} split validates on 'val'; clamped if fewer pools exist).
    pool_keys = list(detector.pool_split)
    train_pool = config.get("train_pool", pool_keys[0])
    val_pool = config.get("val_pool", pool_keys[-1] if len(pool_keys) > 1 else pool_keys[0])

    master = np.random.SeedSequence(int(seed))
    rngs = nnx.Rngs(jax.random.PRNGKey(int(master.spawn(1)[0].generate_state(1)[0])))
    sample_seq = master.spawn(1)[0]
    train_seq = master.spawn(1)[0]

    model = detopt.nn.from_config(detector, config=config["regressor"], rngs=rngs)
    reg_def, params, state = nnx.split(model, nnx.Param, nnx.Variable)
    members = model.ensemble() if hasattr(model, "ensemble") else None
    opt = make_optimizer(config["training"]["optimizer"])
    opt_state = opt.init(params)

    M = int(jax.tree.leaves(detector.event_spec())[0].shape[0])  # per-hit count
    specs = (detector.event_spec(), jax.ShapeDtypeStruct((M,), jnp.int32), detector.target_spec())

    def fill(n, pool, seq):
        """Sample ``n`` RAW events at the fixed design from ``pool`` into a buffer (raw event,
        mask, raw target). ``combine_encoded`` + ``normalize_target`` run per batch in the kernels."""
        buf = RingBuffer(n, specs, device=device)
        filled = 0
        bar = tqdm(total=n, desc=f"sample[{pool}]", disable=not progress)
        while filled < n:
            chunk = min(sample_chunk, n - filled)
            phys = detector.decode_design(jnp.broadcast_to(theta[None, :], (chunk, design_dim)))
            ev = detector.sample_events(seq.spawn(1)[0], phys, pool=pool)
            buf.push(ev["X"], ev["mask"], ev["target"])
            filled += chunk
            bar.update(chunk)
        bar.close()
        return buf

    def _net_loss(params, state, drop_key, event_b, mask_b, target_b, count):
        reg = nnx.merge(reg_def, params, state)
        feats = detector.combine_encoded(event_b, theta)  # fixed design (encoded), broadcast per hit
        pred = _forward(reg, feats, mask_b, members, count, deterministic=False, rngs=nnx.Rngs(drop_key))
        loss = jnp.mean(detector.loss(pred, detector.normalize_target(target_b)))
        _, _, new_state = nnx.split(reg, nnx.Param, nnx.Variable)
        return loss, new_state

    draw = (members or 1) * batch

    @jax.jit
    def train_chunk(params, state, opt_state, key, event_buf, mask_buf, tgt_buf, n):
        """``val_every`` scan-folded SGD steps over the fixed training buffer."""

        def step(carry, k):
            params, state, opt_state = carry
            k_idx, k_drop = jax.random.split(k)
            idx = jax.random.randint(k_idx, (draw,), 0, n)
            event_b = jax.tree.map(lambda a: a[idx], event_buf)  # raw Event minibatch
            target_b = jax.tree.map(lambda a: a[idx], tgt_buf)  # raw Target minibatch
            (loss, new_state), grads = jax.value_and_grad(_net_loss, has_aux=True)(
                params, state, k_drop, event_b, mask_buf[idx], target_b, batch
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, state, opt_state), loss

        (params, state, opt_state), losses = jax.lax.scan(step, (params, state, opt_state), jax.random.split(key, val_every))
        return params, state, opt_state, losses

    @jax.jit
    def validate(params, state, event_buf, mask_buf, tgt_buf):
        """Per-key validation metric (MSE) over the whole validation buffer."""
        reg = nnx.merge(reg_def, params, state)
        n_chunks = val_samples // eval_batch
        cut = n_chunks * eval_batch
        reshape = lambda a: a[:cut].reshape((n_chunks, eval_batch) + a.shape[1:])
        event_c, mask_c, tgt_c = jax.tree.map(reshape, event_buf), reshape(mask_buf), jax.tree.map(reshape, tgt_buf)

        def step(acc, chunk):
            ev_chunk, m, t = chunk
            feats = detector.combine_encoded(ev_chunk, theta)
            pred = _forward_shared(reg, feats, m, members, deterministic=True)
            md = detector.metric(pred, _target_for(detector.normalize_target(t), members))
            return {k: acc[k] + jnp.sum(md[k]) for k in acc}, None

        init = {name: jnp.float32(0.0) for name in labels}
        acc, _ = jax.lax.scan(step, init, (event_c, mask_c, tgt_c))
        count = n_chunks * eval_batch * (members or 1)
        return {k: acc[k] / count for k in labels}

    # --- sample the buffers, then train + validate ----------------------------
    train_buf = fill(train_samples, train_pool, sample_seq)
    val_buf = fill(val_samples, val_pool, sample_seq)
    n_train = jnp.int32(len(train_buf))

    print(f"design (encoded->physical): {detector.decode_design(theta)}")
    print(f"train={train_samples} val={val_samples}  steps={steps} batch={batch}  members={members or 1}")

    history = []
    done = 0
    while done < steps:
        params, state, opt_state, losses = train_chunk(params, state, opt_state, _key(train_seq), *train_buf.buffers(), n_train)
        done += val_every
        train_loss = float(jnp.mean(losses))
        val = validate(params, state, *val_buf.buffers())
        history.append((done, train_loss, {k: float(val[k]) for k in labels}))
        if progress:
            vstr = " ".join(f"{k}={float(val[k]):.3f}" for k in labels)
            print(f"step {done}/{steps}  train={train_loss:.4f}  val[{vstr}]")

    if output:
        os.makedirs(output, exist_ok=True)
        _plot(history, labels, os.path.join(output, "verify.png"))
    return history


def _plot(history, labels, path):
    from matplotlib.figure import Figure

    steps = [h[0] for h in history]
    train = [h[1] for h in history]
    fig = Figure(figsize=(11, 5))
    ax_train, ax_val = fig.subplots(1, 2)
    ax_train.plot(steps, train, ".-")
    ax_train.set(title="regressor training loss", xlabel="step", ylabel="MSE", yscale="log")
    for name in labels:
        ax_val.plot(steps, [h[2][name] for h in history], ".-", label=name)
    ax_val.set(title="validation MSE (normalized)", xlabel="step", yscale="log")
    ax_val.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    import gearup

    gearup.gearup(verify).with_config("config/subgradient.yaml")()
