"""Can a plain DENSE MLP regress the HNL target at the nominal design, given the SAME
design-informed ``combine`` features the DeepSet regressor sees?

Isolates the regressor architecture from the physics. Samples events at the nominal design
+/- eps on the disjoint TRAIN pool, builds the per-event ``combine`` feature tensor (M, F),
and trains a dense MLP on the FLATTENED features -> normalized 6-vec target; evaluates on the
held-out VAL pool.

Per the experiment spec -- a DeepSet ignores mask=0 hits by construction (masked aggregation),
but a plain MLP processes every slot, so the padded slots must be handled explicitly:
  * use the normal ``normalize`` + ``combine`` pipeline (design-informed features);
  * NON-EXISTING (padded) hits get RANDOM station/view/layer/straw (so combine gathers random
    geometry) and the TDC feature set to -1 (the sentinel);
  * hits are PERMUTED within each event (no positional leakage); NO mask is fed.

Run on GPU: ``python scripts/debug_mlp.py`` (reads config/lfi.yaml). Override ``key=value``
(e.g. ``buffer=49152 eps=0.1 train_steps=8000``).
"""

import sys
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import optax

import detopt


def main(buffer=32768, val_buffer=8192, eps=0.1, train_steps=6000, batch=512, config="config/lfi.yaml"):
    buffer, val_buffer, train_steps, batch, eps = int(buffer), int(val_buffer), int(train_steps), int(batch), float(eps)
    cfg = yaml.safe_load(open(config))
    det = detopt.detector.from_config(cfg["detector"])
    dd = int(det.design_dim())
    M = det.max_hits_per_event
    F = det.combined_feature_dim
    labels = [f"y{i}" for i in range(int(det.target_shape()[0]))]  # per-axis target components (net outputs)
    nd = cfg["nominal_design"]
    theta0 = jnp.asarray(det.encode_design(nd), jnp.float32)
    keys = list(det.pool_split)
    train_pool, val_pool = keys[0], (keys[1] if len(keys) > 1 else keys[0])
    rng = np.random.default_rng(0)
    counts = np.array([det.n_stations, det.n_views_per_station, det.n_layers_per_view, det.n_straws], np.float32)
    print(f"M={M} F={F} input_dim={M * F}  design_dim={dd}  eps={eps}  pools train={train_pool!r}/val={val_pool!r}")

    def featurize(X, mask, theta, key):
        # Real combine() features, but with padded slots given RANDOM geometry + TDC=-1
        # (a plain MLP processes every slot, unlike the DeepSet's masked aggregation).
        # Returns UNpermuted (B, M, F); permutation is applied per-step (augmentation).
        B = X.shape[0]
        m = jnp.asarray(mask).astype(bool)[..., None]  # (B,M,1)
        X = jnp.asarray(X)
        rand_idx = jnp.floor(jax.random.uniform(key, (B, M, 4)) * jnp.asarray(counts))  # random station/view/...
        X = jnp.concatenate([jnp.where(m, X[..., :4], rand_idx), X[..., 4:5]], axis=-1)  # randomize padded geometry
        feats = det.combine(det.normalize(X), theta)  # (B,M,F): [TDC, z, yL, yR, field]
        feats = feats.at[..., 0].set(jnp.where(m[..., 0], feats[..., 0], -1.0))  # TDC=-1 for padded
        return feats  # (B, M, F)

    def permute_flat(feats, key):  # per-event hit permutation -> flatten (B, M*F)
        B = feats.shape[0]
        perm = jnp.argsort(jax.random.uniform(key, (B, M)), axis=1)
        return jnp.take_along_axis(feats, perm[..., None], axis=1).reshape(B, M * F)

    def fill(n, pool, key):
        Fs, Ts = [], []
        got = 0
        while got < n:
            c = min(2048, n - got)
            theta = theta0[None, :] + eps * jax.random.normal(jax.random.PRNGKey(int(rng.integers(1 << 30))), (c, dd))
            out = det.sample_events(int(rng.integers(1 << 30)), det.decode_design(theta), pool=pool)
            key, kf = jax.random.split(key)
            Fs.append(featurize(out["X"], out["mask"], theta, kf))
            Ts.append(jnp.asarray(det.normalize_target(out["targets"])))
            got += c
        return jnp.concatenate(Fs), jnp.concatenate(Ts)

    Xtr, Ytr = fill(buffer, train_pool, jax.random.PRNGKey(1))
    Xva, Yva = fill(val_buffer, val_pool, jax.random.PRNGKey(2))
    print(f"train buffer {Xtr.shape}  val {Xva.shape}")

    # dense MLP
    def init(k, sizes):
        ps = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            k, kw = jax.random.split(k)
            ps.append((jax.random.normal(kw, (i, o)) * (2.0 / i) ** 0.5, jnp.zeros(o)))
        return ps

    sizes = [M * F, 64, 64, 32, len(labels)]
    params = init(jax.random.PRNGKey(0), sizes)
    n_params = sum(int(W.size + b.size) for W, b in params)
    print(f"MLP sizes={sizes}  params={n_params}  (buffer={Xtr.shape[0]} -> {Xtr.shape[0] / n_params:.2f} samples/param)")

    def fwd(params, x):
        for W, b in params[:-1]:
            x = jax.nn.gelu(x @ W + b)
        W, b = params[-1]
        return x @ W + b

    opt = optax.adamw(1e-3, weight_decay=1e-4)
    ost = opt.init(params)

    @jax.jit
    def step(params, ost, key):
        k_idx, k_perm = jax.random.split(key)
        idx = jax.random.randint(k_idx, (batch,), 0, Xtr.shape[0])
        xb = permute_flat(Xtr[idx], k_perm)  # fresh hit permutation each step (augmentation)

        def L(p):
            return jnp.mean((fwd(p, xb) - Ytr[idx]) ** 2)

        l, g = jax.value_and_grad(L)(params)
        u, ost = opt.update(g, ost, params)
        return optax.apply_updates(params, u), ost, l

    for i in range(train_steps):
        params, ost, l = step(params, ost, jax.random.PRNGKey(i))

    def rmse(X, Y):
        return np.sqrt(np.asarray(jnp.mean((fwd(params, permute_flat(X, jax.random.PRNGKey(7))) - Y) ** 2, axis=0)))

    tr, va = rmse(Xtr, Ytr), rmse(Xva, Yva)
    print("normalized RMSE (sigma units; 1.0 = mean predictor):")
    print(f"  TRAIN  total={tr.mean():.3f}  " + " ".join(f"{n}={tr[i]:.3f}" for i, n in enumerate(labels)))
    print(f"  VAL    total={va.mean():.3f}  " + " ".join(f"{n}={va[i]:.3f}" for i, n in enumerate(labels)))


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    main(**kw)
