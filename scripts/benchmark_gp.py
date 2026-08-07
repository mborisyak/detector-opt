#!/usr/bin/env python3
"""Performance test: GP hyperparameter-fit optimiser comparison.

The CV-NLL objective for GP hyperparameters is full-batch, exact-gradient,
deterministic, smooth and low-dimensional -- the regime where a line-search
quasi-Newton method (L-BFGS) should dominate a fixed-LR stochastic optimiser
(Adam). This benchmark builds the same k-fold CV-NLL objective that
``detopt.bo.jax_gp.fit_gp`` minimises and compares, on identical data:

  * **fit**   -- multi-restart (cold) optimisation, best over restarts;
  * **refit** -- single warm-started run from a previous state's hyperparameters;

each with **Adam** and **L-BFGS + line search**. It reports solution quality
(best-seen CV-NLL) and wall-clock (post-JIT-warmup). ``noise`` is a per-
observation standard deviation, as in the GP API.

Run::

    python scripts/benchmark_gp.py
    python scripts/benchmark_gp.py --n 80 --d 6 --restarts 8 --lbfgs-iters 25
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from detopt.bo.jax_gp import GPHParams, _make_cv_nll

# --------------------------------------------------------------------------- #
# Problem + objective (mirrors fit_gp's standardised CV-NLL)
# --------------------------------------------------------------------------- #


def make_dataset(n, d, seed, noise_std):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype("float32")
    y = (50.0 + 20.0 * np.sin(X[:, 0]) - 10.0 * X[:, min(1, d - 1)] + noise_std * rng.standard_normal(n)).astype("float32")
    noise = np.full(n, noise_std, dtype="float32")  # per-observation std
    return jnp.asarray(X), jnp.asarray(y), jnp.asarray(noise)


def build_nll(X, y, noise, n_folds, key_fold, normalize_y=True):
    """Standardised k-fold CV-NLL as a function of GPHParams.

    The GP is now scale-agnostic, so standardisation happens here (mirroring
    :class:`detopt.bo.BayesianOptimizer`): subtract the mean and the *signal*
    std (marginal var minus the observation-noise var), and scale ``noise`` by
    ``1 / y_std``.
    """
    if normalize_y:
        y_var = jnp.var(y)
        signal_var = jnp.maximum(y_var - jnp.mean(noise**2), 1e-2 * y_var)
        y_mean, y_std = jnp.mean(y), jnp.sqrt(signal_var) + 1e-9
    else:
        y_mean, y_std = jnp.float32(0.0), jnp.float32(1.0)
    y_cv = (y - y_mean) / y_std
    noise_cv = jnp.broadcast_to(noise / y_std, (X.shape[0],))  # std scales by 1/y_std
    return _make_cv_nll(key_fold, X, y_cv, noise_cv, n_folds)


def sample_inits(key, n_restarts, d, bounds):
    (ls_lo, ls_hi), (amp_lo, amp_hi) = bounds
    k_ls, k_amp = jax.random.split(key)
    return GPHParams(
        log_lengthscale=jax.random.uniform(k_ls, (n_restarts, d), minval=ls_lo, maxval=ls_hi),
        log_amplitude=jax.random.uniform(k_amp, (n_restarts,), minval=amp_lo, maxval=amp_hi),
    )


# --------------------------------------------------------------------------- #
# Optimiser runs (best-seen-iterate -> budget-monotone)
# --------------------------------------------------------------------------- #


def run_adam(nll_fn, hp0, n_steps, learning_rate):
    opt = optax.adam(learning_rate)
    grad_fn = jax.value_and_grad(nll_fn)

    def step(carry, _):
        hp, opt_state = carry
        value, grad = grad_fn(hp)
        updates, opt_state = opt.update(grad, opt_state, hp)
        hp = optax.apply_updates(hp, updates)
        return (hp, opt_state), value

    (hp_final, _), trace = jax.lax.scan(step, (hp0, opt.init(hp0)), None, length=n_steps)
    return jnp.min(trace), hp_final  # best-seen NLL (monotone in budget)


def run_lbfgs(nll_fn, hp0, max_iters):
    opt = optax.lbfgs()  # default: zoom (Wolfe) line search
    value_and_grad = optax.value_and_grad_from_state(nll_fn)

    def step(carry, _):
        hp, opt_state = carry
        value, grad = value_and_grad(hp, state=opt_state)
        updates, opt_state = opt.update(grad, opt_state, hp, value=value, grad=grad, value_fn=nll_fn)
        hp = optax.apply_updates(hp, updates)
        return (hp, opt_state), value

    (hp_final, _), trace = jax.lax.scan(step, (hp0, opt.init(hp0)), None, length=max_iters)
    return jnp.min(trace), hp_final


def timeit(fn, repeats):
    fn().block_until_ready()  # warmup / compile once
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn().block_until_ready()
    return (time.perf_counter() - t0) / repeats


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


def benchmark(
    n,
    d,
    n_folds,
    n_restarts,
    adam_steps,
    adam_lr,
    lbfgs_iters,
    repeats,
    seed,
    noise_std,
):
    key = jax.random.PRNGKey(seed)
    _, key_fold, key_init = jax.random.split(key, 3)

    X, y, noise = make_dataset(n, d, seed, noise_std)
    nll = build_nll(X, y, noise, n_folds, key_fold)
    bounds = ((-2.0, 2.0), (-2.0, 2.0))
    inits = sample_inits(key_init, n_restarts, d, bounds)

    adam_one = lambda hp0: run_adam(nll, hp0, adam_steps, adam_lr)
    lbfgs_one = lambda hp0: run_lbfgs(nll, hp0, lbfgs_iters)

    # "Previous state" for warm-started refit: best hp of an (untimed) multi-start
    # with the production optimiser (L-BFGS).
    nlls0, hps0 = jax.vmap(lbfgs_one)(inits)
    warm_hp = jax.tree.map(lambda a: a[jnp.argmin(nlls0)], hps0)

    def fit(run_one):  # multi-start: best over restarts
        nlls, _ = jax.vmap(run_one)(inits)
        return jnp.min(jnp.where(jnp.isfinite(nlls), nlls, jnp.inf))

    def refit(run_one):  # single warm-started run
        return run_one(warm_hp)[0]

    runners = {
        ("fit (multi-start)", "Adam"): (jax.jit(lambda: fit(adam_one)), adam_steps),
        ("fit (multi-start)", "L-BFGS"): (jax.jit(lambda: fit(lbfgs_one)), lbfgs_iters),
        ("refit (warm-start)", "Adam"): (jax.jit(lambda: refit(adam_one)), adam_steps),
        ("refit (warm-start)", "L-BFGS"): (
            jax.jit(lambda: refit(lbfgs_one)),
            lbfgs_iters,
        ),
    }

    print(
        f"\nGP fit/refit x optimiser benchmark "
        f"(n={n}, d={d}, folds={n_folds}, restarts={n_restarts}, noise_std={noise_std})"
    )
    print("-" * 78)
    print(f"{'mode':<22}{'optimiser':<12}{'best CV-NLL':>16}{'iters':>10}{'time (ms)':>16}")
    print("-" * 78)
    results = {}
    for (mode, opt_name), (fn, iters) in runners.items():
        nll_val = float(fn())
        t = timeit(fn, repeats)
        results[(mode, opt_name)] = (nll_val, t)
        print(f"{mode:<22}{opt_name:<12}{nll_val:>16.4f}{iters:>10}{t*1e3:>16.2f}")
    print("-" * 78)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--adam-steps", type=int, default=200)
    p.add_argument("--adam-lr", type=float, default=0.05)
    p.add_argument("--lbfgs-iters", type=int, default=25)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=2.0, help="observation noise std")
    args = p.parse_args()
    benchmark(
        args.n,
        args.d,
        args.folds,
        args.restarts,
        args.adam_steps,
        args.adam_lr,
        args.lbfgs_iters,
        args.repeats,
        args.seed,
        args.noise,
    )
