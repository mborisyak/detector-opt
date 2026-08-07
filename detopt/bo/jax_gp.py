"""ARD-RBF Gaussian process with k-fold-CV hyperparameter fitting -- REFERENCE ONLY.

.. warning::

   Nothing in production uses this module. :class:`detopt.bo.BayesianOptimizer`
   builds its surrogate with scikit-learn (see
   :mod:`detopt.bo.bayesian_optimization`); this JAX implementation is retained
   as an independent, differentiable reference to TEST against -- notably for
   checking the analytic EI gradient in the sklearn driver against JAX autodiff,
   and by ``scripts/benchmark_gp.py``. Prefer the sklearn path for anything real:
   this one is jitted, so it compiles once per distinct dataset size and retains
   every executable, which leaks tens of MB per BO iteration over a long run.

It fits hyperparameters by minimising a predictive k-fold CV NLL, where sklearn
maximises the log MARGINAL likelihood -- a deliberate difference, so the two are
not expected to select identical hyperparameters.

Hyperparameters travel as a :class:`GPHParams` namedtuple (a JAX pytree), so
every transform — ``grad``, ``vmap``, ``tree_map`` updates — operates on it
directly. No packed arrays, no positional index magic.

This module is deliberately *scale-agnostic*: it fits and predicts on ``y``
exactly as supplied. Standardising the observations (and rescaling the noise)
is the caller's job — :class:`detopt.bo.BayesianOptimizer` owns that. The GP
itself never shifts or scales ``y``.

Observation noise is **not** fitted: every entry-point takes a mandatory
per-observation standard-deviation vector ``noise (n,)`` (heteroscedastic,
supplied by the caller). Only the lengthscales and the amplitude are fitted, by
minimising the sum of per-fold predictive negative log-likelihoods (predictive
k-fold cross-validation). The inner optimiser is L-BFGS with a zoom/Wolfe line
search (see :func:`_lbfgs_optimise`); :func:`fit_gp` runs it from multiple
restarts, :func:`refit_gp` warm-starts a single run from a previous state.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.scipy.stats import norm as jax_norm

__all__ = [
    "GPHParams",
    "GPState",
    "rbf_kernel",
    "gp_factorize",
    "gp_predict",
    "gp_posterior",
    "fit_gp",
    "refit_gp",
    "expected_improvement",
    "optimise_ei",
]


class GPHParams(NamedTuple):
    """Fitted log-space GP hyperparameters (a JAX pytree).

    Leaves may carry a leading batch axis (e.g. one row per restart under
    ``vmap``): ``log_lengthscale`` is then ``(batch, d)`` and ``log_amplitude``
    ``(batch,)``. Observation noise is supplied separately, not fitted.
    """

    log_lengthscale: jax.Array  # (d,)
    log_amplitude: jax.Array  # ()


class GPState(NamedTuple):
    """Fitted GP state: training data + Cholesky factors of the noisy kernel.

    ``L`` and ``alpha`` are computed on ``y`` as supplied (the caller is
    responsible for any standardisation). ``y_train`` is kept so the acquisition
    reads the incumbent (the best observed ``y``) straight off the state -- no
    separate ``y_best`` argument to thread through EI. Precomputing the
    factorisation once amortises the O(n^3) cost across many :func:`gp_predict`
    calls (e.g. every L-BFGS step of EI maximisation).
    """

    X_train: jax.Array  # (n, d)
    y_train: jax.Array  # (n,)   observations (the EI incumbent is their max)
    hyper_parameters: GPHParams
    L: jax.Array  # (n, n)  lower-triangular Cholesky factor
    alpha: jax.Array  # (n,)    K_noisy^{-1} y


def rbf_kernel(x1: jax.Array, x2: jax.Array, log_lengthscale: jax.Array, log_amplitude: jax.Array) -> jax.Array:
    """ARD-RBF kernel between two batches of points."""
    lengthscale = jnp.exp(log_lengthscale)
    amplitude2 = jnp.exp(2.0 * log_amplitude)
    diff = (x1[:, None, :] - x2[None, :, :]) / lengthscale[None, None, :]
    sq = jnp.sum(diff * diff, axis=-1)
    return amplitude2 * jnp.exp(-0.5 * sq)


def _clamp_hp(hp, log_lengthscale_bounds, log_amplitude_bounds):
    """Clamp log-hyperparameters into their (inclusive) prior box.

    The unconstrained L-BFGS fit can drift outside the prior ranges; clamping
    keeps the lengthscale at the domain scale and the amplitude bounded, which
    is what keeps the kernel matrix well-conditioned.
    """
    ls_low, ls_high = log_lengthscale_bounds
    amp_low, amp_high = log_amplitude_bounds
    return GPHParams(
        log_lengthscale=jnp.clip(hp.log_lengthscale, ls_low, ls_high),
        log_amplitude=jnp.clip(hp.log_amplitude, amp_low, amp_high),
    )


def gp_factorize(
    X_train: jax.Array,
    y_train: jax.Array,
    hyper_parameters: GPHParams,
    noise: jax.Array,
) -> GPState:
    """Precompute Cholesky factors of the noisy training kernel.

    ``noise`` is the (mandatory) per-observation standard deviation ``(n,)``; its
    variance ``noise**2`` is placed on the kernel diagonal. ``y`` is used exactly
    as supplied (the caller standardises it if desired).
    """
    n = X_train.shape[0]
    y_train = jnp.asarray(y_train)

    # No Cholesky jitter: every observation carries a positive noise variance on
    # the diagonal, and the lengthscale/amplitude are clamped to their prior box
    # (see _clamp_hp), so K stays well-conditioned on its own.
    diag = jnp.broadcast_to(jnp.asarray(noise) ** 2, (n,))
    K = rbf_kernel(
        X_train,
        X_train,
        hyper_parameters.log_lengthscale,
        hyper_parameters.log_amplitude,
    )
    K = K + jnp.diag(diag)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
    return GPState(
        X_train=X_train,
        y_train=y_train,
        hyper_parameters=hyper_parameters,
        L=L,
        alpha=alpha,
    )


def gp_predict(state: GPState, X_test: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Closed-form GP posterior ``(mean, var)`` at ``X_test`` (in ``y``'s units)."""
    hyper_parameters = state.hyper_parameters
    Ks = rbf_kernel(
        state.X_train,
        X_test,
        hyper_parameters.log_lengthscale,
        hyper_parameters.log_amplitude,
    )
    Kss_diag = jnp.exp(2.0 * hyper_parameters.log_amplitude) * jnp.ones(X_test.shape[0])
    mean = Ks.T @ state.alpha
    v = jax.scipy.linalg.cho_solve((state.L, True), Ks)
    var = jnp.clip(Kss_diag - jnp.sum(Ks * v, axis=0), 1e-12, None)
    return mean, var


def gp_posterior(
    X_train: jax.Array,
    y_train: jax.Array,
    X_test: jax.Array,
    hyper_parameters: GPHParams,
    noise: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Closed-form GP posterior ``(mean, var)`` at ``X_test`` (factorise + predict).

    Re-factorises on every call (O(n^3)), so it is only for one-shot posteriors
    and for the CV objective (:func:`_make_cv_nll`), where the kernel genuinely
    changes with every hyperparameter candidate -- there the re-factorisation is
    intrinsic. The BO/EI hot loop must NOT use it: it factorises once via
    :func:`gp_factorize` and reuses the factors through :func:`gp_predict`.
    ``noise`` is the per-observation std ``(n,)`` for the training points.
    """
    return gp_predict(gp_factorize(X_train, y_train, hyper_parameters, noise), X_test)


def _make_kfold_stacks(key: jax.Array, X: jax.Array, y: jax.Array, noise: jax.Array, n_folds: int) -> tuple[jax.Array, ...]:
    """Shuffle the data into ``n_folds`` train/val splits, stacked for one vmap.

    The ``n`` points are shuffled and cut into ``n_folds`` equal validation
    blocks of ``n_val = n // n_folds`` points. Fold ``f`` **validates on block
    ``f`` and trains on every OTHER block** -- ``(n_folds - 1) * n_val`` points,
    i.e. everything except the held-out block (ordinary k-fold; there is no
    "1 train / 1 val"). Returns stacked ``(n_folds, n_train, ...)`` train and
    ``(n_folds, n_val, ...)`` val tensors.

    Equal block sizes keep those shapes static for JIT; the price is dropping the
    ``n mod n_folds`` leftover points. That only bites at small ``n``, which is
    exactly where :func:`fit_gp` switches to leave-one-out (``n_folds = n`` =>
    ``n_val = 1``, nothing dropped). So ``n = 5`` with 5 folds is genuine LOO
    (1 val, 4 train per fold); ``n = 9`` runs as 9-fold LOO, never "5 of 9".
    ``noise`` is the per-observation std, split alongside ``y``.
    """
    n_total = X.shape[0]
    n_val = n_total // n_folds
    n_eff = n_val * n_folds
    perm = jax.random.permutation(key, n_total)[:n_eff].reshape(n_folds, n_val)

    fold_idx = jnp.arange(n_folds)
    other = (fold_idx[:, None] + 1 + jnp.arange(n_folds - 1)[None, :]) % n_folds
    train_idx = perm[other].reshape(n_folds, (n_folds - 1) * n_val)
    val_idx = perm
    return (
        X[train_idx],
        y[train_idx],
        noise[train_idx],
        X[val_idx],
        y[val_idx],
        noise[val_idx],
    )


def _make_cv_nll(key_fold, X, y, noise, n_folds):
    """Build the predictive k-fold CV negative-log-likelihood on ``(X, y, noise)``.

    Leave-one-out (``k = n``, one point per fold) below ``2 * n_folds`` so no
    point is dropped in the low-data regime; ordinary ``n_folds``-fold above it.
    """
    n = X.shape[0]
    n_folds = n if n < 2 * n_folds else n_folds
    X_train, y_train, noise_train, X_val, y_val, noise_val = _make_kfold_stacks(key_fold, X, y, noise, n_folds)

    def kfold_nll(hyper_parameters: GPHParams) -> jax.Array:
        means, variances = jax.vmap(lambda Xt, yt, nt, Xv: gp_posterior(Xt, yt, Xv, hyper_parameters, nt))(
            X_train, y_train, noise_train, X_val
        )
        # noise_val is a std -> its variance enters the predictive variance.
        pred_var = variances + noise_val**2 + 1e-9  # latent var + held-out obs noise
        nll = 0.5 * (jnp.log(2.0 * jnp.pi * pred_var) + jnp.square(y_val - means) / pred_var)
        return jnp.sum(nll)

    return kfold_nll


def _lbfgs_optimise(nll_fn, hp_init, n_steps):
    """L-BFGS (with zoom/Wolfe line search) on ``nll_fn`` from ``hp_init``.

    Runs ``n_steps`` L-BFGS iterations and returns the **best-seen** iterate
    ``(hp_best, nll_best)`` rather than the last one: the line search can still
    drift after the gradient flattens on an ill-conditioned objective, so the
    best iterate is the robust choice.
    """
    opt = optax.lbfgs()  # default: zoom (Wolfe) line search
    value_and_grad = optax.value_and_grad_from_state(nll_fn)

    def step(carry, _):
        hp, opt_state, best_hp, best_nll = carry
        value, grad = value_and_grad(hp, state=opt_state)
        improved = value < best_nll
        best_hp = jax.tree.map(lambda b, cur: jnp.where(improved, cur, b), best_hp, hp)
        best_nll = jnp.where(improved, value, best_nll)
        updates, opt_state = opt.update(grad, opt_state, hp, value=value, grad=grad, value_fn=nll_fn)
        hp = optax.apply_updates(hp, updates)
        return (hp, opt_state, best_hp, best_nll), None

    init = (hp_init, opt.init(hp_init), hp_init, jnp.inf)
    (_, _, best_hp, best_nll), _ = jax.lax.scan(step, init, None, length=n_steps)
    return best_hp, best_nll


def fit_gp(
    key: jax.Array,
    X: jax.Array,
    y: jax.Array,
    noise: jax.Array,
    *,
    n_restarts: int,
    n_steps: int,
    log_lengthscale_prior_bounds: tuple[float, float],
    log_amplitude_prior_bounds: tuple[float, float],
    n_folds: int = 5,
) -> GPState:
    """Multi-restart k-fold-CV fit of GP lengthscales + amplitude.

    Fits and refits on ``y`` exactly as supplied (no standardisation — the
    caller owns that). Observation noise is not fitted; ``noise`` (the mandatory
    per-observation standard deviation ``(n,)``) enters the predictive likelihood
    directly. Each restart draws an independent initial point uniformly from
    per-parameter log-space ranges and is optimised with L-BFGS (``n_steps``
    iterations); the lowest-CV-NLL hyperparameters are then refit on the full
    ``(X, y, noise)`` and returned as a ready-to-use :class:`GPState`.

    Parameters
    ----------
    key
        PRNG key for fold permutation and prior sampling.
    X
        ``(n, d)`` design matrix.
    y
        ``(n,)`` observations.
    noise
        ``(n,)`` per-observation standard deviation (heteroscedastic). Mandatory.
    n_restarts, n_steps
        Number of random restarts and L-BFGS iterations per restart.
    log_lengthscale_prior_bounds, log_amplitude_prior_bounds
        ``(low, high)`` inclusive log-space ranges for the prior-sampled starts.
    n_folds
        Number of CV folds (default 5). Falls back to leave-one-out (``n`` folds)
        when ``n < 2 * n_folds`` so no point is dropped in the low-data regime;
        returns the prior midpoint unfitted only when ``n < 2``.

    Returns
    -------
    GPState
        GP refit on the full data with the best CV hyperparameters.
    """
    n, d = X.shape
    ls_low, ls_high = log_lengthscale_prior_bounds
    amp_low, amp_high = log_amplitude_prior_bounds

    X = jnp.asarray(X)
    y = jnp.asarray(y)
    noise = jnp.asarray(noise, dtype=X.dtype)

    prior_midpoint = GPHParams(
        log_lengthscale=jnp.full((d,), 0.5 * (ls_low + ls_high), dtype=jnp.float32),
        log_amplitude=jnp.float32(0.5 * (amp_low + amp_high)),
    )

    # Below 2 points there is nothing to hold out -> return the prior midpoint.
    # (_make_cv_nll falls back to leave-one-out for the rest of the low-data regime.)
    if n < 2:
        return gp_factorize(X, y, prior_midpoint, noise)

    key_fold, key_ls, key_amp = jax.random.split(key, 3)
    nll_fn = _make_cv_nll(key_fold, X, y, noise, n_folds)

    n_runs = max(n_restarts, 1)
    hp0_batch = GPHParams(
        log_lengthscale=jax.random.uniform(key_ls, (n_runs, d), minval=ls_low, maxval=ls_high, dtype=jnp.float32),
        log_amplitude=jax.random.uniform(key_amp, (n_runs,), minval=amp_low, maxval=amp_high, dtype=jnp.float32),
    )

    run = jax.jit(jax.vmap(lambda hp0: _lbfgs_optimise(nll_fn, hp0, n_steps)))
    hp_finals, last_nlls = run(hp0_batch)
    masked_nlls = jnp.where(jnp.isfinite(last_nlls), last_nlls, jnp.inf)
    best_idx = jnp.argmin(masked_nlls)
    best = jax.tree.map(lambda leaf: leaf[best_idx], hp_finals)

    ok = jnp.isfinite(masked_nlls[best_idx])
    best_hp = jax.tree.map(lambda b, m: jnp.where(ok, b, m), best, prior_midpoint)

    # L-BFGS is unconstrained, so clamp the fit back into the prior box. This
    # keeps the lengthscale at the domain scale and the amplitude bounded, which
    # is what keeps the kernel well-conditioned (no Cholesky jitter needed).
    best_hp = _clamp_hp(best_hp, log_lengthscale_prior_bounds, log_amplitude_prior_bounds)

    # Refit on the full (X, y, noise) with the chosen hyperparameters.
    return gp_factorize(X, y, best_hp, noise)


def refit_gp(
    key: jax.Array,
    state: GPState,
    X: jax.Array,
    y: jax.Array,
    noise: jax.Array,
    *,
    n_steps: int,
    n_folds: int = 5,
) -> GPState:
    """Warm-start refit from an existing :class:`GPState` (single run, no multi-start).

    Reuses ``state``'s hyperparameters as the initial point and runs a single
    L-BFGS optimisation of the CV-NLL on the new ``(X, y, noise)``, then refits
    on the full data. Cheaper than :func:`fit_gp` when growing the dataset
    between BO iterations and the optimum moves little. Diverged fits fall back
    to the warm-start hyperparameters.
    """
    n = X.shape[0]
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    noise = jnp.asarray(noise, dtype=X.dtype)

    hp_init = state.hyper_parameters
    if n < 2:  # nothing to hold out -> keep the warm-start hyperparameters
        return gp_factorize(X, y, hp_init, noise)

    nll_fn = _make_cv_nll(key, X, y, noise, n_folds)
    hp_final, last_nll = jax.jit(lambda hp: _lbfgs_optimise(nll_fn, hp, n_steps))(hp_init)

    ok = jnp.isfinite(last_nll)
    hp_used = jax.tree.map(lambda f, i: jnp.where(ok, f, i), hp_final, hp_init)
    return gp_factorize(X, y, hp_used, noise)


# --------------------------------------------------------------------------- #
# Expected-Improvement acquisition (was detopt/bo/acquisition.py).
#
# Both the analytic EI score and a multi-restart L-BFGS maximiser of EI over the
# unit cube. The objective is MINIMISED; the incumbent is the best (lowest)
# observed value, read straight off the GP state. EI itself is always maximised.
# --------------------------------------------------------------------------- #

def expected_improvement(X: jax.Array, state: GPState) -> jax.Array:
    """Analytic Expected Improvement at ``X`` for **minimisation**.

    ``X`` has shape ``(m, d)``. Returns a ``(m,)`` vector of EI values. The
    incumbent is the best (lowest) observed objective, ``state.y_train.min()``;
    improvement is ``max(y_best - y, 0)``.
    """
    mean, var = gp_predict(state, X)
    sigma = jnp.sqrt(var)
    y_best = jnp.min(state.y_train)
    z = (y_best - mean) / sigma
    return (y_best - mean) * jax_norm.cdf(z) + sigma * jax_norm.pdf(z)


def optimise_ei(
    key: jax.Array,
    state: GPState,
    *,
    n_restarts: int,
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Multi-restart L-BFGS maximisation of EI over the unit cube ``[0, 1]^d``.

    The GP/EI always live in the normalised design cube ``[0, 1]^d``. The search
    is parameterised in unconstrained space via a logistic map (``x =
    sigmoid(z)``), so L-BFGS (with a zoom/Wolfe line search) stays unconstrained
    while ``x`` is confined to ``(0, 1)``. Each of ``n_restarts`` independent
    starts runs ``n_steps`` iterations; the best-seen iterate is kept.

    Returns ``(x_best, ei_best)`` -- the maximiser in ``[0, 1]^d`` and its EI.
    """
    d = state.X_train.shape[1]
    x0 = jax.random.uniform(key, (n_restarts, d), minval=0.0, maxval=1.0, dtype=jnp.float32)
    x0 = jnp.clip(x0, 1e-4, 1.0 - 1e-4)
    z0 = jnp.log(x0) - jnp.log1p(-x0)  # logit

    def neg_ei(z):
        x = jax.nn.sigmoid(z)
        return -expected_improvement(x[None, :], state)[0]

    def run_one(z_init):
        opt = optax.lbfgs()  # default: zoom (Wolfe) line search
        value_and_grad = optax.value_and_grad_from_state(neg_ei)

        def step(carry, _):
            z, opt_state, best_z, best_v = carry
            v, g = value_and_grad(z, state=opt_state)
            improved = v < best_v
            best_z = jnp.where(improved, z, best_z)
            best_v = jnp.where(improved, v, best_v)
            updates, opt_state = opt.update(g, opt_state, z, value=v, grad=g, value_fn=neg_ei)
            z = optax.apply_updates(z, updates)
            return (z, opt_state, best_z, best_v), None

        init = (z_init, opt.init(z_init), z_init, jnp.inf)
        (_, _, best_z, best_v), _ = jax.lax.scan(step, init, None, length=n_steps)
        return jax.nn.sigmoid(best_z), -best_v

    x_finals, ei_finals = jax.jit(jax.vmap(run_one))(z0)  # (n_restarts, d), (n_restarts,)
    best_idx = jnp.argmax(ei_finals)
    return x_finals[best_idx], ei_finals[best_idx]
