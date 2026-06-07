"""Tests for the detopt.bo subpackage (JAX GP + EI acquisition)."""

import numpy as np
import jax
import jax.numpy as jnp

import detopt


def _toy_dataset(n=20, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype("float32")
    y = (X[:, 0] ** 2 - 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)).astype("float32")
    y = (y - y.mean()) / (y.std() + 1e-9)
    return X, y


def _noise(n, std=0.1):
    """Per-observation standard deviation vector."""
    return jnp.full((n,), std, dtype=jnp.float32)


_GP_CFG = dict(
    n_folds=5,
    n_restarts=2,
    n_steps=30,
    log_lengthscale_prior_bounds=(-2.0, 2.0),
    log_amplitude_prior_bounds=(-2.0, 2.0),
)
_EI_CFG = dict(n_restarts=4, n_steps=30)
_REFIT_CFG = dict(n_folds=5, n_steps=30)


def test_fit_gp_returns_state(seed):
    X, y = _toy_dataset(n=20, d=3, seed=seed)
    state = detopt.bo.fit_gp(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(y), _noise(20), **_GP_CFG)
    assert isinstance(state, detopt.bo.GPState)
    hp = state.hyper_parameters
    assert hp._fields == ("log_lengthscale", "log_amplitude")  # noise not fitted
    assert hp.log_lengthscale.shape == (3,) and hp.log_amplitude.shape == ()
    assert np.all(np.isfinite(np.asarray(hp.log_lengthscale)))
    assert np.isfinite(float(hp.log_amplitude))
    # Refit on full data => state can predict on all training points.
    mean, var = detopt.bo.gp_predict(state, jnp.asarray(X))
    assert mean.shape == (20,) and np.all(np.isfinite(np.asarray(mean)))


def test_fit_gp_fallback_for_tiny_n(seed):
    """With fewer than 2 points there's nothing to hold out -> prior midpoint."""
    X = np.zeros((1, 3), dtype="float32")
    y = np.zeros((1,), dtype="float32")
    state = detopt.bo.fit_gp(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(y), _noise(1), **_GP_CFG)
    ls_mid = 0.5 * sum(_GP_CFG["log_lengthscale_prior_bounds"])
    amp_mid = 0.5 * sum(_GP_CFG["log_amplitude_prior_bounds"])
    hp = state.hyper_parameters
    np.testing.assert_allclose(np.asarray(hp.log_lengthscale), np.full((3,), ls_mid, "float32"))
    np.testing.assert_allclose(float(hp.log_amplitude), amp_mid, rtol=1e-6)


def test_fit_gp_loo_in_low_data_regime(seed):
    """For 2 <= n < 2*n_folds, fit via leave-one-out (uses all n points, no drop)."""
    # n=7 with n_folds=5 -> LOO (7 folds); a real fit, not the prior midpoint.
    X, y = _toy_dataset(n=7, d=3, seed=seed)
    state = detopt.bo.fit_gp(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(y), _noise(7), **_GP_CFG)
    ls_mid = 0.5 * sum(_GP_CFG["log_lengthscale_prior_bounds"])
    hp = state.hyper_parameters
    assert np.all(np.isfinite(np.asarray(hp.log_lengthscale)))
    # The fit moved off the prior midpoint (LOO actually optimised the NLL).
    assert not np.allclose(np.asarray(hp.log_lengthscale), ls_mid)
    mean, var = detopt.bo.gp_predict(state, jnp.asarray(X))
    assert mean.shape == (7,) and np.all(np.asarray(var) > 0)


def test_gp_posterior_interpolates_training_points(seed):
    """At a training point the predictive variance must stay below the prior."""
    X, y = _toy_dataset(n=20, d=3, seed=seed)
    state = detopt.bo.fit_gp(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(y), _noise(20), **_GP_CFG)
    mean, var = detopt.bo.gp_predict(state, jnp.asarray(X[:3]))
    assert mean.shape == (3,) and var.shape == (3,)
    # The GP is scale-agnostic now: prior variance is the bare amplitude^2.
    prior_var = float(np.exp(2.0 * float(state.hyper_parameters.log_amplitude)))
    assert float(jnp.max(var)) <= prior_var + 1e-6


def test_optimise_ei_returns_point_within_bounds(seed):
    X, y = _toy_dataset(n=20, d=3, seed=seed)
    state = detopt.bo.fit_gp(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(y), _noise(20), **_GP_CFG)
    x_best, ei_val = detopt.bo.optimise_ei(jax.random.PRNGKey(seed + 1), state, **_EI_CFG)
    arr = np.asarray(x_best)
    assert arr.shape == (3,)
    # EI always searches the unit cube [0, 1]^d.
    assert np.all(arr >= 0.0 - 1e-5) and np.all(arr <= 1.0 + 1e-5)
    assert np.isfinite(ei_val)


def test_per_observation_noise(seed):
    """Larger per-observation noise (std) -> looser interpolation of training targets."""
    X, y = _toy_dataset(n=20, d=3, seed=seed)
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    small, large = _noise(20, 1e-2), _noise(20, 0.7)

    hp = detopt.bo.fit_gp(jax.random.PRNGKey(seed), Xj, yj, small, **_GP_CFG).hyper_parameters

    f_small = detopt.bo.gp_factorize(Xj, yj, hp, small)
    f_large = detopt.bo.gp_factorize(Xj, yj, hp, large)
    pred_small, _ = detopt.bo.gp_predict(f_small, Xj)
    pred_large, _ = detopt.bo.gp_predict(f_large, Xj)
    err_small = float(jnp.mean(jnp.abs(pred_small - yj)))
    err_large = float(jnp.mean(jnp.abs(pred_large - yj)))
    assert err_large > err_small


def test_refit_gp_warm_start(seed):
    """refit_gp warm-starts from a state and returns a usable GPState."""
    X, y = _toy_dataset(n=30, d=3, seed=seed)
    Xj, yj, noise = jnp.asarray(X), jnp.asarray(y), _noise(30)
    state = detopt.bo.fit_gp(jax.random.PRNGKey(seed), Xj, yj, noise, **_GP_CFG)

    refit = detopt.bo.refit_gp(jax.random.PRNGKey(seed + 1), state, Xj, yj, noise, **_REFIT_CFG)
    assert isinstance(refit, detopt.bo.GPState)
    hp = refit.hyper_parameters
    assert np.all(np.isfinite(np.asarray(hp.log_lengthscale)))
    assert np.isfinite(float(hp.log_amplitude))
    mean, var = detopt.bo.gp_predict(refit, Xj)
    assert mean.shape == (30,)
    assert np.all(np.isfinite(np.asarray(mean))) and np.all(np.asarray(var) > 0)
