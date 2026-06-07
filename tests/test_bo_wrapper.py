"""Tests for detopt.bo.BayesianOptimizer (owns X-normalisation; y is raw)."""

import numpy as np

from detopt.bo import BayesianOptimizer

_GP_CFG = dict(
    n_folds=5,
    n_restarts=2,
    n_steps=20,
    log_lengthscale_prior_bounds=(-2.0, 2.0),
    log_amplitude_prior_bounds=(-2.0, 2.0),
)
_EI_CFG = dict(n_restarts=4, n_steps=20)


def _make_bo(d=3, bound=3.0, n_init=5, seed=0):
    bounds = np.stack([np.full(d, -bound), np.full(d, bound)], axis=1)
    return BayesianOptimizer(bounds, gp=_GP_CFG, ei=_EI_CFG, n_init=n_init, seed=seed)


def test_x_normalisation_round_trip():
    bo = _make_bo(bound=3.0)
    X = np.array([[-3.0, 0.0, 3.0], [1.0, -2.0, 2.5]], dtype="float32")
    np.testing.assert_allclose(bo.to_nominal(bo.to_unit(X)), X, atol=1e-5)
    # Edges of the box map to the unit-cube corners.
    np.testing.assert_allclose(bo.to_unit(np.full(3, -3.0)), np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(bo.to_unit(np.full(3, 3.0)), np.ones(3), atol=1e-6)


def test_append_requires_noise():
    """noise is mandatory (the GP is heteroscedastic)."""
    bo = _make_bo()
    import pytest

    with pytest.raises(TypeError):
        bo.append(np.zeros((2, 3), "float32"), np.zeros(2, "float32"))


def test_propose_random_during_init():
    """Before ``n_init`` observations, proposals are random and carry no GP info."""
    bo = _make_bo(d=3, bound=3.0, n_init=5)
    x = bo.propose()
    assert x.shape == (3,)
    assert np.all(x >= -3.0 - 1e-5) and np.all(x <= 3.0 + 1e-5)
    assert bo.last_info is None


def test_propose_uses_gp_after_init():
    """Once enough points exist, propose() fits the GP and returns a bounded design."""
    rng = np.random.default_rng(1)
    bo = _make_bo(d=2, bound=3.0, n_init=5)
    for _ in range(12):
        x = rng.uniform(-3.0, 3.0, size=2).astype("float32")
        # Minimise ||x||^2 (a simple convex objective).
        bo.append(x, float(np.sum(x**2)), noise=1e-2)
    x_next = bo.propose()
    assert x_next.shape == (2,)
    assert np.all(x_next >= -3.0 - 1e-5) and np.all(x_next <= 3.0 + 1e-5)
    assert bo.last_info is not None
    assert np.isfinite(bo.last_info["ei"])
