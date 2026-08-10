"""The sklearn driver's analytic EI gradient, checked against JAX autodiff.

``BayesianOptimizer._ei_and_grad`` differentiates Expected Improvement by hand so the
acquisition polish does not have to rely on finite differences (EI underflows to ~1e-15
on a flat surrogate, where differencing is pure cancellation noise). Hand-derived
gradients are exactly the kind of thing that is silently wrong, so this rebuilds the
SAME posterior in JAX from the fitted sklearn parameters and compares against
``jax.grad`` -- an independent implementation, not a perturbation of the same code.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.scipy.linalg import cho_solve as jax_cho_solve  # noqa: E402
from jax.scipy.stats import norm as jax_norm  # noqa: E402

from detopt.bo import BayesianOptimizer  # noqa: E402


@pytest.fixture(autouse=True)
def enable_x64():
    """float64 for the autodiff reference, restored on the way out.

    ``_ei_and_grad`` is numpy float64, so the JAX side has to match or the
    comparison measures float32 rounding rather than the derivation. x64 is a
    *global* JAX flag and there is no context manager for it in jax 0.10, so
    setting it at module scope would silently flip every later test in the
    session -- which breaks :mod:`detopt.bo.jax_gp` (its float32 literals make
    ``lax.cond`` branches disagree on dtype). Hence: on for this module's tests
    only, restored afterwards.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)

_GP_CFG = dict(
    n_folds=5,
    n_restarts=2,
    log_lengthscale_prior_bounds=(-2.0, 2.0),
    log_amplitude_prior_bounds=(-2.0, 2.0),
)
_EI_CFG = dict(n_restarts=4, n_steps=20)


def _fitted(d, n, seed):
    """A fitted sklearn GP on a smooth objective + the centered incumbent."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    bo = BayesianOptimizer(d, gp=_GP_CFG, ei=_EI_CFG, n_init=5, seed=seed)
    X = rng.random((n, d))
    y = np.sin(2.0 * X @ w) + 0.5 * ((X - 0.5) ** 2).sum(1)
    for i in range(n):
        bo.append(X[i], float(y[i]), noise=0.02)
    y_centered = bo.y - bo.y.mean()
    return bo._fit(bo.X, y_centered, bo.noise), float(np.min(y_centered))


def _jax_ei(model, x, y_best):
    """EI rebuilt in JAX from the fitted sklearn parameters (the autodiff target).

    The closed form below is the ARD-RBF one, which is valid because these fixtures build the
    optimiser with no exchangeable block -- there the kernel reduces exactly to ``sigma^2 * RBF``
    (pinned by ``test_without_blocks_it_is_exactly_an_ard_rbf``). Asserted rather than assumed, so
    this reference cannot silently drift out of agreement with the kernel it is checking."""
    assert len(model.kernel_.blocks) == 0, "the ARD closed form below does not hold with a symmetry"
    X_train = jnp.asarray(model.X_train_)
    L = jnp.asarray(model.L_)
    alpha = jnp.asarray(np.ravel(model.alpha_))
    length_scale = jnp.asarray(model.kernel_.coordinate_length_scales())
    amplitude2 = jnp.asarray(float(model.kernel_.constant_value))

    scaled = (X_train - x) / length_scale
    k = amplitude2 * jnp.exp(-0.5 * jnp.sum(scaled * scaled, axis=1))
    v = jax_cho_solve((L, True), k)
    mean = k @ alpha
    sigma = jnp.sqrt(jnp.maximum(amplitude2 - k @ v, 1e-12))
    z = (y_best - mean) / sigma
    return (y_best - mean) * jax_norm.cdf(z) + sigma * jax_norm.pdf(z)


@pytest.mark.parametrize("d, n, seed", [(3, 20, 0), (8, 40, 1), (8, 120, 2)])
def test_ei_gradient_matches_autodiff(d, n, seed):
    model, y_best = _fitted(d, n, seed)
    grad_fn = jax.grad(lambda x: _jax_ei(model, x, y_best))
    rng = np.random.default_rng(seed + 100)
    for _ in range(5):
        x = rng.random(d)
        value, grad = BayesianOptimizer._ei_and_grad(model, x, y_best)
        expected_value = float(_jax_ei(model, jnp.asarray(x), y_best))
        expected_grad = np.asarray(grad_fn(jnp.asarray(x)))
        assert value == pytest.approx(expected_value, rel=1e-9, abs=1e-15)
        # Relative to the gradient's own scale: EI can be tiny, and then so is its slope.
        scale = max(float(np.max(np.abs(expected_grad))), 1e-12)
        assert np.max(np.abs(grad - expected_grad)) / scale < 1e-7
