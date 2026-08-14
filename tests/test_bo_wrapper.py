"""Tests for detopt.bo.BayesianOptimizer (owns X-normalisation; y is raw).

The class works in ONE space, the scaled cube ``[0, 1]^d``: :meth:`append` takes scaled designs and
:meth:`propose` returns one, with no transform inside. These tests pin that contract, and in
particular that the cube is searched UNIFORMLY -- the defect this replaced handed the optimiser a
quantile-encoded box, so a uniform draw piled against the design bounds and a third of the
evaluations landed where nothing could be learned.
"""

import numpy as np
import pytest

from detopt.bo import BayesianOptimizer

_GP_CFG = dict(
    n_folds=5,
    n_restarts=2,
    n_steps=20,
    log_lengthscale_prior_bounds=(-2.0, 2.0),
    log_amplitude_prior_bounds=(-2.0, 2.0),
)
_EI_CFG = dict(n_restarts=4, n_steps=20)


def _make_bo(d=3, n_init=5):
    """``propose`` is seeded PER CALL by the driver, so there is no constructor seed to pass."""
    return BayesianOptimizer(d, gp=_GP_CFG, ei=_EI_CFG, n_init=n_init)


def test_append_requires_noise():
    """noise is mandatory (the GP is heteroscedastic)."""
    bo = _make_bo()
    with pytest.raises(TypeError):
        bo.append(np.zeros((2, 3), "float32"), np.zeros(2, "float32"))


def test_propose_random_during_init():
    """Before ``n_init`` observations, proposals are random and carry no GP info."""
    bo = _make_bo(d=3, n_init=5)
    x = bo.propose(0)
    assert x.shape == (3,)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert bo.last_info is None


def test_propose_uses_gp_after_init():
    """Once enough points exist, propose() fits the GP and returns a scaled design in the cube."""
    rng = np.random.default_rng(1)
    bo = _make_bo(d=2, n_init=5)
    for _ in range(12):
        x = rng.uniform(0.0, 1.0, size=2).astype("float32")
        # Minimise ||x - 0.5||^2 (a simple convex objective with an INTERIOR optimum).
        bo.append(x, float(np.sum((x - 0.5) ** 2)), noise=1e-2)
    x_next = bo.propose(0)
    assert x_next.shape == (2,)
    assert np.all(x_next >= 0.0) and np.all(x_next <= 1.0)
    assert bo.last_info is not None
    assert np.isfinite(bo.last_info["ei"])


def test_gp_proposals_stay_in_the_cube():
    """Every EI-driven proposal is inside ``[0, 1]^d``, including against an objective whose optimum
    sits ON a corner -- the EI polish must clip rather than run out of the box."""
    rng = np.random.default_rng(2)
    bo = _make_bo(d=3, n_init=5)
    for step in range(20):
        x = np.asarray(bo.propose(3 + step), dtype=np.float32)
        assert np.all(x >= 0.0) and np.all(x <= 1.0), x
        bo.append(x, float(np.sum(x)), noise=1e-2)  # minimised at the all-zero CORNER


def test_initial_proposals_are_uniform_in_the_cube():
    """The initial random proposals are UNIFORM in the cube -- so they are a uniform NOMINAL design.

    This is the regression test for the parameterisation defect: under the old quantile encoding a
    uniform draw in the searched box put ~57% of proposals in the outer 20% of every coordinate and
    only ~6% in the middle 14%. Affine scaling must reproduce the uniform 20% / 14%.
    """
    xs = []
    for seed in range(200):
        # One scrambled Sobol BLOCK per seed, consumed row by row -- the rows only advance as
        # observations arrive, so each proposal is appended before the next is asked for.
        bo = _make_bo(d=4, n_init=8)
        for _ in range(8):
            x = np.asarray(bo.propose(seed), dtype=np.float64)
            xs.append(x)
            bo.append(x.astype(np.float32), 0.0, noise=1e-2)
    x = np.asarray(xs)  # (1600, 4)

    outer = float(np.mean((x < 0.1) | (x > 0.9)))  # outer 20% of the range
    middle = float(np.mean((x > 0.43) & (x < 0.57)))  # middle 14%
    assert 0.16 < outer < 0.24, f"outer-20% occupancy {outer:.3f}, expected ~0.20"
    assert 0.11 < middle < 0.17, f"middle-14% occupancy {middle:.3f}, expected ~0.14"
    # and no coordinate is systematically off-centre
    assert np.all(np.abs(x.mean(axis=0) - 0.5) < 0.05), x.mean(axis=0)
