"""Tests for the detector-agnostic SetRegressor."""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

import detopt


def test_set_regressor_shape_and_mask_independence(seed):
    """Output shape is ``(B, T)``; padded slots must not influence the prediction."""
    n_features_in, target_dim = 8, 6
    B, M = 4, 32

    reg = detopt.nn.SetRegressor(
        n_features_in=n_features_in,
        target_dim=target_dim,
        features=[[16, 24], [24, 12]],
        p_dropout=None,
        rngs=nnx.Rngs(seed),
    )

    rng = np.random.default_rng(seed)
    features = jnp.asarray(rng.standard_normal((B, M, n_features_in)).astype("float32"))
    # First half of the slots are valid; second half is padded.
    mask = jnp.asarray(np.concatenate([np.ones((B, M // 2)), np.zeros((B, M // 2))], axis=1).astype("int32"))
    pred = reg(features, mask, deterministic=True)
    assert pred.shape == (B, target_dim)

    # Perturbing only the padded slots should not change the prediction.
    perturb = rng.standard_normal((B, M // 2, n_features_in)).astype("float32") * 10.0
    features_perturbed = features.at[:, M // 2 :, :].add(jnp.asarray(perturb))
    pred_perturbed = reg(features_perturbed, mask, deterministic=True)
    err = float(jnp.max(jnp.abs(pred - pred_perturbed)))
    assert err < 1e-4, f"padded-hit perturbation leaked into output: {err}"


def test_set_regressor_is_detector_agnostic(seed):
    """Constructed and called without any detector reference."""
    reg = detopt.nn.SetRegressor(
        n_features_in=5,
        target_dim=3,
        features=[[8, 8]],
        rngs=nnx.Rngs(seed),
    )

    B, M = 2, 7
    features = jnp.ones((B, M, 5), dtype=jnp.float32)
    mask = jnp.ones((B, M), dtype=jnp.int32)
    pred = reg(features, mask, deterministic=True)
    assert pred.shape == (B, 3)


def test_set_regressor_factory_from_detector(seed):
    """``detopt.nn.from_config`` builds a SetRegressor from the detector."""
    detector = detopt.detector.FreeStrawDetector(
        n_stations=4,
        n_views_per_station=4,
        n_layers_per_view=2,
        n_straws_per_layer=200,
        max_particles=2,
        max_B=0.15,
        B_sigma=300.0,
        z0=8957.0,
        layer_bounds=(8200.0, 9750.0),
    )
    rngs = nnx.Rngs(seed)
    reg = detopt.nn.from_config(
        detector,
        config={"set-regressor": {"features": [[8, 8]], "p_dropout": None}},
        rngs=rngs,
    )
    assert isinstance(reg, detopt.nn.SetRegressor)

    B, M, F = 3, detector.max_hits_per_event, detector.combined_feature_dim
    features = jnp.zeros((B, M, F), dtype=jnp.float32)
    mask = jnp.ones((B, M), dtype=jnp.int32)
    pred = reg(features, mask, deterministic=True)
    assert pred.shape == (B, detector.target_dim())


def test_single_model_reports_no_ensemble(seed):
    """A plain SetRegressor declares itself a single model."""
    reg = detopt.nn.SetRegressor(n_features_in=5, target_dim=3, features=[[8]], rngs=nnx.Rngs(seed))
    assert reg.ensemble() is None


def test_set_ensemble_regressor_shape_and_members(seed):
    """``__call__`` maps ``(N, B, M, F) -> (N, B, T)`` and members differ.

    On an identical batch fed to every member, predictions must vary across the
    ensemble axis (members are initialised and trained independently).
    """
    N, B, M, F, T = 4, 3, 16, 8, 6
    reg = detopt.nn.SetRegressor(
        n_features_in=F, target_dim=T, features=[[16, 16], [16, 8]], n_models=N, rngs=nnx.Rngs(seed)
    )
    assert reg.ensemble() == N

    rng = np.random.default_rng(seed)
    one = jnp.asarray(rng.standard_normal((B, M, F)).astype("float32"))
    feats = jnp.broadcast_to(one, (N, B, M, F))
    mask = jnp.ones((N, B, M), dtype=jnp.int32)
    pred = reg(feats, mask, deterministic=True)
    assert pred.shape == (N, B, T)
    assert float(jnp.std(pred, axis=0).mean()) > 1e-3, "members produced identical predictions"


def test_set_ensemble_regressor_mask_independence(seed):
    """Padded slots must not influence any member's prediction."""
    N, B, M, F, T = 3, 2, 32, 5, 4
    reg = detopt.nn.SetRegressor(n_features_in=F, target_dim=T, features=[[12, 12]], n_models=N, rngs=nnx.Rngs(seed))
    rng = np.random.default_rng(seed)
    feats = jnp.asarray(rng.standard_normal((N, B, M, F)).astype("float32"))
    mask = jnp.asarray(np.concatenate([np.ones((N, B, M // 2)), np.zeros((N, B, M // 2))], axis=2).astype("int32"))
    pred = reg(feats, mask, deterministic=True)
    perturb = rng.standard_normal((N, B, M // 2, F)).astype("float32") * 10.0
    feats_p = feats.at[:, :, M // 2 :, :].add(jnp.asarray(perturb))
    pred_p = reg(feats_p, mask, deterministic=True)
    assert float(jnp.max(jnp.abs(pred - pred_p))) < 1e-4


def test_set_ensemble_regressor_factory_and_ensemble_query(seed):
    """``from_config`` builds the ensemble and reports its member count."""
    detector = detopt.detector.DebugDetector()
    reg = detopt.nn.from_config(
        detector,
        config={"set-regressor": {"features": [[8, 8]], "n_models": 5}},
        rngs=nnx.Rngs(seed),
    )
    assert isinstance(reg, detopt.nn.SetRegressor)
    assert reg.ensemble() == 5
