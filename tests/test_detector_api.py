"""Tests for the Detector contract from detector-spec.md.

Event generation (``__call__``) needs a data source and is not exercised here;
these cover the differentiable / network-facing surface: shapes, design
encode/decode, event normalize, combine, and the loss/metric dicts.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import detopt


def _make_detector():
    return detopt.detector.StrawDetector(
        station_z=[8407.0, 8607.0, 9307.0, 9507.0],
        n_views_per_station=4,
        n_layers_per_view=2,
        n_straws_per_layer=200,
        max_particles=2,
        max_B=0.15,
        B_sigma=300.0,
        z0=8957.0,
        layer_bounds=(8200.0, 9750.0),
        constrain_stereo_angles=True,
        optimize_stations=True,
        optimize_gaps=True,
        optimize_bfield=True,
    )


def test_contract_surface():
    d = _make_detector()
    # No mutable "current design".
    assert not hasattr(d, "update_from_yaml_design")
    for name in (
        "__call__",
        "encode_design",
        "decode_design",
        "normalize",
        "denormalize",
        "combine",
        "loss",
        "metric",
        "target_mean",
        "target_std",
    ):
        assert hasattr(d, name), f"missing {name!r}"


def test_shape_invariants():
    d = _make_detector()
    assert d.max_hits_per_event == 2 * d.max_particles * d.n_layers
    assert d.event_shape() == (d.max_hits_per_event, 5)
    assert d.raw_feature_dim == 5
    assert d.target_shape() == (6,)
    # design = positions(n_layers) + angles(n_layers) + B(1)
    assert d.design_dim() == 2 * d.n_layers + 1
    assert d.combined_event_shape() == (d.max_hits_per_event, 5 + d.design_dim())
    assert d.combined_feature_dim == 5 + d.design_dim()


def test_encode_decode_roundtrip():
    d = _make_detector()
    phys = d.get_current_design_array()
    enc = d.encode_design(phys)
    back = np.asarray(d.decode_design(enc))
    np.testing.assert_allclose(back, phys, rtol=1e-3, atol=1e-2)


def test_encode_decode_jittable():
    d = _make_detector()
    phys = jnp.asarray(d.get_current_design_array())
    enc = jax.jit(d.encode_design)(phys)
    assert enc.shape == (d.design_dim(),)
    assert bool(jnp.all(jnp.isfinite(enc)))


def test_normalize_denormalize_roundtrip():
    d = _make_detector()
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((4, d.max_hits_per_event, 5)).astype("float32"))
    back = d.denormalize(d.normalize(X))
    assert float(jnp.max(jnp.abs(X - back))) < 1e-3


def test_combine_shape_and_broadcast():
    d = _make_detector()
    B, M = 3, d.max_hits_per_event
    rng = np.random.default_rng(1)
    X_norm = jnp.asarray(rng.standard_normal((B, M, 5)).astype("float32"))
    d_enc = jnp.asarray(rng.standard_normal(d.design_dim()).astype("float32"))

    feats_1d = d.combine(X_norm, d_enc)
    feats_2d = d.combine(X_norm, jnp.broadcast_to(d_enc[None, :], (B, d.design_dim())))
    assert feats_1d.shape == (B, M, d.combined_feature_dim)
    np.testing.assert_allclose(np.asarray(feats_1d), np.asarray(feats_2d), rtol=1e-5)
    # The first 5 columns are exactly the normalised event features.
    np.testing.assert_allclose(np.asarray(feats_1d[..., :5]), np.asarray(X_norm), rtol=1e-5)


def test_combine_differentiable_wrt_design():
    d = _make_detector()
    B, M = 2, d.max_hits_per_event
    rng = np.random.default_rng(2)
    X_norm = jnp.asarray(rng.standard_normal((B, M, 5)).astype("float32"))
    d_enc = jnp.asarray(rng.standard_normal(d.design_dim()).astype("float32"))

    grad = jax.grad(lambda e: jnp.sum(d.combine(X_norm, e)))(d_enc)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.sum(jnp.abs(grad))) > 0.0


def test_loss_is_array_and_matches_mse():
    d = _make_detector()
    rng = np.random.default_rng(3)
    target = jnp.asarray(rng.standard_normal((5, 6)).astype("float32") * 10.0)
    pred = jnp.asarray(rng.standard_normal((5, 6)).astype("float32"))
    out = d.loss(pred, target)
    expected = jnp.mean(jnp.square(pred - d.normalize_target(target)), axis=-1)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5)
    assert out.shape == (5,)


def test_metric_is_array():
    d = _make_detector()
    rng = np.random.default_rng(4)
    target = jnp.asarray(rng.standard_normal((5, 6)).astype("float32") * 10.0)
    pred = jnp.asarray(rng.standard_normal((5, 6)).astype("float32"))
    m = d.metric(pred, target)
    assert m.shape == (5,)


def test_yaml_helpers_are_pure():
    d = _make_detector()
    snapshot = (
        tuple(d.station_z),
        tuple(d.view_angles),
        float(d.layer_z_gap),
        float(d.view_z_gap),
        float(d.max_B),
    )
    yaml_params = d.get_current_yaml_design()
    enc = d.encode_yaml_design(yaml_params)
    dec = d.decode_yaml_design(enc)
    _ = d.yaml_to_design_array(dec)
    after = (
        tuple(d.station_z),
        tuple(d.view_angles),
        float(d.layer_z_gap),
        float(d.view_z_gap),
        float(d.max_B),
    )
    assert snapshot == after
