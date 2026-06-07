"""Tests for the DebugDetector (debug-detector-spec.md)."""

import numpy as np
import jax
import jax.numpy as jnp

import detopt


def _make():
    return detopt.detector.DebugDetector()


def test_contract_surface_and_shapes():
    d = _make()
    assert d.design_dim() == d.n_stations + d.n_views_per_station + 1  # 9
    assert d.max_hits_per_event == 2 * d.n_stations * d.n_views_per_station  # 32
    assert d.event_shape() == (d.max_hits_per_event, 4)
    assert d.raw_feature_dim == 4
    assert d.target_shape() == (6,)
    # combine -> [energy, norm_station_z, norm_station_y, norm_tilt, norm_B]
    assert d.combined_feature_dim == 5
    assert d.combined_event_shape() == (d.max_hits_per_event, 5)
    for name in (
        "__call__",
        "encode_design",
        "decode_design",
        "normalize",
        "combine",
        "loss",
        "metric",
    ):
        assert hasattr(d, name)


def test_call_returns_four_tuple_with_shapes():
    d = _make()
    B = 16
    design = np.broadcast_to(d.get_current_design_array()[None, :], (B, d.design_dim()))
    gt, X, mask, y = d(0, design)
    assert gt.shape == (B, 8)
    assert X.shape == (B, d.max_hits_per_event, 4)
    assert mask.shape == (B, d.max_hits_per_event)
    assert y.shape == (B, 6)
    # padded slots are zeroed; real slots within range
    assert int(mask.max()) <= 1 and int(mask.min()) >= 0
    assert np.all(np.asarray(X)[np.asarray(mask) == 0] == 0)


def test_encode_decode_roundtrip_and_jittable():
    d = _make()
    phys = d.get_current_design_array()
    enc = jax.jit(d.encode_design)(jnp.asarray(phys))
    back = np.asarray(d.decode_design(enc))
    np.testing.assert_allclose(back, phys, rtol=1e-3, atol=1e-2)
    assert enc.shape == (d.design_dim(),)


def test_combine_shape_and_differentiable():
    d = _make()
    B, M = 4, d.max_hits_per_event
    design = np.broadcast_to(d.get_current_design_array()[None, :], (B, d.design_dim()))
    _gt, X, _mask, _y = d(1, design)
    X_norm = d.normalize(jnp.asarray(X))
    d_enc = jnp.asarray(d.encode_design(d.get_current_design_array()))
    feats = d.combine(X_norm, d_enc)
    assert feats.shape == (B, M, d.combined_feature_dim)
    grad = jax.grad(lambda e: jnp.sum(d.combine(X_norm, e)))(d_enc)
    assert bool(jnp.all(jnp.isfinite(grad))) and float(jnp.sum(jnp.abs(grad))) > 0.0


def test_acceptance_meets_90_percent():
    """Spec target: >=90% of events have both daughters hit all 4 stations."""
    d = _make()
    acc = np.mean([d.acceptance_fraction(s, 4000) for s in range(3)])
    assert acc >= 0.90, f"acceptance {acc:.3f} below 0.90 target"
