"""Tests for the DebugDetector (debug-detector-spec.md)."""

import numpy as np
import jax
import jax.numpy as jnp

import detopt
from detopt.detector.debug import DebugEvent, DebugTarget


def _make():
    return detopt.detector.DebugDetector()


def test_contract_surface_and_shapes():
    d = _make()
    assert d.design_dim() == d.n_stations + d.n_views_per_station + 1  # 9
    assert d.max_hits_per_event == 2 * d.n_stations * d.n_views_per_station  # 32
    # event_spec is a DebugEvent of ShapeDtypeStruct: int [station, view, straw] + float energy.
    es = d.event_spec()
    assert isinstance(es, DebugEvent)
    assert es.station.shape == (d.max_hits_per_event,) and es.station.dtype == np.int32
    assert es.energy.dtype == np.float32
    assert d.target_dim() == 6
    assert d.ground_truth_dim() == 6  # conditioning == 6-vec target
    # combine -> [energy, norm_station_z, wire_y_left, wire_y_right, field_strength]
    assert d.combined_feature_dim == 5
    assert d.combined_event_shape() == (d.max_hits_per_event, 5)
    for name in (
        "__call__",
        "encode_design",
        "decode_design",
        "combine",
        "combine_encoded",
        "loss",
        "metric",
        "normalize_ground_truth",
    ):
        assert hasattr(d, name)
    assert not hasattr(d, "normalize")  # event normalization folded into combine_encoded


def test_call_returns_records_with_shapes():
    d = _make()
    B = 16
    design = np.broadcast_to(d.get_current_design_array()[None, :], (B, d.design_dim()))
    gt, event, mask, y = d(0, design)
    assert isinstance(event, DebugEvent) and isinstance(y, DebugTarget) and isinstance(gt, DebugTarget)
    assert event.station.shape == (B, d.max_hits_per_event) and event.station.dtype == np.int32
    assert mask.shape == (B, d.max_hits_per_event)
    assert y.vertex.shape == (B, 3) and y.momentum.shape == (B, 3)
    assert int(mask.max()) <= 1 and int(mask.min()) >= 0
    # padded slots are zeroed in every event field
    m0 = np.asarray(mask) == 0
    for f in event:
        assert np.all(np.asarray(f)[m0] == 0)


def test_encode_decode_roundtrip_and_jittable():
    d = _make()
    phys = d.get_current_design_array()
    enc = jax.jit(d.encode_design)(jnp.asarray(phys))
    back = np.asarray(d.flatten_design(d.decode_design(enc)))  # decode -> DebugDesign -> flat
    np.testing.assert_allclose(back, phys, rtol=1e-3, atol=1e-2)
    assert enc.shape == (d.design_dim(),)


def test_combine_encoded_shape_and_differentiable():
    d = _make()
    B, M = 4, d.max_hits_per_event
    design = np.broadcast_to(d.get_current_design_array()[None, :], (B, d.design_dim()))
    _gt, event, _mask, _y = d(1, design)
    d_enc = jnp.asarray(d.encode_design(d.get_current_design_array()))
    feats = d.combine_encoded(event, d_enc)
    assert feats.shape == (B, M, d.combined_feature_dim)
    grad = jax.grad(lambda e: jnp.sum(d.combine_encoded(event, e)))(d_enc)
    assert bool(jnp.all(jnp.isfinite(grad))) and float(jnp.sum(jnp.abs(grad))) > 0.0


def test_acceptance_meets_90_percent():
    """Spec target: >=90% of events have both daughters hit all 4 stations."""
    d = _make()
    acc = np.mean([d.acceptance_fraction(s, 4000) for s in range(3)])
    assert acc >= 0.90, f"acceptance {acc:.3f} below 0.90 target"
