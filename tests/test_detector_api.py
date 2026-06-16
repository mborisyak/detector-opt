"""Tests for the Detector contract from detector-spec.md.

Event generation (``__call__``) needs a data source and is not exercised here;
these cover the differentiable / network-facing surface: shapes, design
encode/decode, event normalize, combine, and the loss/metric dicts.
"""

import inspect

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import detopt


def _make_detector():
    return detopt.detector.FreeStrawDetector(
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


def _nominal_design(d):
    """A neutral design for the contract tests, built outside the detector."""
    return detopt.detector.free_design_array([8407.0, 8607.0, 9307.0, 9507.0], n_layers_per_view=d.n_layers_per_view, B=d.max_B)


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
        "normalize_target",
        "denormalize_predictions",
    ):
        assert hasattr(d, name), f"missing {name!r}"


def test_shape_invariants():
    d = _make_detector()
    # max_hits_per_event is now a decoupled constructor param (default 384), not 2*max_particles*n_layers.
    assert d.max_hits_per_event == 384
    assert d.event_shape() == (d.max_hits_per_event, 5)
    assert d.raw_feature_dim == 5
    assert d.target_shape() == (6,)
    # design = positions(n_layers) + angles(n_layers) + B(1)
    assert d.design_dim() == 2 * d.n_layers + 1
    # combine() gathers per-hit design context into a compact fixed width
    # [TDC, norm_z, wire_y_left, wire_y_right] (field is fixed -> not a feature).
    assert d.combined_event_shape() == (d.max_hits_per_event, 4)
    assert d.combined_feature_dim == 4


def test_encode_decode_roundtrip():
    d = _make_detector()
    phys = _nominal_design(d)
    enc = d.encode_design(phys)
    # decode_design returns a named dict now; flatten it back to compare with the flat physical.
    back = np.asarray(d.flatten_design(d.decode_design(enc)))
    np.testing.assert_allclose(back, phys, rtol=1e-3, atol=1e-2)


def test_design_dict_contract():
    d = _make_detector()
    spec = d.design_spec()  # Mapping[str, Shape]
    assert sum(int(np.prod(shape)) for shape in spec.values()) == d.design_dim()
    assert set(spec) == set(d.design_bounds())
    # flatten/unflatten round-trip on the flat physical vector
    a = np.arange(d.design_dim(), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(d.flatten_design(d.unflatten_design(a))), a)
    # decode -> dict keyed by the spec names
    enc = jnp.asarray(np.random.default_rng(0).standard_normal(d.design_dim()).astype("float32"))
    assert set(d.decode_design(enc)) == set(spec)


def test_encode_decode_jittable():
    d = _make_detector()
    phys = jnp.asarray(_nominal_design(d))
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
    assert feats_1d.shape == (B, M, d.combined_feature_dim)  # compact fixed width
    # A 1-D design broadcasts to the per-row design.
    np.testing.assert_allclose(np.asarray(feats_1d), np.asarray(feats_2d), rtol=1e-5)


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
    # loss compares predictions against the ALREADY-NORMALIZED target (the caller normalizes).
    target = jnp.asarray(rng.standard_normal((5, 6)).astype("float32"))
    pred = jnp.asarray(rng.standard_normal((5, 6)).astype("float32"))
    out = d.loss(pred, target)
    expected = jnp.mean(jnp.square(pred - target), axis=-1)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5)
    assert out.shape == (5,)


def test_metric_is_per_sample_dict():
    d = _make_detector()
    rng = np.random.default_rng(4)
    target = jnp.asarray(rng.standard_normal((5, 6)).astype("float32"))
    pred = jnp.asarray(rng.standard_normal((5, 6)).astype("float32"))
    m = d.metric(pred, target)
    assert set(m) == set(d.metric_labels())  # keys == declared labels
    for v in m.values():
        assert v.shape == (5,)  # per-sample
    np.testing.assert_allclose(np.asarray(m["loss"]), np.asarray(d.loss(pred, target)), rtol=1e-5)


def test_no_split_or_mutable_state():
    """The detector mirrors DebugDetector: no train/val split, no loader, no
    'current yaml design' state -- design is always passed in."""
    d = _make_detector()
    assert "split" not in inspect.signature(d.__call__).parameters
    for removed in ("loader", "get_current_yaml_design", "encode_yaml_design", "yaml_design_shape"):
        assert not hasattr(d, removed), f"unexpected leftover: {removed!r}"


def _synthetic_event(n_primaries=2):
    """A tiny hand-built daughter_data dict: a few charged tracks fired downstream
    through the detector (no data file needed)."""
    pos = np.tile(np.array([0.0, 0.0, 8300.0], np.float32), (n_primaries, 1))
    pos[:, 0] = np.linspace(-5.0, 5.0, n_primaries)  # spread in x so they cross straws
    mom = np.tile(np.array([20.0, 0.0, 3000.0], np.float32), (n_primaries, 1))  # MeV/c, +z
    return {
        "masses": np.full(n_primaries, 139.57, np.float32),  # pi+-
        "charges": np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_primaries)], np.float32),
        "positions": pos,
        "momenta": mom,
        "times": np.zeros(n_primaries, np.float32),
        "offsets": np.array([0, n_primaries], np.int32),
    }


def test_simulate_debug_outputs():
    """simulate_debug adds a per-hit process_id buffer + a flat MC-particle tree
    (process_id / parent_id / properties) without disturbing the default path.

    Secondary channels are off here for a clean, deterministic 2-primary event (the
    physics rates are matched/validated elsewhere); this just checks the API surface.
    """
    d = detopt.detector.FreeStrawDetector(
        n_stations=4,
        n_views_per_station=4,
        n_layers_per_view=2,
        n_straws_per_layer=200,
        max_B=0.15,
        B_sigma=300.0,
        z0=8957.0,
        layer_bounds=(8200.0, 9750.0),
        max_particles=8,
        max_hits_per_event=64,
        enable_decay=False,
        scatter_xX0=0.0,
        lambda_conv_cm=0.0,
        delta_Tcut=1e9,
        noise_rate=0.0,
    )
    design = _nominal_design(d)[None, :]
    dd = _synthetic_event(2)
    out = d.simulate_debug(dd, design, np.random.default_rng(0))

    # per-hit process_id is parallel to mask, width = max_hits_per_event.
    mask = out["mask"].astype(bool)
    assert out["process_ids"].shape == out["mask"].shape == (1, 64)
    assert out["process_ids"].dtype == np.int32
    assert 0 < int(mask.sum()) <= 64  # tracks fire straws, capped at max_hits_per_event

    # tree: one row per tracked particle (here just the 2 primaries), arrays aligned.
    P = len(out["pdg"])
    assert P == 2  # secondaries off -> exactly the two primaries
    for key in ("process_id", "parent_id", "n_hits", "t0", "event_index"):
        assert len(out[key]) == P
    assert out["momentum"].shape == (P, 3) and out["position"].shape == (P, 3)

    # primaries are kPPrimary=0 with parent -1.
    assert np.all(out["process_id"] == 0)
    assert np.all(out["parent_id"] == -1)

    # every emitted hit carries a valid process code; with min-TDC dedup the emitted
    # count is <= the particles' total tube entries (shared straws collapse to one).
    assert set(np.unique(out["process_ids"][mask]).tolist()) <= {0}
    assert int(mask.sum()) <= int(out["n_hits"].sum())

    # the operational solver path: build a transient InputEvents from the pool + a
    # (1,2) boundary span; returns (X, mask, None) with X width = max_hits_per_event.
    ie = d._make_input_events(dd)
    boundaries = np.array([[0, len(dd["masses"])]], dtype=np.int32)
    X, m, traj = d._run_solver(boundaries, design, np.random.default_rng(0), input_events=ie)
    assert X.shape == (1, 64, 5) and m.shape == (1, 64) and traj is None


def test_pool_split_resolve():
    """resolve_pool_split normalizes Sequence/Mapping/None into fractions summing to 1."""
    from detopt.detector.common import Detector

    assert Detector.resolve_pool_split(None) == {0: 1.0}
    r = Detector.resolve_pool_split([3, 1])  # Sequence -> int keys, normalized
    assert set(r) == {0, 1} and abs(sum(r.values()) - 1.0) < 1e-6 and abs(r[0] - 0.75) < 1e-6
    r = Detector.resolve_pool_split({"train": 0.9, "val": 0.1})
    assert abs(r["train"] - 0.9) < 1e-6 and abs(r["val"] - 0.1) < 1e-6


def test_debug_detector_pools_disjoint():
    """DebugDetector pools are independent event streams: different pools -> different events."""
    d = detopt.detector.DebugDetector(pool_split={"train": 0.8, "val": 0.2})
    assert list(d.pool_split) == ["train", "val"]
    design = np.zeros((64, d.design_dim()), np.float32)
    _, _, _, t_train = d(0, design, pool="train")
    _, _, _, t_val = d(0, design, pool="val")
    assert not np.allclose(np.asarray(t_train), np.asarray(t_val))
    # the default pool (no arg) matches the first key
    _, _, _, t_default = d(0, design)
    assert np.allclose(np.asarray(t_default), np.asarray(t_train))
