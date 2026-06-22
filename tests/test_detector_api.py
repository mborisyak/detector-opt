"""Tests for the Detector contract from detector-spec.md.

Event generation (``__call__``) needs a data source and is not exercised here;
these cover the differentiable / network-facing surface: record specs, design
encode/decode, combine/combine_encoded, and the loss/metric dicts.
"""

import inspect

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import detopt
from detopt.detector.straw import StrawEvent, StrawGroundTruth


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


def _fab_event(d, B, rng):
    """A fabricated batched ``StrawEvent`` (random valid hit address + TDC) for combine tests."""
    M = d.max_hits_per_event
    ri = lambda hi: rng.integers(0, hi, (B, M)).astype(np.int32)
    return StrawEvent(
        station=ri(d.n_stations),
        view=ri(d.n_views_per_station),
        layer=ri(d.n_layers_per_view),
        straw=ri(d.n_straws),
        tdc=(440.0 + 30.0 * rng.standard_normal((B, M))).astype(np.float32),
    )


def test_contract_surface():
    d = _make_detector()
    # No mutable "current design".
    assert not hasattr(d, "update_from_yaml_design")
    for name in (
        "__call__",
        "encode_design",
        "decode_design",
        "combine",
        "combine_encoded",
        "event_spec",
        "target_spec",
        "ground_truth_spec",
        "loss",
        "metric",
        "normalize_target",
        "denormalize_predictions",
        "normalize_ground_truth",
    ):
        assert hasattr(d, name), f"missing {name!r}"
    # Event normalization is folded into combine_encoded -- no standalone event normalize/denormalize.
    assert not hasattr(d, "normalize")
    assert not hasattr(d, "denormalize")


def test_shape_invariants():
    d = _make_detector()
    # max_hits_per_event is a decoupled constructor param (default 384).
    assert d.max_hits_per_event == 384
    # event_spec is a StrawEvent of ShapeDtypeStruct with honest dtypes (int address + float TDC).
    es = d.event_spec()
    assert isinstance(es, StrawEvent)
    assert es.station.shape == (d.max_hits_per_event,) and es.station.dtype == np.int32
    assert es.tdc.dtype == np.float32
    assert d.target_dim() == 6  # HNL [vertex(3), momentum(3)]
    assert d.ground_truth_dim() == 7  # [mass(1), p(3), vertex(3)]
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
    # decode_design returns a Design namedtuple now; flatten it back to compare with the flat physical.
    back = np.asarray(d.flatten_design(d.decode_design(enc)))
    np.testing.assert_allclose(back, phys, rtol=1e-3, atol=1e-2)


def test_design_record_contract():
    d = _make_detector()
    spec = d.design_spec()  # Design namedtuple of jax.ShapeDtypeStruct
    assert sum(int(np.prod(leaf.shape)) for leaf in spec) == d.design_dim()
    assert set(spec._fields) == set(d.design_bounds())  # bounds keyed by the Design field names
    # flatten/unflatten round-trip on the flat physical vector (via the tensor codec)
    a = np.arange(d.design_dim(), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(d.flatten_design(d.unflatten_design(a))), a)
    # decode -> Design namedtuple whose fields are the spec fields
    enc = jnp.asarray(np.random.default_rng(0).standard_normal(d.design_dim()).astype("float32"))
    assert d.decode_design(enc)._fields == spec._fields


def test_encode_decode_jittable():
    d = _make_detector()
    phys = jnp.asarray(_nominal_design(d))
    enc = jax.jit(d.encode_design)(phys)
    assert enc.shape == (d.design_dim(),)
    assert bool(jnp.all(jnp.isfinite(enc)))


def test_combine_encoded_shape_and_broadcast():
    d = _make_detector()
    B = 3
    rng = np.random.default_rng(1)
    event = _fab_event(d, B, rng)
    d_enc = jnp.asarray(rng.standard_normal(d.design_dim()).astype("float32"))

    feats_1d = d.combine_encoded(event, d_enc)
    feats_2d = d.combine_encoded(event, jnp.broadcast_to(d_enc[None, :], (B, d.design_dim())))
    assert feats_1d.shape == (B, d.max_hits_per_event, d.combined_feature_dim)  # compact fixed width
    # A 1-D design broadcasts to the per-row design.
    np.testing.assert_allclose(np.asarray(feats_1d), np.asarray(feats_2d), rtol=1e-5)


def test_combine_matches_combine_encoded():
    """``combine(event, design)`` defaults to ``combine_encoded(event, encode_design(design))``."""
    d = _make_detector()
    rng = np.random.default_rng(5)
    event = _fab_event(d, 3, rng)
    enc = d.encode_design(jnp.asarray(_nominal_design(d)))
    a = d.combine(event, d.decode_design(enc))  # physical Design -> encode -> combine_encoded
    b = d.combine_encoded(event, enc)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4)


def test_combine_differentiable_wrt_design():
    d = _make_detector()
    rng = np.random.default_rng(2)
    event = _fab_event(d, 2, rng)
    d_enc = jnp.asarray(rng.standard_normal(d.design_dim()).astype("float32"))

    grad = jax.grad(lambda e: jnp.sum(d.combine_encoded(event, e)))(d_enc)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.sum(jnp.abs(grad))) > 0.0


def test_target_roundtrip_and_record_type():
    """normalize_target(Target) -> flat array; denormalize_predictions(array) -> Target."""
    d = _make_detector()
    from detopt.detector.straw import HNLTarget

    rng = np.random.default_rng(7)
    t = HNLTarget(
        vertex=rng.standard_normal((5, 3)).astype(np.float32), momentum=rng.standard_normal((5, 3)).astype(np.float32)
    )
    tn = d.normalize_target(t)
    assert tn.shape == (5, d.target_dim())
    back = d.denormalize_predictions(tn)
    assert isinstance(back, HNLTarget)
    np.testing.assert_allclose(np.asarray(back.vertex), np.asarray(t.vertex), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(back.momentum), np.asarray(t.momentum), rtol=1e-3, atol=1e-3)


def test_normalize_ground_truth():
    d = _make_detector()
    rng = np.random.default_rng(6)
    gt = StrawGroundTruth(
        mass=rng.standard_normal((4, 1)).astype(np.float32),
        momentum=rng.standard_normal((4, 3)).astype(np.float32),
        vertex=rng.standard_normal((4, 3)).astype(np.float32),
    )
    out = d.normalize_ground_truth(gt)
    assert out.shape == (4, d.ground_truth_dim())  # [mass, p(3), vertex(3)] = 7


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
    ``simulate_debug`` / ``_run_solver`` return the RAW ``(n, M, 5)`` float buffer (the
    typed ``Event`` packing happens only in ``sample_events`` / ``__call__``).
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
    flat = lambda t: np.concatenate([np.asarray(x) for x in t], axis=-1)  # Target record -> flat array
    _, _, _, t_train = d(0, design, pool="train")
    _, _, _, t_val = d(0, design, pool="val")
    assert not np.allclose(flat(t_train), flat(t_val))
    # the default pool (no arg) matches the first key
    _, _, _, t_default = d(0, design)
    assert np.allclose(flat(t_default), flat(t_train))
