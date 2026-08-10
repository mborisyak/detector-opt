"""Tests for the Detector contract from detector-spec.md.

Event generation (``__call__``) needs a data source and is not exercised here;
these cover the differentiable / network-facing surface: record specs, design
nominal<->scaled, combine/combine_scaled, and the loss/metric dicts.
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
        "to_scaled",
        "to_nominal",
        "combine",
        "combine_scaled",
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
    # Event normalization is folded into combine_scaled -- no standalone event normalize/denormalize.
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
    assert d.target_dim() == 9  # unified daughter target [vertex(3), p1(3), p2(3)]
    assert d.ground_truth_dim() == 7  # [mass(1), p(3), vertex(3)]
    # design = positions(n_layers) + angles(n_layers) + B(1)
    assert d.design_dim() == 2 * d.n_layers + 1
    # combine() gathers per-hit design context into a compact fixed width
    # [TDC, norm_z, wire_y_left, wire_y_right] (field is fixed -> not a feature).
    assert d.combined_event_shape() == (d.max_hits_per_event, 4)
    assert d.combined_feature_dim == 4


def test_scaled_nominal_roundtrip():
    d = _make_detector()
    phys = _nominal_design(d)
    enc = d.to_scaled(phys)
    assert np.all(np.asarray(enc) >= 0.0) and np.all(np.asarray(enc) <= 1.0)  # scaled IS the unit cube
    # to_nominal returns a Design namedtuple; flatten it back to compare with the flat nominal.
    back = np.asarray(d.flatten_design(d.to_nominal(enc)))
    # The map is affine per coordinate, so the round trip is exact to float32.
    np.testing.assert_allclose(back, phys, rtol=1e-6, atol=1e-3)


def test_scaled_cube_covers_exactly_the_design_bounds():
    """The unit cube's corners ARE the design bounds -- no saturation, nothing outside.

    Under the quantile encoding the box's interior mapped to a design piled against the bounds and
    the corners were only reached as theta -> +-inf. Affine scaling makes ``u = 0`` / ``u = 1`` the
    bounds exactly, and every interior point admissible."""
    d = _make_detector()
    spec, bounds = d.design_spec(), d.design_bounds()
    lo = np.asarray(d._to_nominal_flat(jnp.zeros(d.design_dim(), jnp.float32)))
    hi = np.asarray(d._to_nominal_flat(jnp.ones(d.design_dim(), jnp.float32)))
    offset = 0
    for field, leaf in zip(spec._fields, spec):
        n = int(np.prod(leaf.shape))
        expected_lo, expected_hi = bounds[field]
        np.testing.assert_allclose(lo[offset:offset + n], expected_lo, rtol=1e-6)
        np.testing.assert_allclose(hi[offset:offset + n], expected_hi, rtol=1e-6)
        offset += n
    # A uniform draw in the cube is a uniform NOMINAL design: it stays inside the bounds.
    u = np.random.default_rng(0).uniform(0.0, 1.0, (256, d.design_dim())).astype("float32")
    p = np.asarray(jax.vmap(d._to_nominal_flat)(jnp.asarray(u)))
    assert np.all(np.isfinite(p))
    assert np.all(p >= np.minimum(lo, hi) - 1e-3) and np.all(p <= np.maximum(lo, hi) + 1e-3)


def test_design_record_contract():
    d = _make_detector()
    spec = d.design_spec()  # Design namedtuple of jax.ShapeDtypeStruct
    assert sum(int(np.prod(leaf.shape)) for leaf in spec) == d.design_dim()
    assert set(spec._fields) == set(d.design_bounds())  # bounds keyed by the Design field names
    # flatten/unflatten round-trip on the flat physical vector (via the tensor codec)
    a = np.arange(d.design_dim(), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(d.flatten_design(d.unflatten_design(a))), a)
    # to_nominal -> Design namedtuple whose fields are the spec fields
    enc = jnp.asarray(np.random.default_rng(0).uniform(0.0, 1.0, d.design_dim()).astype("float32"))
    assert d.to_nominal(enc)._fields == spec._fields


def test_scaling_jittable():
    d = _make_detector()
    phys = jnp.asarray(_nominal_design(d))
    enc = jax.jit(d.to_scaled)(phys)
    assert enc.shape == (d.design_dim(),)
    assert bool(jnp.all(jnp.isfinite(enc)))


def test_combine_scaled_shape_and_broadcast():
    d = _make_detector()
    B = 3
    rng = np.random.default_rng(1)
    event = _fab_event(d, B, rng)
    d_enc = jnp.asarray(rng.uniform(0.0, 1.0, d.design_dim()).astype("float32"))

    feats_1d = d.combine_scaled(event, d_enc)
    feats_2d = d.combine_scaled(event, jnp.broadcast_to(d_enc[None, :], (B, d.design_dim())))
    assert feats_1d.shape == (B, d.max_hits_per_event, d.combined_feature_dim)  # compact fixed width
    # A 1-D design broadcasts to the per-row design.
    np.testing.assert_allclose(np.asarray(feats_1d), np.asarray(feats_2d), rtol=1e-5)


def test_combine_matches_combine_scaled():
    """``combine(event, design)`` defaults to ``combine_scaled(event, to_scaled(design))``."""
    d = _make_detector()
    rng = np.random.default_rng(5)
    event = _fab_event(d, 3, rng)
    enc = d.to_scaled(jnp.asarray(_nominal_design(d)))
    a = d.combine(event, d.to_nominal(enc))  # nominal Design -> scale -> combine_scaled
    b = d.combine_scaled(event, enc)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4)


def test_combine_differentiable_wrt_design():
    d = _make_detector()
    rng = np.random.default_rng(2)
    event = _fab_event(d, 2, rng)
    d_enc = jnp.asarray(rng.uniform(0.0, 1.0, d.design_dim()).astype("float32"))

    grad = jax.grad(lambda e: jnp.sum(d.combine_scaled(event, e)))(d_enc)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.sum(jnp.abs(grad))) > 0.0


def test_target_roundtrip_and_record_type():
    """normalize_target(Target) -> flat array; denormalize_predictions(array) -> Target."""
    d = _make_detector()
    from detopt.detector.straw import DaughterTarget

    rng = np.random.default_rng(7)
    t = DaughterTarget(
        vertex=rng.standard_normal((5, 3)).astype(np.float32),
        p1=rng.standard_normal((5, 3)).astype(np.float32),
        p2=rng.standard_normal((5, 3)).astype(np.float32),
    )
    tn = d.normalize_target(t)
    assert tn.shape == (5, d.target_dim())  # 9
    back = d.denormalize_predictions(tn)
    assert isinstance(back, DaughterTarget)
    np.testing.assert_allclose(np.asarray(back.vertex), np.asarray(t.vertex), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(back.p1), np.asarray(t.p1), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(back.p2), np.asarray(t.p2), rtol=1e-3, atol=1e-3)


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


def test_loss_is_array_and_matches_objective():
    d = _make_detector()
    rng = np.random.default_rng(3)
    # the UNIFIED daughter objective on the ALREADY-NORMALIZED 9-vec: vertex + perm-inv daughters + HNL-sum.
    target = jnp.asarray(rng.standard_normal((5, 9)).astype("float32"))
    pred = jnp.asarray(rng.standard_normal((5, 9)).astype("float32"))
    out = d.loss(pred, target)
    vp, vt = pred[:, :3], target[:, :3]
    p1p, p2p, p1t, p2t = pred[:, 3:6], pred[:, 6:9], target[:, 3:6], target[:, 6:9]
    lv = 0.5 * jnp.mean(jnp.square(vp - vt), -1)
    ld = 0.25 * jnp.minimum(jnp.mean(jnp.square(p1p - p1t) + jnp.square(p2p - p2t), -1),
                            jnp.mean(jnp.square(p2p - p1t) + jnp.square(p1p - p2t), -1))
    lh = 0.25 * jnp.mean(jnp.square(0.5 * ((p1p + p2p) - (p1t + p2t))), -1)
    np.testing.assert_allclose(np.asarray(out), np.asarray(lv + ld + lh), rtol=1e-5)
    assert out.shape == (5,)


def test_metric_is_per_sample_dict():
    d = _make_detector()
    rng = np.random.default_rng(4)
    target = jnp.asarray(rng.standard_normal((5, 9)).astype("float32"))
    pred = jnp.asarray(rng.standard_normal((5, 9)).astype("float32"))
    m = d.metric(pred, target)
    assert set(m) == set(d.metric_labels())  # keys == declared labels
    for v in m.values():
        assert v.shape == (5,)  # per-sample
    np.testing.assert_allclose(np.asarray(m["loss"]), np.asarray(d.loss(pred, target)), rtol=1e-5)


def test_no_split_or_mutable_state():
    """No train/val split, no loader, no 'current yaml design' state -- the design is always passed in,
    and the train/val split lives in the scripts (the detector is a deterministic function)."""
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
    ``simulate_debug`` packs the ``StrawEvent`` + the derived ``mask`` (``tdc >= 0``); ``_run_solver``
    returns the raw ``hits_idx``/``tdc`` buffers the engine fills.
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

    # the operational solver path: build a Pool from the synthetic event + a (1,2) boundary span;
    # the engine fills hits_idx (1, M, 4) uint32 + tdc (1, M) f32 (M = max_hits_per_event).
    from detopt.detector.straw import Pool

    pool = Pool(dd["masses"], dd["charges"], dd["positions"], dd["momenta"], dd["times"])
    boundaries = np.array([[0, len(dd["masses"])]], dtype=np.int32)
    layers, angles, Bs = d._design_to_geometry(design)
    hits_idx, tdc, traj = d._run_solver(pool, boundaries, layers, angles, Bs, np.array([1], np.uint32))
    assert hits_idx.shape == (1, 64, 4) and tdc.shape == (1, 64) and traj is None


def test_call_is_deterministic_by_event_index():
    """The new contract: ``detector(design, event_index)`` is a DETERMINISTIC function -- the same index
    reproduces the event, a repeated index reproduces it within the batch, and ``size()`` reports the
    event count (``None`` for the infinite analytic source)."""
    from analytic import analytic_detector, DESIGN

    d = analytic_detector()
    assert d.size() is None  # analytic = infinite source
    flat = lambda t: np.concatenate([np.asarray(x) for x in t], axis=-1)  # Target record -> flat array
    _, e1, _, t1 = d(DESIGN, np.array([3, 3, 7]))
    _, e2, _, t2 = d(DESIGN, np.array([3, 3, 7]))
    assert np.array_equal(np.asarray(e1.tdc), np.asarray(e2.tdc))            # reproducible across calls
    assert np.array_equal(np.asarray(e1.tdc)[0], np.asarray(e1.tdc)[1])      # repeated index -> identical
    assert not np.allclose(flat(t1)[0], flat(t1)[2])                          # different index -> different
    assert np.allclose(flat(t1), flat(t2))                                    # whole batch reproducible
