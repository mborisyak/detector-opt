"""Simple differentiable track fit: a fixed-dt, fixed-step Boris propagator in JAX.

A self-contained baseline for "what momentum resolution does plain tracking get out of the stereo
straw detector, from wire positions alone (no drift/TDC)". For each event we take the two HNL
daughters' fired straws (each as its wire position -- the sheared coordinate ``Y = y - x*tan(angle)``
at the straw centre, quantised at the 2 cm pitch, no TDC), forward-propagate two charged tracks with
a Boris pusher that IGNORES ALL MATERIAL (no energy loss, no scattering, no secondaries), predict
each track's crossing of every detector plane, and minimise the MSE between predicted and measured
wire coordinate over the assigned hits. We fit each track's full initial state with optax, seeded
from a field-off straight-line fit, then read off the recovered momenta.

Each track is parametrised the standard way at a fixed start plane ``z_start``:
``(x0, y0, tx, ty, q/p)`` -- transverse position, the two slopes ``dx/dz, dy/dz``, and the signed
inverse momentum (charge = its sign, so charge is fit continuously, no left/right branch). The
field bend is *linear* in ``q/p``, which makes the fit well conditioned. Each parameter is rescaled
to O(1) (``PARAM_SCALE``) so a single Adam learning rate moves them all.

The Boris physics, field model and units mirror ``detopt/detector/straw_detector.c`` exactly:
B is along x, Gaussian in z (``B*exp(-0.5*((z-z0)/B_sigma)^2)``), the rotation is in the y-z plane
(tracks bend in y). Positions are cm; the bend uses ``rot = K_BORIS*dt*Bx*(q/p)`` with ``q/p`` in
1/MeV (= ``q/E`` for the ultrarelativistic daughters, p >> m, so beta = 1 and |p| = 1/|q/p|).

Truth is used ONLY to (a) associate each fired straw with one of the two daughters and (b) anchor
the start z -- NEVER for the fitted positions or momenta. Unsupervised hit-to-track association is a
follow-up. This touches only the detector's public interface; it modifies no detector.

    python scripts/track_fit.py seed=0 n_events=2048
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import detopt  # noqa: E402
from detopt.utils.events import shuffled_event_index  # noqa: E402

# Track-fit math (Boris propagator, |p| readout, drift-circle residual, closest-approach vertex, retina
# seed lines) is canonical in the detector package now; import it (underscore aliases keep callers --
# oursim_retina / fairship_retina -- working). Constants live there too.
from detopt.detector.track_solver import (  # noqa: E402
    K_BORIS,
    C_CM_PER_NS,
    PARAM_SCALE,
    QOP_MIN,
    p_vec as _p_vec,
    boris_traj as _boris_traj,
    vertex_doca as _vertex_doca,
    make_residual as _make_residual,
)


def _make_fit(layer_z, layer_tan, z_start, field, prior, dt, n_steps, n_iters, lr, coef):
    """Build the vmapped per-event MAP fit. `field` = (B, z0, B_sigma); `prior` = (mom_mean(3),
    mom_sigma(3), sigma_hit) -- the daughter-momentum Gaussian prior and the wire-hit resolution.
    Loss = NLL + coef * NLP : Gaussian hit likelihood + (coef-weighted) momentum prior."""
    B, z0f, Bsig = field
    mom_mean, mom_sigma, sigma_hit = (jnp.asarray(prior[0]), jnp.asarray(prior[1]), float(prior[2]))
    scale = jnp.asarray(PARAM_SCALE)

    def sheared_pred(track_params):
        x0, y0, tx, ty, qop = track_params * scale  # internal O(1) -> physical
        zt, xt, yt = _boris_traj(x0, y0, z_start, tx, ty, qop, field, dt, n_steps)
        xp = jnp.interp(layer_z, zt, xt)  # (n_planes,)
        yp = jnp.interp(layer_z, zt, yt)
        return yp - xp * layer_tan  # predicted wire (sheared-Y) at every plane

    def event_loss(params, hit_layer, hit_Y, hit_track, hit_valid):
        Ypred = jnp.stack([sheared_pred(params[0]), sheared_pred(params[1])])  # (2, n_planes)
        r2 = (Ypred[hit_track, hit_layer] - hit_Y) ** 2 * hit_valid
        nll = 0.5 * jnp.sum(r2) / (sigma_hit * sigma_hit)  # Gaussian hit likelihood (sum over hits)
        p_vec = _p_vec(params * scale, jnp)  # (2, 3) daughter momenta (GeV)
        nlp = 0.5 * jnp.sum(((p_vec - mom_mean) / mom_sigma) ** 2)  # momentum prior, both daughters
        return nll + coef * nlp

    def fit_event(seed_params, hit_layer, hit_Y, hit_track, hit_valid):
        # Clip before Adam: near q/p->0 the momentum prior's gradient spikes (|p| -> cap), which would
        # otherwise poison Adam's second-moment state and freeze the curvature.
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

        def step(carry, _):
            p, st = carry
            loss, g = jax.value_and_grad(event_loss)(p, hit_layer, hit_Y, hit_track, hit_valid)
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), loss

        (p, _), losses = jax.lax.scan(step, (seed_params, opt.init(seed_params)), None, length=n_iters)
        return p, losses[-1]

    def track_rms(params, hit_layer, hit_Y, hit_track, hit_valid):
        """Per-track RMS wire residual (cm) at the fitted params -> (2,), for a convergence cut."""
        Ypred = jnp.stack([sheared_pred(params[0]), sheared_pred(params[1])])
        err2 = (Ypred[hit_track, hit_layer] - hit_Y) ** 2 * hit_valid
        sel = lambda t: (hit_track == t) * hit_valid
        return jnp.stack([jnp.sqrt(jnp.sum(err2 * (hit_track == t)) / jnp.maximum(jnp.sum(sel(t)), 1.0)) for t in (0, 1)])

    return jax.jit(jax.vmap(fit_event)), jax.jit(jax.vmap(track_rms))


# ---------------------------------------------------------------------------------------------- #
# Host-side preprocessing: fired straws -> wire coords, truth association, straight-line seed.
# ---------------------------------------------------------------------------------------------- #
def _resolve_design(config):
    """The design dict from the config. ``config['design']`` is either an already-expanded dict
    (gearup expands the top-level ``design`` ref through ``config/design/``) or a bare name/path we
    load from ``config/design/<name>.yaml`` (so ``design=stereo-lfi-427`` works on the CLI)."""
    import os
    import yaml

    d = config["design"]
    if isinstance(d, str):
        path = d if os.path.exists(d) else os.path.join("config", "design", f"{d}.yaml")
        d = yaml.safe_load(open(path))
    return d


def z_start_of(layer_z):
    return float(layer_z.min() - 5.0)


def _wire_y(detector, straw, layer_in_view):
    """Straw wire centre = sheared coordinate Y at x=0 (cm). Matches straw_detector.c."""
    stagger = np.where(layer_in_view & 1, 0.5, -0.5) * detector.layer_y_offset
    return (straw + 0.5) * detector.straw_pitch - detector.layer_height + stagger


def _build_event(detector, X_e, mask_e, tracks_e, tmask_e, layer_z, layer_tan, z_start, M, intBdl, min_hits=4, min_stations=1):
    """One event -> padded hit arrays (hit_layer, hit_Y, hit_track, hit_valid) and the (2,5) seed
    (INTERNAL O(1) params). Returns None unless BOTH daughters have >= min_hits straw hits across
    >= min_stations stations (the pre-fit half of the FairShip track-quality selection)."""
    per_station = detector.n_views_per_station * detector.n_layers_per_view
    valid = mask_e > 0
    # X_e is a per-event StrawEvent (int address fields + float tdc).
    station = np.asarray(X_e.station)[valid].astype(np.int64)
    view = np.asarray(X_e.view)[valid].astype(np.int64)
    liv = np.asarray(X_e.layer)[valid].astype(np.int64)
    straw = np.asarray(X_e.straw)[valid].astype(np.int64)
    layer = station * per_station + view * detector.n_layers_per_view + liv
    Ymeas = _wire_y(detector, straw, liv)
    tan_h = layer_tan[layer]

    # Truth association: assign each fired straw to the daughter whose GT crossing (sheared) is nearest.
    Yg = tracks_e[:, layer, 1] - tracks_e[:, layer, 0] * tan_h  # (2, n_hits) daughter sheared-Y at the layer
    crossed = tmask_e[:, layer] > 0  # (2, n_hits)
    cost = np.where(crossed, np.abs(Yg - Ymeas[None, :]), np.inf)
    track = np.argmin(cost, axis=0)
    keep = np.isfinite(cost.min(axis=0))  # drop hits no daughter reached (rare)
    layer, Ymeas, tan_h, track = layer[keep], Ymeas[keep], tan_h[keep], track[keep]

    seed = np.zeros((2, 5), np.float64)
    for t in range(2):
        sel = track == t
        if sel.sum() < min_hits or len(np.unique(layer[sel] // per_station)) < min_stations:
            return None
        zL, aL, YL = layer_z[layer[sel]], tan_h[sel], Ymeas[sel]
        dz = zL - z_start
        # Field-off straight line at z_start: Y = (y0 + ty*dz) - (x0 + tx*dz)*tan(a).
        A = np.stack([-aL, -dz * aL, np.ones_like(zL), dz], axis=1)
        (x0, tx, y0, ty), *_ = np.linalg.lstsq(A, YL, rcond=None)
        # q/p and sign from the y-kink across the magnet, using axial (tan~0) hits.
        ax = np.abs(aL) < 1e-6
        qop_gev = 1.0 / 50.0  # default: a moderately stiff 50 GeV stick if no kink resolvable
        slope = lambda z, y: float(((z - z.mean()) * (y - y.mean())).sum() / max(((z - z.mean()) ** 2).sum(), 1e-9))
        if ax.sum() >= 4:
            up, dn = ax & (zL < detector.z0), ax & (zL > detector.z0)
            if up.sum() >= 2 and dn.sum() >= 2:
                theta = slope(zL[dn], YL[dn]) - slope(zL[up], YL[up])  # signed kink -> signed q/p
                p_gev = np.clip(0.3 * intBdl / max(abs(theta), 1e-3), 0.5, 500.0)
                qop_gev = np.sign(theta or 1.0) / p_gev
        qop_mev = qop_gev / 1000.0
        seed[t] = np.array([x0, y0, tx, ty, qop_mev]) / PARAM_SCALE

    pad = lambda a, fill, dt: np.concatenate([a.astype(dt), np.full(M - len(a), fill, dt)])[:M]
    n = min(len(layer), M)
    return (
        seed.astype(np.float32),
        pad(layer[:n], 0, np.int32),
        pad(Ymeas[:n], 0.0, np.float32),
        pad(track[:n], 0, np.int32),
        pad(np.ones(n), 0.0, np.float32),
    )


def run_pipeline(
    detector, config, *, seed, n_events, dt, n_steps, n_iters, lr, coef, min_hits, min_stations, chi2_cut, doca_cut, out, label
):
    """Sample -> select (pre-fit hit/station cuts) -> MAP fit -> select (post-fit chi2/DoCA cuts) ->
    report network-identical RMSE + DoCA. The selection knobs reproduce FairShip's ShipAna track
    quality + candidate cuts (MEAS_CUT, CHI2_CUT, DOCA_CUT); weak defaults keep all reconstructable
    events. Shared by `fit` (weak) and `scripts/fairship_select.py` (FairShip clean)."""
    sigma_hit = detector.straw_pitch / np.sqrt(12.0)
    phys = np.asarray(detector.flatten_design(detector.decode_design(detector.encode_design(_resolve_design(config)))), np.float32)
    positions, angles, _ = detector._design_to_geometry(phys[None, :])
    layer_z = np.asarray(positions[0], np.float64)
    layer_tan = np.tan(np.asarray(angles[0], np.float64))
    z_start = z_start_of(layer_z)
    intBdl = abs(detector.max_B) * np.sqrt(2 * np.pi) * detector.B_sigma / 100.0  # T*m

    design = np.broadcast_to(phys[None, :], (int(n_events), phys.shape[0]))
    event_index = shuffled_event_index(detector.size(), int(n_events), int(seed))
    ev = detector._simulate(design, event_index, z_planes=layer_z.astype(np.float32), n_tracks=2)
    X, mask = ev["X"], ev["mask"]
    # The solver now emits an ordered per-track trajectory; scatter it back to the per-plane
    # tracks/tmask layout _build_event expects (each crossing matched to its plane by its z).
    traj, ncr = ev["traj"], ev["n_cross"]  # (n,2,P,3), (n,2)
    lz = layer_z.astype(np.float32)
    P, N = lz.shape[0], int(n_events)
    tracks = np.zeros((N, 2, P, 2), np.float32)
    tmask = np.zeros((N, 2, P), np.int32)
    vc = np.arange(P)[None, None, :] < ncr[:, :, None]
    pp = np.clip(np.searchsorted(lz, traj[..., 2]), 0, P - 1)
    ei = np.broadcast_to(np.arange(N)[:, None, None], (N, 2, P))[vc]
    es = np.broadcast_to(np.arange(2)[None, :, None], (N, 2, P))[vc]
    pv = pp[vc]
    tracks[ei, es, pv, 0] = traj[..., 0][vc]
    tracks[ei, es, pv, 1] = traj[..., 1][vc]
    tmask[ei, es, pv] = 1
    tr = ev["target"]  # DaughterTarget; true 9-vec [vertex(3), p1(3), p2(3)]
    targets = np.concatenate([np.asarray(tr.vertex), np.asarray(tr.p1), np.asarray(tr.p2)], axis=-1)
    M = detector.max_hits_per_event

    built, tgt = [], []
    for e in range(int(n_events)):
        X_e = type(X)(*(np.asarray(a)[e] for a in X))  # per-event StrawEvent
        b = _build_event(
            detector,
            X_e,
            mask[e],
            tracks[e],
            tmask[e],
            layer_z,
            layer_tan,
            z_start,
            M,
            intBdl,
            min_hits=min_hits,
            min_stations=min_stations,
        )
        if b is None:
            continue
        built.append(b)
        tgt.append(targets[e])  # true 9-vec [vertex(3) cm, p1(3) GeV, p2(3) GeV]
    print(f"events sampled={n_events}")
    print(
        f"pre-fit selection (both daughters >={min_hits} hits, >={min_stations} stations): "
        f"{len(built)}/{n_events} = {100*len(built)/n_events:.1f}%"
    )

    stack = lambda i: jnp.asarray(np.stack([b[i] for b in built]))
    seeds, hl, hY, ht, hv = stack(0), stack(1), stack(2), stack(3), stack(4)

    prior = (detector.daughter_momentum_mean, detector.daughter_momentum_sigma, sigma_hit)
    fit_fn, eval_rms = _make_fit(
        jnp.asarray(layer_z),
        jnp.asarray(layer_tan),
        z_start,
        (float(detector.max_B), detector.z0, detector.B_sigma),
        prior,
        dt,
        n_steps,
        n_iters,
        lr,
        float(coef),
    )
    params, _ = fit_fn(seeds, hl, hY, ht, hv)
    trms = np.asarray(eval_rms(params, hl, hY, ht, hv))  # (N, 2) per-track final wire RMS (cm)
    nh = np.stack([(np.asarray(hv) * (np.asarray(ht) == t)).sum(1) for t in (0, 1)], axis=1)  # (N, 2) hits/track
    return _report(detector, np.asarray(params), trms, nh, np.stack(tgt), z_start, sigma_hit, chi2_cut, doca_cut, out, label)


def _report(detector, params, trms, nh, true9, z_start, sigma_hit, chi2_cut, doca_cut, out, label, robust=False):
    """Fitted (N,2,5) internal params -> per-component error + DoCA, with the FairShip-style post-fit
    selection (per-track chi2/Ndf + pair DoCA). Shared by both fit paths. ``robust=True`` summarises
    each component by the median and 0.25-quantile of the ABSOLUTE error (tail-insensitive) instead of
    the (mean-based, outlier-sensitive) RMSE -- used by the retina fit, whose multistart failures /
    wrong-charge events give heavy tails that dominate an RMSE."""
    ph = np.asarray(params) * PARAM_SCALE  # (N, 2, 5): x0, y0 (cm), tx, ty, q/p (1/MeV)
    p_vec = _p_vec(ph, np)  # (N, 2, 3) daughter momenta (GeV); |p| smoothly capped (MAP prior bounds it)
    vertex, doca = _vertex_doca(ph, z_start)
    pred9 = np.concatenate([vertex, p_vec[:, 0], p_vec[:, 1]], axis=1)  # (N, 9)

    # Post-fit selection: per-track chi2/Ndf (Ndf = nhits - 5) and the pair DoCA (ShipAna CHI2_CUT/DOCA_CUT).
    chi2ndf = nh * (trms / sigma_hit) ** 2 / np.maximum(nh - 5, 1)  # (N, 2)
    ev_ok = (chi2ndf < chi2_cut).all(1) & (doca < doca_cut)
    print(
        f"post-fit selection (both tracks chi2/Ndf < {chi2_cut:g}, DoCA < {doca_cut:g} cm): "
        f"{ev_ok.sum()}/{len(ph)} = {100*ev_ok.mean():.1f}%   "
        f"median track RMS = {np.median(trms[ev_ok]):.3f} cm (floor {sigma_hit:.3f})"
    )

    # Network-IDENTICAL RMSE: detector.prediction_errors (permutation-matched, vertex cm / momenta GeV).
    pred_norm = np.asarray(detector.normalize_target(pred9[ev_ok]))
    true_norm = np.asarray(detector.normalize_target(true9[ev_ok]))
    errs = detector.prediction_errors(pred_norm, true_norm)
    fairship = {"p_x": 0.007, "p_y": 0.009, "p_z": 0.45}  # FairShip drift-fit RMSE (GeV), for reference
    # Per-component summary stat (|error|): robust median + 0.25-quantile, or the mean-based RMSE.
    summary = (lambda r: (float(np.median(np.abs(r))), float(np.quantile(np.abs(r), 0.25)))) if robust else None
    head = "per-component |error|  median / q0.25" if robust else "per-component RMSE (identical to network metric_real_rmse)"
    print(f"\n{head}  N={ev_ok.sum()} events:")
    for k, (resid, unit) in errs.items():
        resid = np.asarray(resid)
        ref = f"   (FairShip ~{fairship[k]:g} {unit})" if k in fairship else ""
        if robust:
            med, q25 = summary(resid)
            print(f"   {k:9s}  median={med:9.3f}  q0.25={q25:9.3f} {unit}{ref}")
        else:
            print(f"   {k:9s} = {np.sqrt(np.mean(resid ** 2)):9.3f} {unit}{ref}")
    dp = np.sqrt((np.asarray(errs["p_x"][0]) ** 2 + np.asarray(errs["p_y"][0]) ** 2 + np.asarray(errs["p_z"][0]) ** 2))
    p_match = np.concatenate([np.linalg.norm(true9[ev_ok, 3:6], axis=1), np.linalg.norm(true9[ev_ok, 6:9], axis=1)])
    dpop = dp / p_match
    tail = f"q0.25={100*np.quantile(dpop, 0.25):.2f}%" if robust else f"mean={100*np.mean(dpop):.2f}%"
    print(f"   |dp|/p   : median={100*np.median(dpop):.2f}%  {tail}   (FairShip drift-fit ~1%)")
    dca = doca[ev_ok]
    print(
        f"   DoCA [cm]: median={np.median(dca):.2f}  q0.25={np.quantile(dca, 0.25):.2f}  "
        f"p90={np.percentile(dca, 90):.2f}   (FairShip ~1 cm)"
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, k in zip(axes.ravel(), errs.keys()):
        resid, unit = np.asarray(errs[k][0]), errs[k][1]
        hi = np.percentile(np.abs(resid), 99)
        ax.hist(np.clip(resid, -hi, hi), bins=100, color="C0", alpha=0.85)
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(f"{k} residual  [{unit}]")
        if robust:
            med, q25 = summary(resid)
            ax.set_title(f"{k}:  median={med:.3g}  q0.25={q25:.3g} {unit}")
        else:
            ax.set_title(f"{k}:  RMSE = {np.sqrt(np.mean(resid ** 2)):.3g} {unit}")
    fig.suptitle(f"{label} -- network-identical residuals  N={ev_ok.sum()}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    return errs


# ============================================================================================== #
# Artificial-retina fit: truth-FREE pattern recognition + fit in one differentiable objective.
#
# The retina response of a track is  R(theta) = sum_hits exp(-(d_i/sigma)^2),  where d_i is the
# perpendicular distance from the track to fired straw i's WIRE (a 3D line). For our straws the wire
# sits at fixed z, tilted by the stereo angle, so d_i = |Ypred_sheared - Ymeas| * cos(angle) (the
# sheared residual scaled to the true perpendicular distance -- see derivation in the call site).
# No hit-to-track assignment and no MC truth: we fit BOTH daughters jointly and let each hit be
# claimed by whichever track is nearest (max over tracks), so maximizing R separates the two tracks
# onto the two daughters. sigma is annealed wide->narrow so a coarse seed still captures the hits.
# Multistart: seeds come from a coarse retina line-scan of the upstream y-view (top-2 lines) crossed
# with both charge signs; we keep the start with the largest final response.
# ============================================================================================== #
# Fixed multistart bank: 2 line-pairings x 2 charge signs x 2 charge signs = 8 starts.
N_RETINA_STARTS = 8


def _build_event_retina(detector, X_e, mask_e, layer_z, layer_tan, z_start, M, min_hits, min_stations):
    """One event -> (seed_bank (8,2,5) internal params, padded hit_layer, hit_Y, hit_valid). NO MC
    truth: hits are all fired straws; the 2-track seeds come from a retina line-scan of the upstream
    y-view. Returns None unless the event has >= min_hits straws across >= min_stations stations."""
    per_station = detector.n_views_per_station * detector.n_layers_per_view
    valid = mask_e > 0
    station = np.asarray(X_e.station)[valid].astype(np.int64)
    view = np.asarray(X_e.view)[valid].astype(np.int64)
    liv = np.asarray(X_e.layer)[valid].astype(np.int64)
    straw = np.asarray(X_e.straw)[valid].astype(np.int64)
    layer = station * per_station + view * detector.n_layers_per_view + liv
    Ymeas = _wire_y(detector, straw, liv)
    if len(layer) < min_hits or len(np.unique(layer // per_station)) < min_stations:
        return None

    zL = layer_z[layer]
    tan = layer_tan[layer]
    up_y = (np.abs(tan) < 1e-6) & (zL < detector.z0)  # upstream axial (y-view) hits: ~straight, field-off
    if up_y.sum() >= 2:
        lines = _retina_lines(zL[up_y] - z_start, Ymeas[up_y], s=4.0)
    else:  # too few y-view hits: split a single guess by +/- slope so the two seeds differ
        b0 = float(np.median(Ymeas)) if len(Ymeas) else 0.0
        lines = [(0.05, b0), (-0.05, b0)]

    qop_seed = 1.0 / (3.0 * 1000.0)  # |p| ~ 3 GeV seed (1/MeV); the fit refines magnitude + sign
    seeds = []
    for (k0, b0), (k1, b1) in ((lines[0], lines[1]), (lines[1], lines[0])):  # pairing + swap
        for s0 in (+1.0, -1.0):
            for s1 in (+1.0, -1.0):
                seeds.append([[0.0, b0, 0.0, k0, s0 * qop_seed], [0.0, b1, 0.0, k1, s1 * qop_seed]])
    seeds = (np.asarray(seeds, np.float64) / PARAM_SCALE).astype(np.float32)  # (8, 2, 5) internal O(1)

    pad = lambda a, fill, dt: np.concatenate([a.astype(dt), np.full(M - len(a), fill, dt)])[:M]
    n = min(len(layer), M)
    return seeds, pad(layer[:n], 0, np.int32), pad(Ymeas[:n], 0.0, np.float32), pad(np.ones(n), 0.0, np.float32)


_BFGS_STAGES, _BFGS_MAXITER = 8, 30  # optimizer="bfgs": BFGS-to-convergence at each of 8 annealed sigma stages


def _make_retina_fit(layer_z, layer_tan, z_start, field, prior, dt, n_steps, n_iters, lr, coef, s_hi, s_lo, optimizer="adam"):
    """Joint 2-track retina fit with multistart. ``prior`` = (mom_mean(3), mom_sigma(3), sigma_hit).
    Returns ``(fit_fn, eval_fn)``: ``fit_fn(seed_bank, layer, Y, valid) -> best (2,5) params`` (the
    start with the largest response); ``eval_fn`` gives per-track wire-RMS + hit counts under the
    nearest-track assignment (for the chi2 selection)."""
    mom_mean, mom_sigma = jnp.asarray(prior[0]), jnp.asarray(prior[1])
    scale = jnp.asarray(PARAM_SCALE)
    residual = _make_residual(layer_z, layer_tan, z_start, field, dt, n_steps)
    # Rank the multistart at a COARSE sharpness (>= the hit scale): at a tight s_lo every start's
    # response collapses to ~0 and argmax picks a garbage start. The coarse rank only has to tell a
    # right-topology fit from a wrong one; the sharp s_lo still drives the optimisation itself.
    s_rank = max(float(s_lo), float(prior[2]))

    def response(params, hit_layer, hit_Y, hit_r, hit_valid, s):
        e = jnp.exp(-((residual(params, hit_layer, hit_Y, hit_r) / s) ** 2))  # (2, n_hits) retina response
        return jnp.sum(jnp.max(e, axis=0) * hit_valid)  # each hit claimed by its nearest track

    def loss(params, hit_layer, hit_Y, hit_r, hit_valid, s):
        nlp = 0.5 * jnp.sum(((_p_vec(params * scale, jnp) - mom_mean) / mom_sigma) ** 2)  # gentle |p| prior
        return -response(params, hit_layer, hit_Y, hit_r, hit_valid, s) + coef * nlp

    def fit_one(seed, hit_layer, hit_Y, hit_r, hit_valid):
        if optimizer == "bfgs":
            # Staged BFGS: minimise to convergence at each annealed sharpness, warm-starting the next
            # stage. BFGS assumes a fixed objective, so the anneal is done in OUTER stages (not per step).
            from jax.scipy.optimize import minimize

            stages = s_hi * (s_lo / s_hi) ** (jnp.arange(_BFGS_STAGES) / max(_BFGS_STAGES - 1, 1))

            def bstage(x, s):
                res = minimize(
                    lambda p: loss(p.reshape(2, 5), hit_layer, hit_Y, hit_r, hit_valid, s),
                    x, method="BFGS", options={"maxiter": _BFGS_MAXITER},
                )
                return res.x, None

            x, _ = jax.lax.scan(bstage, seed.reshape(-1), stages)
            p = x.reshape(2, 5)
            return p, response(p, hit_layer, hit_Y, hit_r, hit_valid, s_rank)
        # Cosine-decay the LR alongside the sharpness anneal: once s is tight the response gradient
        # vanishes, and a constant-LR Adam random-walks off the good wide-s solution (tight runs ended
        # up WORSE than position-only). Decaying lr->0 freezes the fit at its converged point.
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(lr, n_iters)))

        def step(carry, i):
            p, st = carry
            s = s_hi * (s_lo / s_hi) ** (i / n_iters)  # anneal sharpness wide -> narrow
            _l, g = jax.value_and_grad(loss)(p, hit_layer, hit_Y, hit_r, hit_valid, s)
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), None

        (p, _), _ = jax.lax.scan(step, (seed, opt.init(seed)), jnp.arange(n_iters))
        return p, response(p, hit_layer, hit_Y, hit_r, hit_valid, s_rank)  # rank starts at the coarse scale

    def fit_event(seed_bank, hit_layer, hit_Y, hit_r, hit_valid):
        ps, Rs = jax.vmap(lambda sd: fit_one(sd, hit_layer, hit_Y, hit_r, hit_valid))(seed_bank)
        return ps[jnp.argmax(Rs)]  # keep the multistart winner

    def eval_event(params, hit_layer, hit_Y, hit_r, hit_valid):
        d = jnp.abs(residual(params, hit_layer, hit_Y, hit_r))  # (2, n_hits) distance to the drift circle
        assign = jnp.argmin(d, axis=0)  # nearest-track assignment (truth-free), for the chi2/RMS report
        d_assigned = jnp.min(d, axis=0)
        rms = jnp.stack(
            [
                jnp.sqrt(jnp.sum((d_assigned**2) * (assign == t) * hit_valid) / jnp.maximum(jnp.sum((assign == t) * hit_valid), 1.0))
                for t in (0, 1)
            ]
        )
        nh = jnp.stack([jnp.sum((assign == t) * hit_valid) for t in (0, 1)])
        return rms, nh

    return jax.jit(jax.vmap(fit_event)), jax.jit(jax.vmap(eval_event))


_NU_INIT = -2.197  # logit(0.1): initial per-event noise fraction pi_noise ~ 0.1 when it is learned


def _make_mixture_fit(
    layer_z, layer_tan, z_start, field, prior, dt, n_steps, n_iters, lr, s_hi, s_lo, noise_W, coef, pi_fixed=None, noise_kappa=0.0
):
    """Joint 2-track fit by MAXIMUM A POSTERIORI under a PROPER generative mixture (unlike the retina's
    robust alignment score, this IS a likelihood). Each hit is drawn from track 0, track 1 -- a Gaussian
    on the drift-circle residual, width = measurement sigma ``s_lo`` -- or a UNIFORM noise band of width
    ``noise_W`` (density 1/W):

        p(hit | theta) = (1 - pi)/2 * [N(r0; 0, s^2) + N(r1; 0, s^2)]  +  pi * (1/W)

    The per-event noise fraction ``pi`` is either HARD-FIXED to ``pi_fixed`` (then the only params are the
    2x5 track params), or LEARNED as ``sigmoid(nu)`` with a Beta(1, 1+noise_kappa) prior (log-density
    ``noise_kappa*log(1-pi)``) that favours low ``pi``. On top sits a Gaussian PRIOR ON MOMENTA, weight
    ``coef`` (``mom_mean``/``mom_sigma`` from ``prior``); here it is a genuine prior, not a retina
    regulariser. Deterministic annealing: the Gaussian width is annealed ``s_hi -> s_lo`` to smooth the
    non-convex hit-assignment landscape, ending at the true resolution so the final objective is the exact
    mixture log-posterior. Returns ``(fit_fn, eval_fn)`` matching :func:`_make_retina_fit` (fit_fn -> best
    ``(2,5)`` track params)."""
    residual = _make_residual(layer_z, layer_tan, z_start, field, dt, n_steps)
    mom_mean, mom_sigma = jnp.asarray(prior[0]), jnp.asarray(prior[1])
    scale = jnp.asarray(PARAM_SCALE)
    sigma, logU = float(s_lo), -jnp.log(float(noise_W))
    half_log_2pi = 0.5 * jnp.log(2.0 * jnp.pi)
    learn_pi = pi_fixed is None

    def mixture_ll(track, pi_n, hit_layer, hit_Y, hit_r, hit_valid, s):
        r = residual(track, hit_layer, hit_Y, hit_r)  # (2, n_hits) drift-circle residual, cm
        log_track = -0.5 * (r / s) ** 2 - jnp.log(s) - half_log_2pi + jnp.log(0.5 * (1.0 - pi_n))  # (2, n_hits)
        log_noise = jnp.broadcast_to(jnp.log(pi_n) + logU, (1, r.shape[1]))  # (1, n_hits) uniform band
        ll = logsumexp(jnp.concatenate([log_track, log_noise], 0), axis=0)  # (n_hits,) per-hit mixture
        return jnp.sum(ll * hit_valid)

    def mom_penalty(track):
        return 0.5 * jnp.sum(((_p_vec(track * scale, jnp) - mom_mean) / mom_sigma) ** 2)  # -log N(p; mean, sigma^2)

    def neg_objective(params, hit_layer, hit_Y, hit_r, hit_valid, s):
        track = params[0] if learn_pi else params
        pi_n = jax.nn.sigmoid(params[1]) if learn_pi else pi_fixed
        nll = -mixture_ll(track, pi_n, hit_layer, hit_Y, hit_r, hit_valid, s)
        log_pi_prior = noise_kappa * jnp.log1p(-pi_n) if learn_pi else 0.0  # Beta(1, 1+kappa): favours pi -> 0
        return nll + coef * mom_penalty(track) - log_pi_prior

    def fit_one(seed, hit_layer, hit_Y, hit_r, hit_valid):
        params0 = (seed, jnp.asarray(_NU_INIT)) if learn_pi else seed
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(lr, n_iters)))

        def step(carry, i):
            p, st = carry
            s = s_hi * (sigma / s_hi) ** (i / n_iters)  # anneal inlier width wide -> true resolution
            _l, g = jax.value_and_grad(neg_objective)(p, hit_layer, hit_Y, hit_r, hit_valid, s)
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), None

        (p, _), _ = jax.lax.scan(step, (params0, opt.init(params0)), jnp.arange(n_iters))
        track = p[0] if learn_pi else p
        return track, -neg_objective(p, hit_layer, hit_Y, hit_r, hit_valid, sigma)  # rank by final log-posterior

    def fit_event(seed_bank, hit_layer, hit_Y, hit_r, hit_valid):
        ts, scores = jax.vmap(lambda sd: fit_one(sd, hit_layer, hit_Y, hit_r, hit_valid))(seed_bank)
        return ts[jnp.argmax(scores)]  # (2,5) multistart winner

    def eval_event(params, hit_layer, hit_Y, hit_r, hit_valid):
        d = jnp.abs(residual(params, hit_layer, hit_Y, hit_r))  # (2, n_hits) distance to the drift circle
        assign = jnp.argmin(d, axis=0)  # nearest-track assignment (truth-free), for the chi2/RMS report
        d_assigned = jnp.min(d, axis=0)
        rms = jnp.stack(
            [
                jnp.sqrt(jnp.sum((d_assigned**2) * (assign == t) * hit_valid) / jnp.maximum(jnp.sum((assign == t) * hit_valid), 1.0))
                for t in (0, 1)
            ]
        )
        nh = jnp.stack([jnp.sum((assign == t) * hit_valid) for t in (0, 1)])
        return rms, nh

    return jax.jit(jax.vmap(fit_event)), jax.jit(jax.vmap(eval_event))


def run_retina_pipeline(
    detector, config, *, seed, n_events, dt, n_steps, n_iters, lr, coef, s_hi, s_lo, min_hits, min_stations, chi2_cut, doca_cut, out, label
):
    """Truth-free retina fit pipeline: sample -> per-event retina-seeded multistart fit -> report
    (network-identical RMSE + DoCA). Mirrors :func:`run_pipeline` but uses no MC truth at any step."""
    sigma_hit = detector.straw_pitch / np.sqrt(12.0)
    s_lo = float(s_lo) if s_lo is not None else sigma_hit
    phys = np.asarray(detector.flatten_design(detector.decode_design(detector.encode_design(_resolve_design(config)))), np.float32)
    positions, angles, _ = detector._design_to_geometry(phys[None, :])
    layer_z = np.asarray(positions[0], np.float64)
    layer_tan = np.tan(np.asarray(angles[0], np.float64))
    z_start = z_start_of(layer_z)

    design = np.broadcast_to(phys[None, :], (int(n_events), phys.shape[0]))
    event_index = shuffled_event_index(detector.size(), int(n_events), int(seed))
    ev = detector._simulate(design, event_index)  # no z_planes: the retina needs no truth crossings
    X, mask = ev["X"], ev["mask"]
    tr = ev["target"]  # DaughterTarget; true 9-vec [vertex(3), p1(3), p2(3)] -- used ONLY to score resolution
    targets = np.concatenate([np.asarray(tr.vertex), np.asarray(tr.p1), np.asarray(tr.p2)], axis=-1)
    M = detector.max_hits_per_event

    built, tgt = [], []
    for e in range(int(n_events)):
        X_e = type(X)(*(np.asarray(a)[e] for a in X))
        b = _build_event_retina(detector, X_e, mask[e], layer_z, layer_tan, z_start, M, min_hits, min_stations)
        if b is None:
            continue
        built.append(b)
        tgt.append(targets[e])
    print(f"events sampled={n_events}")
    print(f"pre-fit selection (>={min_hits} straws, >={min_stations} stations): {len(built)}/{n_events} = {100*len(built)/n_events:.1f}%")

    stack = lambda i: jnp.asarray(np.stack([b[i] for b in built]))
    seeds, hl, hY, hv = stack(0), stack(1), stack(2), stack(3)
    hr = jnp.zeros_like(hY)  # position-only: no drift radius (residual = track-to-wire distance)
    prior = (detector.daughter_momentum_mean, detector.daughter_momentum_sigma, sigma_hit)
    fit_fn, eval_fn = _make_retina_fit(
        jnp.asarray(layer_z), jnp.asarray(layer_tan), z_start,
        (float(detector.max_B), detector.z0, detector.B_sigma),
        prior, dt, n_steps, n_iters, lr, float(coef), float(s_hi), s_lo,
    )
    params = fit_fn(seeds, hl, hY, hr, hv)  # (N, 2, 5) internal params (multistart winners)
    trms, nh = eval_fn(params, hl, hY, hr, hv)
    return _report(
        detector, np.asarray(params), np.asarray(trms), np.asarray(nh), np.stack(tgt), z_start, sigma_hit,
        chi2_cut, doca_cut, out, label, robust=True,
    )


def fit(
    seed=0, n_events=2048, dt=0.4, n_steps=160, n_iters=1500, lr=0.05, coef=1.0,
    retina=False, s_hi=30.0, s_lo=None, out=None, **config,
):
    """Boris track fit. ``retina=False`` (default): truth-assigned MSE fit, weak selection (the
    baseline). ``retina=True``: truth-FREE artificial-retina fit (distance-to-wire response) +
    multistart -- no MC hit-to-track association at all."""
    detector = detopt.detector.from_config(config["detector"])
    if retina:
        return run_retina_pipeline(
            detector, config, seed=seed, n_events=n_events, dt=dt, n_steps=n_steps, n_iters=n_iters,
            lr=lr, coef=coef, s_hi=s_hi, s_lo=s_lo, min_hits=8, min_stations=3, chi2_cut=1e9, doca_cut=1e9,
            out=out or "track_fit_retina.png", label="Retina track fit + multistart (truth-free, distance-to-wire)",
        )
    return run_pipeline(
        detector, config, seed=seed, n_events=n_events, dt=dt, n_steps=n_steps, n_iters=n_iters, lr=lr, coef=coef,
        min_hits=4, min_stations=1, chi2_cut=9.0, doca_cut=1e9,
        out=out or "track_fit.png", label="Simple Boris track fit (weak selection)",
    )


if __name__ == "__main__":
    import gearup

    gearup.gearup(fit).with_config("config/regression.yaml")()
