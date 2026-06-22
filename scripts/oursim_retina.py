"""Retina on OUR simulation (magnet-consistent), FairShip-reconstructed events, three measurements.

For every event FairShip reconstructed (non-NaN ``reco``), this pushes its two HNL daughters -- the
boundary crossings from ``particles`` -- through OUR straw solver at the V2023 geometry
(``initial_stereo``) and OUR ~1 T*m field, requesting each daughter's exact ``(x, y)`` crossing of all
32 layer planes (the solver's ``tracks`` output). From those *magnet-consistent* crossings it builds
three per-hit measurements and fits the retina (same engine, ``track_fit``) with each:

  V1 position-only   : nearest wire centre (2cm pitch)         -- sigma = pitch/sqrt(12)
  V2 distance-to-wire: wire centre + |crossing-wire| drift radius -- sigma = drift_sigma
  V3 exact (x, y)    : the exact crossing (no quantization)     -- sigma = drift_sigma (ceiling)

and reports per-component RMSE next to FairShip's own ``reco``, all on the identical event set. Unlike
``fairship_retina.py`` (which fits FairShip's digitized straws with our field -- a magnet mismatch),
here the field that GENERATES the hits also FITS them.

    python scripts/oursim_retina.py n_files=20 n_iters=1000
"""

import glob

import numpy as np
import jax.numpy as jnp

import detopt
from detopt.detector import straw_detector
import track_fit as T
import fairship_retina as FR  # _errs, _master_table, _pred9


def _load_reco(n_files):
    """Concatenate data/mc files that carry ``reco`` -> truth, reco, particles, global event_index."""
    fs = [f for f in sorted(glob.glob("data/mc/*.npz")) if "reco" in np.load(f, allow_pickle=True).files][: int(n_files)]
    truth, reco, pa, EI, bz, eoff = [], [], [], [], None, 0
    for f in fs:
        d = np.load(f, allow_pickle=True)
        truth.append(np.asarray(d["truth"], np.float32))
        reco.append(np.asarray(d["reco"], np.float32))
        pa.append(np.asarray(d["particles"], np.float32))
        EI.append(np.asarray(d["event_index"], np.int64) + eoff)
        eoff += d["truth"].shape[0]
        if bz is None:
            bz = float(d["boundary_z"])
    return np.concatenate(truth), np.concatenate(reco), np.concatenate(pa), np.concatenate(EI), bz, len(fs)


def _input_events(pa, ei, bz, ev_rows):
    """Build solver ``InputEvents`` from the two daughters (first 2 particles) of each event, plus the
    ``(n,2)`` [start,end) boundaries into that flat particle list. Mass/momentum -> MeV (solver units)."""
    order = np.argsort(ei, kind="stable")
    ei_s = ei[order]
    mass, charge, pos, mom, t0, bnds, base = [], [], [], [], [], [], 0
    for e in ev_rows:
        lo = np.searchsorted(ei_s, e, "left")
        rows = order[lo : lo + 2]  # daughters are the event's first two particles
        for r in rows:
            mass.append(pa[r, 1] * 1000.0)
            charge.append(pa[r, 0])
            pos.append([pa[r, 5], pa[r, 6], bz])  # crossing (x, y) at the boundary plane z
            mom.append([pa[r, 2] * 1000.0, pa[r, 3] * 1000.0, pa[r, 4] * 1000.0])
            t0.append(pa[r, 7])
        bnds.append([base, base + len(rows)])
        base += len(rows)
    ie = straw_detector.InputEvents(
        np.asarray(mass, np.float32), np.asarray(charge, np.float32),
        np.asarray(pos, np.float32), np.asarray(mom, np.float32), np.asarray(t0, np.float32),
    )
    return ie, np.asarray(bnds, np.int32)


def _crossings(det, ie, bnds, design, layer_z, batch=4096):
    """Run the solver in batches, returning each daughter's exact crossings ``tracks (n,2,32,2)`` and
    ``track_mask (n,2,32)`` at the layer planes (boundaries index the full ``ie``, so batching is safe).

    The C solver now emits an ordered per-track trajectory ``traj (nb,2,m,3)`` (= (x,y,z) of each
    daughter's first m in-aperture crossings) + ``n_cross (nb,2)``; here we scatter those back into the
    per-plane ``tracks``/``tmask`` layout ``_build_hits`` expects, matching each crossing to its plane by
    its stored z (== a ``layer_z`` value, exact)."""
    n, m = bnds.shape[0], layer_z.shape[0]
    lz = np.asarray(layer_z, np.float32)
    tracks = np.zeros((n, 2, m, 2), np.float32)
    tmask = np.zeros((n, 2, m), np.int32)
    rng = np.random.default_rng(0)
    for i in range(0, n, batch):
        b = bnds[i : i + batch]
        nb = b.shape[0]
        traj = np.zeros((nb, 2, m, 3), np.float32)
        ncr = np.zeros((nb, 2), np.int32)
        pid = np.full((nb, 2), -1, np.int32)
        det._run_solver(b, np.repeat(design[None], nb, 0), rng, input_events=ie, z_planes=lz, traj=traj, n_cross=ncr, part_idx=pid, primaries=True)
        valid = np.arange(m)[None, None, :] < ncr[:, :, None]  # (nb,2,m) filled crossing slots
        p = np.clip(np.searchsorted(lz, traj[..., 2]), 0, m - 1)  # plane index per crossing slot
        ei = np.broadcast_to((i + np.arange(nb))[:, None, None], (nb, 2, m))[valid]
        es = np.broadcast_to(np.arange(2)[None, :, None], (nb, 2, m))[valid]
        pv = p[valid]
        tracks[ei, es, pv, 0] = traj[..., 0][valid]
        tracks[ei, es, pv, 1] = traj[..., 1][valid]
        tmask[ei, es, pv] = 1
    return tracks, tmask


def _build_hits(tracks, tmask, layer_z, layer_tan, n_straws, pitch, z_start, z0):
    """From the crossings build the three measurements + multistart seeds. Per event the hits are the
    (<=2 daughters)x(32 planes) crossings, flattened to M=64 slots. Returns
    ``(seeds, hit_layer, hit_Y_wire, hit_Y_exact, hit_r, hit_valid)``."""
    n, _, nz = tmask.shape
    gg = np.broadcast_to(np.arange(nz), (n, 2, nz))  # global layer per slot
    tan, cos = layer_tan[gg], 1.0 / np.sqrt(1.0 + layer_tan[gg] ** 2)
    stagger = np.where((gg & 1) == 1, 0.5, -0.5) * (pitch / 2.0)  # half-pitch layer stagger (= layer_y_offset/2)
    x, y = tracks[..., 0], tracks[..., 1]
    c = y - x * tan  # exact sheared crossing (perpendicular-to-wire coordinate)
    straw = np.clip(np.round((c - stagger + pitch * n_straws / 2.0) / pitch - 0.5), 0, n_straws - 1)
    wire = (straw + 0.5) * pitch - pitch * n_straws / 2.0 + stagger  # nearest wire centre
    r = np.abs(c - wire) * cos  # true drift radius (perpendicular distance crossing->wire)
    valid = (tmask > 0).astype(np.float32)
    flat = lambda a: a.reshape(n, 2 * nz)
    HL, HYw, HYe, HR, HV = flat(gg).astype(np.int32), flat(wire).astype(np.float32), flat(c).astype(np.float32), flat(r).astype(np.float32), flat(valid)

    qop = 1.0 / (3.0 * 1000.0)
    up = (tmask > 0) & (np.abs(tan) < 1e-6) & (layer_z[gg] < z0)  # upstream axial crossings -> straight seed lines
    seeds = []
    for e in range(n):
        zc, cc = layer_z[gg[e][up[e]]], wire[e][up[e]]
        if zc.shape[0] >= 2:
            lines = T._retina_lines(zc - z_start, cc, s=4.0)
        else:
            b = float(np.median(wire[e][tmask[e] > 0])) if (tmask[e] > 0).any() else 0.0
            lines = [(0.05, b), (-0.05, b)]
        bank = []
        for (k0, b0), (k1, b1) in ((lines[0], lines[1]), (lines[1], lines[0])):
            for s0 in (+1.0, -1.0):
                for s1 in (+1.0, -1.0):
                    bank.append([[0.0, b0, 0.0, k0, s0 * qop], [0.0, b1, 0.0, k1, s1 * qop]])
        seeds.append((np.asarray(bank, np.float64) / T.PARAM_SCALE).astype(np.float32))
    return np.stack(seeds), HL, HYw, HYe, HR, HV


def _no_material(det):
    """Turn OFF every material interaction in the C solver: multiple scattering, Bethe-Bloch energy
    loss, delta-ray production, photon conversion, decay-in-flight, noise. The signal daughters then
    propagate as clean helices (only the field bends them). Rebuilds the SimParams the solver reads."""
    det.scatter_xX0 = 0.0
    det.enable_eloss = 0
    det.enable_decay = 0
    det.eloss_wall_coef = 0.0
    det.eloss_gas_const = 0.0
    det.delta_const = 0.0  # no delta-ray production
    det.delta_Tcut = 1.0e9
    det.lambda_conv_cm = 1.0e12  # no photon conversion
    det.noise_rate = 0.0
    det._sim_params = straw_detector.SimParams(
        det.max_dt, det.max_time, det.dt_fixed, det.max_steps, 64,
        det.scatter_xX0, det.lambda_conv_cm, det.noise_rate, det.enable_decay,
        det.delta_const, det.wall_thickness, det.delta_Tcut, det.enable_eloss,
        det.eloss_wall_coef, det.eloss_gas_const, det.eloss_I, det.eloss_min_ke,
    )


def run(n_files=20, dt=0.4, n_steps=160, n_iters=1000, lr=0.05, coef=1.0, s_hi=30.0, drift_sigma=0.05, material=True, optimizer="adam", **config):
    det = detopt.detector.StereoTracking()
    if not material:
        _no_material(det)
        print("MATERIAL INTERACTIONS OFF: no scattering / energy-loss / delta-rays / conversion / decay")
    sigma_hit = det.straw_pitch / np.sqrt(12.0)
    design = np.asarray(det.flatten_design(__import__("yaml").safe_load(open("config/design/initial_stereo.yaml"))), np.float32)
    layers, angles, _Bs = det._design_to_geometry(design[None])
    layer_z, layer_tan = np.asarray(layers[0], np.float64), np.asarray(angles[0], np.float64)
    z_start = float(layer_z.min() - 5.0)

    truth, reco, pa, ei, bz, nf = _load_reco(n_files)
    true9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], axis=1)
    reco9 = np.concatenate([reco[:, 4:7], reco[:, 7:10], reco[:, 10:13]], axis=1)
    reco_ok = ~np.isnan(reco).any(1)
    ev_rows = np.nonzero(reco_ok)[0]
    print(f"loaded {nf} files: {truth.shape[0]} events, {reco_ok.sum()} reconstructed by FairShip; pushing daughters through OUR solver")

    ie, bnds = _input_events(pa, ei, bz, ev_rows)
    tracks, tmask = _crossings(det, ie, bnds, design, layer_z.astype(np.float32))
    print(f"  our sim: mean planes crossed/daughter = {tmask.sum(2).mean():.1f}/32")
    seeds, hl, hYw, hYe, hr, hv = _build_hits(tracks, tmask, layer_z, layer_tan, det.n_straws, det.straw_pitch, z_start, det.z0)
    seeds, hl, hYw, hYe, hr, hv = (jnp.asarray(a) for a in (seeds, hl, hYw, hYe, hr, hv))
    zero = jnp.zeros_like(hYw)
    prior = (det.daughter_momentum_mean, det.daughter_momentum_sigma, sigma_hit)
    common = (jnp.asarray(layer_z), jnp.asarray(layer_tan), z_start, (float(det.max_B), det.z0, det.B_sigma), prior, dt, n_steps, n_iters, lr, float(coef), float(s_hi))

    def pred_full(p9):
        a = np.full_like(true9, np.nan)
        a[ev_rows] = p9
        return a

    results = []
    print(f"optimizer = {optimizer}")
    f1, _ = T._make_retina_fit(*common, sigma_hit, optimizer)
    results.append(("our V1 pos-only", FR._errs(det, pred_full(FR._pred9(f1(seeds, hl, hYw, zero, hv), z_start)), true9, reco_ok)))
    f2, _ = T._make_retina_fit(*common, float(drift_sigma), optimizer)
    results.append(("our V2 dist-to-wire", FR._errs(det, pred_full(FR._pred9(f2(seeds, hl, hYw, hr, hv), z_start)), true9, reco_ok)))
    f3, e3 = T._make_retina_fit(*common, float(drift_sigma), optimizer)
    p3 = f3(seeds, hl, hYe, zero, hv)
    trms = np.asarray(e3(p3, hl, hYe, zero, hv)[0])  # per-track RMS of _boris_traj to the EXACT crossings
    print(f"V3 track-RMS of the fit to the exact crossings: median={np.median(trms):.4f} cm  p90={np.percentile(trms,90):.4f} cm")
    print("  (if this is sub-mm but |dp|/p is %-level, the fit reproduces the hits at the WRONG momentum")
    print("   -> the retina propagator (_boris_traj) disagrees with the C solver that made them.)")
    results.append(("our V3 exact (x,y)", FR._errs(det, pred_full(FR._pred9(p3, z_start)), true9, reco_ok)))
    results.append(("FairShip reco", FR._errs(det, reco9, true9, reco_ok)))
    FR._master_table(results, reco_ok)


if __name__ == "__main__":
    import gearup

    gearup.gearup(run).with_config("config/regression.yaml")()
