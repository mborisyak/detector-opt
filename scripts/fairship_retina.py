"""Position-only artificial-retina fit on FairShip's DIGITIZED straws, vs FairShip's own reco.

Runs our distance-to-wire retina (no TDC, no MC truth in the measurement) on the REAL FairShip fired
straws (``digi_straw``, dropping ``digi_invalid``) for the events FairShip reconstructed (non-NaN
``reco`` row), and reports per-component RMSE for (a) our retina and (b) FairShip's ``reco``, both vs
``truth``, on the identical event set.

The only geometry needed to turn a fired-straw index into a wire ``(z, sheared-Y)`` -- per-plane z,
the stereo tan, the wire pitch/offset -- is CALIBRATED once from the MC hit cloud (a fixed-survey
quantity averaged over all hits, not a per-event measurement); the per-event input is purely the
straw address. FairShip numbers straws opposite to our nominal `_wire_y`, so this calibration (not
the project's parametrized geometry) is what makes the wire positions exact (residual ~ pitch/sqrt12).

    python scripts/fairship_retina.py n_files=20 n_iters=1000
"""

import glob

import numpy as np
import jax
import jax.numpy as jnp

import detopt
import track_fit as T  # the retina engine (objective, multistart, Boris propagator) lives here

PER_STATION = 8  # views(4) * layers(2): global layer = (station-1)*8 + view*2 + layer
N_GLOBAL = 32


def _load(n_files):
    """Concatenate the new-format data/mc files into flat arrays with global event indices. Only files
    carrying the full new schema (``reco`` + ``digi_*`` + ``hits``) are used; others are skipped."""
    need = ("truth", "reco", "digi_straw", "digi_invalid", "digi_event_index", "hits")
    fs = [f for f in sorted(glob.glob("data/mc/*.npz")) if need[1] in np.load(f, allow_pickle=True).files]
    fs = fs[: int(n_files)]
    truth, reco, ds, inv, deidx, hits = [], [], [], [], [], []
    off = 0
    for f in fs:
        d = np.load(f, allow_pickle=True)
        truth.append(np.asarray(d["truth"], np.float32))
        reco.append(np.asarray(d["reco"], np.float32))
        ds.append(np.asarray(d["digi_straw"], np.int64))
        inv.append(np.asarray(d["digi_invalid"], bool))
        deidx.append(np.asarray(d["digi_event_index"], np.int64) + off)
        hits.append(np.asarray(d["hits"], np.float32))
        off += d["truth"].shape[0]
    return (
        np.concatenate(truth), np.concatenate(reco), np.concatenate(ds),
        np.concatenate(inv), np.concatenate(deidx), np.concatenate(hits), len(fs),
    )


def _calibrate_geometry(ds, hits, valid):
    """Fixed survey geometry from the MC hit cloud (valid hits only): per global layer the plane z
    and stereo tan, and per (view, layer) the wire-coordinate line ``c(straw) = a*straw + b`` (FairShip's
    own straw numbering). Returns ``(layer_z(32,), layer_tan(32,), wire_ab {(view,layer):(a,b)})``."""
    st, vw, ly, sw = ds[valid, 0] - 1, ds[valid, 1], ds[valid, 2], ds[valid, 3]
    x, y, z = hits[valid, 0], hits[valid, 1], hits[valid, 2]
    layer_z = np.zeros(N_GLOBAL, np.float64)
    layer_tan = np.zeros(N_GLOBAL, np.float64)
    wire_ab = {}
    for v in range(4):
        for l in range(2):
            m = (vw == v) & (ly == l)
            a, tan, b = np.linalg.lstsq(np.stack([sw[m], x[m], np.ones(m.sum())], 1), y[m], rcond=None)[0]
            wire_ab[(v, l)] = (a, b)  # wire sheared coord c = y - x*tan = a*straw + b
            for s in range(4):
                g = s * PER_STATION + v * 2 + l
                layer_z[g] = z[(st == s) & (vw == v) & (ly == l)].mean()
                layer_tan[g] = tan
    return layer_z, layer_tan, wire_ab


def _build_events(ds, deidx, valid, ev_rows, hits, layer_z, layer_tan, wire_ab, z_start, z0, M):
    """Per reco'd event -> (seed_bank (8,2,5), hit_layer (M,), hit_Y (M,), hit_r (M,), hit_valid (M,)).
    ``hit_Y`` = wire sheared-Y (from the straw index); ``hit_r`` = the TRUE drift radius |d_perp| extracted
    from the cm MC hit (perpendicular distance from the crossing to the wire). Seeds come from a retina
    line-scan of the upstream y-view wires (no truth)."""
    st, vw, ly, sw = ds[:, 0] - 1, ds[:, 1], ds[:, 2], ds[:, 3]
    a_arr = np.array([[wire_ab[(v, l)][0] for l in range(2)] for v in range(4)])  # (4,2) pitch
    b_arr = np.array([[wire_ab[(v, l)][1] for l in range(2)] for v in range(4)])  # (4,2) offset
    gl_all = st * PER_STATION + vw * 2 + ly  # global layer per digi
    c_all = a_arr[vw, ly] * sw + b_arr[vw, ly]  # wire sheared-Y per digi
    tan_all = layer_tan[gl_all]
    cos_all = 1.0 / np.sqrt(1.0 + tan_all**2)
    c_hit = hits[:, 1] - hits[:, 0] * tan_all  # MC crossing's sheared coord (= y - x*tan)
    r_all = np.abs(c_hit - c_all) * cos_all  # true drift radius |d_perp| (cm), perpendicular to the wire
    # group valid digis by event via one sort (searchsorted slices, O(log D) per event)
    order = np.argsort(deidx, kind="stable")
    de_s, gl_s, c_s, r_s, val_s = deidx[order], gl_all[order], c_all[order], r_all[order], valid[order]
    qop = 1.0 / (3.0 * 1000.0)  # 3 GeV seed; fit refines magnitude + sign
    seeds, HL, HY, HR, HV = [], [], [], [], []
    for e in ev_rows:
        lo, hi = np.searchsorted(de_s, e, "left"), np.searchsorted(de_s, e, "right")
        sel = val_s[lo:hi]
        g, c, r = gl_s[lo:hi][sel], c_s[lo:hi][sel], r_s[lo:hi][sel]
        zL, tanL = layer_z[g], layer_tan[g]
        up_y = (np.abs(tanL) < 1e-6) & (zL < z0)  # upstream axial wires -> ~straight seed lines
        if up_y.sum() >= 2:
            lines = T._retina_lines(zL[up_y] - z_start, c[up_y], s=4.0)
        else:
            b = float(np.median(c)) if len(c) else 0.0
            lines = [(0.05, b), (-0.05, b)]
        bank = []
        for (k0, b0), (k1, b1) in ((lines[0], lines[1]), (lines[1], lines[0])):
            for s0 in (+1.0, -1.0):
                for s1 in (+1.0, -1.0):
                    bank.append([[0.0, b0, 0.0, k0, s0 * qop], [0.0, b1, 0.0, k1, s1 * qop]])
        seeds.append((np.asarray(bank, np.float64) / T.PARAM_SCALE).astype(np.float32))
        n = min(len(g), M)
        pad = lambda v, fill, dt: np.concatenate([np.asarray(v[:n], dt), np.full(M - n, fill, dt)])
        HL.append(pad(g, 0, np.int32)); HY.append(pad(c, 0.0, np.float32))
        HR.append(pad(r, 0.0, np.float32)); HV.append(pad(np.ones(n), 0.0, np.float32))
    return np.stack(seeds), np.stack(HL), np.stack(HY), np.stack(HR), np.stack(HV)


def _pred9(params, z_start):
    """Fitted (N,2,5) internal params -> [vertex(3), p1(3), p2(3)] (N,9)."""
    ph = np.asarray(params) * T.PARAM_SCALE
    vertex, _doca = T._vertex_doca(ph, z_start)
    p_vec = T._p_vec(ph, np)
    return np.concatenate([vertex, p_vec[:, 0], p_vec[:, 1]], axis=1)


def _errs(det, pred9, true9, on):
    """Per-component RMSE dict + |dp|/p (median, RMS) for one method, on the selected events."""
    e = det.prediction_errors(np.asarray(det.normalize_target(pred9[on])), np.asarray(det.normalize_target(true9[on])))
    out = {k: float(np.sqrt(np.mean(np.asarray(r) ** 2))) for k, (r, u) in e.items()}
    dp = np.sqrt(sum(np.asarray(e[k][0]) ** 2 for k in ("p_x", "p_y", "p_z")))
    pm = np.concatenate([np.linalg.norm(true9[on, 3:6], axis=1), np.linalg.norm(true9[on, 6:9], axis=1)])
    out["|dp|/p med%"] = float(100 * np.median(dp / pm))
    out["|dp|/p RMS%"] = float(100 * np.sqrt(np.mean((dp / pm) ** 2)))
    return out


def _master_table(results, on):
    """Print the master comparison: rows = quantities, columns = methods. ``results`` = [(name, errs)]."""
    rows = ["vertex_x", "vertex_y", "vertex_z", "p_x", "p_y", "p_z", "|dp|/p med%", "|dp|/p RMS%"]
    units = {"vertex_x": "cm", "vertex_y": "cm", "vertex_z": "cm", "p_x": "GeV", "p_y": "GeV", "p_z": "GeV"}
    names = [n for n, _ in results]
    w = max(len(n) for n in names) + 2
    print(f"\n=== MASTER COMPARISON (FairShip-reconstructed events, N={int(on.sum())}; RMSE unless noted) ===")
    print("  " + "quantity".ljust(14) + "".join(n.rjust(w) for n in names))
    for q in rows:
        cells = "".join(f"{e[q]:.3f}".rjust(w) for _, e in results)
        print("  " + f"{q} [{units.get(q,'%')}]".ljust(14) + cells)


def run(n_files=20, dt=0.4, n_steps=160, n_iters=1000, lr=0.05, coef=1.0, s_hi=30.0, drift_sigma=0.02, M=256, **config):
    det = detopt.detector.Stereo4Feature()  # for the target normalization + permutation-matched metric only
    sigma_hit = det.straw_pitch / np.sqrt(12.0)
    truth, reco, ds, inv, deidx, hits, nf = _load(n_files)
    valid = ~inv
    layer_z, layer_tan, wire_ab = _calibrate_geometry(ds, hits, valid)
    z_start = float(layer_z.min() - 5.0)

    true9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], axis=1)  # [vertex, p1, p2]
    reco9 = np.concatenate([reco[:, 4:7], reco[:, 7:10], reco[:, 10:13]], axis=1)
    reco_ok = ~np.isnan(reco).any(1)  # events FairShip reconstructed
    ev_rows = np.nonzero(reco_ok)[0]
    print(f"loaded {nf} files: {truth.shape[0]} events, {reco_ok.sum()} reconstructed by FairShip ({100*reco_ok.mean():.1f}%)")

    seeds, hl, hY, hr, hv = _build_events(ds, deidx, valid, ev_rows, hits, layer_z, layer_tan, wire_ab, z_start, det.z0, M)
    seeds, hl, hY, hr, hv = (jnp.asarray(seeds), jnp.asarray(hl), jnp.asarray(hY), jnp.asarray(hr), jnp.asarray(hv))
    zero_r = jnp.zeros_like(hY)
    prior = (det.daughter_momentum_mean, det.daughter_momentum_sigma, sigma_hit)
    field = (float(det.max_B), det.z0, det.B_sigma)
    common = (jnp.asarray(layer_z), jnp.asarray(layer_tan), z_start, field, prior, dt, n_steps, n_iters, lr, float(coef), float(s_hi))

    pred_full = lambda p9: (lambda a: (a.__setitem__(ev_rows, p9), a)[1])(np.full_like(true9, np.nan))
    results = []
    # (1) position-only retina: hit_r = 0, sharpness annealed to the wire-pitch resolution.
    pos_fit, _ = T._make_retina_fit(*common, sigma_hit)
    pos = pos_fit(seeds, hl, hY, zero_r, hv)
    results.append(("retina pos-only", _errs(det, pred_full(_pred9(pos, z_start)), true9, reco_ok)))
    # (2) distance-to-wire retina: hit_r = true drift radius, sharpness annealed to drift_sigma.
    drift_fit, _ = T._make_retina_fit(*common, float(drift_sigma))
    drift = drift_fit(seeds, hl, hY, hr, hv)
    results.append(("retina drift(truth)", _errs(det, pred_full(_pred9(drift, z_start)), true9, reco_ok)))
    # (3) FairShip's own reco.
    results.append(("FairShip reco", _errs(det, reco9, true9, reco_ok)))

    _master_table(results, reco_ok)


if __name__ == "__main__":
    import gearup

    gearup.gearup(run).with_config("config/regression.yaml")()
