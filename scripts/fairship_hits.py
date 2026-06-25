"""Evaluate our trackers on FairShip's OWN digitized daughter hits, with FairShip's real spectrometer
field map (cubic-upsampled, JAX-looked-up) and geometry calibrated from the MC hit cloud. Produces the
same per-method ``{pred9, doca, nhits, n_stations, chi2}`` store as the sim path, so scripts/tracking.py
can render it as the third report regime. The four targets map to FairShip's per-hit data (all parallel,
track-labeled): V1 tubes <- digi_straw wire, V2 TDC <- digi_tdc, V3 r_drift <- MC (x,y) (smeared),
V4 hits <- MC (x,y) exact."""
import glob

import numpy as np
import jax
import jax.numpy as jnp
import uproot
from scipy.interpolate import RegularGridInterpolator

import detopt.tracking as tracking
from detopt.tracking.base import Z_START_MARGIN_CM

PER_STATION, N_GLOBAL, M = 8, 32, 256  # FairShip layout: 4 stations x (4 views x 2 layers); max hits/event
FIELD_PATH = "/home/max/dev/FairShip/files/MainSpectrometerField.root"
FIELD_OFFSET = 8957.0  # field-map local z=0 -> global z (the magnet centre; matches our z0, calibrated)
METHODS = ["retina-tubes", "retina-tdc", "retina-drift", "retina-hits",
           "nll-tubes", "nll-tdc", "nll-drift", "nll-hits"]


def load_field(crop=350.0, crop_z=650.0, fine=2.5):
    """FairShip Bx map (local frame, Tesla) cropped to the tracker region and CUBIC-upsampled onto a fine
    grid (scipy, host); the JAX runtime lookup is trilinear on the fine grid = cubic-smooth. The same
    K_BORIS applies (Tesla units). Returns ``(BX_fine, X0, DX, Y0, DY, ZL0, DZ)``."""
    d = uproot.open(FIELD_PATH)["Data"].arrays(library="np")
    BX = d["Bx"].reshape(95, 82, 199).astype(np.float64)  # x outer, y mid, z inner
    xg, yg, zg = np.arange(-470, 471, 10.0), np.arange(-405, 406, 10.0), np.arange(-990, 991, 10.0)
    xi, yi, zi = np.abs(xg) <= crop, np.abs(yg) <= crop, np.abs(zg) <= crop_z
    rgi = RegularGridInterpolator((xg[xi], yg[yi], zg[zi]), BX[np.ix_(xi, yi, zi)], method="cubic")
    xf = np.arange(xg[xi][0], xg[xi][-1] + 1e-6, fine)
    yf = np.arange(yg[yi][0], yg[yi][-1] + 1e-6, fine)
    zf = np.arange(zg[zi][0], zg[zi][-1] + 1e-6, fine)
    GX, GY, GZ = np.meshgrid(xf, yf, zf, indexing="ij")
    return jnp.asarray(rgi((GX, GY, GZ)).astype(np.float32)), float(xf[0]), fine, float(yf[0]), fine, float(zf[0]), fine


def make_bx(field):
    """``field`` -> ``bx_at(x, y, z_global) -> Bx`` (Tesla), trilinear on the cubic-upsampled grid."""
    BX, X0, DX, Y0, DY, ZL0, DZ = field

    def bx_at(x, y, z):
        ix, iy, iz = (x - X0) / DX, (y - Y0) / DY, (z - FIELD_OFFSET - ZL0) / DZ
        return jax.scipy.ndimage.map_coordinates(BX, [ix, iy, iz], order=1, mode="nearest")
    return bx_at


def _load(n_files):
    # Shared loader (detopt.data.fairship_loader) + the MC hit cloud this script also needs.
    from detopt.data.fairship_loader import load_fairship_digi
    return load_fairship_digi(int(n_files), columns=(
        "truth", "reco", "digi_straw", "digi_invalid", "digi_event_index", "hits", "hit_track", "digi_tdc"))


def _calibrate(ds, hits, valid):
    """Survey geometry from the MC hit cloud: per global layer z + stereo tan, per (view,layer) the wire
    line ``c = a*straw + b``."""
    st, vw, ly, sw = ds[valid, 0] - 1, ds[valid, 1], ds[valid, 2], ds[valid, 3]
    x, y, z = hits[valid, 0], hits[valid, 1], hits[valid, 2]
    lz, lt, wab = np.zeros(N_GLOBAL), np.zeros(N_GLOBAL), {}
    for v in range(4):
        for l in range(2):
            m = (vw == v) & (ly == l)
            a, tan, b = np.linalg.lstsq(np.stack([sw[m], x[m], np.ones(m.sum())], 1), y[m], rcond=None)[0]
            wab[(v, l)] = (a, b)
            for s in range(4):
                g = s * PER_STATION + v * 2 + l
                lz[g] = z[(st == s) & (vw == v) & (ly == l)].mean(); lt[g] = tan
    return lz, lt, wab


def _arrays(data, lz, lt, wab, fit_rows, smear_seed=0):
    """Per event in ``fit_rows``, the padded daughter-hit blocks (global layer + the 4 measurement leaves):
    ``c_wire`` (V1/V2/V3 wire), ``c_hit`` (V4 exact crossing), ``r_drift`` (smeared, V3), ``tdc`` (relative
    to the event's earliest daughter hit, V2), and ``valid``. Each ``(len(fit_rows), M)``."""
    ds, hits, ht, tdc, deidx, inv = (data["digi_straw"], data["hits"], data["hit_track"], data["digi_tdc"],
                                     data["digi_event_index"], data["digi_invalid"])
    daud = np.isin(ht, (3, 4)) & ~inv
    st, vw, ly, sw = ds[:, 0] - 1, ds[:, 1], ds[:, 2], ds[:, 3]
    g_all = st * PER_STATION + vw * 2 + ly
    a_arr = np.array([[wab[(v, l)][0] for l in range(2)] for v in range(4)])
    b_arr = np.array([[wab[(v, l)][1] for l in range(2)] for v in range(4)])
    c_wire = a_arr[vw, ly] * sw + b_arr[vw, ly]
    tan_all = lt[g_all]; cos_all = 1.0 / np.sqrt(1.0 + tan_all ** 2)
    c_hit = hits[:, 1] - hits[:, 0] * tan_all
    rng = np.random.default_rng(smear_seed)
    r_drift = np.abs(np.abs(c_hit - c_wire) * cos_all + rng.normal(0, 0.012, c_hit.shape))  # FairShip-realistic V3
    order = np.argsort(deidx, kind="stable")
    de_s = deidx[order]
    g_s, cw_s, ch_s, r_s, td_s, da_s = (g_all[order], c_wire[order], c_hit[order], r_drift[order], tdc[order], daud[order])
    blocks = {k: [] for k in ("G", "CW", "CH", "R", "TD", "V")}
    for e in fit_rows:
        lo, hi = np.searchsorted(de_s, e, "left"), np.searchsorted(de_s, e, "right")
        sel = da_s[lo:hi]
        g, cw, ch, r, td = g_s[lo:hi][sel], cw_s[lo:hi][sel], ch_s[lo:hi][sel], r_s[lo:hi][sel], td_s[lo:hi][sel]
        td = td - td.min() if len(td) else td  # FairShip tdc is absolute -> relative to the event's earliest hit
        n = min(len(g), M)
        pad = lambda v, fill, dt: np.concatenate([np.asarray(v[:n], dt), np.full(M - n, fill, dt)])
        blocks["G"].append(pad(g, 0, np.int32)); blocks["CW"].append(pad(cw, 0.0, np.float32))
        blocks["CH"].append(pad(ch, 0.0, np.float32)); blocks["R"].append(pad(r, 0.0, np.float32))
        blocks["TD"].append(pad(td, 0.0, np.float32)); blocks["V"].append(pad(np.ones(n), 0.0, np.float32))
    return {k: np.stack(v) for k, v in blocks.items()}


def _tracker(name, det, design, lz, lt, bx_at):
    """A tracker for one config, patched onto FairShip's calibrated geometry + real field map."""
    import yaml
    tr = tracking.from_config(yaml.safe_load(open(f"config/tracking/{name}.yaml")), det, design)
    tr.layer_z, tr.layer_tan = np.asarray(lz, np.float64), np.asarray(lt, np.float64)
    tr.z_start, tr.n_stations, tr.field = float(lz.min() - Z_START_MARGIN_CM), N_GLOBAL // tr.per_station, bx_at
    tr.residual = tr._make_residual(); tr._fit = tr._build_fit(0.0); tr._assign = jax.jit(jax.vmap(tr._nearest_track))
    return tr


# Per version, which (hit_Y, hit_r) leaves to feed: tubes=wire/0, tdc=wire/tdc, drift=wire/drift, hits=cross/0.
_LEAVES = {"tubes": ("CW", None), "tdc": ("CW", "TD"), "drift": ("CW", "R"), "hits": ("CH", None)}


def _fit_method(name, det, design, lz, lt, bx_at, blk, chunk):
    tr = _tracker(name, det, design, lz, lt, bx_at)
    yk, rk = _LEAVES[name.split("-")[1]]
    HL, HY = blk["G"], blk[yk]
    HR = blk[rk] if rk is not None else np.zeros_like(blk["CW"])
    Vv = blk["V"]
    B = HL.shape[0]
    parts = []
    for i in range(0, B, chunk):
        sl = slice(i, i + chunk)
        hl, hY, hr, vv = jnp.asarray(HL[sl]), jnp.asarray(HY[sl]), jnp.asarray(HR[sl]), jnp.asarray(Vv[sl])
        bank = jnp.asarray(tr._seed_bank(HY[sl], Vv[sl]))
        params = tr._fit(bank, hl, hY, hr, vv)
        pred9, doca = tr._predict(params)
        assign, nearest = tr._assign(params, hl, hY, hr, vv)
        nh, ns, ch2 = tr._quality(np.asarray(assign), np.asarray(nearest), HL[sl], Vv[sl])
        parts.append((pred9, doca, nh, ns, ch2))
    cat = lambda c: np.concatenate([p[c] for p in parts], 0)
    return dict(pred9=cat(0), doca=cat(1), nhits=cat(2), n_stations=cat(3), chi2=cat(4))


def store(det, design, n_files, n_events, chunk=1024):
    """Fit all 8 trackers on the first ``n_events`` events' FairShip daughter hits. Returns ``(store, true9,
    reco9, reco_ok)`` for those rows -- same shape as the sim path so the report renders it identically."""
    data, nf = _load(n_files)
    valid = ~data["digi_invalid"]
    lz, lt, wab = _calibrate(data["digi_straw"], data["hits"], valid)
    truth, reco = data["truth"], data["reco"]
    true9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], 1)
    reco9 = np.concatenate([reco[:, 4:7], reco[:, 7:10], reco[:, 13:16]], 1)
    reco_ok = ~np.isnan(reco).any(1)
    fit_rows = np.arange(min(int(n_events), truth.shape[0]))
    blk = _arrays(data, lz, lt, wab, fit_rows)
    bx_at = make_bx(load_field())
    n_hit = int((blk["V"].sum(1) > 0).sum())
    print(f"  FairShip hits: {nf} files, fit {fit_rows.shape[0]} events ({n_hit} with daughter hits, "
          f"{reco_ok[fit_rows].sum()} FairShip-reco'd)")
    out = {}
    for name in METHODS:
        out[name] = _fit_method(name, det, design, lz, lt, bx_at, blk, chunk)
        print(f"  [fs] {name:13s} fit done")
    return out, true9[fit_rows], reco9[fit_rows], reco_ok[fit_rows]
