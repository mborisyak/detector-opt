#!/usr/bin/env python3
"""FairSHiP reco vs our NLL tracker vs the MLOE network, on the SAME paired events, per angle.

Alignment: the tracker store is indexed by truth row (the first n_events rows); the network path is
indexed by pack_fairship_events' `rows` (the events that have valid digis). Both are mapped back to
truth rows and intersected.

  three_way.py <trackers.npz> <checkpoint> <config.yaml> [method]
"""
import sys

import numpy as np
import yaml
import jax
import jax.numpy as jnp
from flax import nnx

import detopt
import detopt.detector
import detopt.nn
import detopt.utils.io
from detopt.data.fairship_loader import load_fairship_digi, pack_fairship_events
from detopt.nn.trainer.common import regressor_rngs

ANG = {"a000": 0.000, "a315": 3.295, "a457": 4.570, "a630": 6.253, "a826": 8.400, "a1022": 10.220}
MEAS_CUT, MIN_STATIONS, CHI2_CUT, DOCA_CUT = 25, 3, 4.0, 2.0
MAD_SE = 1.1664
NPZ = "/home/max/dev/detopt/data/fairship-npz"

z = np.load(sys.argv[1], allow_pickle=True)
CK, CFG = sys.argv[2], sys.argv[3]
method = sys.argv[4] if len(sys.argv) > 4 else "nll-tdc"

cfg = yaml.safe_load(open(CFG))
dd = dict(cfg["detector"])
k0 = next(iter(dd))
dd[k0] = dict(dd[k0])
dd[k0]["data_dir"] = "/home/max/dev/detopt/data/mc/numpy_newFS"
det = detopt.detector.from_config(dd)

reg = detopt.nn.from_config(det, config=cfg["regressor"],
                            rngs=regressor_rngs(int(cfg.get("seed", 0))), design=True)
rdef, params, state = nnx.split(reg, nnx.Param, nnx.Variable)
mgr = detopt.utils.io.get_checkpointer(CK)
rest, _s, _d, _a = detopt.utils.io.restore_training_checkpoint(mgr, regressor=(params, state))
mgr.close()
nnx.replace_by_pure_dict(params, rest)
model = nnx.merge(rdef, params, state)
mean, std = (np.asarray(a) for a in det._target_norm_arrays())


@jax.jit
def fwd(f, m):
    return model(f, m, deterministic=True)


mad = lambda e: 1.4826 * float(np.median(np.abs(e - np.median(e))))


def hnl(a, t9):
    hp, ht = a[:, 3:6] + a[:, 6:9], t9[:, 3:6] + t9[:, 6:9]
    dm = np.linalg.norm(hp, axis=1) - np.linalg.norm(ht, axis=1)
    dirn = hp / np.linalg.norm(hp, axis=1, keepdims=True)
    return (dm, dm / np.linalg.norm(ht, axis=1),
            np.hypot(hp[:, 0], hp[:, 1]) - np.hypot(ht[:, 0], ht[:, 1]),
            np.linalg.norm(np.cross(-a[:, :3], dirn), axis=1))


print(f"FairSHiP vs {method} vs MLOE network, paired events (all three reconstructed)")
print(f"{'angle':>7} {'n':>5} {'method':>10} | {'|p| core':>16} | {'dp/p core':>16} | "
      f"{'pT core':>16} | {'IP core':>14}")

for lab, a in ANG.items():
    if f"{lab}__true9" not in z.files:
        continue
    data, _ = load_fairship_digi(None, data_glob=f"{NPZ}/{lab}/*.npz", columns=(
        "truth", "reco", "digi_straw", "digi_invalid", "digi_event_index", "digi_tdc", "hits"))
    rc = np.asarray(data["reco"])
    reco9_all = np.concatenate([rc[:, 4:7], rc[:, 7:10], rc[:, 13:16]], 1)
    fs_ok_all = ~np.isnan(rc).any(1)

    g = lambda k: z[f"{lab}__{method}__{k}"]
    pred_t, true9_t = g("pred9"), z[f"{lab}__true9"]
    n_rows = pred_t.shape[0]
    trk_ok = ((g("nhits") >= MEAS_CUT).all(1) & (g("n_stations") >= MIN_STATIONS).all(1)
              & (g("chi2") < CHI2_CUT).all(1) & (g("doca") <= DOCA_CUT) & ~np.isnan(pred_t).any(1))

    ev, mask, rows, truth_p = pack_fairship_events(
        data, n_stations=det.n_stations, n_views_per_station=det.n_views_per_station,
        n_layers_per_view=det.n_layers_per_view, n_straws=det.n_straws,
        max_hits=det.max_hits_per_event)

    # truth rows kept by all three
    keep = np.zeros(rc.shape[0], bool)
    keep[:n_rows] = trk_ok
    keep &= fs_ok_all
    sel_net = np.isin(rows, np.nonzero(keep)[0])
    rows_sel = rows[sel_net]
    if rows_sel.size < 20:
        print(f"{a:7.3f} {rows_sel.size:5d} |  (too few paired events)")
        continue

    # network on exactly those events, per-view straw map from the MC hit cloud
    ds, hits, inv = data["digi_straw"], data["hits"], data["digi_invalid"]
    v = ~inv
    ab = {}
    for vw in range(4):
        for ly in range(2):
            m = v & (ds[:, 1] == vw) & (ds[:, 2] == ly)
            if m.sum() < 50:
                continue
            A = np.stack([ds[m, 3], hits[m, 0], np.ones(m.sum())], 1)
            c = np.linalg.lstsq(A, hits[m, 1], rcond=None)[0]
            ab[(vw, ly)] = (float(c[0]), float(c[2]))
    sw = np.asarray(ev.straw).astype(np.float64)
    vw_a, ly_a = np.asarray(ev.view), np.asarray(ev.layer)
    y = np.zeros_like(sw)
    for (vw, ly), (aa, bb) in ab.items():
        s = (vw_a == vw) & (ly_a == ly)
        y[s] = aa * sw[s] + bb
    stagger = np.where((ly_a & 1) == 1, 0.5 * det.layer_y_offset, -0.5 * det.layer_y_offset)
    straw = np.clip(np.rint((y - stagger + det.layer_height) / det.straw_pitch - 0.5),
                    0, det.n_straws - 1).astype(np.int32)
    ev = ev._replace(straw=straw)
    ev = jax.tree.map(lambda x: np.asarray(x)[sel_net], ev)
    m_sel = mask[sel_net]

    theta = np.asarray(det.to_scaled(np.array([np.radians(a)], np.float32)), np.float32)
    P = []
    for i in range(0, m_sel.shape[0], 512):
        sl = slice(i, min(i + 512, m_sel.shape[0]))
        e = jax.tree.map(lambda x: jnp.asarray(x[sl]), ev)
        mm = jnp.asarray(m_sel[sl])
        P.append(np.asarray(fwd(det.combine_scaled(e, theta, mask=mm, reveal_design=True), mm)))
    net = np.concatenate(P, 0)
    if net.ndim == 3:
        net = net.mean(0)
    net = net * std + mean

    t9 = np.concatenate([truth_p[:, 4:7], truth_p[:, 8:11], truth_p[:, 12:15]], 1)[sel_net]
    n = rows_sel.size
    series = (("FairSHiP", reco9_all[rows_sel]),
              (method, pred_t[rows_sel]),
              ("MLOE", net))
    for i, (who, arr) in enumerate(series):
        dm, dpp, dpt, ipv = hnl(arr, t9)
        f = lambda e: f"{mad(e):8.4f}+-{MAD_SE * mad(e) / np.sqrt(n):6.4f}"
        tag = f"{a:7.3f} {n:5d}" if i == 0 else " " * 13
        print(f"{tag} {who:>10} | {f(dm)} | {f(dpp)} | {f(dpt)} | "
              f"{mad(ipv):6.3f}+-{MAD_SE * mad(ipv) / np.sqrt(n):5.3f}")
