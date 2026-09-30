#!/usr/bin/env python3
"""Our network vs FairSHiP's own reconstruction, on the SAME events, per stereo angle.

For each angle: take the events FairSHiP reconstructed, run the meta network on FairSHiP's digis for
exactly those events, and compare both against MC truth on the HNL quantities.

The straw index is flipped (FairSHiP numbers straws top-down, our combine bottom-up). The layer z
spacing mismatch (our 5 cm view gap vs FairSHiP's 12 cm) is NOT corrected here, so the network
numbers still carry that systematic.
"""

import os
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

ANG = {"a000": 0.000, "a315": 3.295, "a457": 4.570,
       "a630": 6.253, "a826": 8.400, "a1022": 10.220}
MAD_SE = 1.1664
NPZ = "/home/max/dev/detopt/data/fairship-npz"
CK = sys.argv[1]
CFG = sys.argv[2]

cfg = yaml.safe_load(open(CFG))
d = dict(cfg["detector"])
k0 = next(iter(d))
d[k0] = dict(d[k0])
d[k0]["data_dir"] = "/home/max/dev/detopt/data/mc/numpy_newFS"
det = detopt.detector.from_config(d)

reg = detopt.nn.from_config(det, config=cfg["regressor"],
                            rngs=regressor_rngs(int(cfg.get("seed", 0))), design=True)
rd, params, state = nnx.split(reg, nnx.Param, nnx.Variable)
mgr = detopt.utils.io.get_checkpointer(CK)
rest, _s, _d, _a = detopt.utils.io.restore_training_checkpoint(mgr, regressor=(params, state))
mgr.close()
nnx.replace_by_pure_dict(params, rest)
model = nnx.merge(rd, params, state)
mean, std = (np.asarray(a) for a in det._target_norm_arrays())


@jax.jit
def fwd(f, m):
    return model(f, m, deterministic=True)


def mad(e):
    return 1.4826 * float(np.median(np.abs(e - np.median(e))))


def hnl(v, p1, p2, t9):
    hp, ht = p1 + p2, t9[:, 3:6] + t9[:, 6:9]
    dm = np.linalg.norm(hp, axis=1) - np.linalg.norm(ht, axis=1)
    dpp = dm / np.linalg.norm(ht, axis=1)
    dpt = np.hypot(hp[:, 0], hp[:, 1]) - np.hypot(ht[:, 0], ht[:, 1])
    dirn = hp / np.linalg.norm(hp, axis=1, keepdims=True)
    ipv = np.linalg.norm(np.cross(-v, dirn), axis=1)
    return dm, dpp, dpt, ipv


print(f"{'angle':>7} {'n':>6} | {'|p| core':>17} | {'dp/p core':>17} | {'pT core':>17} | {'IP core':>15}")
for lab, a in ANG.items():
    data, _ = load_fairship_digi(None, data_glob=os.path.join(NPZ, lab, "*.npz"), columns=(
        "truth", "reco", "digi_straw", "digi_invalid", "digi_event_index", "digi_tdc", "hits"))
    ev, mask, rows, truth = pack_fairship_events(
        data, n_stations=det.n_stations, n_views_per_station=det.n_views_per_station,
        n_layers_per_view=det.n_layers_per_view, n_straws=det.n_straws,
        max_hits=det.max_hits_per_event)
    rc = np.asarray(data["reco"])[rows]
    ok = ~np.isnan(rc).any(1)
    if ok.sum() < 20:
        print(f"{a:7.3f} {int(ok.sum()):6d} |  (FairSHiP reconstructed too few events)")
        continue

    t9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], 1)[ok]
    r9 = np.concatenate([rc[:, 4:7], rc[:, 7:10], rc[:, 13:16]], 1)[ok]

    # Per (view, layer) wire line y = a*straw + b, surveyed from the MC hit cloud, then inverted
    # through our own straw_y(index) so the hit lands at the right y. a and b differ between Y and
    # stereo views and both depend on the angle, so a single constant offset cannot work.
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
    vw_a = np.asarray(ev.view)
    ly_a = np.asarray(ev.layer)
    y = np.zeros_like(sw)
    for (vw, ly), (aa, bb) in ab.items():
        sel = (vw_a == vw) & (ly_a == ly)
        y[sel] = aa * sw[sel] + bb
    stagger = np.where((ly_a & 1) == 1, 0.5 * det.layer_y_offset, -0.5 * det.layer_y_offset)
    idx = (y - stagger + det.layer_height) / det.straw_pitch - 0.5
    straw = np.clip(np.rint(idx), 0, det.n_straws - 1).astype(np.int32)
    ev = ev._replace(straw=straw)
    ev = jax.tree.map(lambda x: np.asarray(x)[ok], ev)
    m_ok = mask[ok]
    theta = np.asarray(det.to_scaled(np.array([np.radians(a)], np.float32)), np.float32)
    P = []
    n = m_ok.shape[0]
    for i in range(0, n, 512):
        sl = slice(i, min(i + 512, n))
        e = jax.tree.map(lambda x: jnp.asarray(x[sl]), ev)
        mm = jnp.asarray(m_ok[sl])
        P.append(np.asarray(fwd(det.combine_scaled(e, theta, mask=mm, reveal_design=True), mm)))
    pred = np.concatenate(P, 0)
    if pred.ndim == 3:
        pred = pred.mean(0)
    phys = pred * std + mean

    for who, arr in (("FairSHiP", r9), ("network ", phys)):
        dm, dpp, dpt, ipv = hnl(arr[:, :3], arr[:, 3:6], arr[:, 6:9], t9)
        f = lambda e: f"{mad(e):8.4f}+-{MAD_SE * mad(e) / np.sqrt(n):7.4f}"
        tag = f"{a:7.3f} {n:6d}" if who == "FairSHiP" else " " * 14
        print(f"{tag} | {f(dm)} | {f(dpp)} | {f(dpt)} | {mad(ipv):7.3f}+-{MAD_SE * mad(ipv) / np.sqrt(n):5.3f}  {who}")
