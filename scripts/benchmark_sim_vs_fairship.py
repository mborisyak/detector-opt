"""Delta-Y + hit precision/recall of OUR simulation (material OFF) against FairShip's MC daughter hits.

For the events FairShip reconstructed, push the two HNL daughters through our solver (clean helices) and
read their crossings AT FAIRSHIP'S OWN PLANES (z + stereo angle calibrated from its MC hit cloud, so the
planes align -- our proposed design's z's do not). Compare to FairShip's MC hits of the daughters
(``hits`` positions with ``hit_track in {3,4}``); both are turned into a sheared-Y straw index and matched
per plane within +-1 STRAW (one pitch).

    recall    = MC daughter hits with one of our hits at the same plane within +-1 straw
    precision = our hits with an MC daughter hit at the same plane within +-1 straw
    delta-Y   = (our crossing - matched MC hit) sheared-Y, cm

    python scripts/benchmark_sim_vs_fairship.py n_files=10
"""
import glob
from collections import defaultdict

import numpy as np
import yaml

import detopt
import oursim_retina as OS
import fairship_retina as FR

DAUGHTERS = (3, 4)  # FairShip MCTrack ids of the two HNL daughters (~32 clean hits each)
STRAW_TOL = 1  # match within +-1 straw (one pitch)


def _straw(c, stagger, pitch, nstr):
    """Sheared-Y wire coordinate -> nearest straw index on OUR grid (clipped)."""
    return np.clip(np.round((c - stagger + pitch * nstr / 2.0) / pitch - 0.5), 0, nstr - 1).astype(int)


def run(n_files=10, **config):
    fs = [f for f in sorted(glob.glob("data/mc/*.npz")) if "hit_track" in np.load(f, allow_pickle=True).files][: int(n_files)]
    reco, pa, EI, hits, ht, hei, ds, inv = [], [], [], [], [], [], [], []
    eoff, bz = 0, None
    for f in fs:
        d = np.load(f, allow_pickle=True)
        reco.append(np.asarray(d["reco"], np.float32))
        pa.append(np.asarray(d["particles"], np.float32))
        EI.append(np.asarray(d["event_index"], np.int64) + eoff)
        hits.append(np.asarray(d["hits"], np.float32))
        ht.append(np.asarray(d["hit_track"], np.int64))
        hei.append(np.asarray(d["hit_event_index"], np.int64) + eoff)
        ds.append(np.asarray(d["digi_straw"], np.int64))
        inv.append(np.asarray(d["digi_invalid"], bool))
        eoff += d["reco"].shape[0]
        bz = float(d["boundary_z"]) if bz is None else bz
    reco, pa, EI = np.concatenate(reco), np.concatenate(pa), np.concatenate(EI)
    hits, ht, hei = np.concatenate(hits), np.concatenate(ht), np.concatenate(hei)
    ds, inv = np.concatenate(ds), np.concatenate(inv)
    ev_rows = np.nonzero(~np.isnan(reco).any(1))[0]

    det = detopt.detector.StereoTracking()
    OS._no_material(det)
    pitch, nstr = det.straw_pitch, det.n_straws
    design = np.asarray(det.flatten_design(yaml.safe_load(open("config/design/initial_stereo.yaml"))), np.float32)
    # FairShip's OWN plane z + stereo angle, calibrated from its MC hit cloud -> our crossings land on the
    # SAME planes as the MC hits (our proposed design's plane z's differ by ~8 cm, which would mis-bin).
    oz, otan, _ = FR._calibrate_geometry(ds, hits, ~inv)
    oz, otan = np.asarray(oz, np.float64), np.asarray(otan, np.float64)
    nz = oz.shape[0]
    ie, bnds = OS._input_events(pa, EI, bz, ev_rows)
    tracks, tmask = OS._crossings(det, ie, bnds, design, oz.astype(np.float32))  # our clean daughter crossings at FairShip planes

    # OUR crossings -> (sheared-Y c, straw) per (event, plane)
    n = bnds.shape[0]
    gg = np.broadcast_to(np.arange(nz), (n, 2, nz))
    stag = np.where((gg & 1) == 1, 0.5, -0.5) * (pitch / 2.0)
    c_our = tracks[..., 1] - tracks[..., 0] * otan[gg]
    s_our = _straw(c_our, stag, pitch, nstr)
    our = defaultdict(list)
    for i, e in enumerate(ev_rows):
        for dd in range(2):
            for g in range(nz):
                if tmask[i, dd, g] > 0:
                    our[(int(e), g)].append((int(s_our[i, dd, g]), float(c_our[i, dd, g])))

    # MC daughter hits -> nearest plane by z -> (sheared-Y c, straw) on OUR grid
    keep = np.isin(ht, DAUGHTERS) & np.isin(hei, ev_rows)
    hx, hy, hz, he = hits[keep, 0], hits[keep, 1], hits[keep, 2], hei[keep]
    gm = np.abs(hz[:, None] - oz[None, :]).argmin(1)  # nearest global plane
    c_mc = hy - hx * otan[gm]
    s_mc = _straw(c_mc, np.where((gm & 1) == 1, 0.5, -0.5) * (pitch / 2.0), pitch, nstr)
    mc = defaultdict(list)
    for e, g, s, c in zip(he, gm, s_mc, c_mc):
        mc[(int(e), int(g))].append((int(s), float(c)))

    # match per (event, plane) within +-STRAW_TOL straws; collect precision/recall + signed delta-Y
    our_total = our_rec = mc_total = mc_rec = 0
    dy = []
    for key in set(our) | set(mc):
        os_, ms = our.get(key, []), mc.get(key, [])
        for s, c in os_:
            our_total += 1
            near = [(abs(s - sm), cc) for sm, cc in ms if abs(s - sm) <= STRAW_TOL]
            if near:
                our_rec += 1
                dy.append(c - min(near)[1])  # our - nearest matched MC, sheared-Y
        for sm, _cc in ms:
            mc_total += 1
            if any(abs(s - sm) <= STRAW_TOL for s, _c in os_):
                mc_rec += 1
    dy = np.asarray(dy)
    precision, recall = our_rec / max(our_total, 1), mc_rec / max(mc_total, 1)
    print(f"\n=== our sim (material OFF) vs FairShip MC daughter hits, {len(ev_rows)} reco'd events ===")
    print(f"match window: same plane, +-{STRAW_TOL} straw ({STRAW_TOL * pitch:.0f} cm)")
    print(f"our crossings: {our_total}   MC daughter hits: {mc_total}")
    print(f"precision = {precision:.4f}   recall = {recall:.4f}")
    print(f"delta-Y (our - matched MC, sheared-Y cm): median={np.median(dy):+.3f}  mean={dy.mean():+.3f}  "
          f"std={dy.std():.3f}  |median|={np.median(np.abs(dy)):.3f}")


if __name__ == "__main__":
    import gearup

    gearup.gearup(run).with_config("config/regression.yaml")()
