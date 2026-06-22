"""Faithfulness check: does OUR solver (material OFF) fire the same wires as FairShip?

On the events FairShip reconstructed, push the two daughters through our solver with all material
interactions OFF (clean helices), and compare the wires our daughters fire to the wires FairShip
actually digitized -- in PHYSICAL sheared-Y position per (station,view,layer), since FairShip numbers
straws opposite to us. Reports, per our fired wire, the distance to the nearest FairShip wire on the
same plane+event (a match within pitch/2 == same physical straw), the recall (our wires FairShip
confirms), hit multiplicity, and the per-station breakdown (upstream should match best; FairShip's
material scattering + secondaries explain downstream divergence and its extra hits).

    python scripts/check_oursim_vs_fairship.py n_files=10
"""

import glob
from collections import defaultdict

import numpy as np
import yaml

import detopt
import fairship_retina as FR
import oursim_retina as OS

PER_STATION = 8


def run(n_files=10, max_B=None, **config):
    fs = [f for f in sorted(glob.glob("data/mc/*.npz")) if "reco" in np.load(f, allow_pickle=True).files][: int(n_files)]
    truth, reco, pa, EI, ds, inv, dei, hits, hei, bz, eoff = [], [], [], [], [], [], [], [], [], None, 0
    for f in fs:
        d = np.load(f, allow_pickle=True)
        truth.append(np.asarray(d["truth"], np.float32)); reco.append(np.asarray(d["reco"], np.float32))
        pa.append(np.asarray(d["particles"], np.float32)); EI.append(np.asarray(d["event_index"], np.int64) + eoff)
        ds.append(np.asarray(d["digi_straw"], np.int64)); inv.append(np.asarray(d["digi_invalid"], bool))
        dei.append(np.asarray(d["digi_event_index"], np.int64) + eoff); hits.append(np.asarray(d["hits"], np.float32))
        hei.append(np.asarray(d["hit_event_index"], np.int64) + eoff)
        eoff += d["truth"].shape[0]
        if bz is None:
            bz = float(d["boundary_z"])
    reco = np.concatenate(reco); pa = np.concatenate(pa); EI = np.concatenate(EI)
    ds = np.concatenate(ds); inv = np.concatenate(inv); dei = np.concatenate(dei); hits = np.concatenate(hits); hei = np.concatenate(hei)
    valid = ~inv
    ev_rows = np.nonzero(~np.isnan(reco).any(1))[0]

    fz, ftan, fab = FR._calibrate_geometry(ds, hits, valid)  # FairShip geometry from its own hit cloud
    det = detopt.detector.StereoTracking()
    OS._no_material(det)
    if max_B is not None:
        det.max_B = float(max_B)  # override field to test the FairShip-matched bend
    print(f"field: max_B={det.max_B:.4f} T  (int B dz = {det.max_B*np.sqrt(2*np.pi)*det.B_sigma/100:.3f} T*m)")
    pitch, nstr = det.straw_pitch, det.n_straws
    design = np.asarray(det.flatten_design(yaml.safe_load(open("config/design/initial_stereo.yaml"))), np.float32)
    layers, angles, _ = det._design_to_geometry(design[None])
    oz, otan = np.asarray(layers[0], np.float64), np.asarray(angles[0], np.float64)
    ie, bnds = OS._input_events(pa, EI, bz, ev_rows)
    tracks, tmask = OS._crossings(det, ie, bnds, design, oz.astype(np.float32))

    n, _, nz = tmask.shape
    gg = np.broadcast_to(np.arange(nz), (n, 2, nz))
    stag = np.where((gg & 1) == 1, 0.5, -0.5) * (pitch / 2.0)
    c_our = tracks[..., 1] - tracks[..., 0] * otan[gg]  # our crossing's sheared coord
    straw = np.clip(np.round((c_our - stag + pitch * nstr / 2.0) / pitch - 0.5), 0, nstr - 1)
    w_our = (straw + 0.5) * pitch - pitch * nstr / 2.0 + stag  # our fired wire position

    # FairShip fired wires (valid), grouped by (event, global layer) in physical sheared-Y
    st, vw, ly, sw = ds[:, 0] - 1, ds[:, 1], ds[:, 2], ds[:, 3]
    fgl = st * PER_STATION + vw * 2 + ly
    a = np.array([[fab[(v, l)][0] for l in range(2)] for v in range(4)])
    b = np.array([[fab[(v, l)][1] for l in range(2)] for v in range(4)])
    w_fair = a[vw, ly] * sw + b[vw, ly]
    sel = valid & np.isin(dei, ev_rows)
    fdict = defaultdict(list)
    for e, g, w in zip(dei[sel], fgl[sel], w_fair[sel]):
        fdict[(int(e), int(g))].append(w)
    fair_count = defaultdict(int)
    for e in dei[sel]:
        fair_count[int(e)] += 1

    dists, matched, total = [], 0, 0
    pst_m, pst_t = np.zeros(4), np.zeros(4)
    our_mult = np.zeros(n)
    for i, e in enumerate(ev_rows):
        for dd in range(2):
            for g in range(nz):
                if tmask[i, dd, g] <= 0:
                    continue
                total += 1; s = g // PER_STATION; pst_t[s] += 1; our_mult[i] += 1
                cand = fdict.get((int(e), int(g)))
                if cand:
                    dmin = min(abs(w_our[i, dd, g] - c) for c in cand)
                    dists.append(dmin)
                    if dmin < pitch / 2.0:
                        matched += 1; pst_m[s] += 1
    dists = np.asarray(dists)
    fair_mult = np.array([fair_count[int(e)] for e in ev_rows], float)
    print(f"\n=== our sim (material OFF) vs FairShip digi, {n} reco'd events ===")
    print(f"hit multiplicity / event:  ours(2 daughters) median={np.median(our_mult):.0f}   FairShip median={np.median(fair_mult):.0f}")
    print(f"our fired wires: {total}; with a FairShip wire on the same plane: {len(dists)} ({100*len(dists)/total:.1f}%)")
    print(f"SAME wire (|dpos| < pitch/2 = {pitch/2:.1f}cm): {matched}/{total} = {100*matched/total:.1f}%")
    print(f"wire-position offset |our - nearest FairShip|: median={np.median(dists):.3f}  p90={np.percentile(dists,90):.3f}  mean={dists.mean():.3f} cm")
    print("per-station same-wire recall:  " + "  ".join(f"st{s+1}={100*pst_m[s]/max(pst_t[s],1):.1f}%" for s in range(4)))

    # Clean trajectory check: our crossing (x,y) vs the nearest FairShip MC hit (x,y,z) in the same
    # event -- no digitization, no wire grid. Isolates geometry+field fidelity. Per station.
    hsel = np.isin(hei, ev_rows)
    hbye = defaultdict(list)
    for e, p in zip(hei[hsel], hits[hsel]):
        hbye[int(e)].append(p)
    hbye = {e: np.asarray(v, np.float32) for e, v in hbye.items()}
    st_d = [[], [], [], []]
    for i, e in enumerate(ev_rows):
        He = hbye.get(int(e))
        if He is None:
            continue
        for dd in range(2):
            for g in range(nz):
                if tmask[i, dd, g] <= 0:
                    continue
                near = He[np.abs(He[:, 2] - oz[g]) < 15.0]  # FairShip hits on this plane
                if near.shape[0] == 0:
                    continue
                d2 = (near[:, 0] - tracks[i, dd, g, 0]) ** 2 + (near[:, 1] - tracks[i, dd, g, 1]) ** 2
                st_d[g // PER_STATION].append(float(np.sqrt(d2.min())))
    print("\nclean trajectory check -- our crossing vs nearest FairShip MC hit (transverse, cm):")
    for s in range(4):
        a = np.asarray(st_d[s])
        print(f"  station {s+1}: median={np.median(a):7.3f}  p90={np.percentile(a,90):7.3f}  (n={len(a)})")


if __name__ == "__main__":
    import gearup

    gearup.gearup(run).with_config("config/regression.yaml")()
