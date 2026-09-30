#!/usr/bin/env python3
"""Our trackers vs FairSHiP's own reco on the SAME events, per stereo angle, HNL quantities."""
import sys

import numpy as np

ANG = {"a000": 0.000, "a315": 3.295, "a457": 4.570, "a630": 6.253, "a826": 8.400, "a1022": 10.220}
MEAS_CUT, MIN_STATIONS, CHI2_CUT, DOCA_CUT = 25, 3, 4.0, 2.0
MAD_SE = 1.1664
mad = lambda e: 1.4826 * float(np.median(np.abs(e - np.median(e))))

z = np.load(sys.argv[1], allow_pickle=True)
method = sys.argv[2] if len(sys.argv) > 2 else "nll-tdc"


def hnl(a, t9):
    hp, ht = a[:, 3:6] + a[:, 6:9], t9[:, 3:6] + t9[:, 6:9]
    dm = np.linalg.norm(hp, axis=1) - np.linalg.norm(ht, axis=1)
    dirn = hp / np.linalg.norm(hp, axis=1, keepdims=True)
    return (dm, dm / np.linalg.norm(ht, axis=1),
            np.hypot(hp[:, 0], hp[:, 1]) - np.hypot(ht[:, 0], ht[:, 1]),
            np.linalg.norm(np.cross(-a[:, :3], dirn), axis=1))


print(f"{method} vs FairSHiP, paired on events both reconstructed")
print(f"{'angle':>7} {'n':>6} | {'|p| core':>16} | {'dp/p core':>16} | {'pT core':>16} | {'IP core':>14}")
for lab, a in ANG.items():
    if f"{lab}__true9" not in z.files:
        continue
    g = lambda k: z[f"{lab}__{method}__{k}"]
    pred, true9 = g("pred9"), z[f"{lab}__true9"]
    # reco9 is not in the archive (the final save overwrote the incremental one); rebuild it
    # from the source npz exactly as fairship_hits.store does, then take the same leading rows.
    from detopt.data.fairship_loader import load_fairship_digi
    src, _ = load_fairship_digi(None, columns=("truth", "reco"),
                                data_glob=f"/home/max/dev/detopt/data/fairship-npz/{lab}/*.npz")
    rc = np.asarray(src["reco"])
    reco9 = np.concatenate([rc[:, 4:7], rc[:, 7:10], rc[:, 13:16]], 1)
    ours_ok = ((g("nhits") >= MEAS_CUT).all(1) & (g("n_stations") >= MIN_STATIONS).all(1)
               & (g("chi2") < CHI2_CUT).all(1) & (g("doca") <= DOCA_CUT))
    n_rows = pred.shape[0]
    both = ours_ok & z[f"{lab}__reco_ok"][:n_rows] & ~np.isnan(pred).any(1)
    n = int(both.sum())
    if n < 20:
        print(f"{a:7.3f} {n:6d} |  (too few paired events)")
        continue
    for who, arr in (("FairSHiP", reco9[:n_rows][both]), (method, pred[both])):
        dm, dpp, dpt, ipv = hnl(arr, true9[:n_rows][both])
        f = lambda e: f"{mad(e):8.4f}+-{MAD_SE * mad(e) / np.sqrt(n):6.4f}"
        tag = f"{a:7.3f} {n:6d}" if who == "FairSHiP" else " " * 14
        print(f"{tag} | {f(dm)} | {f(dpp)} | {f(dpt)} | "
              f"{mad(ipv):6.3f}+-{MAD_SE * mad(ipv) / np.sqrt(n):5.3f}  {who}")
