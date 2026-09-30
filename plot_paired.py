#!/usr/bin/env python3
"""Our tracker vs FairSHiP's own reco, HNL metrics against stereo angle, on paired events.

  plot_paired.py <trackers.npz> <out_prefix> [method]
"""
import sys

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

from detopt.data.fairship_loader import load_fairship_digi

ANG = {"a000": 0.000, "a315": 3.295, "a457": 4.570, "a630": 6.253, "a826": 8.400, "a1022": 10.220}
DEFAULT_DEG, BO_DEG = 4.570, 10.220
MEAS_CUT, MIN_STATIONS, CHI2_CUT, DOCA_CUT = 25, 3, 4.0, 2.0
MAD_SE = 1.1664
MIN_PAIRED = 20

z = np.load(sys.argv[1], allow_pickle=True)
out_prefix = sys.argv[2]
method = sys.argv[3] if len(sys.argv) > 3 else "nll-tdc"

mad = lambda e: 1.4826 * float(np.median(np.abs(e - np.median(e))))
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))


def hnl(a, t9):
    hp, ht = a[:, 3:6] + a[:, 6:9], t9[:, 3:6] + t9[:, 6:9]
    dm = np.linalg.norm(hp, axis=1) - np.linalg.norm(ht, axis=1)
    dirn = hp / np.linalg.norm(hp, axis=1, keepdims=True)
    return dict(
        p=dm,
        dpp=dm / np.linalg.norm(ht, axis=1),
        pt=np.hypot(hp[:, 0], hp[:, 1]) - np.hypot(ht[:, 0], ht[:, 1]),
        ip=np.linalg.norm(np.cross(-a[:, :3], dirn), axis=1),
    )


QUANT = [("p", "HNL $|p|$", "GeV/$c$"), ("dpp", "HNL $\\Delta p / p$", ""),
         ("pt", "HNL $p_\\mathrm{T}$", "GeV/$c$"), ("ip", "HNL IP", "cm")]

res = {}
for lab, a in ANG.items():
    if f"{lab}__true9" not in z.files:
        continue
    g = lambda k: z[f"{lab}__{method}__{k}"]
    pred, true9 = g("pred9"), z[f"{lab}__true9"]
    src, _ = load_fairship_digi(None, columns=("truth", "reco"),
                                data_glob=f"/home/max/dev/detopt/data/fairship-npz/{lab}/*.npz")
    rc = np.asarray(src["reco"])
    reco9 = np.concatenate([rc[:, 4:7], rc[:, 7:10], rc[:, 13:16]], 1)
    n_rows = pred.shape[0]
    ours_ok = ((g("nhits") >= MEAS_CUT).all(1) & (g("n_stations") >= MIN_STATIONS).all(1)
               & (g("chi2") < CHI2_CUT).all(1) & (g("doca") <= DOCA_CUT))
    both = ours_ok & z[f"{lab}__reco_ok"][:n_rows] & ~np.isnan(pred).any(1)
    n = int(both.sum())
    if n < MIN_PAIRED:
        continue
    res[a] = dict(n=n,
                  fs=hnl(reco9[:n_rows][both], true9[:n_rows][both]),
                  ours=hnl(pred[both], true9[:n_rows][both]))

angles = sorted(res)
xs = np.array(angles)

fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for r, (stat, fn, lab) in enumerate((("rmse", rmse, "RMSE"), ("core", mad, "robust $\\sigma$"))):
    for ax, (key, title, unit) in zip(axes[r], QUANT):
        for who, colour, name in (("fs", "tab:green", "FairSHiP"),
                                  ("ours", "tab:purple", method)):
            ys = np.array([fn(res[a][who][key]) for a in angles])
            ns = np.array([res[a]["n"] for a in angles])
            es = ys * (MAD_SE / np.sqrt(ns) if stat == "core" else 1.0 / np.sqrt(2 * (ns - 1)))
            ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=5, capsize=3, lw=1.5,
                        color=colour, label=name)
        ax.axvline(DEFAULT_DEG, color="grey", lw=1, ls=":")
        ax.axvline(BO_DEG, color="crimson", lw=1, ls=":")
        ax.set_xlabel("stereo angle, deg")
        ax.set_ylabel(f"{lab}, {unit}" if unit else lab)
        ax.set_title(f"{title} --- {lab}")
        ax.grid(alpha=0.25)
        if r == 0 and key == "p":
            ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{out_prefix}.png", dpi=140)
plt.close(fig)
print(f"wrote {out_prefix}.png  (angles: {', '.join(f'{a:.3f}' for a in angles)})")

with open(f"{out_prefix}.txt", "w") as fh:
    fh.write(f"{method} vs FairSHiP, paired events, HNL metrics\n\n")
    for stat, fn in (("RMSE", rmse), ("core", mad)):
        fh.write(f"=== {stat} ===\n")
        fh.write(f"{'angle':>8} {'n':>6} {'who':>10} " + " ".join(f"{q[1][:12]:>14}" for q in QUANT) + "\n")
        for a in angles:
            for who, name in (("fs", "FairSHiP"), ("ours", method)):
                fh.write(f"{a:8.3f} {res[a]['n']:6d} {name:>10} "
                         + " ".join(f"{fn(res[a][who][q[0]]):14.4f}" for q in QUANT) + "\n")
        fh.write("\n")
print(f"wrote {out_prefix}.txt")
