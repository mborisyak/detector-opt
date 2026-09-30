#!/usr/bin/env python3
"""Plots, histograms and tables for our classical trackers across the stereo-angle scan.

  plot_trackers.py <trackers.npz> <out_prefix> [hist_method]

Writes, per statistic (robust sigma / RMSE): a 3x3 grid, one component per panel, one line per tracker;
a 3x3 histogram grid for one tracker with the angles overlaid; and a text table.
"""

import sys

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

ANGLES_DEG = {"a000": 0.000, "a315": 3.295, "a457": 4.570,
              "a630": 6.253, "a826": 8.400, "a1022": 10.220}
DEFAULT_DEG, BO_DEG = 4.570, 10.220
NAMES = ["vertex_x", "vertex_y", "vertex_z", "p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z"]
ROWS = [("vertex", [0, 1, 2], "cm"), ("$p_1$", [3, 4, 5], "GeV/$c$"), ("$p_2$", [6, 7, 8], "GeV/$c$")]
MEAS_CUT, MIN_STATIONS, CHI2_CUT, DOCA_CUT = 25, 3, 4.0, 2.0
MAD_SE = 1.1664

path, out_prefix = sys.argv[1:3]
hist_method = sys.argv[3] if len(sys.argv) > 3 else "nll-tdc"
z = np.load(path, allow_pickle=True)

labels = [l for l in ANGLES_DEG if f"{l}__true9" in z.files]
methods = sorted({k.split("__")[1] for k in z.files if k.count("__") == 2})


def accept(label, method):
    g = lambda k: z[f"{label}__{method}__{k}"]
    return ((g("nhits") >= MEAS_CUT).all(1) & (g("n_stations") >= MIN_STATIONS).all(1)
            & (g("chi2") < CHI2_CUT).all(1) & (g("doca") <= DOCA_CUT))


def errors(label, method):
    pred = z[f"{label}__{method}__pred9"]
    true = z[f"{label}__true9"][: pred.shape[0]]
    ok = accept(label, method)
    return (pred - true)[ok], float(ok.mean())


def stat(e):
    n = e.size
    if n < 5:
        return dict(core=np.nan, core_err=np.nan, rms=np.nan, rms_err=np.nan)
    rms = float(np.std(e, ddof=1))
    core = 1.4826 * float(np.median(np.abs(e - np.median(e))))
    return dict(core=core, core_err=MAD_SE * core / np.sqrt(n),
                rms=rms, rms_err=rms / np.sqrt(2 * (n - 1)))


tab, acc, raw = {}, {}, {}
for m in methods:
    for l in labels:
        e, a = errors(l, m)
        raw[(m, l)] = e
        acc[(m, l)] = a
        tab[(m, l)] = [stat(e[:, k]) for k in range(9)] if e.size else [stat(np.array([]))] * 9

xs = np.array([ANGLES_DEG[l] for l in labels])

for key, lab in (("core", "robust $\\sigma$"), ("rms", "RMSE")):
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    for r, (title, idx, unit) in enumerate(ROWS):
        for c, (k, comp) in enumerate(zip(idx, "xyz")):
            ax = axes[r, c]
            for j, m in enumerate(methods):
                ys = np.array([tab[(m, l)][k][key] for l in labels])
                es = np.array([tab[(m, l)][k][key + "_err"] for l in labels])
                ok = np.isfinite(ys)
                if ok.sum():
                    ax.errorbar(xs[ok], ys[ok], yerr=es[ok], fmt="o-", ms=4, capsize=2, lw=1.2,
                                color=plt.cm.tab10(j % 10), label=m)
            ax.axvline(DEFAULT_DEG, color="grey", lw=1, ls=":")
            ax.axvline(BO_DEG, color="crimson", lw=1, ls=":")
            ax.set_xlabel("stereo angle, deg")
            ax.set_ylabel(f"{lab}, {unit}")
            ax.set_title(f"{title} ${comp}$")
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"our trackers on FairShip hits: {lab} vs stereo angle "
                 f"(dotted grey = default, red = BO optimum)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out_prefix}-{key}.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out_prefix}-{key}.png")

# histograms for one tracker, angles overlaid
if hist_method in methods:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for r, (title, idx, unit) in enumerate(ROWS):
        for c, (k, comp) in enumerate(zip(idx, "xyz")):
            ax = axes[r, c]
            pooled = np.concatenate([raw[(hist_method, l)][:, k] for l in labels
                                     if raw[(hist_method, l)].size])
            lo, hi = np.quantile(pooled, [0.005, 0.995])
            pad = 0.1 * (hi - lo)
            bins = np.linspace(lo - pad, hi + pad, 70)
            for j, l in enumerate(labels):
                e = raw[(hist_method, l)]
                if not e.size:
                    continue
                ax.hist(e[:, k], bins=bins, histtype="step", density=True, lw=1.3,
                        color=plt.cm.viridis(j / max(len(labels) - 1, 1)),
                        label=f"{ANGLES_DEG[l]:.2f}$^\\circ$")
            ax.set_xlabel(f"{title} {comp} error, {unit}")
            ax.set_ylabel("prob. density")
            ax.set_title(f"{title} ${comp}$")
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, title="stereo angle")
    fig.suptitle(f"our trackers on FairShip hits ({hist_method}): error distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out_prefix}-hist.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out_prefix}-hist.png")

with open(f"{out_prefix}.txt", "w") as fh:
    for m in methods:
        fh.write(f"===== {m} =====\n")
        fh.write(f"{'angle':>8} {'accept':>8} " + " ".join(f"{n:>22}" for n in NAMES) + "\n")
        for l in labels:
            fh.write(f"{ANGLES_DEG[l]:8.3f} {acc[(m, l)]:8.3f} " + " ".join(
                f"{tab[(m, l)][k]['core']:11.3f} +-{tab[(m, l)][k]['core_err']:8.3f}"
                for k in range(9)) + "\n")
        fh.write("\n")
print(f"wrote {out_prefix}.txt  (robust sigma = 1.4826 MAD; acceptance under our cuts)")
