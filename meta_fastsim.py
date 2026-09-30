#!/usr/bin/env python3
"""MLOE reconstruction vs stereo angle, on OUR fast simulation.

No FairShip hits are involved, so neither the straw-mirror nor the layer-z mismatch applies.

  python meta_fastsim.py <checkpoint> <config.yaml> <out_prefix> [n_events]
"""

import os
import sys

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import curve_fit
import jax
import jax.numpy as jnp
from flax import nnx

import detopt
import detopt.detector
import detopt.nn
import detopt.utils.io
from detopt.nn.trainer.common import regressor_rngs

ANGLES_DEG = [0.000, 3.295, 4.570, 6.253, 8.400, 10.220]
DEFAULT_DEG, BO_DEG = 4.570, 10.220
NAMES = ["vertex_x", "vertex_y", "vertex_z", "p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z",
         "pHNL_x", "pHNL_y", "pHNL_z", "pHNL_abs", "pHNL_pt", "ip_reco", "ip_true", "pHNL_dpp"]
CHUNK = 512

checkpoint, config_path, out_prefix = sys.argv[1:4]
n_events = int(sys.argv[4]) if len(sys.argv) > 4 else 8192

config = yaml.safe_load(open(config_path))
det_cfg = dict(config["detector"])
key = next(iter(det_cfg))
det_cfg[key] = dict(det_cfg[key])
det_cfg[key]["data_dir"] = os.environ.get("DETOPT_MC", "/home/max/dev/detopt/data/mc/numpy_newFS")
detector = detopt.detector.from_config(det_cfg)

reg = detopt.nn.from_config(detector, config=config["regressor"],
                            rngs=regressor_rngs(int(config.get("seed", 0))), design=True)
reg_def, params, state = nnx.split(reg, nnx.Param, nnx.Variable)
manager = detopt.utils.io.get_checkpointer(checkpoint)
restored, _s, _d, _a = detopt.utils.io.restore_training_checkpoint(manager, regressor=(params, state))
manager.close()
nnx.replace_by_pure_dict(params, restored)
model = nnx.merge(reg_def, params, state)

mean, std = (np.asarray(a) for a in detector._target_norm_arrays())
MAD_SE = 1.1664


@jax.jit
def fwd(feats, m):
    return model(feats, m, deterministic=True)


def gauss_core(e, core_mad, window=2.5, nbins=60):
    """Gaussian sigma fitted to the central +-window*MAD of the distribution, with its fit error.
    Directly comparable to the per-histogram Gaussian fits macro/ship.py reports."""
    med = np.median(e)
    sel = e[np.abs(e - med) < window * max(core_mad, 1e-12)]
    if sel.size < 50:
        return np.nan, np.nan
    counts, edges = np.histogram(sel, bins=nbins, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    model = lambda x, a, mu, sg: a * np.exp(-0.5 * ((x - mu) / sg) ** 2)
    try:
        popt, pcov = curve_fit(model, centres, counts,
                               p0=[counts.max(), med, max(core_mad, 1e-9)], maxfev=20000)
        return abs(float(popt[2])), float(np.sqrt(abs(pcov[2, 2])))
    except Exception:
        return np.nan, np.nan


def at_angle(deg):
    design = np.array([np.radians(deg)], np.float32)
    theta = np.asarray(detector.to_scaled(design), np.float32)
    idx = np.arange(n_events, dtype=np.int64)
    _gt, event, mask, target = detector(design, idx)
    true9 = np.concatenate([np.asarray(target.vertex), np.asarray(target.p1),
                            np.asarray(target.p2)], 1)
    preds = []
    for i in range(0, n_events, CHUNK):
        sl = slice(i, min(i + CHUNK, n_events))
        ev = jax.tree.map(lambda a: jnp.asarray(a[sl]), event)
        m = jnp.asarray(mask[sl])
        preds.append(np.asarray(fwd(detector.combine_scaled(ev, theta, mask=m, reveal_design=True), m)))
    pred = np.concatenate(preds, 0)
    if pred.ndim == 3:
        pred = pred.mean(axis=0)
    phys = pred * std + mean
    err = phys - true9
    # HNL momentum is the sum of the two daughters; |p| error is on the magnitude.
    hp, ht = phys[:, 3:6] + phys[:, 6:9], true9[:, 3:6] + true9[:, 6:9]
    mag = np.linalg.norm(hp, axis=1) - np.linalg.norm(ht, axis=1)
    pt = np.hypot(hp[:, 0], hp[:, 1]) - np.hypot(ht[:, 0], ht[:, 1])

    def ip(vertex, mom):
        """Closest approach of the line (vertex, mom) to the target at the origin."""
        d = mom / np.linalg.norm(mom, axis=1, keepdims=True)
        return np.linalg.norm(np.cross(-vertex, d), axis=1)

    ip_reco = ip(phys[:, :3], hp)
    ip_true = ip(true9[:, :3], ht)
    dpp = mag / np.linalg.norm(ht, axis=1)
    return np.concatenate([err, hp - ht, mag[:, None], pt[:, None],
                           ip_reco[:, None], ip_true[:, None], dpp[:, None]], axis=1)


rows = {}
raw = {}
for deg in ANGLES_DEG:
    err = at_angle(deg)
    raw[deg] = err
    n = err.shape[0]
    per = []
    for k in range(len(NAMES)):
        e = err[:, k]
        rms = float(np.sqrt(np.mean(e ** 2)))   # RMSE about zero, bias included
        core = 1.4826 * float(np.median(np.abs(e - np.median(e))))
        gs, ge = gauss_core(e, core)
        per.append(dict(rms=rms, rms_err=rms / np.sqrt(2 * (n - 1)),
                        core=core, core_err=MAD_SE * core / np.sqrt(n),
                        gauss=gs, gauss_err=ge))
    rows[deg] = per
    print(f"  {deg:6.3f} deg done ({n} events)", flush=True)

with open(f"{out_prefix}.txt", "w") as fh:
    fh.write(f"MLOE on our fast simulation, {n_events} events per angle\n\n")
    for stat in ("core", "gauss", "rms"):
        fh.write(f"=== {stat} (+- error) ===\n")
        fh.write(f"{'angle':>8} " + " ".join(f"{c:>20}" for c in NAMES) + "\n")
        for deg in ANGLES_DEG:
            fh.write(f"{deg:8.3f} " + " ".join(
                f"{rows[deg][k][stat]:11.3f} +-{rows[deg][k][stat + '_err']:7.3f}" for k in range(len(NAMES))) + "\n")
        fh.write("\n")
print(f"wrote {out_prefix}.txt")

ROWS = [("vertex", [0, 1, 2], ["cm", "cm", "cm"], ["$x$", "$y$", "$z$"]),
        ("$p_1$", [3, 4, 5], ["GeV/$c$"] * 3, ["$x$", "$y$", "$z$"]),
        ("$p_2$", [6, 7, 8], ["GeV/$c$"] * 3, ["$x$", "$y$", "$z$"]),
        ("$p_\mathrm{HNL}$", [9, 10, 11], ["GeV/$c$"] * 3, ["$x$", "$y$", "$z$"]),
        ("HNL", [12, 13, 14], ["GeV/$c$", "GeV/$c$", "cm"], ["$|p|$", "$p_\mathrm{T}$", "IP"])]
xs = np.array(ANGLES_DEG)

for stat, colour in (("core", "tab:orange"), ("rms", "tab:blue")):
    lab = "robust $\\sigma$" if stat == "core" else "RMSE"
    fig, axes = plt.subplots(5, 3, figsize=(13, 15))
    for r, (title, idx, units, comps) in enumerate(ROWS):
        for c, (k, comp, unit) in enumerate(zip(idx, comps, units)):
            ax = axes[r, c]
            ys = np.array([rows[d][k][stat] for d in ANGLES_DEG])
            es = np.array([rows[d][k][stat + "_err"] for d in ANGLES_DEG])
            ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=5, capsize=3, lw=1.5,
                        color=colour)
            ax.axvline(DEFAULT_DEG, color="grey", lw=1, ls=":")
            ax.axvline(BO_DEG, color="crimson", lw=1, ls=":")
            ax.set_xlabel("stereo angle, deg")
            ax.set_ylabel(f"{lab}, {unit}")
            ax.set_title(f"{title} {comp}")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}-{stat}.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out_prefix}-{stat}.png")


# ---------------------------------------------------------------- histograms
def softrange(vals, q=0.005, margin=0.1):
    lo, hi = np.quantile(vals, [q, 1 - q])
    pad = margin * (hi - lo)
    return lo - pad, hi + pad


fig, axes = plt.subplots(5, 3, figsize=(13, 15))
for r, (title, idx, units, comps) in enumerate(ROWS):
    for c, (k, comp, unit) in enumerate(zip(idx, comps, units)):
        ax = axes[r, c]
        pooled = np.concatenate([raw[d][:, k] for d in ANGLES_DEG])
        lo, hi = softrange(pooled)
        bins = np.linspace(lo, hi, 80)
        for j, d in enumerate(ANGLES_DEG):
            ax.hist(raw[d][:, k], bins=bins, histtype="step", density=True, lw=1.3,
                    color=plt.cm.viridis(j / (len(ANGLES_DEG) - 1)), label=f"{d:.2f}$^\\circ$")
        ax.set_xlabel(f"{title} {comp}, {unit}")
        ax.set_ylabel("prob. density")
        ax.set_title(f"{title} {comp}")
        ax.grid(alpha=0.25)
        if r == 0 and c == 0:
            ax.legend(fontsize=8, title="stereo angle")
fig.tight_layout()
fig.savefig(f"{out_prefix}-hist.png", dpi=140)
plt.close(fig)
print(f"wrote {out_prefix}-hist.png")

# ---------------------------------------------------------------- slide figures: HNL |p|, p_T, IP
SLIDE = [(12, "HNL $|p|$", "GeV/$c$"), (16, "HNL $\\Delta p / p$", ""),
         (13, "HNL $p_\\mathrm{T}$", "GeV/$c$"), (14, "HNL IP", "cm")]

fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for r, (stat, lab, col) in enumerate((("rms", "RMSE", "tab:blue"), ("core", "robust $\\sigma$", "tab:orange"))):
    for ax, (k, title, unit) in zip(axes[r], SLIDE):
        ys = np.array([rows[d][k][stat] for d in ANGLES_DEG])
        es = np.array([rows[d][k][stat + "_err"] for d in ANGLES_DEG])
        ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=5, capsize=3, lw=1.5, color=col)
        ax.axvline(DEFAULT_DEG, color="grey", lw=1, ls=":")
        ax.axvline(BO_DEG, color="crimson", lw=1, ls=":")
        ax.set_xlabel("stereo angle, deg")
        ax.set_ylabel(f"{lab}, {unit}" if unit else lab)
        ax.set_title(f"{title} --- {lab}")
        ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{out_prefix}-slides.png", dpi=140)
plt.close(fig)
print(f"wrote {out_prefix}-slides.png")

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, (k, title, unit) in zip(axes, SLIDE):
    pooled = np.concatenate([raw[d][:, k] for d in ANGLES_DEG])
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    pad = 0.1 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 70)
    for j, d in enumerate(ANGLES_DEG):
        ax.hist(raw[d][:, k], bins=bins, histtype="step", density=True, lw=1.4,
                color=plt.cm.viridis(j / (len(ANGLES_DEG) - 1)), label=f"{d:.2f}$^\\circ$")
    ax.set_xlabel(f"{title}, {unit}" if unit else title)
    ax.set_ylabel("prob. density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8, title="stereo angle")
fig.tight_layout()
fig.savefig(f"{out_prefix}-slides-hist.png", dpi=140)
plt.close(fig)
print(f"wrote {out_prefix}-slides-hist.png")
