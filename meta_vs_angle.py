#!/usr/bin/env python3
"""Evaluate the trained `meta` network on FairShip's own digitized hits, at each scan angle.

The `meta` network is design-conditioned (it is handed the scaled design as part of the per-hit
features), so ONE checkpoint can be scored at every angle of the scan. For each angle we:

  * load that angle's Ship2NumPy npz (FairShip's own digis),
  * pack them into the fired-straw StrawEvent the network's combine consumes,
  * build features with the design set to that angle,
  * forward the network and compare to MC truth.

Caveat carried through to the plot: the network was trained on our fast simulation, so on FairShip
hits this is an out-of-distribution estimate (different digitization, efficiency and secondaries).

  python meta_vs_angle.py <npz_root> <checkpoint> <config.yaml> <out_prefix>
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
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

ANGLES_DEG = {
    "a000": 0.000, "a315": 3.295, "a457": 4.570,
    "a630": 6.253, "a826": 8.400, "a1022": 10.220,
}
COMPONENTS = ["vertex_x", "vertex_y", "vertex_z", "p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z"]
CHUNK = 512

npz_root, checkpoint, config_path, out_prefix = sys.argv[1:5]
config = yaml.safe_load(open(config_path))

# The detector is only used for its geometry, combine and target normalisation; point data_dir at the
# local MC pool so construction succeeds (no simulation is run here).
det_cfg = dict(config["detector"])
key = next(iter(det_cfg))
det_cfg[key] = dict(det_cfg[key])
det_cfg[key]["data_dir"] = os.environ.get("DETOPT_MC", "/home/max/dev/detopt/data/mc/numpy_newFS")
detector = detopt.detector.from_config(det_cfg)

# --- restore the meta network -------------------------------------------------------------------
seed = int(config.get("seed", 0))
reg = detopt.nn.from_config(detector, config=config["regressor"], rngs=regressor_rngs(seed), design=True)
reg_def, params, state = nnx.split(reg, nnx.Param, nnx.Variable)
manager = detopt.utils.io.get_checkpointer(checkpoint)
restored, _state, _design, _aux = detopt.utils.io.restore_training_checkpoint(manager, regressor=(params, state))
manager.close()
nnx.replace_by_pure_dict(params, restored)
model = nnx.merge(reg_def, params, state)
print(f"restored {checkpoint}", flush=True)

mean, std = detector._target_norm_arrays()
mean, std = np.asarray(mean), np.asarray(std)


@jax.jit
def predict(feats, mask):
    return model(feats, mask, deterministic=True)


def evaluate(label):
    """Per-event physical prediction and truth for one angle."""
    files = sorted(glob.glob(os.path.join(npz_root, label, "*.npz")))
    if not files:
        return None
    data, nf = load_fairship_digi(None, data_glob=os.path.join(npz_root, label, "*.npz"))
    event, mask, rows, truth = pack_fairship_events(
        data, n_stations=detector.n_stations, n_views_per_station=detector.n_views_per_station,
        n_layers_per_view=detector.n_layers_per_view, n_straws=detector.n_straws,
        max_hits=detector.max_hits_per_event,
    )
    true9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], 1)

    angle_rad = np.radians(ANGLES_DEG[label])
    theta = np.asarray(detector.to_scaled(np.array([angle_rad], np.float32)), np.float32)

    preds = []
    n = jax.tree.leaves(event)[0].shape[0]
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        ev = jax.tree.map(lambda a: jnp.asarray(a[sl]), event)
        m = jnp.asarray(mask[sl])
        feats = detector.combine_scaled(ev, theta, mask=m, reveal_design=True)
        preds.append(np.asarray(predict(feats, m)))
    pred = np.concatenate(preds, 0)
    if pred.ndim == 3:            # ensemble: average the members
        pred = pred.mean(axis=0)
    phys = pred * std + mean
    print(f"  {label}: {nf} files, {n} events", flush=True)
    return phys, true9[: phys.shape[0]]


def core(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def control(label):
    """Same network, same design, but on OUR OWN simulated events -- the network's native input."""
    angle = np.radians(ANGLES_DEG[label])
    design = np.array([angle], np.float32)
    theta = np.asarray(detector.to_scaled(design), np.float32)
    idx = np.arange(int(os.environ.get("CONTROL_N", 4096)), dtype=np.int64)
    _gt, event, mask, target = detector(design, idx)
    true9 = np.concatenate([np.asarray(target.vertex), np.asarray(target.p1), np.asarray(target.p2)], 1)
    preds = []
    for i in range(0, idx.size, CHUNK):
        sl = slice(i, min(i + CHUNK, idx.size))
        ev = jax.tree.map(lambda a: jnp.asarray(a[sl]), event)
        m = jnp.asarray(mask[sl])
        preds.append(np.asarray(predict(detector.combine_scaled(ev, theta, mask=m, reveal_design=True), m)))
    pred = np.concatenate(preds, 0)
    if pred.ndim == 3:
        pred = pred.mean(axis=0)
    return pred * std + mean, true9


rows = {}
ctrl = {}
for label in ANGLES_DEG:
    got = evaluate(label)
    if got is None:
        continue
    phys, true9 = got
    err = phys - true9
    rows[label] = dict(
        n=err.shape[0],
        per=[dict(mean=float(np.mean(err[:, k])), rms=float(np.std(err[:, k], ddof=1)),
                  core=core(err[:, k])) for k in range(9)],
    )
    cphys, ctrue = control(label)
    cerr = cphys - ctrue
    ctrl[label] = dict(per=[dict(rms=float(np.std(cerr[:, k], ddof=1)), core=core(cerr[:, k]))
                            for k in range(9)])
    print(f"  {label}: control on our own sim done", flush=True)

# --- table ---------------------------------------------------------------------------------------
with open(f"{out_prefix}.txt", "w") as fh:
    fh.write("meta network on FairShip hits (out-of-distribution: trained on our fast simulation)\n\n")
    for stat in ("core", "rms"):
        fh.write(f"=== {stat} ===\n")
        fh.write(f"{'angle':>8} {'n':>7} " + " ".join(f"{c:>10}" for c in COMPONENTS) + "\n")
        for label in sorted(rows, key=lambda l: ANGLES_DEG[l]):
            r = rows[label]
            fh.write(f"{ANGLES_DEG[label]:8.3f} {r['n']:7d} "
                     + " ".join(f"{p[stat]:10.3f}" for p in r["per"]) + "\n")
        fh.write("\n")
print(f"wrote {out_prefix}.txt")

# --- plot ----------------------------------------------------------------------------------------
GROUPS = [("vertex", [0, 1, 2], "cm"), ("$p_1$", [3, 4, 5], "GeV/$c$"), ("$p_2$", [6, 7, 8], "GeV/$c$")]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
labels = sorted(rows, key=lambda l: ANGLES_DEG[l])
xs = np.array([ANGLES_DEG[l] for l in labels])
for row, stat in enumerate(("core", "rms")):
    for ax, (title, idx, unit) in zip(axes[row], GROUPS):
        for j, (k, comp) in enumerate(zip(idx, "xyz")):
            col = plt.cm.tab10(j)
            ys = np.array([rows[l]["per"][k][stat] for l in labels])
            ax.plot(xs, ys, "o-", ms=5, color=col, label=f"{comp}, FairShip hits")
            cy = np.array([ctrl[l]["per"][k][stat] for l in labels])
            ax.plot(xs, cy, "s--", ms=4, color=col, alpha=0.65, label=f"{comp}, our sim")
        ax.axvline(4.570, color="grey", lw=1, ls=":")
        ax.axvline(10.220, color="crimson", lw=1, ls=":")
        ax.set_xlabel("stereo angle, deg")
        ax.set_ylabel(unit)
        ax.set_title(f"{title} --- {stat}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
fig.suptitle("meta network: error vs stereo angle. Solid = FairShip digis, dashed = our own simulation "
             "(its training distribution). Dotted grey = stock, red = BO optimum.", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{out_prefix}.png", dpi=140)
print(f"wrote {out_prefix}.png")
