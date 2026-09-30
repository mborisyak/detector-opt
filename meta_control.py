#!/usr/bin/env python3
"""Control for meta_vs_angle.py: run the SAME restore + combine + forward path on detopt's OWN
simulated events, at a design the network actually saw during training.

If the core widths here are small (well below the target sigmas) while the FairShip-hit numbers are
at or above them, the degradation is a genuine domain shift. If they are equally bad, the evaluation
pipeline is wrong and the FairShip numbers mean nothing.
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
from detopt.nn.trainer.common import regressor_rngs

checkpoint, config_path = sys.argv[1:3]
angle_rad = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
n_events = int(sys.argv[4]) if len(sys.argv) > 4 else 4096

config = yaml.safe_load(open(config_path))
det_cfg = dict(config["detector"])
key = next(iter(det_cfg))
det_cfg[key] = dict(det_cfg[key])
det_cfg[key]["data_dir"] = os.environ.get("DETOPT_MC", "/home/max/dev/detopt/data/mc/numpy_newFS")
detector = detopt.detector.from_config(det_cfg)

seed = int(config.get("seed", 0))
reg = detopt.nn.from_config(detector, config=config["regressor"], rngs=regressor_rngs(seed), design=True)
reg_def, params, state = nnx.split(reg, nnx.Param, nnx.Variable)
manager = detopt.utils.io.get_checkpointer(checkpoint)
restored, _s, _d, _a = detopt.utils.io.restore_training_checkpoint(manager, regressor=(params, state))
manager.close()
nnx.replace_by_pure_dict(params, restored)
model = nnx.merge(reg_def, params, state)

mean, std = (np.asarray(a) for a in detector._target_norm_arrays())
design = np.array([angle_rad], np.float32)
theta = np.asarray(detector.to_scaled(design), np.float32)

# Simulate our own events at this design (the network's native input distribution).
idx = np.arange(n_events, dtype=np.int64)
_gt, event, mask, target = detector(design, idx)
true9 = np.concatenate([np.asarray(target.vertex), np.asarray(target.p1), np.asarray(target.p2)], 1)


@jax.jit
def predict(feats, m):
    return model(feats, m, deterministic=True)


preds = []
for i in range(0, n_events, 512):
    sl = slice(i, min(i + 512, n_events))
    ev = jax.tree.map(lambda a: jnp.asarray(a[sl]), event)
    m = jnp.asarray(mask[sl])
    preds.append(np.asarray(predict(detector.combine_scaled(ev, theta, mask=m, reveal_design=True), m)))
pred = np.concatenate(preds, 0)
if pred.ndim == 3:
    pred = pred.mean(axis=0)
phys = pred * std + mean
err = phys - true9

names = ["vertex_x", "vertex_y", "vertex_z", "p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z"]
sig = np.concatenate([detector.decay_sigma, detector.daughter_momentum_sigma, detector.daughter_momentum_sigma])
print(f"CONTROL: our own simulation, angle {np.degrees(angle_rad):.3f} deg, {n_events} events")
print(f"{'component':>10} {'core':>10} {'rms':>10} {'target sigma':>13} {'core/sigma':>11}")
for k, n in enumerate(names):
    e = err[:, k]
    core = 1.4826 * float(np.median(np.abs(e - np.median(e))))
    print(f"{n:>10} {core:10.3f} {float(np.std(e, ddof=1)):10.3f} {sig[k]:13.3f} {core / sig[k]:11.3f}")
