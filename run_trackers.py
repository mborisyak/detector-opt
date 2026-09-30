#!/usr/bin/env python3
"""Run our classical trackers (retina + NLL, 4 measurement versions each) on FairShip's own digis,
per stereo-angle point of the scan.

Two patches to scripts/fairship_hits.py, applied here rather than in place:
  * load_field derives the grid from the file instead of assuming the old 95x82x199 map;
  * FIELD_PATH points at the ECN3 MgB2 map the scan was simulated with.

  python run_trackers.py <npz_root> <out_prefix> [n_events] [methods...]
"""

import os
import sys
import time

sys.path.insert(0, "scripts")

import numpy as np
import jax.numpy as jnp
import uproot
import yaml
from scipy.interpolate import RegularGridInterpolator

import detopt
import detopt.detector
import fairship_hits as FH

FIELD = "/home/max/dev/FairShip/files/2025_02_12_SHiP_SpectrometerField_ECN3_MgB2.root"
DET_CFG = "config/detector/stereo_tracker_truth.yaml"
DES_CFG = "config/design/initial_stereo.yaml"
ANGLES_DEG = {"a000": 0.000, "a315": 3.295, "a457": 4.570,
              "a630": 6.253, "a826": 8.400, "a1022": 10.220}


def load_field(crop=350.0, crop_z=650.0, fine=2.5):
    """Bx map cropped to the tracker region and cubic-upsampled. Grid read from the file."""
    d = uproot.open(FIELD)["Data"].arrays(library="np")
    xg, yg, zg = (np.unique(d[k]) for k in ("x", "y", "z"))
    BX = d["Bx"].reshape(xg.size, yg.size, zg.size).astype(np.float64)
    xi, yi, zi = np.abs(xg) <= crop, np.abs(yg) <= crop, np.abs(zg) <= crop_z
    rgi = RegularGridInterpolator((xg[xi], yg[yi], zg[zi]), BX[np.ix_(xi, yi, zi)], method="cubic")
    xf = np.arange(xg[xi][0], xg[xi][-1] + 1e-6, fine)
    yf = np.arange(yg[yi][0], yg[yi][-1] + 1e-6, fine)
    zf = np.arange(zg[zi][0], zg[zi][-1] + 1e-6, fine)
    GX, GY, GZ = np.meshgrid(xf, yf, zf, indexing="ij")
    print(f"  field grid {xg.size}x{yg.size}x{zg.size} -> cropped {xf.size}x{yf.size}x{zf.size}", flush=True)
    return (jnp.asarray(rgi((GX, GY, GZ)).astype(np.float32)),
            float(xf[0]), fine, float(yf[0]), fine, float(zf[0]), fine)


FH.load_field = load_field
FH.FIELD_PATH = FIELD

npz_root, out_prefix = sys.argv[1:3]
n_events = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
methods = sys.argv[4:] or FH.METHODS

det = detopt.detector.from_config(yaml.safe_load(open(DET_CFG)))
design = np.asarray(det.flatten_design(yaml.safe_load(open(DES_CFG))), np.float32)
FH.METHODS = methods

out = {}
for label in ANGLES_DEG:
    glob_pat = os.path.join(npz_root, label, "*.npz")
    if not __import__("glob").glob(glob_pat):
        continue
    print(f"=== {label} ({ANGLES_DEG[label]:.3f} deg) ===", flush=True)
    t0 = time.time()
    orig = FH._load
    FH._load = lambda nf, _g=glob_pat: (
        __import__("detopt.data.fairship_loader", fromlist=["load_fairship_digi"]).load_fairship_digi(
            None, data_glob=_g, columns=("truth", "reco", "digi_straw", "digi_invalid",
                                         "digi_event_index", "hits", "hit_track", "digi_tdc")))
    store, true9, reco9, reco_ok = FH.store(det, design, None, n_events, 512)
    FH._load = orig
    out[label] = dict(store=store, true9=true9, reco9=reco9, reco_ok=reco_ok)
    # Save after every angle so a kill cannot lose the completed ones.
    np.savez_compressed(
        f"{out_prefix}.npz",
        **{f"{lb}__{m}__{k}": v
           for lb, r in out.items() for m, d in r["store"].items() for k, v in d.items()},
        **{f"{lb}__true9": r["true9"] for lb, r in out.items()},
        **{f"{lb}__reco9": r["reco9"] for lb, r in out.items()},
        **{f"{lb}__reco_ok": r["reco_ok"] for lb, r in out.items()},
    )
    print(f"  saved {out_prefix}.npz ({len(out)} angles)", flush=True)
    print(f"  {time.time() - t0:.0f}s", flush=True)

np.savez_compressed(
    f"{out_prefix}.npz",
    **{f"{lab}__{m}__{k}": v
       for lab, r in out.items() for m, d in r["store"].items() for k, v in d.items()},
    **{f"{lab}__true9": r["true9"] for lab, r in out.items()},
    **{f"{lab}__reco_ok": r["reco_ok"] for lab, r in out.items()},
)
print(f"wrote {out_prefix}.npz")
