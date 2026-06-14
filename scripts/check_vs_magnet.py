"""Validate the StrawDetector simulation against a single-magnet subset of data/mc.

data/mc mixes magnets (filename sim_<HNLmass_MeV>-<seed>-<magnet>.npz); only the
files whose magnet matches our configured field can be checked positionally. For
the chosen magnet we compare, against the FairShip MC truth (`hits`):

  * per-z transverse miss (sim track interpolated to each MC hit's z)
  * hits / event  (sim vs MC real tracks)
  * hit z-occupancy and transverse-y distributions
  * a 3D overlay of trajectories on MC hits for a few events

    python scripts/check_vs_magnet.py [magnet] [data_dir]

Writes output/check_<magnet>.png and prints a summary.
"""

import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import yaml  # noqa: E402

from detopt.detector.stereo_straw import StereoStrawDetector, stereo_design_array  # noqa: E402


def _nominal_design(cfg_path="config/bo.yaml"):
    nd = yaml.safe_load(open(cfg_path))["nominal_design"]
    return stereo_design_array(nd["station_z"], nd["stereo_angle"], nd["B"])
from detopt.data.ship2numpy import load_ship2numpy_events  # noqa: E402

MAGNET = sys.argv[1] if len(sys.argv) > 1 else "V13_3500"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/mc"


def event_dd(ev, e):
    """Single-event daughter_data (flat slice + offsets) for _run_solver."""
    o = ev["offsets"]
    s = slice(int(o[e]), int(o[e + 1]))
    n = int(o[e + 1]) - int(o[e])
    return {
        "masses": ev["masses"][s],
        "charges": ev["charges"][s],
        "positions": ev["positions"][s],
        "momenta": ev["momenta"][s],
        "times": ev["times"][s],
        "offsets": np.array([0, n], dtype=np.int32),
    }


def event_npart(ev, e):
    o = ev["offsets"]
    return int(o[e + 1]) - int(o[e])


def track_at_z(tp, z):
    t = tp[np.abs(tp).sum(1) > 0]
    if len(t) < 2:
        return None
    t = t[np.argsort(t[:, 2])]
    if z < t[0, 2] or z > t[-1, 2]:
        return None
    return np.array([np.interp(z, t[:, 2], t[:, 0]), np.interp(z, t[:, 2], t[:, 1])])


def main():
    files = sorted(f for f in glob.glob(f"{DATA}/*.npz") if MAGNET in Path(f).name)
    if not files:
        raise SystemExit(f"no files matching magnet {MAGNET!r} in {DATA}")
    det = StereoStrawDetector(max_dt=0.3, max_time=300.0, max_particles=25)
    design = _nominal_design()[None, :]
    layers_z = det._design_to_geometry(design)[0][0]
    per_station = det.n_views_per_station * det.n_layers_per_view
    rng = np.random.default_rng(0)
    print(f"magnet {MAGNET}: {len(files)} files | field max_B={det.max_B} B_sigma={det.B_sigma}")

    zs, miss, sim_mult, mc_mult, sim_z, mc_z, sim_y, mc_y = [], [], [], [], [], [], [], []
    overlay = []  # (event daughter dict, mc hits) for the 3D panel, from the first file
    for fi, f in enumerate(files):
        raw = np.load(f, allow_pickle=True)
        hits = np.asarray(raw["hits"], np.float32)
        he = np.asarray(raw["hit_event_index"], np.int64)
        real = np.asarray(raw["hit_track"], np.int64) >= 0
        ev = load_ship2numpy_events(f, det.max_particles)
        for e in range(ev["n_events"]):
            if event_npart(ev, e) == 0 or not np.any((he == e) & real):
                continue
            dd = event_dd(ev, e)
            X, mask, traj = det._run_solver(dd, design, rng)
            h = X[0, mask[0].astype(bool)]
            sim_mult.append(len(h))
            for row in h:
                k = int(row[0]) * per_station + int(row[1]) * det.n_layers_per_view + int(row[2])
                sim_z.append(float(layers_z[k]))
                yoff = (0.5 if int(row[2]) & 1 else -0.5) * det.layer_y_offset
                sim_y.append((row[3] + 0.5) * det.straw_pitch - det.layer_height + yoff)
            mc = hits[(he == e) & real]
            mc_mult.append(len(mc))
            mc_z.extend(mc[:, 2].tolist())
            mc_y.extend(mc[:, 1].tolist())
            npp = event_npart(ev, e)
            for hh in mc:
                ds = [np.hypot(*(track_at_z(traj[0, p], hh[2]) - hh[:2])) for p in range(min(npp, traj.shape[1])) if track_at_z(traj[0, p], hh[2]) is not None]
                if ds:
                    zs.append(hh[2])
                    miss.append(min(ds))
            if fi == 0 and len(overlay) < 6:
                overlay.append((dd, mc))

    zs, miss, sim_mult, mc_mult = map(np.asarray, (zs, miss, sim_mult, mc_mult))
    print(f"events {len(sim_mult)}  hits sim {sim_mult.sum():,} mc {mc_mult.sum():,}")
    print(f"hits/event  sim {sim_mult.mean():.1f}  MC {mc_mult.mean():.1f}  (ratio {sim_mult.mean()/max(mc_mult.mean(),1):.2f})")
    print("transverse miss (cm) by z:")
    for lo, hi in [(8312, 8600), (8600, 8900), (8900, 9200), (9200, 9527)]:
        m = miss[(zs >= lo) & (zs < hi)]
        if len(m):
            print(f"  z[{lo},{hi}] ({(lo-8312)/100:.1f}-{(hi-8312)/100:.1f}m): median {np.median(m):.2f}  p90 {np.percentile(m,90):.1f}  N={len(m)}")
    print(f"OVERALL miss median {np.median(miss):.2f} cm  p90 {np.percentile(miss,90):.1f}")

    # ---- figure: distributions (top row) + 3D overlays (bottom) ----
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(2, 3, 1)
    ax.hist(mc_mult, bins=30, alpha=0.5, color="k", label="MC"); ax.hist(sim_mult, bins=30, alpha=0.5, color="C0", label="sim")
    ax.set(title=f"{MAGNET}: hits/event", xlabel="n hits"); ax.legend()
    ax = fig.add_subplot(2, 3, 2)
    ax.hist(mc_z, bins=80, density=True, alpha=0.5, color="k", label="MC"); ax.hist(sim_z, bins=80, density=True, alpha=0.5, color="C0", label="sim")
    ax.set(title="hit z occupancy", xlabel="z (cm)"); ax.legend()
    ax = fig.add_subplot(2, 3, 3)
    ax.hist(miss, bins=np.linspace(0, 20, 60), color="C2"); ax.set(title="per-z transverse miss", xlabel="cm", yscale="log")
    for i, (dd, mc) in enumerate(overlay[:3]):
        _, mask, traj = det._run_solver(dd, design, rng)
        a = fig.add_subplot(2, 3, 4 + i, projection="3d")
        for p in range(min(int(dd["offsets"][1]), traj.shape[1])):
            t = traj[0, p]; t = t[np.abs(t).sum(1) > 0]
            if len(t):
                a.plot(t[:, 2], t[:, 0], t[:, 1], lw=1.0)
        if len(mc):
            a.scatter(mc[:, 2], mc[:, 0], mc[:, 1], s=4, c="k", alpha=0.4)
        a.set(title=f"event {i}: {len(mc)} MC hits", xlabel="z", ylabel="x", zlabel="y")
    fig.tight_layout()
    out = Path(f"output/check_{MAGNET}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
