"""Distributional check: simulated straw hits vs FairShip MC truth.

With the magnet matched, compare aggregate distributions (not just per-event
alignment) between the StrawDetector simulation and the MC-truth `hits` in
ship2numpy.npz:

  (1) hits / event           (2) hit z-occupancy (station pattern)
  (3) per-z transverse miss  (4) hit transverse-y spread

    python scripts/check_distributions.py [npz_path] [n_events]

Writes output/check_distributions.png and prints summary stats.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import yaml  # noqa: E402

from detopt.detector.free_straw import FreeStrawDetector, free_design_array  # noqa: E402


def _nominal_design(det, cfg_path="config/detector/straw.yaml"):
    nd = yaml.safe_load(open(cfg_path))["nominal_design"]
    return free_design_array(
        nd["station_z"], n_layers_per_view=det.n_layers_per_view,
        view_angles=nd["view_angles"], view_z_gap=nd["view_z_gap"],
        layer_z_gap=nd["layer_z_gap"], B=nd["B"],
    )

NPZ = sys.argv[1] if len(sys.argv) > 1 else "ship2numpy.npz"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400


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


def at_z(tp, z):
    t = tp[np.abs(tp).sum(1) > 0]
    if len(t) < 2:
        return None
    t = t[np.argsort(t[:, 2])]
    if z < t[0, 2] or z > t[-1, 2]:
        return None
    return np.array([np.interp(z, t[:, 2], t[:, 0]), np.interp(z, t[:, 2], t[:, 1])])


def main():
    raw = np.load(NPZ, allow_pickle=True)
    hits = np.asarray(raw["hits"], np.float32)
    he = np.asarray(raw["hit_event_index"], np.int64)
    real = np.asarray(raw["hit_track"], np.int64) >= 0

    det = FreeStrawDetector(data_dir=NPZ, max_dt=0.1, max_time=300.0, max_particles=25)
    ev = det._events
    design = _nominal_design(det)[None, :]
    layers_z = det._design_to_geometry(design)[0][0]  # (n_layers,) z of each global layer
    per_station = det.n_views_per_station * det.n_layers_per_view
    rng = np.random.default_rng(0)

    sim_mult, mc_mult, sim_z, mc_z, sim_y, mc_y, miss = [], [], [], [], [], [], []
    cand = [e for e in range(det.n_events) if event_npart(ev, e) > 0 and np.any((he == e) & real)][:N]
    for e in cand:
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
            d = [np.hypot(*(at_z(traj[0, p], hh[2]) - hh[:2])) for p in range(min(npp, traj.shape[1])) if at_z(traj[0, p], hh[2]) is not None]
            if d:
                miss.append(min(d))

    sim_mult, mc_mult, miss = map(np.asarray, (sim_mult, mc_mult, miss))
    print(f"events: {len(cand)}")
    print(f"hits/event   sim mean {sim_mult.mean():.1f} median {np.median(sim_mult):.0f} | MC mean {np.mean(mc_mult):.1f} median {np.median(mc_mult):.0f}")
    print(f"transverse miss (cm): median {np.median(miss):.2f}  p90 {np.percentile(miss,90):.2f}")
    print(f"hit z (cm):  sim [{min(sim_z):.0f},{max(sim_z):.0f}] | MC [{min(mc_z):.0f},{max(mc_z):.0f}]")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].hist(mc_mult, bins=30, alpha=0.5, label="MC", color="k")
    ax[0, 0].hist(sim_mult, bins=30, alpha=0.5, label="sim", color="C0")
    ax[0, 0].set(title="hits / event", xlabel="n hits"); ax[0, 0].legend()
    ax[0, 1].hist(mc_z, bins=80, alpha=0.5, label="MC", color="k", density=True)
    ax[0, 1].hist(sim_z, bins=80, alpha=0.5, label="sim", color="C0", density=True)
    ax[0, 1].set(title="hit z occupancy", xlabel="z (cm)"); ax[0, 1].legend()
    ax[1, 0].hist(miss, bins=np.linspace(0, 30, 60), color="C2")
    ax[1, 0].set(title="per-z transverse miss (sim track -> MC hit)", xlabel="cm", yscale="log")
    ax[1, 1].hist(mc_y, bins=80, alpha=0.5, label="MC", color="k", density=True)
    ax[1, 1].hist(sim_y, bins=80, alpha=0.5, label="sim", color="C0", density=True)
    ax[1, 1].set(title="hit transverse y", xlabel="y (cm)"); ax[1, 1].legend()
    fig.tight_layout()
    out = Path("output/check_distributions.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
