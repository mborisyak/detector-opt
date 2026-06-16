"""Simulate ship2numpy events through StrawDetector and check vs FairShip MC truth.

For a handful of events we (1) propagate the daughters through the straw solver,
(2) overlay the resulting trajectories on the MC-truth hit positions (`hits`
(x,y,z) in the npz), and (3) quantify the MC-hit-to-trajectory residual. If the
field/geometry model is faithful, the tracks thread the MC hits.

    python scripts/check_vs_mc.py [npz_path] [n_events]

Writes output/check_vs_mc.png and prints residual + multiplicity stats.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import yaml  # noqa: E402

from detopt.detector.free_straw import FreeStrawDetector, free_design_array  # noqa: E402


def _nominal_design(det, cfg_path="config/detector/nominal_design.yaml"):
    """Read the nominal design from config (not detector state) and build the array."""
    nd = yaml.safe_load(open(cfg_path))["nominal_design"]
    return free_design_array(
        nd["station_z"],
        n_layers_per_view=det.n_layers_per_view,
        view_angles=nd["view_angles"],
        view_z_gap=nd["view_z_gap"],
        layer_z_gap=nd["layer_z_gap"],
        B=nd["B"],
    )


NPZ = sys.argv[1] if len(sys.argv) > 1 else "ship2numpy.npz"
N_SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def event_daughters(events, e):
    """Single-event daughter_data dict (flat slice + offsets) for _run_solver."""
    o = events["offsets"]
    s = slice(int(o[e]), int(o[e + 1]))
    n = int(o[e + 1]) - int(o[e])
    return {
        "masses": events["masses"][s],
        "charges": events["charges"][s],
        "positions": events["positions"][s],
        "momenta": events["momenta"][s],
        "times": events["times"][s],
    }


def run_one(det, dd, design, rng):
    """Solve a single hand-built event dict and return (X, mask, trajectories)."""
    ie = det._make_input_events(dd)
    n = len(dd["masses"])
    traj = np.zeros((1, det.max_particles, det.n_t, 3), dtype=np.float32)
    X, mask, _ = det._run_solver(np.array([[0, n]], np.int32), design, rng, input_events=ie, trajectories=traj)
    return X, mask, traj


def event_nparticles(events, e):
    o = events["offsets"]
    return int(o[e + 1]) - int(o[e])


def _frame_corners(det, z, angle):
    """Lab-frame corners of a layer's parallelogram (matches viz/straw._frame_corners):
    vertical sides at x = +/-width; top/bottom edges sheared by the stereo tilt."""
    w, h = det.layer_width, det.layer_height
    s = w * np.tan(angle)
    x = np.array([-w, -w, w, w], dtype=np.float32)
    y = np.array([-h - s, h - s, h + s, -h + s], dtype=np.float32)
    return x, y, float(z)


def draw_stations(ax, det, design):
    """Overlay the detector's station/layer planes: one wireframe parallelogram per
    layer, drawn in the plot's (z, x, y) axis order so tracks/hits sit in context."""
    positions, angles = det._design_to_geometry(design)[:2]
    positions, angles = positions[0], angles[0]
    for k in range(det.n_layers):
        x, y, z = _frame_corners(det, float(positions[k]), float(angles[k]))
        xl, yl = np.append(x, x[0]), np.append(y, y[0])  # close the loop
        ax.plot(np.full_like(xl, z), xl, yl, color="0.6", lw=0.5, alpha=0.4)


def main():
    raw = np.load(NPZ, allow_pickle=True)
    hits_xyz = np.asarray(raw["hits"], np.float32)
    hit_ev = np.asarray(raw["hit_event_index"], np.int64)
    hit_trk = np.asarray(raw["hit_track"], np.int64)
    real = hit_trk >= 0  # exclude hit_track==-2 (untracked shower secondaries)

    # Fine max_dt so the trajectory polyline is densely sampled: the MC-hit ->
    # nearest-trajectory-vertex residual is otherwise dominated by the ~30*max_dt
    # cm sample spacing, not by physics.
    det = FreeStrawDetector(data_dir=NPZ, max_dt=0.1, max_time=300.0, max_particles=25)
    events = det._events
    design = _nominal_design(det)[None, :]
    rng = np.random.default_rng(0)

    # Pick events that actually have both daughters and MC hits.
    have_hits = set(np.unique(hit_ev).tolist())
    cand = [e for e in range(det.n_events) if e in have_hits and event_nparticles(events, e) > 0]
    show = cand[:N_SHOW]

    # ---- per-event residual + multiplicity over a larger sample -----------------
    sim_mult, mc_mult, residuals = [], [], []
    for e in cand[:200]:
        _, mask, traj = run_one(det, event_daughters(events, e), design, rng)
        sim_mult.append(int(mask.sum()))
        mc = hits_xyz[(hit_ev == e) & real]  # real tracks only
        mc_mult.append(len(mc))
        pts = traj.reshape(-1, 3)
        pts = pts[np.abs(pts).sum(1) > 0]  # filled trajectory samples
        if len(pts) and len(mc):
            # nearest trajectory point to each MC hit (cm)
            for h in mc[:: max(1, len(mc) // 50)]:
                residuals.append(float(np.min(np.linalg.norm(pts - h, axis=1))))

    sim_mult, mc_mult, residuals = map(np.asarray, (sim_mult, mc_mult, residuals))
    print(f"events checked: {len(sim_mult)}")
    print(f"hits/event  sim: mean {sim_mult.mean():.1f}  MC: mean {mc_mult.mean():.1f}")
    print(
        f"MC-hit -> nearest-trajectory residual (cm): "
        f"median {np.median(residuals):.2f}  p90 {np.percentile(residuals, 90):.2f}  mean {residuals.mean():.2f}"
    )

    # ---- 3D overlays ------------------------------------------------------------
    # Axis limits = the detector volume: z over the layer span, x over the straw
    # half-length, y over the straw half-height (so strays don't blow out the view).
    zl = det._design_to_geometry(design)[0][0]
    zlo, zhi = float(zl.min()), float(zl.max())
    zmar = 0.05 * (zhi - zlo)
    ncol = 3
    nrow = int(np.ceil(len(show) / ncol))
    fig = plt.figure(figsize=(6 * ncol, 5 * nrow))
    for i, e in enumerate(show):
        _, mask, traj = run_one(det, event_daughters(events, e), design, rng)
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        draw_stations(ax, det, design)  # detector station/layer frames for context
        npart = event_nparticles(events, e)
        for p in range(min(npart, traj.shape[1])):
            t = traj[0, p]
            t = t[np.abs(t).sum(1) > 0]
            if len(t):
                ax.plot(t[:, 2], t[:, 0], t[:, 1], lw=1.0, alpha=0.8)
        mc = hits_xyz[(hit_ev == e) & real]
        if len(mc):
            ax.scatter(mc[:, 2], mc[:, 0], mc[:, 1], s=4, c="k", alpha=0.4, label="MC hits")
        ax.set_title(f"event {e}: {npart} daughters, {len(mc)} MC hits")
        ax.set_xlabel("z (cm)")
        ax.set_ylabel("x (cm)")
        ax.set_zlabel("y (cm)")
        ax.set_xlim(zlo - zmar, zhi + zmar)
        ax.set_ylim(-det.layer_width, det.layer_width)
        ax.set_zlim(-det.layer_height, det.layer_height)
    fig.tight_layout()
    out = Path("output/check_vs_mc.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
