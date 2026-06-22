"""Overlay our simulated trajectories on the FairShip MC truth hits.

Builds the straw detector straight from ``config/detector/straw.yaml`` (whose
geometry/field are taken from FairShip -- SST strawtubes_config.yaml +
MainSpectrometerField.root), runs our solver on each MC event's boundary-crossing
daughters, and draws our trajectories (lines) over the MC-truth ``hits`` (points)
for a few events. If the geometry/field model is faithful the tracks thread the
MC hits.

Run headless:  ``python scripts/viz.py [data/mc/sim_1000-0-V2023.npz] [out_dir]``
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never block
import matplotlib.pyplot as plt
import numpy as np
import yaml

import detopt
from detopt.utils.viz.straw import daughter_polylines

CFG = "config/detector/straw.yaml"


def build_detector(data_path):
    """FreeStrawDetector from the (FairShip-derived) config + the nominal design array."""
    cfg = yaml.safe_load(open(CFG))
    det = detopt.detector.FreeStrawDetector(**cfg["straw"], data_dir=data_path)
    nd = yaml.safe_load(open("config/detector/nominal_design.yaml"))["nominal_design"]
    design = detopt.detector.free_design_array(
        nd["station_z"],
        n_layers_per_view=det.n_layers_per_view,
        view_angles=nd["view_angles"],
        view_z_gap=nd["view_z_gap"],
        layer_z_gap=nd["layer_z_gap"],
        B=nd["B"],
    )
    return det, design


def event_daughters(events, e):
    """Single-event daughter_data dict (flat slice + offsets) for ``_run_solver``."""
    o = events["offsets"]
    s = slice(int(o[e]), int(o[e + 1]))
    n = int(o[e + 1]) - int(o[e])
    return {
        "masses": events["masses"][s],
        "charges": events["charges"][s],
        "positions": events["positions"][s],
        "momenta": events["momenta"][s],
        "times": events["times"][s],
        "offsets": np.array([0, n], dtype=np.int32),
    }, n


def _overlay(ax, lines, mc, proj):
    """Draw our daughter polylines + MC hit points in projection ``proj``
    ('zy' = bending plane, 'zx' = non-bending). ``lines`` = per-daughter (k,3) [x,y,z]."""
    a, b = (2, 1) if proj == "zy" else (2, 0)  # (z, y) or (z, x)
    for t in lines:
        if len(t) > 1:
            ax.plot(t[:, a], t[:, b], lw=1.0, alpha=0.8, zorder=2)
    if len(mc):
        ax.scatter(mc[:, a], mc[:, b], s=6, c="k", alpha=0.45, zorder=3, label="MC hits")
    ax.set_xlabel("z [cm]")
    ax.set_ylabel(("y" if proj == "zy" else "x") + " [cm]")


def main(data_path="data/mc/sim_1000-0-V2023.npz", out="output/viz", n_show=6):
    os.makedirs(out, exist_ok=True)
    det, design = build_detector(data_path)
    events = det._events
    design_b = design[None, :]
    rng = np.random.default_rng(0)
    # z-axis (the x of these 2D plots) limited to the detector span, with a small margin.
    z_layers = np.asarray(design[: det.n_layers], dtype=np.float32)
    z_margin = 0.05 * (z_layers.max() - z_layers.min())
    zlim = (z_layers.min() - z_margin, z_layers.max() + z_margin)

    raw = np.load(data_path, allow_pickle=True)
    hits = np.asarray(raw["hits"], np.float32)
    he = np.asarray(raw["hit_event_index"], np.int64)
    real = np.asarray(raw["hit_track"], np.int64) >= 0  # exclude untracked shower secondaries
    have_hits = set(np.unique(he[real]).tolist())

    # events that have both daughters and MC hits
    cand = [
        e for e in range(det.n_events) if e in have_hits and (int(events["offsets"][e + 1]) - int(events["offsets"][e])) > 0
    ]
    show = cand[:n_show]
    print(
        f"detector: {det.n_events} events, boundary_z={det.boundary_z} cm, n_straws={det.n_straws}, "
        f"B_peak={design[-1]:.3f} T, B_sigma={det.B_sigma} cm"
    )
    print(f"overlaying {len(show)} events (our trajectories + MC hits)")

    ncol = 2
    nrow = int(np.ceil(len(show) / ncol))
    for proj, tag in [("zy", "bending plane z-y"), ("zx", "non-bending z-x")]:
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4 * nrow), squeeze=False)
        for i, e in enumerate(show):
            dd, n = event_daughters(events, e)
            _, mask, lines = daughter_polylines(det, dd, design_b, rng)
            mc = hits[(he == e) & real]
            ax = axes[i // ncol][i % ncol]
            _overlay(ax, lines, mc, proj)
            ax.set_xlim(*zlim)  # restrict z-axis to the detector span (ignore stray MC hits)
            # vertical axis spans the full detector extent (height for z-y, width for z-x)
            half = det.layer_height if proj == "zy" else det.layer_width
            ax.set_ylim(-half, half)
            ax.set_title(f"event {e}: {n} daughters, {len(mc)} MC hits, {int(mask[0].sum())} sim hits")
            if i == 0:
                ax.legend(loc="best", fontsize=8)
        for j in range(len(show), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"Our trajectories vs FairShip MC hits ({tag})")
        fig.tight_layout()
        path = os.path.join(out, f"viz_overlay_{proj}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  wrote {path}")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "data/mc/sim_1000-0-V2023.npz"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/viz"
    main(data, out_dir)
