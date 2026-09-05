"""Faithful 3D pyvista view of the straw detector for a few events.

Builds the detector from ``config/detector/straw.yaml`` (same as ``scripts/viz.py``),
solves a handful of MC events with the trajectory buffer on, and writes one 3D PNG
per event: daughter/secondary trajectories threading the highlighted fired straws,
with a wireframe frame parallelogram per layer (unfired straws hidden).

Run headless:  ``python scripts/viz3d.py [data/mc/sim_1000-0-V2023.npz] [out_dir]``
"""

import os

import numpy as np
import yaml

import pyvista as pv

import detopt
from detopt.utils.viz import straw as viz3d

CFG = "config/detector/straw.yaml"


def build_detector(data_path):
    """FreeStrawDetector + nominal physical design array from the FairShip-derived config."""
    cfg = yaml.safe_load(open(CFG))
    straw_cfg = dict(cfg["straw"])
    straw_cfg["data_dir"] = data_path  # the config ships `data_dir: null`, so override rather than pass twice
    det = detopt.detector.FreeStrawDetector(**straw_cfg)
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
    """Single-event daughter pool (flat slice + offsets) for ``_run_solver``."""
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


def main(data_path="data/mc/sim_1000-0-V2023.npz", out="output/viz3d", n_show=3, interactive=False):
    # Interactive: open a live window per event. Otherwise render off-screen to PNG
    # (with a virtual framebuffer if one is available).
    if interactive:
        pv.OFF_SCREEN = False
    else:
        pv.OFF_SCREEN = True
        try:
            pv.start_xvfb()
        except Exception:
            pass  # no xvfb -> rely on an OSMesa/EGL VTK build, else run with a display
        os.makedirs(out, exist_ok=True)
    det, design = build_detector(data_path)
    events = det._events
    design_b = design[None, :]
    rng = np.random.default_rng(0)

    cand = [e for e in range(det.n_events) if int(events["offsets"][e + 1]) - int(events["offsets"][e]) > 0]
    show = cand[:n_show]
    print(f"detector: {det.n_events} events, n_layers={det.n_layers}, n_straws={det.n_straws}, B_peak={design[-1]:.3f} T")

    for e in show:
        dd, n = event_daughters(events, e)
        X, mask, lines = viz3d.daughter_polylines(det, dd, design_b, rng)
        title = f"event {e}: {n} HNL daughters, {int(mask[0].sum())} fired straws"
        path = None if interactive else os.path.join(out, f"viz3d_event_{e}.png")
        viz3d.show_event(
            det,
            design,
            X[0],
            mask[0],
            lines,
            n_daughters=n,
            off_screen=not interactive,
            screenshot=path,
            title=title,
        )
        print(f"  shown event {e}" if interactive else f"  wrote {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Faithful 3D pyvista view of the straw detector.")
    ap.add_argument("data", nargs="?", default="data/mc/sim_1000-0-V2023.npz")
    ap.add_argument("out", nargs="?", default="output/viz3d")
    ap.add_argument("--show", default=3, type=int)
    ap.add_argument("-i", "--interactive", action="store_true", help="open a live window per event instead of saving PNGs")
    args = ap.parse_args()
    main(args.data, args.out, n_show=args.show, interactive=args.interactive)
