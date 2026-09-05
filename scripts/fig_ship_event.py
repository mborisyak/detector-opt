"""Reproduce the SHiP straw-tracker event display used as a paper figure.

Renders one event of the FairShip-derived `straw` geometry at the nominal design: fired straws in
red, the two HNL daughter trajectories in colour, and the boundary of every one of the 32 layers as
a wireframe. The defaults below are the accepted figure -- event 566, a 25 degree swing off the
side-on view, orthographic so proportions can be read off the image -- so a bare

    python scripts/fig_ship_event.py

regenerates it byte-for-byte given the same MC file.

VTK renders into a fixed canvas and leaves whatever white space is left over, and it anchors the
title and the orientation gizmo to canvas corners rather than to the scene. Both are therefore
disabled here and the white border is trimmed afterwards, which is why the output size is not the
canvas size. Run headless: the module falls back to xvfb, then to an OSMesa/EGL VTK build.
"""

import os

import numpy as np
import yaml

import pyvista as pv

import detopt
from detopt.utils.viz import straw as viz3d

CFG = "config/detector/straw.yaml"
NOMINAL = "config/detector/nominal_design.yaml"


def build_detector(data_path):
    """FreeStrawDetector + nominal physical design array from the FairShip-derived config."""
    cfg = yaml.safe_load(open(CFG))
    straw_cfg = dict(cfg["straw"])
    straw_cfg["data_dir"] = data_path  # the config ships `data_dir: null`, so override rather than pass twice
    det = detopt.detector.FreeStrawDetector(**straw_cfg)
    nd = yaml.safe_load(open(NOMINAL))["nominal_design"]
    design = detopt.detector.free_design_array(
        nd["station_z"],
        n_layers_per_view=det.n_layers_per_view,
        view_angles=nd["view_angles"],
        view_z_gap=nd["view_z_gap"],
        layer_z_gap=nd["layer_z_gap"],
        B=nd["B"],
    )
    return det, design


def event_daughters(events, event):
    """Single-event daughter pool (flat slice + offsets) for the solver."""
    offsets = events["offsets"]
    span = slice(int(offsets[event]), int(offsets[event + 1]))
    n_daughters = int(offsets[event + 1]) - int(offsets[event])
    pool = {key: events[key][span] for key in ("masses", "charges", "positions", "momenta", "times")}
    pool["offsets"] = np.array([0, n_daughters], dtype=np.int32)
    return pool, n_daughters


def trim_white_border(path, pad):
    """Crop the uniform white margin VTK leaves around the scene, keeping `pad` pixels."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image)
    ink = (pixels < 250).any(axis=2)
    rows, columns = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(columns) == 0:
        return image.size
    top, bottom = max(0, rows[0] - pad), min(pixels.shape[0], rows[-1] + 1 + pad)
    left, right = max(0, columns[0] - pad), min(pixels.shape[1], columns[-1] + 1 + pad)
    cropped = image.crop((left, top, right, bottom))
    cropped.save(path)
    return cropped.size


def main(
    data_path="data/mc/sim_1000-0-V2023.npz",
    out="output/figures/fig_ship_event.png",
    event=566,
    seed=12345,
    azimuth=-25.0,
    elevation=8.0,
    zoom=1.0,
    window=(3200, 2240),
    pad=12,
    show_grid=False,
    frame_width=5.0,
    fill_opacity=0.15,
    frame_opacity=1.0,
    track_scale=2.0,
    grid_font_size=27,
    all_straws=False,
    all_straw_opacity=0.12,
    frame_granularity="view",
    bank=0.0,
    interactive=False,
):
    # Interactive opens a live window on $DISPLAY for inspecting the geometry by hand; otherwise
    # render off-screen to PNG.
    pv.OFF_SCREEN = not interactive
    if not interactive:
        try:
            pv.start_xvfb()
        except Exception:
            pass  # no xvfb -> rely on an OSMesa/EGL VTK build, else run with a display
    pv.global_theme.window_size = list(window) if not interactive else [1400, 950]
    try:
        pv.global_theme.anti_aliasing = "ssaa"
    except Exception:
        pass
    if not interactive:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    det, design = build_detector(data_path)
    pool, n_daughters = event_daughters(det._events, event)
    # The solver re-simulates secondaries on every call, so the seed is what makes the hit count
    # (and therefore the figure) reproducible rather than merely similar.
    X, mask, lines = viz3d.daughter_polylines(det, pool, design[None, :], np.random.default_rng(seed + event))
    n_fired = int(mask[0].sum())

    viz3d.show_event(
        det, design, X[0], mask[0], lines,
        n_daughters=n_daughters, off_screen=not interactive, screenshot=None if interactive else out,
        title=None,  # the caption carries this; a VTK title is anchored to the canvas corner
        # VTK line widths are in PIXELS, so they do not scale with the canvas: a hairline that
        # looks right at 1000 px is invisible at 3200 px and vanishes again when the figure is
        # downsampled into a text-width slot. Keep this proportional to `window`.
        # tab10 entries 1-3, repeating on the fourth view so the two Y-views (angle 0) match and
        # the +/- stereo pair are distinct.
        view_colors=("#1f77b4", "#ff7f0e", "#2ca02c"),
        fill_opacity=fill_opacity, frame_granularity=frame_granularity, bank=bank,
        all_straws=all_straws, all_straw_opacity=all_straw_opacity,
        frame_opacity=frame_opacity, frame_width=frame_width, track_scale=track_scale,
        azimuth=azimuth, elevation=elevation, zoom=zoom,
        parallel_projection=True, show_axes=False, show_grid=show_grid, grid_font_size=grid_font_size,
    )
    if interactive:
        print(f"event {event}: {n_daughters} daughters, {n_fired} fired straws (window closed)")
        return
    size = trim_white_border(out, pad)
    print(f"event {event}: {n_daughters} daughters, {n_fired} fired straws -> {out} {size[0]}x{size[1]}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render the SHiP straw-tracker event display used as a paper figure.")
    ap.add_argument("data", nargs="?", default="data/mc/sim_1000-0-V2023.npz")
    ap.add_argument("out", nargs="?", default="output/figures/fig_ship_event.png")
    ap.add_argument("--event", type=int, default=566)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--azimuth", type=float, default=-25.0,
        help="degrees off the side-on view; z renders at cos(azimuth). Negative swings the near "
             "corner to the upstream end, so station 1 faces the viewer",
    )
    ap.add_argument("--elevation", type=float, default=8.0)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--grid", action="store_true", help="draw the labelled box and ticks (illegible below ~0.9 textwidth)")
    ap.add_argument(
        "--frame-width", type=float, default=5.0,
        help="layer-boundary line width in PIXELS; scale it with the canvas or the lines wash out when downsampled",
    )
    ap.add_argument("--fill-opacity", type=float, default=0.15,
                    help="per-layer translucent surface; 8 layers per station accumulate, so keep it low")
    ap.add_argument("--frames", choices=("layer", "view", "station"), default="view",
                    help="one outline per layer (32), per stereo view (16) or per station (4)")
    ap.add_argument("--frame-opacity", type=float, default=1.0,
                    help="layer outline opacity; 0 leaves only the translucent surfaces")
    ap.add_argument("--straw-opacity", type=float, default=0.12, help="opacity of the inactive grey straws")
    ap.add_argument("--all-straws", action="store_true", help="also draw the inactive straws, faint grey")
    ap.add_argument("--track-scale", type=float, default=2.0,
                    help="daughter trajectory tube radius, in units of the straw radius (1 cm)")
    ap.add_argument("--grid-font-size", type=int, default=27, help="axis tick/label size, in canvas px")
    ap.add_argument("--bank", type=float, default=0.0, help="orbit the viewpoint about the beam axis, degrees")
    ap.add_argument("-i", "--interactive", action="store_true", help="open a live window on $DISPLAY instead of writing a PNG")
    args = ap.parse_args()
    main(
        args.data, args.out, event=args.event, seed=args.seed, azimuth=args.azimuth,
        elevation=args.elevation, zoom=args.zoom, show_grid=args.grid, frame_width=args.frame_width,
        interactive=args.interactive, fill_opacity=args.fill_opacity, frame_opacity=args.frame_opacity, track_scale=args.track_scale, grid_font_size=args.grid_font_size,
        all_straws=args.all_straws, all_straw_opacity=args.straw_opacity, frame_granularity=args.frames, bank=args.bank,
    )
