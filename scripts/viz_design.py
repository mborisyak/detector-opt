"""Draw the z-y projection of a stereo-straw detector design.

Reads a physical design from ``config/design/*.yaml`` ({stations, angle}), builds
the StereoStrawDetector with the geometry from ``config/bo.yaml`` (magnet, layer
extents, field), and draws the bending-plane (z-y) layout:

  * the spectrometer magnet as a grey box (z0 +/- magnet_half_cm, full y aperture)
  * each straw layer as a vertical line at its z (full +/- layer_height y span)
  * a transparent overlay of our simulated trajectories (a few events, one colour
    each) threaded through this design's geometry

Run headless:
    python scripts/viz_design.py [design.yaml | "all"] [data_dir] [out_dir]

With no args it renders every config/design/*.yaml to output/design/.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never block
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from detopt.detector.stereo_straw import StereoStrawDetector  # noqa: E402
from detopt.data.ship2numpy import load_ship2numpy_events  # noqa: E402
from detopt.utils.viz.straw import daughter_polylines  # noqa: E402

BO_CFG = "config/bo.yaml"


def build_detector(data_dir):
    """StereoStrawDetector from the geometry block of config/bo.yaml."""
    cfg = yaml.safe_load(open(BO_CFG))["detector"]["stereo_straw"]
    cfg = {**cfg, "data_dir": data_dir}
    return StereoStrawDetector(**cfg)


def load_design(path, det):
    """Read {stations, angle} -> flat physical design array [station_z(n), angle]."""
    d = yaml.safe_load(open(path))
    return np.asarray([*d["stations"], float(d["angle"])], np.float32)


def layer_z(det, design):
    """Per-layer z positions (n_layers,) for the bending-plane projection."""
    positions, _angles, _B = det._design_to_geometry(design[None, :])
    return positions[0]


def _event_dd(ev, e):
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
    }, n


def sample_trajectories(det, design, data_dir, n_events, rng):
    """Run our solver on a few events at this design -> list of per-particle (z, y) polylines."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        return []
    ev = load_ship2numpy_events(files[0])
    design_b = design[None, :]
    cand = [e for e in range(ev["n_events"]) if (int(ev["offsets"][e + 1]) - int(ev["offsets"][e])) > 0]
    rng.shuffle(cand)
    events = []
    for e in cand[:n_events]:
        dd, n = _event_dd(ev, e)
        _, _, lines3d = daughter_polylines(det, dd, design_b, rng)  # per-daughter (k,3) [x,y,z]
        lines = [t[:, [2, 1]] for t in lines3d if len(t) > 1]  # (z, y) polylines
        if lines:
            events.append(lines)
    return events


def draw(det, design, events, title, out_path):
    z = layer_z(det, design)
    half_y = det.layer_height
    zlo, zhi = det.layer_bounds

    fig, ax = plt.subplots(figsize=(12, 6))

    # spectrometer magnet: grey box spanning its z-width and the full y aperture
    z0, mh = det.z0, det.magnet_half_cm
    ax.axvspan(z0 - mh, z0 + mh, color="0.7", alpha=0.5, zorder=0)
    ax.text(z0, half_y * 0.92, "magnet", ha="center", va="top", color="0.35", fontsize=9)

    # straw layers: a vertical line per layer at its z
    for i, zi in enumerate(z):
        ax.plot([zi, zi], [-half_y, half_y], color="steelblue", lw=0.8, alpha=0.7, zorder=1, label="layer" if i == 0 else None)

    # transparent event overlay: our simulated trajectories, one colour per event
    cmap = plt.get_cmap("autumn")
    for k, lines in enumerate(events):
        c = cmap(k / max(len(events) - 1, 1))
        for j, ln in enumerate(lines):
            ax.plot(ln[:, 0], ln[:, 1], color=c, lw=0.7, alpha=0.15, zorder=2, label="events" if (k == 0 and j == 0) else None)

    ax.set_xlim(zlo, zhi)  # z-axis spans the full detector bounds (layer_bounds)
    ax.set_ylim(-half_y, half_y)
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main(target="all", data_dir="data/mc", out="output/design", n_events=200):
    os.makedirs(out, exist_ok=True)
    paths = sorted(glob.glob("config/design/*.yaml")) if target == "all" else [target]
    det = build_detector(data_dir)
    rng = np.random.default_rng(0)
    print(
        f"detector: n_stations={det.n_stations}, n_layers={det.n_layers}, "
        f"magnet z0={det.z0} +/-{det.magnet_half_cm} cm, y-aperture +/-{det.layer_height:.0f} cm, "
        f"z-bounds {det.layer_bounds}"
    )
    for p in paths:
        design = load_design(p, det)
        events = sample_trajectories(det, design, data_dir, n_events, rng)  # solved at THIS design
        name = os.path.splitext(os.path.basename(p))[0]
        title = f"{name}: z-y projection (angle={design[-1]:.4f} rad)"
        print(f"{name}: overlaying {len(events)} simulated events")
        draw(det, design, events, title, os.path.join(out, f"zy_{name}.png"))


if __name__ == "__main__":
    tgt = sys.argv[1] if len(sys.argv) > 1 else "all"
    dd = sys.argv[2] if len(sys.argv) > 2 else "data/mc"
    od = sys.argv[3] if len(sys.argv) > 3 else "output/design"
    main(tgt, dd, od)
