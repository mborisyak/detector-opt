"""3D visualisation of the :class:`~detopt.detector.debug.DebugDetector`.

Renders one event with pyvista: the four stations (each a stack of tilted
views), the magnet and decay volume, the fired straws, the decay vertex, and
the two daughter trajectories. Consumes the dict from
:meth:`DebugDetector.sample_events` so all geometry is read straight off the
detector and the physical design that produced the event.

Example
-------
    import detopt
    from detopt.utils.viz import debug as viz_debug

    det = detopt.detector.DebugDetector()
    sample = det.sample_events(0, det.get_current_design_array())
    viz_debug.show(det, sample, off_screen=True)
"""

import numpy as np
import pyvista as pv

__all__ = ["show"]

# Daughter colours (momenta order is [muon, pion]; see DebugDetector.generate_events).
_DAUGHTER_COLORS = ("red", "dodgerblue")
_DAUGHTER_NAMES = ("muon", "pion")
# Per-view straw colours.
_VIEW_PALETTE = ("gold", "limegreen", "magenta", "cyan", "orange", "white")


def _view_corners(w, h, tilt, z):
    """Parallelogram corners of a view in the plane ``z`` (sides parallel to y).

    Straws run along x; the lattice is sheared by ``tilt`` over the half-width so
    the straw coordinate is ``c = y - (tilt/w)*x``. The view spans ``c in [-h, h]``,
    ``x in [-w, w]``.
    """
    return np.array(
        [
            [-w, -h - tilt, z],
            [-w, h - tilt, z],
            [w, h + tilt, z],
            [w, -h + tilt, z],
        ],
        dtype=np.float32,
    )


def _straw_line(w, tilt, c, z):
    """End points of the straw whose centre coordinate is ``c`` (line along x)."""
    return np.array([[-w, c - tilt, z], [w, c + tilt, z]], dtype=np.float32)


def _box(plotter, z0, z1, w, h, color, opacity):
    mesh = pv.Box(bounds=(-w, w, -h, h, z0, z1))
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        style="surface",
        show_edges=True,
        edge_color=color,
        line_width=1.0,
    )


def show(
    detector,
    sample,
    event=0,
    off_screen=True,
    screenshot="debug_detector.png",
    max_straws_drawn=16,
    view_z_spread=2.0,
):
    """Render ``sample[... event]`` and (optionally) save a screenshot.

    Parameters
    ----------
    detector : DebugDetector -- supplies the static geometry constants.
    sample : dict from ``detector.sample_events`` (batched; one row per design).
    event : which batch row to draw.
    off_screen : render without opening a window (writes ``screenshot``).
    max_straws_drawn : context straws drawn per view (evenly spaced); fired straws
      are always drawn on top regardless of this cap.
    view_z_spread : the four co-located views are nudged apart in z by this much
      (cm) purely so they are visually separable; the physics keeps them at the
      station's z.
    """
    w = detector.view_half_width
    h = detector.view_half_height
    n_straws = detector.n_straws
    pitch = detector.straw_pitch
    zm0, zm1 = detector.magnet_z
    dvz0, dvz1 = detector.decay_volume_z

    station_z = np.asarray(sample["station_z"])[event]  # (S,)
    tilt = np.asarray(sample["tilt"])[event]  # (V,)
    valid = np.asarray(sample["valid"])[event]  # (2, S, V)
    straw = np.asarray(sample["straw"])[event]  # (2, S, V)
    trajectories = np.asarray(sample["trajectories"])[event]  # (2, 4, 3)
    vertex = np.asarray(sample["vertex"])[event]  # (3,)

    S = station_z.shape[0]
    V = tilt.shape[0]

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    # Volumes: decay region and magnet.
    _box(plotter, dvz0, dvz1, w, h, color="lightgray", opacity=0.06)
    _box(plotter, zm0, zm1, w, h, color="purple", opacity=0.12)

    # Stations -> views -> straws.
    context_idx = np.unique(np.linspace(0, n_straws - 1, min(max_straws_drawn, n_straws)).astype(int))
    for s in range(S):
        z_station = float(station_z[s])
        for v in range(V):
            t = float(tilt[v])
            z = z_station + (v - 0.5 * (V - 1)) * view_z_spread  # visual separation
            color = _VIEW_PALETTE[v % len(_VIEW_PALETTE)]

            frame = pv.PolyData(_view_corners(w, h, t, z), faces=np.array([[4, 0, 1, 2, 3]]))
            plotter.add_mesh(frame, color="black", style="wireframe", opacity=0.4, line_width=1.0)

            # context straws (sparse) for orientation
            for k in context_idx:
                c = (k + 0.5) * pitch - h
                plotter.add_mesh(
                    pv.lines_from_points(_straw_line(w, t, c, z)),
                    color=color,
                    opacity=0.25,
                    line_width=1.0,
                )

            # fired straws (any daughter), drawn as thick tubes on top
            fired = np.unique(straw[valid[:, s, v].astype(bool), s, v])
            for k in fired:
                if 0 <= int(k) < n_straws:
                    c = (int(k) + 0.5) * pitch - h
                    tube = pv.lines_from_points(_straw_line(w, t, c, z)).tube(radius=0.6 * pitch)
                    plotter.add_mesh(tube, color=color, opacity=0.95)

    # Daughter trajectories.
    for p in range(trajectories.shape[0]):
        poly = trajectories[p]  # (4, 3)
        plotter.add_mesh(
            pv.Spline(poly, max(64, poly.shape[0])),
            color=_DAUGHTER_COLORS[p % len(_DAUGHTER_COLORS)],
            line_width=4,
        )

    # Decay vertex.
    plotter.add_mesh(pv.Sphere(radius=2.0, center=vertex), color="black")

    plotter.add_legend(
        [(_DAUGHTER_NAMES[p], _DAUGHTER_COLORS[p]) for p in range(trajectories.shape[0])],
        bcolor="white",
    )
    plotter.show_grid(xtitle="x [cm]", ytitle="y [cm]", ztitle="z [cm]")
    plotter.camera_position = [
        (-400.0, 250.0, 200.0),
        (0.0, 0.0, 0.5 * (dvz0 + 400.0)),
        (0.0, 1.0, 0.0),
    ]
    plotter.reset_camera()
    if screenshot is not None:
        plotter.show(screenshot=screenshot)
    else:
        plotter.show()
    return plotter
