"""Faithful 3D view of the straw detector (pyvista).

Draws, for a single solved event, exactly the three things that matter:

  * the daughter/secondary **trajectories** (the C solver's per-particle paths),
  * the **fired straws**, highlighted as their true tubes in the lab frame,
  * one wireframe **frame parallelogram** per layer to show the layer planes.

Unfired straws are hidden (only the fired tubes are drawn). The geometry is
reconstructed from the detector + the physical design array the same way the C
solver lays it out, so a faithful field/geometry model has the trajectories
threading the fired tubes.

Each layer is drawn as a **parallelogram** with vertical sides (parallel to y at
x = +/-width): the straws are stacked in y at the straw pitch and the wires are
tilted by the stereo ``angle`` from x, ends locked to the x = +/-width sides. The
hit ``straw`` index maps to a y position ``y = (2*straw+1)*r - height + yoff``
(``r`` the straw radius, ``yoff`` the half-pitch stagger). NB: the C solver
currently models the stereo as a pure *rotation* about z (a rotated rectangle),
which agrees with this shear to O(angle^2); see the note in the module.
"""

import numpy as np

try:
    import pyvista as pv
except Exception:  # pragma: no cover - pyvista is an optional viz dep
    pv = None

__all__ = ["show_event", "daughter_polylines"]


def daughter_polylines(detector, dd, design, rng, *, z_grid=None, m=None):
    """Per-daughter ``[x, y, z]`` polylines from the C solver's crossing-track buffer.

    Solves the single hand-built event ``dd`` once at ``design`` and samples each of the first
    ``m`` primaries' (the HNL daughters') in-aperture ``(x, y)`` crossings at the reference planes
    ``z_grid`` (default: a fine grid over ``detector.layer_bounds``). Returns ``(X, mask, lines)``
    with ``lines[p]`` a ``(k, 3)`` ``[x, y, z]`` array (z increasing) of the planes daughter ``p``
    actually crossed -- the drop-in replacement for the old per-particle trajectory slice (secondary
    paths are no longer recorded; only the first ``m`` primaries). ``m`` defaults to the event's
    particle count."""
    if z_grid is None:
        lo, hi = detector.layer_bounds
        z_grid = np.linspace(float(lo), float(hi), 400)
    from detopt.detector.straw import Pool

    pool = Pool(dd["masses"], dd["charges"], dd["positions"], dd["momenta"], dd["times"])
    n = len(np.asarray(dd["masses"]))
    m = n if m is None else int(m)
    zp = np.ascontiguousarray(z_grid, dtype=np.float32)
    traj = np.zeros((1, m, zp.shape[0], 3), dtype=np.float32)
    n_cross = np.zeros((1, m), dtype=np.int32)
    part_idx = np.full((1, m), -1, dtype=np.int32)
    layers, angles, Bs = detector._design_to_geometry(design)
    hits_idx, tdc, _ = detector._run_solver(
        pool, np.array([[0, n]], np.int32), layers, angles, Bs,
        rng.integers(1, 2**32, size=1, dtype=np.uint32), z_planes=zp,
        traj=traj, n_cross=n_cross, part_idx=part_idx, primaries=True,
    )
    lines = [traj[0, s, : int(n_cross[0, s])] for s in range(m)]  # each (n_cross, 3) ordered (x, y, z) polyline
    # show_event consumes the dense (1, M, 5) [station, view, layer, straw, tdc] layout (padding tdc = -1,
    # masked out below) -- reassemble it from the engine's hits_idx + tdc buffers.
    X = np.concatenate([hits_idx.astype(np.float32), np.asarray(tdc, np.float32)[..., None]], axis=-1)
    return X, (tdc >= 0).astype(np.int32), lines


def _require_pyvista():
    if pv is None:
        raise ImportError("pyvista is required for the 3D straw view (pip-free env: install it yourself).")


def _layer_geometry(detector, design):
    """Per-layer ``(z, angle)`` from the physical free-design array ``[z(n), angle(n), B]``."""
    n = detector.n_layers
    design = np.asarray(design, dtype=np.float32).ravel()
    return design[:n], design[n : 2 * n]


def _frame_corners(detector, z, angle):
    """Lab-frame corners of a layer's parallelogram: the left/right sides are vertical
    (parallel to y at x = +/-width); the top/bottom edges are sheared by the stereo
    tilt -- the wires run at ``angle`` from x with their ends locked to x = +/-width."""
    w, h = detector.layer_width, detector.layer_height
    s = w * np.tan(angle)
    x = np.array([-w, -w, w, w], dtype=np.float32)
    y = np.array([-h - s, h - s, h + s, -h + s], dtype=np.float32)
    return np.stack([x, y, np.full_like(x, z)], axis=-1)


def _straw_endpoints(detector, z, angle, layer_in_view, straw):
    """Lab-frame endpoints of one fired straw: stacked in y at the straw pitch, the wire
    tilted by ``angle`` from x with its ends on the vertical x = +/-width sides."""
    r = detector.straw_pitch / 2.0  # straw radius (== layer_height / n_straws)
    yoff = 0.5 * detector.layer_y_offset if (int(layer_in_view) & 1) else -0.5 * detector.layer_y_offset
    y_i = (2 * int(straw) + 1) * r - detector.layer_height + yoff
    w = detector.layer_width
    s = w * np.tan(angle)
    return np.array([[-w, y_i - s, z], [w, y_i + s, z]], dtype=np.float32)


def show_event(
    detector,
    design,
    X,
    mask,
    trajectories=None,
    *,
    n_daughters=None,
    off_screen=True,
    screenshot=None,
    title=None,
    straw_color="red",
    frame_color="#b3b3b3",
):
    """Render one solved event in 3D.

    Args:
        detector: the ``StrawDetector`` (geometry source).
        design: the physical design array ``[layer_z(n), layer_angle(n), B]``.
        X: one event's dense hits ``(M, 5)`` = ``[station, view, layer_in_view, straw, time]``.
        mask: ``(M,)`` 1 for a real hit. Only fired straws are drawn.
        trajectories: optional sequence of per-daughter ``(k, 3)`` ``[x, y, z]`` polylines
            (from :func:`daughter_polylines`) -- one per HNL daughter. (A single
            ``(n_particles, n_t, 3)`` array still works: it is iterated per particle.)
        n_daughters: number of primaries (the HNL daughters) to highlight; any extra
            polylines are drawn as muted thin secondaries. ``None`` -> all daughters.
        off_screen: render without a window (headless); pair with ``screenshot``.
        screenshot: path to save a PNG (None -> show interactively if not off_screen).
    """
    _require_pyvista()
    positions, angles = _layer_geometry(detector, design)
    per_station = detector.n_views_per_station * detector.n_layers_per_view
    straw_r = detector.straw_pitch / 2.0

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    # Layer frames: one wireframe parallelogram per layer plane. Accumulate the
    # corner extent so the camera can be framed on the detector volume alone (the
    # trajectories/secondaries may fly well outside it).
    all_corners = []
    for k in range(detector.n_layers):
        corners = _frame_corners(detector, float(positions[k]), float(angles[k]))
        all_corners.append(corners)
        loop = pv.lines_from_points(np.vstack([corners, corners[0]]))
        plotter.add_mesh(loop, color=frame_color, line_width=1.0, opacity=0.4)
    all_corners = np.concatenate(all_corners, axis=0)
    lo, hi = all_corners.min(axis=0), all_corners.max(axis=0)
    det_bounds = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]  # xmin,xmax,ymin,ymax,zmin,zmax

    # Fired straws only: highlight each as its true tube; unfired straws are hidden.
    m = np.asarray(mask).astype(bool)
    hits = np.asarray(X)[m]
    for station, view, lpv, straw, _t in hits:
        k = int(station) * per_station + int(view) * detector.n_layers_per_view + int(lpv)
        if not (0 <= k < detector.n_layers):
            continue
        ends = _straw_endpoints(detector, float(positions[k]), float(angles[k]), lpv, straw)
        tube = pv.lines_from_points(ends).tube(radius=straw_r, n_sides=12)
        plotter.add_mesh(tube, color=straw_color, opacity=1.0, smooth_shading=True)

    # All tracked particles. HNL daughters (slots 0..n_daughters-1) are highlighted:
    # thick, opaque, vivid. Secondaries (delta-rays / conversion pairs / decay muons)
    # are drawn muted and thin so the daughters stand out.
    if trajectories is not None:
        n_d = len(trajectories) if n_daughters is None else int(n_daughters)
        daughter_palette = ["royalblue", "limegreen", "magenta", "cyan", "darkorange", "purple"]
        for p, pts in enumerate(trajectories):
            pts = np.asarray(pts)
            pts = pts[np.abs(pts).sum(axis=1) > 0]
            # Cut the portion that leaves the detector volume (clip to the frame box).
            inside = (
                (pts[:, 0] >= det_bounds[0])
                & (pts[:, 0] <= det_bounds[1])
                & (pts[:, 1] >= det_bounds[2])
                & (pts[:, 1] <= det_bounds[3])
                & (pts[:, 2] >= det_bounds[4])
                & (pts[:, 2] <= det_bounds[5])
            )
            pts = pts[inside]
            if len(pts) <= 1:
                continue
            line = pv.lines_from_points(pts)
            if p < n_d:  # HNL daughter -> highlighted
                plotter.add_mesh(
                    line.tube(radius=0.8 * straw_r, n_sides=10), color=daughter_palette[p % len(daughter_palette)], opacity=1.0
                )
            else:  # tracked secondary -> muted, thin
                plotter.add_mesh(line.tube(radius=0.25 * straw_r, n_sides=6), color="#777777", opacity=0.5)

    plotter.add_axes(line_width=2)
    plotter.show_grid(color="#999999")
    if title:
        plotter.add_text(title, font_size=10, color="black")

    # Orient: look along -x so increasing z renders left->right and +y points up; then
    # rotate ~30 deg about the (vertical y) up-axis so the x depth is visible. Frame on
    # the detector volume only (ignore stray secondaries/trajectories that fly outside).
    z_c = 0.5 * (det_bounds[4] + det_bounds[5])
    focal = (0.0, 0.0, z_c)
    span = max(hi - lo)
    pos = (-span, 0.0, z_c)  # camera on -x side, looking along +x so increasing z renders left->right
    plotter.camera_position = [pos, focal, (0.0, 1.0, 0.0)]
    plotter.reset_camera(bounds=det_bounds)  # fit to the detector box, keep orientation
    plotter.camera.Azimuth(30)  # swing ~30 deg about the up (y) axis for x-depth
    plotter.camera.zoom(1.25)

    if screenshot is not None:
        plotter.screenshot(screenshot)
        plotter.close()
        return screenshot
    plotter.show()
    return None
