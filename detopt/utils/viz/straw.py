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
(``r`` the straw radius, ``yoff`` the half-pitch stagger).

NB: this shear is what the C solver does too -- its acceptance test is
``|x| <= width and |Y| <= height`` with ``Y = y - x*tan(angle)``, and the straw
index is inverted from that same ``Y`` (``straw_detector.c``, "outside the
parallelogram frame"). An earlier version of this note claimed the solver used a
pure rotation about z agreeing to O(angle^2); it does not, and drawing a rotated
rectangle here would put the frames and tubes somewhere the hits do not come from.
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
    tilt -- the wires run at ``angle`` from x with their ends locked to x = +/-width.

    This is a SHEAR, not a rotation, and it is deliberate: it mirrors the solver's own acceptance
    test ``|x| <= width and |y - x*tan(angle)| <= height`` (``straw_detector.c``, "outside the
    parallelogram frame"), from which the straw index is inverted. Drawing a rotated rectangle here
    instead would look better -- the eye reads a parallelogram as a rectangle turned in depth -- but
    it would no longer be the geometry the hits come from."""
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
    track_scale=0.8,
    frame_color="#b3b3b3",
    frame_opacity=0.4,
    frame_width=1.0,
    fill_color="#808080",
    fill_opacity=0.0,
    view_colors=None,
    all_straws=False,
    all_straw_color="#9a9a9a",
    all_straw_opacity=0.12,
    all_straw_width=1.0,
    frame_granularity="layer",
    bank=0.0,
    azimuth=30.0,
    elevation=0.0,
    zoom=1.25,
    parallel_projection=False,
    show_axes=True,
    show_grid=True,
    grid_font_size=None,
    depth_peeling=True,
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
        frame_opacity, frame_width: the per-layer wireframes are the layer boundaries; at the
            0.4 / 1.0 default they are a faint backdrop, raise both to make them read as structure.
        azimuth: degrees swung about the vertical (y) axis away from the side-on view. 0 puts z
            across the image with x pointing straight away from the viewer; larger values trade
            that for depth, and z then renders at ``cos(azimuth)`` of its true length.
        elevation: degrees the camera is tipped down after the azimuth swing.
        zoom: camera zoom applied last. Above ~1.2 the axis labels start leaving the canvas.
        parallel_projection: orthographic instead of perspective. Equal world lengths then render
            equal, so proportions can be read off the image; near and far stations come out the
            same size, which is the check that nothing is being scaled.
        show_axes: draw the corner orientation gizmo. It is anchored to the canvas corner rather
            than to the scene, so it holds the cropped margin open.
        fill_color, fill_opacity: a translucent surface per layer. Wireframes occlude nothing, so
            without a fill all 32 outlines show through one another and the stack reads as
            interpenetrating. 0.0 (the default) keeps the historical pure-wireframe look.
        view_colors: optional sequence of colours, one per stereo view within a station, cycled.
            Overrides ``frame_color`` and tints the fill to match. With the nominal view angles
            ``[0, +a, -a, 0]``, a 3-colour sequence gives the two Y-views the same colour.
        show_grid: draw the labelled bounding box and tick marks.
    """
    _require_pyvista()
    positions, angles = _layer_geometry(detector, design)
    per_station = detector.n_views_per_station * detector.n_layers_per_view
    straw_r = detector.straw_pitch / 2.0

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    if depth_peeling and (fill_opacity > 0.0 or all_straws or frame_opacity < 1.0):
        # Belt and braces. The fills are already added back-to-front above, which is exact for
        # non-intersecting parallel planes. Peeling covers anything else translucent -- but note it
        # is SILENTLY DISABLED when SSAA anti-aliasing is on: enable_depth_peeling() still returns
        # True while GetLastRenderingUsedDepthPeeling() reports 0, which is why the sort exists.
        peeled = plotter.enable_depth_peeling(number_of_peels=int(__import__('os').environ.get('PEELS', 16)), occlusion_ratio=0.0)
        if peeled is False:
            import warnings
            warnings.warn("depth peeling unavailable: translucent layers will blend in draw order")

    # Layer frames: one wireframe parallelogram per layer plane. Accumulate the
    # corner extent so the camera can be framed on the detector volume alone (the
    # trajectories/secondaries may fly well outside it).
    # Which layers get an outline. The two layers of a view sit 1.732 cm apart in a scene ~1138 cm
    # deep, so their outlines are visually duplicate; and within a station the views are sheared
    # 0/+a/-a/0, so their outlines cross one another whatever colour they carry. Drawing one
    # outline per view, or one per station, is what actually removes that.
    if frame_granularity == "view":
        drawn = range(0, detector.n_layers, detector.n_layers_per_view)
    elif frame_granularity == "station":
        drawn = range(0, detector.n_layers, per_station)
    else:
        drawn = range(detector.n_layers)

    corners_by_layer = [_frame_corners(detector, float(positions[k]), float(angles[k])) for k in range(detector.n_layers)]
    _stacked = np.concatenate(corners_by_layer, axis=0)
    _lo, _hi = _stacked.min(axis=0), _stacked.max(axis=0)
    _span = max(_hi - _lo)
    _zc = 0.5 * (_lo[2] + _hi[2])
    # The camera is only applied at the end, but its DIRECTION is fixed by bank/azimuth/elevation
    # and is unaffected by the later reset_camera (which only slides along the view axis). Build a
    # scratch camera with the same operations and read the direction off it, so the translucent
    # fills can be added back-to-front. For parallel planes that never intersect, painter's order is
    # exact -- and unlike depth peeling it is not silently disabled by SSAA.
    _b = np.radians(bank)
    _scratch = pv.Camera()
    _scratch.SetPosition(-_span * np.cos(_b), -_span * np.sin(_b), _zc)
    _scratch.SetFocalPoint(0.0, 0.0, _zc)
    _scratch.SetViewUp(-np.sin(_b), np.cos(_b), 0.0)
    _scratch.Azimuth(azimuth)
    _scratch.Elevation(elevation)
    view_dir = np.array(_scratch.GetDirectionOfProjection(), dtype=float)

    pending_fills = []
    all_corners = []
    for k in range(detector.n_layers):
        corners = corners_by_layer[k]
        all_corners.append(corners)

        if all_straws:
            # Every wire in EVERY layer -- outside the granularity filter below, which only thins
            # the outlines. Drawn as lines rather than tubes: a 2 cm straw in a ~1100 cm scene is
            # sub-pixel, so tubing 10k of them costs a great deal and shows nothing. Grey and faint,
            # so they read as inactive; the fired ones are drawn opaque further down.
            n = detector.n_straws
            ends = np.concatenate(
                [_straw_endpoints(detector, float(positions[k]), float(angles[k]), k % detector.n_layers_per_view, i)
                 for i in range(n)], axis=0
            )
            cells = np.hstack([np.full((n, 1), 2, dtype=np.int64), np.arange(2 * n, dtype=np.int64).reshape(n, 2)])
            wires = pv.PolyData(ends.astype(np.float64), lines=cells.ravel())
            plotter.add_mesh(wires, color=all_straw_color, opacity=all_straw_opacity, line_width=all_straw_width)

        if k not in drawn:
            continue
        # One colour per stereo view, so the pattern of view angles is legible: the two views at
        # angle 0 share a colour and the +/- stereo pair get their own.
        if view_colors is None:
            layer_color = frame_color
        else:
            view = (k % per_station) // detector.n_layers_per_view
            layer_color = view_colors[view % len(view_colors)]
        loop = pv.lines_from_points(np.vstack([corners, corners[0]]))
        plotter.add_mesh(loop, color=layer_color, line_width=frame_width, opacity=frame_opacity)
        if fill_opacity > 0.0:
            # A faint surface per layer. Wireframes alone occlude nothing, so all 32 outlines show
            # through one another and the stack reads as interpenetrating rather than stacked.
            quad = pv.PolyData(corners.astype(np.float64), faces=np.array([4, 0, 1, 2, 3]))
            tint = fill_color if view_colors is None else layer_color
            pending_fills.append((float(corners.mean(axis=0) @ view_dir), quad, tint))

    # Farthest first: larger projection on the view direction is further from the camera.
    for _, quad, tint in sorted(pending_fills, key=lambda item: -item[0]):
        plotter.add_mesh(quad, color=tint, opacity=fill_opacity, show_edges=False, lighting=False)

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
                    line.tube(radius=track_scale * straw_r, n_sides=10), color=daughter_palette[p % len(daughter_palette)], opacity=1.0
                )
            else:  # tracked secondary -> muted, thin
                plotter.add_mesh(line.tube(radius=0.31 * track_scale * straw_r, n_sides=6), color="#777777", opacity=0.5)

    if show_axes:
        plotter.add_axes(line_width=2)
    if show_grid:
        # Default VTK tick text is sized for a screen-resolution canvas; on a 3200 px
        # canvas scaled into a text-width slot it lands around 3 pt.
        grid_kwargs = {"color": "#999999"}
        if grid_font_size is not None:
            grid_kwargs["font_size"] = grid_font_size
        plotter.show_grid(**grid_kwargs)
    if title is not None:
        plotter.add_text(title, font_size=10, color="black")

    # Orient: look along -x so increasing z renders left->right and +y points up; then
    # rotate ~30 deg about the (vertical y) up-axis so the x depth is visible. Frame on
    # the detector volume only (ignore stray secondaries/trajectories that fly outside).
    z_c = 0.5 * (det_bounds[4] + det_bounds[5])
    focal = (0.0, 0.0, z_c)
    span = max(hi - lo)
    # `bank` orbits the viewpoint about the beam axis z, carrying the up-vector with it so z stays
    # horizontal in the image. Unlike a camera roll -- which rotates the image plane and therefore
    # cannot change whether projected outlines cross -- this changes the projection: the layer
    # shear lies in the x-y plane, so at bank = 90 deg it is viewed edge-on and disappears.
    b = np.radians(bank)
    pos = (-span * np.cos(b), -span * np.sin(b), z_c)
    up = (-np.sin(b), np.cos(b), 0.0)
    plotter.camera_position = [pos, focal, up]
    plotter.reset_camera(bounds=det_bounds)  # fit to the detector box, keep orientation
    plotter.camera.Azimuth(azimuth)  # swing about the up (y) axis for x-depth
    plotter.camera.Elevation(elevation)
    if parallel_projection:
        plotter.enable_parallel_projection()
    plotter.camera.zoom(zoom)
    # reset_camera() fixed the clipping range for the PRE-rotation camera; the azimuth/elevation
    # swing then moves geometry across it and the near stations get sliced by the near plane.
    # Recompute it last, against the full scene rather than the detector box, so trajectories
    # leaving the volume are not clipped either.
    plotter.renderer.ResetCameraClippingRange()

    if screenshot is not None:
        plotter.screenshot(screenshot)
        plotter.close()
        return screenshot
    plotter.show()
    # The camera object outlives the closed window, so the view the user left the scene at can be
    # read back here. Returned rather than logged so nothing has to hook into the interactor --
    # attaching observers or timers to a live plotter is what breaks interactivity.
    try:
        return plotter.camera_position
    except Exception:
        return None
