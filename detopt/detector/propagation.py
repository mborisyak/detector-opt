"""Pluggable propagation engine for a straw-geometry detector.

A `PropagationEngine` turns a batch of *already-sampled* particle events into digitized straw hits: it
propagates each particle through the per-event layer geometry + magnetic field and records the fired
straws with their drift-time TDC. It is **modeled after the C `straw_detector.solve`** (see
`StrawDetector._run_solver`, `straw.py:~630`): the engine RECEIVES the events — it does NOT sample them —
so the event SOURCE (a sampler) is a separate, orthogonal component. Any source can feed any engine, e.g.
analytically-generated events can be pushed through the realistic C solver, or pool events through the
simplified one.

Two implementations share this interface (so a network's features are engine-agnostic — train on one,
apply to the other):
  * ``RealisticEngine``  — a thin wrapper over the C ODE solver (Boris push through the Gaussian field,
                           secondaries, drift TDC).
  * ``SimplifiedEngine`` — analytic straight -> bend (box field) -> straight, with the SAME drift TDC.

Responsibilities split:
  * SOURCE (sampler)  -> ``(input_events, boundaries, target, ground_truth)`` (pool OR analytic).
  * ENGINE (this)     -> ``(X, mask, traj)`` from the events + the design geometry + field.
  * DETECTOR          -> geometry/design/combine/target/normalization; packs ``X`` -> StrawEvent etc.
"""
import math
from typing import Optional

import numpy as np

from . import straw_detector
from ..utils.det_random import det_normals

_STRAW_TDC_EMPTY = 1.0e30  # unfired-straw sentinel (matches straw_detector.c STRAW_TDC_EMPTY)


class PropagationEngine:
    """Abstract engine. Subclasses implement :meth:`solve`. Constructed with the FIXED straw layout
    (design-independent geometry) the detector supplies; per-event design + field arrive in `solve`.

    Buffer ownership (zero-alloc on BOTH sides):
      * The DETECTOR owns the NN-facing OUTPUT buffers (``X``/``mask``/``traj``): it allocates+tracks them
        once and reuses them every call, passing them into :meth:`solve` to be filled in place.
      * The ENGINE owns its engine-specific SCRATCH buffers: it allocates them once at construction
        (sized from the layout) and reuses them every call -- the realistic engine's C ``Scratch``, the
        simplified engine's analytic working arrays. These are private to the engine, never NN-facing.
    So nothing on the hot path is heap-allocated by either side."""

    def __init__(self, *, n_views_per_station: int, n_layers_per_view: int, n_straws: int,
                 n_layers: int, max_hits_per_event: int, layer_width: float, layer_height: float,
                 straw_pitch: float, layer_y_offset: float, B_sigma: float, z0: float):
        # Fixed layout + field SHAPE (matches the C `Layout`/Geometry): aperture, pitch, stagger, hit cap,
        # magnet z-width + centre. All design-INDEPENDENT -> engine hyper-parameters. (The field STRENGTH
        # `Bs` is a per-event design output and arrives in `solve`, not here -- it can be a DOF.)
        self.n_views_per_station = int(n_views_per_station)
        self.n_layers_per_view = int(n_layers_per_view)
        self.n_straws = int(n_straws)
        self.n_layers = int(n_layers)
        self.max_hits_per_event = int(max_hits_per_event)
        self.layer_width = float(layer_width)     # x half-extent (straw_length / 2)
        self.layer_height = float(layer_height)   # y half-extent (n_straws * pitch / 2)
        self.straw_pitch = float(straw_pitch)
        self.layer_y_offset = float(layer_y_offset)  # half-pitch stagger between the 2 layers in a view
        self.B_sigma = float(B_sigma)             # magnet z-width (fixed field shape)
        self.z0 = float(z0)                       # magnet centre z (fixed field shape)

    def solve(
        self,
        pool,                       # detector's `Pool`: `.c` (C InputEvents) + `.positions/.momenta/.charges`
        boundaries: np.ndarray,     # (n, 2) int32  [start, end) spans into the pool, per event
        layers: np.ndarray,         # (n, n_layers) float32  per-layer z (cm)            -- design output
        angles: np.ndarray,         # (n, n_layers) float32  per-layer view angle (rad)  -- design output
        Bs: np.ndarray,             # (n,) float32  per-event on-axis field strength     -- design output (DOF-capable)
        hits_idx: np.ndarray,       # (n, M, 4) uint32 OUTPUT -> [station, view, layer, straw]
        tdc: np.ndarray,            # (n, M) float32 OUTPUT    -> min-subtracted TDC; < 0 = no hit (the mask)
        *,
        seeds: np.ndarray,          # (n,) uint32 per-event seeds (the detector derives them from event_index)
        z_planes: Optional[np.ndarray] = None,   # (m,) reference z's for optional trajectory recording
        traj: Optional[np.ndarray] = None,       # (n, n_tracks, m, 3) OUTPUT, or None to skip
        n_cross: Optional[np.ndarray] = None,    # (n, n_tracks) OUTPUT crossing count, or None
        part_idx: Optional[np.ndarray] = None,   # (n, n_tracks) OUTPUT particle index, or None
        primaries: bool = False,
        process_ids: Optional[np.ndarray] = None,  # (n, M) int32 debug OUTPUT (realistic only), or None
        tree: Optional[dict] = None,               # MC-tree debug buffers (realistic only), or None
    ) -> None:
        """Propagate the detector's `pool` through the per-event design (`layers`, `angles`, `Bs`) and the
        engine's fixed field SHAPE (`self.B_sigma`, `self.z0`), digitizing the straw hits INTO the
        caller-allocated buffers `hits_idx` (addresses) + `tdc` (< 0 = no hit). The detector pre-inits
        `hits_idx=0`, `tdc=-1`; the engine overwrites only the real hits. The realistic engine hands the
        packed C `InputEvents` to the C solver (and `process_ids`/`tree` for debug); the simplified one
        reads `pool.positions/.momenta/.charges`. `Bs` is the per-event on-axis peak (Gaussian convention).
        """
        raise NotImplementedError


class RealisticEngine(PropagationEngine):
    """The realistic simulation: a thin wrapper over the C ``straw_detector.solve`` (Boris push through
    the GAUSSIAN field + secondaries + drift TDC). OWNS the C ``SimParams``/``Layout`` and the single-event
    ``Scratch`` (all allocated once at construction); ``solve`` packs the detector's ``Pool`` into a C
    ``InputEvents`` per call (cheap header checks) and hands it to the solver."""

    def __init__(self, sim_params, layout, *, n_views_per_station, n_layers_per_view, n_straws, n_layers,
                 max_hits_per_event, layer_width, layer_height, straw_pitch, layer_y_offset, B_sigma, z0):
        super().__init__(n_views_per_station=n_views_per_station, n_layers_per_view=n_layers_per_view,
                         n_straws=n_straws, n_layers=n_layers, max_hits_per_event=max_hits_per_event,
                         layer_width=layer_width, layer_height=layer_height, straw_pitch=straw_pitch,
                         layer_y_offset=layer_y_offset, B_sigma=B_sigma, z0=z0)
        self._sim_params = sim_params            # C SimParams (physics constants), built by the detector
        self._layout = layout                    # C Layout (structural counts + fixed z0/B_sigma)
        n_cells = self.n_layers * self.n_straws  # single-event per-straw scratch (sparse set)
        self._tdc = np.full(n_cells, _STRAW_TDC_EMPTY, dtype=np.float32)
        self._proc = np.zeros(n_cells, dtype=np.int32)
        self._fired_idx = np.zeros(n_cells, dtype=np.int32)
        self._scratch = straw_detector.Scratch(self._tdc, self._proc, self._fired_idx)

    def solve(self, pool, boundaries, layers, angles, Bs, hits_idx, tdc, *, seeds,
              z_planes=None, traj=None, n_cross=None, part_idx=None, primaries=False,
              process_ids=None, tree=None):
        # Pack the pool arrays into the C InputEvents (C does the validation -- a few header checks per
        # array, cheap, so no caching); build the optional debug buffers; hand it all to the C solver,
        # which fills `hits_idx` (n,M,4 uint32) + `tdc` (n,M f32, <0 = no hit) in place.
        ie = straw_detector.InputEvents(pool.masses, pool.charges, pool.positions, pool.momenta, pool.times)
        debug = None
        if process_ids is not None or tree is not None:
            debug = straw_detector.DebugBuffers(
                process_ids,
                None if tree is None else tree["int"], None if tree is None else tree["float"],
                None if tree is None else tree["event"], None if tree is None else tree["count"],
            )
        straw_detector.solve(
            self._sim_params, self._layout, ie, self._scratch,
            np.ascontiguousarray(seeds, np.uint32), np.ascontiguousarray(boundaries, np.int32), layers, angles,
            np.ascontiguousarray(Bs, np.float32), hits_idx, tdc,
            z_planes, traj, n_cross, part_idx, int(bool(primaries)), debug,
        )


class SimplifiedEngine(PropagationEngine):
    """The fast analytic simulation: each input particle is pushed straight -> bend (BOX field over
    z0 +/- B_sigma) -> straight to every layer z, then digitized with the SAME drift-time TDC as the C
    solver (drift radius from the sheared closest approach; ``ToF + t_drift + t_prop``, min-subtracted per
    event). No secondaries. Vectorized numpy; fills the detector's X/mask buffers in place."""

    C_CM_PER_NS = 29.9792458
    V_DRIFT = 1.0 / 300.0          # cm/ns (30 ns/mm)  -- matches straw_detector.c STRAW_VDRIFT
    SIGMA_SPATIAL = 0.012          # cm (120 um)

    def __init__(self, *, n_views_per_station, n_layers_per_view, n_straws, n_layers, max_hits_per_event,
                 layer_width, layer_height, straw_pitch, layer_y_offset, B_sigma, z0):
        super().__init__(n_views_per_station=n_views_per_station, n_layers_per_view=n_layers_per_view,
                         n_straws=n_straws, n_layers=n_layers, max_hits_per_event=max_hits_per_event,
                         layer_width=layer_width, layer_height=layer_height, straw_pitch=straw_pitch,
                         layer_y_offset=layer_y_offset, B_sigma=B_sigma, z0=z0)
        nv, nl = self.n_views_per_station, self.n_layers_per_view
        per_station = nv * nl
        gl = np.arange(self.n_layers)
        self._layer_station = (gl // per_station).astype(np.int32)         # global layer -> station
        within = gl % per_station
        self._layer_view = (within // nl).astype(np.int32)                # -> view-in-station (0..3)
        self._layer_lpv = (within % nl).astype(np.int32)                  # -> layer-in-view (0,1)
        # half-pitch stagger between the two layers of a view (same parity rule as straw.py).
        self._y_stagger = np.where((self._layer_lpv & 1) == 1,
                                   0.5 * self.layer_y_offset, -0.5 * self.layer_y_offset).astype(np.float32)
        # Box field region (FIXED, from the field-shape hyper-parameters): constant peak over z0 +/- B_sigma.
        # The box peak per event is derived from the per-event Gaussian peak `Bs` so both engines realize the
        # same int(B dz): box_peak * 2sigma = Bs * sigma * sqrt(2pi)  ->  box_peak = Bs * sqrt(2pi)/2.
        self._zm0 = self.z0 - self.B_sigma
        self._zm1 = self.z0 + self.B_sigma
        self._box_factor = math.sqrt(2.0 * math.pi) / 2.0

    def solve(self, pool, boundaries, layers, angles, Bs, hits_idx, tdc, *, seeds,
              z_planes=None, traj=None, n_cross=None, part_idx=None, primaries=False,
              process_ids=None, tree=None):
        n = int(boundaries.shape[0])
        # The detector pre-inits hits_idx=0 + tdc=-1; we overwrite only the real hits (padding stays -1).
        # The analytic propagator reads the pool's numpy arrays directly (mass/time don't enter the
        # straight->bend->straight + drift TDC). `boundaries` (n,2) carve out each event's particle span;
        # gather just the sampled particles (no python loop) + their event index.
        positions, momenta, charges = pool.positions, pool.momenta, pool.charges
        counts = (boundaries[:, 1] - boundaries[:, 0]).astype(np.int64)   # (n,) particles per sampled event
        H = int(counts.sum())
        part_event = np.repeat(np.arange(n, dtype=np.int32), counts)      # (H,) event index of each particle
        within = np.arange(H) - np.repeat(np.cumsum(counts) - counts, counts)  # 0..c-1 within each event
        row = boundaries[part_event, 0].astype(np.int64) + within         # (H,) row into the pool
        pos = np.asarray(positions, np.float32)[row]
        mom = np.asarray(momenta, np.float32)[row]
        chg = np.asarray(charges, np.float32)[row]
        layer_z = np.asarray(layers, np.float32)[part_event]              # (H, n_layers) each particle's event z's
        ang = np.asarray(angles, np.float32)[part_event]                  # (H, n_layers) per-layer view angle (rad)
        box_peak = (np.asarray(Bs, np.float32)[part_event] * self._box_factor)  # (H,) per-particle box peak
        # Per-hit smear key: a hash of (this particle's EVENT seed, its within-event index) -- so the drift
        # noise depends only on the event index, NOT the batch position (repeated index -> identical event).
        key = (np.asarray(seeds, np.uint64)[part_event] * np.uint64(0x9E3779B97F4A7C15)
               + (within.astype(np.uint64) + np.uint64(1)))

        x, y, path = self._propagate(pos, mom, chg, layer_z, box_peak)    # (H, n_layers)
        tdc_vals, straw, valid = self._digitize(x, y, path, ang, key)     # (H, n_layers)

        # scatter into per-event hits, dedup earliest TDC per (layer, straw), keep M earliest -> buffers.
        self._emit(part_event, tdc_vals, straw, valid, hits_idx, tdc)

    # ---- propagation: straight -> box-field arc (y bends, x is free; B along x) -> straight ---- #
    def _propagate(self, pos, mom, charge, layer_z, box_peak):
        x0, y0, z0 = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]               # (P,1)
        px, py, pz = mom[:, 0:1], mom[:, 1:2], mom[:, 2:3]
        pz = np.where(np.abs(pz) < 1e-6, 1e-6, pz)
        q = charge[:, None]
        zm0, zm1 = self._zm0, self._zm1                                   # box field region (z0 +/- B_sigma)
        B = box_peak[:, None]                                            # (P,1) per-event box peak (from Bs)
        x = x0 + (px / pz) * (layer_z - z0)                              # x never bends (B || x)
        y_lin = y0 + (py / pz) * (layer_z - z0)
        # arc in the (z, y) plane between zm0 and zm1
        p_yz = np.sqrt(py**2 + pz**2)
        denom = np.maximum(0.3 * np.abs(B) * np.maximum(np.abs(q), 1e-6), 1e-12)
        R = p_yz / denom * 100.0                                         # bend radius (cm)
        y_e = y0 + (py / pz) * (zm0 - z0)                                # y at field entry
        sgn = np.sign(q) * np.where(B == 0, 1.0, np.sign(B))
        Cz = zm0 + R * sgn * (-py / p_yz)
        Cy = y_e + R * sgn * (pz / p_yz)
        sq = np.sqrt(np.maximum(R**2 - (zm1 - Cz) ** 2, 0.0))
        y_lin_exit = y_e + (py / pz) * (zm1 - zm0)
        yp, ym = Cy + sq, Cy - sq
        y_exit = np.where(np.abs(yp - y_lin_exit) <= np.abs(ym - y_lin_exit), yp, ym)
        rz, ry = (zm1 - Cz), (y_exit - Cy)
        tz, ty = np.where(-ry > 0, -ry, ry), np.where(-ry > 0, rz, -rz)
        slope_exit = ty / np.where(np.abs(tz) < 1e-9, 1e-9, tz)
        straight = R > 1e5
        y_exit = np.where(straight, y_lin_exit, y_exit)
        slope_exit = np.where(straight, py / pz, slope_exit)
        y_after = y_exit + slope_exit * (layer_z - zm1)
        y = np.where(layer_z < zm0, y_lin, y_after)                      # before field: straight; after: bent
        path = np.abs(layer_z - z0) * np.sqrt(1.0 + (px / pz) ** 2 + (py / pz) ** 2)  # ~ToF path length
        return x.astype(np.float32), y.astype(np.float32), path.astype(np.float32)

    # ---- digitization: straw + drift radius + the real TDC (matches straw_detector.c:723-728) ---- #
    def _digitize(self, x, y, path, ang, key):
        tan = np.tan(ang)
        ca = np.cos(ang)
        pitch, height, ns = self.straw_pitch, self.layer_height, self.n_straws
        c = y - x * tan                                                 # sheared (wire-frame) coordinate
        straw_f = (c + height - self._y_stagger) / pitch - 0.5
        straw = np.clip(np.round(straw_f), 0, ns - 1).astype(np.int32)
        wire = (straw + 0.5) * pitch - height + self._y_stagger
        dr = np.abs(c - wire) * ca                                      # drift radius (cm)
        # Deterministic per-hit spatial smear keyed by (event seed, within-event particle, layer).
        noise = det_normals(key, dr.shape[1])                          # (H, n_layers) standard normals
        t_drift = np.abs(dr + noise * self.SIGMA_SPATIAL) / self.V_DRIFT
        t_prop = (self.layer_width - x) / (np.maximum(ca, 1e-6) * self.C_CM_PER_NS)
        tof = path / self.C_CM_PER_NS
        tdc = (tof + t_drift + t_prop).astype(np.float32)
        in_x = np.abs(x) <= self.layer_width
        in_y = np.abs(c) <= height
        valid = in_x & in_y
        return tdc, straw, valid

    # ---- emit: per-event dedup (earliest TDC per cell) + the M earliest-TDC straws -> hits_idx, tdc ---- #
    # Addresses are 0-based (station/view/layer-in-view/straw), matching the C solver (emit_event:
    # station=k/per_station, view, layer=k%n_lpv, straw=idx%n_straws), and the TDC is relative to the
    # earliest hit in the event -- so simplified and realistic events are interchangeable. Padding rows
    # keep the detector's pre-init (hits_idx=0, tdc=-1, i.e. no hit).
    def _emit(self, part_event, tdc_vals, straw, valid, hits_idx, tdc):
        n, M = hits_idx.shape[0], self.max_hits_per_event
        P, L = tdc_vals.shape
        layer_idx = np.broadcast_to(np.arange(L, dtype=np.int32), (P, L))
        for e in range(n):
            sel = part_event == e
            v = valid[sel].ravel()
            if not np.any(v):
                continue
            t = tdc_vals[sel].ravel()[v]                     # (H,) candidate-hit TDCs
            li = layer_idx[sel].ravel()[v]                   # (H,) global layer index
            sw = straw[sel].ravel()[v]                       # (H,) straw index
            # dedup: keep the earliest-TDC hit per (global-layer, straw) cell.
            cell = li.astype(np.int64) * self.n_straws + sw
            order = np.argsort(t, kind="stable")
            _, first = np.unique(cell[order], return_index=True)
            keep = order[first]
            # the M earliest-TDC of the deduped hits.
            best = keep[np.argsort(t[keep], kind="stable")[:M]]
            m = int(best.shape[0])
            lb, tb = li[best], t[best]
            hits_idx[e, :m, 0] = self._layer_station[lb]     # station 0..n_stations-1
            hits_idx[e, :m, 1] = self._layer_view[lb]        # view 0..3
            hits_idx[e, :m, 2] = self._layer_lpv[lb]         # layer-in-view 0..n_lpv-1
            hits_idx[e, :m, 3] = sw[best]                    # straw 0..n_straws-1
            tdc[e, :m] = tb - tb.min()                       # TDC relative to earliest in event (>= 0)
