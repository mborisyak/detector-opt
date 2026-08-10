"""StrawDetector with a fixed stereo-view layout and a compact design space.

Each station carries a fixed [0, +a, -a, 0] view layout (horizontal, +stereo,
-stereo, horizontal). The only optimizable design is:

    [ station_z (n_stations) , stereo_angle a ]

so dimension ``n_stations + 1`` instead of the base ``2*n_layers + 1``. The
magnetic field and the intra-station gaps are fixed. The station-z scaling is
**coupled and ordered**: each station has a z width (``station_width``) and the
map back to nominal is sequential, so stations are always ordered, never overlap
each other, and never intrude into the spectrometer magnet (centred at ``z0``,
half-width ``magnet_half_cm``). Upstream stations sit below ``z0 - magnet_half``,
downstream ones above ``z0 + magnet_half`` (see ``_station_lo_hi``).
"""

from typing import NamedTuple

import jax
import numpy as np

from .straw import Pool, StrawDetector, scale, unscale

__all__ = ["StereoStrawDetector", "StereoDesign"]


class StereoDesign(NamedTuple):
    """Compact stereo design: per-station z centres + one shared stereo angle (field fixed)."""

    stations: jax.Array  # (..., n_stations)
    view_angle: jax.Array  # (..., 1)  stereo view angle in radians (FairShip's `view_angle`)


class StereoStrawDetector(StrawDetector):
    def __init__(
        self,
        # Station structure: the first n_stations_upstream sit upstream of the magnet,
        # the rest downstream -- this (not a nominal station_z) sets each station's bounds.
        n_stations_upstream: int = 2,
        n_stations_downstream: int = 2,
        n_layers_per_view: int = 2,
        n_straws_per_layer: int = 316,  # FairShip SST V2023 straws per layer (digi_straw index 1..316); 2 cm pitch -> |y| <= 316 cm
        straw_pitch: float = 2.0,  # cm; = outer straw diameter (tightly packed) -> straw radius 1 cm
        straw_length: float = 400.0,  # cm; FairShip SST aperture width 200 cm (half)
        # Stereo design scheme: intra-station view/layer z spacing + optimizable angle bound.
        layer_z_gap: float = 1.732,
        view_z_gap: float = 5.0,
        stereo_bound: tuple = (0.0, 0.2),
        magnet_half_cm: float = 140.0,  # magnet z half-width (FairShip YokeDepth)
        station_width: float = 100.0,  # station z full-width (= 2 x FairShip strawtubes station_length 50)
        station_clearance: float = 0.0,  # minimum z gap between adjacent station footprints (0 = may touch)
        # Physics / field -- FairShip V2023 MainSpectrometerField (NEGATIVE polarity). Parametrized by the
        # FIELD INTEGRAL int Bx dz (T.m, the magnet's bending power) + the Gaussian width B_sigma (cm); the
        # on-axis peak is DERIVED (max_B = field_integral / (B_sigma*sqrt(2pi))). int Bx dz = -1.07 T.m is
        # the faithful value (the sign matters -- the old +0.20 peak bent the wrong way -> 12-18 cm
        # divergence downstream); the width is not separately faithful (208 & 283 cm both give -1.07 T.m).
        field_integral: float = -1.069,  # T.m, int Bx dz  (== -0.205 T peak x 208 cm)
        z0: float = 8957.0,
        B_sigma: float = 208.0,
        layer_bounds: tuple = (8000.0, 10000.0),
        dt=None,
        max_dt: float = 1.0,
        max_time: float = 200.0,
        max_particles: int = 5,
        max_hits_per_event: int = 384,
        material: str = "kapton",
        wall_thickness: float = 0.0036,
        # Physics-process defaults below match the base StrawDetector: tuned so the
        # per-hit process composition approximately matches the FairShip MC truth.
        delta_Tcut: float = 0.28,
        lambda_conv_cm=95.0,
        enable_decay: bool = True,
        noise_rate: float = 0.0,
        scatter_xX0: float = 2.5e-4,
        # Target/conditioning normalization is FIXED in the base StrawDetector (see its
        # decay_mean/.../mass_* defaults) so a data swap can't re-scale the loss; not
        # overridden here. To recompute from data, edit the base defaults to None.
        data_dir=None,
        boundary_z=None,
        engine: str = "realistic",  # 'realistic' (C ODE solver) | 'simplified' (analytic straight->bend->straight)
        # The no-data analytic INPUT-EVENT source (momentum_mean/vertex_mean/hnl_mass/...) is generic HNL
        # physics owned by the BASE StrawDetector and forwarded through -- it is NOT a stereo-geometry
        # concern, so this class doesn't name it.
        **kwargs,
    ):
        # Derive the on-axis Gaussian peak (T) from the field integral (T.m) + width (cm). z is in cm
        # while int Bx dz is quoted in T.m, hence the 100x: peak * B_sigma[cm] * sqrt(2pi) = int Bx dz[T.cm]
        # = 100 * int Bx dz[T.m].
        max_B = field_integral * 100.0 / (B_sigma * np.sqrt(2.0 * np.pi))
        super().__init__(
            n_stations=int(n_stations_upstream) + int(n_stations_downstream),
            n_views_per_station=4,  # fixed [0, +a, -a, 0] stereo layout
            n_layers_per_view=n_layers_per_view,
            n_straws_per_layer=n_straws_per_layer,
            straw_pitch=straw_pitch,
            straw_length=straw_length,
            max_B=max_B,
            z0=z0,
            B_sigma=B_sigma,
            layer_bounds=layer_bounds,
            # angle features (combine) normalised by the stereo angle range
            angles_bounds=(-float(stereo_bound[1]), float(stereo_bound[1])),
            dt=dt,
            max_dt=max_dt,
            max_time=max_time,
            max_particles=max_particles,
            max_hits_per_event=max_hits_per_event,
            material=material,
            wall_thickness=wall_thickness,
            delta_Tcut=delta_Tcut,
            lambda_conv_cm=lambda_conv_cm,
            enable_decay=enable_decay,
            noise_rate=noise_rate,
            scatter_xX0=scatter_xX0,
            data_dir=data_dir,
            boundary_z=boundary_z,
            engine=engine,
            **kwargs,  # the no-data analytic INPUT-EVENT params (momentum_mean/...) -> base StrawDetector
        )
        self.field_integral = float(field_integral)  # T.m, int Bx dz (config param; self.max_B is derived)
        self.n_stations_upstream = int(n_stations_upstream)
        self.stereo_bound = (float(stereo_bound[0]), float(stereo_bound[1]))
        self.magnet_half_cm = float(magnet_half_cm)
        self.station_width = float(station_width)
        self.station_clearance = float(station_clearance)
        self.layer_z_gap = float(layer_z_gap)
        self.view_z_gap = float(view_z_gap)

        # Fixed per-layer maps (global layer -> station, z-offset within station,
        # stereo-angle sign): ordering is station -> view -> layer-in-view.
        # station_z is the CENTRE of each station: offset the per-layer z by -span/2 so
        # the views/layers are laid out symmetrically about the station-centre design dof.
        span = (4 - 1) * self.view_z_gap + (self.n_layers_per_view - 1) * self.layer_z_gap
        half_span = 0.5 * span
        pattern = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)  # [0,+a,-a,0]
        st, zoff, sign = [], [], []
        for s in range(self.n_stations):
            for v in range(4):
                for l in range(self.n_layers_per_view):
                    st.append(s)
                    zoff.append(v * self.view_z_gap + l * self.layer_z_gap - half_span)
                    sign.append(pattern[v])
        self._layer_station = np.array(st, dtype=np.int64)
        self._layer_zoff = np.array(zoff, dtype=np.float32)
        self._layer_anglesign = np.array(sign, dtype=np.float32)
        # The event SOURCE (data gather OR the no-data analytic HNL generator) lives entirely in the base
        # StrawDetector -- this class is ONLY the stereo geometry parametrization.

    def _station_lo_hi(self, k, prev_z):
        """Coupled ``(lo, hi)`` bound on the station-CENTRE z at global index ``k``.

        Stations have a z width (``station_width``) and cannot overlap each other or
        the magnet (``z0 +/- magnet_half_cm``). The bound is sequential: it depends on
        the previous station's z (``prev_z``; ``None`` for the first station of a side).
        Upstream stations live below the magnet, downstream ones above; each reserves
        room for the stations still to come between it and the magnet edge / detector
        edge. ``prev_z`` may be a (batched) array -> the returned bound is too.
        """
        w = self.station_width
        h = 0.5 * w  # station half-width: keep this clear of the magnet face / detector edge
        p = w + self.station_clearance  # min center-to-center pitch (footprints + clearance)
        n_up = self.n_stations_upstream
        n_dn = self.n_stations - n_up
        min_z, max_z = self.layer_bounds
        if k < n_up:  # upstream of the magnet (most-upstream first)
            i = k
            lo = (min_z + h) if prev_z is None else prev_z + p
            hi = (self.z0 - self.magnet_half_cm - h) - (n_up - 1 - i) * p
        else:  # downstream of the magnet (nearest-magnet first)
            j = k - n_up
            lo = (self.z0 + self.magnet_half_cm + h) if prev_z is None else prev_z + p
            hi = (max_z - h) - (n_dn - 1 - j) * p
        return lo, hi

    # ------------------------------------------------------------------ #
    # Compact design space: [station_z(n_stations), stereo_angle]
    # ------------------------------------------------------------------ #
    def design_shape(self):
        # [station_z(n_stations), stereo_angle];  peak field is FIXED at max_B (not a dof)
        return (self.n_stations + 1,)

    def design_spec(self):
        # StereoDesign filled with ShapeDtypeStruct; flatten/unflatten are generic (base, via tensor).
        f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
        return StereoDesign(stations=f(self.n_stations), view_angle=f(1))

    def design_bounds(self):
        # station-CENTRE range (footprint must fit inside layer_bounds) + the stereo angle range
        lo, hi = self.layer_bounds
        return {
            "stations": (lo + 0.5 * self.station_width, hi - 0.5 * self.station_width),
            "view_angle": (self.stereo_bound[0], self.stereo_bound[1]),
        }

    def _as_stereo_design(self, design, xp):
        """Normalize a physical design -- a :class:`StereoDesign` namedtuple OR its flat
        ``[station_z(n_stations), view_angle]`` array -- to a BATCHED ``StereoDesign``. The ONE positional
        split (the flat<->named boundary) lives HERE; the geometry + combine then read ``.stations`` /
        ``.view_angle`` by name, never by index. ``xp`` is ``numpy`` / ``jax.numpy``."""
        if isinstance(design, StereoDesign):
            return design
        d = xp.asarray(design)
        d = d if d.ndim == 2 else d[None, :]
        ns = self.n_stations
        return StereoDesign(stations=d[:, :ns], view_angle=d[:, ns:ns + 1])

    def _expand(self, design, xp):
        """Per-detector ``StereoDesign(stations, view_angle)`` -- or its flat encoding -- expanded to
        per-layer ``(positions, angles)``. ``xp`` is ``numpy`` / ``jax.numpy``; batched on the leading
        axis (a 1-D flat input is squeezed back on output)."""
        single = not isinstance(design, StereoDesign) and xp.asarray(design).ndim == 1
        d = self._as_stereo_design(design, xp)
        positions = d.stations[:, self._layer_station] + self._layer_zoff  # (B, n_layers)
        angles = d.view_angle * self._layer_anglesign                       # (B, n_layers)
        if single:
            positions, angles = positions[0], angles[0]
        return positions, angles

    def _design_to_geometry(self, design):
        d = self._as_stereo_design(design, np)  # StereoDesign (batched); accepts a namedtuple or flat array
        positions, angles = self._expand(d, np)
        # Field is fixed at max_B (not a design dof).
        Bs = np.full(d.stations.shape[0], self.max_B, dtype=np.float32)
        return positions.astype(np.float32), angles.astype(np.float32), Bs

    # ------------------------------------------------------------------ #
    # Nominal <-> scaled: compact physical <-> [0, 1]  (B fixed, not a design dof)
    #
    # The stations are ORDERED and magnet-excluding, so station k's admissible window is coupled to
    # station k-1 and the "each coordinate on its own range" rule does not hold literally here: u_k is
    # the fraction of the REMAINING admissible window ``_station_lo_hi(k, prev_z)``, not of the single
    # global pair in ``design_bounds()['stations']``. The loop is identical in both directions and to
    # the quantile version it replaces; only the per-window transform is now affine.
    # ------------------------------------------------------------------ #
    def _station_window(self, k, prev_z, jnp):
        """Station ``k``'s admissible ``(lo, hi)``, with a collapsed-or-inverted window widened to a
        positive ``1e-3``. The guard is needed in BOTH directions: forward it stops a /0, and backward
        a negative ``(hi - lo)`` would make z DECREASE in u and leave the box entirely -- silently
        breaking the ordering the sequential window exists to enforce (the quantile map it replaces
        still produced a finite, if meaningless, z)."""
        lo, hi = self._station_lo_hi(k, prev_z)
        return lo, lo + jnp.maximum(hi - lo, 1e-3)

    def _to_scaled_flat(self, design):
        import jax.numpy as jnp

        d = jnp.asarray(design, jnp.float32)
        ns = self.n_stations
        # Station z's: invert the sequential coupled windows. We have the physical z's, so each
        # station's (lo, hi) follows from the previous station's z directly.
        us, prev = [], None
        for k in range(ns):
            if k == self.n_stations_upstream:
                prev = None  # new side downstream of the magnet
            z_k = d[..., k]
            lo, hi = self._station_window(k, prev, jnp)
            us.append((z_k - lo) / (hi - lo))
            prev = z_k
        z_u = jnp.stack(us, axis=-1)
        a_u = scale(d[..., ns : ns + 1], self.stereo_bound)
        return jnp.concatenate([z_u, a_u], axis=-1)

    def _to_nominal_flat(self, design_scaled):
        import jax.numpy as jnp

        u = jnp.asarray(design_scaled, jnp.float32)
        ns = self.n_stations
        # Station z's: un-scale sequentially so the window for each station depends on the previous
        # station's z -> ordered, non-overlapping, magnet-excluding by construction. ``u_k = 0`` is a
        # reachable hard corner (station k packed one pitch behind k-1); it is admissible, where the
        # quantile map only approached it as theta -> -inf.
        zs, prev = [], None
        for k in range(ns):
            if k == self.n_stations_upstream:
                prev = None  # new side downstream of the magnet
            lo, hi = self._station_window(k, prev, jnp)
            z_k = lo + u[..., k] * (hi - lo)
            zs.append(z_k)
            prev = z_k
        z_d = jnp.stack(zs, axis=-1)
        a_d = unscale(u[..., ns : ns + 1], self.stereo_bound)
        return jnp.concatenate([z_d, a_d], axis=-1)


    def _scaled_to_layer_geometry(self, design_scaled):
        import jax.numpy as jnp

        phys = self._to_nominal_flat(design_scaled)  # (B, ns+1) flat physical
        positions, angles = self._expand(phys, jnp)  # (B, n_layers) each
        # Field is fixed at max_B (not a design dof).
        B_field = jnp.full((phys.shape[0],), self.max_B, dtype=jnp.float32)
        return positions, angles, B_field
