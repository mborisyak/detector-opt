"""StrawDetector with a fixed stereo-view layout and a compact design space.

Each station carries a fixed [0, +a, -a, 0] view layout (horizontal, +stereo,
-stereo, horizontal). The only optimizable design is:

    [ station_z (n_stations) , stereo_angle a ]

so dimension ``n_stations + 1`` instead of the base ``2*n_layers + 1``. The
magnetic field and the intra-station gaps are fixed. Station z-bounds exclude
the spectrometer magnet (centred at ``z0``, half-length ``magnet_half_cm``, from
the FairShip geometry), so the optimiser never drops a station inside the magnet:
upstream stations stay below ``z0 - magnet_half`` and downstream ones above
``z0 + magnet_half``.
"""

import numpy as np

from .straw import StrawDetector
from ..utils.encoding import normal_to_uniform_jax, uniform_to_normal_jax

__all__ = ["StereoStrawDetector", "stereo_design_array"]


def stereo_design_array(station_z, stereo_angle=0.0798, B=0.20):
    """Build a compact stereo design ``[station_z(n), stereo_angle, B]`` from a
    nominal layout. Lives OUTSIDE the detector (caller-owned initial design)."""
    return np.concatenate([np.asarray(station_z, np.float32), [float(stereo_angle), float(B)]]).astype(np.float32)


class StereoStrawDetector(StrawDetector):
    def __init__(
        self,
        # Station structure: the first n_stations_upstream sit upstream of the magnet,
        # the rest downstream -- this (not a nominal station_z) sets each station's bounds.
        n_stations_upstream: int = 2,
        n_stations_downstream: int = 2,
        n_layers_per_view: int = 2,
        n_straws_per_layer: int = 200,
        straw_pitch: float = 2.0,
        straw_length: float = 400.0,
        layer_y_offset: float = 1.0,
        # Stereo design scheme: intra-station view/layer z spacing + optimizable angle bound.
        layer_z_gap: float = 1.732,
        view_z_gap: float = 5.0,
        stereo_bound: tuple = (0.0, 0.2),
        magnet_half_cm: float = 140.0,
        # Physics / field
        max_B: float = 0.20,
        z0: float = 8957.0,
        B_sigma: float = 286.0,
        layer_bounds: tuple = (8000.0, 10000.0),
        dt=None,
        max_dt: float = 1.0,
        max_time: float = 200.0,
        max_particles: int = 5,
        material: str = "kapton",
        wall_thickness: float = 0.0036,
        delta_Tcut: float = 0.5,
        lambda_conv_cm=None,
        enable_decay: bool = False,
        noise_rate: float = 0.0,
        scatter_xX0: float = 0.0,
        decay_mean=None,
        decay_sigma=None,
        momentum_mean=None,
        momentum_sigma=None,
        data_dir=None,
        boundary_z=None,
    ):
        super().__init__(
            n_stations=int(n_stations_upstream) + int(n_stations_downstream),
            n_views_per_station=4,  # fixed [0, +a, -a, 0] stereo layout
            n_layers_per_view=n_layers_per_view,
            n_straws_per_layer=n_straws_per_layer,
            straw_pitch=straw_pitch,
            straw_length=straw_length,
            layer_y_offset=layer_y_offset,
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
            material=material,
            wall_thickness=wall_thickness,
            delta_Tcut=delta_Tcut,
            lambda_conv_cm=lambda_conv_cm,
            enable_decay=enable_decay,
            noise_rate=noise_rate,
            scatter_xX0=scatter_xX0,
            decay_mean=decay_mean,
            decay_sigma=decay_sigma,
            momentum_mean=momentum_mean,
            momentum_sigma=momentum_sigma,
            data_dir=data_dir,
            boundary_z=boundary_z,
        )
        self.n_stations_upstream = int(n_stations_upstream)
        self.stereo_bound = (float(stereo_bound[0]), float(stereo_bound[1]))
        self.magnet_half_cm = float(magnet_half_cm)
        self.layer_z_gap = float(layer_z_gap)
        self.view_z_gap = float(view_z_gap)

        # Fixed per-layer maps (global layer -> station, z-offset within station,
        # stereo-angle sign): ordering is station -> view -> layer-in-view.
        pattern = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)  # [0,+a,-a,0]
        st, zoff, sign = [], [], []
        for s in range(self.n_stations):
            for v in range(4):
                for l in range(self.n_layers_per_view):
                    st.append(s)
                    zoff.append(v * self.view_z_gap + l * self.layer_z_gap)
                    sign.append(pattern[v])
        self._layer_station = np.array(st, dtype=np.int64)
        self._layer_zoff = np.array(zoff, dtype=np.float32)
        self._layer_anglesign = np.array(sign, dtype=np.float32)

        # Magnet-aware per-station z bounds on the station base position (the span
        # of its views/layers is reserved so the whole station stays out of the magnet).
        # The first n_stations_upstream stations are upstream of the magnet; rest downstream.
        span = (4 - 1) * self.view_z_gap + (self.n_layers_per_view - 1) * self.layer_z_gap
        lo, hi = [], []
        for s in range(self.n_stations):
            if s < self.n_stations_upstream:  # upstream of the magnet
                lo.append(self.layer_bounds[0])
                hi.append(self.z0 - self.magnet_half_cm - span)
            else:  # downstream
                lo.append(self.z0 + self.magnet_half_cm)
                hi.append(self.layer_bounds[1] - span)
        self._station_lo = np.array(lo, dtype=np.float32)
        self._station_hi = np.array(hi, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Compact design space: [station_z(n_stations), stereo_angle]
    # ------------------------------------------------------------------ #
    def design_shape(self):
        # [station_z(n_stations), stereo_angle, B]
        return (self.n_stations + 2,)

    def _expand(self, design, xp):
        """Compact ``[station_z(n_stations), a]`` -> per-layer ``(positions, angles)``.

        ``xp`` is ``numpy`` or ``jax.numpy``; batched on the leading axis.
        """
        design = xp.asarray(design)
        single = design.ndim == 1
        if single:
            design = design[None, :]
        ns = self.n_stations
        station_z = design[:, :ns]  # (B, ns)
        alpha = design[:, ns : ns + 1]  # (B, 1)
        positions = station_z[:, self._layer_station] + self._layer_zoff  # (B, n_layers)
        angles = alpha * self._layer_anglesign  # (B, n_layers)
        if single:
            positions, angles = positions[0], angles[0]
        return positions, angles

    def _design_to_geometry(self, design):
        design = np.asarray(design, np.float32)
        d2 = design[None, :] if design.ndim == 1 else design
        positions, angles = self._expand(d2, np)
        n_batch, m = positions.shape
        widths = np.full((n_batch, m), self.layer_width, dtype=np.float32)
        heights = np.full((n_batch, m), self.layer_height, dtype=np.float32)
        Bs = d2[:, self.n_stations + 1].astype(np.float32)  # peak field is the last design dof
        return positions.astype(np.float32), angles.astype(np.float32), widths, heights, Bs

    # ------------------------------------------------------------------ #
    # Encode/decode: compact physical <-> N(0,1)  (B fixed, not a design dof)
    # ------------------------------------------------------------------ #
    def encode_design(self, design):
        import jax.numpy as jnp

        d = jnp.asarray(design, jnp.float32)
        ns = self.n_stations
        z_e = uniform_to_normal_jax(d[..., :ns], jnp.asarray(self._station_lo), jnp.asarray(self._station_hi))
        a_e = uniform_to_normal_jax(d[..., ns : ns + 1], self.stereo_bound[0], self.stereo_bound[1])
        b_e = uniform_to_normal_jax(d[..., ns + 1 : ns + 2], 0.0, self.max_B)
        return jnp.concatenate([z_e, a_e, b_e], axis=-1)

    def decode_design(self, encoded_design):
        import jax.numpy as jnp

        e = jnp.asarray(encoded_design, jnp.float32)
        ns = self.n_stations
        z_d = normal_to_uniform_jax(e[..., :ns], jnp.asarray(self._station_lo), jnp.asarray(self._station_hi))
        a_d = normal_to_uniform_jax(e[..., ns : ns + 1], self.stereo_bound[0], self.stereo_bound[1])
        b_d = normal_to_uniform_jax(e[..., ns + 1 : ns + 2], 0.0, self.max_B)
        return jnp.concatenate([z_d, a_d, b_d], axis=-1)

    def _decode_to_layer_geometry(self, d_enc):
        import jax.numpy as jnp

        phys = self.decode_design(d_enc)  # (B, ns+2) compact
        positions, angles = self._expand(phys, jnp)  # (B, n_layers) each
        B_field = phys[:, self.n_stations + 1]  # peak field per design
        return positions, angles, B_field
