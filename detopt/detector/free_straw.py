"""StrawDetector with the free per-layer design scheme.

The design is the full per-layer geometry directly:

    [ layer_z (n_layers) , layer_angle (n_layers) , B ]

so ``design_dim = 2*n_layers + 1``. Bounds: each ``layer_z`` in ``layer_bounds``,
each angle in ``angle_bounds``, ``B`` in ``[0, max_B]``. This is the concrete
design subclass registered as ``"straw"``; the base :class:`StrawDetector` carries
no design scheme.
"""

import numpy as np

from .straw import StrawDetector
from ..utils.encoding import normal_to_uniform_jax, uniform_to_normal_jax

__all__ = ["FreeStrawDetector", "free_design_array"]


def free_design_array(
    station_z,
    n_views_per_station=4,
    n_layers_per_view=2,
    view_angles=(0.0, 0.0798, -0.0798, 0.0),
    view_z_gap=5.0,
    layer_z_gap=1.732,
    B=0.20,
):
    """Build a free per-layer design ``[layer_z(n), layer_angle(n), B]`` from a
    nominal station layout. Lives OUTSIDE the detector: the initial/nominal design
    is the caller's concern (diagnostics, baselines), not detector state."""
    positions, angles = [], []
    for z_station in station_z:
        for v in range(n_views_per_station):
            base = z_station + v * view_z_gap
            ang = view_angles[v] if v < len(view_angles) else view_angles[-1]
            for _ in range(n_layers_per_view):
                positions.append(base)
                angles.append(ang)
                base += layer_z_gap
    return np.concatenate([positions, angles, [float(B)]]).astype(np.float32)


class FreeStrawDetector(StrawDetector):
    # ------------------------------------------------------------------ #
    # Design space: [layer_z(n), layer_angle(n), B]
    # ------------------------------------------------------------------ #
    def design_shape(self):
        # positions + angles + magnetic field strength
        return (2 * self.n_layers + 1,)

    def _design_to_geometry(self, design):
        """Physical ``[positions(n), angles(n), B]`` -> per-layer
        ``(layers, angles, widths, heights, Bs)`` for the C solver."""
        design = np.asarray(design, dtype=np.float32)
        if design.ndim == 1:
            design = design[None, :]
        n_batch = design.shape[0]
        m = self.n_layers

        layers = design[:, :m].astype(np.float32)
        angles = design[:, m : 2 * m].astype(np.float32)
        Bs = design[:, 2 * m].astype(np.float32)
        widths = np.full((n_batch, m), self.layer_width, dtype=np.float32)
        heights = np.full((n_batch, m), self.layer_height, dtype=np.float32)
        return layers, angles, widths, heights, Bs

    # ------------------------------------------------------------------ #
    # Encode/decode (constrained <-> N(0,1)), differentiable
    # ------------------------------------------------------------------ #
    def encode_design(self, design):
        import jax.numpy as jnp

        design = jnp.asarray(design, dtype=jnp.float32)
        n = self.n_layers
        pos_e = uniform_to_normal_jax(design[..., :n], *self.layer_bounds)
        ang_e = uniform_to_normal_jax(design[..., n : 2 * n], *self.angle_bounds)
        B_e = uniform_to_normal_jax(design[..., 2 * n : 2 * n + 1], 0.0, self.max_B)
        return jnp.concatenate([pos_e, ang_e, B_e], axis=-1)

    def decode_design(self, encoded_design):
        import jax.numpy as jnp

        enc = jnp.asarray(encoded_design, dtype=jnp.float32)
        n = self.n_layers
        pos_d = normal_to_uniform_jax(enc[..., :n], *self.layer_bounds)
        ang_d = normal_to_uniform_jax(enc[..., n : 2 * n], *self.angle_bounds)
        B_d = normal_to_uniform_jax(enc[..., 2 * n : 2 * n + 1], 0.0, self.max_B)
        return jnp.concatenate([pos_d, ang_d, B_d], axis=-1)

    def _decode_to_layer_geometry(self, d_enc):
        """Encoded design -> per-layer ``(positions(B,n), angles(B,n), B(B))``.
        The free design *is* per-layer, so decode + split."""
        phys = self.decode_design(d_enc)
        n = self.n_layers
        return phys[:, :n], phys[:, n : 2 * n], phys[:, 2 * n]
