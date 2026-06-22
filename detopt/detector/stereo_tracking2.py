"""Stereo daughter-tracking detector whose per-hit features ALSO carry the hit's
discrete address as fine-grained continuous positions, on top of the continuous geometry.

Identical design space, daughter-kinematics target and loss as :class:`StereoTracking`
(it subclasses it); only :meth:`combine_encoded` is extended. Where the parent hands the network

    [TDC, norm(layer z), wire_y_left, wire_y_right]

this appends the hit's *position* within each level of the hierarchy, normalized to ``[-1, 1]``:

    [..., station_pos, view-in-station_pos, layer-in-view_pos, rbf(straw_pos)[n_rbf]]

The three positions are scalar normalized indices (``linspace(-1, 1, N)[idx]``); the straw is
given an ``n_rbf``-wide RADIAL-BASIS soft-encoding of its normalized centre instead -- centres
on ``linspace(-1, 1, n_rbf)`` with bandwidth ``sigma = 2 / (n_rbf - 1)`` -- so the network sees a
smooth, low-dimensional, locality-preserving straw position. The combined feature dim is
therefore ``4 + 3 + n_rbf``.
"""

import jax.numpy as jnp
import numpy as np

from .stereo_tracking import StereoTracking

__all__ = ["StereoTracking2"]


class StereoTracking2(StereoTracking):
    def __init__(self, n_rbf: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.n_rbf = int(n_rbf)
        # RBF indexing points + bandwidth, in the straw's normalized centre space [-1, 1].
        self._rbf_centers = jnp.linspace(-1.0, 1.0, self.n_rbf, dtype=jnp.float32)
        self._rbf_sigma = 2.0 / (self.n_rbf - 1)

    def combined_event_shape(self):
        # combine() -> [TDC, norm z, y_left, y_right] ++ pos(station, view, layer) ++ rbf(straw)
        return (self.max_hits_per_event, 4 + 3 + self.n_rbf)

    def _normalized_index(self, idx, n):
        """Map a 0-based hit index ``idx`` over ``n`` slots to its position in ``[-1, 1]``
        (``linspace(-1, 1, n)[idx]``); a single-slot level collapses to all-zeros."""
        if n <= 1:
            return jnp.zeros(idx.shape, jnp.float32)
        return (2.0 / (n - 1)) * idx.astype(jnp.float32) - 1.0

    def combine_encoded(self, event, encoded_design):
        """Parent continuous features ``[TDC, norm z, wire_y_left, wire_y_right]`` concatenated
        with the hit's address as fine-grained positions in ``[-1, 1]``:

          * ``station`` -- the station's physical z (the decoded design dof) normalized by the
            detector z-extent, ``2 (z_station - centre) / width`` (so it carries the actual,
            design-dependent station spacing, not a bare index; differentiable through the decode);
          * ``view-in-station`` / ``layer-in-view`` -- scalar normalized indices ``linspace(-1, 1, N)``;
          * ``straw`` -- an ``n_rbf``-wide RBF soft-encoding of its normalized centre.

        The discrete address is read straight off the raw ``StrawEvent`` int fields (no rounding
        needed). Masked/empty hits carry index 0 in every block; the hit ``mask`` is threaded
        separately, so that is harmless."""
        base = super().combine_encoded(event, encoded_design)  # (B, M, 4) continuous geometry features

        station = jnp.asarray(event.station, jnp.int32)
        view = jnp.asarray(event.view, jnp.int32)
        layer = jnp.asarray(event.layer, jnp.int32)
        straw = jnp.asarray(event.straw, jnp.int32)
        B, M = station.shape

        # Station: physical station-centre z (decoded design dof), normalized to ~[-1, 1] by the
        # detector z-extent -> 2 (z - centre) / width (same convention as the parent's norm_z).
        d_enc = jnp.asarray(encoded_design, jnp.float32)
        if d_enc.ndim == 1:
            d_enc = jnp.broadcast_to(d_enc[None, :], (B, d_enc.shape[0]))
        station_z_all = self._decode_flat(d_enc)[..., : self.n_stations]  # (B, n_stations) physical z
        z_station = jnp.take_along_axis(station_z_all, station, axis=1)  # (B, M)
        z_mid = 0.5 * (self.layer_bounds[0] + self.layer_bounds[1])
        z_half = max(0.5 * (self.layer_bounds[1] - self.layer_bounds[0]), 1e-6)
        station_pos = (z_station - z_mid) / z_half

        # View/layer: fine-grained position within their level of the hierarchy, normalized to [-1, 1].
        positions = jnp.stack(
            [
                station_pos,
                self._normalized_index(view, self.n_views_per_station),
                self._normalized_index(layer, self.n_layers_per_view),
            ],
            axis=-1,
        )  # (B, M, 3)

        # Straw: RBF soft-encoding of its normalized centre against fixed indexing points.
        straw_pos = self._normalized_index(straw, self.n_straws)  # (B, M) straw centre in [-1, 1]
        centers = self._rbf_centers
        rbf = jnp.exp(-0.5 * jnp.square(straw_pos[..., None] - centers) / (self._rbf_sigma ** 2))  # (B, M, n_rbf)

        return jnp.concatenate([base, positions, rbf], axis=-1)
