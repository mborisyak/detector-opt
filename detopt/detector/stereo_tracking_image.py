"""Stereo daughter-tracking detector that emits the layer-wise data as an IMAGE, for a CNN.

A sibling of :class:`StereoLayerWise` under the shared :class:`StereoLayerGrid` base (same TDC
grid + per-layer positions); only the combine differs. Instead of ``(B, n_layers, 5 + n_straws)`` rows
it emits a channels-LAST image

    (B, n_layers, n_straws, 4)   channels = [station_z_norm, view_norm, straw_norm, TDC]

-- the natural input for the hierarchical CNN (:class:`detopt.nn.ConvRegressor`), whose straw / layer
/ view / station convolutions run over the station->view->layer-ordered layer axis and the straw axis.
The layer-within-view index is NOT a channel: that structure is captured by the layer-level conv.
``element_mask`` (all-ones over the layers) is inherited -- the CNN ignores the set-mask, since hit
absence is encoded as ``TDC = -1`` in the grid.
"""

import jax.numpy as jnp

from .stereo_tracking_layerwise import StereoLayerGrid

__all__ = ["StereoImage"]


class StereoImage(StereoLayerGrid):
    def combined_event_shape(self):
        # channels-last image: (n_layers, n_straws, 6) = [station_z, view, angle, y_offset, straw, TDC]
        return (self.n_layers, self.n_straws, 6)

    def combine_scaled(self, event, design_scaled, mask=None):
        """Raw ``StrawEvent`` + SCALED design -> image ``(B, n_layers, n_straws, 6)``. REQUIRES ``mask``.

        Per pixel ``(layer l, straw s)``: ``[station_z_norm[l], view_norm[l], angle_norm[l],
        y_offset_norm[l], straw_norm[s], TDC[l, s]]``. ``station_z_norm`` and ``angle_norm`` (the layer's
        stereo tilt) are design-dependent (the gradient flows through them); the y_offset/straw/TDC
        channels are not. (The layer-within-view index is NOT a channel -- it is captured by the
        layer-level conv.)"""
        grid, B = self._tdc_grid(event, mask)  # (B, n_layers, n_straws)
        station_z_norm, view_norm, _layer_norm, angle_norm, y_offset_norm = self._layer_positions(design_scaled, B)
        straw_norm = jnp.linspace(-1.0, 1.0, self.n_straws, dtype=jnp.float32)  # (n_straws,) fixed straw position

        shape = (B, self.n_layers, self.n_straws)
        station_ch = jnp.broadcast_to(station_z_norm[:, :, None], shape)  # per-layer, over straws
        view_ch = jnp.broadcast_to(view_norm[:, :, None], shape)
        angle_ch = jnp.broadcast_to(angle_norm[:, :, None], shape)
        y_offset_ch = jnp.broadcast_to(y_offset_norm[:, :, None], shape)
        straw_ch = jnp.broadcast_to(straw_norm[None, None, :], shape)  # per-straw, over layers
        return jnp.stack([station_ch, view_ch, angle_ch, y_offset_ch, straw_ch, grid], axis=-1)  # (B,n_layers,n_straws,6)
