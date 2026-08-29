"""Stereo daughter-tracking detector with a PER-HIT combine, for the continuous-convolutional regressor.

Same design space / daughter target / loss as :class:`Stereo4Feature` (it subclasses it); the combine
emits one feature vector per HIT (not per layer / per straw-grid):

    (B, M, 7)   = [station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm, straw_norm, tdc_norm]

``station_z_norm`` and ``angle_norm`` are design-dependent (the gradient flows through them); the rest
are the hit's address (view/layer/straw, the wire stagger) + its normalized TDC. ``element_mask`` is the
hit mask (default) -- the continuous-conv regressor treats each hit as a graph node.

WITH THE DESIGN WITHHELD those two design-dependent features are exactly what goes, leaving

    (B, M, 5)   = [view_norm, layer_norm, y_offset_norm, straw_norm, tdc_norm]

-- the hit's address and its reading, which is all this detector can say without knowing where the
stations sit or how the views are tilted. The measurement does not depend on the design, so
``design_scaled=None`` is honoured and treated exactly like ``reveal_design=False``.
"""

import jax.numpy as jnp

from .stereo_straw import StereoStrawDetector

__all__ = ["StereoHits"]


class StereoHits(StereoStrawDetector):
    def __init__(self, tdc_scale: float = 62.0, tdc_clip: float = 1.0e4, **kwargs):
        super().__init__(**kwargs)
        self.tdc_scale = float(tdc_scale)
        self.tdc_clip = float(tdc_clip)

    def combined_event_shape(self, design: bool = True):
        # 7 with the design; without it station_z and the stereo angle are unknowable -> 5
        return (self.max_hits_per_event, 7 if design else 5)

    def element_mask(self, event, mask):
        return mask  # element == hit (the continuous-conv regressor treats each hit as a graph node)

    def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
        """Raw ``StrawEvent`` + SCALED design -> per-HIT features ``(B, M, 7)``, or ``(B, M, 5)`` with
        the design withheld -- ``station_z_norm`` and ``angle_norm`` are the design and are dropped,
        the address and the TDC are not. (``mask`` accepted but not needed -- masked hits are gated by
        the regressor's hit mask.)"""
        station = jnp.asarray(event.station, jnp.int32)  # (B, M)
        view = jnp.asarray(event.view, jnp.int32)
        layer = jnp.asarray(event.layer, jnp.int32)
        straw = jnp.asarray(event.straw, jnp.int32)
        tdc = jnp.asarray(event.tdc, jnp.float32)
        B, M = station.shape
        per_station = self.n_views_per_station * self.n_layers_per_view
        g = jnp.clip(station * per_station + view * self.n_layers_per_view + layer, 0, self.n_layers - 1)  # (B,M)

        if design_scaled is None:
            d_scaled = jnp.zeros((B, self.design_dim()), jnp.float32)  # unused: reveal_design is False below
            reveal_design = False
        else:
            d_scaled = jnp.asarray(design_scaled, jnp.float32)
        if d_scaled.ndim == 1:
            d_scaled = jnp.broadcast_to(d_scaled[None, :], (B, d_scaled.shape[0]))
        phys = self._to_nominal_flat(d_scaled)  # (B, n_stations + 1) physical [station_z..., stereo_angle]
        z_hit = jnp.take_along_axis(phys[..., : self.n_stations], jnp.clip(station, 0, self.n_stations - 1), axis=1)
        z_mid = 0.5 * (self.layer_bounds[0] + self.layer_bounds[1])
        z_half = max(0.5 * (self.layer_bounds[1] - self.layer_bounds[0]), 1e-6)
        station_z_norm = (z_hit - z_mid) / z_half  # (B, M) design-dependent

        alpha = phys[..., self.n_stations]  # (B,)
        anglesign = jnp.asarray(self._layer_anglesign, jnp.float32)[g]  # (B, M) the hit's [0,+1,-1,0] view sign
        angle_bound = max(abs(self.stereo_bound[1]), 1e-6)
        angle_norm = (alpha[:, None] * anglesign) / angle_bound  # (B, M) design-dependent

        view_axis = (jnp.linspace(-1.0, 1.0, self.n_views_per_station, dtype=jnp.float32)
                     if self.n_views_per_station > 1 else jnp.zeros(1, jnp.float32))
        layer_axis = (jnp.linspace(-1.0, 1.0, self.n_layers_per_view, dtype=jnp.float32)
                      if self.n_layers_per_view > 1 else jnp.zeros(1, jnp.float32))
        view_norm = view_axis[jnp.clip(view, 0, self.n_views_per_station - 1)]  # (B, M)
        layer_norm = layer_axis[jnp.clip(layer, 0, self.n_layers_per_view - 1)]  # (B, M)

        y_offset_norm = jnp.where((layer & 1) == 1, 0.5 * self.layer_y_offset, -0.5 * self.layer_y_offset) \
            / self.straw_pitch  # (B, M) wire half-pitch stagger, +/-0.25
        straw_norm = (2.0 / max(self.n_straws - 1, 1)) * jnp.clip(straw, 0, self.n_straws - 1).astype(jnp.float32) - 1.0
        tdc_norm = jnp.clip(tdc, 0.0, self.tdc_clip) / self.tdc_scale  # (B, M) >= 0

        if not reveal_design:
            # station_z and angle are the design; view/layer/y_offset/straw/TDC are the hit's address.
            return jnp.stack([view_norm, layer_norm, y_offset_norm, straw_norm, tdc_norm], axis=-1)  # (B, M, 5)
        return jnp.stack([station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm, straw_norm, tdc_norm],
                         axis=-1)  # (B, M, 7)
