"""Stereo daughter-tracking detectors whose set element is a LAYER, not a hit.

``StereoLayerGrid`` is the shared base for the per-LAYER combines: it scatters the fired straws into a
dense per-layer TDC grid and exposes the per-layer position features (station-z, view, layer, stereo
angle, wire-stagger). Its two combine LEAVES lay that grid out differently:

- :class:`StereoLayerWise` -- a SET of ``(B, n_layers, 5 + n_straws)`` per-layer rows
  ``[station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm]`` ++ the per-layer TDC grid;
- :class:`StereoImage` -- a channels-last IMAGE (see ``stereo_tracking_image.py``).

The straw TDCs are scattered into a dense per-layer grid: a fired straw holds
``clip(tdc, 0, tdc_clip) / tdc_scale`` (>= 0), an unfired straw holds ``-1``. The position features'
station-z / stereo angle are design-dependent (differentiable); the TDC grid is not. The per-element
mask is all-ones ``(B, n_layers)`` -- every layer is always a valid element -- so a plain
:class:`SetRegressor` pools over the 32 layers and the causal ``PredictiveSetRegressor`` streams them.
"""

import jax
import jax.numpy as jnp

from .stereo_straw import StereoStrawDetector

__all__ = ["StereoLayerGrid", "StereoLayerWise"]


class StereoLayerGrid(StereoStrawDetector):
    """Shared base for the per-LAYER combine detectors: the TDC-grid scatter + per-layer position
    features + the all-valid per-layer element mask. The combine itself is ABSTRACT (the leaves lay the
    grid out as a set row vs an image)."""

    def __init__(self, tdc_scale: float = 62.0, tdc_clip: float = 1.0e4, **kwargs):
        super().__init__(**kwargs)
        self.tdc_scale = float(tdc_scale)  # fired straw TDC -> tdc/tdc_scale; ~our-sim TDC median (ns)
        self.tdc_clip = float(tdc_clip)  # clip pathological TDCs before scaling (FairShip has 1e16 outliers)

    def element_mask(self, event, mask):
        # Every layer is always a valid element (empty layers are an all -1 TDC row, not masked out).
        B = jnp.asarray(event.station).shape[0]
        return jnp.ones((B, self.n_layers), jnp.int32)

    def _tdc_grid(self, event, mask):
        """Scatter the fired straws into a dense ``(B, n_layers, n_straws)`` TDC grid (default ``-1``).

        REQUIRES ``mask``: padded ``StrawEvent`` slots carry index 0 / tdc 0 and are indistinguishable
        from a real hit at straw 0 of layer 0 by value, so the scatter must know which hits are real.
        Real hit -> ``clip(tdc, 0, tdc_clip)/tdc_scale`` (>= 0); padded hit -> ``-1`` (the floor). A
        masked scatter-MAX is order-independent: a real hit always beats the floor / a padded write.
        Returns ``(grid, B)``."""
        if mask is None:
            raise ValueError(f"{type(self).__name__}.combine_scaled requires the per-hit mask")
        station = jnp.asarray(event.station, jnp.int32)  # (B, M)
        view = jnp.asarray(event.view, jnp.int32)
        layer = jnp.asarray(event.layer, jnp.int32)
        straw = jnp.asarray(event.straw, jnp.int32)
        tdc = jnp.asarray(event.tdc, jnp.float32)
        m = jnp.asarray(mask, jnp.float32)  # (B, M) 1 = real hit
        B, M = station.shape
        per_station = self.n_views_per_station * self.n_layers_per_view
        g = jnp.clip(station * per_station + view * self.n_layers_per_view + layer, 0, self.n_layers - 1)  # (B,M)
        s_col = jnp.clip(straw, 0, self.n_straws - 1)  # (B, M)
        tdc_enc = jnp.clip(tdc, 0.0, self.tdc_clip) / self.tdc_scale  # (B, M) >= 0
        write_val = jnp.where(m > 0, tdc_enc, -1.0)  # (B, M)
        bidx = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32)[:, None], (B, M))
        grid = jnp.full((B, self.n_layers, self.n_straws), -1.0, jnp.float32)
        grid = grid.at[bidx, g, s_col].max(write_val, mode="drop")  # (B, n_layers, n_straws)
        return grid, B

    def _layer_positions(self, design_scaled, B):
        """Per-layer ``(station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm)``, each
        ``(B, n_layers)`` in ~``[-1, 1]``.

        ``station_z_norm = (z_layer - centre)/half`` (physical station z) and ``angle_norm = alpha *
        anglesign / stereo_bound`` (the layer's stereo tilt) are design-dependent (differentiable);
        ``view_norm`` / ``layer_norm`` are fixed normalized indices ``linspace(-1, 1, N)``;
        ``y_offset_norm = y_stagger / straw_pitch`` (= +/-0.25) is the fixed wire half-pitch stagger."""
        per_station = self.n_views_per_station * self.n_layers_per_view
        gl = jnp.arange(self.n_layers, dtype=jnp.int32)  # (n_layers,)
        layer_station = gl // per_station  # (n_layers,) global layer -> station
        within = gl % per_station
        layer_view = within // self.n_layers_per_view  # (n_layers,) view-in-station
        layer_lpv = within % self.n_layers_per_view  # (n_layers,) layer-in-view

        d_scaled = jnp.asarray(design_scaled, jnp.float32)
        if d_scaled.ndim == 1:
            d_scaled = jnp.broadcast_to(d_scaled[None, :], (B, d_scaled.shape[0]))
        phys = self._to_nominal_flat(d_scaled)  # (B, n_stations + 1) physical [station_z..., stereo_angle]
        z_layer = phys[..., : self.n_stations][:, layer_station]  # (B, n_layers) each layer's station z
        z_mid = 0.5 * (self.layer_bounds[0] + self.layer_bounds[1])
        z_half = max(0.5 * (self.layer_bounds[1] - self.layer_bounds[0]), 1e-6)
        station_z_norm = (z_layer - z_mid) / z_half  # (B, n_layers)

        # Per-layer stereo angle: decoded ``alpha`` times the view's [0,+1,-1,0] sign, normalized by the
        # angle bound. Design-dependent (gradient flows through alpha) -> restores the angle design dof.
        alpha = phys[..., self.n_stations]  # (B,)
        anglesign = jnp.asarray(self._layer_anglesign, jnp.float32)  # (n_layers,)
        angle_bound = max(abs(self.stereo_bound[1]), 1e-6)
        angle_norm = (alpha[:, None] * anglesign[None, :]) / angle_bound  # (B, n_layers) ~[-1, 1]

        view_axis = (jnp.linspace(-1.0, 1.0, self.n_views_per_station, dtype=jnp.float32)
                     if self.n_views_per_station > 1 else jnp.zeros(1, jnp.float32))
        layer_axis = (jnp.linspace(-1.0, 1.0, self.n_layers_per_view, dtype=jnp.float32)
                      if self.n_layers_per_view > 1 else jnp.zeros(1, jnp.float32))
        view_norm = jnp.broadcast_to(view_axis[layer_view][None, :], (B, self.n_layers))
        layer_norm = jnp.broadcast_to(layer_axis[layer_lpv][None, :], (B, self.n_layers))

        # Per-layer wire half-pitch stagger (the two layers of a view are offset by +/-pitch/4 in y),
        # normalized to pitch units (= +/-0.25). Same parity rule the C solver / Stereo4Feature use.
        y_stagger = jnp.where((layer_lpv & 1) == 1, 0.5 * self.layer_y_offset, -0.5 * self.layer_y_offset)  # (n_layers,)
        y_offset_norm = jnp.broadcast_to((y_stagger / self.straw_pitch)[None, :], (B, self.n_layers))
        return station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm


class StereoLayerWise(StereoLayerGrid):
    """Per-LAYER SET combine: ``(B, n_layers, 5 + n_straws)`` rows for a :class:`SetRegressor`."""

    def combined_event_shape(self):
        # element axis = global layers; per-layer feature = [station_z, view, layer, angle, y_offset] ++ TDC grid
        return (self.n_layers, 5 + self.n_straws)

    def combine_scaled(self, event, design_scaled, mask=None):
        """Raw ``StrawEvent`` + SCALED design -> per-LAYER features ``(B, n_layers, 5 + n_straws)``:
        ``[station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm]`` ++ the per-layer TDC grid.
        REQUIRES ``mask``."""
        grid, B = self._tdc_grid(event, mask)
        station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm = self._layer_positions(design_scaled, B)
        positions = jnp.stack([station_z_norm, view_norm, layer_norm, angle_norm, y_offset_norm], axis=-1)  # (B,n_layers,5)
        return jnp.concatenate([positions, grid], axis=-1)  # (B, n_layers, 5 + n_straws)
