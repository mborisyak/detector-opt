"""Stereo daughter-tracking detector whose SET ELEMENT is a LAYER carrying its own identity.

A sibling of :class:`StereoLayerWise` and :class:`StereoStrip` under the shared
:class:`StereoLayerGrid` base -- same TDC grid, same scatter, only the combine differs. Every element
is one layer's whole row of straw readings, prefixed by WHO that layer is:

    (B, n_layers, n_layers + n_straws)   blind    -- identity as a ONE-HOT index
    (B, n_layers, 2 + n_straws)          revealed -- identity as (station_z_norm, angle_norm)

⚠️ THIS IS A SUBSTITUTION, NOT A COLUMN DROP, and that is deliberate -- unlike ``stereo_strip`` and
``stereo_address_design``, where withholding removes trailing columns and leaves the measurement
block byte-identical. Here both conditions name the element; they differ in WHAT the name means. Blind,
a layer is an anonymous token and the network must learn a separate read-out per slot. Revealed, the
layer announces where it sits and how it is tilted, so the network can share structure between layers
that happen to be similar. The TDC block is identical either way, so the measurement is untouched and
only the identity changes.

The one-hot is the right blind encoding rather than a normalized index: a scalar index asserts an
ORDER and a METRIC over layers ("layer 7 is close to layer 8, and twice layer 3.5"), which is
geometry the blind condition is supposed to withhold. One-hot asserts only distinctness.

``station_z_norm`` and ``angle_norm`` are exactly the two design-dependent per-layer quantities
:meth:`StereoLayerGrid._layer_positions` computes, so the revealed condition is differentiable in the
design. The fixed index features that class also returns (``view_norm``, ``layer_norm``,
``y_offset_norm``) are NOT carried: they are address, not design, and the one-hot already says which
layer this is.

The per-element mask is all-ones ``(B, n_layers)`` from the base class -- every layer is always a
valid element, whether or not any straw in it fired.
"""

import jax.numpy as jnp

from .stereo_tracking_layerwise import StereoLayerGrid

__all__ = ["StereoLayerSet"]


class StereoLayerSet(StereoLayerGrid):
  """Layers as set elements, identified by a one-hot index (blind) or by (z, angle) (revealed)."""

  def combined_event_shape(self, design: bool = True):
    return (self.n_layers, (2 if design else self.n_layers) + self.n_straws)

  def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
    """Raw ``StrawEvent`` + SCALED design -> ``(B, n_layers, identity + n_straws)``.

        REQUIRES ``mask``: the scatter has to know which of the padded hit slots are real."""
    grid, B = self._tdc_grid(event, mask)
    if design_scaled is None or not reveal_design:
      identity = jnp.broadcast_to(jnp.eye(self.n_layers, dtype=jnp.float32)[None, :, :],
                                  (B, self.n_layers, self.n_layers))
    else:
      station_z_norm, _view, _layer, angle_norm, _y_offset = self._layer_positions(design_scaled, B)
      identity = jnp.stack([station_z_norm, angle_norm], axis=-1)
    return jnp.concatenate([identity, grid], axis=-1)
