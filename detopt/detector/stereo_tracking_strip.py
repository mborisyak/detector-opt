"""Stereo daughter-tracking detector that emits the TDC grid as a 1-D IMAGE along the STRAW axis, with
the layers as CHANNELS.

A sibling of :class:`StereoImage` under the shared :class:`StereoLayerGrid` base -- same TDC grid, same
scatter, only the combine differs. Where the image emits ``(B, n_layers, n_straws, 6)`` with layer and
straw both spatial, this emits

    (B, n_straws, n_layers)          blind
    (B, n_straws, n_layers + D)      design revealed, D = design_dim()

so a 1-D convolution runs along the STRAW axis and every layer is a separate channel of the same
pixel. That is the natural form for a depthwise-separable stack -- depthwise width-5 over straws,
then a width-1 conv mixing layers at each pixel.

⚠️ WHAT THIS REPRESENTATION THROWS AWAY, and it is the point of comparing it. Collapsing the layers
into channels destroys the layer AXIS, so the per-layer geometry :class:`StereoImage` carries as
channels -- ``station_z``, ``view``, ``angle``, ``y_offset``, each varying ALONG that axis -- has
nowhere to live. A per-layer scalar cannot be a channel when the layers ARE the channels. What
survives is the reading itself plus, when revealed, the design as whole-image constants.

THE DESIGN ENTERS AS CONSTANT CHANNELS. ``design_scaled`` is ``D`` scalars broadcast along the straw
axis, appended after the layer channels. They are constant per image, so the depthwise stage cannot
use them and only the width-1 channel mixing can -- which is the honest encoding here, not a
limitation to work around: the design is a property of the whole event, not of a straw.

WITH THE DESIGN WITHHELD -- ``reveal_design=False`` or ``design_scaled=None``, treated alike -- the
trailing ``D`` channels are simply absent and the layer channels are untouched, so withholding is a
clean channel drop and the two conditions differ in information rather than in representation.

⚠️ ABSENCE IS ``TDC = -1`` AND THAT SENTINEL IS IN BAND. The grid writes ``clip(tdc, 0, tdc_clip) /
tdc_scale`` for a real hit and ``-1`` for an empty straw. A dense convolution has no mask to hide
behind, so ``-1`` is read as a value like any other. ``occupancy`` appends a second block of
``n_layers`` channels holding 1 where a straw fired and 0 where it did not, which makes absence
separable from any reading; it is OFF by default so the emitted image is the TDC grid alone.
"""

import jax.numpy as jnp

from .stereo_tracking_layerwise import StereoLayerGrid

__all__ = ["StereoStrip"]


class StereoStrip(StereoLayerGrid):
  """The TDC grid as ``(B, n_straws, n_layers)`` -- straws spatial, layers as channels."""

  def __init__(self, occupancy: bool = False, **kwargs):
    super().__init__(**kwargs)
    self.occupancy = bool(occupancy)

  def _channels(self, design: bool = True):
    return self.n_layers * (2 if self.occupancy else 1) + (self.design_dim() if design else 0)

  def combined_event_shape(self, design: bool = True):
    return (self.n_straws, self._channels(design))

  def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
    """Raw ``StrawEvent`` + SCALED design -> ``(B, n_straws, n_layers [+ n_layers] [+ D])``.

    REQUIRES ``mask``: the scatter has to know which of the padded hit slots are real."""
    grid, B = self._tdc_grid(event, mask)
    strip = jnp.swapaxes(grid, 1, 2)
    if self.occupancy:
      strip = jnp.concatenate([strip, (strip > -1.0).astype(jnp.float32)], axis=-1)
    if design_scaled is None or not reveal_design:
      return strip
    scaled = jnp.asarray(design_scaled, jnp.float32)
    if scaled.ndim == 1:
      scaled = jnp.broadcast_to(scaled[None, :], (B, scaled.shape[0]))
    constant = jnp.broadcast_to(scaled[:, None, :], (B, self.n_straws, scaled.shape[-1]))
    return jnp.concatenate([strip, constant], axis=-1)
