"""Stereo straw detector whose combine hands the network the raw HIT ADDRESS and, when the design is
revealed, the SCALED DESIGN appended as trailing per-hit columns.

    revealed  [TDC, station, view, layer, straw, *design]   (M, 5 + D)
    withheld  [TDC, station, view, layer, straw]            (M, 5)

WHAT THIS SEPARATES, and why it is not `stereo_tracking`. The four-feature combine RESOLVES the design
into each hit's own geometry -- ``norm_z`` is that layer's z, ``wire_y_left``/``wire_y_right`` its wire
endpoints -- so the design is present but ENTANGLED with the measurement, and a hit's geometry is
handed over pre-computed. Here the measurement is the bare address and the design is a separate block
of columns, so the network must combine them itself. Withholding is then a clean column drop rather
than a change of representation: the ADDRESS is identical in both cases and only the design block
goes, which is what makes revealed and withheld comparable on this detector and not on that one.

THE INDICES ARE NORMALISED TO ``[0, 1]`` (``straw.address_combine``), which is the scaled design's own
convention, so every column except ``TDC`` shares one scale; ``TDC`` keeps the standardisation the
four-feature combine uses.

THE DESIGN IS PER EVENT, BROADCAST OVER HITS. Every hit of an event was measured at the same design,
so the block is constant along the element axis. It still breaks nothing: the address varies per hit
and is what distinguishes them.
"""

import jax.numpy as jnp

from .stereo_straw import StereoStrawDetector
from .straw import N_ADDRESS_FEATURES, address_combine, address_shape

__all__ = ['StereoAddressDesign']


class StereoAddressDesign(StereoStrawDetector):
  """Stereo geometry + the ADDRESS combine with the scaled design as trailing columns."""

  def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
    address = address_combine(self, event, mask=mask)
    if design_scaled is None or not reveal_design:
      return address
    design = jnp.asarray(design_scaled, jnp.float32)
    if design.ndim == 1:
      design = jnp.broadcast_to(design[None, :], address.shape[:-2] + (design.shape[0], ))
    block = jnp.broadcast_to(design[..., None, :], address.shape[:-1] + (design.shape[-1], ))
    return jnp.concatenate([address, block], axis=-1)

  def combined_event_shape(self, design: bool = True):
    elements, features = address_shape(self)
    return (elements, features + (self.design_dim() if design else 0))

  def element_mask(self, event, mask):
    return mask
