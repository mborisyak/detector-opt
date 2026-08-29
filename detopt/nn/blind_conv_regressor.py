"""The alpha-conv stack fed the IMAGE ONLY: the design overlay is withheld.

The mnist window detector's `combine` emits `(rows, columns, 2)` -- channel 0 the image, channel 1 the
binary window whose extent IS the design. Every other conv architecture here reads both. This one
slices channel 0 and builds an otherwise bit-identical `AlphaConvRegressor` over a single input
channel, so the network is never told which window produced the image.

WHAT IT CONTROLS FOR. The image is already windowed -- what falls outside is zeroed -- so a
design-blind network is not blind to the design's EFFECT, only to its DECLARATION. The comparison
therefore separates two things that the conditioned architectures confound: how much a network gains
from being told the window, against how much it can infer from which pixels survived. A window over
dark pixels is the case where the two differ most, because there the zeroing leaves no trace.

Composition, not subclassing: this module OWNS an `AlphaConvRegressor` built for one input channel
rather than overriding any of its behaviour, so the two stay identical by construction and no concrete
method is replaced.

`ensemble()` returns `None` and there is no `n_models` argument, as in the stack it controls for.
"""

from collections.abc import Sequence

from flax import nnx

from .alpha_conv_regressor import AlphaConvRegressor
from .common import Model, Shape

__all__ = ['BlindConvRegressor']

IMAGE_CHANNELS = 1


class BlindConvRegressor(Model):
  """``(features, mask) -> logits`` over the image channel alone, the design overlay discarded.

  Parameters are those of :class:`AlphaConvRegressor` and are forwarded unchanged; only the input
  channel count differs, because the window channel never reaches the network.
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, channels: Sequence[int] = (16, 32, 64),
    blocks: int = 2, kernel_size: int = 3, p_dropout: float | None = None, *, rngs: nnx.Rngs
  ):
    if len(input_shape) != 3:
      raise ValueError(f'expected a channels-last image shape (rows, columns, channels), got {tuple(input_shape)}')
    if int(input_shape[-1]) < IMAGE_CHANNELS + 1:
      raise ValueError(
        f'expected at least {IMAGE_CHANNELS + 1} channels so there is a design overlay to withhold, '
        f'got {int(input_shape[-1])}'
      )
    self.rngs = rngs
    self.n_channels_in = int(input_shape[-1])
    self.target_dim = int(target_shape[0])
    image_shape = tuple(input_shape[:-1]) + (IMAGE_CHANNELS, )
    self.body = AlphaConvRegressor(
      image_shape, target_shape, ground_truth_shape, channels=channels, blocks=blocks, kernel_size=kernel_size,
      p_dropout=p_dropout, rngs=rngs
    )

  def ensemble(self) -> int | None:
    return None

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    """``features (..., rows, columns, C)`` -> ``(..., target_dim)`` logits, reading channel 0 only."""
    return self.body(features[..., :IMAGE_CHANNELS], mask, deterministic=deterministic, rngs=rngs)
