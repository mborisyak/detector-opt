"""Alpha-resnet over a 1-D strip: depthwise-separable convolutions along the STRAW axis, layers as
channels.

For :class:`detopt.detector.StereoStrip`, which emits ``(B, n_straws, n_layers [+ D])``. One spatial
axis, so every convolution is 1-D and the channel count is the layer count.

    block b: X = channel-wise 1 ( depthwise K, stride S )    in -> out channels, strip / S
             X += alpha * [ channel-wise 1 ( depthwise K ) ]   x units, shape preserving
    head:    channel-wise 1  (widths[-1] -> prod(target_shape)) at the single pixel, reshaped to the target

THE FACTORISATION. A depthwise convolution is channel-INDEPENDENT: ``feature_group_count = channels``,
so each layer's strip is filtered on its own and nothing mixes across layers. The width-1 convolution
that follows mixes channels at a single pixel and does no spatial work. Together they are the
depthwise-separable factorisation -- ``C*K + C*C'`` parameters where a dense 1-D conv costs
``C*C'*K`` -- and here it separates "what does this layer's strip look like" from "how do the layers
combine at this straw".

ALPHA. Each residual branch is scaled by a PER-CHANNEL parameter initialised to ZERO, so the stack is
the identity map at initialisation and depth costs nothing in conditioning; the optimiser decides how
much of each branch to switch on. Same device and same reason as
:mod:`detopt.nn.alpha_conv_regressor`, which is the 2-D sibling.

⚠️ EVERY CHANNEL-WISE OPERATION IS A LINEAR LAYER, NOT A WIDTH-1 CONVOLUTION. On channels-last data
the two compute the same map, but the convolution is dispatched to cuDNN, and cuDNN has NO valid
algorithm for a width-1 kernel over a strip of extent 1 -- which is exactly what the last block
produces. MEASURED: a first submission of 25 jobs died 25/25 inside the first training step with
``INTERNAL: Autotuning failed for HLO ... No valid config found!`` on shapes ``f32[1,12,1]`` and
``f32[16,16,1]``, across 7 distinct hosts including ones where other jobs succeeded the same day. It
is the shapes, not the nodes. A depthwise clipped to width 1 is a per-channel scale and is written as
one for the same reason.

EVERY KERNEL IS CLIPPED TO THE STRIP IT ACTS ON. The strip shrinks fast -- at stride 5 it runs
316 -> 64 -> 13 -> 3 -> 1 -- so the last block's width-9 kernel would be mostly padding over a 3-wide
input, and its residual unit would convolve a single pixel. Each block therefore takes
``min(kernel_size, positions)``: the downsample uses its INPUT extent, the residual units their
OUTPUT extent. Nothing is padded that has no data behind it.

THE STRIDE LIVES IN THE DEPTHWISE, WHICH IS WHAT MAKES THIS SAFE ON SPARSE DATA. Each block reduces
the strip with a width-``K`` depthwise at stride ``S``, so every pixel is inside some kernel window
before the reduction. Putting the stride in the width-1 projection instead would SUBSAMPLE -- one
pixel in every ``S``, the rest never looked at -- and this grid is 0.4% occupied, ~41 hits per event
over 10112 cells, so MEASURED, a bare stride-3 subsample of the raw strip keeps only 35% of them. The
residual units that follow are stride 1 and shape preserving, which is what lets their alphas start
at zero and leave each block an exact identity beyond its downsample.

THE HEAD IS CHANNEL-WISE AND THE TARGET IS RESHAPED, NOT FLAT. This detector's target is a structure --
``vertex``, ``p1``, ``p2``, each ``(3,)`` -- which flattens to ``(3, 3)`` rather than to a vector, so
the head emits ``prod(target_shape)`` channels at the single pixel and reshapes. Taking
``target_shape[0]`` instead, as the image regressors do for a flat logit target, would emit 3 numbers
where the task wants 9. The
constructor RAISES if the block count and the stride do not leave exactly one position, because a
channel-wise head has no spatial extent to reduce and silently averaging one away would hide a
mis-specified stack.
"""

from typing import Sequence

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape

__all__ = ["StripRegressor"]


class DepthwiseSeparable(nnx.Module):
  """``depthwise(kernel, stride) -> channel-wise``, the residual branch of a block.

  The channel-wise half is an ``nnx.Linear``, not a width-1 ``nnx.Conv``. On channels-last data the
  two are the same map, but the convolution goes through cuDNN and a width-1 kernel over a 1-pixel
  strip has no valid cuDNN algorithm -- see the module docstring. A depthwise of width 1 is likewise
  a per-channel scale and is expressed as one."""

  def __init__(self, channels: int, kernel_size: int, stride: int, p_dropout: float | None, *, rngs: nnx.Rngs):
    self.channels = int(channels)
    self.depthwise = nnx.Conv(
      self.channels, self.channels, kernel_size=(int(kernel_size), ), strides=(int(stride), ), padding='SAME',
      feature_group_count=self.channels, rngs=rngs
    ) if int(kernel_size) > 1 else None
    self.scale = None if int(kernel_size) > 1 else nnx.Param(jnp.ones((self.channels, )))
    self.pointwise = nnx.Linear(self.channels, self.channels, rngs=rngs)
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None

  def __call__(self, h, *, deterministic: bool = True, rngs=None):
    if self.dropout is not None:
      h = self.dropout(h, deterministic=deterministic, rngs=rngs)
    spatial = self.depthwise(jax.nn.celu(h)) if self.depthwise is not None else self.scale.value * jax.nn.celu(h)
    return self.pointwise(jax.nn.celu(spatial))


class StripBlock(nnx.Module):
  """``depthwise K stride S -> channel-wise 1 stride 1`` (in -> out channels, strip / S), then
  ``units`` repetitions of ``X += alpha * channel-wise 1 ( depthwise K )`` at constant shape."""

  def __init__(
    self, in_channels: int, out_channels: int, kernel_size: int, stride: int, units: int,
    p_dropout: float | None, *, rngs: nnx.Rngs, positions_in: int, positions_out: int
  ):
    down_kernel = min(int(kernel_size), int(positions_in))
    unit_kernel = min(int(kernel_size), int(positions_out))
    self.down = nnx.Conv(
      int(in_channels), int(in_channels), kernel_size=(down_kernel, ), strides=(int(stride), ),
      padding='SAME', feature_group_count=int(in_channels), rngs=rngs
    )
    self.project = nnx.Linear(int(in_channels), int(out_channels), rngs=rngs)
    self.units = nnx.List([
      DepthwiseSeparable(int(out_channels), unit_kernel, 1, p_dropout, rngs=rngs) for _ in range(int(units))
    ])
    self.alphas = nnx.List([nnx.Param(jnp.zeros((int(out_channels), ))) for _ in range(int(units))])

  def __call__(self, h, *, deterministic: bool = True, rngs=None):
    h = self.project(self.down(h))
    for unit, alpha in zip(self.units, self.alphas):
      h = h + alpha.value * unit(h, deterministic=deterministic, rngs=rngs)
    return h


class StripRegressor(Model):
  """``(features, mask) -> target`` over a 1-D strip.

  Parameters
  ----------
  channels : one width per block; the strip is divided by ``stride`` at each.
  kernel_size : width of every depthwise kernel.
  stride : spatial reduction per block.
  p_dropout : dropout inside each residual branch, inert until the alphas leave zero.
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape,
    channels: Sequence[int] = (32, 24, 16, 12), kernel_size: int = 9, stride: int = 5, units: int = 3,
    p_dropout: float | None = None, *, rngs: nnx.Rngs
  ):
    if len(input_shape) != 2:
      raise ValueError(f'expected a 1-D strip shape (positions, channels), got {tuple(input_shape)}')
    widths = [int(c) for c in channels]
    if len(widths) < 1:
      raise ValueError('channels must name at least one block width')
    self.rngs = rngs
    self.n_channels_in = int(input_shape[-1])
    self.target_shape = tuple(int(d) for d in target_shape)
    self.target_dim = int(np.prod(self.target_shape))
    extents, positions = [], int(input_shape[0])
    for _ in widths:
      nxt = -(-positions // int(stride))
      extents.append((positions, nxt))
      positions = nxt
    self.blocks = nnx.List([
      StripBlock(
        widths[i - 1] if i else self.n_channels_in, widths[i], kernel_size, stride, units, p_dropout, rngs=rngs,
        positions_in=extents[i][0], positions_out=extents[i][1]
      ) for i in range(len(widths))
    ])
    if positions != 1:
      raise ValueError(
        f'the stack leaves a strip {positions} wide, not 1: {input_shape[0]} at stride {stride} over '
        f'{len(widths)} blocks. The head is a CHANNEL-WISE map and has no spatial extent to reduce, so '
        f'the block count and the stride must drive the strip to a single pixel.'
      )
    self.output = nnx.Linear(widths[-1], self.target_dim, rngs=rngs)

  def ensemble(self) -> int | None:
    return None

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    """``features (..., positions, C)`` -> ``(..., target_dim)``. ``mask`` is ignored: the strip is a
    DENSE grid with no absent elements -- an empty straw is a value in it, not a missing row."""
    h = features
    for block in self.blocks:
      h = block(h, deterministic=deterministic, rngs=rngs)
    flat = jnp.squeeze(self.output(jax.nn.celu(h)), axis=-2)
    return jnp.reshape(flat, flat.shape[:-1] + self.target_shape)
