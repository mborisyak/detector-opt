"""The :mod:`detopt.nn.alpha_conv_regressor` stack with EXPLICIT per-block design conditioning.

Same stem, same stage widths, same number of residual units, same ``celu``, same zero-initialised
per-channel ``alpha``, same strided ``celu -> Conv`` between stages, same pooled linear head. ONE
thing differs: the design enters every residual unit as NUMBERS, not only as the window mask at the
input. Semi-hypernetwork rather than hypernetwork -- the design conditions the ACTIVATIONS through a
learned embedding, it does not generate the weights.

    embedding = Linear(celu(Linear(design)))                       ``(..., k)``
    block:  m = Conv1x1([h, broadcast(embedding), resample(image)])  ``C + k + 2 -> C``
            h <- h + alpha * Conv(celu(Dropout(m)))

so the block BODY keeps the width and the resolution it had, the skip is still the untouched block
input, and ``alpha = 0`` at initialisation still makes the whole unit the identity.

HOW THE DESIGN REACHES THE MODEL. It is RECOVERED from the mask channel, because nothing else can
carry it: :meth:`detopt.detector.MNISTDetector.combine_scaled` emits two channels (occluded image,
binary window) and a trainer hands a model only ``(features, mask)``. The window is an axis-aligned
rectangle, ``window[r, c] = in_rows[r] * in_columns[c]``, so a max over each spatial axis separates
it into its two per-axis indicators, and each indicator's COUNT is the extent while its CENTROID is
the centre -- giving ``(x_lo, x_hi, y_lo, y_hi)``, which is the design sorted into its canonical
order. Recovery is therefore exact up to PIXEL QUANTISATION and no further: the mask only records
which pixel CENTRES fall inside the window, so an edge is placed to within half a pixel. An empty
window carries no design at all and reads as zeros.

The recovered vector is the SORTED design. Each coordinate pair of ``MNISTDesign`` is unordered and
names the same window either way, so nothing recoverable is lost.

HOW THE IMAGE IS RESAMPLED. 2x2 AREA AVERAGE per stage transition, applied to both input channels.
A stride-2 ``SAME`` convolution takes a side of ``n`` to ``ceil(n / 2)``, and so does a 2x2 window at
stride 2 with ``SAME`` padding, so the resampled image lines up with the block's own resolution at
every stage without a shape table. Averaging rather than subsampling because subsampling drops three
pixels in four: a narrow window could vanish entirely from the mask channel, whereas the average
keeps the mask reading the FRACTION of each coarse cell the window covers and the image channel
reading the mean intensity under it.

``ensemble()`` returns ``None`` and there is no ``n_models`` argument, exactly as in the alpha-conv
stack this is the conditioned twin of: a config asking for members raises rather than quietly
training one network.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape

__all__ = [
  'SemiHyperConvRegressor', 'SemiHyperConvResidual', 'SemiHyperConvStage', 'DesignEmbedding', 'area_pool', 'recover_window'
]


def area_pool(x):
  """2x2 area average over the two spatial axes of a channels-last image ``(..., rows, columns, C)``.

  ``SAME`` padding, so a side of ``n`` becomes ``ceil(n / 2)`` -- what a stride-2 ``SAME`` convolution
  does. The divisor is counted per output cell rather than assumed to be 4, so an odd side averages
  its partial edge cell over the pixels that are really there."""
  window = (1, 2, 2, 1)
  strides = (1, 2, 2, 1)
  leading = x.shape[:-3]
  flat = jnp.reshape(x, (-1, ) + x.shape[-3:])
  total = jax.lax.reduce_window(flat, 0.0, jax.lax.add, window, strides, 'SAME')
  ones = jnp.ones((1, ) + flat.shape[-3:-1] + (1, ), flat.dtype)
  count = jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, strides, 'SAME')
  return jnp.reshape(total / count, leading + total.shape[1:])


def _axis_extent(indicator, n_cells):
  """A binary per-cell indicator of a contiguous run -> ``(lo, hi)`` in normalised coordinates.

  ``lo = centroid - count / (2 n)`` and ``hi = centroid + count / (2 n)`` over the cell CENTRES, which
  for a contiguous run of cells ``[j_lo, j_hi]`` is exactly ``(j_lo / n, (j_hi + 1) / n)``. An empty
  indicator reads ``(0, 0)``."""
  centres = (jnp.arange(n_cells, dtype=jnp.float32) + 0.5) / float(n_cells)
  count = jnp.sum(indicator, axis=-1)
  centroid = jnp.sum(indicator * centres, axis=-1) / jnp.maximum(count, 1.0)
  half = 0.5 * count / float(n_cells)
  filled = count > 0
  return jnp.where(filled, centroid - half, 0.0), jnp.where(filled, centroid + half, 0.0)


def recover_window(features):
  """``features (..., rows, columns, 2)`` -> the window ``(..., 4)`` as ``(x_lo, x_hi, y_lo, y_hi)``.

  Channel 1 is the binary window and it factorises as ``in_rows[r] * in_columns[c]``, so a max over
  each spatial axis recovers the two per-axis indicators. ``x`` runs along the COLUMN axis and ``y``
  along the ROW axis, matching the detector. Exact up to half a pixel (see the module docstring)."""
  window = features[..., 1]
  columns = jnp.max(window, axis=-2)
  rows = jnp.max(window, axis=-1)
  x_lo, x_hi = _axis_extent(columns, window.shape[-1])
  y_lo, y_hi = _axis_extent(rows, window.shape[-2])
  return jnp.stack([x_lo, x_hi, y_lo, y_hi], axis=-1)


class DesignEmbedding(nnx.Module):
  """``design (..., d) -> (..., channels)`` through one hidden ``celu`` layer."""

  def __init__(self, design_dim: int, features: int, channels: int, *, rngs: nnx.Rngs):
    self.hidden = nnx.Linear(int(design_dim), int(features), rngs=rngs)
    self.output = nnx.Linear(int(features), int(channels), rngs=rngs)

  def __call__(self, design):
    return self.output(jax.nn.celu(self.hidden(design)))


class SemiHyperConvResidual(nnx.Module):
  """One residual unit whose BODY is fed the block input, the design embedding and the resampled
  input image: ``h -> h + alpha * Conv(celu(Dropout(Conv1x1([h, embedding, image]))))``.

  The 1x1 convolution maps ``channels + embedding_channels + image_channels`` back to ``channels``,
  so everything after it is shaped exactly as the unconditioned unit. ``alpha`` is per-CHANNEL and
  zero-initialised, so the unit is the identity at initialisation."""

  def __init__(
    self, channels: int, embedding_channels: int, image_channels: int, kernel_size: int, p_dropout: float | None, *,
    rngs: nnx.Rngs
  ):
    self.channels = int(channels)
    kernel = (int(kernel_size), int(kernel_size))
    self.mix = nnx.Conv(
      self.channels + int(embedding_channels) + int(image_channels), self.channels, kernel_size=(1, 1), padding='SAME',
      rngs=rngs
    )
    self.conv = nnx.Conv(self.channels, self.channels, kernel_size=kernel, padding='SAME', rngs=rngs)
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None
    self.alpha = nnx.Param(jnp.zeros((self.channels, )))

  def __call__(self, h, embedding, image, *, deterministic: bool = True, rngs=None):
    spatial = jnp.broadcast_to(embedding[..., None, None, :], h.shape[:-1] + embedding.shape[-1:])
    branch = self.mix(jnp.concatenate([h, spatial, image], axis=-1))
    if self.dropout is not None:
      branch = self.dropout(branch, deterministic=deterministic, rngs=rngs)
    branch = self.conv(jax.nn.celu(branch))
    return h + self.alpha[...] * branch


class SemiHyperConvStage(nnx.Module):
  """``blocks`` conditioned residual units at one width, then -- unless this is the last stage -- a
  strided ``celu -> Conv`` that halves the resolution and moves to the next width."""

  def __init__(
    self, channels: int, out_channels: int | None, blocks: int, embedding_channels: int, image_channels: int, kernel_size: int,
    p_dropout: float | None, *, rngs: nnx.Rngs
  ):
    if blocks < 1:
      raise ValueError(f'blocks must be at least 1, got {blocks}')
    kernel = (int(kernel_size), int(kernel_size))
    self.units = nnx.List([
      SemiHyperConvResidual(channels, embedding_channels, image_channels, kernel_size, p_dropout, rngs=rngs)
      for _ in range(int(blocks))
    ])
    down = out_channels is not None
    self.downsample = nnx.data(
      nnx.Conv(int(channels), int(out_channels), kernel_size=kernel, strides=(2, 2), padding='SAME', rngs=rngs)
    ) if down else None

  def __call__(self, h, embedding, image, *, deterministic: bool = True, rngs=None):
    for unit in self.units:
      h = unit(h, embedding, image, deterministic=deterministic, rngs=rngs)
    if self.downsample is not None:
      h = self.downsample(jax.nn.celu(h))
    return h


class SemiHyperConvRegressor(Model):
  """``(features, mask) -> logits``, the alpha-conv stack with the design injected at every block.

  Parameters
  ----------
  channels : one width per stage; each stage but the last ends in a stride-2 conv, so the resolution
      halves per entry.
  blocks : residual units per stage.
  kernel_size : side of every square kernel in the residual bodies and the strided convolutions; the
      conditioning convolution is always 1x1.
  p_dropout : dropout inside each residual branch, applied to the MIXED tensor (the body's input).
  embedding_channels : width ``k`` of the design embedding that every block is handed.
  embedding_features : hidden width of the embedding network.
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, channels: Sequence[int] = (16, 32, 64),
    blocks: int = 2, kernel_size: int = 3, p_dropout: float | None = None, embedding_channels: int = 8,
    embedding_features: int = 32, *, rngs: nnx.Rngs
  ):
    if len(input_shape) != 3:
      raise ValueError(f'expected a channels-last image shape (rows, columns, channels), got {tuple(input_shape)}')
    if int(input_shape[-1]) != 2:
      raise ValueError(f'the window detector emits (image, mask); got {int(input_shape[-1])} channels')
    widths = [int(c) for c in channels]
    if len(widths) < 1:
      raise ValueError('channels must name at least one stage width')
    if int(embedding_channels) < 1:
      raise ValueError(f'embedding_channels must be at least 1, got {embedding_channels}')
    self.rngs = rngs
    self.n_channels_in = int(input_shape[-1])
    self.target_dim = int(target_shape[0])
    self.design_dim = 4
    self.embedding = DesignEmbedding(self.design_dim, int(embedding_features), int(embedding_channels), rngs=rngs)
    self.stem = nnx.Conv(
      self.n_channels_in, widths[0], kernel_size=(int(kernel_size), int(kernel_size)), padding='SAME', rngs=rngs
    )
    self.stages = nnx.List([
      SemiHyperConvStage(
        widths[i], widths[i + 1] if i + 1 < len(widths) else None, int(blocks), int(embedding_channels), self.n_channels_in,
        int(kernel_size), p_dropout, rngs=rngs
      ) for i in range(len(widths))
    ])
    self.output = nnx.Linear(widths[-1], self.target_dim, rngs=rngs)

  def ensemble(self) -> int | None:
    return None

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    """``features (..., rows, columns, 2)`` -> ``(..., target_dim)`` logits. ``mask`` is ignored: the
    input is a DENSE image with no absent elements."""
    embedding = self.embedding(recover_window(features))
    image = features
    h = self.stem(features)
    for stage in self.stages:
      h = stage(h, embedding, image, deterministic=deterministic, rngs=rngs)
      if stage.downsample is not None:
        image = area_pool(image)
    return self.output(jnp.mean(jax.nn.celu(h), axis=(-3, -2)))
