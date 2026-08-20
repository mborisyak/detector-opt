"""Plain (non-residual, non-gated) CNN over a DENSE image -- the CONTROL for
:mod:`detopt.nn.alpha_conv_regressor`.

Same external contract as the alpha version: a channels-last image ``(B, rows, columns, C)`` in --
what :class:`detopt.detector.MNISTDetector` emits, the digit occluded by the visible window beside
the window mask -- and ``target_dim`` LOGITS out (the detector's ``loss`` applies the softmax;
nothing here normalises). Same hyper-parameter names and meanings (``channels``, ``blocks``,
``kernel_size``, ``p_dropout``), so a config differs from the alpha one in the registry key alone.

    stem conv -> [plain units, strided conv]* -> CELU -> global mean pool -> linear

THE ONE DIFFERENCE, AND IT IS THE VARIABLE UNDER TEST. A unit here is ``h <- Conv(celu(Dropout(h)))``
where the alpha version's is ``h <- h + alpha * Conv(celu(Dropout(h)))``: no skip connection and no
zero-initialised per-channel gate, so the stack is NOT the identity at initialisation and depth is
paid for in conditioning. Every other choice is held fixed on purpose -- the same operator order
inside a unit, the same ``celu -> strided Conv`` between stages, the same ``celu -> global mean pool
-> linear`` head, the same parameter-free hardcoded ``jax.nn.celu``, no BatchNorm and hence no
running statistics for the functional train loop to carry.

Two consequences of removing the gate are worth stating rather than discovering. Parameter count
falls only by the alphas themselves (one per channel per unit), so the two networks are the same size
to within a fraction of a percent and a difference between them is not a capacity difference. And
``p_dropout`` now sits on the MAIN path instead of inside a branch scaled by ``alpha = 0``, so it is
live from the first step here where in the alpha version it is inert until the alphas move.

SINGLE NETWORK BY DESIGN, NOT BY OMISSION, exactly as in the alpha version: ``ensemble()`` returns
``None`` and there is no ``n_models`` argument, so a config asking for members raises instead of
quietly training one network and reporting as though it had four, and the trainer takes its
single-net branch.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape

__all__ = ['PlainConvRegressor', 'PlainConvUnit', 'PlainConvStage']


class PlainConvUnit(nnx.Module):
  """One plain unit ``h -> Conv(celu(Dropout(h)))``. Resolution and width are unchanged (``SAME``
  padding, unit stride)."""

  def __init__(self, channels: int, kernel_size: int, p_dropout: float | None, *, rngs: nnx.Rngs):
    self.channels = int(channels)
    kernel = (int(kernel_size), int(kernel_size))
    self.conv = nnx.Conv(self.channels, self.channels, kernel_size=kernel, padding='SAME', rngs=rngs)
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None

  def __call__(self, h, *, deterministic: bool = True, rngs=None):
    if self.dropout is not None:
      h = self.dropout(h, deterministic=deterministic, rngs=rngs)
    return self.conv(jax.nn.celu(h))


class PlainConvStage(nnx.Module):
  """``blocks`` plain units at one width, then -- unless this is the last stage -- a strided
  ``celu -> Conv`` that halves the resolution and moves to the next width. ``out_channels`` is
  ``None`` at the last stage, where the head pools instead."""

  def __init__(
    self, channels: int, out_channels: int | None, blocks: int, kernel_size: int, p_dropout: float | None,
    *, rngs: nnx.Rngs
  ):
    if blocks < 1:
      raise ValueError(f'blocks must be at least 1, got {blocks}')
    kernel = (int(kernel_size), int(kernel_size))
    self.units = nnx.List([PlainConvUnit(channels, kernel_size, p_dropout, rngs=rngs) for _ in range(int(blocks))])
    down = out_channels is not None
    self.downsample = nnx.data(
      nnx.Conv(int(channels), int(out_channels), kernel_size=kernel, strides=(2, 2), padding='SAME', rngs=rngs)
    ) if down else None

  def __call__(self, h, *, deterministic: bool = True, rngs=None):
    for unit in self.units:
      h = unit(h, deterministic=deterministic, rngs=rngs)
    if self.downsample is not None:
      h = self.downsample(jax.nn.celu(h))
    return h


class PlainConvRegressor(Model):
  """``(features, mask) -> logits`` plain CNN over a dense channels-last image.

  Parameters
  ----------
  channels : one width per stage; each stage but the last ends in a stride-2 conv, so the resolution
      halves per entry.
  blocks : plain convolutional units per stage.
  kernel_size : side of every square kernel.
  p_dropout : dropout before each unit's convolution, live from the first step (see the module
      docstring).
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, channels: Sequence[int] = (16, 32, 64),
    blocks: int = 2, kernel_size: int = 3, p_dropout: float | None = None, *, rngs: nnx.Rngs
  ):
    if len(input_shape) != 3:
      raise ValueError(f'expected a channels-last image shape (rows, columns, channels), got {tuple(input_shape)}')
    widths = [int(c) for c in channels]
    if len(widths) < 1:
      raise ValueError('channels must name at least one stage width')
    self.rngs = rngs
    self.n_channels_in = int(input_shape[-1])
    self.target_dim = int(target_shape[0])
    self.stem = nnx.Conv(
      self.n_channels_in, widths[0], kernel_size=(int(kernel_size), int(kernel_size)), padding='SAME', rngs=rngs
    )
    self.stages = nnx.List([
      PlainConvStage(
        widths[i], widths[i + 1] if i + 1 < len(widths) else None, int(blocks), int(kernel_size), p_dropout, rngs=rngs
      ) for i in range(len(widths))
    ])
    self.output = nnx.Linear(widths[-1], self.target_dim, rngs=rngs)

  def ensemble(self) -> int | None:
    return None

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    """``features (..., rows, columns, C)`` -> ``(..., target_dim)`` logits. ``mask`` is ignored: the
    input is a DENSE image with no absent elements, and what is invisible is already zeroed by the
    detector's window (whose extent the mask CHANNEL carries)."""
    h = self.stem(features)
    for stage in self.stages:
      h = stage(h, deterministic=deterministic, rngs=rngs)
    return self.output(jnp.mean(jax.nn.celu(h), axis=(-3, -2)))
