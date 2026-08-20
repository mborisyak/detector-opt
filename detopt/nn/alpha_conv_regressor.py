"""Tiny residual CNN over a DENSE image, with the zero-initialised per-unit alpha of
:mod:`detopt.nn.alpha_set_regressor` carried over to channels.

Every other regressor here consumes the ``(B, M, F)`` set/hit contract or the stereo layer-wise
grid; this one takes a plain channels-last image ``(B, rows, columns, C)`` -- what
:class:`detopt.detector.MNISTDetector` emits, the digit occluded by the visible window beside the
window mask -- and returns ``target_dim`` LOGITS (the detector's ``loss`` applies the softmax;
nothing here normalises).

    stem conv -> [residual units, strided conv]* -> CELU -> global mean pool -> linear

Each residual unit is ``h <- h + alpha * Conv(celu(h))`` with ``alpha`` a PER-CHANNEL parameter
initialised to ZERO, exactly as the set version does it. So at initialisation the residual part is
the IDENTITY map and depth costs nothing in conditioning: a deeper stack starts as well-behaved as a
shallow one and the optimiser decides how much of each branch to switch on. That is what makes
capacity cheap here, where BO scores many designs and depth must not be paid for in optimisation
difficulty. No BatchNorm-style machinery is needed to make the depth trainable, and none is used --
which also keeps the model free of running statistics the functional train loop would have to carry.

⚠️ ``p_dropout`` sits INSIDE the residual branch, and a branch scaled by ``alpha = 0`` cannot change
its output, so dropout is INERT until the alphas move off zero. A first-design comparison with and
without it can therefore come out bit-identical; that is the initialisation, not evidence that
dropout does nothing.

SINGLE NETWORK BY DESIGN, NOT BY OMISSION. ``ensemble()`` returns ``None`` and there is no
``n_models`` argument, so a config asking for members raises instead of quietly training one network
and reporting as though it had four; the trainer then draws ONE minibatch per step and takes the
single-net branch. The missing member axis is a decision -- ensembling a convolutional stack would
need it threaded through every ``nnx.Conv``, which the shared ``EnsembleLinear`` /
``EnsembleLinear`` layers do not provide -- and it should not be read as something left undone.

The activation is a HARDCODED ``jax.nn.celu`` at its default ``alpha = 1``: parameter-free, so unlike
the learnable ``EnsembleLeakyTanh`` of the set regressors it adds no per-unit capacity. CELU is
CONTINUOUSLY DIFFERENTIABLE -- no kink at the origin, unlike ``relu`` or a leaky one -- and its
negative arm decays to ``-alpha`` rather than vanishing, so no channel goes permanently dead behind a
zero alpha. Smoothness is worth having here because the residual branches start inert and the only
early gradient path to the stem and head runs through the activation.

At the default ``channels=(16, 32, 64)``, ``blocks=2``, ``kernel_size=3`` over a 2-channel input the
network holds about 122k parameters, and the resolution runs 28 -> 14 -> 7 for a 28x28 image.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape

__all__ = ['AlphaConvRegressor', 'AlphaConvResidual', 'AlphaConvStage']


class AlphaConvResidual(nnx.Module):
  """One residual unit ``h -> h + alpha * Conv(celu(Dropout(h)))``, ``alpha`` zero-init.

  ``alpha`` is per-CHANNEL rather than scalar, for the same reason the set version's is per-unit:
  different feature maps switch on at different rates, and a single scalar gate would make the whole
  branch one knob. Resolution and width are unchanged by a unit (``SAME`` padding, unit stride)."""

  def __init__(self, channels: int, kernel_size: int, p_dropout: float | None, *, rngs: nnx.Rngs):
    self.channels = int(channels)
    kernel = (int(kernel_size), int(kernel_size))
    self.conv = nnx.Conv(self.channels, self.channels, kernel_size=kernel, padding='SAME', rngs=rngs)
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None
    self.alpha = nnx.Param(jnp.zeros((self.channels, )))

  def __call__(self, h, *, deterministic: bool = True, rngs=None):
    branch = h
    if self.dropout is not None:
      branch = self.dropout(branch, deterministic=deterministic, rngs=rngs)
    branch = self.conv(jax.nn.celu(branch))
    return h + self.alpha[...] * branch


class AlphaConvStage(nnx.Module):
  """``blocks`` residual units at one width, then -- unless this is the last stage -- a strided
  ``celu -> Conv`` that halves the resolution and moves to the next width. ``out_channels`` is
  ``None`` at the last stage, where the head pools instead."""

  def __init__(
    self, channels: int, out_channels: int | None, blocks: int, kernel_size: int, p_dropout: float | None,
    *, rngs: nnx.Rngs
  ):
    if blocks < 1:
      raise ValueError(f'blocks must be at least 1, got {blocks}')
    kernel = (int(kernel_size), int(kernel_size))
    self.units = nnx.List(
      [AlphaConvResidual(channels, kernel_size, p_dropout, rngs=rngs) for _ in range(int(blocks))]
    )
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


class AlphaConvRegressor(Model):
  """``(features, mask) -> logits`` residual CNN over a dense channels-last image.

  Parameters
  ----------
  channels : one width per stage; each stage but the last ends in a stride-2 conv, so the resolution
      halves per entry.
  blocks : residual units per stage.
  kernel_size : side of every square kernel.
  p_dropout : dropout INSIDE each residual branch -- inert until the alphas leave zero (see the
      module docstring).
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
      AlphaConvStage(
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
