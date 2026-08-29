"""Residual CNN whose per-channel residual gate is GENERATED from the design -- the lightest
hypernetwork this stack admits.

    embedding  = Linear(celu(Linear(design)))            ``(..., k)``
    block:  h <- h + alpha(embedding) * Conv(celu(Dropout(h)))

`AlphaConvRegressor` already carries a per-CHANNEL `alpha` on every residual unit, zero-initialised so
each unit starts as the identity. That gate is the natural thing for a hypernetwork to own: it decides
how much each block contributes, per feature map, and nothing else in the unit changes. Here `alpha` is
no longer a free parameter but the output of a linear head reading the design embedding, which makes
this a hypernetwork in the strict sense -- it GENERATES parameters -- unlike the semi-hypernetwork,
which concatenates the design as extra input channels and leaves every weight static.

COST. A generated gate is `k * C + C` parameters per block against the semi-hypernetwork's
`(C + k + 2) * C + C` for its 1x1 mix, roughly a quarter as many at the widths this task uses. The
generator is the only addition; stem, convolutions, downsamples and head are bit-identical to the
alpha-conv stack, so a difference between the two is the conditioning and not the capacity elsewhere.

IDENTITY AT INITIALISATION IS PRESERVED, and it is the reason this works at all. The gate head is
zero-initialised in BOTH kernel and bias, so `alpha(d) = 0` for every design before training and each
unit is the identity exactly as in the unconditioned stack. Dropout inside the branch is likewise
inert until the gates leave zero.

THE DESIGN IS RECOVERED FROM THE MASK CHANNEL, not passed in: the trainer hands a model only
`(features, mask)`. `recover_window` and `DesignEmbedding` are imported from
`semi_hyper_conv_regressor` rather than duplicated, so both conditioned architectures read the design
by the identical route and any error in it moves them together.

`zero_design` IS THE CAPACITY-MATCHED GATE CONTROL, AND IT IS NOT A DESIGN-BLIND BASELINE. It feeds
the embedding a ZERO design vector while leaving every parameter, the generator and the
identity-at-init untouched, so `alpha` becomes one learned constant per channel instead of a function
of the design -- the same gate the unconditioned stack learns freely, reached through the same weights.

⛔️ WHAT IT DOES NOT VARY: the design INFORMATION. The design reaches this network TWICE -- once
through the generator, and once through the features, where the window channel IS the design and
`recover_window` reads it straight back out. `zero_design` withholds only the first copy, so both
conditions see which window they are looking at. A comparison against it answers "does conditioning
the GATE on the design pay", not "does knowing the design pay". Read it as the former only.

THE SECOND COPY CANNOT SIMPLY BE DELETED, because the window is the measurement's own support: the
aperture physically occludes the digit. Withholding the design from the FEATURES means keeping
`image * window` and dropping the channel that announces where the window was, which is
`Detector.combine(..., reveal_design=False)` and a property of the training arm rather than of this
module.

`ensemble()` returns `None` and there is no `n_models` argument, as in the alpha-conv stack this is the
conditioned twin of: a config asking for members raises rather than quietly training one network.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape
from .semi_hyper_conv_regressor import DesignEmbedding, recover_window

__all__ = ['AlphaHyperConvRegressor', 'AlphaHyperConvResidual', 'AlphaHyperConvStage']

DESIGN_DIM = 4


class AlphaHyperConvResidual(nnx.Module):
  """One residual unit ``h -> h + alpha(embedding) * Conv(celu(Dropout(h)))``.

  The gate head is zero-initialised in kernel and bias, so the unit is the identity at initialisation
  for every design. Resolution and width are unchanged by a unit (``SAME`` padding, unit stride)."""

  def __init__(self, channels: int, embedding_channels: int, kernel_size: int, p_dropout: float | None, *, rngs: nnx.Rngs):
    self.channels = int(channels)
    kernel = (int(kernel_size), int(kernel_size))
    self.conv = nnx.Conv(self.channels, self.channels, kernel_size=kernel, padding='SAME', rngs=rngs)
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None
    self.gate = nnx.Linear(
      int(embedding_channels), self.channels, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs
    )

  def __call__(self, h, embedding, *, deterministic: bool = True, rngs=None):
    branch = h
    if self.dropout is not None:
      branch = self.dropout(branch, deterministic=deterministic, rngs=rngs)
    branch = self.conv(jax.nn.celu(branch))
    alpha = self.gate(embedding)[..., None, None, :]
    return h + alpha * branch


class AlphaHyperConvStage(nnx.Module):
  """``blocks`` conditioned residual units at one width, then -- unless this is the last stage -- a
  strided ``celu -> Conv`` that halves the resolution and moves to the next width. ``out_channels`` is
  ``None`` at the last stage, where the head pools instead."""

  def __init__(
    self, channels: int, out_channels: int | None, embedding_channels: int, blocks: int, kernel_size: int,
    p_dropout: float | None, *, rngs: nnx.Rngs
  ):
    if blocks < 1:
      raise ValueError(f'blocks must be at least 1, got {blocks}')
    kernel = (int(kernel_size), int(kernel_size))
    self.units = nnx.List([
      AlphaHyperConvResidual(channels, embedding_channels, kernel_size, p_dropout, rngs=rngs) for _ in range(int(blocks))
    ])
    down = out_channels is not None
    self.downsample = nnx.data(
      nnx.Conv(int(channels), int(out_channels), kernel_size=kernel, strides=(2, 2), padding='SAME', rngs=rngs)
    ) if down else None

  def __call__(self, h, embedding, *, deterministic: bool = True, rngs=None):
    for unit in self.units:
      h = unit(h, embedding, deterministic=deterministic, rngs=rngs)
    if self.downsample is not None:
      h = self.downsample(jax.nn.celu(h))
    return h


class AlphaHyperConvRegressor(Model):
  """``(features, mask) -> logits`` residual CNN whose residual gates are generated from the design.

  Parameters
  ----------
  channels : one width per stage; each stage but the last ends in a stride-2 conv, so the resolution
      halves per entry.
  blocks : residual units per stage.
  kernel_size : side of every square kernel.
  embedding_features : width of the design embedding's hidden layer.
  embedding_channels : width of the design embedding handed to every gate head.
  p_dropout : dropout INSIDE each residual branch -- inert until the gates leave zero.
  zero_design : withhold the design from the generator, giving the capacity-matched baseline.
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, channels: Sequence[int] = (16, 32, 64),
    blocks: int = 2, kernel_size: int = 3, embedding_features: int = 16, embedding_channels: int = 8,
    p_dropout: float | None = None, zero_design: bool = False, *, rngs: nnx.Rngs
  ):
    if len(input_shape) != 3:
      raise ValueError(f'expected a channels-last image shape (rows, columns, channels), got {tuple(input_shape)}')
    widths = [int(c) for c in channels]
    if len(widths) < 1:
      raise ValueError('channels must name at least one stage width')
    self.rngs = rngs
    self.zero_design = bool(zero_design)
    self.n_channels_in = int(input_shape[-1])
    self.target_dim = int(target_shape[0])
    self.embedding = DesignEmbedding(DESIGN_DIM, int(embedding_features), int(embedding_channels), rngs=rngs)
    self.stem = nnx.Conv(
      self.n_channels_in, widths[0], kernel_size=(int(kernel_size), int(kernel_size)), padding='SAME', rngs=rngs
    )
    self.stages = nnx.List([
      AlphaHyperConvStage(
        widths[i], widths[i + 1] if i + 1 < len(widths) else None, int(embedding_channels), int(blocks), int(kernel_size),
        p_dropout, rngs=rngs
      ) for i in range(len(widths))
    ])
    self.output = nnx.Linear(widths[-1], self.target_dim, rngs=rngs)

  def ensemble(self) -> int | None:
    return None

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    """``features (..., rows, columns, 2)`` -> ``(..., target_dim)`` logits. ``mask`` is ignored: the
    input is a DENSE image with no absent elements."""
    if features.shape[-1] < 2:
      raise ValueError(
        f'the design-conditioned gate needs the window channel, but these features carry '
        f'{features.shape[-1]} channel(s) -- this is the design-free layout, on which a hypernetwork '
        f'has no design to read'
      )
    window = recover_window(features)
    if self.zero_design:
      window = jnp.zeros_like(window)
    embedding = self.embedding(window)
    h = self.stem(features)
    for stage in self.stages:
      h = stage(h, embedding, deterministic=deterministic, rngs=rngs)
    return self.output(jnp.mean(jax.nn.celu(h), axis=(-3, -2)))
