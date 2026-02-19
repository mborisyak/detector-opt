import inspect
from typing import Sequence

import math

import jax
import jax.numpy as jnp
import jax.nn as jnn

from flax import nnx
from ..detector import Detector
from .common import Model, Block, SiLU, LeakyTanh, LeakyReLU, bayes_aggregate

__all__ = [
  'DeepSetLFI',
  'AlphaResLFI'
]

class DeepSetLFI(Model):
  def condition_shape(self):
    return (6,)

  def output_shape(self):
    return ()

  def __init__(
    self, detector: Detector, features: Sequence[Sequence[int]], p_dropout: float | None = None, *,
    rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)

    output_dim = math.prod(self.output_shape())
    ### B, station z, view offset, view angle, layer offset, straw_offset, 6 labels
    n_design = 6 + 6

    n_stations, n_views, n_layers, n_straws, n_f = self.input_shape()
    self.blocks: list[Block] = []

    n_features = n_design + n_f

    dropout = lambda: () if p_dropout is None else (nnx.Dropout(rate=p_dropout, rngs=rngs),)

    for block_def in features:
      units = (n_features, *block_def)
      self.blocks.append(
        Block(
          *(
            Block(*dropout(), nnx.Linear(n_in, n_out, rngs=rngs), LeakyReLU())
            for n_in, n_out in zip(units[:-2], units[1:-1])
          ),
          [nnx.Linear(units[-2], units[-1], rngs=rngs), nnx.Linear(units[-2], units[-1], rngs=rngs)]
        )
      )
      n_features = 2 * units[-1]

    *_, last = features
    *_, n_latent = last

    self.output = nnx.Linear(2 * n_latent, output_dim, rngs=rngs)

  def combine(self, X, design, y):
    n_b, n_s, n_v, n_l, n_straw, n_f = X.shape
    _, n_d = design.shape

    view_offset = jnp.linspace(-1, 1, num=n_v)
    layer_offset = jnp.linspace(-1, 1, num=n_l)
    straw_offset = jnp.linspace(-1, 1, num=n_straw)

    stations, angles, magnetic_strength = design[:, :n_s], design[:, n_s:-1], design[:, -1]

    shape = (n_b, n_s, n_v, n_l, n_straw, 1)

    station_offset = jnp.broadcast_to(stations[:, :, None, None, None, None], shape=shape)
    angles_br = jnp.reshape(angles, shape=(n_b, n_v, n_s))
    angles_br = jnp.broadcast_to(angles_br[:, :, :, None, None, None], shape=shape)
    view_offsets = jnp.broadcast_to(view_offset[None, None, :, None, None, None], shape=shape)
    layer_offset = jnp.broadcast_to(layer_offset[None, None, None, :, None, None], shape=shape)
    straw_offset = jnp.broadcast_to(straw_offset[None, None, None, None, :, None], shape=shape)
    B = jnp.broadcast_to(magnetic_strength[:, None, None, None, None, None], shape=shape)
    y_br = jnp.broadcast_to(y[:, None, None, None, None, :], shape=(n_b, n_s, n_v, n_l, n_straw, 6))

    return jnp.concatenate([X, station_offset, angles_br, view_offsets, layer_offset, straw_offset, B, y_br], axis=-1)

  def __call__(self, X: jax.Array, design: jax.Array, y: jax.Array, *, deterministic: bool = True):
    result = self.combine(X, design, y)

    *rest, last = self.blocks

    for block in rest:
      mus, log_sigmas = block(result, deterministic=deterministic)
      mu, sigma = bayes_aggregate(mus, log_sigmas, axis=-2, keepdims=False)
      result = jnp.concatenate([mu, sigma], axis=-1)

    mus, log_sigmas = last(result)
    mu, sigma = bayes_aggregate(mus, log_sigmas, axis=-2, keepdims=False)
    result = jnp.concatenate([mu, sigma], axis=-1)

    result = self.output(result)

    return jnp.reshape(result, shape=(result.shape[0], *self.output_shape()))

class AlphaResLFI(Model):
  def __init__(
    self, detector: Detector,
    n_hidden: int, depth: int, p_dropout: float | None=0.2,
    *, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    input_dim, design_dim = math.prod(detector.output_shape()), math.prod(detector.design_shape())
    target_dim = math.prod(detector.target_shape())

    n_in = input_dim + design_dim + target_dim
    self.embedding = nnx.Linear(n_in, n_hidden, rngs=rngs)

    self.hidden: list[list[nnx.Module]] = list()
    self.alphas: list[nnx.Param[jax.Array]] = list()

    for i in range(depth):
      block: list[nnx.Module] = list()

      block.append(LeakyReLU())
      if p_dropout is not None:
        block.append(
          nnx.Dropout(p_dropout, rngs=rngs)
        )
      block.append(
        nnx.Linear(n_hidden, n_hidden, rngs=rngs)
      )
      self.alphas.append(
        nnx.Param(jnp.zeros(shape=(n_hidden, )), )
      )

    self.output: list[nnx.Module] = [
      LeakyReLU(),
      nnx.Linear(n_hidden, 1, rngs=rngs),
    ]

  def __call__(self, X: jax.Array, design: jax.Array, target: jax.Array, *, deterministic: bool=True):
    n, *_ = X.shape

    X = jnp.reshape(X, shape=(n, -1))
    design = jnp.reshape(design, shape=(n, -1))
    target = jnp.reshape(target, shape=(n , -1))

    result = jnp.concatenate([X, design, target], axis=-1)
    result = self.embedding(result)

    for block, alpha in zip(self.hidden, self.alphas):
      hidden = result
      for layer in block:
        if hasattr(layer, 'deterministic'):
          hidden = layer(hidden, deterministic=deterministic)
        else:
          hidden = layer(hidden)

      result = result + alpha.value * hidden

    for layer in self.output:
      result = layer(result)

    return jnp.reshape(result, shape=(n, ))