import inspect
from typing import Sequence

import math

import jax
import jax.numpy as jnp
import jax.nn as jnn

from flax import nnx

from ..detector import Detector
from .common import Model, Block, SiLU, LeakyTanh, bayes_aggregate

__all__ = [
  'Discriminator',
  'DeepSetLFI'
]

class Discriminator(Model):
  def __call__(self, X: jax.Array, design: jax.Array, ground_truth: jax.Array, *, deterministic: bool=True):
    raise NotImplementedError()

class DeepSetLFI(Discriminator):
  def __init__(
    self, detector: Detector,
    features: Sequence[Sequence[int]], p_dropout: float | None = 0.1,
    *, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    target_dim = 1
    ground_truth_dim = math.prod(self.ground_truth_shape)

    ### position + angle + B
    n_design = 3 + ground_truth_dim

    n_layers, n_straws = self.input_shape

    dropout = lambda: () if p_dropout is None else (nnx.Dropout(rate=p_dropout, rngs=rngs), )

    self.blocks: list[Block] =  []

    n_features = n_design + n_straws
    for block_def in features:
      units = (n_features, *block_def)
      self.blocks.append(
        Block(
          *(
            Block(*dropout(), nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out, ))
            for n_in, n_out in zip(units[:-2], units[1:-1])
          ),
          Block(*dropout(), nnx.Linear(units[-2], units[-1], rngs=rngs))
        )
      )
      n_features = 2 * units[-1]

    *_, last = features
    *_, n_latent = last

    self.output = nnx.Linear(n_latent, target_dim, rngs=rngs)

  def combine(self, X, design, ground_truth):
    n_b, n_l, n_s = X.shape
    _, n_d = design.shape
    n_gt = math.prod(ground_truth.shape[1:])

    X = jnp.reshape(X, shape=(n_b, n_l, n_s))
    ground_truth = jnp.reshape(ground_truth, shape=(n_b, n_gt))
    ground_truth = jnp.broadcast_to(ground_truth[:, None, :], shape=(n_b, n_l, n_gt))

    positions, angles, magnetic_strength = design[:, :n_l], design[:, n_l:2 * n_l], design[:, -1]
    positions = jnp.broadcast_to(positions[:, :, None], shape=(n_b, n_l, 1))
    angles = jnp.broadcast_to(angles[:, :, None], shape=(n_b, n_l, 1))
    magnetic_strength = jnp.broadcast_to(magnetic_strength[:, None, None], shape=(n_b, n_l, 1))

    return jnp.concatenate([X, positions, angles, magnetic_strength, ground_truth], axis=-1)

  def __call__(self, X: jax.Array, design: jax.Array, ground_truth: jax.Array, *, deterministic: bool = True):
    result = self.combine(X, design, ground_truth)

    *rest, last = self.blocks

    for block in rest:
      mus = block(result)
      mu = jnp.mean(mus, axis=1, keepdims=True)
      mu = jnp.broadcast_to(mu, shape=mus.shape)
      result = jnp.concatenate([mus, mu], axis=-1)

    mus = last(result)
    result = jnp.mean(mus, axis=1, keepdims=False)

    result = self.output(result)

    return jnp.reshape(result, shape=(result.shape[0], ))