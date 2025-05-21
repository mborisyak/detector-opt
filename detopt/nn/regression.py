from typing import Sequence

import math

import jax
import jax.numpy as jnp

from flax import nnx

__all__ = [
  'Regressor',
  'MLP',
  'AlphaResNet',
  'HyperResNet'
]

class Regressor(nnx.Module):
  @classmethod
  def from_config(
      cls, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
      config, *, rngs: nnx.Rngs
  ):
    config = {k: v for k, v in config.items()}

    # if 'activation' in config:
    #   config['activation'] = getattr(nnx, config['activation'])

    return cls(
      input_shape=input_shape, design_shape=design_shape, target_shape=target_shape,
      rngs=rngs, **config
    )

  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
    *, rngs: nnx.Rngs
  ):
    self.input_shape = input_shape
    self.design_shape = design_shape
    self.target_shape = target_shape

    self.rngs = rngs

  def __call__(self, X: jax.Array, design: jax.Array):
    raise NotImplementedError()

class MLP(Regressor):
  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
    hidden_units: Sequence[int], *, rngs: nnx.Rngs
  ):
    super().__init__(input_shape, design_shape, target_shape, rngs=rngs)
    input_dim, design_dim, output_dim = math.prod(input_shape), math.prod(design_shape), math.prod(target_shape)

    units = (input_dim + design_dim, *hidden_units, output_dim)

    self.layers = [
      nnx.Linear(n_in, n_out, rngs=rngs)
      for n_in, n_out in zip(units[:-1], units[1:])
    ]

  def __call__(self, X: jax.Array, design: jax.Array):
    n, *_ = X.shape
    X = jnp.reshape(X, shape=(n, -1))
    design = jnp.reshape(design, shape=(n, -1))

    result = jnp.concatenate([X, design], axis=-1)

    *hidden, output = self.layers

    for layer in hidden:
      result = nnx.silu(layer(result))

    result = output(result)
    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))

class AlphaResNet(Regressor):
  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
    n_hidden: int, depth: int, *, rngs: nnx.Rngs
  ):
    super().__init__(input_shape, design_shape, target_shape, rngs=rngs)
    input_dim, design_dim, output_dim = math.prod(input_shape), math.prod(design_shape), math.prod(target_shape)

    n_in = input_dim + design_dim
    self.embedding = nnx.Linear(n_in, n_hidden, rngs=rngs)

    self.hidden = list()
    self.alphas = list()

    for i in range(depth):
      self.hidden.append(
        nnx.LinearGeneral(n_hidden, n_hidden, rngs=rngs)
      )
      self.alphas.append(
        nnx.Param(jnp.zeros(shape=()), )
      )

    self.output = nnx.Linear(n_hidden, output_dim, rngs=rngs)

  def __call__(self, X: jax.Array, design: jax.Array):
    n, *_ = X.shape
    X = jnp.reshape(X, shape=(n, -1))
    design = jnp.reshape(design, shape=(n, -1))

    result = jnp.concatenate([X, design], axis=-1)
    result = nnx.silu(self.embedding(result))

    for layer, alpha in zip(self.hidden, self.alphas):
      hidden = nnx.silu(layer(result))
      result = result + alpha * hidden

    result = self.output(result)
    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))

class HyperResNet(Regressor):
  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
    n_hidden_input: int, n_hidden_design: int,depth: int, *, rngs: nnx.Rngs
  ):
    super().__init__(input_shape, design_shape, target_shape, rngs=rngs)
    input_dim, design_dim, output_dim = math.prod(input_shape), math.prod(design_shape), math.prod(target_shape)

    self.embedding_input = nnx.Linear(input_dim, n_hidden_input, rngs=rngs)
    self.embedding_design = nnx.Linear(design_dim, n_hidden_design, rngs=rngs)

    self.hidden = list()
    self.alphas = list()

    for i in range(depth):
      self.hidden.append(
        nnx.LinearGeneral(n_hidden_input + n_hidden_design, n_hidden_input, rngs=rngs)
      )
      self.alphas.append(
        nnx.Param(jnp.zeros(shape=()), )
      )

    self.output = nnx.Linear(n_hidden_input, output_dim, rngs=rngs)

  def __call__(self, X: jax.Array, design: jax.Array):
    n, *_ = X.shape
    X = jnp.reshape(X, shape=(n, -1))
    design = jnp.reshape(design, shape=(n, -1))

    input_latent = nnx.silu(self.embedding_input(X))
    design_embedding = nnx.silu(self.embedding_design(design))

    for layer, alpha in zip(self.hidden, self.alphas):
      joint = jnp.concatenate([input_latent, design_embedding], axis=-1)
      hidden = nnx.silu(layer(joint))
      input_latent = input_latent + alpha * hidden

    result = self.output(input_latent)
    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))
