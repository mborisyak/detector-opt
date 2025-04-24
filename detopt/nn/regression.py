from typing import Sequence

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
  def from_config(cls, config, input_dim: int, design_dim: int, output_dim: int, *, rngs: nnx.Rngs):
    config = {k: v for k, v in config.items()}

    if 'activation' in config:
      config['activation'] = getattr(nnx, config['activation'])

    return cls(
      input_dim=input_dim, design_dim=design_dim, output_dim=output_dim,
      rngs=rngs, **config
    )

  def __init__(self, input_dim: int, design_dim: int, output_dim: int, *, rngs: nnx.Rngs, activation=nnx.swish):
    self.input_dim = input_dim
    self.design_dim = design_dim
    self.output_dim = output_dim

    self.rngs = rngs
    self.activation = activation

  def __call__(self, X: jax.Array, design: jax.Array):
    raise NotImplementedError()

class MLP(Regressor):
  def __init__(
    self, input_dim: int, design_dim: int, output_dim: int,
    hidden_units: Sequence[int], *, rngs: nnx.Rngs, activation=nnx.swish
  ):
    super().__init__(input_dim, design_dim, output_dim, rngs=rngs, activation=activation)

    units = (input_dim + design_dim, *hidden_units, output_dim)

    self.layers = [
      nnx.Linear(n_in, n_out, rngs=rngs)
      for n_in, n_out in zip(units[:-1], units[1:])
    ]

  def __call__(self, X: jax.Array, design: jax.Array):
    result = jnp.concatenate([X, design], axis=-1)

    *hidden, output = self.layers

    for layer in hidden:
      result = self.activation(layer(result))

    return output(result)

class AlphaResNet(Regressor):
  def __init__(
    self, input_dim: int, design_dim: int, output_dim: int,
    n_hidden: int, depth: int, *, rngs: nnx.Rngs, activation=nnx.swish
  ):
    super().__init__(input_dim, design_dim, output_dim, rngs=rngs, activation=activation)

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
    result = jnp.concatenate([X, design], axis=-1)
    result = self.activation(self.embedding(result))

    for layer, alpha in zip(self.hidden, self.alphas):
      hidden = self.activation(layer(result))
      result = result + alpha * hidden

    return self.output(result)

class HyperResNet(Regressor):
  def __init__(
    self, input_dim: int, design_dim: int, output_dim: int,
    n_hidden_input: int, n_hidden_design: int,depth: int, *, rngs: nnx.Rngs, activation=nnx.swish
  ):
    super().__init__(input_dim, design_dim, output_dim, rngs=rngs, activation=activation)

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
    
    super().__init__(input_dim, design_dim, output_dim, rngs=rngs, activation=activation)

  def __call__(self, X: jax.Array, design: jax.Array):
    input_latent = self.activation(self.embedding_input(X))
    design_embedding = self.activation(self.embedding_design(design))

    for layer, alpha in zip(self.hidden, self.alphas):
      joint = jnp.concatenate([input_latent, design_embedding], axis=-1)
      hidden = self.activation(layer(joint))
      input_latent = input_latent + alpha * hidden

    return self.output(input_latent)