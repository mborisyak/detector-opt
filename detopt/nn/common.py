from typing import Sequence

import jax
import jax.numpy as jnp

from flax import nnx

__all__ = [
  'MLP',
  'AlphaResNet'
]

class MLP(nnx.Module):
  def __init__(self, units: Sequence[int], *, rngs: nnx.Rngs, activation=nnx.leaky_relu):
    self.layers = [
      nnx.Linear(n_in, n_out, rngs=rngs)
      for n_in, n_out in zip(units[:-1], units[1:])
    ]

    self.activation = activation

  def __call__(self, X: jax.Array):
    result = X

    *hidden, output = self.layers

    for layer in hidden:
      result = self.activation(layer(result))

    return output(result)

class AlphaResNet(nnx.Module):
  def __init__(self, n_in: int, n_out: int, n_hidden: int, depth: int, *, rngs: nnx.Rngs, activation=nnx.swish):
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

    self.output = nnx.Linear(n_hidden, n_out, rngs=rngs)

    self.activation = activation

  def __call__(self, X: jax.Array):
    result = self.activation(self.embedding(X))

    for layer, alpha in zip(self.hidden, self.alphas):
      hidden = self.activation(layer(result))
      result = result + alpha * hidden

    return self.output(result)