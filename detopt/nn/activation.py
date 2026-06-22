import jax.nn
import jax.numpy as jnp

from flax import nnx

__all__ = [
  'LeakyTanh', 'LeakyReLU'
]

class LeakyTanh(nnx.Module):
  def __init__(self, dim: int):
    self.positive = nnx.Param(jnp.ones(shape=(dim,)))
    self.negative = nnx.Param(jnp.ones(shape=(dim,)))

  def __call__(self, x):
    pos, neg = self.positive[...], self.negative[...]
    return jax.nn.tanh(x) + pos * jax.nn.softplus(x) - neg * jax.nn.softplus(-x)

class LeakyReLU(nnx.Module):
  def __init__(self, dim: int):
    self.positive = nnx.Param(jnp.ones(shape=(dim,)))
    self.negative = nnx.Param(jnp.zeros(shape=(dim,)))

  def __call__(self, x):
    pos, neg = self.positive[...], self.negative[...]
    return pos * jax.nn.relu(x) - neg * jax.nn.relu(-x)