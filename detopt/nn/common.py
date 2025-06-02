from typing import Sequence
import inspect

import math

import jax
import jax.numpy as jnp

from flax import nnx

from ..detector import Detector

__all__ = [
  'CELu', 'SiLU', 'LeakyTanh', 'Softplus', 'gated_leaky_tanh',
  'apply_with_kwargs',
  'Block', 'bayes_aggregate'
]

class Model(nnx.Module):
  @classmethod
  def from_config(cls, detector: Detector, config, *, rngs: nnx.Rngs):
    return cls(detector, rngs=rngs, **config)

  def __init__(self, detector: Detector, *, rngs: nnx.Rngs):
    self.input_shape = detector.output_shape()
    self.design_shape = detector.design_shape()
    self.target_shape = detector.target_shape()
    self.ground_truth_shape = detector.ground_truth_shape()

    self.rngs = rngs

  def regularization(self):
    _, parameters, _ = nnx.split(self, nnx.Param, ...)
    reg = sum(
      jnp.sum(jnp.square(p)) for p in jax.tree.leaves(parameters)
    )

    n = sum(math.prod(p.shape) for p in jax.tree.leaves(parameters))

    return reg / n

class LeakyTanh(nnx.Module):
  def __init__(self, *shape):
    self.positive = nnx.Param(jnp.ones(shape=shape, ))
    self.negative = nnx.Param(jnp.ones(shape=shape, ))

  def __call__(self, x):
    return jax.nn.tanh(x) + self.positive.value * jax.nn.softplus(x) - self.negative * jax.nn.softplus(-x)

def gated_leaky_tanh(x, alpha, beta):
    return jax.nn.tanh(x) + alpha * jax.nn.softplus(x) - beta * jax.nn.softplus(-x)

class CELu(nnx.Module):
  def __init__(self, n):
    self.alpha = nnx.Param(jnp.ones(shape=(n, ), ))

  def __call__(self, X):
    return jax.nn.celu(X, self.alpha.value)

class SiLU(nnx.Module):
  def __call__(self, X):
    return jax.nn.silu(X)

class Softplus(nnx.Module):
  def __call__(self, X):
    return jax.nn.softplus(X)

class LeakyReLU(nnx.Module):
  def __call__(self, X):
    return jax.nn.leaky_relu(X, 0.05)

class MaxPool(nnx.Module):
  def __call__(self, X):
    return jax.lax.reduce_window(
      X, init_value=-jnp.inf, computation=jax.lax.max,
      window_dimensions=(1, 2, 2, 1), window_strides=(1, 2, 2, 1), padding='SAME'
    )

def has_kwargs(f, name):
  signature = inspect.signature(f)

  return any(
    param_name == name or param.kind == inspect.Parameter.VAR_KEYWORD
    for param_name, param in signature.parameters.items()
  )

### I can't believe I have to write this...
def apply_with_kwargs(f, args, kwargs):
  signature = inspect.signature(f)
  has_var_kw = any(
    param.kind == inspect.Parameter.VAR_KEYWORD
    for _, param in signature.parameters.items()
  )

  if has_var_kw:
    return f(*args, **kwargs)
  else:
    filtered = {
      k: v
      for k, v in kwargs.items()
      if k in signature.parameters
    }
    return f(*args, **filtered)

def eval_with_kwargs(module, args, kwargs):
  if isinstance(module, nnx.Module):
    return apply_with_kwargs(module, args, kwargs)

  elif isinstance(module, (tuple, list)):
    return [
      apply_with_kwargs(m, args, kwargs)
      for m in module
    ]
  else:
    raise ValueError('a module should be either nnx.Module or list/tuple of them.')

class Block(nnx.Module):
  def __init__(self, *modules):
    self.modules = modules

  def __call__(self, *args, **kwargs):
    result = args
    *first, last = self.modules
    for module in first:
      result = eval_with_kwargs(module, result, kwargs)

      if isinstance(result, jax.Array):
        result = (result, )

    result = eval_with_kwargs(last, result, kwargs)
    return result

def bayes_aggregate(mu, log_sigma, axis, keepdims=False):
  # inv_sigma_sqr = jnp.exp(-2 * log_sigma)
  inv_sigma_sqr = jax.nn.softplus(-log_sigma)

  mu_inv_sigma_sqr = jnp.sum(mu * inv_sigma_sqr, axis=axis, keepdims=keepdims)
  inv_sum_inv_sigma_sqr = 1 / (1 + jnp.sum(inv_sigma_sqr, axis=axis, keepdims=keepdims))

  mu_aggregated = mu_inv_sigma_sqr * inv_sum_inv_sigma_sqr
  sigma_aggregated = jnp.sqrt(inv_sum_inv_sigma_sqr)

  return mu_aggregated, sigma_aggregated