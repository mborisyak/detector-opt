import math
import numpy as np
import scipy as sp

__all__ = [
  'uniform_to_normal',
  'normal_to_uniform'
]

INV_SQRT_2 = math.sqrt(0.5)
SQRT_2 = math.sqrt(2.0)
LOG_2 = math.log(2.0)

def uniform_to_normal(xs, low, high):
  """
  A transformation that converts a uniformly distributed variable U[low, high] into N(0, 1).
  """
  ### uniform = 1 - exp(-x / 2)
  ### normal = erfinv(2 * uniform - 1) * sqrt(2)
  if hasattr(xs, 'dtype'):
    eps = np.finfo(xs.dtype).eps
  else:
    eps = np.finfo(np.float32).eps

  c = (low + high) / 2
  d = (high - low) / 2

  us = np.clip((xs - c) / d, eps - 1, 1 - eps)

  return sp.special.erfinv(us) * SQRT_2

def normal_to_uniform(xs, low, high):
  """
  A transformation that converts a normally distributed variable N(0, 1) into U[low, high].
  """
  us = sp.special.erf(xs * INV_SQRT_2,)
  c = (low + high) / 2
  d = (high - low) / 2
  return us * d + c