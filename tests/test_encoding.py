import math
import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import detopt


def test_encoding(seed):
  from detopt.utils.encoding import uniform_to_normal, normal_to_uniform

  rng = np.random.default_rng(seed)

  xs = np.linspace(-10, 10, num=128)
  print(
    uniform_to_normal(xs, -10, 10)
  )

  print(
    normal_to_uniform(np.linspace(-30, 30, num=128), -10, 10)
  )
