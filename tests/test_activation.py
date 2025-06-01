import os

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

def test_activation(seed, plot_root):
  rng = jax.random.PRNGKey(seed)

  xs = jnp.linspace(-7, 7, num=128)

  n, m = 3, 3

  fig = plt.figure()
  axes = fig.subplots(n, m).ravel()

  for i, axis in enumerate(axes):
    key, rng = jax.random.split(rng, num=2)

    a, b = jax.random.uniform(key, minval=-0.1, maxval=0.1, shape=(2, ))
    b = 0
    ys = a * jax.nn.softplus(xs) - b * jax.nn.softplus(-xs) + jax.nn.sigmoid(xs)

    axis.plot(xs, ys)
    axis.set_title(f'{a:.3f} {b:.3f}')

  fig.tight_layout()
  fig.savefig(plot_root / 'activation.png')
  plt.close(fig)


def test_reduce_window(seed):
  rng = jax.random.PRNGKey(seed)
  X = jax.random.randint(rng, minval=0, maxval=10, shape=(1, 5, 5)).astype(jnp.float32)

  X_ = jax.lax.reduce_window(X, init_value=-jnp.inf, computation=jax.lax.max, window_dimensions=(1, 2, 2), window_strides=(1, 2, 2), padding='SAME')
  print()
  print(X)
  print(X_)