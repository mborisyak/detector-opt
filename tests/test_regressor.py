import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx

import detopt

def test_regressors(seed):
  rngs = nnx.Rngs(seed)

  key_X, key_design = rngs(), rngs()

  n_batch, n_input, n_design, n_output = 7, 11, 13, 5

  X = jax.random.normal(key_X, shape=(n_batch, n_input))
  design = jax.random.normal(key_design, shape=(n_batch, n_design))

  regressor = detopt.nn.MLP(
    input_dim=n_input, design_dim=n_design, output_dim=n_output,
    hidden_units=(16, 24, 32,), rngs=rngs
  )
  y = regressor(X, design)
  assert y.shape == (n_batch, n_output)

  regressor = detopt.nn.AlphaResNet(
    input_dim=n_input, design_dim=n_design, output_dim=n_output,
    n_hidden=24, depth=5, rngs=rngs
  )
  y = regressor(X, design)
  assert y.shape == (n_batch, n_output)

  regressor = detopt.nn.HyperResNet(
    input_dim=n_input, design_dim=n_design, output_dim=n_output,
    n_hidden_input=24, n_hidden_design=16, depth=5, rngs=rngs
  )
  y = regressor(X, design)
  assert y.shape == (n_batch, n_output)