import math
import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import detopt


def test_cvae(seed):
  rng = jax.random.PRNGKey(seed)

  rng, key_rngs, key_X, key_design = jax.random.split(rng, num=4)
  rngs = nnx.Rngs(key_rngs)

  n_batch, input_shape, n_design, n_output = 5, (16, 32), 2 * 16 + 1, 3

  X = jax.random.normal(key_X, shape=(n_batch, *input_shape))
  design = jax.random.normal(key_design, shape=(n_batch, n_design))

  vae = detopt.nn.CVAE(input_shape, (n_design, ), (n_output, ), features=(7, 11, 13), rngs=rngs)

  mean, log_sigma = vae.encode(X, design)
  rng, key_eps = jax.random.split(rng, num=2)
  latent = mean + jax.random.normal(key_eps, shape=mean.shape) * jnp.exp(log_sigma)
  print('decode!')
  X_reco = vae.decode(latent, design)

  assert X_reco.shape == X.shape
  assert latent.shape == (n_batch, 13)

def test_deep_set_vae(seed):
  rng = jax.random.PRNGKey(seed)

  rng, key_rngs, key_X, key_design = jax.random.split(rng, num=4)
  rngs = nnx.Rngs(key_rngs)

  n_batch, input_shape, n_design, n_output = 5, (16, 32), 2 * 16 + 1, 3
  n_z = 27

  X = jax.random.normal(key_X, shape=(n_batch, *input_shape))
  design = jax.random.normal(key_design, shape=(n_batch, n_design))

  vae = detopt.nn.DeepSetVAE(
    input_shape, (n_design, ), (n_output, ),
    encoder_features=[(7, 11), (13, 17), (23, n_z)],
    decoder_features=[24, 14, 12],
    rngs=rngs
  )

  mean, log_sigma = vae.encode(X, design)
  rng, key_eps = jax.random.split(rng, num=2)
  latent = mean + jax.random.normal(key_eps, shape=mean.shape) * jnp.exp(log_sigma)
  print('decode!')
  X_reco = vae.decode(latent, design)

  assert X_reco.shape == X.shape
  assert latent.shape == (n_batch, n_z)