import inspect
from typing import Sequence

import math

import jax
import jax.numpy as jnp
import jax.nn as jnn

from flax import nnx

from .common import Model, Block, SiLU, LeakyTanh, Softplus, LeakyReLU, gated_leaky_tanh, bayes_aggregate

__all__ = [
  'VAE',
  'MLPVAE',
  'HyperVAE',
  'CVAE',
  'DeepSetVAE',

  'elbo',
  'cross_entropy_elbo'
]

def elbo(
  X_original: jax.Array, X_reconstructed: jax.Array, latent_mean: jax.Array, latent_sigma: jax.Array,
  sigma_reconstructed: float=1.0, beta: float | None=None, exact: bool=False,
  axes: tuple[int, ...] | int | None=None
):
  """
  Returns Evidence Lower Bound for normally distributed (z | X), (X | z) and z:
    P(z | X) = N(`latent_mean`, `latent_std`);
    P(X | z) = N(`X_reconstructed`, `sigma_reconstructed`);
    P(z) = N(0, 1).

  :param X_original: ground-truth sample;
  :param X_reconstructed: reconstructed sample;
  :param latent_mean: estimated mean of the posterior P(z | X);
  :param latent_log_sigma: estimated log sigma of the posterior P(z | X);
  :param sigma_reconstructed: variance for reconstructed sample, i.e. X | z ~ N(X_original, sigma_reconstructed)
    If a scalar, `Var(X | z) = sigma_reconstructed * I`, if tensor then `Var(X | z) = diag(sigma_reconstructed)`
  :param beta: coefficient for beta-VAE
  :param exact: if true returns exact value of ELBO, otherwise returns rearranged ELBO equal to the original
    up to a multiplicative constant, possibly increasing computational stability for low `sigma_reconstructed`.
  :param axes: axes of reduction for samples, typically, all axes except for batch ones, if `None` all axes expect
    for the first one. Latent batch axes are considered the same as the sample batch axes.

  :return: Evidence Lower Bound (renormalized, if `exact=False`).
  """

  if axes is None:
    sample_axes = range(1, X_original.ndim)
    latent_axes = range(1, latent_mean.ndim)
  else:
    sample_axes = axes
    sample_batch_axes = tuple(i for i in range(X_original.ndim) if i not in sample_axes)
    latent_axes = tuple(i for i in range(latent_mean.ndim) if i not in sample_batch_axes)

  n = max(
    math.prod(X_original.shape[i] for i in sample_axes),
    math.prod(latent_mean.shape[i] for i in latent_axes),
  )

  reconstruction_loss = 0.5 * jnp.sum(jnp.square(X_reconstructed - X_original), axis=sample_axes)

  posterior_penalty = 0.5 * jnp.sum(
    jnp.square(latent_sigma) + jnp.square(latent_mean) - 2 * jnp.log(latent_sigma),
    axis=latent_axes
  )

  return (reconstruction_loss + jnp.square(sigma_reconstructed) * posterior_penalty) / n

def cross_entropy_elbo(
  X_original: jax.Array, logits: jax.Array, latent_mean: jax.Array, latent_sigma: jax.Array,
  axes: tuple[int, ...] | int | None=None
):
  """
  Returns Evidence Lower Bound for normally distributed (z | X), (X | z) and z:
    P(z | X) = N(`latent_mean`, `latent_std`);
    P(X | z) = N(`X_reconstructed`, `sigma_reconstructed`);
    P(z) = N(0, 1).

  :param X_original: ground-truth sample;
  :param logits: reconstructed sample;
  :param latent_mean: estimated mean of the posterior P(z | X);
  :param latent_sigma: estimated log sigma of the posterior P(z | X);
  :param sigma_reconstructed: variance for reconstructed sample, i.e. X | z ~ N(X_original, sigma_reconstructed)
    If a scalar, `Var(X | z) = sigma_reconstructed * I`, if tensor then `Var(X | z) = diag(sigma_reconstructed)`
  :param beta: coefficient for beta-VAE
  :param exact: if true returns exact value of ELBO, otherwise returns rearranged ELBO equal to the original
    up to a multiplicative constant, possibly increasing computational stability for low `sigma_reconstructed`.
  :param axes: axes of reduction for samples, typically, all axes except for batch ones, if `None` all axes expect
    for the first one. Latent batch axes are considered the same as the sample batch axes.

  :return: Evidence Lower Bound (renormalized, if `exact=False`).
  """

  if axes is None:
    sample_axes = range(1, X_original.ndim)
    latent_axes = range(1, latent_mean.ndim)
  else:
    sample_axes = axes
    sample_batch_axes = tuple(i for i in range(X_original.ndim) if i not in sample_axes)
    latent_axes = tuple(i for i in range(latent_mean.ndim) if i not in sample_batch_axes)

  n = max(
    math.prod(X_original.shape[i] for i in sample_axes),
    math.prod(latent_mean.shape[i] for i in latent_axes),
  )

  reconstruction_loss = jnp.sum(
    X_original * jax.nn.softplus(-logits) + (1 - X_original) * jax.nn.softplus(logits),
    axis=sample_axes
  )

  posterior_penalty = 0.5 * jnp.sum(
    jnp.square(latent_sigma) + jnp.square(latent_mean) - 2 * jnp.log(latent_sigma),
    axis=latent_axes
  )

  return (reconstruction_loss + posterior_penalty) / n

class VAE(Model):
  latent_dim: int

  def __call__(self, key: jax.Array, design):
    raise NotImplementedError()

  def encode(self, X, design, deterministic: bool=False):
    raise NotImplementedError()

  def decode(self, latent, design, deterministic: bool=False):
    raise NotImplementedError()

  def sample(self, key: jax.Array, design: jax.Array):
    n, *_ = design.shape
    latent = jax.random.normal(key, shape=(n, self.latent_dim))
    return self.decode(latent, design, deterministic=True)

class MLPVAE(VAE):
  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int], units: Sequence[int],
    *, rngs: nnx.Rngs
  ):
    super().__init__(input_shape, design_shape, target_shape, rngs=rngs)
    sample_dim, design_dim = math.prod(input_shape), math.prod(design_shape)
    *intermediate, latent_dim = units
    units_encoder = (sample_dim + design_dim, *intermediate, latent_dim)
    units_decoder = (latent_dim + design_dim, *reversed(intermediate), sample_dim)

    self.encoder = Block(
      *(
        Block(nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out, ))
        for n_in, n_out in zip(units_encoder[:-2], units_encoder[1:-1])
      ),
      [
        nnx.Linear(units_encoder[-2], units_encoder[-1], rngs=rngs),
        nnx.Linear(units_encoder[-2], units_encoder[-1], rngs=rngs)
      ]
    )
    self.decoder = Block(
      *(
        Block(nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out, ))
        for n_in, n_out in zip(units_decoder[:-2], units_decoder[1:-1])
      ),
      nnx.Linear(units_decoder[-2], units_decoder[-1], rngs=rngs)
    )
    self.latent_dim = latent_dim

  def encode(self, x, design, deterministic: bool=False):
    n_b, *x_dims = x.shape
    x = jnp.reshape(x, (n_b, math.prod(x_dims)))

    n_b, *design_dims = design.shape
    design = jnp.reshape(design, (n_b, math.prod(design_dims)))

    mean, raw_sigma = self.encoder(jnp.concatenate([x, design], axis=1), use_running_average=deterministic)
    return mean, jax.nn.softplus(raw_sigma)

  def decode(self, latent, design, deterministic: bool=False):
    n_b, *design_dims = design.shape
    design = jnp.reshape(design, (n_b, math.prod(design_dims)))

    result = self.decoder(jnp.concatenate([latent, design], axis=1), use_running_average=deterministic)
    result = jnp.reshape(result, (n_b, *self.input_shape))
    return result

class HyperVAE(VAE):
  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
    units: Sequence[int], design_embedding: Sequence[int], *, rngs: nnx.Rngs
  ):
    super().__init__(input_shape, design_shape, target_shape, rngs=rngs)
    sample_dim, design_dim = math.prod(input_shape), math.prod(design_shape)
    *intermediate, latent_dim = units
    units_encoder = (sample_dim, *intermediate, latent_dim)
    units_decoder = (latent_dim, *reversed(intermediate), sample_dim)

    self.encoder_body = [
        nnx.Linear(n_in, n_out, rngs=rngs)
        for n_in, n_out in zip(units_encoder[:-2], units_encoder[1:-1])
    ]
    self.encoder_head = (
      nnx.Linear(units_encoder[-2], units_encoder[-1], rngs=rngs),
      nnx.Linear(units_encoder[-2], units_encoder[-1], rngs=rngs)
    )
    self.decoder_body = [
        nnx.Linear(n_in, n_out, rngs=rngs)
        for n_in, n_out in zip(units_decoder[:-2], units_decoder[1:-1])
    ]
    self.decoder_head = nnx.Linear(units_decoder[-2], units_decoder[-1], rngs=rngs)

    *_, n_emb = design_embedding
    embedding_units = (design_dim, *design_embedding)

    self.design_embedding = Block(*(
      Block(nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out))
      for n_in, n_out in zip(embedding_units[:-1], embedding_units[1:])
    ))

    self.hyper_encoder = [
      (nnx.Linear(n_emb, n_out, rngs=rngs), nnx.Linear(n_emb, n_out, rngs=rngs))
      for n_out in units_encoder[1:-1]
    ]

    self.hyper_decoder = [
      (nnx.Linear(n_emb, n_out, rngs=rngs), nnx.Linear(n_emb, n_out, rngs=rngs))
      for n_out in units_decoder[1:-1]
    ]
    self.latent_dim = latent_dim

  def encode(self, x, design, deterministic: bool=False):
    n_b, *x_dims = x.shape
    x = jnp.reshape(x, (n_b, math.prod(x_dims)))

    n_b, *design_dims = design.shape
    design = jnp.reshape(design, (n_b, math.prod(design_dims)))

    design_emb = self.design_embedding(design)

    result = x
    for layer, (hyper_alpha, hyper_beta) in zip(self.encoder_body, self.hyper_encoder):
      result = layer(result)
      alpha, beta = hyper_alpha(design_emb), hyper_beta(design_emb)
      result = gated_leaky_tanh(result, alpha, beta)

    head_mean, head_raw_sigma = self.encoder_head

    mean, raw_sigma = head_mean(result), head_raw_sigma(result)
    return mean, jax.nn.softplus(raw_sigma)

  def decode(self, latent, design, deterministic: bool=False):
    n_b, *design_dims = design.shape
    design = jnp.reshape(design, (n_b, math.prod(design_dims)))

    design_emb = self.design_embedding(design)

    result = latent
    for layer, (hyper_alpha, hyper_beta) in zip(self.decoder_body, self.hyper_decoder):
      result = layer(result)
      alpha, beta = hyper_alpha(design_emb), hyper_beta(design_emb)
      result = gated_leaky_tanh(result, alpha, beta)

    result = self.decoder_head(result)
    result = jnp.reshape(result, (n_b, *self.input_shape))
    return jax.nn.softplus(result)

class CVAE(VAE):
  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
    features, *, rngs: nnx.Rngs
  ):
    super().__init__(input_shape, design_shape, target_shape, rngs=rngs)
    n_d, n_input = 3, 1
    n_layers, n_straws = input_shape

    features_encoder = (0, *features)

    self.encoder_body: list[tuple[Block, Block]] = list()
    for n_in, n_f in zip(features_encoder[:-2], features_encoder[1:-1]):
      self.encoder_body.append((
        Block(
          nnx.Conv(2 * n_in + n_d + n_input, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
          nnx.Conv(n_f, n_f, kernel_size=(1, n_straws), feature_group_count=n_f, padding='VALID', rngs=rngs),
          SiLU(),
        ),
        Block(
          nnx.Conv(n_f, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
          nnx.Conv(n_f, n_f, kernel_size=(n_layers, 1), feature_group_count=n_f, padding='VALID', rngs=rngs),
          SiLU()
        )
      ))

    *_, n_in, n_f = features_encoder

    self.encoder_head = Block(
      nnx.Conv(2 * n_in + n_d + n_input, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
      nnx.Conv(n_f, n_f, kernel_size=(1, n_straws), feature_group_count=n_f, padding='VALID', rngs=rngs),
      SiLU(),
      nnx.Conv(n_f, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
      [
        nnx.Conv(n_f, n_f, kernel_size=(n_layers, 1), feature_group_count=n_f, padding='VALID', rngs=rngs),
        nnx.Conv(n_f, n_f, kernel_size=(n_layers, 1), feature_group_count=n_f, padding='VALID', rngs=rngs),
      ]
    )

    *rest, n_f, n_z = features
    features_decoder = (n_f, *reversed(rest), n_input)
    self.latent_dim = n_z

    self.decoder_head = Block(
      ### (*, 1, 1, n_z + n_d)
      nnx.ConvTranspose(n_z, n_f, kernel_size=(n_layers, 1), padding='VALID', rngs=rngs),
      ### (*, n_l, 1, n_z + n_d)
      nnx.ConvTranspose(n_f, n_f, kernel_size=(1, n_straws), padding='VALID', rngs=rngs),
      nnx.Conv(n_f, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
      ### (*, n_l, n_s, n_f)
    )

    self.decoder_body: list[tuple[Block, Block, Block]] = list()
    for n_in, n_f in zip(features_decoder[:-1], features_decoder[1:]):
      self.decoder_body.append((
        Block(
          nnx.Conv(n_in, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
          nnx.Conv(n_f, n_f, kernel_size=(1, n_straws), feature_group_count=n_f, padding='VALID', rngs=rngs),
          SiLU(),
        ),
        Block(
          nnx.Conv(n_f, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
          nnx.Conv(n_f, n_f, kernel_size=(n_layers, 1), feature_group_count=n_f, padding='VALID', rngs=rngs),
          SiLU()
        ),
        Block(
          nnx.Conv(n_in + 2 * n_f + n_d, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
        )
      ))

  def encode(self, X, design, deterministic: bool = True):
    n_b, n_l, n_s = X.shape
    _, n_d = design.shape

    X = jnp.reshape(X, shape=(n_b, n_l, n_s, 1))
    positions, angles, magnetic_strength = design[:, :n_l], design[:, n_l:2 * n_l], design[:, -1]
    positions = jnp.broadcast_to(positions[:, :, None, None], shape=(n_b, n_l, n_s, 1))
    angles = jnp.broadcast_to(angles[:, :, None, None], shape=(n_b, n_l, n_s, 1))
    magnetic_strength = jnp.broadcast_to(magnetic_strength[:, None, None, None], shape=(n_b, n_l, n_s, 1))

    X = jnp.concatenate([X, positions, angles, magnetic_strength], axis=-1)
    hidden = X

    for (block_straw_wise, block_layer_wise) in self.encoder_body:
      hidden_straw_wise = block_straw_wise(hidden, deterministic=deterministic)
      hidden_layer_wise = block_layer_wise(hidden_straw_wise, deterministic=deterministic)

      *_, h_sw_c = hidden_straw_wise.shape
      hidden_straw_wise = jnp.broadcast_to(hidden_straw_wise, (n_b, n_l, n_s, h_sw_c))
      *_, h_lw_c = hidden_layer_wise.shape
      hidden_layer_wise = jnp.broadcast_to(hidden_layer_wise, (n_b, n_l, n_s, h_lw_c))
      hidden = jnp.concatenate([X, hidden_straw_wise, hidden_layer_wise], axis=-1)

    mean, raw_sigma = self.encoder_head(hidden, deterministic=deterministic)
    mean = jnp.mean(mean, axis=(1, 2))
    raw_sigma = jnp.mean(raw_sigma, axis=(1, 2))

    return mean, jax.nn.softplus(raw_sigma)

  def decode(self, latent, design, deterministic: bool = True):
    n_b, n_z = latent.shape
    n_l, n_s = self.input_shape
    _, n_d = design.shape

    latent = jnp.reshape(latent, shape=(n_b, 1, 1, n_z))
    positions, angles, magnetic_strength = design[:, :n_l], design[:, n_l:2 * n_l], design[:, -1]
    positions = jnp.broadcast_to(positions[:, :, None, None], shape=(n_b, n_l, n_s, 1))
    angles = jnp.broadcast_to(angles[:, :, None, None], shape=(n_b, n_l, n_s, 1))
    magnetic_strength = jnp.broadcast_to(magnetic_strength[:, None, None, None], shape=(n_b, n_l, n_s, 1))

    design = jnp.concatenate([positions, angles, magnetic_strength], axis=-1)

    result = self.decoder_head(latent)

    for (block_straw_wise, block_layer_wise, block_combine) in self.decoder_body:
      hidden_straw_wise = block_straw_wise(result, deterministic=deterministic)
      hidden_layer_wise = block_layer_wise(hidden_straw_wise, deterministic=deterministic)

      *_, h_sw_c = hidden_straw_wise.shape
      hidden_straw_wise = jnp.broadcast_to(hidden_straw_wise, (n_b, n_l, n_s, h_sw_c))
      *_, h_lw_c = hidden_layer_wise.shape
      hidden_layer_wise = jnp.broadcast_to(hidden_layer_wise, (n_b, n_l, n_s, h_lw_c))
      result = jnp.concatenate([design, result, hidden_straw_wise, hidden_layer_wise], axis=-1)
      result = block_combine(result)

    return jnp.mean(result, axis=3)

class DeepSetVAE(VAE):
  def __init__(
    self, input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
    encoder_features: Sequence[Sequence[int]], decoder_features: Sequence[int], *, rngs: nnx.Rngs
  ):
    super().__init__(input_shape, design_shape, target_shape, rngs=rngs)
    n_l, n_s = input_shape
    ### position + angle + B
    n_design = 3
    n_input = n_s

    n_features = n_design + n_input

    self.encoder: list[Block] =  []
    for block_def in encoder_features:
      units = (n_features, *block_def)
      self.encoder.append(
        Block(
          *(
            Block(nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out, ))
            for n_in, n_out in zip(units[:-2], units[1:-1])
          ),
          [nnx.Linear(units[-2], units[-1], rngs=rngs), nnx.Linear(units[-2], units[-1], rngs=rngs)]
        )
      )
      n_features = 3 * units[-1]

    *_, last = encoder_features
    *_, n_latent = last
    self.latent_dim = n_latent

    units = (n_design + n_latent, *decoder_features)
    self.decoder: Block = Block(*(
      Block(nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out, ))
      for n_in, n_out in zip(units[:-1], units[1:])
    ))

    *_, last_hidden = decoder_features

    self.output = nnx.Linear(last_hidden, n_s, rngs=rngs)

  def combine(self, X, design):
    n_b, n_l, n_s = X.shape
    _, n_d = design.shape

    positions, angles, magnetic_strength = design[:, :n_l], design[:, n_l:2 * n_l], design[:, -1]
    positions = jnp.broadcast_to(positions[:, :, None], shape=(n_b, n_l, 1))
    angles = jnp.broadcast_to(angles[:, :, None], shape=(n_b, n_l, 1))
    magnetic_strength = jnp.broadcast_to(magnetic_strength[:, None, None], shape=(n_b, n_l, 1))

    return jnp.concatenate([X, positions, angles, magnetic_strength], axis=-1)

  def encode(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
    result = self.combine(X, design)

    *rest, last = self.encoder
    for block in rest:
      mus, log_sigmas = block(result)
      mu, sigma = bayes_aggregate(mus, log_sigmas, axis=(1, ), keepdims=True)
      mu = jnp.broadcast_to(mu, shape=mus.shape)
      sigma = jnp.broadcast_to(sigma, shape=mus.shape)
      result = jnp.concatenate([mus, mu, sigma], axis=-1)

    mus, log_sigmas = last(result)
    mu, sigma = bayes_aggregate(mus, log_sigmas, axis=(1, ), keepdims=False)
    return mu, sigma

  def decode(self, latent: jax.Array, design: jax.Array, *, deterministic: bool = True):
    n_b, n_z = latent.shape
    n_l, n_s = self.input_shape

    latent = jnp.broadcast_to(latent[:, None, :], shape=(n_b, n_l, n_z))
    result = self.combine(latent, design)
    result = self.decoder(result)

    return self.output(result)