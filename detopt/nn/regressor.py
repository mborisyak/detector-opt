import inspect
from typing import Sequence

import math

import jax
import jax.numpy as jnp
import jax.nn as jnn

from flax import nnx
from ..detector import Detector
from .common import Model, Block, SiLU, LeakyTanh, bayes_aggregate

__all__ = [
  'Regressor',
  'MLP',
  'AlphaResNet',
  'CNN',
  'HyperResNet',
  'DeepSet'
]

class Regressor(Model):
  def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool=True):
    raise NotImplementedError()

class MLP(Regressor):
  def __init__(
    self, detector: Detector,
    hidden_units: Sequence[int], *, layer_norm: bool=True, p_dropout: float | None=0.2, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    input_dim, design_dim = math.prod(self.input_shape), math.prod(self.design_shape)
    target_dim = math.prod(self.target_shape)

    self.units = (input_dim + design_dim, *hidden_units, target_dim)
    self.layer_norm = layer_norm

    self.layers: list[nnx.Module] = list()
    for i, (n_in, n_out) in enumerate(zip(self.units[1:-1], self.units[:-2])):
        if layer_norm:
          self.layers.append(
            nnx.LayerNorm(n_in, rngs=rngs)
          )
        if p_dropout is not None:
          self.layers.append(
            nnx.Dropout(rate=p_dropout, rngs=rngs)
          )
        self.layers.append(
          nnx.Linear(n_in, n_out, rngs=rngs)
        )
        self.layers.append(SiLU())

    *_, n_in, n_out = self.units
    self.layers.append(
      nnx.Linear(n_in, n_out, rngs=rngs)
    )

  def __call__(self, X: jax.Array, design: jax.Array, deterministic: bool=True):
    n, *_ = X.shape
    X = jnp.reshape(X, shape=(n, -1))
    design = jnp.reshape(design, shape=(n, -1))
    result = jnp.concatenate([X, design], axis=-1)

    for layer in self.layers:
      result = layer(result)

    result = jnp.reshape(result, shape=(result.shape[0], *self.target_shape))
    return result


class AlphaResNet(Regressor):
  def __init__(
    self, detector: Detector,
    n_hidden: int, depth: int, p_dropout: float | None=0.2,
    *, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    input_dim, design_dim = math.prod(self.input_shape), math.prod(self.design_shape)
    target_dim = math.prod(self.target_shape)

    n_in = input_dim + design_dim
    self.embedding = nnx.Linear(n_in, n_hidden, rngs=rngs)

    self.hidden: list[list[nnx.Module]] = list()
    self.alphas: list[nnx.Param[jax.Array]] = list()

    for i in range(depth):
      block: list[nnx.Module] = list()

      block.append(SiLU())
      if p_dropout:
        block.append(
          nnx.Dropout(n_hidden, rngs=rngs)
        )
      block.append(
        nnx.Linear(n_hidden, n_hidden, rngs=rngs)
      )
      self.alphas.append(
        nnx.Param(jnp.zeros(shape=(n_hidden, )), )
      )

    self.output: list[nnx.Module] = [
      SiLU(),
      nnx.Linear(n_hidden, target_dim, rngs=rngs),
    ]

  def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool=True):
    n, *_ = X.shape
    X = jnp.reshape(X, shape=(n, -1))
    design = jnp.reshape(design, shape=(n, -1))

    result = jnp.concatenate([X, design], axis=-1)
    result = self.embedding(result)

    for block, alpha in zip(self.hidden, self.alphas):
      hidden = result
      for layer in block:
        if hasattr(layer, 'deterministic'):
          hidden = layer(hidden, deterministic=deterministic)
        else:
          hidden = layer(hidden)

      result = result + alpha.value * hidden

    for layer in self.output:
      result = layer(jax.nn.celu(result))

    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))

class CNN(Regressor):
  def __init__(
    self, detector: Detector,
    features, p_dropout: float = 0.1,
    *, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    input_dim, design_dim = math.prod(self.input_shape), math.prod(self.design_shape)
    target_dim = math.prod(self.target_shape)

    n_d = 4

    n_layers, n_straws = self.input_shape

    self.blocks: list[tuple[Block, Block]] = list()
    n_features = 0
    for n_f in features[:-1]:
      self.blocks.append((
        Block(
          nnx.Conv(2 * n_features + n_d, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
          nnx.Dropout(rate=p_dropout, rngs=rngs),
          nnx.Conv(n_f, n_f, kernel_size=(1, n_straws), feature_group_count=n_f, padding='VALID', rngs=rngs),
          SiLU(),
        ),
        Block(
          nnx.Conv(n_f, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
          nnx.Dropout(rate=p_dropout, rngs=rngs),
          nnx.Conv(n_f, n_f, kernel_size=(n_layers, 1), feature_group_count=n_f, padding='VALID', rngs=rngs),
          SiLU()
        )
      ))
      n_features = n_f

    n_f = features[-1]

    self.final_block = Block(
      nnx.Conv(2 * n_features + n_d, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
      nnx.Dropout(rate=p_dropout, rngs=rngs),
      nnx.Conv(n_f, n_f, kernel_size=(1, n_straws), feature_group_count=n_f, padding='VALID', rngs=rngs),
      SiLU(),
      nnx.Conv(n_f, n_f, kernel_size=(1, 1), padding='VALID', rngs=rngs),
      nnx.Dropout(rate=p_dropout, rngs=rngs),
      nnx.Conv(n_f, n_f, kernel_size=(n_layers, 1), feature_group_count=n_f, padding='VALID', rngs=rngs),
      SiLU()
    )

    self.output = Block(
      nnx.Linear(n_f, target_dim, rngs=rngs)
    )

  def convolve(self, X, design, deterministic: bool = True):
    n_b, n_l, n_s = X.shape
    _, n_d = design.shape

    X = jnp.reshape(X, shape=(n_b, n_l, n_s, 1))
    positions, angles, magnetic_strength = design[:, :n_l], design[:, n_l:2 * n_l], design[:, -1]
    positions = jnp.broadcast_to(positions[:, :, None, None], shape=(n_b, n_l, n_s, 1))
    angles = jnp.broadcast_to(angles[:, :, None, None], shape=(n_b, n_l, n_s, 1))
    magnetic_strength = jnp.broadcast_to(magnetic_strength[:, None, None, None], shape=(n_b, n_l, n_s, 1))

    X = jnp.concatenate([X, positions, angles, magnetic_strength], axis=-1)
    hidden = X

    for (block_straw_wise, block_layer_wise) in self.blocks:
      hidden_straw_wise = block_straw_wise(hidden, deterministic=deterministic)
      hidden_layer_wise = block_layer_wise(hidden_straw_wise, deterministic=deterministic)

      *_, h_sw_c = hidden_straw_wise.shape
      hidden_straw_wise = jnp.broadcast_to(hidden_straw_wise, (n_b, n_l, n_s, h_sw_c))
      *_, h_lw_c = hidden_layer_wise.shape
      hidden_layer_wise = jnp.broadcast_to(hidden_layer_wise, (n_b, n_l, n_s, h_lw_c))
      hidden = jnp.concatenate([X, hidden_straw_wise, hidden_layer_wise], axis=-1)

    hidden = self.final_block(hidden, deterministic=deterministic)
    _, n_x, n_y, _ = hidden.shape
    assert n_x == n_y == 1

    hidden = jnp.mean(hidden, axis=(1, 2))

    return hidden

  def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
    result = self.convolve(X, design, deterministic=deterministic)
    result = self.output(result)

    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))

class HyperResNet(Regressor):
  def __init__(
    self, detector: Detector,
    n_hidden: int, depth: int, p_dropout: float | None=0.1, *, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    input_dim, design_dim = math.prod(self.input_shape), math.prod(self.design_shape)
    target_dim = math.prod(self.target_shape)

    dropout = lambda: [] if p_dropout is None else [nnx.Dropout(rate=p_dropout, rngs=rngs)]
    self.initial_embeddings = (
      nnx.Linear(input_dim, n_hidden, rngs=rngs),
      nnx.Linear(design_dim, n_hidden, rngs=rngs)
    )
    self.initial_block = Block(
      SiLU(),
      *dropout(),
      nnx.Linear(n_hidden, n_hidden, rngs=rngs),
    )

    self.embeddings = list()
    self.blocks = list()
    self.alphas = list()

    for i in range(depth):
      self.embeddings.append((
        nnx.Linear(input_dim, n_hidden, rngs=rngs),
        nnx.Linear(design_dim, n_hidden, rngs=rngs),
        nnx.Linear(n_hidden, n_hidden, rngs=rngs),
      ))
      self.blocks.append(Block(
        SiLU(),
        *dropout(),
        nnx.Linear(n_hidden, n_hidden, rngs=rngs),
      ))
      self.alphas.append(
        nnx.Param(jnp.zeros(shape=()), )
      )

    self.output = nnx.Linear(n_hidden, target_dim, rngs=rngs)

  def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
    n, *_ = X.shape
    X = jnp.reshape(X, shape=(n, -1))
    design = jnp.reshape(design, shape=(n, -1))

    initial_emb_input, initial_emb_design = self.initial_embeddings
    X_emb = initial_emb_input(X)
    design_emb = initial_emb_design(design)

    latent = self.initial_block(X_emb + design_emb, deterministic=deterministic)

    for embds, block, alpha in zip(self.embeddings, self.blocks, self.alphas):
      emb_input, emb_design, emb_latent = embds
      delta = block(
        emb_input(X) + emb_design(design) + emb_latent(latent),
        deterministic=deterministic
      )
      latent = latent + alpha * delta

    result = self.output(latent)
    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))

class DeepSet(Regressor):
  def __init__(
    self, detector: Detector,
    features: Sequence[Sequence[int]], p_dropout: float = 0.1,
    *, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    input_dim, design_dim = math.prod(self.input_shape), math.prod(self.design_shape)
    target_dim = math.prod(self.target_shape)
    ### position + angle + B
    n_design = 3

    n_layers, n_straws = self.input_shape
    self.blocks: list[Block] =  []

    dropout = lambda: () if p_dropout is None else (nnx.Dropout(rate=p_dropout, rngs=rngs), )

    n_features = n_design + n_straws
    for block_def in features:
      units = (n_features, *block_def)
      self.blocks.append(
        Block(
          *(
            Block(*dropout(), nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out, ))
            for n_in, n_out in zip(units[:-2], units[1:-1])
          ),
          Block(*dropout(), nnx.Linear(units[-2], units[-1], rngs=rngs))
        )
      )
      n_features = 2 * units[-1]

    *_, last = features
    *_, n_latent = last

    self.output = nnx.Linear(n_latent, target_dim, rngs=rngs)

  def combine(self, X, design):
    n_b, n_l, n_s = X.shape
    _, n_d = design.shape

    X = jnp.reshape(X, shape=(n_b, n_l, n_s))
    positions, angles, magnetic_strength = design[:, :n_l], design[:, n_l:2 * n_l], design[:, -1]
    positions = jnp.broadcast_to(positions[:, :, None], shape=(n_b, n_l, 1))
    angles = jnp.broadcast_to(angles[:, :, None], shape=(n_b, n_l, 1))
    magnetic_strength = jnp.broadcast_to(magnetic_strength[:, None, None], shape=(n_b, n_l, 1))

    return jnp.concatenate([X, positions, angles, magnetic_strength], axis=-1)

  def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
    result = self.combine(X, design)

    *rest, last = self.blocks

    for block in rest:
      mus = block(result)
      mu = jnp.mean(mus, axis=1, keepdims=True)
      mu = jnp.broadcast_to(mu, shape=mus.shape)
      result = jnp.concatenate([mus, mu], axis=-1)

    mus = last(result)
    result = jnp.mean(mus, axis=1, keepdims=False)

    result = self.output(result)

    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))

class BayesDeepSet(Regressor):
  def __init__(
              self, detector: Detector,
              features: Sequence[Sequence[int]], p_dropout: float = 0.1,
              *, rngs: nnx.Rngs
  ):
    super().__init__(detector, rngs=rngs)
    input_dim, design_dim = math.prod(self.input_shape), math.prod(self.design_shape)
    target_dim = math.prod(self.target_shape)
    ### position + angle + B
    n_design = 3

    n_layers, n_straws = self.input_shape
    self.blocks: list[Block] = []

    n_features = n_design + n_straws
    for block_def in features:
      units = (n_features, *block_def)
      self.blocks.append(
        Block(
          *(
            Block(nnx.Linear(n_in, n_out, rngs=rngs), LeakyTanh(n_out, ))
            for n_in, n_out in zip(units[:-2], units[1:-1])
          ),
          [nnx.Linear(units[-2], units[-1], rngs=rngs), nnx.Linear(units[-2], units[-1], rngs=rngs)]
        )
      )
      n_features = 3 * units[-1]

    *_, last = features
    *_, n_latent = last

    self.output = nnx.Linear(2 * n_latent, target_dim, rngs=rngs)

  def combine(self, X, design):
    n_b, n_l, n_s = X.shape
    _, n_d = design.shape

    X = jnp.reshape(X, shape=(n_b, n_l, n_s))
    positions, angles, magnetic_strength = design[:, :n_l], design[:, n_l:2 * n_l], design[:, -1]
    positions = jnp.broadcast_to(positions[:, :, None], shape=(n_b, n_l, 1))
    angles = jnp.broadcast_to(angles[:, :, None], shape=(n_b, n_l, 1))
    magnetic_strength = jnp.broadcast_to(magnetic_strength[:, None, None], shape=(n_b, n_l, 1))

    return jnp.concatenate([X, positions, angles, magnetic_strength], axis=-1)

  def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
    result = self.combine(X, design)

    *rest, last = self.blocks

    for block in rest:
      mus, log_sigmas = block(result)
      mu, sigma = bayes_aggregate(mus, log_sigmas, axis=1, keepdims=True)
      mu = jnp.broadcast_to(mu, shape=mus.shape)
      sigma = jnp.broadcast_to(sigma, shape=mus.shape)
      result = jnp.concatenate([mus, mu, sigma], axis=-1)

    mus, log_sigmas = last(result)
    mu, sigma = bayes_aggregate(mus, log_sigmas, axis=1, keepdims=False)
    result = jnp.concatenate([mu, sigma], axis=-1)

    result = self.output(result)

    return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))
