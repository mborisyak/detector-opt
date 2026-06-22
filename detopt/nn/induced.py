from typing import Sequence

import jax
import jax.numpy as jnp
import jax.nn as jnn

from flax import nnx

from ..detector import Detector
from .common import Model
from .set_regressor import masked_weighted_aggregate
from .activation import LeakyTanh

__all__ = ["InducedSetRegressor"]


def cross_attention(xs, inducing, scale_features, *, mask_xs=None):
  # xs: (*, n, t)
  # is: (*, m, t)
  # K: (*, n, m, t)
  K = jax.nn.sigmoid(
    scale_features * (xs[..., :, None, :] - inducing[..., None, :, :])
  ) * mask_xs[..., None, None]

  inducing_total = jnp.sum(K, axis=-2) + 1
  xs_total = jnp.sum(K, axis=-3) + 1

  xs_aggregated = jnp.sum(K * inducing[..., None, :, :], axis=-2) / inducing_total
  inducing_aggregated = jnp.sum(K * xs[..., None, :], axis=-3) / xs_total

  return xs_aggregated, inducing_aggregated


class Block(nnx.Module):
  def __init__(
    self,
    in_dim: int,
    block_def: Sequence[int],
    p_dropout: float | None = None,
    *,
    rngs: nnx.Rngs,
  ):
    *body_dims, last_hidden, out_dim = (in_dim, *block_def)

    layers = []
    for n_in, n_out in zip(body_dims, (*body_dims, last_hidden)[1:]):

      if p_dropout is not None and p_dropout > 0:
        layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
      layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
      layers.append(LeakyTanh(n_out))

    self.layers = nnx.List(layers)
    self.output = nnx.Linear(last_hidden, out_dim, rngs=rngs)

  def __call__(self, x, *, deterministic: bool = True, rngs=None):
    h = x
    for layer in self.layers:
      if isinstance(layer, nnx.Dropout):
        # Explicit rng (a key/Rngs) so dropout is functional under jit/scan.
        h = layer(h, deterministic=deterministic, rngs=rngs)
      else:
        h = layer(h)

    return self.output(h)


class InducedSetRegressor(Model):
  """``(features, mask) -> predictions`` deep set over the ``M*M`` hit pairs. Single net only.

  Parameters mirror :class:`SetRegressor` (``features`` = per-block hidden-width lists,
  ``p_dropout``), plus ``coefficients``: a length-``F`` sequence scaling each feature's
  pair-comparison ``tanh(coeff_f * Δfeature)`` (``None`` -> all ones). Set a feature's
  coefficient to 0 to drop its comparison.
  """

  @classmethod
  def from_config(cls, detector: Detector, config, *, rngs: nnx.Rngs):
    return cls(
      input_shape=detector.combined_event_shape(),
      target_shape=(detector.target_dim(),),
      ground_truth_shape=(detector.ground_truth_dim(),),
      rngs=rngs,
      **config,
    )

  def __init__(
    self,
    input_shape: Sequence[int],
    target_shape: Sequence[int],
    ground_truth_shape: Sequence[int],
    features: Sequence[Sequence[int]],
    induced: int,
    p_dropout: float | None = None,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__(input_shape, target_shape, ground_truth_shape, rngs=rngs)

    self.rngs = rngs

    _, n_in = input_shape
    (n_t,) = target_shape

    self.initial_inducing_points = nnx.Param(jnp.zeros(shape=(induced, n_in)))

    feature_masks = []
    main: list[Block] = []
    induction: list[Block] = []

    *body_defs, head_def = features
    for block_def in body_defs:
      main.append(Block(2 * n_in, block_def, p_dropout=p_dropout, rngs=rngs))
      induction.append(Block(2 * n_in, block_def, p_dropout=p_dropout, rngs=rngs))

      fmask = nnx.Param(jnp.ones(shape=(n_in,)), )
      feature_masks.append(fmask)

      n_in = int(block_def[-1])

    self.main = nnx.List(main)
    self.induction = nnx.List(induction)
    self.main_feature_masks = nnx.List(feature_masks)

    *head_hidden, head_output = head_def
    self.head = Block(2 * n_in, (*head_hidden, 2 * head_output), p_dropout=p_dropout, rngs=rngs)
    self.head_feature_mask = nnx.Param(jnp.ones(shape=(n_in,)), )

    self.output = nnx.Linear(head_output, n_t, rngs=rngs)

  def ensemble(self) -> None:
    return None

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    n_b, *_ = features.shape
    x = features
    inducing = self.initial_inducing_points[...]
    inducing = jnp.broadcast_to(inducing[None], shape=(n_b, *inducing.shape))

    for block_x, block_inducing, fmask in zip(self.main, self.induction, self.main_feature_masks):
      x_inducing, inducing_x = cross_attention(x, inducing, mask_features=fmask, mask_xs=mask)
      x = jnp.concatenate([x, x_inducing], axis=-1)
      inducing = jnp.concatenate([inducing, inducing_x], axis=-1)

      x = block_x(x, deterministic=deterministic, rngs=rngs)
      inducing = block_inducing(inducing, deterministic=deterministic, rngs=rngs)

    fmask = self.head_feature_mask[...]
    x_inducing, _ = cross_attention(x, inducing, mask_xs=mask, mask_features=fmask)
    x = jnp.concatenate([x, x_inducing], axis=-1)

    xw = self.head(x)
    x, w = jnp.split(xw, 2, axis=-1)

    return self.output(masked_weighted_aggregate(x, w, mask))