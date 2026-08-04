"""Plain MLP over the whole element set (optionally an ensemble).

``(features, mask) -> predictions``, flattening the element axis instead of aggregating over it:
masked elements are zeroed and the ``(M, F)`` block becomes one ``M * F`` input vector. Unlike the
set regressors this is NOT permutation-invariant -- the network sees the elements in a fixed order.
This is usable wherever the element axis is the design's own ordering and never varies in length --
the enzyme detector's batch of ``n_experiments`` experiments, where each element carries its own
measurements plus its own design values, so the flat input is exactly
``n_experiments * (measurements + design)``. There it serves as the order-dependent ABLATION:
``config/enzyme.yaml`` uses ``set-regressor``, whose aggregation is permutation-invariant in the
experiments (the batch being an unordered set of (condition, readout) pairs), and this model is what
that invariance is worth measuring against.

Layers are shared with the set regressor (``EnsembleLinear`` / ``EnsembleLeakyTanh``), so
``n_models`` ensembles this the same way: every parameter grows a leading member axis and all
members evaluate in one batched pass.
"""

from typing import Sequence

import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape
from .set_regressor import EnsembleLinear, EnsembleLeakyTanh

__all__ = ['MLPRegressor']


class MLPRegressor(Model):
  """``(features, mask) -> predictions`` MLP over the flattened element set.

  Parameters
  ----------
  features : hidden widths, e.g. ``[128, 128]``.
  n_models : ``None`` for a single network, or an int ``>= 1`` for an ensemble of that many
      independent members.
  p_dropout : optional dropout rate, applied before every hidden layer.
  """

  def __init__(
    self,
    input_shape: Shape,
    target_shape: Shape,
    ground_truth_shape: Shape,
    features: Sequence[int],
    n_models: int | None = None,
    p_dropout: float | None = None,
    *,
    rngs: nnx.Rngs
  ):
    if len(features) < 1:
      raise ValueError('features must contain at least one hidden width')
    self.rngs = rngs
    self.n_elements = int(input_shape[0])
    self.n_features_in = int(input_shape[-1])
    self.n_inputs = self.n_elements * self.n_features_in
    self.target_dim = int(target_shape[0])
    self.n_models = None if n_models is None else int(n_models)
    if self.n_models is not None and self.n_models < 1:
      raise ValueError('n_models must be None or an int >= 1')

    layers = []
    previous = self.n_inputs
    for width in features:
      if p_dropout is not None and p_dropout > 0:
        layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
      layers.append(EnsembleLinear(self.n_models, previous, int(width), rngs=rngs))
      layers.append(EnsembleLeakyTanh(self.n_models, int(width)))
      previous = int(width)
    self.hidden = nnx.List(layers)
    self.output = EnsembleLinear(self.n_models, previous, self.target_dim, rngs=rngs)

  def ensemble(self) -> int | None:
    return self.n_models

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    # features: (..., M, F); mask: (..., M). Leading axes are (B,) for a single net or (N, B) for an
    # ensemble. Masked-out elements are zeroed before the flatten, so padding cannot reach a weight.
    h = features * mask.astype(features.dtype)[..., None]
    h = h.reshape(h.shape[:-2] + (self.n_inputs,))
    for layer in self.hidden:
      if isinstance(layer, nnx.Dropout):
        # Explicit rng (a key/Rngs) so dropout stays functional under jit/scan.
        h = layer(h, deterministic=deterministic, rngs=rngs)
      else:
        h = layer(h)
    return self.output(h)
