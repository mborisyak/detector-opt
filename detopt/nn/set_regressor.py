"""Detector-agnostic set regressor (optionally an ensemble).

Consumes ``(features, mask)`` and produces predictions. With ``n_models=None`` it is a
single network: ``(B, M, F), (B, M) -> (B, T)``. With ``n_models=n`` (an int, ``n>=1``)
it is an ensemble of ``n`` independent members stacked on a leading axis:
``(N, B, M, F), (N, B, M) -> (N, B, T)`` (member ``k`` sees slice ``k``). Every parameter
then carries a leading ensemble axis and the whole ensemble evaluates in one batched
(einsum) pass -- no per-member Python loop. :meth:`SetRegressor.ensemble` returns
``n_models`` so trainers feed ``n`` independent minibatches at training time and average
the ``n`` member predictions at evaluation. ``n_models=1`` is a one-member ensemble (a
leading axis of size 1).

All detector-specific feature engineering (design lookups, normalisation) lives in
:meth:`detopt.detector.Detector.combine`; this module only knows the per-hit feature
dimensionality and the target dimensionality.

The architecture is a stack of shared per-hit MLP blocks interleaved with a simple
learned-weight set aggregation over the ``M`` axis (each hit emits a value and a
non-negative gate ``softplus(w_raw)``; the event representation is the normalised weighted
average ``sum_i value_i * gate_i / (sum_i gate_i + 1)`` over the *live* hits), followed by
a linear head. The mask enters ONLY in that aggregation (the gate is multiplied by the
mask); because the per-hit MLP is pointwise and hits combine solely through the masked
aggregation, padded slots cannot influence the output -- so no other masking of
intermediate activations is needed.
"""

import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape

__all__ = [
  "SetRegressor", "EnsembleSetBlock", "EnsembleLinear", "EnsembleLeakyTanh", "FixedActivation", "make_activation",
  "masked_weighted_aggregate",
]


def masked_weighted_aggregate(value: jax.Array, weight_logit: jax.Array, mask: jax.Array):
  """Simple learned-weight aggregation over the hit (``M``) axis.

    ``value``, ``weight_logit``: ``(..., M, D)``; ``mask``: ``(..., M)`` int/bool.
    Aggregation is over ``M`` = the second-to-last axis (``-2``), so it is agnostic to the
    number of leading batch/ensemble dims (``(B, M, D)`` or ``(N, B, M, D)`` alike).

    Returns the normalised weighted average over the live hits,
    ``sum_i value_i * softplus(w_i) / (sum_i softplus(w_i) + 1)``, dropping the ``M`` axis.
    Padded hits are gated out by the mask. The ``+1`` in the denominator keeps the
    aggregate bounded (and finite when all gates vanish), so the event representation does
    not grow with the number of hits.
    """
  m = mask.astype(jnp.float32)[..., None]  # (..., M, 1)
  gate = jax.nn.softplus(weight_logit) * m  # (..., M, D); padded hits -> 0 gate
  weighted_sum = jnp.sum(value * gate, axis=-2)  # (..., D)
  norm = jnp.sum(gate, axis=-2) + 1.0  # (..., D)
  return weighted_sum / norm


# --------------------------------------------------------------------------- #
# Layers that serve a single net (``n_models=None``: no leading axis, plain
# affine/activation) or an ensemble (``n_models=n``: a leading member axis, every
# op batched over it via einsum/broadcast). One code path for both. Parameters are
# *constructed* with nnx; the forward is bare JAX so all members evaluate at once.
# --------------------------------------------------------------------------- #
class EnsembleLinear(nnx.Module):
  """Affine map, optionally replicated over a leading ensemble axis.

    ``n_models=None``: ``(..., in) -> (..., out)`` (one shared map). ``n_models=n``:
    ``(N, ..., in) -> (N, ..., out)``, member ``k`` using ``kernel[k]`` / ``bias[k]``.
    Members (and the single net) use a Lecun-style normal kernel and zero bias.
    """

  def __init__(self, n_models: int | None, in_dim: int, out_dim: int, *, rngs: nnx.Rngs, dropconnect: float | None = None):
    self.n_models = n_models
    self.in_dim = int(in_dim)
    # Drops individual WEIGHTS during training. `None` is a strict no-op: no mask, no key drawn.
    self.dropconnect = None if dropconnect is None else float(dropconnect)
    if self.dropconnect is not None and not 0.0 <= self.dropconnect < 1.0:
      raise ValueError(f"dropconnect must be None or in [0, 1), got {dropconnect}")
    std = 1.0 / math.sqrt(in_dim)
    kernel_shape = (in_dim, out_dim) if n_models is None else (n_models, in_dim, out_dim)
    bias_shape = (out_dim, ) if n_models is None else (n_models, out_dim)
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), kernel_shape) * std)
    self.bias = nnx.Param(jnp.zeros(bias_shape))

  def regularization(self):
    """``-log p(kernel)`` up to a constant, under the prior that maps a standard normal INPUT to a
        standard normal OUTPUT.

        ``y_j = sum_i W_ji x_i`` with ``x ~ N(0, I)`` has variance ``sum_i W_ji^2``, so ``y ~ N(0, 1)``
        needs ``W_ji ~ N(0, 1/in_dim)`` -- which is the Lecun scale this layer is already INITIALISED
        at, so the prior and the initialisation are the same distribution. The negative log density is
        then ``in_dim * ||W||^2 / 2``, and at initialisation it equals HALF THE KERNEL'S PARAMETER
        COUNT (every standardised coordinate contributes 1/2), which is the check to run on it.

        The BIAS and the activation gains are deliberately absent. A bias is an offset, not a map, so
        no input-to-output variance argument applies to it; and the gains initialise at 1.0 while a
        penalty pulls toward 0, which would drag the nonlinearity from ``tanh(x) + x`` toward plain
        ``tanh(x)`` -- a prior on the SHAPE of the activation rather than on the size of the network's
        weights.
        """
    return 0.5 * self.in_dim * jnp.sum(jnp.square(self.kernel[...]))

  def _kernel(self, deterministic, rngs):
    """The kernel this call multiplies by: the parameter, or a DropConnect draw of it.

    The mask carries the kernel's own shape, so EACH ENSEMBLE MEMBER GETS ITS OWN MASK. One mask per
    call; inverted-scaled, so evaluation uses the plain parameter.
    """
    kernel = self.kernel[...]
    if self.dropconnect is None or deterministic or self.dropconnect == 0.0:
      return kernel
    if rngs is None or not hasattr(rngs, "__getitem__"):
      raise ValueError(
        "dropconnect needs an `nnx.Rngs` on the forward call; a bare key would give "
        "every layer the same mask"
      )
    keep = 1.0 - self.dropconnect
    mask = jax.random.bernoulli(rngs['dropconnect'](), keep, kernel.shape).astype(kernel.dtype)
    return kernel * mask / keep

  def __call__(self, x, *, deterministic: bool = True, rngs=None):
    kernel = self._kernel(deterministic, rngs)
    if self.n_models is None:
      return jnp.einsum("...i,io->...o", x, kernel) + self.bias[...]
    # member axis kept, bias broadcast over the middle (...) axes.
    bias_shape = (x.shape[0], ) + (1, ) * (x.ndim - 2) + (-1, )
    return jnp.einsum("n...i,nio->n...o", x, kernel) + self.bias[...].reshape(bias_shape)


class EnsembleLeakyTanh(nnx.Module):
  """Per-feature (and per-member, when ensembled) LeakyTanh.

    ``tanh(x) + pos*softplus(x) - neg*softplus(-x)`` with learned per-feature gains.
    ``n_models=None``: gains ``(dim,)``; ``n_models=n``: gains ``(n, dim)`` broadcast over
    the middle axes.
    """

  def __init__(self, n_models: int | None, dim: int):
    self.n_models = n_models
    shape = (dim, ) if n_models is None else (n_models, dim)
    self.positive = nnx.Param(jnp.ones(shape))
    self.negative = nnx.Param(jnp.ones(shape))

  def __call__(self, x):
    if self.n_models is None:
      pos, neg = self.positive[...], self.negative[...]
    else:
      shape = (x.shape[0], ) + (1, ) * (x.ndim - 2) + (x.shape[-1], )
      pos = self.positive[...].reshape(shape)
      neg = self.negative[...].reshape(shape)
    return jax.nn.tanh(x) + pos * jax.nn.softplus(x) - neg * jax.nn.softplus(-x)


# --------------------------------------------------------------------------- #
# Activations. ``leaky-tanh`` (the default, above) is LEARNABLE -- two gains per unit,
# i.e. capacity. The alternatives below are FIXED functions with no parameters at all, so
# selecting one removes that capacity without touching anything else in the architecture.
# --------------------------------------------------------------------------- #
_FIXED_ACTIVATIONS = {
  # LeakyTanh FROZEN AT ITS OWN INITIALISATION. `softplus(x) - softplus(-x) == x` exactly,
  # so `tanh(x) + 1*softplus(x) - 1*softplus(-x)` is `tanh(x) + x` -- the same SHAPE the
  # learnable version starts from (unbounded, slope 2 at the origin, slope 1 asymptotically)
  # with none of its learnable capacity. This is the CONTROL that separates "the shape is too
  # expressive" from "the per-unit parameters are the capacity".
  "fixed-leaky-tanh": lambda x: jax.nn.tanh(x) + x,
  "tanh": jax.nn.tanh,  # BOUNDED (|f| <= 1) and slope 1 at the origin: the least expressive option
  "relu": jax.nn.relu,  # unbounded, piecewise linear
  "gelu": jax.nn.gelu,  # unbounded, smooth
  "celu": jax.nn.celu,
}


class FixedActivation(nnx.Module):
  """A parameter-free activation selected by name (see ``_FIXED_ACTIVATIONS``).

    The name is a plain ``str`` attribute, not an ``nnx.Param``, so the module contributes
    NOTHING to the parameter pytree -- which is the point: it is the zero-capacity comparison
    against :class:`EnsembleLeakyTanh`.
    """

  def __init__(self, name: str):
    if name not in _FIXED_ACTIVATIONS:
      known = sorted(_FIXED_ACTIVATIONS) + ["leaky-tanh"]
      raise ValueError(f"unknown activation {name!r}; known: {known}")
    self.name = name

  def __call__(self, x):
    return _FIXED_ACTIVATIONS[self.name](x)


def make_activation(name: str, n_models: int | None, dim: int):
  """``leaky-tanh`` -> the learnable per-unit activation; anything else -> a fixed function."""
  if name == "leaky-tanh":
    return EnsembleLeakyTanh(n_models, dim)
  return FixedActivation(name)


class EnsembleSetBlock(nnx.Module):
  """Shared per-hit MLP block producing ``(value, weight_logit)`` of shape
    ``(..., M, out_dim)`` each; the weight logit becomes a non-negative aggregation gate
    via ``softplus``. Serves a single net (``n_models=None``) or an ensemble; when
    ensembled, dropout's mask is sampled over the full ``(N, ...)`` tensor so each member
    drops independently.
    """

  def __init__(
    self, n_models: int | None, in_dim: int, block_def: Sequence[int], p_dropout: float | None = None,
    activation: str = "leaky-tanh", *, rngs: nnx.Rngs, dropconnect: float | None = None,
  ):
    if len(block_def) < 1:
      raise ValueError("block_def must contain at least one output dimension")

    hidden_dims = tuple(block_def[:-1])
    out_dim = int(block_def[-1])

    layers = []
    prev = in_dim
    for h in hidden_dims:
      # linear -> activation -> dropout. Dropout acts on ACTIVATIONS only: never on a block's input
      # (the raw data for the first block, `[value, aggregate]` for the rest) and never on an
      # aggregate. `p_dropout` None or 0 builds no layer at all.
      layers.append(EnsembleLinear(n_models, prev, h, rngs=rngs, dropconnect=dropconnect))
      layers.append(make_activation(activation, n_models, h))
      if p_dropout is not None and p_dropout > 0:
        layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
      prev = h
    self.shared = nnx.List(layers)
    # EXEMPT from dropconnect: this map emits the value AND the aggregation gate, and noise in the
    # aggregation weights perturbs the pooling itself rather than the map being regularised.
    self.output = EnsembleLinear(n_models, prev, 2 * out_dim, rngs=rngs)

  def regularization(self):
    """This block's kernels: the shared MLP's linears plus its own output map. Dropout and the
        activations contribute nothing -- the first has no parameters, the second has no kernel."""
    shared = sum(layer.regularization() for layer in self.shared if isinstance(layer, EnsembleLinear))
    return shared + self.output.regularization()

  def __call__(self, x, *, deterministic: bool = True, rngs=None):
    h = x
    for layer in self.shared:
      if isinstance(layer, (nnx.Dropout, EnsembleLinear)):
        # Explicit rng (a key/Rngs) so dropout is functional under jit/scan; the linears take
        # it for DropConnect and ignore it when that is off.
        h = layer(h, deterministic=deterministic, rngs=rngs)
      else:
        h = layer(h)  # activations: no randomness
    h = self.output(h, deterministic=deterministic, rngs=rngs)

    mu, sigma_raw = jnp.split(h, 2, axis=-1)

    return mu, sigma_raw


class SetRegressor(Model):
  """``(features, mask) -> predictions`` set regressor, optionally an ensemble.

    Parameters
    ----------
    n_features_in : per-hit feature dimension produced by ``detector.combine``.
    target_dim : ``T``.
    features : sequence of block definitions; each is a sequence of hidden widths whose
        last element is the block's output width. Successive blocks see ``2 * out_dim_prev``
        features (the hit's own ``value`` plus the aggregated event representation broadcast
        back to each hit).
    n_models : ``None`` for a single network, or an int ``>= 1`` for an ensemble of that
        many independent members (``1`` is a one-member ensemble: a leading axis of size 1).
    p_dropout : optional dropout rate on the shared MLPs' ACTIVATIONS -- the block order is
        ``linear -> activation -> dropout``, so it never touches a block's input (the raw features for
        the first block, ``[value, aggregate]`` for the rest), never an aggregate, and never follows a
        block's output map. ``None`` (or 0) builds no dropout layer at all.
    dropconnect : optional rate for dropping individual WEIGHTS. The mask carries the kernel's shape,
        so each ensemble member gets its own; it is inverted-scaled, so evaluation uses the plain
        kernel. Applied to the hidden linears and the read-out head, and EXEMPT on each block's output
        map, which emits the aggregation gate. ``None`` (the default) is a strict no-op: no mask, no
        key drawn.
    """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, features: Sequence[Sequence[int]],
    n_models: int | None = None, p_dropout: float | None = None, activation: str = "leaky-tanh", *, rngs: nnx.Rngs,
    dropconnect: float | None = None,
  ):
    # The universal shapes: per-hit feature count = input_shape[-1], target dim = target_shape[0].
    self.rngs = rngs
    self.n_features_in = int(input_shape[-1])
    self.target_dim = int(target_shape[0])
    self.n_models = None if n_models is None else int(n_models)
    if self.n_models is not None and self.n_models < 1:
      raise ValueError("n_models must be None or an int >= 1")

    blocks: list[EnsembleSetBlock] = []
    n_in = self.n_features_in
    for block_def in features:
      blocks.append(
        EnsembleSetBlock(
          self.n_models, n_in, block_def, p_dropout=p_dropout, activation=activation, rngs=rngs, dropconnect=dropconnect
        )
      )
      # After aggregation the next block sees [value_hit, event_repr].
      n_in = 2 * int(block_def[-1])
    self.blocks = nnx.List(blocks)
    self.output = EnsembleLinear(self.n_models, int(features[-1][-1]), self.target_dim, rngs=rngs, dropconnect=dropconnect)

  def ensemble(self) -> int | None:
    return self.n_models

  def regularization(self):
    """Every kernel this regressor owns: one term per set block, plus the read-out map. Each is
        scaled by its own fan-in (:meth:`EnsembleLinear.regularization`), so a wide layer and a narrow
        one are penalised on the same footing rather than by raw parameter count. For an ensemble the
        members' kernels are summed, since the members are independent draws from the same prior."""
    return sum(block.regularization() for block in self.blocks) + self.output.regularization()

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    # features: (..., M, F); mask: (..., M). Leading axes are (B,) for a single net or
    # (N, B) for an ensemble; all ops act on the trailing (M, feature) axes. Masking
    # lives solely in masked_weighted_aggregate (see module docstring).
    result = features

    *rest, last = self.blocks
    for block in rest:
      value_hit, weight_hit = block(result, deterministic=deterministic, rngs=rngs)
      event_repr = masked_weighted_aggregate(value_hit, weight_hit, mask)  # (..., D)
      event_per_hit = jnp.broadcast_to(jnp.expand_dims(event_repr, -2), value_hit.shape)
      result = jnp.concatenate([value_hit, event_per_hit], axis=-1)

    value_hit, weight_hit = last(result, deterministic=deterministic, rngs=rngs)
    event_repr = masked_weighted_aggregate(value_hit, weight_hit, mask)
    return self.output(event_repr, deterministic=deterministic, rngs=rngs)
