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
    "SetRegressor",
    "EnsembleSetBlock",
    "EnsembleLinear",
    "EnsembleLeakyTanh",
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

    def __init__(self, n_models: int | None, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.n_models = n_models
        std = 1.0 / math.sqrt(in_dim)
        kernel_shape = (in_dim, out_dim) if n_models is None else (n_models, in_dim, out_dim)
        bias_shape = (out_dim,) if n_models is None else (n_models, out_dim)
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), kernel_shape) * std)
        self.bias = nnx.Param(jnp.zeros(bias_shape))

    def __call__(self, x):
        if self.n_models is None:
            return jnp.einsum("...i,io->...o", x, self.kernel[...]) + self.bias[...]
        # member axis kept, bias broadcast over the middle (...) axes.
        bias_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (-1,)
        return jnp.einsum("n...i,nio->n...o", x, self.kernel[...]) + self.bias[...].reshape(bias_shape)


class EnsembleLeakyTanh(nnx.Module):
    """Per-feature (and per-member, when ensembled) LeakyTanh.

    ``tanh(x) + pos*softplus(x) - neg*softplus(-x)`` with learned per-feature gains.
    ``n_models=None``: gains ``(dim,)``; ``n_models=n``: gains ``(n, dim)`` broadcast over
    the middle axes.
    """

    def __init__(self, n_models: int | None, dim: int):
        self.n_models = n_models
        shape = (dim,) if n_models is None else (n_models, dim)
        self.positive = nnx.Param(jnp.ones(shape))
        self.negative = nnx.Param(jnp.ones(shape))

    def __call__(self, x):
        if self.n_models is None:
            pos, neg = self.positive[...], self.negative[...]
        else:
            shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)
            pos = self.positive[...].reshape(shape)
            neg = self.negative[...].reshape(shape)
        return jax.nn.tanh(x) + pos * jax.nn.softplus(x) - neg * jax.nn.softplus(-x)


class EnsembleSetBlock(nnx.Module):
    """Shared per-hit MLP block producing ``(value, weight_logit)`` of shape
    ``(..., M, out_dim)`` each; the weight logit becomes a non-negative aggregation gate
    via ``softplus``. Serves a single net (``n_models=None``) or an ensemble; when
    ensembled, dropout's mask is sampled over the full ``(N, ...)`` tensor so each member
    drops independently.
    """

    def __init__(
        self,
        n_models: int | None,
        in_dim: int,
        block_def: Sequence[int],
        p_dropout: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        if len(block_def) < 1:
            raise ValueError("block_def must contain at least one output dimension")

        hidden_dims = tuple(block_def[:-1])
        out_dim = int(block_def[-1])

        layers = []
        prev = in_dim
        for h in hidden_dims:
            if p_dropout is not None and p_dropout > 0:
                layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
            layers.append(EnsembleLinear(n_models, prev, h, rngs=rngs))
            layers.append(EnsembleLeakyTanh(n_models, h))
            prev = h
        self.shared = nnx.List(layers)
        self.output = EnsembleLinear(n_models, prev, 2 * out_dim, rngs=rngs)

    def __call__(self, x, *, deterministic: bool = True, rngs=None):
        h = x
        for layer in self.shared:
            if isinstance(layer, nnx.Dropout):
                # Explicit rng (a key/Rngs) so dropout is functional under jit/scan.
                h = layer(h, deterministic=deterministic, rngs=rngs)
            else:
                h = layer(h)
        h = self.output(h)

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
    p_dropout : optional dropout rate for the shared MLPs.
    """

    def __init__(
        self,
        input_shape: Shape,
        target_shape: Shape,
        ground_truth_shape: Shape,
        features: Sequence[Sequence[int]],
        n_models: int | None = None,
        p_dropout: float | None = None,
        *,
        rngs: nnx.Rngs,
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
            blocks.append(EnsembleSetBlock(self.n_models, n_in, block_def, p_dropout=p_dropout, rngs=rngs))
            # After aggregation the next block sees [value_hit, event_repr].
            n_in = 2 * int(block_def[-1])
        self.blocks = nnx.List(blocks)
        self.output = EnsembleLinear(self.n_models, int(features[-1][-1]), self.target_dim, rngs=rngs)

    def ensemble(self) -> int | None:
        return self.n_models

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
        return self.output(event_repr)
