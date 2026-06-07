"""Detector-agnostic set regressor.

Consumes ``(features: (B, M, F), mask: (B, M))`` and produces ``(B, T)``.
All detector-specific feature engineering (design lookups, normalisation)
lives in :meth:`detopt.detector.Detector.combine`; this module only knows
about per-hit feature dimensionality and the target dimensionality.

The architecture is a stack of shared per-hit MLP blocks interleaved with a
simple learned-weight set aggregation over the ``M`` axis (each hit emits a
value and a non-negative gate ``softplus(w_raw)``; the event representation is
the normalised weighted average
``sum_i value_i * softplus(w_raw_i) / (sum_i softplus(w_raw_i) + 1)`` over the
live hits), followed by a linear
head on the pooled representation. Operates directly on padded ``(B, M, F)``
tensors instead of a flat ragged layout.
"""

import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from .common import LeakyTanh, Model

__all__ = [
    "SetRegressor",
    "SetBlock",
    "masked_weighted_aggregate",
    "SetEnsembleRegressor",
    "EnsembleSetBlock",
    "EnsembleLinear",
    "EnsembleLeakyTanh",
]


class SetBlock(nnx.Module):
    """Shared per-hit MLP block applied along the last axis.

    Produces ``(value, weight_logit)`` of shape ``(B, M, out_dim)`` each; the
    weight logit becomes a non-negative aggregation gate via ``softplus``.
    """

    def __init__(
        self,
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
            layers.append(nnx.Linear(prev, h, rngs=rngs))
            layers.append(LeakyTanh(h))
            prev = h
        self.shared = nnx.List(layers)
        self.value_head = nnx.Linear(prev, out_dim, rngs=rngs)
        self.weight_head = nnx.Linear(prev, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True, rngs=None):
        h = x
        for layer in self.shared:
            if isinstance(layer, nnx.Dropout):
                # Pass an explicit rng (a key/Rngs) so dropout is functional under
                # jit/scan; it takes precedence over the module's own rng state.
                h = layer(h, deterministic=deterministic, rngs=rngs)
            else:
                h = layer(h)
        return self.value_head(h), self.weight_head(h)


def masked_weighted_aggregate(
    value: jax.Array,
    weight_logit: jax.Array,
    mask: jax.Array,
):
    """Simple learned-weight aggregation over the hit (``M``) axis.

    ``value``, ``weight_logit``: ``(..., M, D)``; ``mask``: ``(..., M)`` int/bool.
    Aggregation is over ``M`` = the second-to-last axis (``-2``), so it is agnostic
    to the number of leading batch dims (``(B, M, D)`` or the ensemble layout
    ``(N, B, M, D)`` alike).

    Returns the normalised weighted average over the live hits,
    ``sum_i value_i * softplus(w_i) / (sum_i softplus(w_i) + 1)``, dropping the
    ``M`` axis. Padded hits are gated out by the mask. The ``+1`` in the
    denominator keeps the aggregate bounded (and finite when all gates vanish),
    so the event representation does not grow with the number of hits.
    """
    m = mask.astype(jnp.float32)[..., None]  # (..., M, 1)
    gate = jax.nn.softplus(weight_logit) * m  # (..., M, D)
    weighted_sum = jnp.sum(value * gate, axis=-2)  # (..., D)
    norm = jnp.sum(gate, axis=-2) + 1.0  # (..., D)
    return weighted_sum / norm


class SetRegressor(Model):
    """``(features, mask) -> (B, target_dim)`` regressor.

    Parameters
    ----------
    n_features_in : per-hit feature dimension produced by ``detector.combine``.
    target_dim : ``T``.
    features : sequence of block definitions; each is a sequence of hidden
        widths whose last element is the block's output width. Successive
        blocks see ``2 * out_dim_prev`` features (the hit's own ``value`` plus
        the aggregated event representation broadcast back to each hit).
    p_dropout : optional dropout rate for the shared MLPs.
    """

    @classmethod
    def from_config(cls, detector, config, *, rngs: nnx.Rngs):
        return cls(
            n_features_in=int(detector.combined_feature_dim),
            target_dim=int(detector.target_dim()),
            rngs=rngs,
            **config,
        )

    def __init__(
        self,
        n_features_in: int,
        target_dim: int,
        features: Sequence[Sequence[int]],
        p_dropout: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        # Do not call Model.__init__ -- it reads detector-specific shapes
        # we no longer rely on.
        self.rngs = rngs
        self.n_features_in = int(n_features_in)
        self.target_dim = int(target_dim)

        blocks: list[SetBlock] = []
        n_in = self.n_features_in
        for block_def in features:
            block = SetBlock(in_dim=n_in, block_def=block_def, p_dropout=p_dropout, rngs=rngs)
            blocks.append(block)
            out_dim = int(block_def[-1])
            # After aggregation the next block sees [value_hit, event_repr].
            n_in = 2 * out_dim
        self.blocks = nnx.List(blocks)

        last_out = int(features[-1][-1])
        self.output = nnx.Linear(last_out, self.target_dim, rngs=rngs)

    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        # features: (B, M, F_in); mask: (B, M). ``rngs`` (a key/Rngs) drives
        # dropout when training; for eval pass deterministic=True (rngs unused).
        m = mask.astype(jnp.float32)[..., None]  # (B, M, 1)
        result = features * m

        *rest, last = self.blocks

        for block in rest:
            value_hit, weight_hit = block(result, deterministic=deterministic, rngs=rngs)
            value_hit = value_hit * m
            event_repr = masked_weighted_aggregate(value_hit, weight_hit, mask)
            # Broadcast the event representation back to each hit.
            event_per_hit = jnp.broadcast_to(event_repr[:, None, :], value_hit.shape)
            result = jnp.concatenate([value_hit, event_per_hit], axis=-1) * m

        value_hit, weight_hit = last(result, deterministic=deterministic, rngs=rngs)
        value_hit = value_hit * m
        event_repr = masked_weighted_aggregate(value_hit, weight_hit, mask)
        return self.output(event_repr)


# --------------------------------------------------------------------------- #
# Ensemble: ``N`` independent set regressors sharing one architecture, stacked
# on a leading ensemble axis. Parameters are *constructed* with nnx (each leaf
# carries the extra ``N`` axis); every forward op is bare JAX (an einsum over the
# members), so one call evaluates all members at once -- no per-member Python loop
# and no nnx machinery in the forward. Trainers feed ``N`` independent minibatches
# at training time (one per member, drawn independently from the same pool) and
# average the ``N`` member predictions at evaluation.
# --------------------------------------------------------------------------- #


class EnsembleLinear(nnx.Module):
    """``N`` independent affine maps applied over a leading ensemble axis.

    ``__call__`` maps ``(N, ..., in_dim) -> (N, ..., out_dim)``; member ``k`` uses
    ``kernel[k]`` / ``bias[k]``. Members are initialised independently
    (Lecun-normal kernel, zero bias).
    """

    def __init__(self, n_models: int, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        std = 1.0 / math.sqrt(in_dim)
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (n_models, in_dim, out_dim)) * std)
        self.bias = nnx.Param(jnp.zeros((n_models, out_dim)))

    def __call__(self, x):
        # x: (N, ..., in) -> (N, ..., out); bias broadcast over the middle axes.
        bias_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (-1,)
        return jnp.einsum("n...i,nio->n...o", x, self.kernel[...]) + self.bias[...].reshape(bias_shape)


class EnsembleLeakyTanh(nnx.Module):
    """Per-member, per-feature :class:`~detopt.nn.common.LeakyTanh`.

    Gains carry a leading ensemble axis; ``__call__`` acts on ``(N, ..., dim)``
    tensors, broadcasting each member's gains over the middle axes.
    """

    def __init__(self, n_models: int, dim: int):
        self.positive = nnx.Param(jnp.ones((n_models, dim)))
        self.negative = nnx.Param(jnp.ones((n_models, dim)))

    def __call__(self, x):
        shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)
        pos = self.positive[...].reshape(shape)
        neg = self.negative[...].reshape(shape)
        return jax.nn.tanh(x) + pos * jax.nn.softplus(x) - neg * jax.nn.softplus(-x)


class EnsembleSetBlock(nnx.Module):
    """Ensemble counterpart of :class:`SetBlock` (a leading ``N`` axis throughout).

    Dropout is shared structurally across members but its mask is sampled over the
    full ``(N, ...)`` tensor, so each member drops independently.
    """

    def __init__(
        self,
        n_models: int,
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
        self.value_head = EnsembleLinear(n_models, prev, out_dim, rngs=rngs)
        self.weight_head = EnsembleLinear(n_models, prev, out_dim, rngs=rngs)

    def __call__(self, x, *, deterministic: bool = True, rngs=None):
        h = x
        for layer in self.shared:
            if isinstance(layer, nnx.Dropout):
                h = layer(h, deterministic=deterministic, rngs=rngs)
            else:
                h = layer(h)
        return self.value_head(h), self.weight_head(h)


class SetEnsembleRegressor(Model):
    """Ensemble of ``n_models`` independent :class:`SetRegressor` networks.

    Architecturally identical to :class:`SetRegressor`, but every parameter carries
    a leading ensemble axis so all members evaluate in one batched (einsum) pass.
    ``__call__`` maps ``(N, B, M, F), (N, B, M) -> (N, B, T)`` (member ``k`` sees
    slice ``k``). :meth:`ensemble` returns ``n_models``, so trainers feed ``N``
    independent minibatches at training time and average the ``N`` predictions at
    evaluation.
    """

    @classmethod
    def from_config(cls, detector, config, *, rngs: nnx.Rngs):
        return cls(
            n_features_in=int(detector.combined_feature_dim),
            target_dim=int(detector.target_dim()),
            rngs=rngs,
            **config,
        )

    def __init__(
        self,
        n_features_in: int,
        target_dim: int,
        features: Sequence[Sequence[int]],
        n_models: int,
        p_dropout: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.n_features_in = int(n_features_in)
        self.target_dim = int(target_dim)
        self.n_models = int(n_models)
        if self.n_models < 1:
            raise ValueError("n_models must be >= 1")

        blocks: list[EnsembleSetBlock] = []
        n_in = self.n_features_in
        for block_def in features:
            blocks.append(EnsembleSetBlock(self.n_models, n_in, block_def, p_dropout=p_dropout, rngs=rngs))
            n_in = 2 * int(block_def[-1])
        self.blocks = nnx.List(blocks)
        self.output = EnsembleLinear(self.n_models, int(features[-1][-1]), self.target_dim, rngs=rngs)

    def ensemble(self) -> int:
        return self.n_models

    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        # features: (N, B, M, F); mask: (N, B, M). The hit axis M is axis 2.
        m = mask.astype(jnp.float32)[..., None]  # (N, B, M, 1)
        result = features * m

        *rest, last = self.blocks

        for block in rest:
            value_hit, weight_hit = block(result, deterministic=deterministic, rngs=rngs)
            value_hit = value_hit * m
            event_repr = masked_weighted_aggregate(value_hit, weight_hit, mask)  # (N, B, D)
            event_per_hit = jnp.broadcast_to(event_repr[:, :, None, :], value_hit.shape)
            result = jnp.concatenate([value_hit, event_per_hit], axis=-1) * m

        value_hit, weight_hit = last(result, deterministic=deterministic, rngs=rngs)
        value_hit = value_hit * m
        event_repr = masked_weighted_aggregate(value_hit, weight_hit, mask)
        return self.output(event_repr)  # (N, B, T)
