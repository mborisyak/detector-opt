"""Pair set regressor: a deep set over all PAIRS of measurements (single net, no ensemble).

Instead of treating the ``M`` hits as the set, this builds the ``M*M`` ordered pairs and runs the
SAME masked deep-set body as :class:`SetRegressor` (shared per-element MLP blocks interleaved with
learned-weight aggregation, then a linear head) over the pair axis. Each pair element is the two
hits' :meth:`combine` features concatenated with one PAIR-WISE comparison feature PER input
feature: ``tanh(coeff_f * (feat_i[f] - feat_j[f]))``. A pair is live only when both its hits are
live (``mask_i AND mask_j``).

The per-feature ``coefficients`` are supplied by the config -- the network does NOT measure
anything. Their natural value is ``1 / (typical daughter-pair Δfeature)`` so a typical difference
drives the ``tanh`` to order 1; measure those deltas with the standalone
``scripts/measure_pair_deltas.py`` and paste the reciprocals into the network config.
"""

from typing import Sequence

import jax.nn
import jax.numpy as jnp
import jax.nn as jnn

from flax import nnx

from ..detector import Detector
from .common import Model, Shape
from .set_regressor import EnsembleSetBlock, EnsembleLinear, masked_weighted_aggregate

__all__ = ["PairSetRegressor"]


class PairSetRegressor(Model):
    """``(features, mask) -> predictions`` deep set over the ``M*M`` hit pairs. Single net only.

    Parameters mirror :class:`SetRegressor` (``features`` = per-block hidden-width lists,
    ``p_dropout``), plus ``coefficients``: a length-``F`` sequence scaling each feature's
    pair-comparison ``tanh(coeff_f * Δfeature)`` (``None`` -> all ones). Set a feature's
    coefficient to 0 to drop its comparison.
    """

    def __init__(
        self,
        input_shape: Shape,
        target_shape: Shape,
        ground_truth_shape: Shape,
        features: Sequence[Sequence[int]],
        coefficients: Sequence[float] | None = None,
        p_dropout: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.n_features_in = int(input_shape[-1])
        self.target_dim = int(target_shape[0])

        coeffs = [1.0] * self.n_features_in if coefficients is None else list(coefficients)
        if len(coeffs) != self.n_features_in:
            raise ValueError(f"coefficients must have one entry per feature ({self.n_features_in}); got {len(coeffs)}")
        # Fixed (non-learned) per-feature scale for the pair comparison. Stored as a static tuple
        # (not a jax array) so nnx does not treat it as trainable/variable state.
        self.coefficients = tuple(float(c) for c in coeffs)

        blocks: list[EnsembleSetBlock] = []
        n_in = 3 * self.n_features_in  # pair element: [feat_i, feat_j, tanh(coeff * (feat_i - feat_j))]
        for block_def in features:
            blocks.append(EnsembleSetBlock(None, n_in, block_def, p_dropout=p_dropout, rngs=rngs))
            n_in = 2 * int(block_def[-1])  # next block sees [value_pair, event_repr]
        self.blocks = nnx.List(blocks)
        self.output = EnsembleLinear(None, int(features[-1][-1]), self.target_dim, rngs=rngs)


    def _pairs(self, features, mask):
        """``(B, M, F), (B, M)`` -> pair features ``(B, M*M, 3F)`` and pair mask ``(B, M*M)``."""
        B, M, F = features.shape
        fi = features[:, :, None, :]  # (B, M, 1, F)
        fj = features[:, None, :, :]  # (B, 1, M, F)
        coeff = jnp.asarray(self.coefficients, jnp.float32)  # (F,)
        cmp = jnp.tanh(coeff * (fi - fj))  # (B, M, M, F); per-feature scaled comparison
        pair = jnp.concatenate([jnp.broadcast_to(fi, (B, M, M, F)), jnp.broadcast_to(fj, (B, M, M, F)), cmp], axis=-1)
        pair_mask = mask[:, :, None] * mask[:, None, :]  # (B, M, M); live only if both hits live
        return pair.reshape(B, M * M, 3 * F), pair_mask.reshape(B, M * M)

    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        # features: (B, M, F); mask: (B, M). Pairs become the set; masking lives solely in
        # masked_weighted_aggregate (over the pair axis -2), as in SetRegressor.
        result, pair_mask = self._pairs(features, mask)

        *rest, last = self.blocks
        for block in rest:
            value, weight = block(result, deterministic=deterministic, rngs=rngs)
            event_repr = masked_weighted_aggregate(value, weight, pair_mask)
            event_per_pair = jnp.broadcast_to(jnp.expand_dims(event_repr, -2), value.shape)
            result = jnp.concatenate([value, event_per_pair], axis=-1)

        value, weight = last(result, deterministic=deterministic, rngs=rngs)
        return self.output(masked_weighted_aggregate(value, weight, pair_mask))

class TrackerRegressor(Model):
    """``(features, mask) -> predictions`` deep set over the ``M*M`` hit pairs. Single net only.

    Parameters mirror :class:`SetRegressor` (``features`` = per-block hidden-width lists,
    ``p_dropout``), plus ``coefficients``: a length-``F`` sequence scaling each feature's
    pair-comparison ``tanh(coeff_f * Δfeature)`` (``None`` -> all ones). Set a feature's
    coefficient to 0 to drop its comparison.
    """

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

        main: list[EnsembleSetBlock] = []
        induction: list[EnsembleSetBlock] = []

        *body_defs, head_def = features
        for block_def in body_defs:
            main.append(Block(2 * n_in, block_def, p_dropout=p_dropout, rngs=rngs))
            induction.append(Block(2 * n_in, block_def, p_dropout=p_dropout, rngs=rngs))

            n_in = int(block_def[-1])

        self.main = nnx.List(main)
        self.induction = nnx.List(induction)

        *head_hidden, head_output = head_def
        self.head = Block(2 * n_in, (*head_hidden, 2 * head_output), p_dropout=p_dropout, rngs=rngs)

        self.output = nnx.Linear(head_output, n_t, rngs=rngs)


    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        n_b, *_ = features.shape
        x = features
        inducing = self.initial_inducing_points[...]
        inducing = jnp.broadcast_to(inducing[None], shape=(n_b, *inducing.shape))

        for block_x, block_inducing in zip(self.main, self.induction):
            x_inducing, inducing_x = cross_attention(x, inducing, mask_xs=mask)
            x = jnp.concatenate([x, x_inducing], axis=-1)
            inducing = jnp.concatenate([inducing, inducing_x], axis=-1)

            x = block_x(x, deterministic=deterministic, rngs=rngs)
            inducing = block_inducing(inducing, deterministic=deterministic, rngs=rngs)

        x_inducing, _ = cross_attention(x, inducing, mask_xs=mask)
        x = jnp.concatenate([x, x_inducing], axis=-1)

        xw = self.head(x)
        x, w = jnp.split(xw, 2, axis=-1)

        return self.output(masked_weighted_aggregate(x, w, mask))
