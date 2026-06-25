"""Predictive (fully causal) set regressor, for the layer-wise stereo detector.

A :class:`~detopt.nn.set_regressor.SetRegressor` variant with NO ensemble whose aggregation is a
CUMULATIVE causal average with a learnable prior, instead of a single pooled average. For ordered
elements (the layers of :class:`StereoLayerWise`, station -> view -> layer), element ``i``'s
representation depends only on elements ``j <= i`` -- so the network emits a *running* per-layer
estimate that sharpens as more layers are seen, seeded by a learned prior (element 0).

Aggregation per block (replacing ``masked_weighted_aggregate``): with ``w_j = softplus(weight_j)*mask_j``
and a learnable prior ``latent_0`` (weight ``w_0 = 1``),

    agg_i = (1 * latent_0 + sum_{j=0..i} w_j * value_j) / (1 + sum_{j=0..i} w_j)        (cumulative)

``agg`` stays per-element ``(B, M, D)`` (NOT collapsed) and is concatenated back per element with NO
broadcast, so causality is preserved through every block. The head maps each element to a prediction,
giving a per-layer stack ``(B, M, T)``; ``__call__`` returns the LAST layer as the point prediction.
The :meth:`loss` override trains with deep supervision: the last layer at full weight, the earlier
layers at a small ``aux_weight``.
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .common import Model
from .set_regressor import EnsembleSetBlock, EnsembleLinear, EnsembleLeakyTanh

__all__ = ["PredictiveSetRegressor", "PredictiveProbRegressor", "PredictiveMixtureRegressor",
           "CausalSetBlock", "causal_cumulative_aggregate"]


def causal_cumulative_aggregate(value, weight_logit, mask, latent_0):
    """Causal cumulative aggregation over the element (``M``) axis with a learnable prior.

    ``value``, ``weight_logit``: ``(..., M, D)``; ``mask``: ``(..., M)``; ``latent_0``: ``(D,)`` prior
    (weight ``w_0 = 1``). Returns ``(..., M, D)`` where element ``i`` aggregates only elements
    ``j <= i`` (an inclusive cumulative sum), so the result is causal/streaming."""
    m = mask.astype(jnp.float32)[..., None]  # (..., M, 1)
    gate = jax.nn.softplus(weight_logit) * m  # (..., M, D); padded elements -> 0 gate
    cum_num = jnp.cumsum(value * gate, axis=-2) + latent_0  # (..., M, D)  prior in numerator (w_0=1)
    cum_den = jnp.cumsum(gate, axis=-2) + 1.0  # (..., M, D)  prior weight w_0=1 in denominator
    return cum_num / cum_den


class CausalSetBlock(nnx.Module):
    """A set block (shared per-element MLP) whose pooling is the causal cumulative aggregate, with a
    per-block learnable prior. Returns ``[value, agg]`` per element ``(..., M, 2*out_dim)`` -- NO
    broadcast, so the next block (and the head) stay causal."""

    def __init__(self, in_dim, block_def, p_dropout=None, *, rngs: nnx.Rngs):
        self.block = EnsembleSetBlock(None, in_dim, block_def, p_dropout=p_dropout, rngs=rngs)
        self.latent_0 = nnx.Param(jnp.zeros((int(block_def[-1]),)))  # learnable prior, dim = block output

    def __call__(self, x, mask, *, deterministic: bool = True, rngs=None):
        value, weight_logit = self.block(x, deterministic=deterministic, rngs=rngs)  # (..., M, D) each
        agg = causal_cumulative_aggregate(value, weight_logit, mask, self.latent_0[...])  # (..., M, D)
        return jnp.concatenate([value, agg], axis=-1)  # (..., M, 2D)


class PredictiveSetRegressor(Model):
    """``(features, mask) -> predictions`` causal set regressor (single net only).

    ``features`` is the per-block hidden-width list (same as :class:`SetRegressor`); ``aux_weight`` is
    the deep-supervision weight on the non-final per-layer losses."""

    def __init__(self, input_shape, target_shape, ground_truth_shape, features, aux_weight=0.3,
                 p_dropout=None, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.n_features_in = int(input_shape[-1])
        self.target_dim = int(target_shape[0])
        self.aux_weight = float(aux_weight)

        blocks, n_in = [], self.n_features_in
        for block_def in features:
            blocks.append(CausalSetBlock(n_in, block_def, p_dropout=p_dropout, rngs=rngs))
            n_in = 2 * int(block_def[-1])  # next block sees [value, agg]
        self.blocks = nnx.List(blocks)
        self.output = EnsembleLinear(None, n_in, self.target_dim, rngs=rngs)  # per-element head on [value, agg]


    def _blocks_state(self, features, mask, *, deterministic=True, rngs=None):
        """All causal blocks -> the per-element TOP-LEVEL state ``(..., M, 2D)`` (the head's input)."""
        h = features
        for block in self.blocks:
            h = block(h, mask, deterministic=deterministic, rngs=rngs)  # (..., M, 2D)
        return h

    def _per_element(self, features, mask, *, deterministic=True, rngs=None):
        """All blocks + head -> per-element (per-layer) prediction stack ``(..., M, T)``."""
        return self.output(self._blocks_state(features, mask, deterministic=deterministic, rngs=rngs))

    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        per = self._per_element(features, mask, deterministic=deterministic, rngs=rngs)  # (..., M, T)
        return per[..., -1, :]  # LAST element (last layer) = point prediction (..., T)

    def _next_layer_loss(self, features, state):
        """Auxiliary per-sample loss term from the next-layer prediction head. Base: none. Subclasses
        (prob / mixture) override -- this is why ``Model.loss`` owns the forward (to read ``features``)."""
        return 0.0

    def loss(self, loss_fn, features, mask, target, *, deterministic=True, rngs=None):
        """Deep-supervision loss ``loss(last) + aux_weight * mean(loss(earlier))`` + the subclass's
        next-layer term. One forward (the top-level state is shared by the head and the next-layer head).

        ``loss_fn`` (the detector's per-sample loss) must reduce only over the target's last axis and
        broadcast leading axes -- so feeding ``(..., M, T)`` vs ``(..., 1, T)`` yields ``(..., M)``."""
        state = self._blocks_state(features, mask, deterministic=deterministic, rngs=rngs)  # (..., M, 2D)
        per_layer = loss_fn(self.output(state), target[..., None, :])  # (..., M) per-sample, per-layer
        deep_sup = per_layer[..., -1] + self.aux_weight * jnp.mean(per_layer[..., :-1], axis=-1)  # (...,)
        return deep_sup + self._next_layer_loss(features, state)

    # ---- shared next-layer-head helpers (used by the Prob / Mixture subclasses) ---- #
    # The next-layer heads pair with `stereo_layerwise`, whose per-layer feature layout is
    # [station_z(0), view(1), layer(2), angle(3), y_offset(4), TDC grid(5:)].
    _N_POS = 5
    _GEOM_IDX = (0, 3, 4)  # station_z, angle, y_offset -> the next layer's location/angle/y-offset

    def _build_head(self, in_dim, out_dim, head_features, rngs):
        layers, prev = [], int(in_dim)
        for h in head_features:
            layers.append(EnsembleLinear(None, prev, int(h), rngs=rngs))
            layers.append(EnsembleLeakyTanh(None, int(h)))
            prev = int(h)
        layers.append(EnsembleLinear(None, prev, int(out_dim), rngs=rngs))
        return nnx.List(layers)

    def _head_inputs(self, features, state):
        """``(x, grid_next)`` for the next-layer heads: ``x = concat(state_i, geom_{i+1})`` over
        ``i=0..M-2`` and ``grid_next`` = layer i+1's TDC grid. Both ``(..., M-1, ·)``."""
        geom = features[..., jnp.asarray(self._GEOM_IDX)]  # (..., M, 3) next-layer location/angle/y-offset
        grid = features[..., self._N_POS:]  # (..., M, n_straws)
        x = jnp.concatenate([state[..., :-1, :], geom[..., 1:, :]], axis=-1)  # (..., M-1, 2D+3)
        return x, grid[..., 1:, :]  # predict layer i+1 from layer i's state


class PredictiveProbRegressor(PredictiveSetRegressor):
    """Predictive regressor + a next-layer HIT-PROBABILITY head: from layer *i*'s top-level state (and
    the next layer's location/angle/y-offset) predict per-straw logits for where hits land on layer
    *i+1*; trained with a BCE term against the next layer's actual hit pattern (``TDC > 0``)."""

    def __init__(self, input_shape, target_shape, ground_truth_shape, features, aux_weight=0.3,
                 p_dropout=None, head_features=(64,), prob_weight=0.1, *, rngs: nnx.Rngs):
        super().__init__(input_shape, target_shape, ground_truth_shape, features, aux_weight=aux_weight,
                         p_dropout=p_dropout, rngs=rngs)
        self.prob_weight = float(prob_weight)
        n_straws = self.n_features_in - self._N_POS
        head_in = 2 * int(features[-1][-1]) + len(self._GEOM_IDX)
        self.head = self._build_head(head_in, n_straws, head_features, rngs)

    def _next_layer_loss(self, features, state):
        x, grid_next = self._head_inputs(features, state)  # (..., M-1, 2D+3), (..., M-1, n_straws)
        logits = x
        for layer in self.head:
            logits = layer(logits)  # (..., M-1, n_straws)
        presence = (grid_next > 0).astype(jnp.float32)
        bce = optax.sigmoid_binary_cross_entropy(logits, presence)  # (..., M-1, n_straws)
        return self.prob_weight * jnp.mean(bce, axis=(-1, -2))  # (...,) per-sample


class PredictiveMixtureRegressor(PredictiveSetRegressor):
    """Predictive regressor + a next-layer MIXTURE-DENSITY head: from layer *i*'s top-level state predict
    a ``n_components``-Gaussian mixture over the normalized straw position ``linspace(-1, 1, n_straws)``;
    trained with the NLL of the next layer's FIRED straw positions (``TDC > 0``) under that mixture."""

    def __init__(self, input_shape, target_shape, ground_truth_shape, features, aux_weight=0.3,
                 p_dropout=None, head_features=(64,), n_components=4, mixture_weight=0.1, *, rngs: nnx.Rngs):
        super().__init__(input_shape, target_shape, ground_truth_shape, features, aux_weight=aux_weight,
                         p_dropout=p_dropout, rngs=rngs)
        self.n_components = int(n_components)
        self.mixture_weight = float(mixture_weight)
        n_straws = self.n_features_in - self._N_POS
        head_in = 2 * int(features[-1][-1]) + len(self._GEOM_IDX)
        self.head = self._build_head(head_in, 3 * self.n_components, head_features, rngs)
        self._n_straws = int(n_straws)  # plain int (no jax array on the module -> nnx.split-safe)

    def _next_layer_loss(self, features, state):
        x, grid_next = self._head_inputs(features, state)  # (..., M-1, 2D+3), (..., M-1, n_straws)
        params = x
        for layer in self.head:
            params = layer(params)  # (..., M-1, 3K)
        w_logits, mu_raw, log_std = jnp.split(params, 3, axis=-1)  # each (..., M-1, K)
        log_w = jax.nn.log_softmax(w_logits, axis=-1)
        mu = jnp.tanh(mu_raw)  # component means in [-1, 1]
        sigma = jax.nn.softplus(log_std) + 1e-3  # > 0
        # log N(straw_pos | mu_k, sigma_k) over (..., M-1, n_straws, K). straw_norm (n_straws,1) broadcasts
        # against (..., M-1, 1, K) -> (..., M-1, n_straws, K).
        straw_norm = jnp.linspace(-1.0, 1.0, self._n_straws, dtype=jnp.float32)  # (n_straws,) target positions
        d = straw_norm[:, None] - mu[..., None, :]  # (..., M-1, n_straws, K)
        log_norm = -0.5 * jnp.square(d / sigma[..., None, :]) - jnp.log(sigma[..., None, :]) \
            - 0.5 * jnp.log(2.0 * jnp.pi)
        log_mix = jax.scipy.special.logsumexp(log_w[..., None, :] + log_norm, axis=-1)  # (..., M-1, n_straws)
        fired = (grid_next > 0).astype(jnp.float32)  # (..., M-1, n_straws)
        nll = -(log_mix * fired).sum(axis=(-1, -2)) / (fired.sum(axis=(-1, -2)) + 1e-6)  # (...,) per-sample
        return self.mixture_weight * nll
