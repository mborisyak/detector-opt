"""Masked (denoising) set regressor over layer-wise features.

A single-net set regressor on layers (works with ``StereoLayerWise``) with a self-supervised
auxiliary task. Two passes per training step:

  1. **normal** -- the usual layer set regressor -> the detector's ``loss_fn``.
  2. **masked**  -- a copy of the input has some TDC-grid cells *flipped* to a random state; the LAST
     block's per-layer embeddings feed a head that predicts WHICH cells were flipped (per-cell BCE).

``final loss = loss_fn + mask_weight * masked_cross_entropy``.

The flip mask is ``uniform_uniform``: a per-event threshold ``u ~ U[0,1]`` is drawn, then each cell flips
iff its own ``U[0,1]`` draw ``< u`` (so the corrupted fraction is itself ``~U[0,1]``). A flipped cell is
redrawn from a fixed distribution fit to the real present-TDC histogram (Gamma over ``tdc_norm``); with
probability ``1 - flip_present_prob`` it instead becomes ABSENT (``-1``) -- so flips cover fake hits,
deleted hits, and perturbed TDCs. Defaults were measured on our-sim data (median ``tdc_norm`` ~ 1 after
``tdc_scale=145``): Gamma(shape 2.888, scale 0.357).
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .common import Model
from .set_regressor import EnsembleSetBlock, EnsembleLinear, EnsembleLeakyTanh, masked_weighted_aggregate

__all__ = ["MaskedSetRegressor"]


class MaskedSetRegressor(Model):
    _N_POS = 5  # stereo_layerwise layout: [station_z, view, layer, angle, y_offset] then the TDC grid

    def __init__(self, input_shape, target_shape, ground_truth_shape, features, head_features=(64,),
                 mask_weight=0.3, flip_tdc_shape=2.888, flip_tdc_scale=0.357, flip_present_prob=0.5,
                 p_dropout=None, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.n_features_in = int(input_shape[-1])
        self.target_dim = int(target_shape[0])
        self.mask_weight = float(mask_weight)
        self.flip_tdc_shape = float(flip_tdc_shape)
        self.flip_tdc_scale = float(flip_tdc_scale)
        self.flip_present_prob = float(flip_present_prob)
        self._n_straws = self.n_features_in - self._N_POS

        blocks, n_in = [], self.n_features_in
        for bd in features:
            blocks.append(EnsembleSetBlock(None, n_in, bd, p_dropout=p_dropout, rngs=rngs))
            n_in = 2 * int(bd[-1])
        self.blocks = nnx.List(blocks)
        d_last = int(features[-1][-1])
        self.output = EnsembleLinear(None, d_last, self.target_dim, rngs=rngs)

        # flip-detection head: last-block per-layer embedding (d_last) -> per-straw flip logits
        head, prev = [], d_last
        for h in head_features:
            head.append(EnsembleLinear(None, prev, int(h), rngs=rngs))
            head.append(EnsembleLeakyTanh(None, int(h)))
            prev = int(h)
        head.append(EnsembleLinear(None, prev, self._n_straws, rngs=rngs))
        self.head = nnx.List(head)


    def _body(self, features, mask, deterministic, rngs):
        """Set-regressor body -> ``(prediction (..., T), last_block_value (..., M, D))``. The last
        block's per-element ``value`` (before its aggregation) is the per-layer embedding."""
        result = features
        *rest, last = self.blocks
        for block in rest:
            value, weight = block(result, deterministic=deterministic, rngs=rngs)
            event_repr = masked_weighted_aggregate(value, weight, mask)  # (..., D)
            result = jnp.concatenate([value, jnp.broadcast_to(event_repr[..., None, :], value.shape)], axis=-1)
        value, weight = last(result, deterministic=deterministic, rngs=rngs)
        event_repr = masked_weighted_aggregate(value, weight, mask)
        return self.output(event_repr), value

    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        return self._body(features, mask, deterministic, rngs)[0]

    def _corrupt(self, features, key):
        """``uniform_uniform`` flip of the TDC grid -> ``(corrupted_features, flip_mask (..., M, n_straws))``."""
        grid = features[..., self._N_POS:]  # (..., M, n_straws)
        ku, kf, kp, kg = jax.random.split(key, 4)
        u = jax.random.uniform(ku, grid.shape[:-2] + (1, 1))  # per-event threshold (..., 1, 1)
        flip = jax.random.uniform(kf, grid.shape) < u  # (..., M, n_straws)
        tdc = jax.random.gamma(kg, self.flip_tdc_shape, grid.shape) * self.flip_tdc_scale  # present draw (>0)
        present = jax.random.uniform(kp, grid.shape) < self.flip_present_prob
        new_val = jnp.where(present, tdc, -1.0)  # present(Gamma TDC) or absent(-1)
        new_grid = jnp.where(flip, new_val, grid)
        corrupted = jnp.concatenate([features[..., : self._N_POS], new_grid], axis=-1)
        return corrupted, flip.astype(jnp.float32)

    def _flip_logits(self, emb):
        h = emb
        for layer in self.head:
            h = layer(h)
        return h  # (..., M, n_straws)

    def loss(self, loss_fn, features, mask, target, *, deterministic=True, rngs=None):
        """Normal pass (``loss_fn``) + masked pass (flip-detection BCE). The masked pass needs an rng;
        without one (eval) it is skipped and only the regression loss is returned."""
        pred = self._body(features, mask, deterministic, rngs)[0]
        main = loss_fn(pred, target)  # (...,) per-sample
        if rngs is None:
            return main
        corrupted, flip_target = self._corrupt(features, rngs.corruption())
        _, emb = self._body(corrupted, mask, deterministic, rngs)  # (..., M, D)
        ce = optax.sigmoid_binary_cross_entropy(self._flip_logits(emb), flip_target)  # (..., M, n_straws)
        return main + self.mask_weight * jnp.mean(ce, axis=(-1, -2))  # (...,)
