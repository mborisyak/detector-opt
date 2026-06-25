"""Hierarchical convolutional regressor over the ``StereoImage`` ``(n_layers, n_straws, 4)``.

Reinterprets the layer-wise data as an image and convolves it with straw / layer / view / station
convolutions whose strides mirror the fixed station->view->layer ordering of the layer axis:

    straw -> layer -> straw -> view -> straw -> station -> max-pool -> linear

- **straw conv** -- kernel ``(1, M)``, stride ``(1, 1)``: convolves the straw (width) axis within each
  layer row (VALID, so the straw axis shrinks by ``M-1`` each).
- **layer / view / station conv** -- kernel ``(k, 1)``, stride ``(k, 1)`` with ``k`` =
  ``n_layers_per_view`` / ``n_views_per_station`` / ``n_stations``: each VALID stride-=kernel conv
  exactly tiles and collapses one level of the hierarchy (``n_layers -> n_st*n_vps -> n_st -> 1``).

All convs are VALID with ``LeakyTanh`` between; a global MAX-pool over the straw axis then feeds a
linear head. Single net (no ensemble); the standard ``Model.loss`` default. The geometry
(``n_stations`` / ``n_views_per_station`` / ``n_layers_per_view``) is config-supplied and asserted
against the image's layer axis.
"""
import jax.numpy as jnp
from flax import nnx

from .common import Model
from .set_regressor import EnsembleLeakyTanh

__all__ = ["ConvRegressor"]


class ConvRegressor(Model):
    def __init__(self, input_shape, target_shape, ground_truth_shape,
                 n_stations, n_views_per_station, n_layers_per_view, straw_kernel, channels,
                 p_dropout=None, *, rngs: nnx.Rngs):
        # image input_shape = (n_layers, n_straws, in_channels); the station/view/layer factorization is a
        # config hyper-parameter (validated against n_layers below).
        self.rngs = rngs
        in_channels, n_layers, n_straws = int(input_shape[-1]), int(input_shape[0]), int(input_shape[1])
        self.target_dim = int(target_shape[0])
        n_st, n_vps, n_lpv = int(n_stations), int(n_views_per_station), int(n_layers_per_view)
        if n_st * n_vps * n_lpv != int(n_layers):
            raise ValueError(f"conv geometry {n_st}*{n_vps}*{n_lpv}={n_st * n_vps * n_lpv} != image n_layers {n_layers}")
        if len(channels) != 6:
            raise ValueError(
                f"`channels` must list 6 conv widths (straw/layer/straw/view/straw/station); got {len(channels)}")
        M = int(straw_kernel)
        c = [int(x) for x in channels]

        # (kernel, stride) per conv, in order: straw, layer-level, straw, view-level, straw, station-level.
        specs = [((1, M), (1, 1)), ((n_lpv, 1), (n_lpv, 1)), ((1, M), (1, 1)),
                 ((n_vps, 1), (n_vps, 1)), ((1, M), (1, 1)), ((n_st, 1), (n_st, 1))]
        convs, acts = [], []
        f_in = int(in_channels)
        for (ksize, strides), f_out in zip(specs, c):
            convs.append(nnx.Conv(f_in, f_out, kernel_size=ksize, strides=strides, padding="VALID", rngs=rngs))
            acts.append(EnsembleLeakyTanh(None, f_out))
            f_in = f_out
        self.convs = nnx.List(convs)
        self.acts = nnx.List(acts)
        self.dropout = nnx.Dropout(rate=p_dropout, rngs=rngs) if (p_dropout is not None and p_dropout > 0) else None
        self.output = nnx.Linear(c[-1], self.target_dim, rngs=rngs)


    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        # features: (B, n_layers, n_straws, in_channels); mask ignored (absence is encoded as TDC=-1).
        h = features
        for conv, act in zip(self.convs, self.acts):
            h = act(conv(h))
            if self.dropout is not None:
                h = self.dropout(h, deterministic=deterministic, rngs=rngs)
        # h: (B, 1, n_straws', C) -- the layer axis is collapsed to 1. MAX-pool over the spatial axes.
        pooled = jnp.max(h, axis=(1, 2))  # (B, C)
        return self.output(pooled)  # (B, target_dim)
