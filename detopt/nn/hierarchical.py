"""Hierarchical (nested) set regressors over ``StereoLayerWise``.

The momentum signal is a long-range cross-station straw correlation (the magnet bend) that a FLAT set
regressor over layers cannot resolve. These regressors aggregate along the detector's NESTING instead:

  * ``DoubleSetRegressor``      -- 2 levels:  straw -> layer -> global
  * ``StructuredSetRegressor``  -- 4 levels:  straw -> layer -> view -> station -> global

Both consume the layer-wise features `(B, n_layers, 5 + n_straws)` = `[station_z, view, layer, angle,
y_offset]` ++ dense TDC/energy grid. Per straw the embedding is `[straw_norm, TDC]` (the straw POSITION
is essential, else the within-layer set is permutation-blind). Aggregation is the dense weighted mean
`Σ(v·softplus(w))/(Σsoftplus(w)+1)` over ALL elements (`masked_weighted_aggregate` with a ones mask).

`features` is a list of block defs (as in `SetRegressor`); each entry is one aggregation ROUND. The same
block def is reused at every level but with SEPARATE parameters (and level-specific input dims). Flow is
a bottom-up sweep with a STALE parent: at round *l* each level aggregates its children from round *l*
(fresh) and conditions on its parent + the global from round *l-1*. Single net (no ensemble); final
`z_global -> EnsembleLinear -> target`.
"""
import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model
from .set_regressor import EnsembleSetBlock, EnsembleLinear, masked_weighted_aggregate

__all__ = ["DoubleSetRegressor", "StructuredSetRegressor"]


def _agg(value, weight_logit):
    """Dense weighted mean over axis -2: ``Σ v·softplus(w) / (Σ softplus(w) + 1)`` (ones mask = all
    elements contribute -- absent straws are real elements carrying TDC=-1, not masked out)."""
    mask = jnp.ones(value.shape[:-1], value.dtype)
    return masked_weighted_aggregate(value, weight_logit, mask)


def _ctx(emb, target_lead):
    """Broadcast a context embedding ``emb (B, *prefix, F)`` to the child's element lead
    ``target_lead = (B, *prefix, *extra)`` -> ``(*target_lead, F)`` (insert singleton axes for *extra)."""
    F = emb.shape[-1]
    n_extra = len(target_lead) - (emb.ndim - 1)
    return jnp.broadcast_to(emb.reshape(emb.shape[:-1] + (1,) * n_extra + (F,)), tuple(target_lead) + (F,))


class _Hier(Model):
    """Shared machinery: geometry, block stacks per node, single-net head, single-pass loss."""

    def __init__(self, input_shape, target_shape, ground_truth_shape, features, n_straws,
                 n_stations, n_views_per_station, n_layers_per_view, p_dropout=None, global_dim=32, *,
                 rngs: nnx.Rngs):
        # The straw/layer/view/station NESTING is a config hyper-parameter (the layer-wise input_shape only
        # carries (n_layers, 5 + n_straws), not the factorization); target dim from target_shape.
        self.rngs = rngs
        self.features = [list(f) for f in features]
        self.n_rounds = len(self.features)
        self.target_dim = int(target_shape[0])
        self.n_straws = int(n_straws)
        self.n_stations = int(n_stations)
        self.n_views = int(n_views_per_station)
        self.n_lpv = int(n_layers_per_view)
        self.n_layers = self.n_stations * self.n_views * self.n_lpv
        self.global_dim = int(global_dim)
        self.p_dropout = p_dropout
        self.global_init = nnx.Param(jax.random.normal(rngs.params(), (self.global_dim,)) * 0.1)
        self.output = EnsembleLinear(None, int(self.features[-1][-1]), self.target_dim, rngs=rngs)
        self._build_nodes(rngs)


    def _dprev(self, l, init_dim):
        """Dim of a node's embedding ENTERING round l: its static init dim at l==0, else last block out."""
        return init_dim if l == 0 else int(self.features[l - 1][-1])

    def _stack(self, static, self_init, parent_init, has_global, has_child, rngs):
        """One node = a per-round list of `EnsembleSetBlock`. ``parent_init`` = parent's init dim (None if
        the immediate parent IS global / there is no parent); ``has_global`` feeds the broadcast global."""
        blocks = []
        for l in range(self.n_rounds):
            in_dim = static + self._dprev(l, self_init)
            if parent_init is not None:
                in_dim += self._dprev(l, parent_init)
            if has_global:
                in_dim += self._dprev(l, self.global_dim)
            if has_child:
                in_dim += int(self.features[l][-1])  # fresh child aggregate (this round)
            blocks.append(EnsembleSetBlock(None, in_dim, self.features[l], p_dropout=self.p_dropout, rngs=rngs))
        return nnx.List(blocks)

    def loss(self, loss_fn, features, mask, target, *, deterministic=True, rngs=None):
        return loss_fn(self(features, mask, deterministic=deterministic, rngs=rngs), target)

    # subclasses implement _build_nodes(rngs) and __call__


class DoubleSetRegressor(_Hier):
    """straw -> layer -> global."""

    def _build_nodes(self, rngs):
        # straw: parent=layer (not global) + global; layer: parent IS global; global: top.
        self.straw = self._stack(2, 2, 5, True, False, rngs)
        self.layer = self._stack(5, 5, None, True, True, rngs)
        self.glob = self._stack(0, self.global_dim, None, False, True, rngs)

    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        B = features.shape[0]
        geom = features[..., :5]  # (B, n_layers, 5)
        grid = features[..., 5:]  # (B, n_layers, n_straws)
        straw_norm = jnp.linspace(-1.0, 1.0, self.n_straws, dtype=features.dtype)
        straw_static = jnp.stack([jnp.broadcast_to(straw_norm, grid.shape), grid], axis=-1)  # (B,nl,ns,2)
        h_straw, h_layer = straw_static, geom
        h_global = jnp.broadcast_to(self.global_init[...], (B, self.global_dim))
        for l in range(self.n_rounds):
            lead_s = h_straw.shape[:-1]
            x_s = jnp.concatenate([straw_static, h_straw, _ctx(h_layer, lead_s), _ctx(h_global, lead_s)], -1)
            v_s, w_s = self.straw[l](x_s, deterministic=deterministic, rngs=rngs)
            straw_agg = _agg(v_s, w_s)  # (B, n_layers, d)
            lead_l = h_layer.shape[:-1]
            x_l = jnp.concatenate([geom, h_layer, _ctx(h_global, lead_l), straw_agg], -1)
            v_l, w_l = self.layer[l](x_l, deterministic=deterministic, rngs=rngs)
            layer_agg = _agg(v_l, w_l)  # (B, d)
            x_g = jnp.concatenate([h_global, layer_agg], -1)[:, None, :]  # (B, 1, in)
            v_g, _ = self.glob[l](x_g, deterministic=deterministic, rngs=rngs)
            h_straw, h_layer, h_global = v_s, v_l, v_g[:, 0, :]
        return self.output(h_global)


class StructuredSetRegressor(_Hier):
    """straw -> layer -> view -> station -> global, mirroring the detector nesting."""

    def _build_nodes(self, rngs):
        # static dims: straw [straw_norm,TDC]=2, layer geom=5, view [station_z,view]=2, station [station_z]=1.
        self.straw = self._stack(2, 2, 5, True, False, rngs)   # parent=layer
        self.layer = self._stack(5, 5, 2, True, True, rngs)    # parent=view
        self.view = self._stack(2, 2, 1, True, True, rngs)     # parent=station
        self.station = self._stack(1, 1, None, True, True, rngs)  # parent IS global
        self.glob = self._stack(0, self.global_dim, None, False, True, rngs)

    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        B = features.shape[0]
        ns, nv, nl = self.n_stations, self.n_views, self.n_lpv
        geom = features[..., :5].reshape(B, ns, nv, nl, 5)
        grid = features[..., 5:].reshape(B, ns, nv, nl, self.n_straws)
        straw_norm = jnp.linspace(-1.0, 1.0, self.n_straws, dtype=features.dtype)
        straw_static = jnp.stack([jnp.broadcast_to(straw_norm, grid.shape), grid], axis=-1)  # (B,ns,nv,nl,nstraw,2)
        layer_static = geom                                  # (B, ns, nv, nl, 5)
        view_static = geom[:, :, :, 0, :2]                   # (B, ns, nv, 2) = [station_z, view]
        station_static = geom[:, :, 0, 0, :1]                # (B, ns, 1) = [station_z]

        h_straw, h_layer, h_view, h_station = straw_static, layer_static, view_static, station_static
        h_global = jnp.broadcast_to(self.global_init[...], (B, self.global_dim))
        for l in range(self.n_rounds):
            ls = h_straw.shape[:-1]
            x = jnp.concatenate([straw_static, h_straw, _ctx(h_layer, ls), _ctx(h_global, ls)], -1)
            v_s, w_s = self.straw[l](x, deterministic=deterministic, rngs=rngs)
            agg_s = _agg(v_s, w_s)  # over straws -> (B, ns, nv, nl, d)

            ll = h_layer.shape[:-1]
            x = jnp.concatenate([layer_static, h_layer, _ctx(h_view, ll), _ctx(h_global, ll), agg_s], -1)
            v_l, w_l = self.layer[l](x, deterministic=deterministic, rngs=rngs)
            agg_l = _agg(v_l, w_l)  # over n_lpv -> (B, ns, nv, d)

            lv = h_view.shape[:-1]
            x = jnp.concatenate([view_static, h_view, _ctx(h_station, lv), _ctx(h_global, lv), agg_l], -1)
            v_v, w_v = self.view[l](x, deterministic=deterministic, rngs=rngs)
            agg_v = _agg(v_v, w_v)  # over views -> (B, ns, d)

            lt = h_station.shape[:-1]
            x = jnp.concatenate([station_static, h_station, _ctx(h_global, lt), agg_v], -1)
            v_t, w_t = self.station[l](x, deterministic=deterministic, rngs=rngs)
            agg_t = _agg(v_t, w_t)  # over stations -> (B, d)

            x = jnp.concatenate([h_global, agg_t], -1)[:, None, :]
            v_g, _ = self.glob[l](x, deterministic=deterministic, rngs=rngs)
            h_straw, h_layer, h_view, h_station, h_global = v_s, v_l, v_v, v_t, v_g[:, 0, :]
        return self.output(h_global)
