"""Continuous-convolutional (kernel message-passing) regressor over per-hit features.

Each hit is a graph node with features `(B, M, F)` (from `StereoHits`). A KERNEL NET maps every
ordered hit PAIR `concat(feat_i, feat_j)` to a non-negative per-channel gate `K (B, M, M, d)`; node *i*
aggregates a masked weighted mean of all hits *j*:

    agg_i = sum_j (K[i,j] * value_j * mask_j) / (sum_j K[i,j] * mask_j + 1)

The body mirrors `SetRegressor` -- `value = block(h); agg = aggregate(value, K, mask); h = concat([value,
agg])` -- then a final masked mean over hits feeds a linear head. Single net (no ensemble).

Two variants (`kernel_per_block`): **False** -> one kernel computed ONCE from the raw `(B,M,F)` pairs and
reused every block (every block's value dim = `k`, so all `features` widths must be equal); **True** -> a
kernel net PER block, recomputed from that block's hidden pairs (`last_k -> new_k`).
"""
import jax
import jax.numpy as jnp
from flax import nnx

from .common import Model
from .set_regressor import EnsembleLinear, EnsembleLeakyTanh

__all__ = ["ContinuousConvRegressor"]


def _mlp(in_dim, widths, out_dim, p_dropout, rngs):
    """Plain per-element MLP `(..., in) -> (..., out)` (LeakyTanh between; optional dropout)."""
    layers, prev = [], int(in_dim)
    for h in widths:
        if p_dropout is not None and p_dropout > 0:
            layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
        layers.append(EnsembleLinear(None, prev, int(h), rngs=rngs))
        layers.append(EnsembleLeakyTanh(None, int(h)))
        prev = int(h)
    layers.append(EnsembleLinear(None, prev, int(out_dim), rngs=rngs))
    return nnx.List(layers)


def _apply_mlp(mlp, x, deterministic, rngs):
    for layer in mlp:
        x = layer(x, deterministic=deterministic, rngs=rngs) if isinstance(layer, nnx.Dropout) else layer(x)
    return x


def kernel_aggregate(K, value, mask):
    """Message-passing aggregation: node *i* gathers a masked weighted mean of all hits *j*.

    ``K (..., M, M, d)`` (gate for ordered pair i<-j), ``value (..., M, d)``, ``mask (..., M)``.
    Returns ``(..., M, d)`` with ``agg_i = sum_j K[i,j]*value_j*mask_j / (sum_j K[i,j]*mask_j + 1)``."""
    gate = K * mask.astype(jnp.float32)[..., None, :, None]  # (..., M, M, d); gate the j axis
    num = jnp.sum(gate * value[..., None, :, :], axis=-2)  # sum over j -> (..., M, d)
    den = jnp.sum(gate, axis=-2) + 1.0  # (..., M, d)
    return num / den


class _PairKernel(nnx.Module):
    """Kernel net: from per-element features ``(..., M, in)`` build the ordered-pair gate
    ``(..., M, M, out)`` = ``softplus(MLP(concat(x_i, x_j)))`` (non-negative)."""

    def __init__(self, in_dim, kernel_features, out_dim, p_dropout=None, *, rngs: nnx.Rngs):
        self.mlp = _mlp(2 * int(in_dim), kernel_features, out_dim, p_dropout, rngs)

    def __call__(self, x, deterministic=True, rngs=None):
        M = x.shape[-2]
        xi = jnp.broadcast_to(x[..., :, None, :], x.shape[:-2] + (M, M, x.shape[-1]))
        xj = jnp.broadcast_to(x[..., None, :, :], xi.shape)
        pair = jnp.concatenate([xi, xj], axis=-1)  # (..., M, M, 2*in)
        return jax.nn.softplus(_apply_mlp(self.mlp, pair, deterministic, rngs))  # (..., M, M, out) >= 0


class ContinuousConvRegressor(Model):
    def __init__(self, input_shape, target_shape, ground_truth_shape, features, kernel_features=(32,),
                 kernel_per_block=False, p_dropout=None, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.n_features_in = int(input_shape[-1])
        self.target_dim = int(target_shape[0])
        self.kernel_per_block = bool(kernel_per_block)

        out_dims = [int(bd[-1]) for bd in features]
        if not self.kernel_per_block and len(set(out_dims)) != 1:
            raise ValueError(f"kernel_per_block=False needs all `features` block widths equal (the one "
                             f"shared kernel's k); got {out_dims}")

        blocks, kernels, n_in = [], [], self.n_features_in
        for bd in features:
            blocks.append(_mlp(n_in, list(bd[:-1]), int(bd[-1]), p_dropout, rngs))  # value MLP -> (..., M, d)
            if self.kernel_per_block:
                kernels.append(_PairKernel(n_in, kernel_features, int(bd[-1]), p_dropout=p_dropout, rngs=rngs))
            n_in = 2 * int(bd[-1])  # next block sees [value, agg]
        self.blocks = nnx.List(blocks)
        if self.kernel_per_block:
            self.kernels = nnx.List(kernels)
        else:  # ONE kernel from the raw input features, reused every block (k = the shared block width)
            self.kernel = _PairKernel(self.n_features_in, kernel_features, out_dims[0], p_dropout=p_dropout, rngs=rngs)
        self.output = EnsembleLinear(None, out_dims[-1], self.target_dim, rngs=rngs)


    def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
        K0 = None if self.kernel_per_block else self.kernel(features, deterministic, rngs)  # (..., M, M, k) once
        h = features
        value = None
        for i, block in enumerate(self.blocks):
            value = _apply_mlp(block, h, deterministic, rngs)  # (..., M, d)
            K = self.kernels[i](h, deterministic, rngs) if self.kernel_per_block else K0  # (..., M, M, d)
            agg = kernel_aggregate(K, value, mask)  # (..., M, d)
            h = jnp.concatenate([value, agg], axis=-1)  # (..., M, 2d)
        # final masked mean over hits of the last block's value -> linear head
        m = mask.astype(jnp.float32)[..., None]  # (..., M, 1)
        pooled = jnp.sum(value * m, axis=-2) / (jnp.sum(m, axis=-2) + 1.0)  # (..., d_last)
        return self.output(pooled)  # (..., target_dim)
