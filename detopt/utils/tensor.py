"""Generic pytree <-> flat-tensor codec used to pack typed records (Event/Target/...) into a single
``(*batch, features)`` tensor and back.

The "def" returned by :func:`structure` / :func:`flatten` is ``(treedef, leaf_specs, batch_dimensions)``
where each ``leaf_specs[i]`` is a :class:`jax.ShapeDtypeStruct` of the leaf's NON-batch shape **and
dtype** -- so :func:`unflatten` restores both the structure and each leaf's original dtype (the
concatenated tensor itself necessarily takes the common promoted dtype). ``structure`` also accepts a
tree of ``jax.ShapeDtypeStruct`` (e.g. a ``*_spec`` record), so a def can be built straight from a
spec without materialising data.

Batch dims are moved to the front before flattening, but the transpose is taken ONLY when they are
not already the leading axes (the common ``batch_dimensions=(0,)`` / ``()`` cases reshape directly).
"""

import math

import jax
import jax.numpy as jnp

__all__ = ["structure", "flatten", "unflatten"]


def _batch_front_perm(ndim, batch_dimensions):
    """Permutation moving ``batch_dimensions`` to the front (their given order), the rest after
    (ascending). Identity iff the batch dims are already the leading axes in order."""
    rest = tuple(i for i in range(ndim) if i not in batch_dimensions)
    return tuple(batch_dimensions) + rest


def _inverse_perm(perm):
    inverse = [0] * len(perm)
    for position, axis in enumerate(perm):
        inverse[axis] = position
    return tuple(inverse)


def structure(tree, batch_dimensions=()):
    """Capture a pytree's invertible layout as ``(treedef, leaf_specs, batch_dimensions)``.

    ``leaf_specs[i]`` is a :class:`jax.ShapeDtypeStruct` of leaf ``i``'s shape with the batch
    dimensions removed, plus its dtype. Works on a tree of arrays or of ``jax.ShapeDtypeStruct``
    (both expose ``.shape`` / ``.dtype``)."""
    batch_dimensions = tuple(batch_dimensions)
    leaves, treedef = jax.tree.flatten(tree)
    leaf_specs = [
        jax.ShapeDtypeStruct(tuple(s for i, s in enumerate(x.shape) if i not in batch_dimensions), x.dtype) for x in leaves
    ]
    return treedef, leaf_specs, batch_dimensions


def flatten(tree, batch_dimensions=(0,)):
    """Pack a pytree's leaves into one ``(*batch, features)`` tensor + an invertible def.

    Each leaf has its ``batch_dimensions`` moved to the front (transposed only if not already
    leading), is flattened over the remaining dims, and the leaves are concatenated on the last
    axis. Per-leaf dtype is recorded in the def; :func:`unflatten` restores it. Returns ``(tensor, def)``."""
    batch_dimensions = tuple(batch_dimensions)
    tree = jax.tree.map(lambda x: x if isinstance(x, jax.Array) else jnp.asarray(x), tree)
    treedef, leaf_specs, _ = structure(tree, batch_dimensions)

    blocks = []
    for x in jax.tree.leaves(tree):
        perm = _batch_front_perm(x.ndim, batch_dimensions)
        x = x if perm == tuple(range(x.ndim)) else jnp.transpose(x, perm)  # transpose only when needed
        n_batch = len(batch_dimensions)
        blocks.append(x.reshape((*x.shape[:n_batch], math.prod(x.shape[n_batch:]))))

    return jnp.concatenate(blocks, axis=-1), (treedef, leaf_specs, batch_dimensions)


def unflatten(tensor_def, tensor, axis=-1):
    """Inverse of :func:`flatten`: split ``tensor`` along ``axis`` per leaf, restore each leaf's
    non-batch shape and dtype, move the batch dims back to their original positions (only when
    needed), and rebuild the pytree from ``tensor_def`` (as returned by :func:`structure`/:func:`flatten`)."""
    treedef, leaf_specs, batch_dimensions = tensor_def
    axis = tensor.ndim + axis if axis < 0 else axis
    leading = tuple(slice(None) for _ in range(axis))

    blocks, offset = [], 0
    for spec in leaf_specs:
        feature_shape = tuple(spec.shape)
        size = math.prod(feature_shape)
        block = tensor[(*leading, slice(offset, offset + size))]
        offset += size
        block = block.reshape((*block.shape[:axis], *feature_shape, *block.shape[axis + 1 :])).astype(spec.dtype)
        ndim = len(batch_dimensions) + len(feature_shape)
        perm = _batch_front_perm(ndim, batch_dimensions)
        if perm != tuple(range(ndim)):
            block = jnp.transpose(block, _inverse_perm(perm))
        blocks.append(block)

    return jax.tree.unflatten(treedef, blocks)
