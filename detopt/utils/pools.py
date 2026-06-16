"""Detector-agnostic event pool for training scripts.

A :class:`Pool` holds preallocated per-event buffers on a JAX device, sized
from **detector-provided shapes** (it knows nothing about hits): ``measurements``,
``mask``, ``targets``, and the per-event ``designs`` (the encoded design that
generated each event -- one entry per event). ``n_current`` is a runtime cursor;
events are written contiguously, and training kernels read fixed-shape buffers
gated on ``n_current`` so one compiled kernel handles a growing pool.

Storing the design per event lets a single pool span *many* designs (the buffer
accumulates across BO iterations) and lets the network's ``combine`` be
design-conditioned -- each event is combined with its own design.

The pool never calls the detector -- scripts sample events explicitly (so the
detector calls are visible there) and append them here.
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["Pool", "RingBuffer"]


class Pool:
    """Preallocated per-event buffers on a JAX device, sized from shapes."""

    def __init__(
        self,
        n_max,
        measurement_shape,
        mask_shape,
        target_shape,
        design_shape,
        device=None,
    ):
        self.n_max = int(n_max)
        self.device = device

        def alloc(per_event_shape, dtype):
            zeros = jnp.zeros((self.n_max, *tuple(per_event_shape)), dtype=dtype)
            return jax.device_put(zeros, device=device)

        self.X = alloc(measurement_shape, jnp.float32)
        self.mask = alloc(mask_shape, jnp.int32)
        self.targets = alloc(target_shape, jnp.float32)
        self.designs = alloc(design_shape, jnp.float32)
        self.n_current = 0

    def append(self, X, mask, targets, designs):
        """Append a chunk: leading axis ``n``, remaining axes match the shapes."""
        n = X.shape[0]
        if self.n_current + n > self.n_max:
            raise RuntimeError(f"Pool overflow: {self.n_current} + {n} > {self.n_max}")
        s = self.n_current
        self.X = self.X.at[s : s + n].set(jnp.asarray(X))
        self.mask = self.mask.at[s : s + n].set(jnp.asarray(mask))
        self.targets = self.targets.at[s : s + n].set(jnp.asarray(targets))
        self.designs = self.designs.at[s : s + n].set(jnp.asarray(designs))
        self.n_current += n

    def buffers(self):
        """The four per-event buffers ``(X, mask, targets, designs)``.

        ``designs`` is the encoded design that generated each event (one entry
        per event), so ``combine`` can be design-conditioned.
        """
        return self.X, self.mask, self.targets, self.designs


class RingBuffer:
    """Fixed-capacity FIFO ring of recent training examples on a JAX device.

    Unlike :class:`Pool` (append-only, sized to the whole budget), the ring keeps only
    the most recent ``capacity`` examples: ``push`` overwrites the oldest once full via a
    modular cursor. It stores **network-ready** ``features`` (already design-``combine``d),
    the per-event ``mask``, and the normalised ``targets`` -- exactly what a scan-folded
    train step samples minibatches from. ``state``/``load_state`` round-trip the arrays +
    cursor so the ring survives a checkpoint/restore.
    """

    def __init__(self, capacity, feature_shape, mask_shape, target_shape, device=None):
        self.capacity = int(capacity)
        self.device = device

        def alloc(per_event_shape, dtype):
            zeros = jnp.zeros((self.capacity, *tuple(per_event_shape)), dtype=dtype)
            return jax.device_put(zeros, device=device)

        self.features = alloc(feature_shape, jnp.float32)
        self.mask = alloc(mask_shape, jnp.int32)
        self.targets = alloc(target_shape, jnp.float32)
        self.cursor = 0  # next write position (mod capacity)
        self.n_filled = 0  # number of valid rows so far (<= capacity)

    def push(self, features, mask, targets):
        """Write a chunk of ``n`` examples, overwriting the oldest once the ring is full."""
        features, mask, targets = (jnp.asarray(features), jnp.asarray(mask), jnp.asarray(targets))
        n = features.shape[0]
        if n > self.capacity:  # an oversized chunk: keep only its last `capacity` rows
            features, mask, targets = features[-self.capacity :], mask[-self.capacity :], targets[-self.capacity :]
            n = self.capacity
        idx = (self.cursor + jnp.arange(n)) % self.capacity
        self.features = self.features.at[idx].set(features)
        self.mask = self.mask.at[idx].set(mask)
        self.targets = self.targets.at[idx].set(targets)
        self.cursor = int((self.cursor + n) % self.capacity)
        self.n_filled = int(min(self.n_filled + n, self.capacity))

    def buffers(self):
        """The three per-example buffers ``(features, mask, targets)`` (full capacity)."""
        return self.features, self.mask, self.targets

    def __len__(self):
        return self.n_filled

    def state(self):
        """Serialisable snapshot (arrays + cursor) for checkpointing."""
        return {
            "features": self.features,
            "mask": self.mask,
            "targets": self.targets,
            "cursor": np.int32(self.cursor),
            "n_filled": np.int32(self.n_filled),
        }

    def load_state(self, state):
        """Restore from a :meth:`state` snapshot."""
        self.features = jax.device_put(jnp.asarray(state["features"]), self.device)
        self.mask = jax.device_put(jnp.asarray(state["mask"]), self.device)
        self.targets = jax.device_put(jnp.asarray(state["targets"]), self.device)
        self.cursor = int(state["cursor"])
        self.n_filled = int(state["n_filled"])
