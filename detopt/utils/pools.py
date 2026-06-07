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

__all__ = ["Pool"]


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
