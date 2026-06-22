"""Detector-agnostic event buffers for training scripts.

Both buffers hold a fixed, positional tuple of **pytree slots** on a JAX device, each
sized from a per-event *spec* -- either a record (a namedtuple of ``jax.ShapeDtypeStruct``,
e.g. ``event_spec()`` / ``target_spec()``) or a plain ``jax.ShapeDtypeStruct`` (e.g. the mask
or the flat physical design). Allocation/append/index all go through ``jax.tree``, so a slot
may carry mixed dtypes (int32 hit indices beside a float32 TDC) without special-casing.

**They store RAW records** -- raw events, raw targets, the raw physical design, raw ground
truth -- NOT design-``combine``d features. ``combine`` + ``normalize_target`` run per training
batch right before the network (encoded/normalised/combined forms are larger; re-deriving them
per batch is the intended trade). ``push``/``append`` and ``buffers`` are positional so callers
keep the ``ring.push(a, b, c)`` / ``*ring.buffers()`` idiom; each item is just a pytree now.
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["Pool", "RingBuffer"]


def _alloc(spec, capacity, device):
    """Allocate a zeroed buffer for one slot: prepend ``capacity`` to every leaf's shape.
    ``spec`` is a record (namedtuple of ShapeDtypeStruct) or a plain ShapeDtypeStruct."""
    return jax.tree.map(
        lambda s: jax.device_put(jnp.zeros((int(capacity),) + tuple(s.shape), s.dtype), device),
        spec,
    )


def _leading_len(item):
    """Leading (batch) length of a chunk -- the leading axis of any one of its leaves."""
    return int(jax.tree.leaves(item)[0].shape[0])


class Pool:
    """Append-only per-event buffers (a positional tuple of pytree slots), sized from specs.

    ``n_current`` is a runtime cursor; events are written contiguously and training kernels
    read the fixed-shape buffers gated on ``n_current``, so one compiled kernel handles a
    growing pool. The pool never calls the detector -- scripts sample events explicitly (so the
    detector calls stay visible) and append them here. Storing the raw physical design per event
    lets one pool span many designs (it accumulates across BO iterations) and lets the network's
    ``combine`` be design-conditioned -- each event is combined with its own design.
    """

    def __init__(self, n_max, specs, device=None):
        self.n_max = int(n_max)
        self.device = device
        self.specs = tuple(specs)
        self.slots = [_alloc(spec, self.n_max, device) for spec in self.specs]
        self.n_current = 0

    def append(self, *chunk):
        """Append a chunk of ``n`` examples (one positional item per slot, matching the specs)."""
        n = _leading_len(chunk[0])
        if self.n_current + n > self.n_max:
            raise RuntimeError(f"Pool overflow: {self.n_current} + {n} > {self.n_max}")
        s = self.n_current
        self.slots = [
            jax.tree.map(lambda b, c: b.at[s : s + n].set(jnp.asarray(c)), buf, item) for buf, item in zip(self.slots, chunk)
        ]
        self.n_current += n

    def buffers(self):
        """The stored slots as a positional tuple of pytrees (full capacity)."""
        return tuple(self.slots)


class RingBuffer:
    """Fixed-capacity FIFO ring of recent examples (a positional tuple of pytree slots).

    Unlike :class:`Pool` (append-only, sized to the whole budget), the ring keeps only the most
    recent ``capacity`` examples: ``push`` overwrites the oldest once full via a modular cursor.
    ``state``/``load_state`` round-trip the slots + cursor so the ring survives a checkpoint.
    """

    def __init__(self, capacity, specs, device=None):
        self.capacity = int(capacity)
        self.device = device
        self.specs = tuple(specs)
        self.slots = [_alloc(spec, self.capacity, device) for spec in self.specs]
        self.cursor = 0  # next write position (mod capacity)
        self.n_filled = 0  # number of valid rows so far (<= capacity)

    def push(self, *chunk):
        """Write a chunk of ``n`` examples (one positional item per slot), overwriting the oldest
        once the ring is full."""
        n = _leading_len(chunk[0])
        if n > self.capacity:  # an oversized chunk: keep only its last `capacity` rows
            chunk = tuple(jax.tree.map(lambda c: c[-self.capacity :], item) for item in chunk)
            n = self.capacity
        idx = (self.cursor + jnp.arange(n)) % self.capacity
        self.slots = [
            jax.tree.map(lambda b, c: b.at[idx].set(jnp.asarray(c)), buf, item) for buf, item in zip(self.slots, chunk)
        ]
        self.cursor = int((self.cursor + n) % self.capacity)
        self.n_filled = int(min(self.n_filled + n, self.capacity))

    def buffers(self):
        """The stored slots as a positional tuple of pytrees (full capacity)."""
        return tuple(self.slots)

    def __len__(self):
        return self.n_filled

    def state(self):
        """Serialisable snapshot (slots + cursor) for checkpointing."""
        return {"slots": tuple(self.slots), "cursor": np.int32(self.cursor), "n_filled": np.int32(self.n_filled)}

    def load_state(self, state):
        """Restore from a :meth:`state` snapshot."""
        self.slots = [jax.tree.map(lambda a: jax.device_put(jnp.asarray(a), self.device), slot) for slot in state["slots"]]
        self.cursor = int(state["cursor"])
        self.n_filled = int(state["n_filled"])
