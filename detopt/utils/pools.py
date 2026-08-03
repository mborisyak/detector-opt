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
    lambda s: jnp.zeros(shape=(capacity, *s.shape), dtype=s.dtype, device=device),
    spec,
  )


def batch_dim(item):
  """Leading (batch) length of a chunk -- the leading axis of any one of its leaves."""
  n, = set(s.shape[0] for s in jax.tree.leaves(item))
  return n

class Buffered:
  def __init__(self, capacity, specs, device=None):
    self.capacity = capacity
    self.device = device
    self.specs = tuple(specs)

    self._buffers = jax.tree.map(
      lambda s: jnp.zeros(shape=(capacity, *s.shape), dtype=s.dtype, device=device),
      specs
    )

    def assign(buffers, index, values):
      return jax.tree.map(
        lambda b, v: b.at[index].set(v),
        buffers, values
      )

    self.assign = jax.jit(assign, donate_argnums=(0,))

  def buffers(self):
    """The stored slots as a positional tuple of pytrees (full capacity)."""
    return self._buffers

class Pool(Buffered):
  """Append-only per-event buffers (a positional tuple of pytree slots), sized from specs.

  ``n_current`` is a runtime cursor; events are written contiguously and training kernels
  read the fixed-shape buffers gated on ``n_current``, so one compiled kernel handles a
  growing pool. The pool never calls the detector -- scripts sample events explicitly (so the
  detector calls stay visible) and append them here. Storing the raw physical design per event
  lets one pool span many designs (it accumulates across BO iterations) and lets the network's
  ``combine`` be design-conditioned -- each event is combined with its own design.
  """

  def __init__(self, capacity, specs, device=None):
    super().__init__(capacity, specs, device=device)
    self.current = 0

  def append(self, *chunk):
    """Append a chunk of ``n`` examples (one positional item per slot, matching the specs)."""
    n = batch_dim(chunk[0])
    if self.current + n > self.capacity:
      raise RuntimeError(f"Pool overflow: {self.current} + {n} > {self.capacity}")

    index = jnp.arange(self.current, self.current + n, dtype=jnp.int32)
    self._buffers = self.assign(self._buffers, index, chunk)
    self.current += n

  def __len__(self):
    return self.current

class RingBuffer(Buffered):
  """Fixed-capacity FIFO ring of recent examples (a positional tuple of pytree slots).

  Unlike :class:`Pool` (append-only, sized to the whole budget), the ring keeps only the most
  recent ``capacity`` examples: ``push`` overwrites the oldest once full via a modular cursor.
  ``state``/``load_state`` round-trip the slots + cursor so the ring survives a checkpoint.
  """

  def __init__(self, capacity, specs, device=None):
    super().__init__(capacity, specs, device=device)
    self.current = 0
    self.cursor = 0  # next write position (mod capacity)
    self.filled = 0  # number of valid rows so far (<= capacity)

  def push(self, *chunk):
    """Write a chunk of ``n`` examples (one positional item per slot), overwriting the oldest
    once the ring is full."""
    n = batch_dim(chunk[0])
    if n > self.capacity:  # an oversized chunk: keep only its last `capacity` rows
      import warnings
      warnings.warn('The chuck is larger than the ring buffer. Undefined behaviour might occur.')

    index = (self.cursor + jnp.arange(n)) % self.capacity
    self._buffers = self.assign(self._buffers, index, chunk)
    self.cursor = (self.cursor + n) % self.capacity
    self.filled = min(self.filled + n, self.capacity)

  def __len__(self):
    return self.filled

  def state(self):
    """Serialisable snapshot (slots + cursor) for checkpointing."""
    return {"slots": tuple(self.slots), "cursor": np.int32(self.cursor), "n_filled": np.int32(self.filled)}

  def load_state(self, state):
    """Restore from a :meth:`state` snapshot."""
    self.slots = [jax.tree.map(lambda a: jax.device_put(jnp.asarray(a), self.device), slot) for slot in state["slots"]]
    self.cursor = int(state["cursor"])
    self.filled = int(state["n_filled"])
