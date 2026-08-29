"""Detector-agnostic event buffers for training scripts.

Both buffers hold a fixed, positional tuple of **pytree slots** on a JAX device, each
sized from a per-event *spec* -- either a record (a namedtuple of ``jax.ShapeDtypeStruct``,
e.g. ``event_spec()`` / ``target_spec()``) or a plain ``jax.ShapeDtypeStruct`` (e.g. the mask
or the flat physical design). Allocation/append/index all go through ``jax.tree``, so a slot
may carry mixed dtypes (int32 hit indices beside a float32 TDC) without special-casing.

**They store RAW records** -- raw events, raw targets, the raw physical design, raw ground
truth -- NOT design-``combine``d features. ``combine`` + ``normalize_target`` run per training
batch right before the network (scaled/normalised/combined forms are larger; re-deriving them
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
  def __init__(self, capacity, specs, device=None, buffers=None):
    """``buffers`` supplies the slots instead of allocating them, checked against ``specs``.

        Allocating zeros and then replacing them holds two pools at once, and XLA keeps that
        high-water mark for the process's life."""
    self.capacity = capacity
    self.device = device
    self.specs = tuple(specs)

    if buffers is None:
      self._buffers = jax.tree.map(
        lambda s: jnp.zeros(shape=(capacity, *s.shape), dtype=s.dtype, device=device),
        specs
      )
    else:
      given, expected = jax.tree.structure(tuple(buffers)), jax.tree.structure(self.specs)
      if given != expected:
        raise ValueError(f"buffers do not match specs: {given} vs {expected}")
      for array, spec in zip(jax.tree.leaves(tuple(buffers)), jax.tree.leaves(self.specs)):
        if array.shape != (capacity, *spec.shape):
          raise ValueError(f"buffer of shape {array.shape} is not a slot of {(capacity, *spec.shape)}")
      self._buffers = jax.tree.map(lambda a: jax.device_put(jnp.asarray(a), device), tuple(buffers))

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
  lets one pool span many designs (it accumulates across BO iterations) and lets ``combine`` be
  given each event's OWN design -- whether that design then reaches the network is the trainer's
  call (``Trainer.reveals_design``), and either way the pool holds the same raw records.
  """

  def __init__(self, capacity, specs, device=None, buffers=None):
    super().__init__(capacity, specs, device=device, buffers=buffers)
    self.current = 0

  def append(self, *chunk):
    """Append a chunk of ``n`` examples (one positional item per slot, matching the specs)."""
    n = batch_dim(chunk[0])
    if self.current + n > self.capacity:
      raise RuntimeError(f"Pool overflow: {self.current} + {n} > {self.capacity}")

    index = jnp.arange(self.current, self.current + n, dtype=jnp.int32)
    self._buffers = self.assign(self._buffers, index, chunk)
    self.current += n

  def state(self):
    """Serialisable snapshot (slots + fill cursor) for checkpointing.

    The CURSOR is as important as the contents: it decides where the next design's window opens and
    therefore which slice of the run's event index that design consumes. Restoring rows without the
    cursor would silently re-use events.
    """
    return {"slots": tuple(self.buffers()), "current": np.int32(self.current)}

  @classmethod
  def load(cls, state, specs, capacity=None, device=None):
    """A pool of ``capacity`` slots holding a :meth:`state` snapshot. Release what you are replacing
        BEFORE calling it.

        CAPACITY IS THE POOL'S, NOT THE SNAPSHOT'S. It defaults to the snapshot's leading dimension --
        the same pool that was saved -- but a caller that has been configured for a LARGER budget
        passes that budget, and the restored rows are placed in the first ``current`` slots of a pool
        sized for it. Taking capacity from the data instead, as this once did, silently discarded a
        raised budget: the pool came back full at the old size and the run ended on its first round
        with "pool exhausted", having added nothing.

        A capacity BELOW the stored fill is refused rather than truncated: those events were paid for
        with detector calls and the fill cursor decides which slice of the event index every later
        design consumes, so dropping rows would corrupt the run rather than shrink it.
        """
    slots = tuple(state["slots"])
    stored = int(jax.tree.leaves(slots)[0].shape[0])
    current = int(state["current"])
    if capacity is None:
      capacity = stored
    capacity = int(capacity)
    if capacity < current:
      raise ValueError(f"capacity {capacity} is below the snapshot's fill {current}: restoring would "
                       f"drop events that have already been paid for")
    if capacity == stored:
      # The saved arrays ARE the buffers: no allocation, no copy, no doubled high-water mark.
      pool = cls(capacity, specs, device=device, buffers=slots)
      pool.current = current
      return pool
    # Growing: allocate at the NEW capacity and place the stored rows into the first `current` slots.
    # The snapshot's leaves are on the host, so only one pool is ever resident on the device.
    pool = cls(capacity, specs, device=device)
    if current > 0:
      index = jnp.arange(current, dtype=jnp.int32)
      head = jax.tree.map(lambda a: jnp.asarray(a)[:current], slots)
      pool._buffers = pool.assign(pool._buffers, index, head)
    pool.current = current
    return pool

  def __len__(self):
    return self.current

class RingBuffer(Buffered):
  """Fixed-capacity FIFO ring of recent examples (a positional tuple of pytree slots).

  Unlike :class:`Pool` (append-only, sized to the whole budget), the ring keeps only the most
  recent ``capacity`` examples: ``push`` overwrites the oldest once full via a modular cursor.
  ``state``/``load_state`` round-trip the slots + cursor so the ring survives a checkpoint.
  """

  def __init__(self, capacity, specs, device=None, buffers=None):
    super().__init__(capacity, specs, device=device, buffers=buffers)
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
