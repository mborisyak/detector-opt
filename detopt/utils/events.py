"""Shared event-index helpers for the deterministic detector API.

The detector is a pure function ``detector(design, event_index)`` with ``size() -> int | None``; it does NOT
sample or own train/val pools. The SCRIPTS (and the trainers, which are script-side infrastructure) own the
randomness and the split: they read ``detector.size()``, build a shuffled ``event_index`` of the requested
length (the first ``n`` of a permutation; warn + oversample by wrapping when ``n > size``), and slice it into
disjoint train / val / design index sets. These helpers are that one shared implementation.
"""

import warnings

import numpy as np

__all__ = ["shuffled_event_index", "split_disjoint", "EventIndexStream", "disjoint_index_streams"]


def shuffled_event_index(size, n, seed, *, name="events"):
    """The first ``n`` events of a shuffled ``[0, size)`` permutation, as an ``(n,) int64`` index array.

    ``size`` is ``detector.size()``: a finite int (data-backed) or ``None`` (infinite -- the analytic
    source, where any index is a fresh deterministic event). When ``n > size`` (finite), warn and
    OVERSAMPLE by wrapping the permutation (``perm[arange(n) % size]``) -- repeated indices reproduce the
    same event by the detector's per-index determinism.
    """
    rng = np.random.default_rng(seed)
    n = int(n)
    if size is None:
        return rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)  # infinite source: any indices
    size = int(size)
    perm = rng.permutation(size)
    if n > size:
        warnings.warn(f"requested {n} {name} > {size} available; oversampling (wrapping the shuffle).",
                      stacklevel=2)
    return perm[np.arange(n) % size].astype(np.int64)


class EventIndexStream:
    """An endless stream of event indices over a fixed ``universe`` (a subset of ``[0, size())``),
    re-shuffled each pass -- for the ring-buffer scripts (subgradient/lfi) that keep requesting fresh
    events across many design steps. ``next_block(k)`` returns the next ``(k,) int64`` indices. For the
    infinite analytic source (``universe=None``) every block is freshly drawn (each index is a new
    deterministic event)."""

    def __init__(self, universe, seed):
        self._universe = None if universe is None else np.asarray(universe, np.int64)
        self._rng = np.random.default_rng(seed)
        self._perm = None if self._universe is None else self._rng.permutation(self._universe)
        self._cursor = 0

    def next_block(self, k):
        k = int(k)
        if self._universe is None:
            return self._rng.integers(0, 2**31 - 1, size=k, dtype=np.int64)
        out, need = [], k
        while need > 0:
            if self._cursor >= self._perm.shape[0]:  # exhausted -> reshuffle the universe
                self._perm = self._rng.permutation(self._universe)
                self._cursor = 0
            take = min(need, self._perm.shape[0] - self._cursor)
            out.append(self._perm[self._cursor:self._cursor + take])
            self._cursor += take
            need -= take
        return np.concatenate(out).astype(np.int64)


def disjoint_index_streams(size, cut_fractions, seed):
    """``len(cut_fractions) + 1`` endless :class:`EventIndexStream`s over DISJOINT universes -- a shuffled
    split of ``[0, size)`` by ``cut_fractions`` (the last stream takes the remainder) -- for the
    ring-buffer scripts' train / design / val event sets. For the infinite analytic source (``size`` is
    ``None``) the streams are infinite and disjoint by seed. E.g. ``disjoint_index_streams(size, [0.45,
    0.45], seed)`` -> ``(train, design, val)`` streams (0.45 / 0.45 / 0.10)."""
    rng = np.random.default_rng(seed)
    n = len(cut_fractions) + 1
    if size is None:
        return [EventIndexStream(None, int(rng.integers(0, 2**31 - 1))) for _ in range(n)]
    universes = split_disjoint(rng.permutation(int(size)), *cut_fractions)
    return [EventIndexStream(u, int(rng.integers(0, 2**31 - 1))) for u in universes]


def split_disjoint(event_index, *fractions):
    """Slice ``event_index`` into disjoint contiguous chunks by normalized ``fractions`` (the last chunk
    takes the remainder). E.g. ``split_disjoint(idx, 0.45, 0.45)`` -> ``(train, design, val)`` index arrays.
    The chunks are disjoint slices of the already-shuffled array, so they share no event (modulo any
    oversample wrapping in ``event_index`` itself)."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    cuts, acc = [], 0.0
    for f in fractions:
        acc += float(f)
        cuts.append(min(n, int(round(acc * n))))
    out, start = [], 0
    for c in (*cuts, n):
        out.append(event_index[start:c])
        start = c
    return tuple(out)
