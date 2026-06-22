"""Detector base class.

Contract (see ``detector-spec.md``). The base declares the interface only;
every method that encodes an implementation choice is abstract, so concrete
detectors decide *how*.

  * The detector owns all static configuration -- geometry constants, design
    constraints (bounds), target normalisation statistics, and its **event
    source**. It MUST NOT carry a mutable "current design"; design parameters
    are always passed in as function arguments.

  * Raw events, targets, ground truth and the physical design are typed
    **namedtuple records** (``Event``, ``Target``, ``GroundTruth``, ``Design``),
    each a pytree of possibly mixed-dtype arrays (e.g. int32 hit indices beside a
    float32 TDC -- no more integers smuggled through floats). The ``*_spec``
    methods report a record's structure as the SAME namedtuple filled with
    ``jax.ShapeDtypeStruct`` (per-event, no batch axis), so pools allocate/index
    them generically with ``jax.tree``.

  * ``__call__(seed, design) -> (ground_truth, event, mask, target)`` generates
    events. ``design`` is **un-encoded** (physical ``Design`` / config Mapping /
    flat array), leading axis ``B``::

        ground_truth : GroundTruth -- generator truth (== conditioning)
        event        : Event       -- per-hit raw features (pytree, leaves (B, M, ...))
        mask         : (B, M) int32 -- 1 for real hits, 0 padding
        target       : Target      -- regression target

    Event generation is host-side (numpy) and non-differentiable.

  * ``encode_design(d) -> d_enc`` is a bijection from the interior of the
    constrained space onto unconstrained R^n; it accepts a ``Design`` namedtuple,
    a config ``Mapping``, or an already-flat physical array. ``decode_design`` is
    its inverse and returns a ``Design``. Both are differentiable / jittable.

  * ``combine_encoded(event, d_enc) -> features`` merges a *raw* event and an
    *encoded* design into a single design-informed event -- it normalises/packs
    the event itself (there is no separate ``normalize``), differentiable w.r.t.
    the encoded design; the hit ``mask`` is threaded separately by the caller.
    ``combine(event, design)`` is the convenience wrapper
    ``combine_encoded(event, encode_design(design))`` used when training from
    buffers that store the raw physical design.

  * ``normalize_target(Target) -> Array`` maps targets into the network's flat
    prediction space; ``denormalize_predictions(Array) -> Target`` is its inverse.
    ``normalize_ground_truth(GroundTruth) -> Array`` standardises the ground truth
    for the discriminator (no inverse -- ground truth is never predicted).

  * ``loss(pred, target)`` / ``metric(pred, target)`` compare a flat prediction
    array against the flat normalised label and return per-sample ``(B,)`` arrays.

  * The ``*_spec`` records are the source of truth; the ``*_dim`` accessors
    (``design_dim``, ``encoded_design_dim``, ``target_dim``, ``ground_truth_dim``,
    ``combined_feature_dim``) are derived for the convenience of the networks.
    ``encoded_design_shape`` and ``combined_event_shape`` stay flat-array shapes.
"""

import math
from typing import Any, Sequence, Mapping

import jax

__all__ = ["Detector", "Shape"]

# A per-field design shape, e.g. ``(n_stations,)`` for a vector field or ``(1,)`` for a scalar one.
Shape = tuple[int, ...]


def _prod(shape):
    return int(math.prod(shape))


def _spec_dim(spec):
    """Total flattened width of a record spec (a namedtuple / pytree of ShapeDtypeStruct)."""
    return int(sum(_prod(leaf.shape) for leaf in jax.tree.leaves(spec)))


class Detector(object):
    """
    Abstract detector interface.
    """

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Detector":
        return cls(**config)

    # ------------------------------------------------------------------ #
    # Record specs: the SAME namedtuple as the record, filled with
    # jax.ShapeDtypeStruct (per-event, no batch axis). Source of truth.
    # ------------------------------------------------------------------ #
    def event_spec(self):
        """Raw per-hit event record structure (defined per detector)."""
        raise NotImplementedError()

    def target_spec(self):
        """Regression target record structure (defined per detector)."""
        raise NotImplementedError()

    def ground_truth_spec(self):
        """Ground-truth (== conditioning) record structure (defined per detector)."""
        raise NotImplementedError()

    def design_shape(self) -> Shape:
        """Flat physical design shape (defined per detector)."""
        raise NotImplementedError()

    def encoded_design_shape(self) -> Shape:
        """Encoded (unconstrained) design shape; may differ from ``design_shape``."""
        return self.design_shape()

    def combined_event_shape(self):
        """Per-hit feature shape ``(M, F)`` produced by :meth:`combine` (a flat float array)."""
        raise NotImplementedError()

    # ------------------------------------------------------------------ #
    # Dimensions derived from the specs / shapes (no batch axis).
    # ------------------------------------------------------------------ #
    def design_dim(self):
        return _prod(self.design_shape())

    def encoded_design_dim(self):
        return _prod(self.encoded_design_shape())

    def target_dim(self):
        return _spec_dim(self.target_spec())

    def ground_truth_dim(self):
        return _spec_dim(self.ground_truth_spec())

    @property
    def combined_feature_dim(self):
        """Per-hit feature count produced by :meth:`combine`."""
        return int(self.combined_event_shape()[-1])

    def __call__(self, seed, design, pool=None):
        """
        Generates events for an un-encoded design. Returns (ground_truth, event, mask, target),
        where event/target/ground_truth are namedtuple records (leaves carry a leading batch axis).

        ``pool`` (int|str, default = first pool key) selects which disjoint event pool to
        sample from -- see ``pool_split`` on the concrete detector.
        """
        raise NotImplementedError()

    ### TODO: this function belongs to utils
    @staticmethod
    def resolve_pool_split(pool_split):
        """Normalise a ``pool_split`` (Sequence | Mapping | None) into an ordered dict of
        ``{key: fraction}`` with fractions summing to 1. A ``Sequence`` keys by position
        (``0, 1, ...``); ``None`` is a single pool ``{0: 1.0}`` over all events. Pool keys
        are int or str; the first key is the default pool."""
        from collections.abc import Mapping, Sequence

        if pool_split is None:
            return {0: 1.0}
        if isinstance(pool_split, Mapping):
            items = {k: float(v) for k, v in pool_split.items()}
        elif isinstance(pool_split, Sequence) and not isinstance(pool_split, (str, bytes)):
            items = {i: float(v) for i, v in enumerate(pool_split)}
        else:
            raise TypeError(f"pool_split must be a Sequence, Mapping, or None; got {type(pool_split)}")
        total = sum(items.values())
        if total <= 0:
            raise ValueError(f"pool_split fractions must sum to > 0; got {items}")
        return {k: v / total for k, v in items.items()}

    # ------------------------------------------------------------------ #
    # Physical design as a typed ``Design`` namedtuple (e.g. stereo: ``(stations, angle)``); the
    # ENCODED design stays a single flat vector. A detector defines ``design_spec`` (the ``Design``
    # namedtuple filled with ``jax.ShapeDtypeStruct`` -- same shape as ``event_spec``/``target_spec``),
    # ``design_bounds``, and the flat physical<->encoded bijection (``_encode_flat`` / ``_decode_flat``).
    # The record<->flat conversion (``flatten_design`` / ``unflatten_design``) is GENERIC here, via the
    # ``tensor`` codec over the design pytree -- exactly how Event/Target records are packed.
    # ------------------------------------------------------------------ #
    def design_spec(self):
        """The ``Design`` namedtuple filled with ``jax.ShapeDtypeStruct`` (per-field shape + dtype),
        in field order. Defined per detector."""
        raise NotImplementedError()

    def design_bounds(self):
        """``{name: (lo, hi)}`` physical per-field bounds, keyed by the ``Design`` field names
        (defined per detector)."""
        raise NotImplementedError()

    def flatten_design(self, design):
        """``Design`` namedtuple / config ``Mapping`` / already-flat array -> flat physical
        ``(..., design_dim)`` (field order). Generic: packs the design pytree with
        :func:`tensor.flatten` -- the last axis of each field is its feature, anything before is batch."""
        import jax.numpy as jnp
        from collections.abc import Mapping
        from ..utils import tensor

        spec = self.design_spec()
        design_type = type(spec)
        if isinstance(design, Mapping):  # config dict (a single design): one value per field -> its spec shape
            design = design_type(*(jnp.asarray(design[name], jnp.float32).reshape(leaf.shape) for name, leaf in zip(spec._fields, spec)))
        elif not isinstance(design, design_type):
            return jnp.asarray(design, jnp.float32)  # already a flat physical vector
        ndim = jax.tree.leaves(design)[0].ndim  # fields share leading (batch) axes; the last axis is the feature
        return tensor.flatten(design, batch_dimensions=tuple(range(ndim - 1)))[0]

    def unflatten_design(self, flat):
        """Flat physical ``(..., design_dim)`` -> ``Design`` namedtuple (inverse of
        :meth:`flatten_design`), via :func:`tensor.unflatten` over ``design_spec``."""
        import jax.numpy as jnp
        from ..utils import tensor

        return tensor.unflatten(tensor.structure(self.design_spec()), jnp.asarray(flat, jnp.float32))

    def encode_design(self, design):
        """Physical design (``Design`` namedtuple, config Mapping, or flat array) -> encoded vector."""
        return self._encode_flat(self.flatten_design(design))

    def decode_design(self, encoded_design):
        """Encoded flat vector -> physical ``Design`` namedtuple."""
        return self.unflatten_design(self._decode_flat(encoded_design))

    def _encode_flat(self, design):
        """Flat physical design array -> encoded vector (the bijection; defined per detector)."""
        raise NotImplementedError()

    def _decode_flat(self, encoded_design):
        """Encoded vector -> flat physical design array (inverse of :meth:`_encode_flat`)."""
        raise NotImplementedError()

    # ------------------------------------------------------------------ #
    # Combine: raw event (+ design) -> flat per-hit network features.
    # ------------------------------------------------------------------ #
    def combine_encoded(self, event, encoded_design):
        """Merge a raw ``Event`` and an ENCODED design into ``features (..., M, F)`` (defined per
        detector). Normalises/packs the event internally; differentiable w.r.t. the encoded design."""
        raise NotImplementedError()

    def combine(self, event, design):
        """Merge a raw ``Event`` and a PHYSICAL design. Default: encode the design, then
        :meth:`combine_encoded`. Detectors may override how they combine."""
        return self.combine_encoded(event, self.encode_design(design))

    # ------------------------------------------------------------------ #
    # Target / ground-truth normalisation.
    # ------------------------------------------------------------------ #
    def normalize_target(self, target):
        """Physical ``Target`` -> standardised flat array (~[-1, 1]); defined per detector."""
        raise NotImplementedError()

    def denormalize_predictions(self, normalised):
        """Inverse of :meth:`normalize_target`: flat normalised array -> physical ``Target``."""
        raise NotImplementedError()

    def normalize_ground_truth(self, ground_truth):
        """Physical ``GroundTruth`` -> standardised flat array (no inverse -- never predicted)."""
        raise NotImplementedError()

    def loss(self, predicted: jax.Array, target: jax.Array) -> jax.Array:
        raise NotImplementedError()

    def metric(self, predicted: jax.Array, target: jax.Array) -> Mapping[str, jax.Array]:
        raise NotImplementedError()

    def loss_label(self) -> str:
        raise NotImplementedError()

    def metric_labels(self) -> Sequence[str]:
        raise NotImplementedError()
