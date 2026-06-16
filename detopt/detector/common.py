"""Detector base class.

Contract (see ``detector-spec.md``). The base declares the interface only;
every method that encodes an implementation choice is abstract, so concrete
detectors decide *how*.

  * The detector owns all static configuration -- geometry constants
    (``n_layers``, ``max_particles``, ...), design constraints (bounds), target
    normalisation statistics, and its **event source**. It MUST NOT carry a
    mutable "current design"; design parameters are always passed in as
    function arguments.

  * ``__call__(seed, design) -> (ground_truth, measurements, mask, target)``
    generates events. ``design`` is **un-encoded** (physical/constrained),
    shape ``(B, design_dim)``; ``B`` is its leading dimension. For hit-based
    detectors::

        ground_truth : (B, ground_truth_dim) float32 -- generator-truth info
        measurements : (B, M, raw_feature_dim) float32 -- per-hit raw features
        mask         : (B, M)                int32   -- 1 for real hits, 0 padding
        target       : (B, target_dim)       float32 -- regression target

    Event generation is host-side (numpy) and non-differentiable.

  * Shape methods (``event_shape``, ``design_shape``, ``combined_event_shape``,
    ``target_shape``, ``ground_truth_shape``) report per-event shapes, no batch
    axis. ``normalized_event_shape`` / ``encoded_design_shape`` default to
    ``event_shape`` / ``design_shape``.

  * ``encode_design(d) -> d_enc`` is a bijection from the interior of the
    constrained space onto unconstrained R^n; ``decode_design`` is its inverse.
    Both are differentiable / jittable so they can run inside the network loss.

  * ``normalize(X) -> X_norm`` brings raw event features into ~[-1, 1]
    (invertible via ``denormalize``). ``combine(X_norm, d_enc) -> features``
    merges a *normalised* event and an *encoded* design into a single
    design-informed event, differentiable w.r.t. both; the hit ``mask`` is
    threaded separately by the caller.

  * ``normalize_target(target)`` maps targets into the network's prediction
    space; ``denormalize_predictions(pred)`` is its inverse, back to physical
    units.

  * ``loss(pred, target)`` / ``metric(pred, target)`` compare a prediction
    against the target and return plain per-sample ``(B,)`` arrays. The default
    is the mean-squared error against the *normalised* target; ``metric``
    defaults to ``loss``.

  * The ``*_shape`` methods are the source of truth; the ``*_dim`` accessors
    (``design_dim``, ``target_dim``, ``raw_feature_dim``, ``combined_feature_dim``)
    are derived from them for the convenience of the networks.
"""

import math
from typing import Any, Sequence, Mapping

import jax

__all__ = ["Detector", "Shape"]

# A per-field design shape, e.g. ``(n_stations,)`` for a vector field or ``(1,)`` for a scalar one.
Shape = tuple[int, ...]


def _prod(shape):
    return int(math.prod(shape))


class Detector(object):
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Detector":
        return cls(**config)

    def event_shape(self):
        raise NotImplementedError()

    def normalized_event_shape(self):
        return self.event_shape()

    def design_shape(self):
        raise NotImplementedError()

    def encoded_design_shape(self):
        return self.design_shape()

    def combined_event_shape(self):
        raise NotImplementedError()

    def target_shape(self):
        raise NotImplementedError()

    def ground_truth_shape(self):
        raise NotImplementedError()

    # ------------------------------------------------------------------ #
    # Dimensions derived from the shapes (no batch axis).
    # ------------------------------------------------------------------ #
    def design_dim(self):
        return _prod(self.design_shape())

    def encoded_design_dim(self):
        return _prod(self.encoded_design_shape())

    def target_dim(self):
        return _prod(self.target_shape())

    @property
    def raw_feature_dim(self):
        """Per-hit raw feature count (last axis of ``event_shape``)."""
        return int(self.event_shape()[-1])

    @property
    def combined_feature_dim(self):
        """Per-hit feature count produced by :meth:`combine`."""
        return int(self.combined_event_shape()[-1])

    def __call__(self, seed, design, pool=None):
        """
        Generates events for an un-encoded design. Returns (ground_truth, measurements, mask, target).

        ``pool`` (int|str, default = first pool key) selects which disjoint event pool to
        sample from -- see ``pool_split`` on the concrete detector. ``None`` uses the default
        pool (all events when no split was configured).
        """
        raise NotImplementedError()

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
    # Physical design as a named dict (e.g. stereo: {stations, angle}); the ENCODED design
    # stays a single flat vector. ``design_spec`` (``Mapping[str, Shape]``) names the dict's
    # fields and their per-field shapes; detectors implement the dict<->flat conversion
    # (``flatten_design`` / ``unflatten_design``), the flat physical<->encoded bijection
    # (``_encode_flat`` / ``_decode_flat``) and ``design_bounds``.
    # ------------------------------------------------------------------ #
    def design_spec(self) -> Mapping[str, Shape]:
        """Ordered ``{name: shape}`` of the physical design dict's fields (defined per detector)."""
        raise NotImplementedError()

    def design_bounds(self):
        """``{name: (lo, hi)}`` physical per-field bounds (defined per detector)."""
        raise NotImplementedError()

    def flatten_design(self, design):
        """Design dict -> flat physical array ``(..., design_dim)`` (spec order; defined per
        detector). A non-dict is returned as-is (already flat)."""
        raise NotImplementedError()

    def unflatten_design(self, flat):
        """Flat physical array ``(..., design_dim)`` -> design dict (inverse of
        :meth:`flatten_design`; defined per detector)."""
        raise NotImplementedError()

    def encode_design(self, design):
        """Physical design (dict or flat array) -> encoded flat vector."""
        return self._encode_flat(self.flatten_design(design))

    def decode_design(self, encoded_design):
        """Encoded flat vector -> physical design DICT."""
        return self.unflatten_design(self._decode_flat(encoded_design))

    def _encode_flat(self, design):
        """Flat physical design array -> encoded vector (the bijection; defined per detector)."""
        raise NotImplementedError()

    def _decode_flat(self, encoded_design):
        """Encoded vector -> flat physical design array (inverse of :meth:`_encode_flat`)."""
        raise NotImplementedError()

    def normalize(self, X):
        raise NotImplementedError()

    def denormalize(self, X_norm):
        raise NotImplementedError()

    def combine(self, X_norm, encoded_design):
        """Merge a normalised event and an encoded design into ``features``."""
        raise NotImplementedError()

    def normalize_target(self, target):
        """Map physical targets into the network's prediction space (~[-1, 1]); defined per detector."""
        raise NotImplementedError()

    def denormalize_predictions(self, normalised):
        """Inverse of :meth:`normalize_target`: back to physical units; defined per detector."""
        raise NotImplementedError()

    def loss(self, predicted: jax.Array, target: jax.Array) -> jax.Array:
        raise NotImplementedError()

    def metric(self, predicted: jax.Array, target: jax.Array) -> Mapping[str, jax.Array]:
        raise NotImplementedError()

    def loss_label(self) -> str:
        raise NotImplementedError()

    def metric_labels(self) -> Sequence[str]:
        raise NotImplementedError()
