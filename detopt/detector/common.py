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
from typing import Any

import jax

__all__ = ["Detector"]


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

    def __call__(self, seed, design):
        """
        Generates events for an un-encoded design. Returns (ground_truth, measurements, mask, target).
        """
        raise NotImplementedError()

    def encode_design(self, design):
        raise NotImplementedError()

    def decode_design(self, encoded_design):
        raise NotImplementedError()

    def normalize(self, X):
        raise NotImplementedError()

    def denormalize(self, X_norm):
        raise NotImplementedError()

    def combine(self, X_norm, encoded_design):
        """Merge a normalised event and an encoded design into ``features``."""
        raise NotImplementedError()

    def normalize_target(self, target):
        """Map physical targets into the network's prediction space (~[-1, 1]).

        Default: per-component standardisation by ``target_mean`` / ``target_std``
        (both required on the concrete detector).
        """
        import jax.numpy as jnp

        target = jnp.asarray(target, dtype=jnp.float32)
        return (target - jnp.asarray(self.target_mean)) / jnp.asarray(self.target_std)

    def denormalize_predictions(self, normalised):
        """Inverse of :meth:`normalize_target`: back to physical units."""
        import jax.numpy as jnp

        normalised = jnp.asarray(normalised, dtype=jnp.float32)
        return normalised * jnp.asarray(self.target_std) + jnp.asarray(self.target_mean)

    def loss(self, predicted: jax.Array, target: jax.Array) -> jax.Array:
        """Per-sample ``(B,)`` MSE between predictions and the normalised target.

        Predictions are expected in the network's (normalised) output space, so the
        target is brought into the same space via :meth:`normalize_target`.
        """
        import jax.numpy as jnp

        predicted = jnp.asarray(predicted, dtype=jnp.float32)
        target_norm = self.normalize_target(target)
        return jnp.mean(jnp.square(predicted - target_norm), axis=-1)

    def metric(self, predicted: jax.Array, target: jax.Array) -> jax.Array:
        return self.loss(predicted, target)
