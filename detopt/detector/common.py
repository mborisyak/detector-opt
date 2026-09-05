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
    events. ``design`` is **NOMINAL** (physical ``Design`` / config Mapping /
    flat array), leading axis ``B``::

        ground_truth : GroundTruth -- generator truth (== conditioning)
        event        : Event       -- per-hit raw features (pytree, leaves (B, M, ...))
        mask         : (B, M) int32 -- 1 for real hits, 0 padding
        target       : Target      -- regression target

    Event generation is host-side (numpy) and non-differentiable.

  * A design lives in exactly TWO spaces. ``to_scaled(d) -> d_scaled`` is a
    bijection from the constrained NOMINAL (physical) space onto the SCALED cube
    ``[0, 1]^n``, each coordinate affinely on its own design range; it accepts a
    ``Design`` namedtuple, a config ``Mapping``, or an already-flat nominal array.
    ``to_nominal`` is its inverse and returns a ``Design``. Both are
    differentiable / jittable. There is no third (unconstrained) space: optimisers
    search the scaled cube directly, so a uniform draw there IS a uniform design.

  * ``combine_scaled(event, d_scaled) -> features`` merges a *raw* event and a
    *scaled* design into the network's input -- it normalises/packs the event
    itself (there is no separate ``normalize``), differentiable w.r.t. the scaled
    design; the hit ``mask`` is threaded separately by the caller.
    ``combine(event, design)`` is the FINAL wrapper
    ``combine_scaled(event, to_scaled(design))`` used when training from
    buffers that store the raw physical design.

    THE DESIGN IS OPTIONAL, in two distinct ways. ``reveal_design=False`` keeps
    the design in the MEASUREMENT and withholds it from the NETWORK; ``design=None``
    says there is no design at all, which a detector whose measurement depends on
    one refuses. Either way the features get NARROWER, never corrupted, and
    ``combined_event_shape(design=False)`` reports that narrower shape. Which of
    the two layouts a run uses is the TRAINER's call (``Trainer.reveals_design``),
    not the detector's, so a detector must implement both.

  * ``normalize_target(Target) -> Array`` maps targets into the network's flat
    prediction space; ``denormalize_predictions(Array) -> Target`` is its inverse.
    ``normalize_ground_truth(GroundTruth) -> Array`` standardises the ground truth
    for the discriminator (no inverse -- ground truth is never predicted).

  * ``loss(pred, target)`` / ``metric(pred, target)`` compare a flat prediction
    array against the flat normalised label and return per-sample ``(B,)`` arrays.

  * The ``*_spec`` records are the source of truth; the ``*_dim`` accessors
    (``design_dim``, ``target_dim``, ``ground_truth_dim``, ``combined_feature_dim(design)``)
    are derived for the convenience of the networks. ``design_shape`` and
    ``combined_event_shape(design)`` stay flat-array shapes; ``design_shape``
    covers both design spaces, which share a width. The two combined accessors
    take the ``design`` flag because their answer depends on it.
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
        """Build the detector from a (yaml-parsed) ``config`` dict, passing its entries straight into
        ``__init__``. Validates the keys against the constructor signature(s) across the MRO, so an
        unknown or mistyped key raises instead of being silently splatted. The single concrete factory
        for every detector (no subclass override)."""
        import inspect

        allowed = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, p in inspect.signature(init).parameters.items():
                if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                allowed.add(name)
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown {cls.__name__} config key(s): {sorted(unknown)}")
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
        """Flat design shape, shared by BOTH spaces: the scaled vector is the nominal design one
        coordinate at a time on ``[0, 1]``, so it has the same width (defined per detector)."""
        raise NotImplementedError()

    def combined_event_shape(self, design: bool = True):
        """The shape of what :meth:`combine` produces, without the batch axis, for the case where the
        design IS (``design=True``) or IS NOT (``design=False``) supplied. Usually ``(M, F)``, and the
        two cases usually differ only in ``F``. ``Model.from_config`` passes this straight through as
        ``input_shape``, so a model must be built for the same case the trainer will feed it."""
        raise NotImplementedError()

    # ------------------------------------------------------------------ #
    # Dimensions derived from the specs / shapes (no batch axis).
    # ------------------------------------------------------------------ #
    def design_dim(self):
        return _prod(self.design_shape())

    def combined_event_shape_for(self, reveal: str):
        """:meth:`combined_event_shape` under a TRAINER's ``reveal`` setting rather than a bool.

        ``'design'`` and ``'zeros'`` both take the design-revealed layout -- ``'zeros'`` differs in the
        VALUE handed to ``combine``, not the width. ``'none'`` takes the design-free one.

        ``'append'`` is the design-free layout WIDENED by ``design_dim``: the trainer concatenates the
        scaled design onto whatever ``combine`` emits without it. It is defined on ``combine``'s OUTPUT,
        so no detector implements it -- the rule is the same whether that output is an element set
        ``(M, F)`` or an image ``(H, W, C)``, where the design becomes ``design_dim`` constant channels.

        ⚠️ ``'none'`` IS NOT ``'design'`` WITH THE DESIGN COLUMNS DELETED. Several detectors emit a WIDER
        design-free layout, substituting an identity encoding so elements stay distinguishable --
        ``stereo_tracking_layerset`` is ``(n_layers, 2 if design else n_layers)``. So ``'append'`` is
        that substitute encoding PLUS the raw design, not the bare readings plus the design."""
        if reveal == 'append':
            *lead, features = self.combined_event_shape(False)
            return (*lead, features + self.design_dim())
        return self.combined_event_shape(reveal != 'none')

    def target_dim(self):
        return _spec_dim(self.target_spec())

    def ground_truth_dim(self):
        return _spec_dim(self.ground_truth_spec())

    def combined_feature_dim(self, design: bool = True):
        """Per-element feature count produced by :meth:`combine`, with the design revealed or not.

        ⚠️ A METHOD, NOT A PROPERTY: the count depends on which of the two layouts is meant, so it
        cannot be read without saying so."""
        return int(self.combined_event_shape(design)[-1])

    def __call__(self, design, event_index):
        """Simulate the events at the integer ``event_index`` for ``design`` -- a DETERMINISTIC function
        (same ``(design, event_index)`` -> same output). Returns ``(ground_truth, event, mask, target)``,
        namedtuple records (leaves carry a leading batch axis). No internal sampling or pools: the scripts
        own the randomness + the train/val split (see :func:`detopt.utils.events.shuffled_event_index`)."""
        raise NotImplementedError()

    def size(self):
        """Number of available events -- a finite int (data-backed) or ``None`` (infinite, e.g. an
        analytic source). Scripts use it to build a shuffled ``event_index``."""
        raise NotImplementedError()

    # ------------------------------------------------------------------ #
    # A design lives in exactly TWO spaces. NOMINAL is the physical design, a typed ``Design``
    # namedtuple (e.g. stereo: ``(stations, angle)``), used for reading a starting design from the
    # config and for writing results; SCALED is a single flat vector in ``[0, 1]^d``, each coordinate
    # on its own design range, and is what every optimiser searches and every network is conditioned
    # on. A detector defines ``design_spec`` (the ``Design`` namedtuple filled with
    # ``jax.ShapeDtypeStruct`` -- same shape as ``event_spec``/``target_spec``), ``design_bounds``,
    # and the flat nominal<->scaled bijection (``_to_scaled_flat`` / ``_to_nominal_flat``).
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

    def design_penalty(self, design):
        """Additive price of the design ITSELF, in loss units, or ``None`` when the task prices nothing.

        ``design`` is a physical design in any form :meth:`flatten_design` accepts. Returns a scalar to
        be ADDED to the trained loss, or ``None`` -- the default -- meaning this detector has no such
        term. Callers MUST test ``is not None`` and drop the term entirely when it is; a detector with
        no cost must not be made to return ``0.0``, because ``0.0`` is a price that was measured and
        ``None`` is the absence of one, and only the second may be omitted from a report.

        IT IS DETERMINISTIC, SO IT CARRIES NO UNCERTAINTY. The value is a function of the design alone,
        with no sampling in it: a caller adds it to the reported loss and leaves the reported ERROR
        untouched.

        IT MUST NEVER REACH THE TRAINER. ``loss_precision`` and the settled test are statements about
        how well the NETWORK has fit its data; a per-design constant added there would move the
        reported loss without changing anything the criterion is measuring, so the bar would stop
        meaning what it says. The term belongs after convergence, where the design is priced once."""
        return None

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

    def to_scaled(self, design):
        """NOMINAL design (``Design`` namedtuple, config Mapping, or flat array) -> SCALED vector."""
        return self._to_scaled_flat(self.flatten_design(design))

    def to_nominal(self, design_scaled):
        """SCALED flat vector -> NOMINAL ``Design`` namedtuple."""
        return self.unflatten_design(self._to_nominal_flat(design_scaled))

    def _to_scaled_flat(self, design):
        """Flat NOMINAL design -> SCALED ``[0, 1]`` vector (the bijection; defined per detector).

        For most detectors this is the per-coordinate affine map of ``design_bounds``; where a
        coordinate's range depends on another (the stereo stations are ordered, so station k's range
        starts at station k-1), the detector implements that coupling here."""
        raise NotImplementedError()

    def _to_nominal_flat(self, design_scaled):
        """SCALED vector -> flat NOMINAL design (inverse of :meth:`_to_scaled_flat`)."""
        raise NotImplementedError()

    # ------------------------------------------------------------------ #
    # Combine: raw event (+ design) -> the network's input.
    # ------------------------------------------------------------------ #
    def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
        """Merge a raw ``Event`` and a SCALED design into the network's input (defined per detector).
        Normalises/packs the event internally; differentiable w.r.t. the scaled design.

        USUALLY one array ``features (..., M, F)``, with the design already resolved into the
        per-element features. It MAY instead be a PYTREE whose leaves share the leading axes, for a
        detector that hands the design apart from the measurement. Whatever this returns is what
        reaches the regressor: nothing between here and ``Model.__call__`` inspects it, EXCEPT the
        trainer's ensemble path, which reshapes it as an array -- so a non-array combine requires
        ``n_models: null``. ``combined_event_shape`` mirrors the structure.

        EVERY IMPLEMENTATION HANDLES THE DESIGN-FREE CASE. ``reveal_design=False`` must still use
        ``design_scaled`` wherever the MEASUREMENT depends on it and drop only what would announce
        WHICH design was used; ``design_scaled=None`` must be honoured by a detector whose measurement
        does not depend on the design, and REFUSED with a ``ValueError`` by one whose does. The result
        is the narrower layout ``combined_event_shape(design=False)`` reports -- never a corrupted or
        zero-filled version of the wide one. See :meth:`combine`.

        ``mask`` (the per-hit validity mask) is OPTIONAL: hit-wise combines ignore it (masked hits
        carry index 0 and are zeroed downstream by the regressor mask). Combines whose element axis
        is NOT the hit axis (e.g. layer-wise, which scatters hits into a per-layer grid) REQUIRE it
        to tell a real hit from a masked slot, and raise if it is ``None``."""
        raise NotImplementedError()

    def combine(self, event, design=None, mask=None, reveal_design: bool = True):
        """Merge a raw ``Event`` and a NOMINAL design: scale it, then :meth:`combine_scaled`. NEVER
        overridden -- the feature layout varies through :meth:`combine_scaled`.

        TWO WAYS TO WITHHOLD THE DESIGN, and they are not the same thing.

        ``reveal_design=False`` says the CALLER HAS the design but the NETWORK IS NOT TOLD IT. The
        detector still uses it wherever the MEASUREMENT depends on it -- the visible-window task applies
        its aperture, so the network sees what was actually captured -- and drops only what would
        announce which design produced it. This is what a design-blind training arm wants.

        ``design=None`` says there IS no design to use. Detectors whose measurement does not depend on
        the design (the straw family, the enzyme family) treat it exactly like ``reveal_design=False``;
        one whose measurement DOES depend on it cannot honour it and raises.

        Either way the features become narrower, never corrupted, and
        ``combined_event_shape(design=False)`` reports their shape."""
        return self.combine_scaled(
            event, None if design is None else self.to_scaled(design), mask=mask, reveal_design=reveal_design
        )

    def element_mask(self, event, mask):
        """Per-ELEMENT validity mask ``(..., n_elements)`` for the regressor aggregation -- the element
        axis matches :meth:`combine_scaled`, so each combine leaf IMPLEMENTS it: a hit-wise combine
        returns the hit ``mask`` unchanged (element == hit); a layer-wise combine returns the all-valid
        per-layer mask. Abstract here (it varies with the combine)."""
        raise NotImplementedError()

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
