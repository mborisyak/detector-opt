"""Deep set whose per-unit residual gate is GENERATED from the design -- the set-regressor twin of
:mod:`detopt.nn.alpha_hyper_conv_regressor`.

    embedding = Linear(LeakyTanh(Linear(design)))              ``(..., k)``
    unit:  h <- h + alpha(embedding) * Linear(LeakyTanh(Dropout(h)))

:class:`detopt.nn.alpha_set_regressor.AlphaSetRegressor` already carries a per-unit `alpha`,
zero-initialised so each unit starts as the identity. That gate is the natural thing for a
hypernetwork to own: it decides how much each unit contributes, per feature, and nothing else in the
unit changes. Here `alpha` is no longer a free parameter but the output of a linear head reading a
design embedding, which makes this a hypernetwork in the strict sense -- it GENERATES parameters.

THE DESIGN IS TAKEN STRAIGHT FROM THE ELEMENT FEATURES, not recovered. `EnzymeDetector.combine_scaled`
appends each experiment's own scaled design values as the LAST `design_dim` columns of that element's
feature vector, so `features[..., -design_dim:]` IS the design, exactly and per element. Nothing like
the conv stack's `recover_window` is needed or wanted here.

⛔️ THIS NEEDS THE DESIGN-REVEALED FEATURE LAYOUT. An arm whose `Trainer.reveals_design()` is `False`
is fed features WITHOUT those columns, and the trailing `design_dim` MEASUREMENTS would be read as a
design. `'straw-geometry'` catches that case by requiring 4 features; `'trailing'` cannot tell the two
layouts apart, so build this model with `design=True` only.

PERMUTATION INVARIANCE IS PRESERVED. The gate for element `i` reads element `i`'s own design values,
so the per-element map stays equivariant and aggregation -- unchanged, `masked_weighted_aggregate` --
makes the whole model invariant. A gate computed from the aggregated set would NOT be equivariant
element-by-element and is deliberately not what this does.

THE EMBEDDING IS COMPUTED ONCE, FROM THE ORIGINAL FEATURES, AND THREADED INTO EVERY UNIT. After the
first block a element's vector is `[value_element, event_representation]` and no longer carries the
design columns, so recomputing it downstream would read the wrong numbers.

IDENTITY AT INITIALISATION IS PRESERVED, and it is the reason this works at all. The gate head is
zero-initialised in BOTH kernel and bias -- `EnsembleLinear` draws a Lecun-normal kernel, so the
kernel is zeroed explicitly after construction -- giving `alpha(d) = 0` for every design before
training, exactly as the free `alpha` is zero-init in the stack this replaces. Dropout inside the
branch is likewise inert until the gates leave zero.

`zero_design` IS THE CAPACITY-MATCHED BASELINE. It feeds the embedding a ZERO design vector while
leaving every parameter and the generator untouched, so `alpha` becomes one learned constant per
feature instead of a function of the design -- the same gate the unconditioned stack learns freely,
reached through the same weights.

⛔️ IT IS NOT A DESIGN-BLIND BASELINE, and comparing against it does NOT vary the design INFORMATION.
The design reaches this network TWICE -- once through the generator, and once through the trailing
feature columns, which the set regressor is built to read and which `zero_design` does not touch. Both
conditions therefore see the design. The comparison answers "does conditioning the GATE on the design
pay", and nothing wider. Withholding the design from the FEATURES is
`Detector.combine(..., reveal_design=False)` and is a property of the training arm, not of this
module.

⚠️ EVERY LINEAR CALL FORWARDS ``deterministic`` AND ``rngs``. `EnsembleLinear` applies DropConnect
only when it is handed both, and defaults to `deterministic=True`, so a call that omits them leaves
`dropconnect` CONFIGURED BUT SILENTLY INERT -- no error, no warning, just no regularisation. Caught
here by asserting that two stochastic passes DIFFER; keep that check if this forward is ever changed.

ENSEMBLING follows the library convention: `n_models=None` is one network, `n_models=n` gives every
layer -- generator included -- its own member, so members have independent generators rather than a
shared one.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape
from .set_regressor import EnsembleLeakyTanh, EnsembleLinear, masked_weighted_aggregate

__all__ = [
  'AlphaHyperSetRegressor', 'EnsembleAlphaHyperResidual', 'AlphaHyperResSetBlock', 'EnsembleDesignEmbedding',
  'design_from_features'
]

STRAW_NORM_Z, STRAW_WIRE_LEFT, STRAW_WIRE_RIGHT = 1, 2, 3


def design_from_features(features, source: str, design_dim: int):
  """The per-element DESIGN vector, extracted from a detector's combined features.

  ``'trailing'`` -- the last ``design_dim`` columns ARE the design, which is how the enzyme detectors
  build their features.

  ``'straw-geometry'`` -- the four-feature straw combine appends NO design vector: it emits
  ``[TDC, norm_z, wire_y_left, wire_y_right]`` and the design reaches the network only through the
  geometry. Two design coordinates are recoverable per hit and MEASUREMENT-FREE given the hit's layer:
  ``norm_z`` is that layer's own z, and ``wire_y_right - wire_y_left = 2 * layer_width * tan(angle) /
  y_scale`` is that layer's stereo angle, in which the fired straw's ``straw_y`` CANCELS EXACTLY.
  Handing the gate the raw wire columns instead would condition it on WHICH STRAW FIRED -- a
  measurement -- so this contrast is taken and not the columns."""
  if source == 'trailing':
    return features[..., -int(design_dim):]
  if source == 'straw-geometry':
    span = features[..., STRAW_WIRE_RIGHT] - features[..., STRAW_WIRE_LEFT]
    return jnp.stack([features[..., STRAW_NORM_Z], span], axis=-1)
  raise ValueError(f"design_features must be 'trailing' or 'straw-geometry', got {source!r}")



class EnsembleDesignEmbedding(nnx.Module):
  """``design (..., d) -> (..., channels)`` through one hidden ``LeakyTanh`` layer, per member."""

  def __init__(
    self, n_models: int | None, design_dim: int, features: int, channels: int, *, rngs: nnx.Rngs,
    dropconnect: float | None = None
  ):
    self.hidden = EnsembleLinear(n_models, int(design_dim), int(features), rngs=rngs, dropconnect=dropconnect)
    self.activation = EnsembleLeakyTanh(n_models, int(features))
    self.output = EnsembleLinear(n_models, int(features), int(channels), rngs=rngs, dropconnect=dropconnect)

  def __call__(self, design, *, deterministic: bool = True, rngs=None):
    hidden = self.hidden(design, deterministic=deterministic, rngs=rngs)
    return self.output(self.activation(hidden), deterministic=deterministic, rngs=rngs)


class EnsembleAlphaHyperResidual(nnx.Module):
  """One residual unit ``h -> h + alpha(embedding) * Linear(LeakyTanh(Dropout(h)))``.

  The gate head is zero-initialised in kernel and bias, so the unit is the identity at initialisation
  for every design."""

  def __init__(
    self, n_models: int | None, width: int, embedding_channels: int, p_dropout: float | None, *, rngs: nnx.Rngs,
    dropconnect: float | None = None
  ):
    self.n_models = n_models
    self.activation = EnsembleLeakyTanh(n_models, width)
    self.linear = EnsembleLinear(n_models, width, width, rngs=rngs, dropconnect=dropconnect)
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None
    self.gate = EnsembleLinear(n_models, int(embedding_channels), width, rngs=rngs, dropconnect=dropconnect)
    # `EnsembleLinear` draws a Lecun-normal kernel; the gate must start at exactly zero so the unit is
    # the identity for every design. The bias is already zeros.
    self.gate.kernel[...] = jnp.zeros_like(self.gate.kernel[...])

  def __call__(self, h, embedding, *, deterministic: bool = True, rngs=None):
    branch = h
    if self.dropout is not None:
      branch = self.dropout(branch, deterministic=deterministic, rngs=rngs)
    branch = self.linear(self.activation(branch), deterministic=deterministic, rngs=rngs)
    return h + self.gate(embedding, deterministic=deterministic, rngs=rngs) * branch


class AlphaHyperResSetBlock(nnx.Module):
  """Per-element conditioned residual stack producing ``(value, weight_logit)`` -- the same pair
  ``EnsembleSetBlock`` produces, so aggregation is unchanged."""

  def __init__(
    self, n_models: int | None, in_dim: int, width: int, depth: int, out_dim: int, embedding_channels: int,
    p_dropout: float | None, *, rngs: nnx.Rngs, dropconnect: float | None = None
  ):
    if depth < 1:
      raise ValueError('depth must be >= 1')
    self.project = EnsembleLinear(n_models, in_dim, width, rngs=rngs, dropconnect=dropconnect)
    self.units = nnx.List([
      EnsembleAlphaHyperResidual(n_models, width, embedding_channels, p_dropout, rngs=rngs, dropconnect=dropconnect)
      for _ in range(depth)
    ])
    self.output = EnsembleLinear(n_models, width, 2 * out_dim, rngs=rngs, dropconnect=dropconnect)

  def __call__(self, x, embedding, *, deterministic: bool = True, rngs=None):
    h = self.project(x, deterministic=deterministic, rngs=rngs)
    for unit in self.units:
      h = unit(h, embedding, deterministic=deterministic, rngs=rngs)
    return jnp.split(self.output(h, deterministic=deterministic, rngs=rngs), 2, axis=-1)


class AlphaHyperSetRegressor(Model):
  """``(features, mask) -> predictions`` deep set whose residual gates are generated from the design.

  Parameters
  ----------
  features : one entry per block, each the block's OUTPUT width.
  width, depth : the residual stack's working width and number of units, shared by every block.
  design_dim : how many TRAILING feature columns are the design (2 for the enzyme detectors). IGNORED
      when ``design_features='straw-geometry'``, which always yields 2.
  design_features : where the design is read from -- ``'trailing'`` (enzyme) or ``'straw-geometry'``
      (the 4-feature straw combine, which appends no design vector). See ``design_from_features``.
  embedding_features, embedding_channels : the design embedding's hidden width and its output width,
      the latter being what every gate head reads.
  n_models : ``None`` for a single network, an int ``>= 1`` for an ensemble.
  p_dropout : dropout INSIDE each residual branch -- inert until the gates leave zero.
  dropconnect : weight dropout on every linear map, the generator included, as the shipped set
      regressor uses.
  zero_design : withhold the design from the GENERATOR, giving the capacity-matched baseline.
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, features: Sequence[int] = (16, 24),
    width: int = 64, depth: int = 4, design_dim: int = 2, embedding_features: int = 16, embedding_channels: int = 8,
    n_models: int | None = None, p_dropout: float | None = None, dropconnect: float | None = None, zero_design: bool = False,
    design_features: str = 'trailing', *, rngs: nnx.Rngs
  ):
    self.rngs = rngs
    self.target_dim = int(target_shape[0])
    self.zero_design = bool(zero_design)
    self.design_features = str(design_features)
    if self.design_features == 'straw-geometry':
      self.n_features_in = int(input_shape[-1])
      if self.n_features_in != 4:
        raise ValueError(f'straw-geometry needs the 4-feature straw combine, got {self.n_features_in} features')
      self.design_dim = 2
    else:
      self.n_features_in = int(input_shape[-1])
      self.design_dim = int(design_dim)
      if self.design_dim < 1 or self.design_dim > self.n_features_in:
        raise ValueError(f'design_dim must be in [1, {self.n_features_in}], got {design_dim}')
    self.n_models = None if n_models is None else int(n_models)
    if self.n_models is not None and self.n_models < 1:
      raise ValueError('n_models must be None or an int >= 1')

    self.embedding = EnsembleDesignEmbedding(
      self.n_models, self.design_dim, int(embedding_features), int(embedding_channels), rngs=rngs, dropconnect=dropconnect
    )
    blocks: list[AlphaHyperResSetBlock] = []
    n_in = self.n_features_in
    for out_dim in features:
      blocks.append(
        AlphaHyperResSetBlock(
          self.n_models, n_in, int(width), int(depth), int(out_dim), int(embedding_channels), p_dropout, rngs=rngs,
          dropconnect=dropconnect
        )
      )
      n_in = 2 * int(out_dim)
    self.blocks = nnx.List(blocks)
    self.output = EnsembleLinear(self.n_models, int(features[-1]), self.target_dim, rngs=rngs, dropconnect=dropconnect)

  def ensemble(self) -> int | None:
    return self.n_models

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    design = design_from_features(features, self.design_features, self.design_dim)
    if self.zero_design:
      design = jnp.zeros_like(design)
    embedding = self.embedding(design, deterministic=deterministic, rngs=rngs)

    result = features
    *rest, last = self.blocks
    for block in rest:
      value_element, weight_element = block(result, embedding, deterministic=deterministic, rngs=rngs)
      event_repr = masked_weighted_aggregate(value_element, weight_element, mask)
      event_per_element = jnp.broadcast_to(jnp.expand_dims(event_repr, -2), value_element.shape)
      result = jnp.concatenate([value_element, event_per_element], axis=-1)

    value_element, weight_element = last(result, embedding, deterministic=deterministic, rngs=rngs)
    event_repr = masked_weighted_aggregate(value_element, weight_element, mask)
    return self.output(event_repr, deterministic=deterministic, rngs=rngs)
