"""Deep set whose per-element MLP is a RESIDUAL stack with zero-initialised per-unit scaling.

The set structure is not negotiable here: the design is a BATCH of interchangeable experiments and
the inference must be permutation-invariant, so aggregation stays exactly as in
:mod:`detopt.nn.set_regressor` (``masked_weighted_aggregate``, reused rather than reimplemented).
What changes is what happens to each element BEFORE aggregation.

Each block projects to a working ``width`` and then applies ``depth`` residual units

    h <- h + alpha * f(h),     f = LeakyTanh -> Linear   (dropout, when asked, inside the branch)

with ``alpha`` a PER-UNIT parameter initialised to ZERO. Two consequences, and the second is the
reason the architecture is worth trying at all:

* **At initialisation the network is the identity map** in its residual part, so depth costs
  nothing in conditioning: a 6-deep stack starts as well-behaved as a 1-deep one and the optimiser
  decides how much of each branch to switch on. That is what makes capacity cheap to add here,
  where the shipped set regressor is a plain stack at widths of 16-24 and adding depth to it
  degrades optimisation.
* **A zero alpha makes the branch inert at step zero.** A peer session measuring the analogous
  architecture found its dropout column bit-identical with and without dropout on the first design
  for exactly this reason -- dropout inside a branch scaled by zero cannot change the output. That
  is an ARTEFACT of the initialisation, not evidence that dropout does nothing, and it disappears
  once the alphas move. Anyone reading a first-design comparison of this model should know that.

Ensembling follows the same convention as the rest of the library: ``n_models=None`` is one network
with no leading axis, ``n_models=n`` gives every layer a leading member axis evaluated in one
einsum. Members are independent -- separate kernels, separate alphas, independent dropout masks --
so the ensemble averages genuinely different functions rather than one function's noise.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape
from .set_regressor import EnsembleLeakyTanh, EnsembleLinear, masked_weighted_aggregate

__all__ = ["AlphaSetRegressor"]


class EnsembleAlphaResidual(nnx.Module):
  """One residual unit ``h -> h + alpha * Linear(LeakyTanh(Dropout(h)))``, ``alpha`` zero-init.

  ``alpha`` is per-unit (and per-member when ensembled) rather than scalar: different features
  switch on at different rates, which is the point of the parameterisation -- a scalar gate makes
  the whole branch a single knob and recovers a plain deep stack as soon as it leaves zero.
  """

  def __init__(self, n_models: int | None, width: int, p_dropout: float | None, *, rngs: nnx.Rngs):
    self.n_models = n_models
    self.activation = EnsembleLeakyTanh(n_models, width)
    self.linear = EnsembleLinear(n_models, width, width, rngs=rngs)
    # One assignment, not `= None` then maybe-a-module: nnx types an attribute on first
    # assignment, and a static `None` cannot later hold a Dropout.
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None
    shape = (width, ) if n_models is None else (n_models, width)
    self.alpha = nnx.Param(jnp.zeros(shape))

  def __call__(self, h, *, deterministic: bool = True, rngs=None):
    branch = h
    if self.dropout is not None:
      branch = self.dropout(branch, deterministic=deterministic, rngs=rngs)
    branch = self.linear(self.activation(branch))
    if self.n_models is None:
      alpha = self.alpha[...]
    else:
      alpha = self.alpha[...].reshape((h.shape[0], ) + (1, ) * (h.ndim - 2) + (h.shape[-1], ))
    return h + alpha * branch


class AlphaResSetBlock(nnx.Module):
  """Per-element residual stack producing ``(value, weight_logit)``, the same pair
  :class:`detopt.nn.set_regressor.EnsembleSetBlock` produces, so aggregation is unchanged."""

  def __init__(
    self, n_models: int | None, in_dim: int, width: int, depth: int, out_dim: int, p_dropout: float | None, *, rngs: nnx.Rngs
  ):
    if depth < 1:
      raise ValueError("depth must be >= 1")
    self.project = EnsembleLinear(n_models, in_dim, width, rngs=rngs)
    self.units = nnx.List([EnsembleAlphaResidual(n_models, width, p_dropout, rngs=rngs) for _ in range(depth)])
    self.output = EnsembleLinear(n_models, width, 2 * out_dim, rngs=rngs)

  def __call__(self, x, *, deterministic: bool = True, rngs=None):
    h = self.project(x)
    for unit in self.units:
      h = unit(h, deterministic=deterministic, rngs=rngs)
    return jnp.split(self.output(h), 2, axis=-1)


class AlphaSetRegressor(Model):
  """``(features, mask) -> predictions`` deep set with residual per-element blocks.

  Parameters
  ----------
  features : one entry per block, each the block's OUTPUT width. The per-element stack inside a
      block is described by ``width``/``depth`` instead of a list of hidden sizes, because the
      residual form needs a constant working width.
  width, depth : the residual stack's working width and number of units, shared by every block.
  n_models : ``None`` for a single network, an int ``>= 1`` for an ensemble.
  p_dropout : dropout INSIDE each residual branch. Note it is activation dropout, not
      drop-connect -- no mask touches a kernel.
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, features: Sequence[int] = (16, 24),
    width: int = 64, depth: int = 4, n_models: int | None = None, p_dropout: float | None = None, *, rngs: nnx.Rngs,
  ):
    self.rngs = rngs
    self.n_features_in = int(input_shape[-1])
    self.target_dim = int(target_shape[0])
    self.n_models = None if n_models is None else int(n_models)
    if self.n_models is not None and self.n_models < 1:
      raise ValueError("n_models must be None or an int >= 1")

    blocks: list[AlphaResSetBlock] = []
    n_in = self.n_features_in
    for out_dim in features:
      blocks.append(AlphaResSetBlock(self.n_models, n_in, int(width), int(depth), int(out_dim), p_dropout, rngs=rngs))
      # After aggregation the next block sees [value_element, event_representation].
      n_in = 2 * int(out_dim)
    self.blocks = nnx.List(blocks)
    self.output = EnsembleLinear(self.n_models, int(features[-1]), self.target_dim, rngs=rngs)

  def ensemble(self) -> int | None:
    return self.n_models

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    # Identical aggregation to SetRegressor -- the elements are a SET and the masking lives in
    # masked_weighted_aggregate. Written out rather than inherited: the blocks differ, and the
    # library's rule is that a concrete forward is not overridden.
    result = features

    *rest, last = self.blocks
    for block in rest:
      value_element, weight_element = block(result, deterministic=deterministic, rngs=rngs)
      event_repr = masked_weighted_aggregate(value_element, weight_element, mask)
      event_per_element = jnp.broadcast_to(jnp.expand_dims(event_repr, -2), value_element.shape)
      result = jnp.concatenate([value_element, event_per_element], axis=-1)

    value_element, weight_element = last(result, deterministic=deterministic, rngs=rngs)
    event_repr = masked_weighted_aggregate(value_element, weight_element, mask)
    return self.output(event_repr)
