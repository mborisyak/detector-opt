"""Deep set carrying TWO residual streams -- the per-element representation and the aggregated
statistic -- both gated by a single design-generated ``alpha``.

    e_i     = Embed(design_i)                                 per element
    E       = masked mean of e_i                              permutation-INVARIANT design summary
    alpha_t = Gate_t(E)                                       (..., D), zero-initialised

    block 0 (PLAIN, no gate, no residual):
              z_0, w_0           = Split(MLP_0(features))     (..., M, D)
              mu_0               = aggregate(z_0, w_0, mask)  (..., D)

    block t>0 (RESIDUAL):
              z'_{t+1}, w'_{t+1} = Split(MLP_t([z_t, broadcast(mu_t)]))
              mu'_{t+1}          = aggregate(z'_{t+1}, w'_{t+1}, mask)
              z_{t+1}            = alpha_t * z'_{t+1} + z_t
              mu_{t+1}           = alpha_t * mu'_{t+1} + mu_t

    out     = Head(mu_T)

HOW THIS DIFFERS from :mod:`detopt.nn.alpha_hyper_set_regressor`, which it does not replace. There the
per-element stream is residual and the aggregate is RECOMPUTED per block and concatenated forward, so
the set statistic has no memory of its own. Here the statistic is a residual stream in its own right:
each block proposes an INCREMENT to it and ``alpha`` decides how much of that increment is taken. A
block can therefore refine the event representation slightly rather than rebuild it, and the depth is
free at initialisation because every increment starts at zero.

ONE ``alpha`` PER BLOCK, APPLIED TO BOTH STREAMS, which is what the specification asks for. It is
per-FEATURE (shape ``(..., D)``), not scalar: different coordinates of the representation should be
allowed to switch on at different rates, exactly as the free per-unit ``alpha`` does in the
alpha-conv/alpha-set stacks. Using one ``alpha`` for both streams ties the element update to the
statistic update, so a block that is switched off is switched off entirely rather than half-acting.

``alpha`` IS PER-EVENT, NOT PER-ELEMENT, AND THAT IS WHAT KEEPS THE MODEL INVARIANT. The design is the
whole batch of experiments, so ``alpha`` is generated from a permutation-invariant summary of the
per-element design embeddings (a masked mean) and then broadcast over the element axis. A per-element
``alpha`` would still be equivariant and would also be defensible, but it would make the gate a
function of WHICH element it sits on, which is not what "the design controls alpha" means.

THE FIRST BLOCK DOES PLAIN AGGREGATION. It has the same MLP shape as the residual blocks but takes the
raw features (there is no ``mu`` to concatenate yet), takes no ``alpha`` and carries no residual: it
simply produces ``z_0`` and aggregates it to ``mu_0``. ``blocks`` counts ALL blocks, so ``blocks = 3``
is one plain block followed by two residual ones.

IDENTITY AT INITIALISATION means something specific here. Every gate head is zero-initialised in kernel
AND bias, so ``alpha = 0`` for every design before training and both streams pass through untouched:
``z_T = z_0`` and ``mu_T = mu_0``. The model therefore starts as EXACTLY ITS FIRST BLOCK -- a plain
one-aggregation deep set -- and the residual blocks switch themselves on from there. That is why depth
costs nothing in conditioning: a 5-block model begins life as the 1-block model.

``zero_design`` IS THE CAPACITY-MATCHED BASELINE: it feeds the embedding a zero design vector while
leaving every parameter and the generator untouched, so ``alpha`` becomes one learned constant per
feature instead of a function of the design. The design columns still reach the MAIN path -- they are
part of the measurement -- only the generator is blinded.

THE DESIGN IS TAKEN FROM THE TRAILING FEATURE COLUMNS. ``EnzymeDetector.combine_scaled`` appends each
experiment's own scaled design values as the last ``design_dim`` columns, so
``features[..., -design_dim:]`` IS the design, exactly and per element.

⛔️ THAT NEEDS THE DESIGN-REVEALED FEATURE LAYOUT. An arm whose ``Trainer.reveals_design()`` is
``False`` is fed features WITHOUT those columns, and the trailing ``design_dim`` MEASUREMENTS would be
read as a design -- the two layouts are not distinguishable from the width alone. Build this model
with ``design=True`` only.

⚠️ EVERY LINEAR CALL FORWARDS ``deterministic`` AND ``rngs``. ``EnsembleLinear`` applies DropConnect
only when handed both and defaults to ``deterministic=True``, so a call that omits them leaves
``dropconnect`` CONFIGURED BUT SILENTLY INERT. Assert that two STOCHASTIC passes DIFFER if this forward
is ever changed.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx

from .common import Model, Shape
from .set_regressor import EnsembleLeakyTanh, EnsembleLinear, make_activation, masked_weighted_aggregate

__all__ = ['AlphaHyperDualSetRegressor', 'DualResidualBlock', 'InvariantDesignGate']


def masked_mean(x, mask):
  """Permutation-invariant mean over the element axis, ignoring masked elements."""
  m = mask.astype(jnp.float32)[..., None]
  return jnp.sum(x * m, axis=-2) / (jnp.sum(m, axis=-2) + 1e-6)


class InvariantDesignGate(nnx.Module):
  """``design (..., M, d) -> alpha (..., D)`` per block: embed per element, average over elements
  (permutation-invariant), then one zero-initialised head per block.

  The heads are zero in kernel AND bias, so every ``alpha`` is exactly 0 before training."""

  def __init__(
    self, n_models: int | None, design_dim: int, features: int, channels: int, width: int, n_blocks: int, activation: str, *,
    rngs: nnx.Rngs, dropconnect: float | None = None
  ):
    self.hidden = EnsembleLinear(n_models, int(design_dim), int(features), rngs=rngs, dropconnect=dropconnect)
    self.activation = make_activation(activation, n_models, int(features))
    self.embed = EnsembleLinear(n_models, int(features), int(channels), rngs=rngs, dropconnect=dropconnect)
    heads = []
    for _ in range(int(n_blocks)):
      head = EnsembleLinear(n_models, int(channels), int(width), rngs=rngs, dropconnect=dropconnect)
      head.kernel[...] = jnp.zeros_like(head.kernel[...])
      heads.append(head)
    self.heads = nnx.List(heads)

  def __call__(self, design, mask, *, deterministic: bool = True, rngs=None):
    h = self.hidden(design, deterministic=deterministic, rngs=rngs)
    e = self.embed(self.activation(h), deterministic=deterministic, rngs=rngs)
    summary = masked_mean(e, mask)
    return [head(summary, deterministic=deterministic, rngs=rngs) for head in self.heads]


class DualResidualBlock(nnx.Module):
  """One block: map its input to ``z'`` and ``w'``, aggregate to ``mu'``, return both.

  ``in_dim`` is the raw feature count for the PLAIN first block and ``2 * width`` for the residual
  blocks, which are fed ``[z, broadcast(mu)]``. The block never sees ``alpha`` -- the caller applies it
  -- so the gate and the map stay separable, and the plain block is this same class used without one."""

  def __init__(
    self, n_models: int | None, in_dim: int, width: int, hidden: int, p_dropout: float | None, activation: str, *,
    rngs: nnx.Rngs, dropconnect: float | None = None
  ):
    self.project = EnsembleLinear(n_models, int(in_dim), int(hidden), rngs=rngs, dropconnect=dropconnect)
    self.activation = make_activation(activation, n_models, int(hidden))
    has_dropout = p_dropout is not None and p_dropout > 0
    self.dropout = nnx.data(nnx.Dropout(rate=p_dropout, rngs=rngs)) if has_dropout else None
    self.output = EnsembleLinear(n_models, int(hidden), 2 * int(width), rngs=rngs)

  def __call__(self, x, mask, *, deterministic: bool = True, rngs=None):
    h = self.project(x, deterministic=deterministic, rngs=rngs)
    h = self.activation(h)
    if self.dropout is not None:
      h = self.dropout(h, deterministic=deterministic, rngs=rngs)
    value, weight_logit = jnp.split(self.output(h, deterministic=deterministic, rngs=rngs), 2, axis=-1)
    return value, masked_weighted_aggregate(value, weight_logit, mask)


class AlphaHyperDualSetRegressor(Model):
  """``(features, mask) -> predictions`` deep set with residual element AND statistic streams, both
  gated by a design-generated per-feature ``alpha``.

  Parameters
  ----------
  width : the working width ``D`` carried by both residual streams, constant across blocks.
  hidden : the per-block MLP's hidden width.
  blocks : TOTAL number of blocks. The first does plain aggregation; the remaining `blocks - 1`
      are residual and gated. `blocks = 1` is exactly a plain one-aggregation deep set.
  design_dim : how many TRAILING feature columns are the design.
  embedding_features, embedding_channels : the design embedding's hidden and output widths.
  n_models : ``None`` for one network, an int ``>= 1`` for an ensemble.
  p_dropout : dropout inside a block's branch -- inert until the gates leave zero.
  dropconnect : weight dropout on every linear map except each block's output, which emits the
      aggregation gate.
  zero_design : withhold the design from the GENERATOR, giving the capacity-matched baseline.
  """

  def __init__(
    self, input_shape: Shape, target_shape: Shape, ground_truth_shape: Shape, width: int = 24, hidden: int = 32,
    blocks: int = 3, design_dim: int = 2, embedding_features: int = 16, embedding_channels: int = 8,
    n_models: int | None = None, p_dropout: float | None = None, dropconnect: float | None = None,
    activation: str = 'leaky-tanh', zero_design: bool = False, *, rngs: nnx.Rngs
  ):
    if int(blocks) < 1:
      raise ValueError('blocks must be at least 1')
    self.rngs = rngs
    self.n_features_in = int(input_shape[-1])
    self.target_dim = int(target_shape[0])
    self.design_dim = int(design_dim)
    self.zero_design = bool(zero_design)
    if self.design_dim < 1 or self.design_dim > self.n_features_in:
      raise ValueError(f'design_dim must be in [1, {self.n_features_in}], got {design_dim}')
    self.n_models = None if n_models is None else int(n_models)
    if self.n_models is not None and self.n_models < 1:
      raise ValueError('n_models must be None or an int >= 1')
    self.width = int(width)

    self.first = DualResidualBlock(
      self.n_models, self.n_features_in, self.width, int(hidden), p_dropout, activation, rngs=rngs, dropconnect=dropconnect
    )
    self.gate = InvariantDesignGate(
      self.n_models, self.design_dim, int(embedding_features), int(embedding_channels), self.width,
      int(blocks) - 1, activation, rngs=rngs, dropconnect=dropconnect
    )
    self.blocks = nnx.List([
      DualResidualBlock(
        self.n_models, 2 * self.width, self.width, int(hidden), p_dropout, activation, rngs=rngs, dropconnect=dropconnect
      ) for _ in range(int(blocks) - 1)
    ])
    self.output = EnsembleLinear(self.n_models, self.width, self.target_dim, rngs=rngs, dropconnect=dropconnect)

  def ensemble(self) -> int | None:
    return self.n_models

  def __call__(self, features, mask, *, deterministic: bool = True, rngs=None):
    design = features[..., -self.design_dim:]
    if self.zero_design:
      design = jnp.zeros_like(design)
    alphas = self.gate(design, mask, deterministic=deterministic, rngs=rngs)

    z, mu = self.first(features, mask, deterministic=deterministic, rngs=rngs)

    for block, alpha in zip(self.blocks, alphas):
      context = jnp.broadcast_to(jnp.expand_dims(mu, -2), z.shape)
      z_increment, mu_increment = block(jnp.concatenate([z, context], axis=-1), mask, deterministic=deterministic, rngs=rngs)
      z = jnp.expand_dims(alpha, -2) * z_increment + z
      mu = alpha * mu_increment + mu

    return self.output(mu, deterministic=deterministic, rngs=rngs)
