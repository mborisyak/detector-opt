"""The spliced norms of ``scripts/probe_scheduled_norm.py`` must not reach the set aggregation.

``SetRegressor`` gets to skip masking its intermediate activations for ONE reason, stated in
``detopt/nn/set_regressor.py``: the per-hit MLP is POINTWISE, so hits meet only in
``masked_weighted_aggregate``, where the gate is multiplied by the mask. A normalisation that
reduced over the hit axis ``M`` would silently void that -- a padded slot's statistics would enter a
live slot's activation and reach the aggregate through the front door.

Both norms here reduce over the trailing FEATURE axis, so the property should survive. These tests
measure that rather than trusting it, at the architecture the probe actually runs
(``features [[32, 48], [48, 32]]``, ``M = 128``, ``F = 5``, single net), for every arm and at both
ends of the C schedule.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
from flax import nnx

import detopt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from probe_scheduled_norm import ARMS, Coefficient, ParameterFreeLayerNorm, ScheduledSoftNorm, splice_norm  # noqa: E402

M, F, T, B = 128, 5, 9, 4
FEATURES = [[32, 48], [48, 32]]


def _model(seed):
  return detopt.nn.SetRegressor(
    input_shape=(M, F), target_shape=(T, ), ground_truth_shape=(1, ), features=FEATURES, n_models=None, p_dropout=None,
    activation="celu", rngs=nnx.Rngs(seed)
  )


def _spliced(arm, seed, carrier):
  model = _model(seed)
  if arm == "layernorm":
    splice_norm(model, ParameterFreeLayerNorm)
  elif arm in ("c_const", "c_decay"):
    splice_norm(model, lambda: ScheduledSoftNorm(carrier))
  return model


def _batch(seed, live):
  rng = np.random.default_rng(seed)
  features = jnp.asarray(rng.standard_normal((B, M, F)).astype("float32"))
  mask = jnp.asarray(np.concatenate([np.ones((B, live)), np.zeros((B, M - live))], axis=1).astype("int32"))
  return rng, features, mask


def test_padded_slots_cannot_reach_the_prediction(seed):
  """Perturbing ONLY masked slots must leave every arm's prediction unchanged.

    The padded block is overwritten with large values and with a constant row -- the latter drives
    ``ParameterFreeLayerNorm``'s variance to zero, the worst case for its epsilon."""
  live = M // 2
  for arm in ARMS:
    for coefficient in (0.0, 0.5, 1.0):
      carrier = Coefficient(jnp.float32(coefficient))
      model = _spliced(arm, seed, carrier)
      rng, features, mask = _batch(seed, live)
      prediction = model(features, mask, deterministic=True)
      assert prediction.shape == (B, T)
      for filler in (rng.standard_normal((B, M - live, F)).astype("float32") * 1000.0, np.full(
        (B, M - live, F), 7.0, dtype="float32"), np.zeros((B, M - live, F), dtype="float32")):
        perturbed = features.at[:, live:, :].set(jnp.asarray(filler))
        moved = float(jnp.max(jnp.abs(prediction - model(perturbed, mask, deterministic=True))))
        assert moved < 1e-4, f"{arm} at C={coefficient}: padded-slot perturbation leaked {moved} into the output"


def test_no_arm_produces_a_non_finite_activation(seed):
  """A degenerate masked row (zero variance, zero norm) must not make anything non-finite."""
  live = M // 2
  for arm in ARMS:
    for coefficient in (0.0, 1.0):
      carrier = Coefficient(jnp.float32(coefficient))
      model = _spliced(arm, seed, carrier)
      _rng, features, mask = _batch(seed, live)
      features = features.at[:, live:, :].set(0.0)
      prediction = model(features, mask, deterministic=True)
      assert bool(jnp.all(jnp.isfinite(prediction))), f"{arm} at C={coefficient} produced a non-finite prediction"


def test_the_aggregation_stays_permutation_invariant(seed):
  """Hits are a SET: permuting them (and the mask with them) must not move the prediction."""
  live = M // 2
  for arm in ARMS:
    carrier = Coefficient(jnp.float32(1.0))
    model = _spliced(arm, seed, carrier)
    rng, features, mask = _batch(seed, live)
    prediction = model(features, mask, deterministic=True)
    order = rng.permutation(M)
    moved = float(jnp.max(jnp.abs(prediction - model(features[:, order, :], mask[:, order], deterministic=True))))
    assert moved < 1e-4, f"{arm}: permuting the hit axis moved the prediction by {moved}"


def test_no_norm_wraps_a_block_output_map(seed):
  """The norms go in each block's SHARED MLP only. ``block.output`` emits the aggregation gate, so a
    norm on it would rescale the gate itself rather than the map being normalised."""
  from detopt.nn.set_regressor import EnsembleLinear

  carrier = Coefficient(jnp.float32(1.0))
  for arm in ("layernorm", "c_const", "c_decay"):
    model = _spliced(arm, seed, carrier)
    spliced = 0
    for block in model.blocks:
      assert isinstance(block.output, EnsembleLinear), f"{arm}: block.output is no longer the bare gate map"
      layers = list(block.shared)
      for i, layer in enumerate(layers):
        if isinstance(layer, (ParameterFreeLayerNorm, ScheduledSoftNorm)):
          spliced += 1
          assert i > 0 and isinstance(layers[i - 1], EnsembleLinear), f"{arm}: a norm does not follow a hidden linear"
    assert spliced == len(FEATURES), f"{arm}: expected one norm per block, found {spliced}"


def test_soft_norm_at_zero_is_exactly_the_identity(seed):
  """At the end of the schedule ``c_decay`` must BE the baseline architecture, not merely resemble it.

    ``x / (1 + C * rms(x))`` at C = 0 is the identity, so the annealed arm's read-out network is the
    unspliced one and the schedule is a training device rather than a different model at test time."""
  live = M // 2
  _rng, features, mask = _batch(seed, live)
  baseline = _model(seed)(features, mask, deterministic=True)
  soft = _spliced("c_decay", seed, Coefficient(jnp.float32(0.0)))(features, mask, deterministic=True)
  moved = float(jnp.max(jnp.abs(baseline - soft)))
  assert moved == 0.0, f"the soft norm at C=0 is not the identity: it moved the prediction by {moved}"


def test_soft_norm_at_one_actually_changes_the_prediction(seed):
  """The counterpart: at C = 1 the arm must NOT be the baseline, or the probe compares nothing."""
  live = M // 2
  _rng, features, mask = _batch(seed, live)
  baseline = _model(seed)(features, mask, deterministic=True)
  soft = _spliced("c_const", seed, Coefficient(jnp.float32(1.0)))(features, mask, deterministic=True)
  assert float(jnp.max(jnp.abs(baseline - soft))) > 1e-4, "the soft norm at C=1 left the prediction unchanged"
