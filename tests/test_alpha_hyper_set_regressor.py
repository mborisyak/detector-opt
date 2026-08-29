"""The design source the alpha-hyper set regressor reads, and the blind twin it is compared against."""

import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

import detopt
from detopt.nn.alpha_hyper_set_regressor import design_from_features


def parameter_count(model):
  return sum(int(np.prod(leaf[...].shape)) for _path, leaf in nnx.state(model, nnx.Param).flat_state())


def straw_features(seed, batch=3, elements=7):
  """A synthetic ``[TDC, norm_z, wire_y_left, wire_y_right]`` batch built from KNOWN parts, so the
  extraction can be checked against the wire offset and the straw position that produced it."""
  rng = np.random.default_rng(seed)
  tdc = rng.standard_normal((batch, elements)).astype('float32')
  norm_z = rng.uniform(-1.0, 1.0, size=(batch, elements)).astype('float32')
  straw_y = rng.uniform(-1.0, 1.0, size=(batch, elements)).astype('float32')
  offset = rng.uniform(0.01, 0.2, size=(batch, elements)).astype('float32')
  features = np.stack([tdc, norm_z, straw_y - offset, straw_y + offset], axis=-1)
  return jnp.asarray(features), norm_z, straw_y, offset


def test_straw_geometry_recovers_the_design_and_cancels_the_straw(seed):
  """``norm_z`` comes through untouched and the span is ``2 * offset``, in which ``straw_y`` cancels."""
  features, norm_z, straw_y, offset = straw_features(seed)
  design = np.asarray(design_from_features(features, 'straw-geometry', 2))
  assert design.shape == features.shape[:-1] + (2, )
  assert float(np.max(np.abs(design[..., 0] - norm_z))) < 1e-6
  assert float(np.max(np.abs(design[..., 1] - 2.0 * offset))) < 1e-6

  # Move every straw and hold the geometry: the raw columns move, the extracted design does not.
  shift = np.asarray(np.random.default_rng(seed + 1).standard_normal(straw_y.shape), 'float32')
  moved = features.at[..., 2].add(jnp.asarray(shift)).at[..., 3].add(jnp.asarray(shift))
  moved_design = np.asarray(design_from_features(moved, 'straw-geometry', 2))
  assert float(np.max(np.abs(np.asarray(moved)[..., 2] - np.asarray(features)[..., 2]))) > 1e-3
  assert float(np.max(np.abs(moved_design - design))) < 1e-6


def test_trailing_source_is_unchanged_and_an_unknown_source_raises(seed):
  features, _norm_z, _straw_y, _offset = straw_features(seed)
  assert float(
    np.max(np.abs(np.asarray(design_from_features(features, 'trailing', 2)) - np.asarray(features)[..., -2:]))
  ) == 0.0
  with pytest.raises(ValueError):
    design_from_features(features, 'wire-columns', 2)


def test_blind_twin_is_capacity_matched_and_identical_at_initialisation(seed):
  """``zero_design`` withholds the design from the GENERATOR only, so the two models have the same
  parameters and -- because every gate head is zero-initialised -- the same output before training."""
  block = dict(
    features=[12, 8], width=12, depth=2, design_features='straw-geometry', embedding_features=8, embedding_channels=4,
    n_models=None
  )
  built = [
    detopt.nn.AlphaHyperSetRegressor(
      input_shape=(7, 4), target_shape=(3, ), ground_truth_shape=(1, ), zero_design=blind, rngs=nnx.Rngs(seed), **block
    ) for blind in (False, True)
  ]
  counts = [parameter_count(model) for model in built]
  assert counts[0] == counts[1]

  features, _norm_z, _straw_y, _offset = straw_features(seed)
  mask = jnp.asarray(np.ones(features.shape[:-1], 'int32'))
  sighted, blind = (np.asarray(m(features, mask)) for m in built)
  assert float(np.max(np.abs(sighted - blind))) < 1e-6
