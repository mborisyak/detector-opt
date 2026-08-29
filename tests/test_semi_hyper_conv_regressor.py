"""`SemiHyperConvRegressor`: the alpha-conv stack with the design injected at every residual unit.

Three things are checked here that the architecture stands or falls on -- that the window RECOVERED
from the mask channel is the design (up to the half pixel a binary mask can place an edge to), that
the resampled image lines up with each stage's own resolution, and that the model keeps the
alpha-conv contract (single network, identity at initialisation, same output shape). The detector
reads a handful of digits rather than all 60000, so construction stays cheap; the tests skip when
the arrow file is not on this machine.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from detopt.detector import MNISTDetector
from detopt.nn import AlphaConvRegressor, SemiHyperConvRegressor
from detopt.nn.semi_hyper_conv_regressor import area_pool, recover_window
from detopt.nn.trainer.common import regressor_rngs

DATA_PATH = '/home/max/dev/data/mnist/mnist-train.arrow'
N_EVENTS = 64
CHANNELS = (16, 24, 32)
BLOCKS = 2
KERNEL_SIZE = 3
EMBEDDING_CHANNELS = 8
EMBEDDING_FEATURES = 32

pytestmark = pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f'no MNIST arrow file at {DATA_PATH}')


@pytest.fixture(scope='module')
def detector():
  return MNISTDetector(data_path=DATA_PATH, n_events=N_EVENTS)


def _alpha(detector):
  return AlphaConvRegressor(
    detector.combined_event_shape(), (detector.target_dim(), ), (detector.ground_truth_dim(), ), channels=CHANNELS,
    blocks=BLOCKS, kernel_size=KERNEL_SIZE, p_dropout=None, rngs=regressor_rngs(0)
  )


def _semi(detector):
  return SemiHyperConvRegressor(
    detector.combined_event_shape(), (detector.target_dim(), ), (detector.ground_truth_dim(), ), channels=CHANNELS,
    blocks=BLOCKS, kernel_size=KERNEL_SIZE, p_dropout=None, embedding_channels=EMBEDDING_CHANNELS,
    embedding_features=EMBEDDING_FEATURES, rngs=regressor_rngs(0)
  )


def _features(detector, design_scaled, n):
  _gt, event, mask, _target = detector(np.broadcast_to(design_scaled, (n, 4)).copy(), np.arange(n, dtype=np.int64))
  return detector.combine_scaled(event, jnp.broadcast_to(jnp.asarray(design_scaled),
                                                         (n, 4))), detector.element_mask(event, mask)


def _designs(seed, n):
  return np.random.default_rng(seed).uniform(size=(n, 4)).astype(np.float32)


def test_the_recovered_window_is_the_sorted_design(detector, seed):
  """The mask channel factorises as `in_rows[r] * in_columns[c]`, so the design is recoverable from
    it -- to HALF A PIXEL, which is all a binary mask over pixel centres can carry."""
  half_pixel = 0.5 / detector.n_columns
  for design in _designs(seed, 32):
    features, _ = _features(detector, design, 4)
    recovered = np.asarray(recover_window(features))
    assert np.allclose(recovered, recovered[0]), 'recovery must not depend on which digit was drawn'
    x_low, x_high, y_low, y_high = (float(np.asarray(v)) for v in detector.window(design))
    if float(np.asarray(features[..., 1]).max()) == 0.0:
      continue  # an empty window carries no design at all
    assert np.abs(recovered[0] - np.array([x_low, x_high, y_low, y_high], np.float32)).max() <= half_pixel


def test_the_recovered_window_is_invariant_to_swapping_a_pair(detector, seed):
  """Each coordinate pair of `MNISTDesign` is unordered and names the same window either way, so
    nothing recoverable is lost by returning the sorted form."""
  for design in _designs(seed, 8):
    swapped = design[[1, 0, 3, 2]]
    straight, _ = _features(detector, design, 2)
    reversed_pairs, _ = _features(detector, swapped, 2)
    assert np.allclose(np.asarray(recover_window(straight)), np.asarray(recover_window(reversed_pairs)))


def test_an_empty_window_recovers_as_zeros(detector):
  """A window of zero extent shows nothing, so there is no design in the mask to read."""
  features, _ = _features(detector, np.array([0.5, 0.5, 0.5, 0.5], np.float32), 2)
  assert float(np.asarray(features).max()) == 0.0
  assert np.allclose(np.asarray(recover_window(features)), 0.0)


def test_area_pool_matches_a_strided_same_convolution(detector):
  """The image is resampled by 2x2 area averaging because that is what puts it on the block's own
    resolution: a stride-2 `SAME` convolution and a stride-2 `SAME` 2x2 window both take `n` to
    `ceil(n / 2)`."""
  features, _ = _features(detector, np.array([0.1, 0.8, 0.2, 0.9], np.float32), 2)
  pooled = area_pool(features)
  rows, columns = features.shape[-3], features.shape[-2]
  assert pooled.shape == features.shape[:-3] + (-(-rows // 2), -(-columns // 2), features.shape[-1])
  assert float(np.asarray(pooled).max()) <= 1.0 and float(np.asarray(pooled).min()) >= 0.0
  # Area, not subsampling: the mask channel's MEAN is preserved up to the padded edge cells.
  assert np.isclose(float(np.asarray(pooled[..., 1]).sum()) * 4.0, float(np.asarray(features[..., 1]).sum()), rtol=0.05)


def test_the_forward_pass_returns_logits_of_the_target_shape(detector, seed):
  model = _semi(detector)
  features, elements = _features(detector, _designs(seed, 1)[0], 16)
  out = model(features, elements)
  assert out.shape == (16, detector.target_dim())
  assert bool(jnp.all(jnp.isfinite(out)))


def test_every_residual_branch_is_inert_at_initialisation(detector, seed):
  """`alpha` is zero-initialised, exactly as in the alpha-conv stack, so at initialisation the whole
    conditioning pathway is switched OFF and the model is its stem, its strided convolutions and its
    head. Depth -- and the design injection -- therefore cost nothing in conditioning at the start.

    The two architectures are NOT the same function at initialisation even so: the conditioning
    parameters are drawn from the same rng stream, which renumbers every draw after them."""
  model = _semi(detector)
  features, elements = _features(detector, _designs(seed, 1)[0], 8)
  h = model.stem(features)
  for stage in model.stages:
    if stage.downsample is not None:
      h = stage.downsample(jax.nn.celu(h))
  skipped = model.output(jnp.mean(jax.nn.celu(h), axis=(-3, -2)))
  assert np.allclose(np.asarray(model(features, elements)), np.asarray(skipped), atol=1e-5)


def test_it_declares_no_ensemble(detector):
  assert _semi(detector).ensemble() is None


def test_the_conditioning_pathway_is_a_small_fraction_of_the_parameters(detector):
  """The 1x1 mixing convolutions and the embedding network are the ONLY parameters added."""

  def count(model):
    _, params, _ = nnx.split(model, nnx.Param, nnx.Variable)
    return sum(int(np.prod(np.shape(leaf))) for leaf in jax.tree.leaves(params))

  base, conditioned = count(_alpha(detector)), count(_semi(detector))
  widths = [int(c) for c in CHANNELS]
  image_channels = int(detector.combined_event_shape()[-1])
  mixing = BLOCKS * sum((c + EMBEDDING_CHANNELS + image_channels) * c + c for c in widths)
  embedding = 4 * EMBEDDING_FEATURES + EMBEDDING_FEATURES + EMBEDDING_FEATURES * EMBEDDING_CHANNELS + EMBEDDING_CHANNELS
  assert conditioned - base == mixing + embedding


def test_the_design_changes_the_output_through_the_embedding(detector, seed):
  """The conditioning pathway is inert at initialisation, so this trains `alpha` off zero first --
    otherwise a broken embedding would look identical to a working one."""
  model = _semi(detector)
  graphdef, params, state = nnx.split(model, nnx.Param, nnx.Variable)
  params = jax.tree.map(lambda leaf: leaf + 0.1, params)
  live = nnx.merge(graphdef, params, state)
  design = np.array([0.1, 0.4, 0.1, 0.4], np.float32)
  features, elements = _features(detector, design, 8)
  # The SAME features, with the embedding fed a different window: only the conditioning path moves.
  reference = live(features, elements)
  shifted = live.output(
    jnp.mean(jax.nn.celu(_forward_with_design(live, features, np.array([0.6, 0.9, 0.6, 0.9], np.float32))), axis=(-3, -2))
  )
  assert not np.allclose(np.asarray(reference), np.asarray(shifted))


def _forward_with_design(model, features, design_scaled):
  """The stack run on ``features`` but conditioned on an arbitrary window -- the lever the test needs
    to move the embedding without moving the image."""
  embedding = model.embedding(jnp.asarray(design_scaled, jnp.float32))
  image = features
  h = model.stem(features)
  for stage in model.stages:
    h = stage(h, embedding, image)
    if stage.downsample is not None:
      image = area_pool(image)
  return h
