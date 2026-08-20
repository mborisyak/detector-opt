"""`ErasureDetector`: the log keep-probability map, the erasure and its determinism, the priced
design, and the shapes at every interface.

The detector reads a handful of digits rather than all 240000, so construction stays cheap; the tests
skip when the data is not on this machine.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from detopt.detector import ErasureDetector
from detopt.detector.mnist import N_CLASSES
from detopt.nn import from_config

EMNIST_IMAGES = '/home/max/dev/data/emnist/emnist-digits-train-images-idx3-ubyte'
EMNIST_LABELS = '/home/max/dev/data/emnist/emnist-digits-train-labels-idx1-ubyte'
MNIST_ARROW = '/home/max/dev/data/mnist/mnist-train.arrow'
N_EVENTS = 512
MIN_KEEP = 3.0e-3

pytestmark = pytest.mark.skipif(not os.path.exists(EMNIST_IMAGES), reason=f'no EMNIST IDX pair at {EMNIST_IMAGES}')


@pytest.fixture(scope='module')
def detector():
  return ErasureDetector(data_path=EMNIST_IMAGES, labels_path=EMNIST_LABELS, n_events=N_EVENTS, min_keep=MIN_KEEP)


@pytest.fixture(scope='module')
def unpriced():
  return ErasureDetector(
    data_path=EMNIST_IMAGES, labels_path=EMNIST_LABELS, n_events=N_EVENTS, min_keep=MIN_KEEP, keep_weight=None
  )


def scaled_draws(seed, n):
  """``n`` uniform draws from the SCALED cube, with both ends kept."""
  return np.concatenate([np.array([0.0, 1.0]), np.random.default_rng(seed).uniform(size=n)]).astype(np.float32)


# --------------------------------------------------------------------------- #
# The design map
# --------------------------------------------------------------------------- #
def test_the_scaled_cube_maps_onto_the_keep_range(detector, seed):
  nominal = np.asarray([
    float(jnp.reshape(detector._to_nominal_flat(jnp.array([u])), (-1, ))[0]) for u in scaled_draws(seed, 64)
  ])
  assert np.all(nominal >= detector.min_keep * (1.0 - 1e-5)) and np.all(nominal <= 1.0)
  assert nominal[0] == pytest.approx(detector.min_keep, rel=1e-5)
  assert nominal[1] == 1.0


def test_the_design_round_trips_through_both_spaces(detector, seed):
  for u in scaled_draws(seed, 32):
    nominal = detector.to_nominal(np.array([u], np.float32))
    assert float(jnp.reshape(detector.to_scaled(nominal), (-1, ))[0]) == pytest.approx(float(u), abs=1e-5)


def test_the_map_is_log_uniform_not_linear(detector):
  middle = float(jnp.reshape(detector._to_nominal_flat(jnp.array([0.5])), (-1, ))[0])
  assert middle == pytest.approx(math.sqrt(detector.min_keep), rel=1e-5)


# --------------------------------------------------------------------------- #
# The erasure
# --------------------------------------------------------------------------- #
def test_the_same_design_and_index_reproduce_the_frame(detector):
  """DETERMINISM: the framework re-reads indices and expects the same event back."""
  first = detector({'keep': 0.3}, np.arange(16))[1].pixels
  second = detector({'keep': 0.3}, np.arange(16))[1].pixels
  assert np.array_equal(np.asarray(first), np.asarray(second))


def test_two_keep_probabilities_erase_independently_on_the_same_digit(detector):
  """The design's own bit pattern folds into the key, so changing it re-rolls the erasure -- but the
    LABEL is a property of the index alone, so no design can move its own label."""
  _, sparse, _, sparse_target = detector({'keep': 0.05}, np.arange(16))
  _, dense, _, dense_target = detector({'keep': 0.9}, np.arange(16))
  assert not np.array_equal(np.asarray(sparse.pixels), np.asarray(dense.pixels))
  assert np.array_equal(np.asarray(sparse_target.digit), np.asarray(dense_target.digit))
  nudged = detector({'keep': 0.9000001}, np.arange(16))[1].pixels
  assert not np.array_equal(np.asarray(dense.pixels), np.asarray(nudged))


def test_survivors_keep_their_exact_intensity_and_the_rest_are_zero(detector):
  """A surviving pixel shows its true intensity exactly; an erased one is 0. Nothing is rescaled."""
  index = np.arange(N_EVENTS)
  clean = detector.images[index % detector.size()]
  pixels = np.asarray(detector({'keep': 0.5}, index)[1].pixels)
  survived = pixels > 0
  assert np.array_equal(pixels[survived], clean[survived])
  assert np.all(pixels[~survived] == 0)


def test_the_surviving_share_matches_the_keep_probability(detector):
  """The erasure is per-pixel and independent, so the surviving share of the ink IS ``p``."""
  index = np.arange(N_EVENTS)
  clean = detector.images[index % detector.size()]
  lit = float((clean > 0).sum())
  for keep in (0.05, 0.2, 0.5):
    survivors = float((np.asarray(detector({'keep': keep}, index)[1].pixels) > 0).sum())
    assert survivors / lit == pytest.approx(keep, rel=0.1)


def test_less_keep_leaves_less_ink(detector):
  index = np.arange(N_EVENTS)
  counts = [float((np.asarray(detector({'keep': k}, index)[1].pixels) > 0).sum()) for k in (0.05, 0.2, 0.5, 1.0)]
  assert counts[0] < counts[1] < counts[2] < counts[3]


# --------------------------------------------------------------------------- #
# Shapes and combine
# --------------------------------------------------------------------------- #
def test_the_shapes_at_every_interface(detector):
  design = {'keep': 0.3}
  ground_truth, event, mask, target = detector(design, np.arange(8))
  assert event.pixels.shape == (8, detector.n_rows, detector.n_columns)
  assert event.pixels.dtype == jnp.uint8
  assert mask.shape == (8, detector.n_rows) and mask.dtype == jnp.int32
  assert target.digit.shape == (8, N_CLASSES) and ground_truth.digit.shape == (8, N_CLASSES)
  assert np.allclose(np.sum(np.asarray(target.digit), axis=-1), 1.0)
  assert detector.combine(event, design, mask=mask).shape == (8, ) + detector.combined_event_shape()
  assert detector.element_mask(event, mask).shape == (8, detector.n_rows)
  assert detector.design_dim() == 1 and detector.target_dim() == N_CLASSES
  assert detector.size() == N_EVENTS


def test_both_channels_are_bounded_and_carry_what_they_should(detector, seed):
  """Channel 0 is the surviving frame on [0, 1] and channel 1 the scaled keep probability. Both are
    bounded at every design, which is why nothing is clipped or stabilised anywhere."""
  for u in scaled_draws(seed, 8):
    design = detector.to_nominal(np.array([u], np.float32))
    _, event, mask, _ = detector(design, np.arange(64))
    features = np.asarray(detector.combine(event, design, mask=mask))
    assert features.shape == (64, detector.n_rows, detector.n_columns, 2)
    assert features[..., 0].min() >= 0.0 and features[..., 0].max() <= 1.0
    assert np.allclose(features[..., 0], np.asarray(event.pixels, np.float32) / 255.0)
    assert np.allclose(features[..., 1], float(u), atol=1e-5)


# --------------------------------------------------------------------------- #
# The price
# --------------------------------------------------------------------------- #
def test_the_price_is_the_weighted_keep_and_absent_when_unconfigured(detector, unpriced, seed):
  """`None` is the ABSENCE of a price and 0.0 is a price that came out zero; they are not the same."""
  free = ErasureDetector(
    data_path=EMNIST_IMAGES, labels_path=EMNIST_LABELS, n_events=N_EVENTS, min_keep=MIN_KEEP, keep_weight=0.0
  )
  assert detector.keep_weight == pytest.approx(math.log(N_CLASSES))
  for u in scaled_draws(seed, 8):
    design = detector.to_nominal(np.array([u], np.float32))
    keep = float(jnp.reshape(detector.flatten_design(design), (-1, ))[0])
    assert float(detector.design_penalty(design)) == pytest.approx(math.log(N_CLASSES) * keep, rel=1e-5)
    assert unpriced.design_penalty(design) is None
    assert float(free.design_penalty(design)) == 0.0


def test_a_complete_frame_costs_exactly_the_no_information_loss(detector):
  """The pinning that makes ln(10) the reference line on the reported total."""
  assert float(detector.design_penalty([1.0])) == pytest.approx(math.log(N_CLASSES), rel=1e-6)
  _, _, _, target = detector({'keep': 0.5}, np.arange(32))
  uniform = jnp.zeros((32, N_CLASSES))
  assert float(jnp.mean(detector.loss(uniform, detector.normalize_target(target)))) == pytest.approx(math.log(N_CLASSES))


def test_the_reported_loss_is_the_trained_loss_plus_the_price(detector, unpriced):
  """The arithmetic `scripts/bo.py` performs, with the term DROPPED rather than coerced to zero when
    the detector prices nothing."""
  design = [0.25]
  trained_loss = 0.375
  penalty = detector.design_penalty(design)
  penalty = None if penalty is None else float(penalty)
  assert penalty == pytest.approx(math.log(N_CLASSES) * 0.25, rel=1e-5)
  assert (trained_loss if penalty is None else trained_loss + penalty) == pytest.approx(trained_loss + penalty)
  penalty = unpriced.design_penalty(design)
  penalty = None if penalty is None else float(penalty)
  assert penalty is None
  assert (trained_loss if penalty is None else trained_loss + penalty) == trained_loss


# --------------------------------------------------------------------------- #
# Contract and reuse
# --------------------------------------------------------------------------- #
def test_the_detector_holds_no_design(detector):
  """NO STATE IN THE DETECTOR: the design arrives per call, from the config."""
  assert not hasattr(detector, 'get_current_design_array')
  assert not hasattr(detector, 'design')
  assert not hasattr(detector, 'nominal_design')


def test_the_loaders_are_shared_with_the_window_detector():
  """Guard against a decode being copied back in: both detectors must call the SAME loaders."""
  from detopt.detector import erasure, mnist

  assert erasure.read_arrow is mnist.read_arrow
  assert erasure.read_idx is mnist.read_idx


def test_the_frames_arrive_upright(detector):
  """`read_idx` transposes EMNIST's column-major frames and this detector must not do it again. The
    check is on the SHAPE of the ink over many frames, not on any single one."""
  ink = detector.images.astype(np.float32) / 255.0
  rows_used = float(np.mean(np.sum(ink.sum(axis=2) > 0.05, axis=1)))
  columns_used = float(np.mean(np.sum(ink.sum(axis=1) > 0.05, axis=1)))
  assert rows_used > columns_used


@pytest.mark.skipif(not os.path.exists(MNIST_ARROW), reason=f'no MNIST arrow file at {MNIST_ARROW}')
def test_the_arrow_layout_loads_when_no_labels_path_is_given():
  detector = ErasureDetector(data_path=MNIST_ARROW, n_events=256, min_keep=MIN_KEEP)
  assert detector.size() == 256 and (detector.n_rows, detector.n_columns) == (28, 28)


def test_a_bad_configuration_raises_rather_than_being_silently_clamped():
  for bad in ({'min_keep': 0.0}, {'min_keep': 1.0}, {'n_events': 0}):
    arguments = {'data_path': EMNIST_IMAGES, 'labels_path': EMNIST_LABELS, 'n_events': N_EVENTS}
    arguments.update(bad)
    with pytest.raises(ValueError):
      ErasureDetector(**arguments)


# --------------------------------------------------------------------------- #
# The network that reads those frames
# --------------------------------------------------------------------------- #
def test_the_regressor_maps_the_frame_to_logits(detector, seed):
  regressor = from_config(
    detector, config={'alpha-conv-regressor': {
      'channels': [16, 24, 32],
      'blocks': 2
    }}, rngs=nnx.Rngs(seed, params=seed + 1, dropout=seed + 2)
  )
  assert regressor.ensemble() is None
  features = jnp.zeros((4, ) + detector.combined_event_shape())
  assert regressor(features, jnp.ones((4, detector.n_rows), jnp.int32)).shape == (4, N_CLASSES)
  _, params, _ = nnx.split(regressor, nnx.Param, nnx.Variable)
  assert sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params)) == 44754


def test_a_gradient_step_moves_the_parameters(detector, seed):
  import optax

  regressor = from_config(
    detector, config={'alpha-conv-regressor': {
      'channels': [8, 16],
      'blocks': 1
    }}, rngs=nnx.Rngs(seed, params=seed + 1, dropout=seed + 2)
  )
  graphdef, params, state = nnx.split(regressor, nnx.Param, nnx.Variable)
  design = {'keep': 0.8}
  _, event, mask, target = detector(design, np.arange(32))
  features = detector.combine(event, design, mask=mask)
  element_mask = detector.element_mask(event, mask)
  normalised = detector.normalize_target(target)

  def loss_fn(parameters):
    merged = nnx.merge(graphdef, parameters, state)
    return jnp.mean(merged.loss(detector.loss, features, element_mask, normalised, deterministic=True))

  optimizer = optax.adam(1e-2)
  before, gradients = jax.value_and_grad(loss_fn)(params)
  updates, _ = optimizer.update(gradients, optimizer.init(params), params)
  assert loss_fn(optax.apply_updates(params, updates)) < before
