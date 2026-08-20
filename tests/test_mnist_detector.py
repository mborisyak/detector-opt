"""`MNISTDetector` (the visible-window task) and the `AlphaConvRegressor` that reads its images.

The window parameterisation, the occlusion, the priced area and the shapes at every interface. The
detector reads a handful of digits from the arrow file rather than all 60000, so construction stays
cheap; the tests skip when that file is not on this machine.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from detopt.detector import MNISTDetector
from detopt.detector.mnist import N_CLASSES
from detopt.nn import from_config

DATA_PATH = '/home/max/dev/data/mnist/mnist-train.arrow'
N_EVENTS = 256

pytestmark = pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f'no MNIST arrow file at {DATA_PATH}')

# Every corner of the design cube: the extreme windows (whole image, empty, degenerate slivers). The
# flat design is `(x1, x2, y1, y2)` -- two UNORDERED pairs, not a corner and an extent.
CORNERS = np.array([[x1, x2, y1, y2] for x1 in (0.0, 1.0) for x2 in (0.0, 1.0) for y1 in (0.0, 1.0) for y2 in (0.0, 1.0)],
                   np.float32)


@pytest.fixture(scope='module')
def detector():
  return MNISTDetector(data_path=DATA_PATH, n_events=N_EVENTS)


@pytest.fixture(scope='module')
def unpriced():
  return MNISTDetector(data_path=DATA_PATH, n_events=N_EVENTS, area_weight=None)


def designs(seed, n):
  """``n`` uniform draws from the design cube, with every corner prepended."""
  return np.concatenate([CORNERS, np.random.default_rng(seed).uniform(size=(n, 4)).astype(np.float32)])


def test_the_window_lies_inside_the_image_for_every_draw(detector, seed):
  """The whole point of the two-corner parameterisation: `[0, 1]^4` maps INTO the image, so the
    design space needs no constraints and no proposal can be infeasible."""
  x0, x1, y0, y1 = (np.asarray(v) for v in detector.window(designs(seed, 256)))
  assert np.all(x0 >= 0.0) and np.all(y0 >= 0.0)
  assert np.all(x1 <= 1.0) and np.all(y1 <= 1.0)
  assert np.all(x1 >= x0) and np.all(y1 >= y0)


def test_the_mask_channel_covers_the_visible_area(detector, seed):
  """The masked pixel FRACTION is the analytic area, up to the one pixel each edge can round by."""
  drawn = designs(seed, 32)
  _, event, mask, _ = detector(drawn[:1], np.arange(1))
  for design in drawn:
    features = detector.combine(event, design, mask=mask)
    window = np.asarray(features[0, ..., 1])
    x0, x1, y0, y1 = (float(v) for v in detector.window(design))
    tolerance = (x1 - x0) / detector.n_columns + (y1 - y0) / detector.n_rows + 1.0 / (detector.n_rows * detector.n_columns)
    assert float(np.mean(window)) == pytest.approx(float(detector.visible_area(design)), abs=tolerance + 1e-6)
    assert set(np.unique(window)) <= {0.0, 1.0}


def test_the_extreme_windows_show_everything_and_nothing(detector):
  _, event, mask, _ = detector(np.zeros(4, np.float32), np.arange(4))
  everything = np.asarray(detector.combine(event, [0.0, 1.0, 0.0, 1.0], mask=mask)[..., 1])
  assert float(np.mean(everything)) == 1.0
  # A pair with equal members is a zero-width window, whichever axis it is on and wherever it sits.
  for empty in ([0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]):
    assert float(np.max(np.asarray(detector.combine(event, empty, mask=mask)[..., 1]))) == 0.0


def test_nothing_outside_the_window_reaches_the_network(detector, seed):
  """The image channel is occluded, so the network cannot see past the aperture."""
  _, event, mask, _ = detector(np.zeros(4, np.float32), np.arange(16))
  for design in designs(seed, 16):
    features = np.asarray(detector.combine(event, design, mask=mask))
    assert float(np.max(features[..., 0] * (1.0 - features[..., 1]))) == 0.0
  # ... and inside it the image is intact, not merely dimmed.
  full = np.asarray(detector.combine(event, [0.0, 1.0, 0.0, 1.0], mask=mask)[..., 0])
  assert np.allclose(full, np.asarray(event.image, np.float32) / 255.0)


def test_the_shapes_at_every_interface(detector):
  design = {'x': [0.25, 0.75], 'y': [0.25, 0.75]}
  ground_truth, event, mask, target = detector(design, np.arange(8))
  assert event.image.shape == (8, detector.n_rows, detector.n_columns)
  assert event.image.dtype == jnp.uint8
  assert mask.shape == (8, detector.n_rows) and mask.dtype == jnp.int32
  assert target.digit.shape == (8, N_CLASSES) and ground_truth.digit.shape == (8, N_CLASSES)
  assert np.allclose(np.sum(np.asarray(target.digit), axis=-1), 1.0)
  assert detector.combine(event, design, mask=mask).shape == (8, ) + detector.combined_event_shape()
  assert detector.element_mask(event, mask).shape == (8, detector.n_rows)
  assert detector.design_dim() == 4 and detector.target_dim() == N_CLASSES
  assert detector.size() == N_EVENTS


def test_the_design_round_trips_and_scaling_is_the_identity(detector, seed):
  """NOMINAL == SCALED here -- the parameterisation is already the unit box."""
  for design in designs(seed, 16):
    assert np.allclose(np.asarray(detector.to_scaled(design)), design)
    assert np.allclose(np.asarray(detector.flatten_design(detector.to_nominal(design))), design)


def test_the_uniform_prediction_scores_ln_ten(detector):
  """The loss is in NATS and undivided, which is what puts it on the same scale as the area price."""
  _, _, _, target = detector(np.zeros(4, np.float32), np.arange(32))
  predicted = jnp.zeros((32, N_CLASSES))
  assert float(jnp.mean(detector.loss(predicted, detector.normalize_target(target)))) == pytest.approx(math.log(N_CLASSES))


def test_the_metric_reports_accuracy(detector):
  _, _, _, target = detector(np.zeros(4, np.float32), np.arange(32))
  normalised = detector.normalize_target(target)
  perfect = 10.0 * normalised
  metrics = detector.metric(perfect, normalised)
  assert set(metrics) == set(detector.metric_labels())
  assert float(jnp.mean(metrics['accuracy'])) == 1.0


def test_the_price_is_the_weighted_area_and_absent_when_unconfigured(detector, unpriced, seed):
  """`None` is the ABSENCE of a price and 0.0 is a price that came out zero; they are not the same."""
  free = MNISTDetector(data_path=DATA_PATH, n_events=N_EVENTS, area_weight=0.0)
  assert detector.area_weight == pytest.approx(math.log(N_CLASSES))
  for design in designs(seed, 16):
    area = float(detector.visible_area(design))
    assert float(detector.design_penalty(design)) == pytest.approx(math.log(N_CLASSES) * area, abs=1e-6)
    assert unpriced.design_penalty(design) is None
    assert float(free.design_penalty(design)) == 0.0
  assert float(detector.design_penalty([0.0, 1.0, 0.0, 1.0])) == pytest.approx(math.log(N_CLASSES))


def test_the_reported_loss_is_the_trained_loss_plus_the_price(detector, unpriced):
  """The arithmetic `scripts/bo.py` performs: `loss = trained_loss + design_penalty`, with the term
    DROPPED rather than coerced to zero when the detector prices nothing."""
  design = [0.1, 0.6, 0.2, 0.7]
  trained_loss = 0.375

  penalty = detector.design_penalty(design)
  penalty = None if penalty is None else float(penalty)
  loss = trained_loss if penalty is None else trained_loss + penalty
  assert penalty == pytest.approx(math.log(N_CLASSES) * float(detector.visible_area(design)), abs=1e-6)
  assert loss == pytest.approx(trained_loss + penalty)

  penalty = unpriced.design_penalty(design)
  penalty = None if penalty is None else float(penalty)
  loss = trained_loss if penalty is None else trained_loss + penalty
  assert penalty is None and loss == trained_loss


def test_the_detector_holds_no_design(detector):
  """NO STATE IN THE DETECTOR: the design arrives per call, from the config."""
  assert not hasattr(detector, 'get_current_design_array')
  assert not hasattr(detector, 'design')
  assert not hasattr(detector, 'nominal_design')


def test_the_same_index_is_the_same_digit_under_every_design(detector, seed):
  """No design can move its own label -- the aperture is applied in combine, not in the event."""
  index = np.arange(16)
  _, first_event, _, first_target = detector(designs(seed, 1)[0], index)
  _, second_event, _, second_target = detector(designs(seed, 1)[-1], index)
  assert np.array_equal(np.asarray(first_event.image), np.asarray(second_event.image))
  assert np.array_equal(np.asarray(first_target.digit), np.asarray(second_target.digit))


# --------------------------------------------------------------------------- #
# The network that reads those images.
# --------------------------------------------------------------------------- #
def _regressor(detector, seed, **config):
  return from_config(detector, config={'alpha-conv-regressor': config}, rngs=nnx.Rngs(seed, params=seed + 1, dropout=seed + 2))


def test_the_regressor_maps_the_image_to_logits(detector, seed):
  regressor = _regressor(detector, seed, channels=[8, 16], blocks=1)
  assert regressor.ensemble() is None
  features = jnp.zeros((4, ) + detector.combined_event_shape())
  assert regressor(features, jnp.ones((4, detector.n_rows), jnp.int32)).shape == (4, N_CLASSES)


def test_the_regressor_refuses_to_be_ensembled(detector, seed):
  """Single network BY DESIGN: `n_models` must fail loudly, not train one net and report four."""
  with pytest.raises(TypeError):
    _regressor(detector, seed, channels=[8, 16], blocks=1, n_models=4)


def test_a_gradient_step_moves_the_parameters(detector, seed):
  import optax

  regressor = _regressor(detector, seed, channels=[8, 16], blocks=1)
  graphdef, params, state = nnx.split(regressor, nnx.Param, nnx.Variable)
  _, event, mask, target = detector(np.zeros(4, np.float32), np.arange(32))
  features = detector.combine(event, [0.1, 0.8, 0.1, 0.8], mask=mask)
  element_mask = detector.element_mask(event, mask)
  normalised = detector.normalize_target(target)

  def loss_fn(parameters):
    merged = nnx.merge(graphdef, parameters, state)
    return jnp.mean(merged.loss(detector.loss, features, element_mask, normalised, deterministic=True))

  optimizer = optax.adam(1e-2)
  opt_state = optimizer.init(params)
  before, gradients = jax.value_and_grad(loss_fn)(params)
  updates, _ = optimizer.update(gradients, opt_state, params)
  moved = optax.apply_updates(params, updates)
  assert max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(moved), jax.tree.leaves(params))) > 0.0
  assert loss_fn(moved) < before


def test_the_residual_branches_start_inert(detector, seed):
  """`alpha` is zero-initialised, so the residual part is the IDENTITY at step zero -- which is what
    makes depth cheap, and also why dropout inside a branch is a no-op until the alphas move."""
  regressor = _regressor(detector, seed, channels=[8, 16], blocks=2)
  features = jnp.asarray(np.random.default_rng(seed).uniform(size=(2, ) + detector.combined_event_shape()), jnp.float32)
  element_mask = jnp.ones((2, detector.n_rows), jnp.int32)
  quiet = regressor(features, element_mask)
  for stage in regressor.stages:
    for unit in stage.units:
      unit.alpha[...] = jnp.zeros_like(unit.alpha[...])
  assert np.allclose(np.asarray(quiet), np.asarray(regressor(features, element_mask)))
  for stage in regressor.stages:
    for unit in stage.units:
      unit.alpha[...] = jnp.ones_like(unit.alpha[...])
  assert not np.allclose(np.asarray(quiet), np.asarray(regressor(features, element_mask)))
