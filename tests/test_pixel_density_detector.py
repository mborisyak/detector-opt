"""`PixelDensityDetector`: the density warp and that SIGMA BITES, the bicubic against a known-smooth
field, what happens outside the frame, the ABSENT price, and the shapes at every interface.

The detector reads a handful of frames rather than all 240000, so construction stays cheap; the tests
skip when the data is not on this machine.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from detopt.detector import PixelDensityDetector
from detopt.detector.mnist import N_CLASSES, MNISTEvent
from detopt.detector.pixel_density import keys_kernel
from detopt.nn import from_config

EMNIST_IMAGES = '/home/max/dev/data/emnist/emnist-digits-train-images-idx3-ubyte'
EMNIST_LABELS = '/home/max/dev/data/emnist/emnist-digits-train-labels-idx1-ubyte'
MNIST_ARROW = '/home/max/dev/data/mnist/mnist-train.arrow'
N_EVENTS = 512
SIGMA_MIN, SIGMA_MAX = 0.2, 5.0

pytestmark = pytest.mark.skipif(not os.path.exists(EMNIST_IMAGES), reason=f'no EMNIST IDX pair at {EMNIST_IMAGES}')


@pytest.fixture(scope='module')
def detector():
  return PixelDensityDetector(
    data_path=EMNIST_IMAGES, labels_path=EMNIST_LABELS, n_events=N_EVENTS, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX
  )


def scaled_draws(seed, n):
  """``n`` uniform draws from the SCALED cube, with both corners kept."""
  return np.concatenate([np.zeros((1, 2)), np.ones((1, 2)),
                         np.random.default_rng(seed).uniform(size=(n, 2))]).astype(np.float32)


def polynomial_frame(detector, coefficients):
  """A frame whose pixel values are a polynomial of degree <= 2 PER AXIS in the normalised
  coordinates -- exactly the family Keys' cubic convolution reproduces without error. Values are
  multiplied by 255 because ``combine_scaled`` divides by it."""
  x = (np.arange(detector.n_columns) + 0.5) / detector.n_columns
  y = (np.arange(detector.n_rows) + 0.5) / detector.n_rows
  field = polynomial(coefficients, x[None, :], y[:, None])
  return MNISTEvent(image=jnp.asarray(field * 255.0, jnp.float32))


def polynomial(coefficients, x, y):
  return sum(c * x**i * y**j for (i, j), c in coefficients.items())


def unit_frame(positions):
  """Probe positions from the detector's symmetric ``[-1, 1]`` frame onto ``[0, 1]``, where the
  pixel-centre arithmetic of these tests is written."""
  return 0.5 * (np.asarray(positions, np.float64) + 1.0)


COEFFICIENTS = {(0, 0): 0.21,
                (1, 0): 0.33,
                (2, 0): -0.27,
                (0, 1): -0.19,
                (1, 1): 0.41,
                (2, 1): 0.15,
                (0, 2): 0.24,
                (1, 2): -0.31,
                (2, 2): 0.18}


# --------------------------------------------------------------------------- #
# The design map
# --------------------------------------------------------------------------- #
def test_the_scaled_cube_maps_onto_the_sigma_range(detector, seed):
  nominal = np.asarray([np.reshape(detector._to_nominal_flat(u), (-1, )) for u in scaled_draws(seed, 32)])
  assert np.all(nominal >= SIGMA_MIN * (1.0 - 1e-5)) and np.all(nominal <= SIGMA_MAX * (1.0 + 1e-5))
  assert nominal[0] == pytest.approx([SIGMA_MIN, SIGMA_MIN], rel=1e-5)
  assert nominal[1] == pytest.approx([SIGMA_MAX, SIGMA_MAX], rel=1e-5)


def test_the_design_round_trips_through_both_spaces(detector, seed):
  for u in scaled_draws(seed, 16):
    nominal = detector.to_nominal(u)
    assert np.reshape(detector.to_scaled(nominal), (-1, )) == pytest.approx(u, abs=1e-5)


def test_the_map_is_log_uniform_not_linear(detector):
  """A factor is the natural step in a concentration, so the middle of the cube is the GEOMETRIC
    mean of the range -- which for this range is exactly the uniform grid, sigma = 1."""
  middle = np.reshape(detector._to_nominal_flat(jnp.array([0.5, 0.5])), (-1, ))
  assert middle == pytest.approx([math.sqrt(SIGMA_MIN * SIGMA_MAX)] * 2, rel=1e-5)


# --------------------------------------------------------------------------- #
# The density warp
# --------------------------------------------------------------------------- #
def test_the_probe_density_is_the_gaussian(detector):
  """THE DEFINING PROPERTY: equal probability mass of ``N(0, sigma^2)`` between consecutive probes,
    so the spacing goes as ``1 / pdf``. Checked against the density directly -- the mass under the
    Gaussian between neighbours is the same for every pair, to the truncation's own normalisation."""
  from scipy.special import erf

  for sigma in (0.3, 0.75, 2.0):
    x = np.asarray(detector.sample_positions({'sigma_x': sigma, 'sigma_y': sigma})[0], np.float64)
    mass = np.diff(0.5 * (1.0 + erf(x / (sigma * math.sqrt(2.0)))))
    assert mass == pytest.approx(np.full(detector.n_grid - 1, mass.mean()), rel=1e-4)


def test_the_corner_probes_are_pinned_to_the_frame_at_every_design(detector, seed):
  """THE FIELD OF VIEW IS FIXED. The outermost probe of each axis sits exactly on the image's edge
    for every design, so no probe is ever spent outside the picture and `sigma` moves only what lies
    between them."""
  for u in scaled_draws(seed, 32):
    for positions in detector.sample_positions(detector.to_nominal(u)):
      positions = np.asarray(positions)
      assert positions[0] == -1.0 and positions[-1] == 1.0
      assert np.all(positions >= -1.0) and np.all(positions <= 1.0)
      assert np.all(np.isfinite(positions))
      assert np.all(np.diff(positions) > 0.0)


def test_the_spacing_grows_from_the_centre_outwards_at_every_design(detector, seed):
  """A Gaussian density can only thin outwards: the gap between neighbours must increase with
    distance from the centre, at EVERY sigma. Nothing may ever cluster at the edge -- that was the
    signature of two earlier, wrong constructions."""
  for u in scaled_draws(seed, 16):
    x = np.asarray(detector.sample_positions(detector.to_nominal(u))[0], np.float64)
    spacing = np.diff(x)
    half = detector.n_grid // 2
    assert np.all(np.diff(spacing[half - 1:]) >= -1e-6)
    assert spacing[0] >= spacing[half - 1] - 1e-6


def test_small_sigma_collapses_the_interior_onto_the_centre(detector):
  """The small end: a magnified crop of the middle, with the corners still pinned. The digit's
    extremities fall outside the sampled region entirely."""
  x = np.asarray(detector.sample_positions({'sigma_x': detector.sigma_min, 'sigma_y': 1.0})[0], np.float64)
  assert np.all(np.abs(x[1:-1]) < 0.55)
  assert x[0] == -1.0 and x[-1] == 1.0
  tighter = np.asarray(detector.sample_positions({'sigma_x': 0.05, 'sigma_y': 1.0})[0], np.float64)
  assert np.max(np.abs(tighter[1:-1])) < np.max(np.abs(x[1:-1]))


def test_large_sigma_tends_to_the_uniform_grid(detector):
  """The large end: the Gaussian is flat across the frame, so the probes spread evenly. This is the
    limit, and it is a legitimate design rather than a failure -- the range's asymmetry is a
    property of the parameterisation and is reported as such."""
  x = np.asarray(detector.sample_positions({'sigma_x': 50.0, 'sigma_y': 1.0})[0], np.float64)
  assert x == pytest.approx(np.linspace(-1.0, 1.0, detector.n_grid), abs=1e-3)
  at_top = np.asarray(detector.sample_positions({'sigma_x': detector.sigma_max, 'sigma_y': 1.0})[0], np.float64)
  spacing = np.diff(at_top)
  assert spacing.min() / spacing.max() > 0.9


def test_the_centre_to_edge_spacing_ratio_is_monotone_in_sigma(detector):
  """The one number that orders the whole design space, and the guard against a parameterisation
    that saturates: it must move over the WHOLE range, with no two designs sharing a grid."""
  ratios = []
  for sigma in (0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0):
    spacing = np.diff(np.asarray(detector.sample_positions({'sigma_x': sigma, 'sigma_y': 1.0})[0], np.float64))
    ratios.append(spacing[detector.n_grid // 2 - 1] / spacing[0])
  assert np.all(np.diff(ratios) > 0.01)
  assert ratios[0] < 0.1 and ratios[-1] > 0.95


def test_the_grid_is_symmetric_about_the_centre(detector, seed):
  for u in scaled_draws(seed, 8):
    x = np.asarray(detector.sample_positions(detector.to_nominal(u))[0])
    assert x + x[::-1] == pytest.approx(np.zeros_like(x), abs=1e-6)


def test_sigma_bites_on_the_grid_and_on_the_combined_event(detector):
  """⚠️ THE CANCELLATION TRAP. Renormalising a fixed set of warped quantiles by their own extremes
    divides `sigma` back out and leaves a silently design-blind detector. Two sigmas must give two
    different grids AND two different read-outs, on each axis independently -- including at the small
    end, where the earlier constructions saturated."""
  uniform = {'sigma_x': 1.0, 'sigma_y': 1.0}
  _, event, mask, _ = detector(uniform, np.arange(8))
  reference = np.asarray(detector.combine(event, uniform, mask=mask)[..., 0])
  reference_grid = [np.asarray(a) for a in detector.sample_positions(uniform)]
  for design in ({'sigma_x': 0.4, 'sigma_y': 1.0}, {'sigma_x': 1.0, 'sigma_y': 0.4}, {'sigma_x': 3.0, 'sigma_y':
                                                                                      3.0}, {'sigma_x': 0.2, 'sigma_y': 0.25}):
    grid = [np.asarray(a) for a in detector.sample_positions(design)]
    assert not all(np.allclose(a, b) for a, b in zip(grid, reference_grid))
    other = np.asarray(detector.combine(event, design, mask=mask)[..., 0])
    assert np.max(np.abs(other - reference)) > 0.05
  close = [np.asarray(a) for a in detector.sample_positions({'sigma_x': 0.21, 'sigma_y': 1.0})]
  far = [np.asarray(a) for a in detector.sample_positions({'sigma_x': 0.2, 'sigma_y': 1.0})]
  assert not np.allclose(close[0], far[0])


def test_the_two_axes_are_separate_coordinates(detector):
  """``sigma_x`` moves the COLUMN positions and leaves the rows alone, and conversely."""
  x0, y0 = [np.asarray(a) for a in detector.sample_positions({'sigma_x': 1.0, 'sigma_y': 1.0})]
  x1, y1 = [np.asarray(a) for a in detector.sample_positions({'sigma_x': 0.5, 'sigma_y': 1.0})]
  assert not np.allclose(x0, x1)
  assert np.allclose(y0, y1)


# --------------------------------------------------------------------------- #
# The interpolant
# --------------------------------------------------------------------------- #
def test_the_kernel_is_a_partition_of_unity_with_four_taps(detector):
  """Keys' cubic convolution: support ``|s| < 2`` and the four taps around any point summing to 1,
    which is what makes a constant frame come back as that constant in the interior."""
  assert float(keys_kernel(jnp.array(0.0))) == pytest.approx(1.0)
  assert float(keys_kernel(jnp.array(1.0))) == pytest.approx(0.0, abs=1e-6)
  assert float(keys_kernel(jnp.array(2.0))) == pytest.approx(0.0, abs=1e-6)
  assert float(keys_kernel(jnp.array(2.5))) == 0.0
  for offset in np.linspace(0.0, 1.0, 11):
    taps = np.asarray(keys_kernel(jnp.asarray(offset - np.array([-1.0, 0.0, 1.0, 2.0]))))
    assert taps.sum() == pytest.approx(1.0, abs=1e-6)


def test_the_bicubic_reproduces_a_known_smooth_field_exactly(detector):
  """Cubic convolution at ``a = -1/2`` is EXACT for polynomials of degree <= 2 per axis, so on such a
    field the read-out must equal the field at the sample positions -- to float precision, in the
    interior where the 4x4 stencil does not reach past the border."""
  event = polynomial_frame(detector, COEFFICIENTS)
  for design in ({'sigma_x': 1.0, 'sigma_y': 1.0}, {'sigma_x': 0.7, 'sigma_y': 1.3}, {'sigma_x': 0.5, 'sigma_y': 0.5}):
    samples = np.asarray(detector.combine(event, design)[..., 0])
    x, y = [unit_frame(a) for a in detector.sample_positions(design)]
    truth = polynomial(COEFFICIENTS, x[None, :], y[:, None])
    interior = ((x * detector.n_columns > 1.5) & (x * detector.n_columns < detector.n_columns - 1.5))[None, :] \
        & ((y * detector.n_rows > 1.5) & (y * detector.n_rows < detector.n_rows - 1.5))[:, None]
    assert np.max(np.abs(samples - truth)[interior]) < 1e-5


def test_the_bicubic_beats_linear_interpolation_on_a_smooth_field(detector):
  """It is BI-CUBIC and not bilinear: on a smooth non-polynomial field the error must be far below
    what `map_coordinates(order=1)` -- the only interpolation jax ships -- reaches at the same points."""
  from jax.scipy.ndimage import map_coordinates

  x = (np.arange(detector.n_columns) + 0.5) / detector.n_columns
  y = (np.arange(detector.n_rows) + 0.5) / detector.n_rows

  def field(x, y):
    return 0.5 + 0.4 * np.sin(3.1 * x + 0.7) * np.cos(2.6 * y - 0.3)

  event = MNISTEvent(image=jnp.asarray(field(x[None, :], y[:, None]) * 255.0, jnp.float32))
  design = {'sigma_x': 0.9, 'sigma_y': 1.1}
  samples = np.asarray(detector.combine(event, design)[..., 0])
  grid_x, grid_y = [unit_frame(a) for a in detector.sample_positions(design)]
  truth = field(grid_x[None, :], grid_y[:, None])
  rows, columns = np.meshgrid(grid_y * detector.n_rows - 0.5, grid_x * detector.n_columns - 0.5, indexing='ij')
  linear = np.asarray(
    map_coordinates(
      jnp.asarray(field(x[None, :], y[:, None])), [jnp.asarray(rows), jnp.asarray(columns)], order=1, mode='constant'
    )
  )
  interior = (columns > 1.5) & (columns < detector.n_columns - 1.5) & (rows > 1.5) & (rows < detector.n_rows - 1.5)
  assert np.max(np.abs(samples - truth)[interior]) < 0.1 * np.max(np.abs(linear - truth)[interior])


def test_outside_the_frame_is_zero_padding_and_never_a_nan(detector):
  """THE BORDER CONTRACT, pinned on the constant frame where the answer is arithmetic, and now at
    EVERY design because the corner probes are pinned to the frame's edges. A probe at ``x = 0`` sits
    at pixel coordinate -0.5, so its four taps carry weights (-1/16, 9/16, 9/16, -1/16); the two
    outside the lattice contribute 0 and the axis returns exactly 1/2. A corner, the product of two
    such axes, returns exactly 1/4. Nothing is clamped and nothing is a NaN."""
  ones = MNISTEvent(image=jnp.full((detector.n_rows, detector.n_columns), 255.0, jnp.float32))
  middle = detector.n_grid // 2
  for sigma in (SIGMA_MIN, 0.5, 1.0, 2.0, SIGMA_MAX):
    samples = np.asarray(detector.combine(ones, {'sigma_x': sigma, 'sigma_y': sigma})[..., 0])
    assert np.all(np.isfinite(samples))
    assert samples[0, 0] == pytest.approx(0.25, abs=1e-4)
    assert samples[0, -1] == pytest.approx(0.25, abs=1e-4)
    assert samples[-1, -1] == pytest.approx(0.25, abs=1e-4)
    assert samples[0, middle] == pytest.approx(0.5, abs=1e-4)
    assert samples[middle, 0] == pytest.approx(0.5, abs=1e-4)
  assert np.asarray(detector.combine(ones, {'sigma_x': 1.0, 'sigma_y': 1.0})[middle, middle, 0]) \
      == pytest.approx(1.0, abs=1e-4)


def test_no_design_produces_a_nan_or_an_unbounded_reading(detector, seed):
  """Cubic convolution RINGS, so the read-out may leave [0, 1] by a little near a sharp stroke; what
    it may never do is diverge or produce a NaN."""
  _, event, mask, _ = detector({'sigma_x': 1.0, 'sigma_y': 1.0}, np.arange(64))
  for u in scaled_draws(seed, 16):
    samples = np.asarray(detector.combine_scaled(event, jnp.asarray(u), mask=mask)[..., 0])
    assert np.all(np.isfinite(samples))
    assert samples.min() > -0.5 and samples.max() < 1.5


# --------------------------------------------------------------------------- #
# Shapes, channels and determinism
# --------------------------------------------------------------------------- #
def test_the_shapes_at_every_interface(detector):
  design = {'sigma_x': 0.8, 'sigma_y': 1.2}
  ground_truth, event, mask, target = detector(design, np.arange(8))
  assert event.image.shape == (8, detector.n_rows, detector.n_columns)
  assert event.image.dtype == jnp.uint8
  assert mask.shape == (8, detector.n_rows) and mask.dtype == jnp.int32
  assert target.digit.shape == (8, N_CLASSES) and ground_truth.digit.shape == (8, N_CLASSES)
  assert np.allclose(np.sum(np.asarray(target.digit), axis=-1), 1.0)
  assert detector.combined_event_shape() == (detector.n_grid, detector.n_grid, 3)
  assert detector.combine(event, design, mask=mask).shape == (8, ) + detector.combined_event_shape()
  assert detector.element_mask(event, mask).shape == (8, detector.n_grid)
  assert detector.design_dim() == 2 and detector.target_dim() == N_CLASSES
  assert detector.size() == N_EVENTS


def test_the_design_planes_carry_the_scaled_design(detector, seed):
  """Channels 1 and 2 are how the design reaches the network: the grid's spacing is not uniform and
    nothing else names the warp."""
  _, event, mask, _ = detector({'sigma_x': 1.0, 'sigma_y': 1.0}, np.arange(16))
  for u in scaled_draws(seed, 8):
    features = np.asarray(detector.combine_scaled(event, jnp.asarray(u), mask=mask))
    assert features.shape == (16, detector.n_grid, detector.n_grid, 3)
    assert np.allclose(features[..., 1], u[0], atol=1e-6)
    assert np.allclose(features[..., 2], u[1], atol=1e-6)


def test_the_event_does_not_depend_on_the_design(detector):
  """The sampling lives in `combine_scaled`, so one pool of frames serves every design and no design
    can move its own label."""
  first = detector({'sigma_x': 0.3, 'sigma_y': 3.0}, np.arange(16))
  second = detector({'sigma_x': 2.0, 'sigma_y': 0.5}, np.arange(16))
  assert np.array_equal(np.asarray(first[1].image), np.asarray(second[1].image))
  assert np.array_equal(np.asarray(first[3].digit), np.asarray(second[3].digit))


def test_the_same_design_and_index_reproduce_the_read_out(detector):
  """DETERMINISM: the framework re-reads indices and expects the same numbers back."""
  design = {'sigma_x': 0.9, 'sigma_y': 1.1}
  _, event, mask, _ = detector(design, np.arange(16))
  first = np.asarray(detector.combine(event, design, mask=mask))
  _, again, mask_again, _ = detector(design, np.arange(16))
  assert np.array_equal(first, np.asarray(detector.combine(again, design, mask=mask_again)))


def test_a_batched_design_reads_each_event_at_its_own_grid(detector):
  """One design per event, which is how the trainer's pools store them."""
  _, event, mask, _ = detector({'sigma_x': 1.0, 'sigma_y': 1.0}, np.arange(4))
  batched = jnp.asarray([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5], [0.5, 0.5]], jnp.float32)
  features = np.asarray(detector.combine_scaled(event, batched, mask=mask))
  for index in range(4):
    single = np.asarray(detector.combine_scaled(MNISTEvent(image=event.image[index]), batched[index]))
    assert np.allclose(features[index], single, atol=1e-6)


# --------------------------------------------------------------------------- #
# The (absent) price
# --------------------------------------------------------------------------- #
def test_the_task_prices_nothing_at_all(detector, seed):
  """`None` is the ABSENCE of a price, not a price of 0.0, and a caller must drop the term entirely.
    The sample budget is fixed at n_grid^2, so the trade-off is already intrinsic."""
  for u in scaled_draws(seed, 8):
    assert detector.design_penalty(detector.to_nominal(u)) is None
  trained_loss = 0.375
  penalty = detector.design_penalty({'sigma_x': 1.0, 'sigma_y': 1.0})
  assert (trained_loss if penalty is None else trained_loss + float(penalty)) == trained_loss


def test_the_uniform_prediction_scores_the_no_information_level(detector):
  _, _, _, target = detector({'sigma_x': 1.0, 'sigma_y': 1.0}, np.arange(32))
  uniform = jnp.zeros((32, N_CLASSES))
  assert float(jnp.mean(detector.loss(uniform, detector.normalize_target(target)))) == pytest.approx(math.log(N_CLASSES))


# --------------------------------------------------------------------------- #
# Contract and reuse
# --------------------------------------------------------------------------- #
def test_the_detector_holds_no_design(detector):
  """NO STATE IN THE DETECTOR: the design arrives per call, from the config."""
  assert not hasattr(detector, 'get_current_design_array')
  assert not hasattr(detector, 'design')
  assert not hasattr(detector, 'nominal_design')


def test_the_loaders_and_records_are_shared_with_the_window_detector():
  """Guard against a decode being copied back in: the frame source must be the SAME loaders and the
    same records the other digit detectors use."""
  from detopt.detector import mnist, pixel_density

  assert pixel_density.read_arrow is mnist.read_arrow
  assert pixel_density.read_idx is mnist.read_idx
  assert pixel_density.MNISTEvent is mnist.MNISTEvent
  assert pixel_density.MNISTTarget is mnist.MNISTTarget


def test_the_frames_arrive_upright(detector):
  """`read_idx` transposes EMNIST's column-major frames and this detector must not do it again. The
    check is on the SHAPE of the ink over many frames, not on any single one."""
  ink = detector.images.astype(np.float32) / 255.0
  rows_used = float(np.mean(np.sum(ink.sum(axis=2) > 0.05, axis=1)))
  columns_used = float(np.mean(np.sum(ink.sum(axis=1) > 0.05, axis=1)))
  assert rows_used > columns_used


@pytest.mark.skipif(not os.path.exists(MNIST_ARROW), reason=f'no MNIST arrow file at {MNIST_ARROW}')
def test_the_arrow_layout_loads_when_no_labels_path_is_given():
  detector = PixelDensityDetector(data_path=MNIST_ARROW, n_events=256)
  assert detector.size() == 256 and (detector.n_rows, detector.n_columns) == (28, 28)


def test_a_bad_configuration_raises_rather_than_being_silently_clamped():
  for bad in ({'n_grid': 1}, {'sigma_min': 0.0}, {'sigma_min': 6.0}, {'n_events': 0}):
    arguments = {'data_path': EMNIST_IMAGES, 'labels_path': EMNIST_LABELS, 'n_events': N_EVENTS}
    arguments.update(bad)
    with pytest.raises(ValueError):
      PixelDensityDetector(**arguments)


# --------------------------------------------------------------------------- #
# The network that reads the grid
# --------------------------------------------------------------------------- #
def test_the_regressor_maps_the_read_out_to_logits(detector, seed):
  regressor = from_config(
    detector, config={'alpha-conv-regressor': {
      'channels': [16, 24, 32],
      'blocks': 2
    }}, rngs=nnx.Rngs(seed, params=seed + 1, dropout=seed + 2)
  )
  assert regressor.ensemble() is None
  features = jnp.zeros((4, ) + detector.combined_event_shape())
  assert regressor(features, jnp.ones((4, detector.n_grid), jnp.int32)).shape == (4, N_CLASSES)
  _, params, _ = nnx.split(regressor, nnx.Param, nnx.Variable)
  assert sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params)) == 44898


def test_the_combine_is_differentiable_in_the_scaled_design(detector):
  """The contract asks combine_scaled to be differentiable w.r.t. the SCALED design, and here that is
    load-bearing rather than formal: the sampling itself is the design, so a zero gradient would mean
    the read-out does not move."""
  _, event, mask, _ = detector({'sigma_x': 1.0, 'sigma_y': 1.0}, np.arange(8))

  def readout(design_scaled):
    return jnp.sum(jnp.square(detector.combine_scaled(event, design_scaled, mask=mask)[..., 0]))

  gradient = np.asarray(jax.grad(readout)(jnp.array([0.5, 0.5], jnp.float32)))
  assert np.all(np.isfinite(gradient))
  assert np.all(np.abs(gradient) > 1e-3)


def test_a_gradient_step_moves_the_parameters(detector, seed):
  import optax

  regressor = from_config(
    detector, config={'alpha-conv-regressor': {
      'channels': [8, 16],
      'blocks': 1
    }}, rngs=nnx.Rngs(seed, params=seed + 1, dropout=seed + 2)
  )
  graphdef, params, state = nnx.split(regressor, nnx.Param, nnx.Variable)
  design = {'sigma_x': 0.8, 'sigma_y': 1.0}
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
