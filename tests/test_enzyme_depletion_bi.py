"""`EnzymeDepletionBiDetector`: the contract surface, the RKC2 integration against the exact
solution, the read-out, the design SET symmetry, the A/B degeneracy and design-blindness.

The reference solution is closed form. With 1:1 stoichiometry ``[B] = [A] + (B0 - A0)``, so the
ping-pong bi-bi system collapses to one ODE in the extent ``x = A0 - [A] = B0 - [B]`` whose
right-hand side is ``q / (K_A/(A0-x) + K_B/(B0-x) + 1)``; separating variables gives

    q t = x + K_A ln(A0 / (A0 - x)) + K_B ln(B0 / (B0 - x)),

strictly increasing and unbounded on ``[0, min(A0, B0))``. The substitution
``x = min(A0, B0) (1 - e^-z)`` makes the right-hand side asymptotically linear in ``z``, which gives
the exact bracket ``z <= q t / K_limiting`` used below. It is an INDEPENDENT check: nothing in the
detector uses it.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import detopt
from detopt.detector import EnzymeDepletionBiDetector
from detopt.detector.enzyme_depletion_bi import (
  EnzymeDepletionBiDesign, EnzymeDepletionBiEvent, EnzymeDepletionBiGroundTruth, EnzymeDepletionBiTarget,
)

N_EVENTS = 512


def exact_extent(t, initial_a, initial_b, velocity, michaelis_a, michaelis_b, n_bisect=80, n_newton=4):
  """The quadrature INVERTED, in float64, by bisection then Newton on ``z = ln(L0 / (L0 - x))``.

  Deliberately an INDEPENDENT re-derivation rather than an import from the calibration script: this
  is the thing the detector's RKC2 chain is checked against, and a shared implementation would let a
  single wrong derivation pass both. It is the PRODUCT form

      q t = x + K_A ln(A0/(A0-x)) + K_B ln(B0/(B0-x)) + (K_A K_B / D) ln(A0 (B0-x) / (B0 (A0-x)))

  with ``D = B0 - A0`` and its finite diagonal branch. The cross term is evaluated through ``log1p``
  of ``D expm1(z) / max(A0, B0)``, which is the same number without the cancellation that kills the
  difference-of-logarithms form as ``D -> 0``."""
  t, initial_a, initial_b, velocity, michaelis_a, michaelis_b = np.broadcast_arrays(
    *[np.asarray(v, np.float64) for v in (t, initial_a, initial_b, velocity, michaelis_a, michaelis_b)]
  )
  a_limits = initial_a <= initial_b
  limiting = np.where(a_limits, initial_a, initial_b)
  total = np.where(a_limits, initial_b, initial_a)
  k_limiting = np.where(a_limits, michaelis_a, michaelis_b)
  k_excess = np.where(a_limits, michaelis_b, michaelis_a)
  delta = total - limiting
  target = velocity * t

  def residual_and_slope(z):
    capped = np.minimum(z, 700.0)
    left = limiting * np.exp(-capped)
    extent = limiting * -np.expm1(-capped)
    grown = np.expm1(capped)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
      finite = np.log1p(delta * grown / total)
      asymptotic = z + np.log(np.where(delta > 0.0, delta, 1.0) / total)
      regular = np.where(z > 700.0, asymptotic, finite) / np.where(delta > 0.0, delta, 1.0)
      cross = np.where(delta > 0.0, regular, grown / limiting)
    value = (extent + k_limiting * capped + k_excess * np.log(total / (delta + left)) + k_limiting * k_excess * cross - target)
    return value, (k_limiting + left) * (k_excess + delta + left) / (delta + left)

  low, high = np.zeros_like(target), target / k_limiting + 1.0
  for _ in range(n_bisect):
    mid = 0.5 * (low + high)
    negative = residual_and_slope(mid)[0] < 0.0
    low = np.where(negative, mid, low)
    high = np.where(negative, high, mid)
  z = 0.5 * (low + high)
  for _ in range(n_newton):
    value, slope = residual_and_slope(z)
    z = np.clip(z - value / slope, low, high)
  return limiting * -np.expm1(-np.minimum(z, 700.0))


@pytest.fixture(scope='module')
def detector():
  """The SHIPPED instrument, at its shipped lattice.

  Deliberately not a cheaper one. A coarser lattice does not merely blur the objective, it destroys
  the design discrimination outright: at ``n_grid = 21`` a near-diagonal batch scores 0.0725 against
  the symmetry-breaking batch's 0.0728 -- indistinguishable -- where at the shipped 31 the same pair
  is 0.210 against 0.083. Testing a cheap lattice would have passed while the objective was blind."""
  return EnzymeDepletionBiDetector(n_experiments=2)


def nominal(detector, initial_a, initial_b):
  return EnzymeDepletionBiDesign(initial_a=jnp.asarray(initial_a, jnp.float32), initial_b=jnp.asarray(initial_b, jnp.float32))


def read_out_times(detector):
  return np.arange(1, detector.n_measurements + 1) * detector.duration / detector.n_measurements


def test_registered():
  assert detopt.detector.__detectors__['enzyme_depletion_bi'] is EnzymeDepletionBiDetector
  built = detopt.detector.from_config({'enzyme_depletion_bi': {'n_experiments': 3, 'n_grid': 9}})
  assert isinstance(built, EnzymeDepletionBiDetector)
  assert built.n_experiments == 3


def test_no_state(detector):
  """The design is passed per call: nothing about it may be stored on the detector."""
  assert not hasattr(detector, 'get_current_design_array')
  for name in dir(detector):
    assert 'current_design' not in name and 'nominal_design' not in name


def test_specs(detector):
  assert detector.design_shape() == (4, )
  assert detector.design_dim() == 4
  assert detector.target_dim() == 3
  assert detector.ground_truth_dim() == 3
  assert detector.combined_event_shape() == (2, detector.n_measurements + 2)
  assert detector.event_spec().extent.shape == (2, detector.n_measurements)
  assert set(detector.design_bounds()) == {'initial_a', 'initial_b'}
  assert detector.size() is None
  assert isinstance(detector.design_spec(), EnzymeDepletionBiDesign)


def test_exchangeable_blocks_are_the_experiment_pairs(detector):
  """The design is a SET of (A0, B0) PAIRS, so the surrogate must permute the two fields TOGETHER."""
  from detopt.bo import _exchangeable_blocks

  assert _exchangeable_blocks(detector, detector.n_experiments) == ((0, 1), (2, 3))


def test_design_round_trip(detector):
  """The two substrates have SEPARATE boxes, so the bijection is per-field, not one shared map."""
  a_values = np.geomspace(*detector.concentration_a_bounds, 7)
  b_values = np.geomspace(*detector.concentration_b_bounds, 7)
  for initial_a, initial_b in zip(a_values, b_values):
    design = nominal(detector, [initial_a, a_values[0]], [initial_b, b_values[-1]])
    scaled = np.asarray(detector.to_scaled(design))
    assert np.all(scaled >= -1e-6) and np.all(scaled <= 1.0 + 1e-6)
    back = detector.to_nominal(jnp.asarray(scaled))
    assert np.allclose(np.asarray(back.initial_a), [initial_a, a_values[0]], rtol=1e-5)
    assert np.allclose(np.asarray(back.initial_b), [initial_b, b_values[-1]], rtol=1e-5)
  # the bijection is affine on the LOG of each field's OWN box, so the box corners are cube corners
  corner = nominal(
    detector, [detector.concentration_a_bounds[0], detector.concentration_a_bounds[1]],
    [detector.concentration_b_bounds[1], detector.concentration_b_bounds[0]]
  )
  assert np.allclose(np.asarray(detector.to_scaled(corner)), [0.0, 1.0, 1.0, 0.0], atol=1e-6)


def test_target_round_trip(detector):
  kinetics = jnp.asarray([[2.8e-3, 0.06, 0.3], [9.0e-4, 0.02, 0.1], [9.0e-3, 0.2, 1.0]], jnp.float32)
  normalised = detector.normalize_target(EnzymeDepletionBiTarget(kinetics=kinetics))
  assert np.all(np.asarray(normalised) >= -1.0 - 1e-5) and np.all(np.asarray(normalised) <= 1.0 + 1e-5)
  assert np.allclose(np.asarray(normalised)[1], [-1.0, -1.0, -1.0], atol=1e-5)
  assert np.allclose(np.asarray(normalised)[2], [1.0, 1.0, 1.0], atol=1e-5)
  back = detector.denormalize_predictions(normalised)
  assert np.allclose(np.asarray(back.kinetics), np.asarray(kinetics), rtol=1e-4)


def test_integration_matches_the_exact_solution(detector):
  """The RKC2 chain the detector hands out, against the closed form, over the whole prior box.

  THE DIAGONAL IS IN THE SWEEP ON PURPOSE. Two independent concentration ranges share no value, so a
  product of two grids never contains a single ``A0 == B0`` point -- and that is where both substrates
  exhaust together and the rate is a removable ``0/0``. Both this sweep and the calibration script
  once missed it and passed while the detector returned NaN on every diagonal design."""
  times = read_out_times(detector)
  worst = 0.0
  for initial_a in (detector.concentration_a_bounds[0], 3.0, detector.concentration_a_bounds[1]):
    for initial_b in (detector.concentration_b_bounds[0], 6.0, detector.concentration_b_bounds[1], initial_a):
      for velocity in detector.velocity_bounds:
        for michaelis_a in (detector.michaelis_a_bounds[0], 0.063, detector.michaelis_a_bounds[1]):
          for michaelis_b in (detector.michaelis_b_bounds[0], 0.316, detector.michaelis_b_bounds[1]):
            got, error = jax.jit(detector._integrate)(initial_a, initial_b, velocity, michaelis_a, michaelis_b)
            truth = exact_extent(times, initial_a, initial_b, velocity, michaelis_a, michaelis_b)
            worst = max(worst, float(np.max(np.abs(np.asarray(got, np.float64) - truth))))
            assert float(error) <= detector.integration_tolerance
  assert worst <= detector.integration_tolerance, worst


def test_the_diagonal_is_integrable(detector):
  """``A0 == B0`` exhausts both substrates together, where the product form of the rate is ``0/0``.

  Every diagonal design in the box must integrate to a FINITE trajectory that matches the closed
  form, for parameters that complete the reaction well inside the window."""
  times = read_out_times(detector)
  worst = 0.0
  for initial in (detector.concentration_a_bounds[0], 1.5, 3.0, 8.0, detector.concentration_a_bounds[1]):
    for velocity in detector.velocity_bounds:
      for michaelis_a in detector.michaelis_a_bounds:
        for michaelis_b in detector.michaelis_b_bounds:
          got, error = jax.jit(detector._integrate)(initial, initial, velocity, michaelis_a, michaelis_b)
          got = np.asarray(got, np.float64)
          assert np.all(np.isfinite(got)), (initial, velocity, michaelis_a, michaelis_b)
          assert float(error) <= detector.integration_tolerance
          truth = exact_extent(times, initial, initial, velocity, michaelis_a, michaelis_b)
          worst = max(worst, float(np.max(np.abs(got - truth))))
  assert worst <= detector.integration_tolerance, worst

  # and the whole pipeline runs on a diagonal batch rather than raising or returning NaN
  diagonal = nominal(detector, [1.5, 8.0], [1.5, 8.0])
  _, event, _, target = detector(diagonal, np.arange(N_EVENTS))
  assert bool(np.all(np.isfinite(np.asarray(event.extent))))
  predicted = np.asarray(detector.estimate(diagonal, event), np.float64)
  assert bool(np.all(np.isfinite(predicted)))


def test_integration_tolerance_is_enforced():
  """An under-resolved chain must RAISE, not return a number: an error is not a result."""
  coarse = EnzymeDepletionBiDetector(n_experiments=1, steps_per_measurement=1, n_stages=2, n_grid=5)
  with pytest.raises(RuntimeError, match='integration_tolerance'):
    coarse(nominal(coarse, [10.0], [25.0]), np.arange(8))


def test_read_out_and_determinism(detector):
  design = nominal(detector, [1.5, 8.0], [12.0, 2.0])
  index = np.arange(N_EVENTS)
  truth, event, mask, target = detector(design, index)
  assert event.extent.shape == (N_EVENTS, 2, detector.n_measurements)
  assert mask.shape == (N_EVENTS, 2) and int(jnp.sum(mask)) == N_EVENTS * 2
  assert np.allclose(np.asarray(truth.kinetics), np.asarray(target.kinetics))

  # the variant depends on event_index ALONE -- a second design re-measures the same enzymes
  _, other_event, _, other_target = detector(nominal(detector, [3.0, 3.0], [6.0, 1.5]), index)
  assert np.allclose(np.asarray(other_target.kinetics), np.asarray(target.kinetics))
  assert not np.allclose(np.asarray(other_event.extent), np.asarray(event.extent))

  # the draw covers its prior box and the read-out noise is the configured sd
  kinetics = np.asarray(target.kinetics, np.float64)
  for column, bounds in enumerate((detector.velocity_bounds, detector.michaelis_a_bounds, detector.michaelis_b_bounds)):
    assert kinetics[:, column].min() > bounds[0] and kinetics[:, column].max() < bounds[1]
    assert kinetics[:, column].min() < bounds[0] * 1.5 and kinetics[:, column].max() > bounds[1] / 1.5
  clean = exact_extent(
    read_out_times(detector)[None, None, :],
    np.array([1.5, 8.0])[None, :, None],
    np.array([12.0, 2.0])[None, :, None], kinetics[:, 0][:, None, None], kinetics[:, 1][:, None, None], kinetics[:, 2][:, None,
                                                                                                                       None]
  )
  residual = np.asarray(event.extent, np.float64) - clean
  assert abs(float(np.std(residual)) / detector.measurement_noise - 1.0) < 0.06
  assert abs(float(np.mean(residual))) < 0.02 * detector.measurement_noise + 3e-4


def test_the_diagonal_is_weak_but_not_degenerate(detector):
  """The design problem, MEASURED against the exact model rather than asserted.

  UNDER THE PING-PONG LAW the diagonal was an EXACT degeneracy: the rate collapsed to
  ``q [A] / ([A] + K_A + K_B)`` and two triples with the same ``q`` and the same ``K_A + K_B`` gave
  literally the same curve. THE PRODUCT FORM DOES NOT DO THAT. Its rate on the diagonal is
  ``q a^2 / ((K_A + a)(K_B + a))``, which depends on the two constants through their PRODUCT as well
  as their sum, so equal-sum pairs separate -- weakly, but by more than the read-out noise at the
  best diagonal concentration.

  This test is the regression guard on the correction: it fails if the constant ``K_A K_B`` term is
  ever dropped from the denominator again, because that is exactly what restores the exact
  degeneracy this asserts is absent."""
  times = read_out_times(detector)
  velocity = math.sqrt(detector.velocity_bounds[0] * detector.velocity_bounds[1])
  diagonal = {}
  for initial in (1.0, 3.0, 10.0):
    one = exact_extent(times, initial, initial, velocity, 0.05, 0.55)
    two = exact_extent(times, initial, initial, velocity, 0.15, 0.45)
    diagonal[initial] = float(np.max(np.abs(one - two)))
    assert diagonal[initial] > 1.0e-3, (initial, diagonal)
  assert max(diagonal.values()) > detector.measurement_noise, diagonal

  separation = [
    float(np.max(np.abs(exact_extent(times, a, b, velocity, 0.05, 0.55) - exact_extent(times, a, b, velocity, 0.15, 0.45))))
    for a, b in ((3.0, 15.0), (3.0, 6.0), (10.0, 2.0))
  ]
  # symmetry breaking still PAYS -- every off-diagonal design beats the best diagonal one -- which is
  # why the batch should break it; what is gone is the reason to exclude the diagonal from the box.
  assert min(separation) > max(diagonal.values()), (separation, diagonal)

  # and the estimator sees the same ordering: an off-diagonal batch recovers K_A better than a
  # diagonal one, but the diagonal one is no longer stuck at the prior.
  index = np.arange(N_EVENTS)
  errors = {}
  for name, design in (('diagonal', nominal(detector, [1.5, 8.0], [1.5, 8.0])), ('off-diagonal', nominal(detector, [1.5, 8.0],
                                                                                                         [12.0, 2.0]))):
    _, event, _, target = detector(design, index)
    predicted = detector.denormalize_predictions(detector.estimate(design, event)).kinetics
    predicted = np.asarray(predicted, np.float64)
    truth = np.asarray(target.kinetics, np.float64)
    errors[name] = float(np.sqrt(np.mean(np.log(predicted[:, 1] / truth[:, 1])**2)))
  assert errors['off-diagonal'] < errors['diagonal'], errors


def test_combine(detector):
  design = nominal(detector, [1.5, 8.0], [12.0, 2.0])
  _, event, mask, _ = detector(design, np.arange(32))
  features = detector.combine(event, design)
  assert features.shape == (32, 2, detector.n_measurements + 2)
  # the last two channels are the SCALED design, broadcast over the batch
  scaled = np.asarray(detector.to_scaled(design))
  assert np.allclose(np.asarray(features[:, :, -2]), scaled[None, :2], atol=1e-6)
  assert np.allclose(np.asarray(features[:, :, -1]), scaled[None, 2:], atol=1e-6)
  # readings are fractions of each experiment's OWN maximum extent min(A0, B0)
  assert np.all(np.asarray(features[:, :, :-2]) < 1.5)
  assert np.allclose(np.asarray(detector.element_mask(event, mask)), np.asarray(mask))


def test_combine_is_not_design_blind(detector):
  """Two DIFFERENT designs must give different features AND a different loss.

  A normalisation that divides the read-out by a design-dependent scale can cancel the design
  exactly, leaving a detector that runs fine and scores every design identically."""
  index = np.arange(N_EVENTS)
  first = nominal(detector, [1.5, 8.0], [12.0, 2.0])
  second = nominal(detector, [2.0, 2.0], [2.0, 2.0])
  features, losses = [], []
  for design in (first, second):
    _, event, _, target = detector(design, index)
    features.append(np.asarray(detector.combine(event, design), np.float64))
    losses.append(float(jnp.mean(detector.loss(detector.estimate(design, event), detector.normalize_target(target)))))
  assert float(np.max(np.abs(features[0][:, :, :-2] - features[1][:, :, :-2]))) > 0.05
  assert abs(losses[0] - losses[1]) > 0.02, losses
  assert losses[0] < losses[1], losses  # the symmetry-breaking batch is the better one


def test_no_information_loss_is_the_prior(detector):
  """The closed-form 1/3 must equal the MEASURED loss of the best constant prediction."""
  _, _, _, target = detector(nominal(detector, [1.5, 8.0], [12.0, 2.0]), np.arange(8192))
  normalised = detector.normalize_target(target)
  guess = jnp.zeros_like(normalised)  # the prior mean in the normalised coordinate
  measured = float(jnp.mean(detector.loss(guess, normalised)))
  assert abs(measured - detector.no_information_loss()) < 0.01
  assert detector.no_information_loss() == pytest.approx(1.0 / 3.0)


def test_estimator_beats_the_prior_and_improves_with_information(detector):
  index = np.arange(N_EVENTS)
  design = nominal(detector, [1.5, 8.0], [12.0, 2.0])
  _, event, _, target = detector(design, index)
  normalised = detector.normalize_target(target)
  loss = float(jnp.mean(detector.loss(detector.estimate(design, event), normalised)))
  assert loss < 0.5 * detector.no_information_loss()
  assert np.all(np.abs(np.asarray(detector.estimate(design, event))) <= 1.0 + 1e-5)

  # a noisier assay is strictly worse
  noisy = EnzymeDepletionBiDetector(n_experiments=2, n_grid=21, measurement_noise=10.0 * detector.measurement_noise)
  _, noisy_event, _, noisy_target = noisy(design, index)
  noisy_loss = float(jnp.mean(noisy.loss(noisy.estimate(design, noisy_event), noisy.normalize_target(noisy_target))))
  assert noisy_loss > loss


def test_estimator_matches_a_float64_reference(detector):
  """The grid log-likelihood is a LARGE number whose useful content is a small difference between
  grid points, so float32 cancellation is a real risk. This rebuilds the same posterior in float64
  numpy from the CLOSED-FORM model and requires the two to agree far inside the loss scale."""
  design = nominal(detector, [1.5, 8.0], [12.0, 2.0])
  index = np.arange(128)
  _, event, _, target = detector(design, index)
  predicted = np.asarray(detector.estimate(design, event), np.float64)

  counts = (detector.n_grid_velocity, detector.n_grid, detector.n_grid)
  axes = [np.linspace(-1.0, 1.0, n) for n in counts]
  bounds = (detector.velocity_bounds, detector.michaelis_a_bounds, detector.michaelis_b_bounds)
  physical_axes = [np.exp(math.log(b[0]) + 0.5 * (a + 1.0) * math.log(b[1] / b[0])) for a, b in zip(axes, bounds)]
  mesh_scaled = [m.reshape(-1) for m in np.meshgrid(*axes, indexing='ij')]
  mesh_physical = [m.reshape(-1) for m in np.meshgrid(*physical_axes, indexing='ij')]
  model = exact_extent(
    read_out_times(detector)[None, None, :],
    np.array([1.5, 8.0])[None, :, None],
    np.array([12.0, 2.0])[None, :, None], mesh_physical[0][:, None, None], mesh_physical[1][:, None, None],
    mesh_physical[2][:, None, None]
  ).reshape(int(np.prod(counts)), -1)
  readings = np.asarray(event.extent, np.float64).reshape(index.size, -1)
  # The likelihood is formed as a MATMUL and chunked over events: the lattice has ~2e5 nodes, so the
  # explicit (events, nodes, read-outs) difference would be tens of gigabytes.
  square = 0.5 * np.sum(model * model, axis=-1)
  scaled_nodes = np.stack(mesh_scaled, axis=-1)
  reference = np.empty((index.size, 3))
  for start in range(0, index.size, 32):
    log_likelihood = (readings[start:start + 32] @ model.T - square[None, :]) / detector.measurement_noise**2
    log_likelihood -= log_likelihood.max(axis=-1, keepdims=True)
    weight = np.exp(log_likelihood)
    weight /= weight.sum(axis=-1, keepdims=True)
    reference[start:start + 32] = weight @ scaled_nodes

  worst = float(np.max(np.abs(predicted - reference)))
  assert worst < 0.02, worst
  normalised = np.asarray(detector.normalize_target(target), np.float64)
  assert abs(float(np.mean((predicted - normalised)**2) - np.mean((reference - normalised)**2))) < 1e-3


def test_estimator_saturates_at_the_prior_without_information():
  """With the read-out noise far above the signal the instrument must return the PRIOR MEAN, so the
  loss lands on `no_information_loss` rather than running past it."""
  blind = EnzymeDepletionBiDetector(n_experiments=1, measurement_noise=1.0e4, n_grid=11)
  design = nominal(blind, [3.0], [12.0])
  _, event, _, target = blind(design, np.arange(1024))
  predicted = blind.estimate(design, event)
  assert float(jnp.max(jnp.abs(predicted))) < 0.05
  loss = float(jnp.mean(blind.loss(predicted, blind.normalize_target(target))))
  assert abs(loss - blind.no_information_loss()) < 0.03


def test_the_lattice_is_fine_enough_to_be_a_bayes_estimator(detector):
  """The posterior mean CANNOT score worse than the prior -- so any per-parameter loss above 1/3 is
  proof the lattice is coarser than the posterior it represents.

  A self-contained check needing no reference value. It is what forced the lattice to be ANISOTROPIC:
  the velocity is determined an order of magnitude more sharply than either constant, so an isotropic
  lattice fine enough for the constants collapses the softmax onto one node in q, loses the
  marginalisation, and returns constants that are biased and over-confident."""
  index = np.arange(N_EVENTS)
  assert detector.n_grid_velocity > detector.n_grid
  for initial_a, initial_b in (([10.0, 10.0], [25.0, 25.0]), ([1.5, 8.0], [12.0, 2.0]), ([0.8, 0.8], [1.0, 1.0])):
    design = nominal(detector, initial_a, initial_b)
    _, event, _, target = detector(design, index)
    predicted = np.asarray(detector.estimate(design, event), np.float64)
    normalised = np.asarray(detector.normalize_target(target), np.float64)
    per_parameter = np.mean((predicted - normalised)**2, axis=0)
    assert np.all(per_parameter <= detector.no_information_loss() + 0.02), (initial_a, initial_b, per_parameter)


def test_design_is_a_set(detector):
  """Permuting the experiments names the SAME experiment, so the objective must be invariant --
  EXACTLY, not up to Monte-Carlo error: the read-out noise follows the batch's own order."""
  index = np.arange(N_EVENTS)
  losses = []
  for initial_a, initial_b in (([1.5, 8.0], [12.0, 2.0]), ([8.0, 1.5], [2.0, 12.0])):
    design = nominal(detector, initial_a, initial_b)
    _, event, _, target = detector(design, index)
    losses.append(float(jnp.mean(detector.loss(detector.estimate(design, event), detector.normalize_target(target)))))
  # exact in exact arithmetic; the residual is float32 reduction order over a 200k-node lattice
  assert abs(losses[0] - losses[1]) < 1e-5

  # The pair travels together, so swapping only A0 is a DIFFERENT BATCH and not a relabelling:
  # {(1.5, 12), (8, 2)} becomes {(8, 12), (1.5, 2)}. The objective must therefore MOVE -- by far more
  # than the permutation moves it, which is nothing. Which of the two is better is an empirical
  # question and deliberately NOT asserted here: `test_the_lattice_is_fine_enough_to_be_a_bayes_estimator`
  # and the calibration script's m = 1 profile are where design quality is measured.
  swapped = nominal(detector, [8.0, 1.5], [12.0, 2.0])
  _, event, _, target = detector(swapped, index)
  other = float(jnp.mean(detector.loss(detector.estimate(swapped, event), detector.normalize_target(target))))
  assert abs(other - losses[0]) > 1.0e-3, (other, losses[0])


def test_rejects_bad_configuration():
  for bad in ({'n_experiments': 0}, {'n_measurements': 1}, {'measurement_noise':
                                                            0.0}, {'duration': -1.0}, {'concentration_a_bounds': (10.0, 0.8)},
              {'concentration_b_bounds': (25.0, 1.0)}, {'michaelis_a_bounds': (0.0, 1.0)}, {'michaelis_b_bounds':
                                                                                            (0.0, 1.0)}, {'n_grid': 1}):
    with pytest.raises(ValueError):
      EnzymeDepletionBiDetector(**bad)
  with pytest.raises(ValueError):
    EnzymeDepletionBiDetector.from_config({'no_such_key': 1})
