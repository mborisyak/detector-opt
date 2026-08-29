"""`EnzymeDepletionDetector`: the contract surface, the RKC2 integration against the exact solution,
the read-out, the design SET symmetry and the analytic instrument.

The reference solution is closed form. Integrating ``dA/dt = -q A/(A+K)`` gives the implicit
``A + K ln A = A0 + K ln A0 - q t``; with ``w = A/K`` that is ``w + ln w = s``, a Lambert-W branch
solved here by Newton in ``ln w`` (stable for every ``s``). It is an INDEPENDENT check: nothing in
the detector uses it.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import detopt
from detopt.detector import EnzymeDepletionDetector
from detopt.detector.enzyme_depletion import (
  EnzymeDepletionDesign, EnzymeDepletionEvent, EnzymeDepletionGroundTruth, EnzymeDepletionTarget,
)

N_EVENTS = 512


def exact_concentration(t, initial, velocity, michaelis, n_newton=60):
  """The closed-form ``[A](t)``, in float64, by Newton on ``e^v + v = s`` with ``v = ln(A/K)``."""
  t, initial, velocity, michaelis = np.broadcast_arrays(*[np.asarray(x, np.float64) for x in (t, initial, velocity, michaelis)])
  s = np.log(initial / michaelis) + (initial - velocity * t) / michaelis
  v = np.where(s > 1.0, np.log(np.maximum(s - np.log(np.maximum(s, 1.0 + 1e-12)), 1e-300)), s)
  for _ in range(n_newton):
    ev = np.exp(np.clip(v, -700.0, 700.0))
    v = v - (ev + v - s) / (ev + 1.0)
  return michaelis * np.exp(np.clip(v, -700.0, 700.0))


@pytest.fixture(scope='module')
def detector():
  return EnzymeDepletionDetector(n_experiments=2)


def nominal(detector, values):
  return EnzymeDepletionDesign(initial_concentration=jnp.asarray(values, jnp.float32))


def test_registered():
  assert detopt.detector.__detectors__['enzyme_depletion'] is EnzymeDepletionDetector
  built = detopt.detector.from_config({'enzyme_depletion': {'n_experiments': 3, 'n_grid': 9}})
  assert isinstance(built, EnzymeDepletionDetector)
  assert built.n_experiments == 3


def test_no_state(detector):
  """The design is passed per call: nothing about it may be stored on the detector."""
  assert not hasattr(detector, 'get_current_design_array')
  for name in dir(detector):
    assert 'current_design' not in name and 'nominal_design' not in name


def test_specs(detector):
  assert detector.design_shape() == (2, )
  assert detector.design_dim() == 2
  assert detector.target_dim() == 2
  assert detector.ground_truth_dim() == 2
  assert detector.combined_event_shape() == (2, detector.n_measurements + 1)
  assert detector.event_spec().concentration.shape == (2, detector.n_measurements)
  assert set(detector.design_bounds()) == {'initial_concentration'}
  assert detector.size() is None
  assert isinstance(detector.design_spec(), EnzymeDepletionDesign)


def test_design_round_trip(detector):
  low, high = detector.concentration_bounds
  values = np.geomspace(low, high, 7)
  for a, b in zip(values[:-1], values[1:]):
    design = nominal(detector, [a, b])
    scaled = np.asarray(detector.to_scaled(design))
    assert np.all(scaled >= -1e-6) and np.all(scaled <= 1.0 + 1e-6)
    back = np.asarray(detector.to_nominal(jnp.asarray(scaled)).initial_concentration)
    assert np.allclose(back, [a, b], rtol=1e-5)
  # the bijection is affine on the LOG, so the box corners are the cube corners
  assert np.allclose(np.asarray(detector.to_scaled(nominal(detector, [low, high]))), [0.0, 1.0], atol=1e-6)


def test_target_round_trip(detector):
  kinetics = jnp.asarray([[3.0e-4, 0.5], [1.0e-4, 0.01], [1.0e-3, 20.0]], jnp.float32)
  normalised = detector.normalize_target(EnzymeDepletionTarget(kinetics=kinetics))
  assert np.all(np.asarray(normalised) >= -1.0 - 1e-5) and np.all(np.asarray(normalised) <= 1.0 + 1e-5)
  assert np.allclose(np.asarray(normalised)[1], [-1.0, -1.0], atol=1e-5)
  assert np.allclose(np.asarray(normalised)[2], [1.0, 1.0], atol=1e-5)
  back = detector.denormalize_predictions(normalised)
  assert np.allclose(np.asarray(back.kinetics), np.asarray(kinetics), rtol=1e-4)


def test_integration_matches_the_exact_solution(detector):
  """The RKC2 chain the detector hands out, against the closed form, over the whole prior box."""
  times = np.arange(1, detector.n_measurements + 1) * detector.duration / detector.n_measurements
  worst = 0.0
  for initial in (detector.concentration_bounds[0], 1.5, detector.concentration_bounds[1]):
    for velocity in detector.velocity_bounds:
      for michaelis in (detector.michaelis_bounds[0], 0.45, detector.michaelis_bounds[1]):
        got, error = jax.jit(detector._integrate)(initial, velocity, michaelis)
        truth = exact_concentration(times, initial, velocity, michaelis)
        worst = max(worst, float(np.max(np.abs(np.asarray(got, np.float64) - truth))))
        assert float(error) <= detector.integration_tolerance
  assert worst <= detector.integration_tolerance, worst


def test_integration_tolerance_is_enforced():
  """An under-resolved chain must RAISE, not return a number: an error is not a result."""
  coarse = EnzymeDepletionDetector(n_experiments=1, steps_per_measurement=1, n_stages=2, n_grid=5)
  with pytest.raises(RuntimeError, match='integration_tolerance'):
    coarse(nominal(coarse, [4.0]), np.arange(8))


def test_read_out_and_determinism(detector):
  design = nominal(detector, [0.6, 3.0])
  index = np.arange(N_EVENTS)
  truth, event, mask, target = detector(design, index)
  assert event.concentration.shape == (N_EVENTS, 2, detector.n_measurements)
  assert mask.shape == (N_EVENTS, 2) and int(jnp.sum(mask)) == N_EVENTS * 2
  assert np.allclose(np.asarray(truth.kinetics), np.asarray(target.kinetics))

  # the variant depends on event_index ALONE -- a second design re-measures the same enzymes
  _, other_event, _, other_target = detector(nominal(detector, [1.0, 2.0]), index)
  assert np.allclose(np.asarray(other_target.kinetics), np.asarray(target.kinetics))
  assert not np.allclose(np.asarray(other_event.concentration), np.asarray(event.concentration))

  # the draw covers its prior box and the read-out noise is the configured sd
  kinetics = np.asarray(target.kinetics, np.float64)
  for column, bounds in enumerate((detector.velocity_bounds, detector.michaelis_bounds)):
    assert kinetics[:, column].min() > bounds[0] and kinetics[:, column].max() < bounds[1]
    assert kinetics[:, column].min() < bounds[0] * 1.5 and kinetics[:, column].max() > bounds[1] / 1.5
  times = np.arange(1, detector.n_measurements + 1) * detector.duration / detector.n_measurements
  clean = exact_concentration(
    times[None, None, :],
    np.array([0.6, 3.0])[None, :, None], kinetics[:, 0][:, None, None], kinetics[:, 1][:, None, None]
  )
  residual = np.asarray(event.concentration, np.float64) - clean
  assert abs(float(np.std(residual)) / detector.measurement_noise - 1.0) < 0.06
  assert abs(float(np.mean(residual))) < 0.01 * detector.measurement_noise + 3e-4


def test_combine(detector):
  design = nominal(detector, [0.6, 3.0])
  _, event, mask, _ = detector(design, np.arange(32))
  features = detector.combine(event, design)
  blind = detector.combine(event, design, reveal_design=False)
  assert blind.shape == (32, 2, detector.n_measurements)
  assert np.allclose(np.asarray(blind), np.asarray(features[..., :-1]))
  assert features.shape == (32, 2, detector.n_measurements + 1)
  # last channel is the SCALED design, broadcast over the batch
  assert np.allclose(np.asarray(features[:, :, -1]), np.asarray(detector.to_scaled(design))[None, :], atol=1e-6)
  # readings are RAW concentrations -- no normalisation, so they are bounded by the design box top
  assert np.all(np.asarray(features[:, :, :-1]) < 1.5 * detector.concentration_bounds[1])
  assert np.allclose(np.asarray(detector.element_mask(event, mask)), np.asarray(mask))


def test_no_information_loss_is_the_prior(detector):
  """The closed-form 1/3 must equal the MEASURED loss of the best constant prediction."""
  _, _, _, target = detector(nominal(detector, [0.6, 3.0]), np.arange(8192))
  normalised = detector.normalize_target(target)
  guess = jnp.zeros_like(normalised)  # the prior mean in the normalised coordinate
  measured = float(jnp.mean(detector.loss(guess, normalised)))
  assert abs(measured - detector.no_information_loss()) < 0.01
  assert detector.no_information_loss() == pytest.approx(1.0 / 3.0)


def test_estimator_beats_the_prior_and_improves_with_information(detector):
  index = np.arange(N_EVENTS)
  design = nominal(detector, [0.6, 3.0])
  _, event, _, target = detector(design, index)
  normalised = detector.normalize_target(target)
  loss = float(jnp.mean(detector.loss(detector.estimate(design, event), normalised)))
  assert loss < 0.5 * detector.no_information_loss()
  assert np.all(np.abs(np.asarray(detector.estimate(design, event))) <= 1.0 + 1e-5)

  # a noisier assay is strictly worse, and a design outside the informative window is worse still
  noisy = EnzymeDepletionDetector(n_experiments=2, measurement_noise=10.0 * detector.measurement_noise)
  _, noisy_event, _, noisy_target = noisy(design, index)
  noisy_loss = float(jnp.mean(noisy.loss(noisy.estimate(design, noisy_event), noisy.normalize_target(noisy_target))))
  assert noisy_loss > loss


def test_estimator_matches_a_float64_reference(detector):
  """The grid log-likelihood is a LARGE number whose useful content is a small difference between
  grid points -- `y . m - |m|^2 / 2` over the read-out, divided by a small variance -- so float32
  cancellation is a real risk. This rebuilds the same posterior in float64 numpy from the closed-form
  model and requires the two to agree far inside the loss scale."""
  design = nominal(detector, [0.6, 3.0])
  index = np.arange(256)
  _, event, _, target = detector(design, index)
  predicted = np.asarray(detector.estimate(design, event), np.float64)

  axis = np.linspace(-1.0, 1.0, detector.n_grid)
  velocity = np.exp(
    np.log(detector.velocity_bounds[0]) + 0.5 *
    (axis + 1.0) * math.log(detector.velocity_bounds[1] / detector.velocity_bounds[0])
  )
  michaelis = np.exp(
    np.log(detector.michaelis_bounds[0]) + 0.5 *
    (axis + 1.0) * math.log(detector.michaelis_bounds[1] / detector.michaelis_bounds[0])
  )
  grid_velocity, grid_michaelis = np.meshgrid(velocity, michaelis, indexing='ij')
  grid_q, grid_k = np.meshgrid(axis, axis, indexing='ij')
  times = np.arange(1, detector.n_measurements + 1) * detector.duration / detector.n_measurements
  model = exact_concentration(
    times[None, None, :],
    np.array([0.6, 3.0])[None, :, None],
    grid_velocity.reshape(-1)[:, None, None],
    grid_michaelis.reshape(-1)[:, None, None]
  ).reshape(detector.n_grid**2, -1)
  readings = np.asarray(event.concentration, np.float64).reshape(index.size, -1)
  log_likelihood = -0.5 * np.sum((readings[:, None, :] - model[None, :, :])**2, axis=-1) / detector.measurement_noise**2
  log_likelihood -= log_likelihood.max(axis=-1, keepdims=True)
  weight = np.exp(log_likelihood)
  weight /= weight.sum(axis=-1, keepdims=True)
  reference = np.stack([weight @ grid_q.reshape(-1), weight @ grid_k.reshape(-1)], axis=-1)

  worst = float(np.max(np.abs(predicted - reference)))
  assert worst < 0.02, worst
  normalised = np.asarray(detector.normalize_target(target), np.float64)
  assert abs(float(np.mean((predicted - normalised)**2) - np.mean((reference - normalised)**2))) < 1e-3


def test_estimator_saturates_at_the_prior_without_information():
  """With the read-out noise far above the signal the instrument must return the PRIOR MEAN, so the
  loss lands on `no_information_loss` rather than running past it."""
  blind = EnzymeDepletionDetector(n_experiments=1, measurement_noise=1.0e4, n_grid=21)
  design = nominal(blind, [1.5])
  _, event, _, target = blind(design, np.arange(1024))
  predicted = blind.estimate(design, event)
  assert float(jnp.max(jnp.abs(predicted))) < 0.05
  loss = float(jnp.mean(blind.loss(predicted, blind.normalize_target(target))))
  assert abs(loss - blind.no_information_loss()) < 0.03


def test_design_is_a_set(detector):
  """Permuting the experiments names the SAME experiment, so the objective must be invariant --
  EXACTLY, not up to Monte-Carlo error: the read-out noise follows the concentration's rank."""
  index = np.arange(N_EVENTS)
  losses = []
  for values in ([0.6, 3.0], [3.0, 0.6]):
    design = nominal(detector, values)
    _, event, _, target = detector(design, index)
    losses.append(float(jnp.mean(detector.loss(detector.estimate(design, event), detector.normalize_target(target)))))
  assert abs(losses[0] - losses[1]) < 1e-6


def test_rejects_bad_configuration():
  for bad in ({'n_experiments': 0}, {'n_measurements': 1}, {'measurement_noise': 0.0}, {'duration': -1.0},
              {'concentration_bounds': (4.0, 0.45)}, {'michaelis_bounds': (0.0, 1.0)}, {'n_grid': 1}):
    with pytest.raises(ValueError):
      EnzymeDepletionDetector(**bad)
  with pytest.raises(ValueError):
    EnzymeDepletionDetector.from_config({'no_such_key': 1})
