"""The semi-analytic Bayes instrument for the binary inhibitor-mechanism task.

Small and CPU-only: every detector here is the reference `enzyme_extremes` configuration with the
read-out cadence and the batch cut down, so the whole module runs in seconds.
"""

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from detopt.analytic import MechanismInstrument, build_detector, sobol_designs
from detopt.utils.config import load_config

REFERENCE = 'config/detector/enzyme_extremes.yaml'


def configuration():
  return load_config(REFERENCE)


def small(noise=0.05, **overrides):
  """The noiseless twin of a cut-down reference detector."""
  arguments = dict(
    n_experiments=2, n_measurements=4, n_steps_per_measurement=40, measurement_noise=0.0, integration_tolerance=0.1 * noise
  )
  arguments.update(overrides)
  return build_detector(configuration(), **arguments)


def instrument(detector, noise=0.05, **overrides):
  arguments = dict(n_events=64, n_library=512, n_noise=2, hedge=1.0e-3, seed=3)
  arguments.update(overrides)
  return MechanismInstrument(detector, noise=noise, **arguments)


def test_read_out_noise_matches_the_detector():
  """The instrument adds the read-out noise itself; the noiseless twin plus that noise must be the
  shipped detector's own output, event for event."""
  noise = 0.05
  clean = small(noise)
  noisy = build_detector(
    configuration(), n_experiments=2, n_measurements=4, n_steps_per_measurement=40, measurement_noise=noise,
    integration_tolerance=0.1 * noise
  )
  design = np.full(clean.design_dim(), 0.5, np.float32)
  nominal = np.asarray(clean._to_nominal_flat(design))
  indices = np.arange(2048, dtype=np.int64)
  _, first, _, first_target = clean(nominal, indices)
  _, second, _, second_target = noisy(nominal, indices)
  residual = np.asarray(second.measurements - first.measurements)
  assert np.allclose(np.asarray(first_target.mechanism), np.asarray(second_target.mechanism))
  assert abs(float(residual.std()) / noise - 1.0) < 0.05
  assert abs(float(residual.mean()) / noise) < 0.05


def test_class_swap_is_an_enzyme_reparameterisation():
  """Swapping ``k1`` and ``k2`` is EXACTLY ``K_B -> K_B r``, ``Ki_D -> Ki_D r``, ``k_cat -> k_cat r``
  with ``r = (1 + I k2) / (1 + I k1)``: the reason one inhibitor level cannot separate the classes
  except through the prior's support."""
  detector = small(n_experiments=1)
  generator = np.random.default_rng(0)
  n = 32
  parameters = {
    name: generator.uniform(low, high, size=n).astype(np.float32)
    for name, (low, high) in detector._parameter_ranges
  }
  half_time = np.full(n, 1.0, np.float32)
  k1 = np.power(10.0, generator.uniform(-1.0, 2.0, size=n)).astype(np.float32)
  k2 = (k1 * np.power(10.0, -generator.uniform(1.0, 2.5, size=n))).astype(np.float32)
  inhibitor_value = 0.3
  ratio = ((1.0 + inhibitor_value * k2) / (1.0 + inhibitor_value * k1)).astype(np.float32)

  def simulate(params, times, first, second):

    def one(p, h, a, b):
      log_k0_cat = detector._calibrate(p, h)
      measurements, _ = detector._run_batch(
        np.float32(0.5) * jnp.ones(1), jnp.ones(1), inhibitor_value * jnp.ones(1),
        np.float32(25.0) * jnp.ones(1), p, log_k0_cat, a, b, jax.random.PRNGKey(0)
      )
      return measurements

    return np.asarray(jax.jit(jax.vmap(one))(params, times, first, second))

  calibrate = jax.jit(jax.vmap(detector._calibrate))
  shifted = dict(parameters)
  shifted['log_K0_B'] = (parameters['log_K0_B'] + np.log(ratio)).astype(np.float32)
  shifted['log_K0i_D'] = (parameters['log_K0i_D'] + np.log(ratio)).astype(np.float32)
  correction = np.exp(np.asarray(calibrate(shifted, half_time)) - np.asarray(calibrate(parameters, half_time)))

  mirrored = simulate(parameters, half_time, k2, k1)
  equivalent = simulate(shifted, (half_time * correction / ratio).astype(np.float32), k1, k2)
  drawn = simulate(parameters, half_time, k1, k2)
  assert float(np.max(np.abs(mirrored - equivalent))) < 1.0e-5
  assert float(np.max(np.abs(mirrored - drawn))) > 1.0e-3


def test_an_inert_inhibitor_leaves_the_guess_level():
  """At a dose far below every compound's Ki the two classes produce the same read-out, so the
  Bayes-optimal answer is the class prior and the loss is the no-information level 1.0."""
  detector = small(inhibitor_bounds=(1.0e-9, 1.1e-9))
  result = instrument(detector, n_events=128, n_library=1024).evaluate(np.full(detector.design_dim(), 0.5))
  assert abs(result.loss - 1.0) < 4.0 * result.standard_error + 0.01
  assert result.effective_sample_size > 0.1 * 1024


def test_a_spread_of_doses_beats_the_guess_level():
  """The positive control, and the shape of it is the physics: the class swap is a reparameterisation
  of the enzyme AT ONE inhibitor level, so a batch that contrasts two doses must carry information
  where a batch at one dose need not. Below the no-information level and above chance accuracy."""
  noise = 0.05
  detector = small(noise, n_experiments=4, n_measurements=8, n_steps_per_measurement=160)
  design = np.full(detector.design_dim(), 0.5)
  design[detector.n_experiments:2 * detector.n_experiments] = [0.2, 0.2, 0.9, 0.9]
  design[2 * detector.n_experiments:3 * detector.n_experiments] = [0.3, 1.0, 0.3, 1.0]
  design[3 * detector.n_experiments:] = 0.6
  result = instrument(detector, noise=noise, n_events=256, n_library=2048).evaluate(design)
  assert result.loss < 1.0 - 3.0 * result.standard_error
  assert result.accuracy > 0.5


def test_the_hedge_bounds_the_loss():
  hedge = 0.05
  detector = small()
  probe = instrument(detector, hedge=hedge)
  result = probe.evaluate(np.full(detector.design_dim(), 0.5))
  assert result.loss <= -math.log(0.5 * hedge) / math.log(2.0)


def test_effective_sample_size_grows_with_the_library():
  detector = small()
  design = np.full(detector.design_dim(), 0.5)
  small_library = instrument(detector, n_library=256).evaluate(design)
  large_library = instrument(detector, n_library=2048).evaluate(design)
  assert large_library.effective_sample_size > 2.0 * small_library.effective_sample_size


def test_events_and_library_do_not_overlap():
  detector = small()
  probe = instrument(detector)
  assert len(np.intersect1d(probe.event_indices, probe.library_indices)) == 0


def test_evaluation_is_deterministic_and_design_dependent():
  detector = small(inhibitor_bounds=(0.3, 1.0))
  probe = instrument(detector, n_events=128, n_library=1024)
  design = np.full(detector.design_dim(), 0.5)
  assert probe.evaluate(design).loss == probe.evaluate(design).loss
  other = design.copy()
  other[2 * detector.n_experiments:3 * detector.n_experiments] = 0.0
  assert probe.evaluate(other).loss != probe.evaluate(design).loss


def test_instrument_rejects_a_noisy_detector_and_a_three_class_library():
  noisy = build_detector(configuration(), n_experiments=2, n_measurements=4, n_steps_per_measurement=40, measurement_noise=0.05)
  with pytest.raises(ValueError):
    MechanismInstrument(noisy, noise=0.05, n_events=8, n_library=16)
  three = build_detector(
    configuration(), n_experiments=2, n_measurements=4, n_steps_per_measurement=40, measurement_noise=0.0,
    mechanism_classes=('mostly_competitive', 'mostly_noncompetitive', 'mostly_uncompetitive')
  )
  with pytest.raises(ValueError):
    MechanismInstrument(three, noise=0.05, n_events=8, n_library=16)


def test_sobol_designs_fill_the_unit_cube():
  points = sobol_designs(6, 32, seed=1)
  assert points.shape == (32, 6)
  assert points.min() >= 0.0 and points.max() <= 1.0
  assert len(np.unique(points, axis=0)) == 32
