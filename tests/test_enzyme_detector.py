"""Tests for the enzyme detector (single-batch design of experiments) + its meta-regressors.

The detector contract (record specs, design encode/decode, combine, loss/metric) plus the two
properties the benchmark rests on: the target is a property of the EVENT alone (so a design cannot
move its own label) and the turnover calibration really does put the half-conversion time where it
claims. The integration guard is checked by deliberately under-resolving the integration, and the
RKC2 stability boundary is measured rather than assumed.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import detopt
from detopt.detector.enzyme import (
  EnzymeDesign, EnzymeEvent, EnzymeGroundTruth, EnzymeTarget, PARAMETER_NAMES
)
from detopt.nn import from_config
from detopt.utils.config import load_config


def _config(**override):
  config = dict(load_config('config/detector/enzyme.yaml')['enzyme'])
  config.update(override)
  return config


def _detector(**override):
  return detopt.detector.EnzymeDetector(**_config(**override))


def _temperatures(detector, n=257):
  """A dense sweep of the whole temperature range, for tests that need to check something EVERYWHERE
  the prior can reach. The detector itself tabulates no temperatures (it bisects), so the resolution
  of a diagnostic sweep is the test's own choice."""
  return jnp.linspace(*detector.temperature_bounds, n, dtype=jnp.float32)


def _design(detector):
  """A spread-out batch, built outside the detector (it holds no design of its own)."""
  low, high = detector.enzyme_fraction_bounds
  t_low, t_high = detector.temperature_bounds
  n = detector.n_experiments
  return {
    'enzyme_fraction': list(np.linspace(low + 0.05, high - 0.05, n)),
    'temperature': list(np.linspace(t_low + 10.0, t_high - 20.0, n))
  }


def test_registered_and_built_from_config():
  """The canonical config builds through the registry, unknown keys are rejected."""
  detector = detopt.detector.from_config(load_config('config/detector/enzyme.yaml'))
  assert isinstance(detector, detopt.detector.EnzymeDetector)
  assert detopt.detector.__detectors__['enzyme'] is detopt.detector.EnzymeDetector
  with pytest.raises(ValueError, match='unknown'):
    detopt.detector.EnzymeDetector.from_config(_config(nonsense=1))


def test_specs():
  detector = _detector()
  n, m = detector.n_experiments, detector.n_measurements
  event = detector.event_spec()
  assert isinstance(event, EnzymeEvent)
  assert event.measurements.shape == (n, m) and event.measurements.dtype == np.float32
  assert isinstance(detector.target_spec(), EnzymeTarget)
  assert isinstance(detector.ground_truth_spec(), EnzymeGroundTruth)
  assert isinstance(detector.design_spec(), EnzymeDesign)
  assert detector.target_dim() == 1
  assert detector.ground_truth_dim() == len(PARAMETER_NAMES) + 1  # + half_time
  assert detector.design_dim() == 2 * n
  assert detector.combined_event_shape() == (n, m + 2)
  assert detector.combined_feature_dim() == m + 2
  assert detector.size() is None  # analytic source


def test_design_scaling_is_a_bijection_onto_the_bounds():
  """Nominal <-> scaled is affine per coordinate, so the unit cube IS the design box.

  The scaled space is BOUNDED (it was unconstrained R^n under the old quantile encoding), so the
  invariant is no longer "any point of R^n lands in the bounds" -- that is false by construction
  here, and an out-of-cube u is simply not a design. What must hold: the round trip is exact, the
  cube's corners are the bounds exactly, and every point of the cube is admissible."""
  detector = _detector()
  design = _design(detector)
  physical = np.asarray(detector.flatten_design(design))
  scaled = detector.to_scaled(design)
  assert np.all(np.asarray(scaled) >= 0.0) and np.all(np.asarray(scaled) <= 1.0)
  assert np.allclose(np.asarray(detector.flatten_design(detector.to_nominal(scaled))), physical, atol=1e-4)

  rng = np.random.default_rng(0)
  cube = jnp.asarray(rng.uniform(0.0, 1.0, (64, detector.design_dim())), jnp.float32)
  # The corners included: u = 0 / u = 1 must BE the bounds, not merely approach them.
  cube = jnp.concatenate([cube, jnp.zeros((1, detector.design_dim()), jnp.float32),
                          jnp.ones((1, detector.design_dim()), jnp.float32)], axis=0)
  nominal = detector.to_nominal(cube)
  for values, (low, high) in ((nominal.enzyme_fraction, detector.enzyme_fraction_bounds),
                              (nominal.temperature, detector.temperature_bounds)):
    values = np.asarray(values)
    span = 1e-6 * (high - low)
    assert np.all((low - span <= values) & (values <= high + span))
    assert np.isclose(values.min(), low, atol=span) and np.isclose(values.max(), high, atol=span)


def test_event_is_deterministic_and_the_target_ignores_the_design():
  """The enzyme and its readout noise are seeded from `event_index` alone: the same call repeats
  exactly, and running the same events under a different design leaves the target untouched --
  otherwise a design could move its own label."""
  detector = _detector()
  index = np.arange(64)
  design = _design(detector)
  first = detector(design, index)
  again = detector(design, index)
  assert np.array_equal(np.asarray(first[1].measurements), np.asarray(again[1].measurements))

  other = {'enzyme_fraction': [0.5] * detector.n_experiments, 'temperature': [35.0] * detector.n_experiments}
  shifted = detector(other, index)
  assert np.allclose(np.asarray(first[3].melting_temperature), np.asarray(shifted[3].melting_temperature))
  assert np.allclose(np.asarray(first[0].parameters), np.asarray(shifted[0].parameters))
  # ... while the measurements themselves DO follow the design.
  assert not np.allclose(np.asarray(first[1].measurements), np.asarray(shifted[1].measurements))


def test_event_shapes_and_ranges():
  detector = _detector()
  index = np.arange(128)
  ground_truth, event, mask, target = detector(_design(detector), index)
  n, m = detector.n_experiments, detector.n_measurements
  assert event.measurements.shape == (128, n, m)
  assert mask.shape == (128, n) and np.all(np.asarray(mask) == 1)  # every experiment is real
  assert target.melting_temperature.shape == (128, 1)
  assert ground_truth.parameters.shape == (128, len(PARAMETER_NAMES))
  assert np.all(np.isfinite(np.asarray(event.measurements)))

  # The target is the DRAWN T_melting, so it must lie inside that parameter's prior range.
  low, high = detector.melting_bounds
  melting = np.asarray(target.melting_temperature)
  assert np.all((low <= melting) & (melting <= high))
  half_time = np.asarray(ground_truth.half_time)
  assert np.all((detector.half_time_bounds[0] <= half_time) & (half_time <= detector.half_time_bounds[1]))
  # [A] decreases: the reaction only consumes it (up to the readout noise).
  measurements = np.asarray(event.measurements)
  assert np.mean(measurements[:, :, -1]) < np.mean(measurements[:, :, 0])


def test_a_single_design_or_one_design_per_event():
  """A single design is broadcast over the event batch; a batched design is used per event."""
  detector = _detector()
  index = np.arange(16)
  design = _design(detector)
  broadcast = detector(design, index)[1].measurements
  batched = detector(
    jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (16,) + jnp.asarray(a).shape),
                 detector.to_nominal(detector.to_scaled(design))),
    index
  )[1].measurements
  assert np.allclose(np.asarray(broadcast), np.asarray(batched), atol=1e-3)


def test_combine_is_design_informed_and_differentiable():
  detector = _detector()
  index = np.arange(8)
  _, event, mask, _ = detector(_design(detector), index)
  encoded = jnp.broadcast_to(detector.to_scaled(_design(detector))[None], (8, detector.design_dim()))

  features = detector.combine_scaled(event, encoded, mask=mask)
  assert features.shape == (8,) + detector.combined_event_shape()
  # The trailing two features are the experiment's own design values, in ~[-1, 1].
  assert np.all(np.abs(np.asarray(features[..., -2:])) <= 1.0 + 1e-5)
  assert np.array_equal(np.asarray(detector.element_mask(event, mask)), np.asarray(mask))

  gradient = jax.grad(lambda d: jnp.sum(detector.combine_scaled(event, d, mask=mask)))(encoded)
  assert np.all(np.isfinite(np.asarray(gradient)))
  assert np.any(np.abs(np.asarray(gradient)) > 0)  # the design really reaches the features


def test_target_normalisation_round_trip_and_loss():
  detector = _detector()
  _, _, _, target = detector(_design(detector), np.arange(32))
  normalised = detector.normalize_target(target)
  assert normalised.shape == (32, 1) and np.all(np.abs(np.asarray(normalised)) <= 1.0 + 1e-5)
  back = detector.denormalize_predictions(normalised)
  assert np.allclose(np.asarray(back.melting_temperature), np.asarray(target.melting_temperature), atol=1e-3)

  predicted = normalised + 0.1
  loss = detector.loss(predicted, normalised)
  assert loss.shape == (32,) and np.allclose(np.asarray(loss), 0.01, atol=1e-6)
  metric = detector.metric(predicted, normalised)
  assert set(metric) == set(detector.metric_labels())
  assert all(v.shape == (32,) for v in metric.values())
  # 0.1 of a normalised temperature is 10% of the half-range -- of the TARGET's own range, the
  # T_melting prior, which is what normalize_target scales by.
  low, high = detector.melting_bounds
  rmse, unit = detector.metric_real_rmse({'melting_temperature': 0.01})['melting_temperature']
  assert unit == 'C' and np.isclose(rmse, 0.1 * 0.5 * (high - low))


def test_ground_truth_normalisation_is_bounded():
  detector = _detector()
  ground_truth, _, _, _ = detector(_design(detector), np.arange(64))
  normalised = np.asarray(detector.normalize_ground_truth(ground_truth))
  assert normalised.shape == (64, detector.ground_truth_dim())
  assert np.all(np.abs(normalised) <= 1.0 + 1e-5)  # every field carries its own prior range


def test_calibration_puts_the_fastest_half_conversion_at_half_time():
  """The physics claim of `_calibrate`: over all temperatures, the earliest time the calibration
  mixture reaches half conversion is `half_time`. Measured by integrating that mixture in real time
  on a fine grid and taking the earliest crossing -- no reuse of the calibration's own quadrature."""
  # Half-times comfortably inside `duration` so the crossing is visible at all (the configured band
  # reaches 2 h, twice the experiment), and a fine time grid.
  detector = _detector(half_time_bounds=[0.4, 0.5], n_measurements=64)
  times = np.asarray(detector.measurement_times)

  def earliest_crossing(index):
    key_parameters, key_half_time, _ = jax.random.split(jax.random.PRNGKey(index), 3)
    parameters = detector._draw_parameters(key_parameters)
    low, high = detector.half_time_bounds
    half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=np.log(low), maxval=np.log(high)))
    log_k0_cat = detector._calibrate(parameters, half_time)
    A0, B0, E0 = detector._initial_state(detector.calibration_fraction)

    def crossing(temperature):
      extent, _ = detector._integrate(A0, B0, E0, temperature, parameters, log_k0_cat,
                                  n_steps=detector.n_steps_per_measurement, n_intervals=detector.n_measurements)
      conversion = extent / jnp.minimum(A0, B0)
      # First grid time at or past half conversion (`duration` if it never gets there).
      reached = conversion >= 0.5
      return jnp.min(jnp.where(reached, jnp.asarray(times), detector.duration))

    return jnp.min(jax.vmap(crossing)(_temperatures(detector))), half_time

  crossing, half_time = jax.jit(jax.vmap(earliest_crossing))(jnp.arange(64, dtype=jnp.int32))
  crossing, half_time = np.asarray(crossing), np.asarray(half_time)
  # The crossing is read off a grid of `duration / n_measurements` steps, so it lands within one
  # step above the true half-time -- or a hair below it, where Euler truncation puts the numerical
  # solution slightly ahead of the exact one.
  step = detector.duration / detector.n_measurements
  assert np.all(crossing >= half_time - 0.05 * step), f'worst {np.min(crossing - half_time):.5f} h'
  assert np.all(crossing <= half_time + step + 1e-6), f'worst {np.max(crossing - half_time):.5f} h'


def test_integration_guard_fires_when_under_resolved():
  """A deliberately coarse step must raise -- never quietly return a wrong reaction. The error the
  guard asserts on is the disagreement between the solve's own dt and dt/2 chains."""
  detector = _detector(n_objective_steps=2, n_steps_per_measurement=1, integration_tolerance=1e-9)
  with pytest.raises(RuntimeError, match='integration_tolerance'):
    detector(_design(detector), np.arange(8))


def test_configuration_is_validated():
  with pytest.raises(ValueError, match='T_melting'):
    _detector(temperature_bounds=[0.0, 50.0])  # the prior's T_melting reaches 80 C
  with pytest.raises(ValueError, match='missing'):
    _detector(parameters={name: [0.0, 1.0] for name in PARAMETER_NAMES if name != 'Q10_cat'})
  with pytest.raises(ValueError, match='unknown'):
    _detector(parameters={**{name: [0.0, 1.0] for name in PARAMETER_NAMES}, 'log_k0_cat': [0.0, 1.0]})
  with pytest.raises(ValueError, match='volume fraction'):
    _detector(objective_fraction=1.5)
  # The design bounds admit the closed [0, 1] but nothing outside it, and must increase.
  with pytest.raises(ValueError, match='volume fraction'):
    _detector(enzyme_fraction_bounds=[0.0, 1.5])
  with pytest.raises(ValueError, match='increasing'):
    _detector(enzyme_fraction_bounds=[0.9, 0.1])
  with pytest.raises(ValueError, match='half_time_bounds'):
    _detector(half_time_bounds=[2.0, 1.0])


# The stiffest corner scripts/check_stability.py finds in this prior: a high-K_M, tightly
# product-inhibited, steeply Arrhenius enzyme that stays folded to the top of the melting range.
STIFFEST = {
  'log_K0_A': -1.61, 'Q10_A': 1.904, 'log_K0_B': 0.0, 'Q10_B': 0.937,
  'log_K0i_C': -3.91, 'Q10_C': 0.7, 'log_K0i_D': -1.2, 'Q10_D': 1.54,
  'Q10_cat': 2.6, 'delta_H': 1.0e5, 'delta_C': 4015.0, 'T_melting': 58.0,
}


def test_rkc2_stability_boundary_matches_theory():
  """The measured real-axis boundary of the implemented step, in float32, against RKC2 theory.

  For s stages the damped Chebyshev stability polynomial reaches about `(2/3)(s^2 - 1)` on the
  negative real axis -- 20x further than explicit Euler's 2 at five stages. Measuring it (rather
  than trusting the formula) is the point: the recursion is evaluated in float32 here."""
  from scripts.check_stability import measure_boundary

  detector = _detector()
  boundary = measure_boundary(detector)
  theory = (2.0 / 3.0) * (detector.n_stages ** 2 - 1)
  assert 0.8 * theory <= boundary <= 1.25 * theory, f'boundary {boundary:.2f} vs theory {theory:.2f}'
  assert boundary > 2.0  # ... and it must beat explicit Euler, or the stages buy nothing


def test_configured_step_is_inside_the_stability_boundary():
  """The configured step must keep `dt * |df/dx|` inside the scheme's measured boundary at the
  stiffest corner of the prior, with margin -- past the boundary the extent of reaction runs beyond
  its physical maximum and never returns, so this is not an accuracy margin."""
  from scripts.check_stability import measure_boundary

  detector = _detector()
  parameters = {name: jnp.asarray(value, jnp.float32) for name, value in STIFFEST.items()}
  # The fastest half-time the calibration may ask for is the stiffest.
  log_k0_cat = detector._calibrate(parameters, jnp.asarray(detector.half_time_bounds[0], jnp.float32))
  A0, B0, E0 = detector._initial_state(detector.calibration_fraction)
  extent = jnp.minimum(A0, B0) * jnp.linspace(0.0, 1.0 - 1e-4, 256, dtype=jnp.float32)
  d_rate = jax.grad(lambda x, T: detector._rate(x, A0, B0, E0, T, parameters, log_k0_cat))
  decay = jax.vmap(lambda T: jnp.max(jnp.abs(jax.vmap(lambda x: d_rate(x, T))(extent))))(_temperatures(detector))

  boundary = measure_boundary(detector)
  for dt in (detector.objective_dt, detector.measurement_dt):
    z = dt * float(jnp.max(decay))
    assert z <= 0.8 * boundary, f'z = dt*|df/dx| = {z:.2f} too close to the boundary {boundary:.2f}'


def test_stiffest_corner_stays_physical():
  """At that corner, both integrations must land inside the physical extent range -- a diverged
  Euler run overshoots it by orders of magnitude."""
  detector = _detector()
  parameters = {name: jnp.asarray(value, jnp.float32) for name, value in STIFFEST.items()}
  log_k0_cat = detector._calibrate(parameters, jnp.asarray(detector.half_time_bounds[0], jnp.float32))
  A0, B0, E0 = detector._initial_state(detector.calibration_fraction)
  extent_max = float(jnp.minimum(A0, B0))
  hot = detector.temperature_bounds[0] + 0.95 * (STIFFEST['T_melting'] - detector.temperature_bounds[0])
  for n_steps, n_intervals in ((detector.n_objective_steps, 1),
                               (detector.n_steps_per_measurement, detector.n_measurements)):
    extents, error = detector._integrate(A0, B0, E0, hot, parameters, log_k0_cat,
                                     n_steps=n_steps, n_intervals=n_intervals)
    final = float(extents[-1])
    assert 0.0 <= final <= extent_max * (1.0 + 1e-3), f'extent {final} outside [0, {extent_max}]'
    assert float(error) <= detector.integration_tolerance


def test_calibrated_turnover_is_hexokinase_like():
  """The E stock is chosen so the CALIBRATED turnover is physical: k_cat at the optimal temperature
  should sit in hexokinase's 30-300 /s band rather than orders of magnitude off it."""
  from detopt.detector.enzyme import vant_hoff

  detector = _detector()

  def turnover(index):
    key_parameters, key_half_time, _ = jax.random.split(jax.random.PRNGKey(index), 3)
    parameters = detector._draw_parameters(key_parameters)
    low, high = detector.half_time_bounds
    half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=np.log(low), maxval=np.log(high)))
    log_k0_cat = detector._calibrate(parameters, half_time)
    optimal, _ = detector._optimal_temperature(parameters, log_k0_cat)
    return vant_hoff(optimal, log_k0_cat, parameters['Q10_cat']) / 3600.0  # per second

  k_cat = np.asarray(jax.jit(jax.vmap(turnover))(jnp.arange(128, dtype=jnp.int32)))
  assert 10.0 < np.median(k_cat) < 1000.0, f'median k_cat(T*) = {np.median(k_cat):.3g}/s is not hexokinase-like'


# ---------------------------------------------------------------------------- #
# The meta-regressor on this detector: the set regressor config/enzyme.yaml uses (one element per
# experiment) and the flat-MLP ablation.
# ---------------------------------------------------------------------------- #
def _set_config(**override):
  return {'set-regressor': dict({'features': [[16, 8], [16, 8]]}, **override)}


def test_set_regressor_shapes(seed):
  detector = _detector()
  regressor = from_config(detector, config=_set_config(), rngs=nnx.Rngs(seed))
  assert regressor.ensemble() is None
  assert regressor.n_features_in == detector.n_measurements + 2  # one element per experiment

  _, event, mask, target = detector(_design(detector), np.arange(8))
  features = detector.combine(event, detector.to_nominal(detector.to_scaled(_design(detector))), mask=mask)
  predicted = regressor(features, detector.element_mask(event, mask), deterministic=True)
  assert predicted.shape == (8, detector.target_dim())
  per_sample = regressor.loss(detector.loss, features, mask, detector.normalize_target(target), deterministic=True)
  assert per_sample.shape == (8,)


def test_set_regressor_is_permutation_invariant_over_experiments(seed):
  """The batch is an unordered set of (condition, readout) pairs -- reordering the experiments cannot
  move the estimate. The set aggregation gives that for free; the flat MLP does not (it is the
  ablation), which is what makes the comparison meaningful."""
  detector = _detector()
  set_regressor = from_config(detector, config=_set_config(), rngs=nnx.Rngs(seed))
  mlp = from_config(detector, config={'mlp-regressor': {'features': [16, 16]}}, rngs=nnx.Rngs(seed))

  _, event, mask, _ = detector(_design(detector), np.arange(8))
  features = detector.combine(event, detector.to_nominal(detector.to_scaled(_design(detector))), mask=mask)
  elements = detector.element_mask(event, mask)
  order = np.roll(np.arange(detector.n_experiments), 1)

  invariant = np.asarray(set_regressor(features, elements, deterministic=True))
  reordered = np.asarray(set_regressor(features[:, order], elements[:, order], deterministic=True))
  assert np.allclose(invariant, reordered, atol=1e-5)
  assert not np.allclose(
    np.asarray(mlp(features, elements, deterministic=True)),
    np.asarray(mlp(features[:, order], elements[:, order], deterministic=True))
  )


def test_mlp_regressor_shapes(seed):
  detector = _detector()
  regressor = from_config(detector, config={'mlp-regressor': {'features': [16, 16]}}, rngs=nnx.Rngs(seed))
  assert regressor.ensemble() is None
  assert regressor.n_inputs == detector.n_experiments * (detector.n_measurements + 2)

  _, event, mask, target = detector(_design(detector), np.arange(8))
  features = detector.combine(event, detector.to_nominal(detector.to_scaled(_design(detector))), mask=mask)
  predicted = regressor(features, detector.element_mask(event, mask), deterministic=True)
  assert predicted.shape == (8, detector.target_dim())
  per_sample = regressor.loss(detector.loss, features, mask, detector.normalize_target(target), deterministic=True)
  assert per_sample.shape == (8,)


def test_mlp_regressor_ensemble(seed):
  """`n_models=n` stacks n independent members on a leading axis, each seeing its own slice."""
  detector = _detector()
  regressor = from_config(
    detector, config={'mlp-regressor': {'features': [16], 'n_models': 3}}, rngs=nnx.Rngs(seed)
  )
  assert regressor.ensemble() == 3
  rng = np.random.default_rng(seed)
  shape = (3, 8) + detector.combined_event_shape()
  features = jnp.asarray(rng.standard_normal(shape), jnp.float32)
  mask = jnp.ones(shape[:-1], jnp.int32)
  predicted = regressor(features, mask, deterministic=True)
  assert predicted.shape == (3, 8, detector.target_dim())
  # Members are independent: the same input gives different predictions.
  same = jnp.broadcast_to(features[0][None], shape)
  members = np.asarray(regressor(same, mask, deterministic=True))
  assert not np.allclose(members[0], members[1])


def test_mlp_regressor_ignores_masked_elements(seed):
  detector = _detector()
  regressor = from_config(detector, config={'mlp-regressor': {'features': [16, 16]}}, rngs=nnx.Rngs(seed))
  rng = np.random.default_rng(seed)
  features = jnp.asarray(rng.standard_normal((4,) + detector.combined_event_shape()), jnp.float32)
  mask = jnp.asarray([[1, 1, 0, 0]] * 4, jnp.int32)
  polluted = features.at[:, 2:].add(100.0)  # only masked-out elements change
  assert np.allclose(
    np.asarray(regressor(features, mask, deterministic=True)),
    np.asarray(regressor(polluted, mask, deterministic=True))
  )
