"""Bacterial growth: cardinal temperatures from a batch of cultures, as a design-of-experiments task.

The benchmark asks the same question as the enzyme detector -- design a whole batch of experiments at
once, then measure how well a regressor trained on that batch's read-out recovers a property of the
organism -- but on a system whose response varies smoothly over the entire design range instead of
through a 5-15 K unfolding cliff:

* **design** -- ``n_experiments`` batch cultures. Each is given its culture TEMPERATURE, its
  INOCULUM (the optical density at t = 0) and its initial SUBSTRATE (g/L of carbon source). Batch,
  not fed-batch: everything is supplied at t = 0 and nothing is fed afterwards, so a design is a few
  numbers rather than a feeding profile.
* **event** -- one STRAIN drawn from a mesophile-to-moderate-thermophile prior, plus its read-out
  noise: ``n_measurements`` optical densities per culture on a fixed time grid, left-CENSORED at the
  reader's detection limit. The draw depends on ``event_index`` ALONE, so the same strain is
  re-measured under every design (common random numbers) and the target is a property of the event.
* **target** -- that strain's optimal growth temperature ``T_opt``, one scalar. A drawn parameter, so
  no design can move its own label.
* **loss** -- squared error of the predicted ``T_opt``, normalised by its prior range.

Why one scalar target does not collapse the design (as ``T_melting`` does on the enzyme, where one
well-placed probe suffices): the growth rate ``mu(T)`` is a broad, strongly ASYMMETRIC curve -- a
gradual rise over 25-35 K from ``T_min`` to ``T_opt``, then a collapse over 5-8 K to ``T_max`` -- so a
SINGLE culture yields one rate and cannot locate the peak at all: the same rate is consistent with the
rising flank of a warm strain or the falling flank of a cool one, and ``mu_opt`` (which varies
severalfold across strains) scales the whole curve and must be divided out by COMPARING temperatures.
Locating ``T_opt`` is a bracketing problem, and a culture that fails to grow is not wasted -- it bounds
``T_max`` (or ``T_min``) from one side. Only a batch whose cultures ALL fail to grow, or all read below
the detection limit, is genuinely uninformative.

Models
------
**Secondary model, the temperature response: CTMI** (Rosso et al.), four parameters with direct
biological meaning, which is why predictive microbiology uses it::

                     mu_opt (T - T_max) (T - T_min)^2
    mu(T) = -------------------------------------------------------------------
            (T_opt - T_min) [ (T_opt - T_min)(T - T_opt)
                              - (T_opt - T_max)(T_opt + T_min - 2T) ]

zero outside ``(T_min, T_max)``. The cardinal temperatures are NOT drawn independently -- they are
strongly correlated in nature and independent draws would produce impossible strains (``T_max <
T_opt``, or a 40 K collapse). ``T_opt`` is drawn over the population span and the two OFFSETS
``T_opt - T_min`` and ``T_max - T_opt`` from their own narrow priors.

**Primary model, the culture itself: MONOD batch kinetics**, integrated rather than the closed-form
logistic, because a real batch culture grows at essentially full rate until the substrate is nearly
exhausted and then stops fairly abruptly::

    dN/dt =  mu(T) S/(K_S + S) N          biomass, substrate-limited
    dS/dt = -(1/Y) mu(T) S/(K_S + S) N    substrate consumed at the yield Y

Temperature enters the RATE only, never the yield, so the plateau reports ``S0 Y`` and the approach
reports ``mu(T)`` -- a separation the design can exploit. ``K_S`` also makes the SUBSTRATE axis carry
more than the plateau height: the length of the exponential phase depends on ``S0/K_S``, and a culture
started below ``K_S`` never reaches ``mu(T)`` at all.

Numerics
--------
The two states are redundant: ``d(N + Y S)/dt = 0``, so ``N + Y S = N0 + Y S0 =: C`` for all time and
the system reduces to ONE state, exactly as the enzyme's 1:1:1:1 stoichiometry reduces to the extent
of reaction::

    dN/dt = mu(T) * u/(K_S + u) * N,      u = max(C - N, 0) / Y

The ``max`` is the conservation law, not a fudge: the substrate cannot go negative, and enforcing that
in the rate keeps a step that would overshoot the plateau from turning into unbounded growth.

The integrator is **RKC2** (:func:`detopt.detector.enzyme.rkc2_coefficients`, imported rather than
re-derived) with the same two-chain error monitor: every solve runs a ``dt`` chain and a ``dt/2``
chain over the same read-out times, RETURNS THE ``dt`` ONE, and reports the largest
``|fine - coarse|`` over those times; :meth:`GrowthDetector.__call__` asserts that against
``integration_tolerance`` host-side, because a jitted kernel cannot raise.

What sets the step here is ACCURACY, not stability. The stiff feature is substrate exhaustion: for
``u << K_S`` the equation linearises to a logistic with rate ``mu C / (Y K_S)``, i.e. a transient of
width ``tau = Y K_S / (mu C)`` -- 25 s at the sharpest corner of the prior x box (``mu_opt`` 2.5/h,
``K_S`` 0.02 g/L, ``Y`` 1.4, ``C`` 1.6 OD) -- during which the last ``Y K_S`` of biomass is made. A
step much longer than ``tau`` does not resolve the elbow and errs by up to that ``Y K_S`` (0.028 OD
at that corner), which is why the step is set from ``tau`` and the two-chain monitor is the thing
that verifies it. The stability boundary is far away by comparison: ``|df/dN| <= mu C / (Y K_S)`` is
143/h there, so ``z = |df/dN| dt = 0.89`` against a MEASURED float32 boundary of 6.18 at three
stages -- a 6.9x margin, the reverse of the enzyme, where stability was binding.

The corner that makes ``tau`` smallest is NOT the corner that makes the two-chain error largest: it
saturates before the first read-out, so both chains agree there. The binding cultures are the ones
whose elbow lands BETWEEN read-out times, and the step count is set from the error's tail over the
whole prior x box (see ``config/detector/growth.yaml``), not from this corner.
"""

import math
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .common import Detector
from .enzyme import rkc2_chain, rkc2_coefficients, rkc2_step
from ..utils import tensor

__all__ = [
  'cardinal_rate', 'PARAMETER_NAMES', 'GrowthDetector', 'GrowthDesign', 'GrowthEvent', 'GrowthTarget',
  'GrowthGroundTruth'
]

# The strain parameters DRAWN per event, in the order they are packed into the ground truth. The two
# cardinal OFFSETS are drawn rather than T_min and T_max themselves: the three cardinal temperatures
# are strongly correlated in nature (the collapse above the optimum is always sharp, the rise below it
# always gradual), and independent draws would produce strains that do not exist.
PARAMETER_NAMES = (
  'T_opt',          # optimal growth temperature (C) -- THE TARGET
  'delta_min',      # T_opt - T_min (K), the gradual lower flank
  'delta_max',      # T_max - T_opt (K), the sharp upper flank
  'log_mu_opt',     # log of the growth rate at T_opt (1/h)
  'log_K_S',        # log of the Monod half-saturation constant (g/L)
  'biomass_yield'   # OD units of biomass per g/L of substrate consumed
)


def cardinal_rate(temperature, T_min, T_opt, T_max, mu_opt):
  """CTMI: the specific growth rate (1/h) at ``temperature``, zero outside ``(T_min, T_max)``.

  Both branches are evaluated under ``jnp.where``, so the denominator is replaced by 1 outside the
  growth window rather than divided by -- outside it the expression has no meaning and can vanish.
  """
  inside = (temperature > T_min) & (temperature < T_max)
  numerator = (temperature - T_max) * jnp.square(temperature - T_min)
  denominator = (T_opt - T_min) * (
    (T_opt - T_min) * (temperature - T_opt) - (T_opt - T_max) * (T_opt + T_min - 2.0 * temperature)
  )
  return jnp.where(inside, mu_opt * numerator / jnp.where(inside, denominator, 1.0), 0.0)


def _scale(values, bounds):
  """NOMINAL ``[low, high]`` -> SCALED ``[0, 1]``, linearly."""
  low, high = bounds
  return (values - low) / (high - low)


def _unscale(values, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL ``[low, high]`` (inverse of :func:`_scale`)."""
  low, high = bounds
  return values * (high - low) + low


def _log_scale(values, bounds):
  """NOMINAL ``[low, high]`` -> SCALED ``[0, 1]``, LOG-linearly (for a decades-spanning quantity)."""
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return (jnp.log(values) - low) / (high - low)


def _log_unscale(values, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL ``[low, high]``, LOG-linearly (inverse of :func:`_log_scale`)."""
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return jnp.exp(values * (high - low) + low)


class GrowthDesign(NamedTuple):
  """One batch of cultures: per culture, the temperature it is incubated at (C), the inoculum it is
  started from (OD) and the substrate it is given (g/L). All three NOMINAL (physical)."""
  temperature: jax.Array  # (n_experiments,)
  inoculum: jax.Array     # (n_experiments,)
  substrate: jax.Array    # (n_experiments,)


class GrowthEvent(NamedTuple):
  """The batch's read-out: the optical density of every culture at every read-out time, with noise and
  the reader's detection limit already applied."""
  measurements: jax.Array  # (n_experiments, n_measurements)


class GrowthTarget(NamedTuple):
  """What the regressor predicts: the strain's optimal growth temperature (C). A DRAWN parameter, so
  it is a property of the strain alone and no design can move its own label."""
  optimal_temperature: jax.Array  # (1,)


class GrowthGroundTruth(NamedTuple):
  """The drawn strain itself (== conditioning): its parameters in ``PARAMETER_NAMES`` order. The
  target's ``T_opt`` is already one of them."""
  parameters: jax.Array  # (len(PARAMETER_NAMES),)


class GrowthDetector(Detector):
  """Single-batch design of ``n_experiments`` bacterial cultures (see the module docstring).

  Every constant is a constructor argument, i.e. lives in the yaml config: the run length and its
  read-out, the strain prior, the design bounds, and the resolution of the integration. Biomass is
  optical density (OD600), substrate g/L, time hours, temperature degrees Celsius.
  """

  def __init__(
    self, *,
    n_experiments: int,
    n_measurements: int,
    parameters: dict,
    duration: float,
    measurement_noise: float,
    detection_limit: float,
    temperature_bounds: tuple,
    inoculum_bounds: tuple,
    substrate_bounds: tuple,
    n_steps_per_measurement: int,
    n_stages: int,
    damping: float,
    integration_tolerance: float
  ):
    self.n_experiments = int(n_experiments)
    self.n_measurements = int(n_measurements)
    self.duration = float(duration)
    self.measurement_noise = float(measurement_noise)
    self.detection_limit = float(detection_limit)
    self.temperature_bounds = (float(temperature_bounds[0]), float(temperature_bounds[1]))
    self.inoculum_bounds = (float(inoculum_bounds[0]), float(inoculum_bounds[1]))
    self.substrate_bounds = (float(substrate_bounds[0]), float(substrate_bounds[1]))
    self.n_steps_per_measurement = int(n_steps_per_measurement)
    self.n_stages = int(n_stages)
    self.damping = float(damping)
    self.integration_tolerance = float(integration_tolerance)
    # RKC2 coefficients: static floats, folded into the jaxpr as constants.
    self._rkc2 = rkc2_coefficients(self.n_stages, self.damping)

    missing, unknown = set(PARAMETER_NAMES) - set(parameters), set(parameters) - set(PARAMETER_NAMES)
    if len(missing) > 0 or len(unknown) > 0:
      raise ValueError(
        f'the prior must give a range for every drawn parameter: missing {sorted(missing)}, unknown {sorted(unknown)}'
      )
    # Drawn uniformly within each range, in PARAMETER_NAMES order; a `log_*` range is therefore
    # log-uniform on the quantity itself.
    self._parameter_ranges = tuple(
      (name, (float(parameters[name][0]), float(parameters[name][1]))) for name in PARAMETER_NAMES
    )
    for name, (low, high) in self._parameter_ranges:
      if not low < high:
        raise ValueError(f'the prior range of {name} must be an increasing (low, high), got ({low}, {high})')
    prior = dict(self._parameter_ranges)
    for name in ('delta_min', 'delta_max'):
      if not prior[name][0] > 0.0:
        raise ValueError(f'{name} is a cardinal-temperature OFFSET and must be strictly positive, got {prior[name]}')
    if not prior['biomass_yield'][0] > 0.0:
      raise ValueError(f'biomass_yield must be strictly positive, got {prior["biomass_yield"]}')

    # The target's own range: T_opt is the drawn parameter the regressor infers, so the loss is scaled
    # by this rather than by the design's temperature range (see `normalize_target`).
    self.optimum_bounds = prior['T_opt']
    for name, bounds in (
      ('temperature_bounds', self.temperature_bounds), ('inoculum_bounds', self.inoculum_bounds),
      ('substrate_bounds', self.substrate_bounds)
    ):
      if not bounds[0] < bounds[1]:
        raise ValueError(f'{name} must be an increasing (low, high), got {bounds}')
    for name, bounds in (('inoculum_bounds', self.inoculum_bounds), ('substrate_bounds', self.substrate_bounds)):
      # Both are LOG-scaled onto the unit cube (they span decades), so neither bound may be zero.
      if not bounds[0] > 0.0:
        raise ValueError(f'{name} is log-scaled and must be strictly positive, got {bounds}')

    # The largest optical density any design in the box can reach on any strain in the prior: the
    # carrying capacity is N0 + Y*S0 by conservation. Used to put the read-out on a fixed scale.
    self.od_ceiling = float(self.inoculum_bounds[1] + prior['biomass_yield'][1] * self.substrate_bounds[1])
    # The read-out times of one culture: evenly spaced, ending at `duration` (t = 0 carries no
    # information -- the OD there is the design's own inoculum).
    self.measurement_times = (self.duration / self.n_measurements) * jnp.arange(
      1, self.n_measurements + 1, dtype=jnp.float32
    )
    self._generate = jax.jit(jax.vmap(self._event))

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return GrowthEvent(measurements=jax.ShapeDtypeStruct((self.n_experiments, self.n_measurements), np.float32))

  def target_spec(self):
    return GrowthTarget(optimal_temperature=jax.ShapeDtypeStruct((1,), np.float32))

  def ground_truth_spec(self):
    return GrowthGroundTruth(parameters=jax.ShapeDtypeStruct((len(PARAMETER_NAMES),), np.float32))

  def design_shape(self):
    return (3 * self.n_experiments,)

  def design_spec(self):
    f = jax.ShapeDtypeStruct((self.n_experiments,), np.float32)
    return GrowthDesign(temperature=f, inoculum=f, substrate=f)

  def design_bounds(self):
    return {
      'temperature': self.temperature_bounds, 'inoculum': self.inoculum_bounds, 'substrate': self.substrate_bounds
    }

  def combined_event_shape(self, design: bool = True):
    # element == culture; its features are its own OD samples + its own (T, N0, S0)
    return (self.n_experiments, self.n_measurements + (3 if design else 0))

  def size(self):
    return None  # an analytic source: every index is a fresh strain

  # ------------------------------------------------------------------ #
  # Design scaling: temperature LINEARLY, the two concentrations LOG-linearly, each from its own
  # bounds. The inoculum and the substrate span decades and the response depends on log(S0/K_S), so a
  # linear map would compress the informative low-substrate region into a sliver of the unit cube.
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    d = jnp.asarray(design, jnp.float32)
    n = self.n_experiments
    return jnp.concatenate([
      _scale(d[..., :n], self.temperature_bounds),
      _log_scale(d[..., n:2 * n], self.inoculum_bounds),
      _log_scale(d[..., 2 * n:3 * n], self.substrate_bounds)
    ], axis=-1)

  def _to_nominal_flat(self, design_scaled):
    e = jnp.asarray(design_scaled, jnp.float32)
    n = self.n_experiments
    return jnp.concatenate([
      _unscale(e[..., :n], self.temperature_bounds),
      _log_unscale(e[..., n:2 * n], self.inoculum_bounds),
      _log_unscale(e[..., 2 * n:3 * n], self.substrate_bounds)
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
    """``features (..., n_experiments, n_measurements + 3)``: each culture's OD samples followed by its
    own three design values, taken STRAIGHT from the scaled design -- already each coordinate on its
    own range in [0, 1], which is what the network wants.

    The OD samples go in on a LOG axis: growth curves are exponential, the quantity a rate estimate is
    the slope of, and the read-out is censored at the detection limit, so the log is bounded below by
    construction. ``mask`` is unused -- every culture of the batch is real (the element axis is the
    design's, not a hit count).

    WITH THE DESIGN WITHHELD -- ``reveal_design=False`` or ``design_scaled=None``, treated alike since
    a culture is grown before it is combined -- the trailing three columns are simply absent and the
    OD samples are unchanged: they are scaled by the FIXED detection-limit-to-ceiling range, not by
    anything the design sets, so nothing leaks back. The network is told WHAT was read and not UNDER
    WHICH conditions. The features are narrower, not corrupted."""
    logarithm = jnp.log(jnp.maximum(event.measurements, self.detection_limit))
    measurements = self._to_unit(logarithm, (math.log(self.detection_limit), math.log(self.od_ceiling)))
    if design_scaled is None or not reveal_design:
      return measurements
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    if design_scaled.ndim == 1:  # one design for the whole event batch
      design_scaled = jnp.broadcast_to(design_scaled[None, :], event.measurements.shape[:-2] + design_scaled.shape)
    n = self.n_experiments
    return jnp.concatenate([
      measurements, design_scaled[..., :n, None], design_scaled[..., n:2 * n, None], design_scaled[..., 2 * n:3 * n, None]
    ], axis=-1)

  def element_mask(self, event, mask):
    return mask  # element == culture

  @staticmethod
  def _to_unit(values, bounds):
    """Map ``[low, high]`` linearly onto ``[-1, 1]``."""
    low, high = bounds
    return (2.0 * values - (low + high)) / (high - low)

  def normalize_target(self, target):
    """By the PRIOR range of ``T_opt``, not by ``temperature_bounds``: the prior spans 25 C of the 45 C
    design range, and scaling by the latter would shrink the target's variance for free. Scaled by its
    own range the target is O(1) (variance 1/3 for a uniform prior), which is what makes
    ``loss_precision`` mean the same thing here as on every other benchmark in this repo."""
    flat, _ = tensor.flatten(target)
    return self._to_unit(flat, self.optimum_bounds)

  def denormalize_predictions(self, normalised):
    low, high = self.optimum_bounds
    physical = 0.5 * (jnp.asarray(normalised, jnp.float32) * (high - low) + (low + high))
    return tensor.unflatten(tensor.structure(self.target_spec()), physical)

  def normalize_ground_truth(self, ground_truth):
    """Physical ``GrowthGroundTruth`` -> standardised flat ``(..., len(PARAMETER_NAMES))``: every
    parameter by its own prior range."""
    low = jnp.asarray([r[0] for _, r in self._parameter_ranges], jnp.float32)
    high = jnp.asarray([r[1] for _, r in self._parameter_ranges], jnp.float32)
    return (2.0 * ground_truth.parameters - (low + high)) / (high - low)

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    low, high = self.optimum_bounds
    return f'MSE (optimal growth temperature / {0.5 * (high - low):.1f} C)'

  def metric_labels(self):
    return ('loss', 'optimal_temperature')

  def loss(self, predicted, target):
    return jnp.mean(jnp.square(predicted - target), axis=-1)

  def metric(self, predicted, target):
    squared = jnp.square(predicted - target)
    return {'loss': jnp.mean(squared, axis=-1), 'optimal_temperature': squared[..., 0]}

  def metric_real_rmse(self, metric_means):
    """Sample-averaged normalised ``optimal_temperature`` metric (from :meth:`metric`) -> real-unit
    RMSE. ``loss`` has no single unit and is omitted."""
    low, high = self.optimum_bounds
    return {'optimal_temperature': (float(np.sqrt(metric_means['optimal_temperature']) * 0.5 * (high - low)), 'C')}

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """Draw the strains at ``event_index`` and run each one's batch of cultures under ``design`` (one
    design broadcast over the batch, or one design per event). DETERMINISTIC: a strain and its
    read-out noise are seeded from ``event_index`` alone, so the same event under two designs is the
    same strain -- the target is a property of the event only."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    temperature, inoculum, substrate = self._resolve_design(design, n)
    measurements, optimum, parameters, error = self._generate(
      temperature, inoculum, substrate, jnp.asarray(event_index, jnp.int32)
    )

    error = float(jnp.max(error))
    if not math.isfinite(error) or error > self.integration_tolerance:
      raise RuntimeError(
        f'integration error {error:.3g} OD over {n} events exceeds integration_tolerance '
        f'{self.integration_tolerance:.3g} OD -- the largest disagreement between the dt and dt/2 '
        f'chains on the biomass at a read-out time; raise n_steps_per_measurement (the substrate '
        f'exhaustion elbow has width Y*K_S/(mu*C), and the step must resolve it).'
      )

    ground_truth = GrowthGroundTruth(parameters=parameters)
    mask = jnp.ones((n, self.n_experiments), jnp.int32)
    return ground_truth, GrowthEvent(measurements=measurements), mask, GrowthTarget(optimal_temperature=optimum[:, None])

  def _resolve_design(self, design, n):
    """``GrowthDesign`` / config ``Mapping`` / flat array -> ``(temperature, inoculum, substrate)``,
    each ``(n, n_experiments)`` (a single design is broadcast over the event batch)."""
    width = 3 * self.n_experiments
    flat = jnp.broadcast_to(jnp.reshape(self.flatten_design(design), (-1, width)), (n, width))
    m = self.n_experiments
    return flat[:, :m], flat[:, m:2 * m], flat[:, 2 * m:]

  def _event(self, temperature, inoculum, substrate, event_index):
    """One event: draw a strain and run its batch. ``temperature`` / ``inoculum`` / ``substrate`` are
    ``(n_experiments,)``; every draw uses ``event_index`` only."""
    key_parameters, key_noise = jax.random.split(jax.random.PRNGKey(event_index), 2)
    parameters = self._draw_parameters(key_parameters)
    measurements, error = self._run_batch(temperature, inoculum, substrate, parameters, key_noise)
    packed = jnp.stack([parameters[name] for name in PARAMETER_NAMES])
    return measurements, parameters['T_opt'], packed, error

  def _draw_parameters(self, key):
    """One strain: every parameter uniform within its own prior range (so a ``log_*`` parameter is
    log-uniform on the quantity itself)."""
    keys = jax.random.split(key, len(self._parameter_ranges))
    return {
      name: jax.random.uniform(k, (), minval=low, maxval=high)
      for (name, (low, high)), k in zip(self._parameter_ranges, keys)
    }

  def _run_batch(self, temperature, inoculum, substrate, parameters, key):
    """The batch's read-out: the optical density of every culture at every read-out time, plus
    independent ``N(0, measurement_noise)`` reader noise, LEFT-CENSORED at the detection limit.

    The censoring is not a detail: a culture that never reaches the limit reads flat at it, which is
    exactly what a plate reader reports and is how a "no growth" result enters the data as a BOUND on
    the cardinal temperatures rather than as a missing value."""
    T_min = parameters['T_opt'] - parameters['delta_min']
    T_max = parameters['T_opt'] + parameters['delta_max']
    mu_opt = jnp.exp(parameters['log_mu_opt'])
    half_saturation = jnp.exp(parameters['log_K_S'])
    biomass_yield = parameters['biomass_yield']

    def run(culture_temperature, N0, S0):
      mu = cardinal_rate(culture_temperature, T_min, parameters['T_opt'], T_max, mu_opt)
      return self._integrate(N0, N0 + biomass_yield * S0, mu, half_saturation, biomass_yield)

    biomass, errors = jax.vmap(run)(temperature, inoculum, substrate)
    reading = biomass + self.measurement_noise * jax.random.normal(key, biomass.shape)
    return jnp.maximum(reading, self.detection_limit), jnp.max(errors)

  # ------------------------------------------------------------------ #
  # Integration
  # ------------------------------------------------------------------ #
  def rkc2_step(self, rate, biomass, dt):
    """One RKC2 step of size ``dt`` on ``dN/dt = rate(N)`` from ``biomass``, with this detector's own
    coefficients: :func:`detopt.detector.enzyme.rkc2_step`, which is what the bare name below resolves
    to (a method body does not see class scope). Kept as a method because the stability probes reach
    for it through the built detector."""
    return rkc2_step(self._rkc2, rate, biomass, dt)

  def _chain(self, rate, initial, *, dt, n_steps):
    """One RKC2 chain: ``n_measurements`` intervals of ``n_steps`` steps of ``dt``, returning the
    biomass at the END of every interval (the read-out times)."""
    return rkc2_chain(self._rkc2, rate, initial, dt=dt, n_steps=n_steps, n_intervals=self.n_measurements)

  def _integrate(self, inoculum, capacity, mu, half_saturation, biomass_yield):
    """RKC2 on the biomass over ``duration``, split into ``n_measurements`` equal intervals. Returns
    the biomass at the END of every interval (the read-out times) and the integration error, ESTIMATED
    INSIDE THE SOLVE.

    Two chains run over the same interval grid -- one at ``dt``, one at ``dt/2`` -- and the error is the
    largest ``|fine - coarse|`` over the interval ends. That is a GLOBAL error estimate on the quantity
    actually consumed downstream (the OD at the read-out times), not a per-step local truncation
    estimate: a local estimate can stay small while the accumulated trajectory drifts. The ``dt`` chain
    is what is RETURNED, paired with that error, so the solution handed out is the one the error was
    measured on. A jitted kernel cannot raise, so the assertion lives in :meth:`__call__`.

    The substrate is recovered from the conserved ``N + Y S = C`` and floored at zero: a step that
    would push the biomass past the carrying capacity leaves no substrate, hence no rate, which is
    what the physics says and what keeps an overshoot from compounding."""
    dt = self.duration / (self.n_measurements * self.n_steps_per_measurement)

    def rate(biomass):
      remaining = jnp.maximum(capacity - biomass, 0.0) / biomass_yield
      return mu * remaining / (half_saturation + remaining) * biomass

    coarse = self._chain(rate, inoculum, dt=dt, n_steps=self.n_steps_per_measurement)
    fine = self._chain(rate, inoculum, dt=0.5 * dt, n_steps=2 * self.n_steps_per_measurement)
    return coarse, jnp.max(jnp.abs(fine - coarse))
