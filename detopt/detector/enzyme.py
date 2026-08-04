"""Enzymatic reaction A + B -> C + D via E: single-batch design of experiments.

The DFG co-design proposal's bioprocess work package asks for the *initial batch* of a wet-lab
characterisation campaign to be designed as a whole -- conventional DoE/BO seeds that batch at
random. Cast into the detector contract, that benchmark runs on the same GP-BO driver and the
same design-conditioned meta-regressor as the SHiP spectrometer:

* **design** -- the batch of ``n_experiments`` initial experiments. Stock solutions of A, B and E
  are mixed at t = 0; the design gives, per experiment, the volume fraction taken by the enzyme
  stock and the temperature the experiment is run at. A and B equally fill the rest of the volume,
  so one fraction fixes the whole initial state.
* **event** -- one enzyme drawn from a deliberately wide "unspecified enzyme" prior, plus its
  readout noise: ``n_measurements`` noisy samples of [A] over each experiment. The draw depends on
  ``event_index`` ALONE, so the same enzyme is re-measured under every design (common random
  numbers), and the target below is a property of the event, never of the design.
* **target** -- that enzyme's optimal temperature: the temperature at which the reference
  1/3 E : 1/3 A : 1/3 B mixture reaches the highest conversion after ``duration``.
* **loss** -- squared error of the predicted optimal temperature. BO therefore searches for the
  batch of experiments from which the optimal temperature is best identified.

Turnover rises with temperature while the enzyme reversibly unfolds, so conversion peaks at an
interior temperature just below the denaturation edge -- the regressor has to locate that edge
from a handful of experiments, which it can only do if the design brackets it.

Numerics
--------
The 1:1:1:1 stoichiometry leaves a single degree of freedom, the extent of reaction ``x``:
``A = A0 - x``, ``B = B0 - x``, ``C = D = x``, with E constant. Experiments and the objective
integrate that one scalar over time, the whole event batch under one ``jax.jit``.

The integrator is **RKC2** -- the second-order Runge-Kutta-Chebyshev scheme (van der Houwen-Sommeijer
/ Verwer) -- because this problem is mildly stiff and explicit Euler is the wrong tool for it.
Product inhibition sets the scale: ``Kapp = K_M (1 + x/Ki)``, so the rate decays over an extent of
order ``Ki``, and the local decay rate ``|df/dx|`` reaches ~4800/h over this prior. Euler is stable
only while ``dt |df/dx| < 2``; RKC2 spends ``n_stages`` rate evaluations per step to push a damped
Chebyshev polynomial along the negative real axis instead, buying a stability boundary that grows
like ``s^2`` -- roughly an eight-fold longer step at five stages, for five evaluations, and second
order accuracy rather than first.

That boundary is **measured, not assumed**: ``scripts/check_stability.py`` applies the implemented
step to ``y' = -lambda y`` in float32 on the target device and finds the largest ``z = lambda dt``
with amplification ``|R(z)| <= 1`` (float32 roundoff in the Chebyshev recursion can eat into the
theoretical value, which is the whole reason to measure it), then searches the prior for the largest
``|df/dx|`` and reports the step this allows. Getting it wrong is not a matter of accuracy -- past
the boundary the extent runs beyond its physical maximum and never comes back.

The error is estimated **inside the solve**: every integration runs TWO chains over the same
read-out times, one at ``dt`` and one at ``dt/2``, RETURNS THE ``dt`` ONE, and reports the largest
``|fine - coarse|`` over those times -- the solution handed out is the one the error was measured on.
That is a global error on the quantity actually used downstream -- the extent at the read-out times,
and hence the conversion the target extremises -- rather than a per-step local truncation estimate,
which can stay small while the accumulated trajectory drifts. Outside the solve,
:meth:`EnzymeDetector.__call__` asserts on it against ``integration_tolerance`` host-side (a jitted
kernel cannot raise).

Neither temperature search needs a scan over temperatures: the calibration optimum and the target
both come from a golden-section search (:meth:`EnzymeDetector._extremum`), which the log-concavity of
the rate in ``T`` justifies. The calibration needs no time stepping at all either -- it is a
quadrature over ``x``.
"""

import math
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .common import Detector
from ..utils import tensor
from ..utils.encoding import uniform_to_normal_jax, normal_to_uniform_jax

### A + B -> C + D via E

ZERO_CELSIUS =  273.15
REFERENCE_TEMPERATURE = 10.0 + ZERO_CELSIUS
INV_TEMPERATURE_SPAN = 1 / ZERO_CELSIUS - 1 / REFERENCE_TEMPERATURE

__all__ = [
  'kinetics', 'vant_hoff', 'gibbs_fraction', 'PARAMETER_NAMES', 'rkc2_coefficients',
  'EnzymeDetector', 'EnzymeDesign', 'EnzymeEvent', 'EnzymeTarget', 'EnzymeGroundTruth'
]

def vant_hoff(T, log_K_0, Q10):
  """
  K_0 --- rate at 0C;
  Q10 - K_ref / K_0, K_ref - rate at the reference temperature, 10C;
  """
  delta = 1 / ZERO_CELSIUS - 1 / (T + ZERO_CELSIUS)
  return jnp.exp(
    log_K_0 + jnp.log(Q10) * delta / INV_TEMPERATURE_SPAN
  )

def gibbs_fraction(temperature, delta_H, delta_C, temperature_melting):
  """
  delta_H --- entalpy of unfolding;
  delta_C --- heat capacity change;
  delta H and delta C are assumed to be normalized by R.
  """
  T = temperature + ZERO_CELSIUS
  T_m = temperature_melting + ZERO_CELSIUS

  dG = delta_H * (1 - T / T_m) - \
      delta_C * ((T_m - T) - T * jnp.log(T_m / T))

  return jax.nn.sigmoid(dG / T)

def kinetics(A, B, C, D, E, temperature, parameters):
  K_A = vant_hoff(temperature, parameters['log_K0_A'], parameters['Q10_A'])
  K_B = vant_hoff(temperature, parameters['log_K0_B'], parameters['Q10_B'])
  Ki_C = vant_hoff(temperature, parameters['log_K0i_C'], parameters['Q10_C'])
  Ki_D = vant_hoff(temperature, parameters['log_K0i_D'], parameters['Q10_D'])

  Kapp_A = K_A * (1 + C / Ki_C)
  Kapp_B = K_B * (1 + D / Ki_D)

  ### actually Arrhenius, but the expression is the same
  k_cat = vant_hoff(temperature, parameters['log_k0_cat'], parameters['Q10_cat'])

  active_enzyme = gibbs_fraction(temperature, parameters['delta_H'], parameters['delta_C'], parameters['T_melting'])

  E_a = active_enzyme * E

  rate = k_cat * E_a * A * B / (A + Kapp_A) / (B + Kapp_B)

  return rate


# The kinetic parameters DRAWN per event, in the order they are packed into the ground truth.
# `log_k0_cat` is deliberately absent: the turnover scale is not drawn but CALIBRATED, so that every
# enzyme reacts on a measurable timescale (see EnzymeDetector._calibrate).
PARAMETER_NAMES = (
  'log_K0_A', 'Q10_A',      # Michaelis constant of A: log-value at 0 C (mM) + per-10C factor
  'log_K0_B', 'Q10_B',      # ... of B
  'log_K0i_C', 'Q10_C',     # competitive inhibition constant of the product C
  'log_K0i_D', 'Q10_D',     # ... of the product D
  'Q10_cat',                # turnover: only its temperature slope is drawn
  'delta_H', 'delta_C',     # unfolding enthalpy / heat-capacity change, both over R (K)
  'T_melting'               # melting temperature (C)
)

# Golden-section contraction factor: the bracket of EnzymeDetector._extremum shrinks by this per step.
GOLDEN_SECTION = 0.5 * (math.sqrt(5.0) - 1.0)



def rkc2_coefficients(n_stages, damping):
  """Verwer's RKC2 coefficients for ``n_stages`` stages and damping ``epsilon``.

  The scheme advances ``w_0 = y`` through a Chebyshev recursion whose stability polynomial is a
  shifted, damped Chebyshev polynomial of degree ``s``: real-axis stability then grows like ``s^2``
  instead of the fixed ``2`` of explicit Euler, at ``s`` rate evaluations per step. The damping
  ``epsilon`` lifts the polynomial off ``|R| = 1`` between its extrema, so the boundary is a strip
  rather than a set of touching points -- without it, a decay rate landing exactly on an extremum
  would sit marginally stable. ``epsilon = 2/13`` is Verwer's standard choice.

  Returns ``(mu_tilde_1, coefficients)``: the first stage's single coefficient, and a
  ``(n_stages - 1, 4)`` array of ``(mu, nu, mu_tilde, gamma_tilde)`` rows for stages ``j = 2..s``,
  laid out for a ``jax.lax.scan`` over the stages.
  """
  s = int(n_stages)
  if s < 2:
    raise ValueError(f'RKC2 needs at least 2 stages, got {s}')
  omega_0 = 1.0 + float(damping) / (s * s)

  # Chebyshev polynomials of the first kind and their first two derivatives at omega_0, by the
  # standard three-term recursions.
  T, dT, ddT = np.zeros(s + 1), np.zeros(s + 1), np.zeros(s + 1)
  T[0], T[1] = 1.0, omega_0
  dT[0], dT[1] = 0.0, 1.0
  ddT[0], ddT[1] = 0.0, 0.0
  for j in range(2, s + 1):
    T[j] = 2.0 * omega_0 * T[j - 1] - T[j - 2]
    dT[j] = 2.0 * T[j - 1] + 2.0 * omega_0 * dT[j - 1] - dT[j - 2]
    ddT[j] = 4.0 * dT[j - 1] + 2.0 * omega_0 * ddT[j - 1] - ddT[j - 2]
  omega_1 = dT[s] / ddT[s]

  b = np.zeros(s + 1)
  for j in range(2, s + 1):
    b[j] = ddT[j] / (dT[j] * dT[j])
  b[0] = b[1] = b[2]  # the conventional extension to the two starting values
  a = 1.0 - b * T

  rows = []
  for j in range(2, s + 1):
    mu_tilde = 2.0 * b[j] * omega_1 / b[j - 1]
    rows.append((2.0 * b[j] * omega_0 / b[j - 1], -b[j] / b[j - 2], mu_tilde, -a[j - 1] * mu_tilde))
  return float(b[1] * omega_1), jnp.asarray(rows, jnp.float32)


class EnzymeDesign(NamedTuple):
  """One batch of initial experiments: per experiment, the volume fraction taken by the enzyme
  stock solution and the temperature (C) the experiment is run at."""
  enzyme_fraction: jax.Array  # (n_experiments,)
  temperature: jax.Array      # (n_experiments,)


class EnzymeEvent(NamedTuple):
  """The batch's readout: the noisy [A] samples (mM) of every experiment."""
  measurements: jax.Array  # (n_experiments, n_measurements)


class EnzymeTarget(NamedTuple):
  """What the regressor predicts: the drawn enzyme's melting temperature (C), i.e. the midpoint of its
  thermal unfolding. A DRAWN parameter, read straight out of the prior -- no conversion is simulated to
  produce it, and it is a property of the enzyme alone, so no design can move its own label."""
  melting_temperature: jax.Array  # (1,)


class EnzymeGroundTruth(NamedTuple):
  """The drawn enzyme itself (== conditioning): its kinetic parameters in ``PARAMETER_NAMES`` order and
  the half-conversion time its turnover was calibrated to (h). The target's ``T_melting`` is already
  one of the parameters."""
  parameters: jax.Array  # (len(PARAMETER_NAMES),)
  half_time: jax.Array   # (1,)


class EnzymeDetector(Detector):
  """Single-batch design of ``n_experiments`` enzymatic experiments (see the module docstring).

  Every constant is a constructor argument, i.e. lives in the yaml config: the stock solutions, the
  experiment length and its readout, the parameter prior, the design bounds, and the resolution of
  each numerical procedure. Concentrations are mM and times hours throughout; temperatures are
  degrees Celsius (``kinetics`` converts internally).
  """

  def __init__(
    self, *,
    n_experiments: int,
    n_measurements: int,
    parameters: dict,
    concentration_A: float,
    concentration_B: float,
    concentration_E: float,
    duration: float,
    measurement_noise: float,
    enzyme_fraction_bounds: tuple,
    temperature_bounds: tuple,
    objective_fraction: float,
    calibration_fraction: float,
    half_time_bounds: tuple,
    n_temperature_steps: int,
    n_quadrature: int,
    n_steps_per_measurement: int,
    n_objective_steps: int,
    n_stages: int,
    damping: float,
    integration_tolerance: float
  ):
    self.n_experiments = int(n_experiments)
    self.n_measurements = int(n_measurements)
    self.concentration_A = float(concentration_A)
    self.concentration_B = float(concentration_B)
    self.concentration_E = float(concentration_E)
    self.duration = float(duration)
    self.measurement_noise = float(measurement_noise)
    self.enzyme_fraction_bounds = (float(enzyme_fraction_bounds[0]), float(enzyme_fraction_bounds[1]))
    self.temperature_bounds = (float(temperature_bounds[0]), float(temperature_bounds[1]))
    self.objective_fraction = float(objective_fraction)
    self.calibration_fraction = float(calibration_fraction)
    self.half_time_bounds = (float(half_time_bounds[0]), float(half_time_bounds[1]))
    self.n_temperature_steps = int(n_temperature_steps)
    self.n_quadrature = int(n_quadrature)
    self.n_steps_per_measurement = int(n_steps_per_measurement)
    self.n_objective_steps = int(n_objective_steps)
    self.n_stages = int(n_stages)
    self.damping = float(damping)
    self.integration_tolerance = float(integration_tolerance)
    # RKC2 coefficients: static floats, folded into the jaxpr as constants.
    self._rkc2 = rkc2_coefficients(self.n_stages, self.damping)

    missing, unknown = set(PARAMETER_NAMES) - set(parameters), set(parameters) - set(PARAMETER_NAMES)
    if len(missing) > 0 or len(unknown) > 0:
      raise ValueError(
        f'the prior must give a range for every drawn parameter: missing {sorted(missing)}, unknown '
        f'{sorted(unknown)} (`log_k0_cat` is calibrated, not drawn)'
      )
    # Drawn uniformly within each range, in PARAMETER_NAMES order; a `log_*` range is therefore
    # log-uniform on the quantity itself.
    self._parameter_ranges = tuple(
      (name, (float(parameters[name][0]), float(parameters[name][1]))) for name in PARAMETER_NAMES
    )
    for name, (low, high) in self._parameter_ranges:
      if not low < high:
        raise ValueError(f'the prior range of {name} must be an increasing (low, high), got ({low}, {high})')

    temperature_low, temperature_high = self.temperature_bounds
    # The target's own range: T_melting is the drawn parameter the regressor infers, so the loss is
    # scaled by this rather than by the design's temperature range (see `normalize_target`).
    self.melting_bounds = melting_low, melting_high = dict(self._parameter_ranges)['T_melting']
    if melting_low < temperature_low or melting_high > temperature_high:
      # Outside the scan an enzyme would be either dead or never denatured at every temperature,
      # putting the conversion peak (and the calibration optimum) beyond the design space.
      raise ValueError(
        f'the T_melting range ({melting_low}, {melting_high}) must lie within temperature_bounds '
        f'({temperature_low}, {temperature_high}) for the conversion peak to be inside the scan'
      )
    # The two REFERENCE mixtures must be strictly interior: they define the target and the turnover
    # calibration, and at 0 or 1 one of the reactants is absent, so the conversion they are the
    # extremum of is 0/0.
    for name, fraction in (
      ('objective_fraction', self.objective_fraction), ('calibration_fraction', self.calibration_fraction)
    ):
      if not 0.0 < fraction < 1.0:
        raise ValueError(f'{name} is a volume fraction and must lie strictly within (0, 1), got {fraction}')
    # The DESIGN bounds are only what can be pipetted, so the closed [0, 1] is allowed: the extremes
    # are useless experiments (no enzyme, or no substrate), not ill-defined ones, and it is the
    # optimiser's job to discover that rather than the detector's to forbid it.
    for index, bound in enumerate(self.enzyme_fraction_bounds):
      if not 0.0 <= bound <= 1.0:
        raise ValueError(f'enzyme_fraction_bounds[{index}] is a volume fraction and must lie within [0, 1], got {bound}')
    if not self.enzyme_fraction_bounds[0] < self.enzyme_fraction_bounds[1]:
      raise ValueError(f'enzyme_fraction_bounds must be an increasing pair, got {self.enzyme_fraction_bounds}')
    if not 0.0 < self.half_time_bounds[0] < self.half_time_bounds[1]:
      raise ValueError(f'half_time_bounds must be an increasing pair of positive times, got {self.half_time_bounds}')

    # The quadrature nodes of the calibration (as a fraction of the half-conversion extent). Built
    # here, eagerly -- never lazily inside a jitted method. Temperatures are not tabulated: both the
    # calibration optimum and the objective peak are found by golden-section search (`_extremum`).
    self._quadrature_nodes = jnp.linspace(0.0, 1.0, self.n_quadrature, dtype=jnp.float32)
    # The read-out times of one experiment: evenly spaced, ending at `duration` (t = 0 carries no
    # information -- [A] there is the design's own A0).
    self.measurement_times = (self.duration / self.n_measurements) * jnp.arange(1, self.n_measurements + 1, dtype=jnp.float32)
    # The uniform RKC2 step of each integration, both bounded by the measured stability boundary
    # (scripts/check_stability.py).
    self.objective_dt = self.duration / self.n_objective_steps
    self.measurement_dt = self.duration / (self.n_measurements * self.n_steps_per_measurement)
    self._generate = jax.jit(jax.vmap(self._event))

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return EnzymeEvent(measurements=jax.ShapeDtypeStruct((self.n_experiments, self.n_measurements), np.float32))

  def target_spec(self):
    return EnzymeTarget(melting_temperature=jax.ShapeDtypeStruct((1,), np.float32))

  def ground_truth_spec(self):
    f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
    return EnzymeGroundTruth(parameters=f(len(PARAMETER_NAMES)), half_time=f(1))

  def design_shape(self):
    return (2 * self.n_experiments,)

  def design_spec(self):
    f = jax.ShapeDtypeStruct((self.n_experiments,), np.float32)
    return EnzymeDesign(enzyme_fraction=f, temperature=f)

  def design_bounds(self):
    return {'enzyme_fraction': self.enzyme_fraction_bounds, 'temperature': self.temperature_bounds}

  def combined_event_shape(self):
    # element == experiment; its features are its own measurements + its own (E0, temperature)
    return (self.n_experiments, self.n_measurements + 2)

  def size(self):
    return None  # an analytic source: every index is a fresh enzyme

  # ------------------------------------------------------------------ #
  # Design encoding: independent per-field bounds <-> N(0, 1)
  # ------------------------------------------------------------------ #
  def _encode_flat(self, design):
    d = jnp.asarray(design, jnp.float32)
    n = self.n_experiments
    return jnp.concatenate([
      uniform_to_normal_jax(d[..., :n], *self.enzyme_fraction_bounds),
      uniform_to_normal_jax(d[..., n:2 * n], *self.temperature_bounds)
    ], axis=-1)

  def _decode_flat(self, encoded_design):
    e = jnp.asarray(encoded_design, jnp.float32)
    n = self.n_experiments
    return jnp.concatenate([
      normal_to_uniform_jax(e[..., :n], *self.enzyme_fraction_bounds),
      normal_to_uniform_jax(e[..., n:2 * n], *self.temperature_bounds)
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_encoded(self, event, encoded_design, mask=None):
    """``features (..., n_experiments, n_measurements + 2)``: each experiment's [A] samples followed
    by its own two design values. The design is decoded back to the physical space and mapped
    LINEARLY onto ~[-1, 1] -- the encoding's own erf scale saturates near the bounds, which would
    squash exactly the extreme designs BO wants to tell apart. ``mask`` is unused: every experiment
    of the batch is real (the element axis is the design's, not a hit count)."""
    encoded_design = jnp.asarray(encoded_design, jnp.float32)
    if encoded_design.ndim == 1:  # one design for the whole event batch
      encoded_design = jnp.broadcast_to(encoded_design[None, :], event.measurements.shape[:-2] + encoded_design.shape)
    physical = self._decode_flat(encoded_design)
    n = self.n_experiments
    # E0 = concentration_E * enzyme_fraction, so the normalised fraction IS the normalised initial
    # enzyme concentration.
    enzyme = self._to_unit(physical[..., :n], self.enzyme_fraction_bounds)
    heat = self._to_unit(physical[..., n:2 * n], self.temperature_bounds)
    # [A] never exceeds half the A stock (the other half of the non-enzyme volume is B).
    measurements = self._to_unit(event.measurements, (0.0, 0.5 * self.concentration_A))
    return jnp.concatenate([measurements, enzyme[..., None], heat[..., None]], axis=-1)

  def element_mask(self, event, mask):
    return mask  # element == experiment

  @staticmethod
  def _to_unit(values, bounds):
    """Map ``[low, high]`` linearly onto ``[-1, 1]``."""
    low, high = bounds
    return (2.0 * values - (low + high)) / (high - low)

  def normalize_target(self, target):
    """By the PRIOR range of ``T_melting``, not by ``temperature_bounds``. The prior spans 13 C of the
    100 C design range, so scaling by the latter would leave a target of variance 0.006 -- finer than
    the precision the objective is measured to, which would make every design indistinguishable by
    construction. Scaled by its own range the target is O(1) (variance 1/3 for a uniform prior)."""
    flat, _ = tensor.flatten(target)
    return self._to_unit(flat, self.melting_bounds)

  def denormalize_predictions(self, normalised):
    low, high = self.melting_bounds
    physical = 0.5 * (jnp.asarray(normalised, jnp.float32) * (high - low) + (low + high))
    return tensor.unflatten(tensor.structure(self.target_spec()), physical)

  def normalize_ground_truth(self, ground_truth):
    """Physical ``EnzymeGroundTruth`` -> standardised flat ``(..., len(PARAMETER_NAMES) + 1)``: every
    parameter by its own prior range, the half-conversion time by its (log) range."""
    low = jnp.asarray([r[0] for _, r in self._parameter_ranges], jnp.float32)
    high = jnp.asarray([r[1] for _, r in self._parameter_ranges], jnp.float32)
    log_half_time_bounds = (math.log(self.half_time_bounds[0]), math.log(self.half_time_bounds[1]))
    return jnp.concatenate([
      (2.0 * ground_truth.parameters - (low + high)) / (high - low),
      self._to_unit(jnp.log(ground_truth.half_time), log_half_time_bounds)
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    low, high = self.melting_bounds
    return f'MSE (melting temperature / {0.5 * (high - low):.1f} C)'

  def metric_labels(self):
    return ('loss', 'melting_temperature')

  def loss(self, predicted, target):
    return jnp.mean(jnp.square(predicted - target), axis=-1)

  def metric(self, predicted, target):
    squared = jnp.square(predicted - target)
    return {'loss': jnp.mean(squared, axis=-1), 'melting_temperature': squared[..., 0]}

  def metric_real_rmse(self, metric_means):
    """Sample-averaged normalised ``melting_temperature`` metric (from :meth:`metric`) -> real-unit
    RMSE. ``loss`` has no single unit and is omitted."""
    low, high = self.melting_bounds
    return {'melting_temperature': (float(np.sqrt(metric_means['melting_temperature']) * 0.5 * (high - low)), 'C')}

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """Draw the enzymes at ``event_index`` and run each one's batch of experiments under ``design``
    (one design broadcast over the batch, or one design per event). DETERMINISTIC: an enzyme and its
    readout noise are seeded from ``event_index`` alone, so the same event under two designs is the
    same enzyme -- the target is a property of the event only."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    fraction, temperature = self._resolve_design(design, n)
    measurements, melting, parameters, half_time, error = self._generate(
      fraction, temperature, jnp.asarray(event_index, jnp.int32)
    )

    error = float(jnp.max(error))
    if not math.isfinite(error) or error > self.integration_tolerance:
      raise RuntimeError(
        f'integration error {error:.3g} mM over {n} events exceeds integration_tolerance '
        f'{self.integration_tolerance:.3g} mM -- the largest disagreement between the dt and dt/2 '
        f'chains on the extent of reaction at a read-out time; raise n_steps_per_measurement '
        f'(or check dt against scripts/check_stability.py).'
      )

    ground_truth = EnzymeGroundTruth(parameters=parameters, half_time=half_time[:, None])
    mask = jnp.ones((n, self.n_experiments), jnp.int32)
    return ground_truth, EnzymeEvent(measurements=measurements), mask, EnzymeTarget(melting_temperature=melting[:, None])

  def _resolve_design(self, design, n):
    """``EnzymeDesign`` / config ``Mapping`` / flat array -> ``(fraction, temperature)``, both
    ``(n, n_experiments)`` (a single design is broadcast over the event batch)."""
    flat = jnp.broadcast_to(
      jnp.reshape(self.flatten_design(design), (-1, 2 * self.n_experiments)), (n, 2 * self.n_experiments)
    )
    return flat[:, :self.n_experiments], flat[:, self.n_experiments:]

  def _initial_state(self, enzyme_fraction):
    """The concentrations right after mixing: the enzyme stock takes ``enzyme_fraction`` of the
    volume, the A and B stocks split the rest equally."""
    rest = 0.5 * (1.0 - enzyme_fraction)
    return self.concentration_A * rest, self.concentration_B * rest, self.concentration_E * enzyme_fraction

  def _rate(self, extent, A0, B0, E0, temperature, parameters, log_k0_cat):
    """The reaction rate at extent ``x``: ``A = A0 - x``, ``B = B0 - x``, ``C = D = x``, E constant.
    ``log_k0_cat`` is passed apart from ``parameters`` because the calibration evaluates the rate
    before it is known."""
    return kinetics(
      A0 - extent, B0 - extent, extent, extent, E0, temperature, dict(parameters, log_k0_cat=log_k0_cat)
    )

  def _event(self, fraction, temperature, event_index):
    """One event: draw an enzyme, calibrate its turnover, and run the batch. ``fraction`` /
    ``temperature`` are ``(n_experiments,)``; every draw uses ``event_index`` only.

    The target is the drawn ``T_melting``, so nothing beyond the batch itself has to be integrated --
    the conversion-vs-temperature objective this replaced cost one integration per golden-section
    contraction, on top of the experiments."""
    key_parameters, key_half_time, key_noise = jax.random.split(jax.random.PRNGKey(event_index), 3)
    parameters = self._draw_parameters(key_parameters)
    low, high = self.half_time_bounds
    half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=math.log(low), maxval=math.log(high)))

    log_k0_cat = self._calibrate(parameters, half_time)
    measurements, batch_error = self._run_batch(fraction, temperature, parameters, log_k0_cat, key_noise)

    packed = jnp.stack([parameters[name] for name in PARAMETER_NAMES])
    return measurements, parameters['T_melting'], packed, half_time, batch_error

  def _draw_parameters(self, key):
    """One enzyme: every parameter uniform within its own prior range."""
    keys = jax.random.split(key, len(self._parameter_ranges))
    return {
      name: jax.random.uniform(k, (), minval=low, maxval=high)
      for (name, (low, high)), k in zip(self._parameter_ranges, keys)
    }

  def _calibrate(self, parameters, half_time):
    """``log_k0_cat`` putting the enzyme's FASTEST half-conversion time exactly at ``half_time``.

    Turnover enters the rate as one multiplicative factor, so scaling ``k_cat`` scales every time by
    the inverse; in particular, which temperature reacts fastest does not depend on that scale. At
    ``log_k0_cat = 0`` the half-conversion time of the calibration mixture is

        tau(T) = int_0^{x_half} dx / rate(x, T),

    a plain quadrature over the extent -- no time stepping -- because the whole state is a function
    of it. Setting ``exp(log_k0_cat) = min_T tau(T) / half_time`` then rescales the fastest
    temperature onto ``half_time``. Denatured temperatures contribute ``tau = inf`` and drop out of
    the minimum, which stays finite because ``T_melting`` is required to lie inside the scan."""
    A0, B0, E0 = self._initial_state(self.calibration_fraction)
    extent = self._quadrature_nodes * (0.5 * jnp.minimum(A0, B0))

    def half_conversion_time(temperature):
      return jnp.trapezoid(1.0 / self._rate(extent, A0, B0, E0, temperature, parameters, 0.0), extent), jnp.zeros(())

    fastest, _ = self._extremum(half_conversion_time, maximise=False)
    return jnp.log(half_conversion_time(fastest)[0]) - jnp.log(half_time)

  def _extremum(self, value, *, maximise):
    """The temperature within ``temperature_bounds`` at which ``value(T)`` is extremal, by GOLDEN-
    SECTION search: ``n_temperature_steps`` bracket contractions, each costing ONE evaluation (the
    other interior point is reused) and shrinking the bracket by 0.618.

    Both quantities searched this way are unimodal in ``T``, which is what the search needs: ``log
    rate`` is a linear Arrhenius / van't Hoff term plus the concave ``-softplus`` of the unfolding
    sigmoid and of the Michaelis saturation, so the rate is log-concave in ``T``; hence
    ``tau(T) = int dx / rate`` is log-convex with a single minimum, and the conversion reached in a
    fixed time has a single maximum. A scan of the whole range would cost one evaluation per grid
    point for the same answer -- and, because XLA fuses the reduction across the scanned points, that
    also compiled ~50x slower (2.5 s at 48 points, 125 s at 384).

    Ties break toward the COOLER temperature (``<=``): a saturated conversion peak is flat-topped, and
    its cool edge is the safer label, overshooting the optimum costing several times what
    undershooting it does.

    ``value`` returns ``(value, integration_error)``; the largest error met anywhere in the search is
    returned with the extremum."""
    sign = -1.0 if maximise else 1.0

    def evaluate(temperature):
      v, error = value(temperature)
      return sign * v, error

    low, high = (jnp.asarray(bound, jnp.float32) for bound in self.temperature_bounds)
    span = high - low
    # Interior points at the golden ratios, left < right.
    left, right = high - GOLDEN_SECTION * span, low + GOLDEN_SECTION * span
    f_left, error_left = evaluate(left)
    f_right, error_right = evaluate(right)

    def contract(carry, _):
      low, high, left, right, f_left, f_right, worst = carry
      keep_left = f_left <= f_right  # better on the left -> the extremum lies in [low, right]
      low, high = jnp.where(keep_left, low, left), jnp.where(keep_left, right, high)
      # One interior point of the new bracket is an interior point of the old one: reuse it, and
      # evaluate only the point it does not cover (below it when the left half was kept, above
      # otherwise).
      reused, f_reused = jnp.where(keep_left, left, right), jnp.where(keep_left, f_left, f_right)
      span = high - low
      fresh = jnp.where(keep_left, high - GOLDEN_SECTION * span, low + GOLDEN_SECTION * span)
      f_fresh, error = evaluate(fresh)
      left, right = jnp.where(keep_left, fresh, reused), jnp.where(keep_left, reused, fresh)
      f_left, f_right = jnp.where(keep_left, f_fresh, f_reused), jnp.where(keep_left, f_reused, f_fresh)
      return (low, high, left, right, f_left, f_right, jnp.maximum(worst, error)), None

    start = (low, high, left, right, f_left, f_right, jnp.maximum(error_left, error_right))
    (low, high, left, right, f_left, f_right, worst), _ = jax.lax.scan(
      contract, start, None, length=self.n_temperature_steps
    )
    return jnp.where(f_left <= f_right, left, right), worst

  def _optimal_temperature(self, parameters, log_k0_cat):
    """The temperature at which the reference mixture reaches the highest conversion after
    ``duration`` -- the event's target. Located by :meth:`_extremum`."""
    A0, B0, E0 = self._initial_state(self.objective_fraction)

    def conversion(temperature):
      extent, error = self._integrate(
        A0, B0, E0, temperature, parameters, log_k0_cat, n_steps=self.n_objective_steps, n_intervals=1
      )
      return extent[-1] / jnp.minimum(A0, B0), error

    return self._extremum(conversion, maximise=True)

  def _run_batch(self, fraction, temperature, parameters, log_k0_cat, key):
    """The batch's readout: [A] at every measurement time of every experiment, plus independent
    ``N(0, measurement_noise)`` readout noise."""
    def run(enzyme_fraction, experiment_temperature):
      A0, B0, E0 = self._initial_state(enzyme_fraction)
      extent, error = self._integrate(
        A0, B0, E0, experiment_temperature, parameters, log_k0_cat,
        n_steps=self.n_steps_per_measurement, n_intervals=self.n_measurements
      )
      return A0 - extent, error

    concentration, errors = jax.vmap(run)(fraction, temperature)
    measurements = concentration + self.measurement_noise * jax.random.normal(key, concentration.shape)
    return measurements, jnp.max(errors)

  def rkc2_step(self, rate, extent, dt):
    """One RKC2 step of size ``dt`` on ``dx/dt = rate(x)`` from ``extent``.

    The Chebyshev recursion of :func:`rkc2_coefficients`: ``n_stages`` rate evaluations, the first of
    which (``rate(w_0)``) is reused by every later stage. The stages run as a ``jax.lax.scan`` over
    their coefficients rather than an unrolled Python loop -- unrolling put ``n_stages`` copies of the
    rate (five exponentials each) inside the time-step body, and with the half-step error monitor and
    the vmap over the temperature scan on top of that, XLA compilation became the dominant cost.
    """
    mu_tilde_1, coefficients = self._rkc2
    slope_0 = rate(extent)

    def stage(carry, row):
      previous, current = carry
      mu, nu, mu_tilde, gamma_tilde = row
      following = ((1.0 - mu - nu) * extent + mu * current + nu * previous
                   + mu_tilde * dt * rate(current) + gamma_tilde * dt * slope_0)
      return (current, following), None

    start = (extent, extent + mu_tilde_1 * dt * slope_0)  # (w_0, w_1)
    (_, final), _ = jax.lax.scan(stage, start, coefficients)
    return final

  def _chain(self, rate, *, dt, n_steps, n_intervals):
    """One RKC2 chain: ``n_intervals`` intervals of ``n_steps`` steps of ``dt``, returning the extent
    of reaction at the END of every interval (the measurement times)."""
    def step(extent, _):
      return self.rkc2_step(rate, extent, dt), None

    def interval(extent, _):
      extent, _ = jax.lax.scan(step, extent, None, length=n_steps)
      return extent, extent

    _, extents = jax.lax.scan(interval, jnp.zeros(()), None, length=n_intervals)
    return extents

  def _integrate(self, A0, B0, E0, temperature, parameters, log_k0_cat, *, n_steps, n_intervals):
    """RKC2 on the extent of reaction over ``duration``, split into ``n_intervals`` equal intervals.
    Returns the extent at the END of every interval (the measurement times) and the integration
    error, ESTIMATED INSIDE THE SOLVE.

    Two chains run over the same interval grid -- one at ``dt``, one at ``dt/2`` -- and the error is
    the largest ``|fine - coarse|`` over the interval ends. That is a GLOBAL error estimate on the
    quantity actually consumed downstream (the extent at the read-out times, and hence the
    conversion the target is an argmax of), not a per-step local truncation estimate: a local
    estimate can stay small while the accumulated trajectory drifts.

    The ``dt`` chain is what is RETURNED, paired with that error -- the solution reported is the one
    the error was measured on, rather than a finer solution whose own error is only inferred. For a
    second-order scheme Richardson puts the returned chain's true error at ``~4/3`` of the estimate,
    which is what it measures (median ratio 1.34 against a ``dt/4`` reference over the prior). It is
    an estimate, not a rigorous bound: when the two chains happen to agree by cancellation it
    under-reports (worst seen 27x, at an absolute error of 4e-6 mM). What makes the guard safe is the
    margin rather than the constant -- the true error runs ~1000x inside ``integration_tolerance``
    and ~10^4 below the read-out noise. A jitted kernel cannot raise, so the assertion lives outside
    the solve.

    ``dt`` must keep ``dt * |df/dx|`` inside the scheme's measured stability boundary everywhere the
    prior can reach; ``scripts/check_stability.py`` measures both the boundary (in float32, on the
    target device) and the worst ``|df/dx|``."""
    dt = self.duration / (n_intervals * n_steps)
    rate = lambda x: self._rate(x, A0, B0, E0, temperature, parameters, log_k0_cat)

    coarse = self._chain(rate, dt=dt, n_steps=n_steps, n_intervals=n_intervals)
    fine = self._chain(rate, dt=0.5 * dt, n_steps=2 * n_steps, n_intervals=n_intervals)
    return coarse, jnp.max(jnp.abs(fine - coarse))
