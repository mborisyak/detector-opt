"""Michaelis-Menten coefficient recovery: the SAME chemistry as ``enzyme.py``, a different question.

``docs/task-mm-recovery.md`` is the specification. The reaction, the rate law, the unfolding model and
the RKC2 solver are unchanged -- what changes is **what the batch has to estimate** and, following from
that, two things about the experiment:

* **the target is a vector of KINETIC COEFFICIENTS**, not the melting temperature. Those coefficients
  are already drawn per event by the melt model; here they are what the regressor reads off the
  progress curves. The number of identifiable coefficients ``p`` is what the batch size ``m`` is then
  matched to -- the melt task's structural failure is ``p = 1`` against ``m = 4``, which leaves most of
  the batch with no job and the response a step rather than a smooth function of the design.
* **SUBSTRATE CONCENTRATION becomes a DESIGN COORDINATE.** A batch that never varies substrate cannot
  separate ``K_M`` from ``k_cat`` at all -- that is the classical reason a Michaelis-Menten design needs
  a sub-saturating *and* a saturating arm. WHICH substrate is varied is decided by the read-out, not by
  taste: the noise sits on the MEASURED species [A], so a sub-saturating [A] arm would have to sit
  inside the read-out noise, whereas [B] can be varied over its whole Michaelis range with the signal
  left large. So the shipped configuration holds [A]0 FIXED and saturating and designs
  ``(enzyme_fraction, temperature, [B]0)`` -- ``3 * n_experiments`` dimensions. Setting
  ``concentration_A_bounds`` instead of ``concentration_A`` makes [A]0 a fourth coordinate, which is the
  variant that needs a sharper read-out; both are the same detector and the same physics.
  The sampling times stay a FIXED grid and are NOT designable (user's decision), which is what keeps
  time positional in the combined element.
* **the temperature box sits BELOW the melt.** With ``temperature_bounds`` under the whole ``T_melting``
  prior the folded fraction is ~1 everywhere (checked at construction against ``min_folded_fraction``),
  so the enzyme stops being a threshold device and becomes a smooth Arrhenius one: every design in the
  box produces a measurable progress curve, and there is no dead volume. ``T_melting`` stays a drawn
  NUISANCE, never predicted.

Anchoring: 25 C, not 0 C
------------------------
``enzyme.py`` parameterises every van 't Hoff constant by its value at **0 C** and a per-10 C factor.
That is fine when the constants are nuisances; it is fatal when they are the TARGET. The design box
probes 10-35 C, so a 0 C-anchored constant is reachable only by extrapolating through its own ``Q10``,
and on this prior that extrapolation alone spreads ``ln K_A`` by about as much as the parameter's own
prior -- a constant that no design can measure, i.e. a built-in floor. The parameters drawn here are
therefore the constants **at 25 C** (where enzyme kinetics are conventionally reported, and a
temperature inside the box); :func:`_to_zero_anchor` converts them to the 0 C form ``kinetics`` expects.
Same physics, same functional form, different coordinates on it.

Turnover calibration
--------------------
``k_cat`` is still CALIBRATED rather than drawn (only its slope ``Q10_cat`` is drawn): every enzyme is
rescaled so a fixed reference assay half-converts at a drawn ``half_time``, which is what makes the
whole prior measurable within one ``duration``. Unlike the melt model the calibration does NOT search
for the fastest temperature -- below the melt the rate is monotone in ``T``, so "fastest" would just be
the top of the design box and would tie the meaning of the drawn ``half_time`` to the box. It is
evaluated at ``calibration_temperature`` (the 25 C anchor) on a fixed reference mixture instead, so the
event -- and therefore the target -- is a property of the enzyme alone.

Integrator rule (repo-wide, non-negotiable)
-------------------------------------------
Every integration runs a ``dt`` chain and a ``dt/2`` chain over the same read-out times, RETURNS the
``dt`` one, reports ``max|fine - coarse|`` at those times, and :meth:`EnzymeMMDetector.__call__` asserts
that against the CONFIGURABLE ``integration_tolerance`` host-side on every call (a jitted kernel cannot
raise). The rate law's stiffness moved with the box -- this task reaches 10x the enzyme loading on 1/25
the substrate -- so the step count and the tolerance are re-measured for it (``scripts/validate_mm.py``)
rather than inherited from ``config/detector/enzyme.yaml``.
"""

import math
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .common import Detector
# `rkc2_step` / `rkc2_chain` are re-exported rather than reimplemented: ONE copy of the scheme lives in
# `enzyme.py`, next to the coefficients it belongs to, and the probes that import them from here keep
# working.
from .enzyme import (INV_TEMPERATURE_SPAN, ZERO_CELSIUS, gibbs_fraction, kinetics, rkc2_chain,
                     rkc2_coefficients, rkc2_step)
from ..utils import tensor

__all__ = [
  'PARAMETER_NAMES', 'ANCHOR_TEMPERATURE', 'EnzymeMMDetector', 'EnzymeMMDesign', 'EnzymeMMEvent',
  'EnzymeMMTarget', 'EnzymeMMGroundTruth'
]

# The temperature the drawn kinetic constants are the value AT (see the module docstring). Enzyme
# kinetics are conventionally reported at 25 C, and it lies inside the design box, so no drawn
# parameter has to be reached by extrapolation.
ANCHOR_TEMPERATURE = 25.0
# log K(0 C) = log K(anchor) - log(Q10) * ANCHOR_SHIFT, from the van 't Hoff form of `vant_hoff`.
ANCHOR_SHIFT = (1.0 / ZERO_CELSIUS - 1.0 / (ANCHOR_TEMPERATURE + ZERO_CELSIUS)) / INV_TEMPERATURE_SPAN

# The kinetic parameters DRAWN per event, in the order they are packed into the ground truth. Same
# population as `enzyme.PARAMETER_NAMES`, but every constant is its value at ANCHOR_TEMPERATURE, which
# is what the name change records. `log_k_cat` is deliberately absent: the turnover SCALE is calibrated,
# not drawn (only its temperature slope Q10_cat is), so recovering an absolute k_cat is not a well-posed
# question here.
PARAMETER_NAMES = (
  'log_K_A', 'Q10_A',    # Michaelis constant of A: log-value at 25 C (mM) + per-10C factor
  'log_K_B', 'Q10_B',    # ... of B
  'log_Ki_C', 'Q10_C',   # competitive inhibition constant of the product C
  'log_Ki_D', 'Q10_D',   # ... of the product D
  'Q10_cat',             # turnover: only its temperature slope is drawn
  'delta_H', 'delta_C',  # unfolding enthalpy / heat-capacity change, both over R (K)
  'T_melting'            # melting temperature (C) -- a NUISANCE: the box sits below it
)

# Which drawn parameter feeds which argument of `kinetics`, once re-anchored to 0 C.
_KINETIC_NAMES = {
  'log_K_A': 'log_K0_A', 'log_K_B': 'log_K0_B', 'log_Ki_C': 'log_K0i_C', 'log_Ki_D': 'log_K0i_D'
}
# The Q10 that re-anchors each of them.
_ANCHOR_PARTNER = {'log_K_A': 'Q10_A', 'log_K_B': 'Q10_B', 'log_Ki_C': 'Q10_C', 'log_Ki_D': 'Q10_D'}

# Physical unit of each drawn parameter, for `metric_real_rmse`. The `log_*` are natural logarithms of
# a concentration in mM, so their error is a LOG error -- a factor, not a difference.
_PARAMETER_UNITS = {
  'log_K_A': 'ln mM', 'log_K_B': 'ln mM', 'log_Ki_C': 'ln mM', 'log_Ki_D': 'ln mM',
  'Q10_A': '', 'Q10_B': '', 'Q10_C': '', 'Q10_D': '', 'Q10_cat': '',
  'delta_H': 'K', 'delta_C': 'K', 'T_melting': 'C'
}


def _to_zero_anchor(parameters):
  """The drawn 25 C-anchored constants -> the 0 C-anchored dict :func:`kinetics` takes.

  A pure change of coordinates on the same van 't Hoff curve: ``K(T) = exp(log K_anchor + log(Q10) *
  (delta(T) - delta(anchor)) / span)`` is ``exp(log K_0 + log(Q10) * delta(T) / span)`` with
  ``log K_0 = log K_anchor - log(Q10) * delta(anchor) / span``. Nothing about the physics moves; what
  moves is which value of ``K(T)`` is the number the regressor has to recover.
  """
  converted = {
    _KINETIC_NAMES[name]: parameters[name] - jnp.log(parameters[_ANCHOR_PARTNER[name]]) * ANCHOR_SHIFT
    for name in _KINETIC_NAMES
  }
  return dict(
    {name: parameters[name] for name in PARAMETER_NAMES if name not in _KINETIC_NAMES}, **converted
  )


def _scale(values, bounds):
  """NOMINAL ``[low, high]`` -> SCALED ``[0, 1]``, linearly."""
  low, high = bounds
  return (values - low) / (high - low)


def _unscale(values, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL ``[low, high]`` (inverse of :func:`_scale`)."""
  low, high = bounds
  return values * (high - low) + low


def _log_scale(values, bounds):
  """NOMINAL ``[low, high]`` -> SCALED ``[0, 1]``, LOGARITHMICALLY.

  The three multiplicative design coordinates (enzyme loading and the two substrate concentrations)
  are scaled this way because that is how an experimenter varies them -- a dilution series -- and
  because a linear map would make the box unusable: ``[A]0`` spans 0.01-2 mM, so a uniform draw puts
  99.5% of its mass above 0.01 mM and the sub-saturating half that identifies ``K_M`` at all would
  essentially never be sampled. The map is still a bijection of the same box onto ``[0, 1]``; it
  changes the SEARCH MEASURE, not the admissible set.
  """
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return (jnp.log(values) - low) / (high - low)


def _log_unscale(values, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL ``[low, high]``, logarithmically (inverse of :func:`_log_scale`)."""
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return jnp.exp(values * (high - low) + low)


class EnzymeMMDesign(NamedTuple):
  """One batch of initial experiments, SHIPPED layout: per experiment, the volume fraction taken by the
  enzyme stock, the temperature (C) it is run at, and the initial concentration of the varied substrate
  B (mM). [A]0 is a fixed assay constant here -- it is the MEASURED species, so varying it downwards is
  what the read-out noise forbids."""
  enzyme_fraction: jax.Array  # (n_experiments,)
  temperature: jax.Array      # (n_experiments,)
  concentration_B: jax.Array  # (n_experiments,)


class EnzymeMMSubstrateDesign(NamedTuple):
  """The same batch with BOTH substrate concentrations designable -- the layout selected by giving
  ``concentration_A_bounds`` instead of ``concentration_A``. Kept because it is the variant a sharper
  read-out would allow, and the two are compared on Fisher information rather than on preference."""
  enzyme_fraction: jax.Array  # (n_experiments,)
  temperature: jax.Array      # (n_experiments,)
  concentration_A: jax.Array  # (n_experiments,)
  concentration_B: jax.Array  # (n_experiments,)


class EnzymeMMEvent(NamedTuple):
  """The batch's readout: the noisy [A] samples (mM) of every experiment."""
  measurements: jax.Array  # (n_experiments, n_measurements)


class EnzymeMMTarget(NamedTuple):
  """What the regressor predicts: the drawn enzyme's kinetic coefficients at 25 C, in
  ``target_parameters`` order. DRAWN parameters, read straight out of the prior -- no conversion is
  simulated to produce them, and they are properties of the enzyme alone, so no design can move its own
  label."""
  coefficients: jax.Array  # (p,)


class EnzymeMMGroundTruth(NamedTuple):
  """The drawn enzyme itself (== conditioning): its kinetic parameters in ``PARAMETER_NAMES`` order and
  the half-conversion time its turnover was calibrated to (h). The target is a SLICE of the former."""
  parameters: jax.Array  # (len(PARAMETER_NAMES),)
  half_time: jax.Array   # (1,)


class EnzymeMMDetector(Detector):
  """Single-batch design of ``n_experiments`` enzymatic experiments whose target is the enzyme's own
  Michaelis-Menten coefficients (see the module docstring).

  Every constant is a constructor argument, i.e. lives in the yaml config. Concentrations are mM, times
  hours, temperatures degrees Celsius (``kinetics`` converts internally).
  """

  def __init__(
    self, *,
    n_experiments: int,
    n_measurements: int,
    parameters: dict,
    target_parameters: list,
    normalization_reference: str = None,
    concentration_E: float,
    duration: float,
    measurement_noise: float,
    enzyme_fraction_bounds: tuple,
    temperature_bounds: tuple,
    concentration_B_bounds: tuple,
    concentration_A: float = None,
    concentration_A_bounds: tuple = None,
    calibration_fraction: float,
    calibration_concentration_A: float,
    calibration_concentration_B: float,
    calibration_temperature: float,
    half_time_bounds: tuple,
    min_folded_fraction: float,
    n_quadrature: int,
    n_steps_per_measurement: int,
    n_stages: int,
    damping: float,
    integration_tolerance: float
  ):
    self.n_experiments = int(n_experiments)
    self.n_measurements = int(n_measurements)
    self.concentration_E = float(concentration_E)
    self.duration = float(duration)
    self.measurement_noise = float(measurement_noise)
    self.enzyme_fraction_bounds = (float(enzyme_fraction_bounds[0]), float(enzyme_fraction_bounds[1]))
    self.temperature_bounds = (float(temperature_bounds[0]), float(temperature_bounds[1]))
    self.concentration_B_bounds = (float(concentration_B_bounds[0]), float(concentration_B_bounds[1]))
    # [A]0 is EITHER a fixed assay constant (`concentration_A`) OR a design coordinate
    # (`concentration_A_bounds`), never both and never neither. Which one is a MEASUREMENT decision:
    # [A] is the species the photometer reads, so a sub-saturating [A]0 arm would have to sit inside the
    # read-out noise, while [B]0 can be swept over its whole Michaelis range with the signal left large.
    if (concentration_A is None) == (concentration_A_bounds is None):
      raise ValueError(
        'give exactly one of concentration_A (a fixed, saturating assay concentration of the MEASURED '
        'species) or concentration_A_bounds (making it a design coordinate, which needs a read-out fine '
        f'enough to resolve its sub-saturating arm); got {concentration_A} / {concentration_A_bounds}'
      )
    self.concentration_A = None if concentration_A is None else float(concentration_A)
    self.concentration_A_bounds = None if concentration_A_bounds is None else \
        (float(concentration_A_bounds[0]), float(concentration_A_bounds[1]))
    self.calibration_fraction = float(calibration_fraction)
    self.calibration_concentration_A = float(calibration_concentration_A)
    self.calibration_concentration_B = float(calibration_concentration_B)
    self.calibration_temperature = float(calibration_temperature)
    self.half_time_bounds = (float(half_time_bounds[0]), float(half_time_bounds[1]))
    self.min_folded_fraction = float(min_folded_fraction)
    self.n_quadrature = int(n_quadrature)
    self.n_steps_per_measurement = int(n_steps_per_measurement)
    self.n_stages = int(n_stages)
    self.damping = float(damping)
    self.integration_tolerance = float(integration_tolerance)
    # RKC2 coefficients: static floats, folded into the jaxpr as constants.
    self._rkc2 = rkc2_coefficients(self.n_stages, self.damping)

    missing, unknown = set(PARAMETER_NAMES) - set(parameters), set(parameters) - set(PARAMETER_NAMES)
    if len(missing) > 0 or len(unknown) > 0:
      raise ValueError(
        f'the prior must give a range for every drawn parameter: missing {sorted(missing)}, unknown '
        f'{sorted(unknown)} (the turnover scale is calibrated, not drawn)'
      )
    # Drawn uniformly within each range, in PARAMETER_NAMES order; a `log_*` range is therefore
    # log-uniform on the quantity itself.
    self.parameter_ranges = tuple(
      (name, (float(parameters[name][0]), float(parameters[name][1]))) for name in PARAMETER_NAMES
    )
    for name, (low, high) in self.parameter_ranges:
      if not low < high:
        raise ValueError(f'the prior range of {name} must be an increasing (low, high), got ({low}, {high})')

    self.target_parameters = tuple(str(name) for name in target_parameters)
    if len(self.target_parameters) == 0:
      raise ValueError('target_parameters must name at least one drawn parameter')
    unknown = [name for name in self.target_parameters if name not in PARAMETER_NAMES]
    if len(unknown) > 0:
      raise ValueError(f'target_parameters names {unknown}, which are not drawn parameters {PARAMETER_NAMES}')
    if len(set(self.target_parameters)) != len(self.target_parameters):
      raise ValueError(f'target_parameters repeats a parameter: {self.target_parameters}')
    ranges = dict(self.parameter_ranges)
    # THE MSE IS NORMALISED PER PARAMETER BY THAT PARAMETER'S OWN PRIOR RANGE (see `normalize_target`),
    # UNLESS `normalization_reference` names a parameter whose range every `log_K*` target borrows.
    #
    # WHY THE OPTION EXISTS. Own-range normalisation is SELF-CANCELLING: widening a parameter's prior
    # also widens its normaliser, so the normalised component keeps variance 1/3 and the task's ceiling
    # does not move. A wider prior therefore buys NO span on its own. Borrowing one common scale for the
    # Michaelis constants makes a widening visible -- a `K` drawn over a wider range than the reference
    # lands outside [-1, 1] and carries variance above 1/3, which is exactly the extra room a harder
    # parameter should contribute. It also puts the two `K`s in the same units, so "a factor of two
    # wrong" costs the same whichever constant it is, which is the honest statistic for two quantities
    # of the same kind. Each component is still CENTRED on its own prior mean, so it stays zero-mean;
    # only the SCALE is shared. Non-`log_K*` targets (e.g. `Q10_cat`, a different kind of quantity) keep
    # their own range.
    self.normalization_reference = None if normalization_reference is None else str(normalization_reference)
    if self.normalization_reference is not None and self.normalization_reference not in ranges:
      raise ValueError(
        f'normalization_reference {self.normalization_reference!r} is not a drawn parameter {PARAMETER_NAMES}'
      )
    bounds = []
    for name in self.target_parameters:
      low, high = ranges[name]
      if self.normalization_reference is not None and name.startswith('log_K'):
        reference_low, reference_high = ranges[self.normalization_reference]
        centre, half = 0.5 * (low + high), 0.5 * (reference_high - reference_low)
        low, high = centre - half, centre + half
      bounds.append((low, high))
    self.target_bounds = tuple(bounds)
    self._target_low = jnp.asarray([low for low, _ in self.target_bounds], jnp.float32)
    self._target_high = jnp.asarray([high for _, high in self.target_bounds], jnp.float32)
    self._target_index = tuple(PARAMETER_NAMES.index(name) for name in self.target_parameters)

    # THE DEFINING PROPERTY OF THIS TASK: the design box sits BELOW the whole melting prior, so the
    # enzyme is a smooth Arrhenius device rather than a threshold one and every design in the box
    # produces a measurable progress curve. Checked here, at construction, rather than asserted in a
    # comment -- and checked NUMERICALLY on the folded fraction, because "below the melting point" is
    # not the same statement as "folded": a low unfolding enthalpy against a high heat-capacity change
    # puts the COLD arm of the Gibbs-Helmholtz stability curve inside the box.
    melting_low, melting_high = ranges['T_melting']
    if melting_low <= self.temperature_bounds[1]:
      raise ValueError(
        f'the T_melting prior ({melting_low}, {melting_high}) must lie strictly ABOVE temperature_bounds '
        f'{self.temperature_bounds}: this task measures kinetics below the melt, not the melt itself'
      )
    folded = self._worst_folded_fraction()
    if not folded > self.min_folded_fraction:
      raise ValueError(
        f'the folded fraction falls to {folded:.4f} somewhere in temperature_bounds x the '
        f'(delta_H, delta_C, T_melting) prior, below min_folded_fraction {self.min_folded_fraction}: '
        f'the unfolding transition (hot or COLD) enters the design box, so the read-out is a melt curve '
        f'rather than a Michaelis-Menten one'
      )

    # The design layout, in `flatten_design` field order: `(field, bounds, logarithmic)`. The three
    # MULTIPLICATIVE coordinates are scaled logarithmically (see `_log_scale`); the temperature, which is
    # an additive coordinate on 1/T through the van 't Hoff form, is scaled linearly.
    layout = [('enzyme_fraction', self.enzyme_fraction_bounds, True), ('temperature', self.temperature_bounds, False)]
    if self.concentration_A_bounds is not None:
      layout.append(('concentration_A', self.concentration_A_bounds, True))
    layout.append(('concentration_B', self.concentration_B_bounds, True))
    self._design_layout = tuple(layout)
    self._design_type = EnzymeMMSubstrateDesign if self.concentration_A_bounds is not None else EnzymeMMDesign
    if self._design_type._fields != tuple(name for name, _, _ in self._design_layout):
      raise ValueError(f'{self._design_type.__name__} fields do not match the design layout')

    for name, bounds in (
      [('enzyme_fraction_bounds', self.enzyme_fraction_bounds),
       ('concentration_B_bounds', self.concentration_B_bounds),
       ('half_time_bounds', self.half_time_bounds)]
      + ([] if self.concentration_A_bounds is None else [('concentration_A_bounds', self.concentration_A_bounds)])
    ):
      # All three are LOG-scaled design/prior ranges, so both ends must be strictly positive.
      if not 0.0 < bounds[0] < bounds[1]:
        raise ValueError(f'{name} must be an increasing pair of positive values, got {bounds}')
    if self.enzyme_fraction_bounds[1] > 1.0:
      raise ValueError(f'enzyme_fraction is a volume fraction and cannot exceed 1, got {self.enzyme_fraction_bounds}')
    if not 0.0 < self.calibration_fraction <= 1.0:
      raise ValueError(f'calibration_fraction is a volume fraction and must lie in (0, 1], got {self.calibration_fraction}')
    for name, value in (
      ('calibration_concentration_A', self.calibration_concentration_A),
      ('calibration_concentration_B', self.calibration_concentration_B)
    ):
      if not value > 0.0:
        raise ValueError(f'{name} must be positive, got {value}')

    # The quadrature nodes of the calibration (as a fraction of the half-conversion extent). Built
    # here, eagerly -- never lazily inside a jitted method.
    self._quadrature_nodes = jnp.linspace(0.0, 1.0, self.n_quadrature, dtype=jnp.float32)
    # The read-out times of one experiment: evenly spaced, ending at `duration`. A FIXED grid, the same
    # for every experiment and every design -- which is what lets the combined element carry time
    # positionally (plan section 4: sampling times are not a design coordinate).
    self.measurement_times = (self.duration / self.n_measurements) * jnp.arange(1, self.n_measurements + 1, dtype=jnp.float32)
    self.measurement_dt = self.duration / (self.n_measurements * self.n_steps_per_measurement)
    self._generate = jax.jit(jax.vmap(self._event))

  def _worst_folded_fraction(self):
    """Smallest ``gibbs_fraction`` over ``temperature_bounds`` x the unfolding prior -- MEASURED on a
    grid, at construction.

    ``dG`` is increasing in ``delta_H`` and decreasing in ``delta_C``, so the worst corner in those two
    is ``(min delta_H, max delta_C)``; neither ``T`` nor ``T_melting`` can be cornered that way (the hot
    end of the box is worst against a LOW melting point, the cold end against a HIGH one, because the
    stability curve has two roots), so both are swept."""
    ranges = dict(self.parameter_ranges)
    temperature = jnp.linspace(self.temperature_bounds[0], self.temperature_bounds[1], 65, dtype=jnp.float32)
    melting = jnp.linspace(ranges['T_melting'][0], ranges['T_melting'][1], 33, dtype=jnp.float32)
    folded = gibbs_fraction(temperature[:, None], ranges['delta_H'][0], ranges['delta_C'][1], melting[None, :])
    return float(jnp.min(folded))

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return EnzymeMMEvent(measurements=jax.ShapeDtypeStruct((self.n_experiments, self.n_measurements), np.float32))

  def target_spec(self):
    return EnzymeMMTarget(coefficients=jax.ShapeDtypeStruct((len(self.target_parameters),), np.float32))

  def ground_truth_spec(self):
    f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
    return EnzymeMMGroundTruth(parameters=f(len(PARAMETER_NAMES)), half_time=f(1))

  def design_shape(self):
    return (len(self._design_layout) * self.n_experiments,)

  def design_spec(self):
    f = jax.ShapeDtypeStruct((self.n_experiments,), np.float32)
    return self._design_type(*(f for _ in self._design_layout))

  def design_bounds(self):
    return {name: bounds for name, bounds, _ in self._design_layout}

  def combined_event_shape(self):
    # element == experiment; its features are its own measurements + its own design coordinates
    return (self.n_experiments, self.n_measurements + len(self._design_layout))

  def size(self):
    return None  # an analytic source: every index is a fresh enzyme

  # ------------------------------------------------------------------ #
  # Design scaling: each field onto [0, 1] from its own bounds
  # ------------------------------------------------------------------ #
  def _blocks(self, flat):
    """Flat design ``(..., n_fields * n_experiments)`` -> one ``(..., n_experiments)`` block per field,
    in ``_design_layout`` order."""
    n = self.n_experiments
    return tuple(flat[..., k * n:(k + 1) * n] for k in range(len(self._design_layout)))

  def _to_scaled_flat(self, design):
    blocks = self._blocks(jnp.asarray(design, jnp.float32))
    return jnp.concatenate([
      (_log_scale if logarithmic else _scale)(block, bounds)
      for block, (_, bounds, logarithmic) in zip(blocks, self._design_layout)
    ], axis=-1)

  def _to_nominal_flat(self, design_scaled):
    blocks = self._blocks(jnp.asarray(design_scaled, jnp.float32))
    return jnp.concatenate([
      (_log_unscale if logarithmic else _unscale)(block, bounds)
      for block, (_, bounds, logarithmic) in zip(blocks, self._design_layout)
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled, mask=None):
    """``features (..., n_experiments, n_measurements + n_design_fields)``: each experiment's [A] samples
    followed by its own design values, taken STRAIGHT from the scaled design -- already each coordinate
    on its own range in [0, 1], which is what the network wants.

    The samples are scaled by ONE FIXED reference -- the top of the [A]0 box -- not by each
    experiment's own [A]0. This is the same rule `enzyme_inhibitor` uses, and it was a CHANGE:

    THE OLD RULE divided by the experiment's own [A]0, reporting the FRACTION of [A] left, on the
    argument that a single linear map would squash low-[A]0 experiments into the bottom percent of the
    range, and that nothing is lost because [A]0 is appended as a feature. The information argument is
    correct and the conditioning argument is fatal. Read-out noise is ABSOLUTE, so dividing by [A]0
    multiplies it by 1/[A]0 -- up to 100x at the bottom of a three-decade box. MEASURED on the proxy's
    own best design at m=4: features ran -8.49 to +7.21 with 4.9% of entries outside +-2, against
    [-0.53, +1.89] and 0.00% for `enzyme_inhibitor`. An experiment carrying no signal produced the
    LOUDEST inputs in the batch, saturating the units that touched it (tanh' ~ 1e-6 at |x| = 8) while
    the informative experiments sat near +-1 and trained normally. The network scored 0.8660 where
    predicting the prior mean scores 0.3340.

    Recovering [A] from the fraction requires MULTIPLYING two inputs, which a two-block MLP of widths
    24 and 16 is a poor instrument for even before saturation. So the old rule lost no information and
    all of the conditioning.

    Under the fixed reference a low-[A]0 experiment does occupy a small part of the range -- which is
    the honest representation, since 0.01 mM of signal under 0.05 mM of noise IS nothing -- and its
    features are bounded rather than amplified.

    ``mask`` is unused: every experiment of the batch is real (the element axis is the design's, not a
    hit count)."""
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    if design_scaled.ndim == 1:  # one design for the whole event batch
      design_scaled = jnp.broadcast_to(design_scaled[None, :], event.measurements.shape[:-2] + design_scaled.shape)
    blocks = self._blocks(design_scaled)
    reference = self.concentration_A if self.concentration_A_bounds is None else self.concentration_A_bounds[1]
    measurements = self._to_unit(event.measurements, (0.0, float(reference)))
    return jnp.concatenate([measurements] + [block[..., None] for block in blocks], axis=-1)

  def element_mask(self, event, mask):
    return mask  # element == experiment

  @staticmethod
  def _to_unit(values, bounds):
    """Map ``[low, high]`` linearly onto ``[-1, 1]``."""
    low, high = bounds
    return (2.0 * values - (low + high)) / (high - low)

  def normalize_target(self, target):
    """EACH COMPONENT BY ITS OWN PRIOR RANGE -- the requirement that makes the vector target a single
    loss (plan section 6.2).

    The components are on wildly different scales (a ``log K_M`` range is 2.3 nats wide, a ``Q10`` range
    is 0.8 wide), so a raw MSE would be the widest component's error and the optimiser would design for
    that one alone. Scaled by its own prior half-range every component is on ``[-1, 1]`` and contributes
    comparably, and the loss is dimensionless. Note that the ``log_*`` components are normalised in LOG
    space, which is also the right statistics: a factor-of-two error in ``K_M`` should cost the same
    whether ``K_M`` is 0.02 or 2 mM.

    It also keeps the convention every measurement in this project is read against: a uniform prior maps
    onto ``[-1, 1]`` with variance 1/3, so predicting the prior mean scores 1/3 and ``loss_precision``
    keeps the meaning it has on the other benchmarks."""
    flat, _ = tensor.flatten(target)
    return (2.0 * flat - (self._target_low + self._target_high)) / (self._target_high - self._target_low)

  def denormalize_predictions(self, normalised):
    physical = 0.5 * (jnp.asarray(normalised, jnp.float32) * (self._target_high - self._target_low)
                      + (self._target_low + self._target_high))
    return tensor.unflatten(tensor.structure(self.target_spec()), physical)

  def normalize_ground_truth(self, ground_truth):
    """Physical ``EnzymeMMGroundTruth`` -> standardised flat ``(..., len(PARAMETER_NAMES) + 1)``: every
    parameter by its own prior range, the half-conversion time by its (log) range."""
    low = jnp.asarray([r[0] for _, r in self.parameter_ranges], jnp.float32)
    high = jnp.asarray([r[1] for _, r in self.parameter_ranges], jnp.float32)
    log_half_time_bounds = (math.log(self.half_time_bounds[0]), math.log(self.half_time_bounds[1]))
    return jnp.concatenate([
      (2.0 * ground_truth.parameters - (low + high)) / (high - low),
      self._to_unit(jnp.log(ground_truth.half_time), log_half_time_bounds)
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    return f'MSE (kinetic coefficients / own prior half-range, mean over {len(self.target_parameters)})'

  def metric_labels(self):
    return ('loss',) + self.target_parameters

  def loss(self, predicted, target):
    return jnp.mean(jnp.square(predicted - target), axis=-1)

  def metric(self, predicted, target):
    """Per-sample loss AND the per-parameter squared errors -- which direction the design is failing is
    the diagnostic a vector target exists to give."""
    squared = jnp.square(predicted - target)
    metrics = {'loss': jnp.mean(squared, axis=-1)}
    for index, name in enumerate(self.target_parameters):
      metrics[name] = squared[..., index]
    return metrics

  def metric_real_rmse(self, metric_means):
    """Sample-averaged per-parameter metrics -> real-unit RMSE. ``loss`` has no single unit and is
    omitted."""
    out = {}
    for index, name in enumerate(self.target_parameters):
      low, high = self.target_bounds[index]
      out[name] = (float(np.sqrt(metric_means[name]) * 0.5 * (high - low)), _PARAMETER_UNITS[name])
    return out

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """Draw the enzymes at ``event_index`` and run each one's batch of experiments under ``design`` (one
    design broadcast over the batch, or one design per event). DETERMINISTIC: an enzyme and its readout
    noise are seeded from ``event_index`` alone, so the same event under two designs is the same enzyme
    -- the target is a property of the event only."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    fraction, temperature, initial_A, initial_B = self._resolve_design(design, n)
    measurements, coefficients, parameters, half_time, error = self._generate(
      fraction, temperature, initial_A, initial_B, jnp.asarray(event_index, jnp.int32)
    )

    error = float(jnp.max(error))
    if not math.isfinite(error) or error > self.integration_tolerance:
      raise RuntimeError(
        f'integration error {error:.3g} mM over {n} events exceeds integration_tolerance '
        f'{self.integration_tolerance:.3g} mM -- the largest disagreement between the dt and dt/2 chains '
        f'on the extent of reaction at a read-out time; raise n_steps_per_measurement (or check dt '
        f'against the measured stability boundary, scripts/validate_mm.py --mode stability).'
      )

    ground_truth = EnzymeMMGroundTruth(parameters=parameters, half_time=half_time[:, None])
    mask = jnp.ones((n, self.n_experiments), jnp.int32)
    return ground_truth, EnzymeMMEvent(measurements=measurements), mask, EnzymeMMTarget(coefficients=coefficients)

  def _resolve_design(self, design, n):
    """``Design`` namedtuple / config ``Mapping`` / flat array -> ``(fraction, temperature, [A]0, [B]0)``,
    each ``(n, n_experiments)`` (a single design is broadcast over the event batch). [A]0 is filled from
    the fixed assay constant when it is not a design coordinate."""
    width = int(self.design_dim())
    flat = jnp.broadcast_to(jnp.reshape(self.flatten_design(design), (-1, width)), (n, width))
    return self._split(flat)

  def _split(self, flat):
    """Flat design -> ``(fraction, temperature, [A]0, [B]0)``, [A]0 broadcast from the assay constant
    when it is fixed. The one place the two layouts differ."""
    blocks = self._blocks(flat)
    if self.concentration_A_bounds is None:
      initial_A = jnp.full_like(blocks[0], self.concentration_A)
      return blocks[0], blocks[1], initial_A, blocks[2]
    return blocks[0], blocks[1], blocks[2], blocks[3]

  # ------------------------------------------------------------------ #
  # Forward model
  # ------------------------------------------------------------------ #
  def measurement_sigma(self, concentration):
    """The read-out's own standard deviation at ``concentration`` (mM): the photometer's ABSOLUTE
    Gaussian floor, ``measurement_noise``, independent of the concentration being read. There is no
    proportional term, so the same sigma applies at the top and at the bottom of the design box, and the
    config bounds that are stated from the noise floor are stated against this one number.

    Returned with ``concentration``'s own shape, so callers can weight per measurement (the Fisher
    information in ``scripts/validate_mm.py`` indexes it as an array)."""
    return jnp.full_like(concentration, self.measurement_noise)

  def readout(self, design, parameters, half_time):
    """The NOISELESS [A] read-out ``(n_experiments, n_measurements)`` of ONE enzyme under ONE design.

    The forward model, differentiable in ``parameters`` (a dict over :data:`PARAMETER_NAMES`) and
    ``half_time`` -- this is what the identifiability analysis pushes ``jax.jacfwd`` through
    (``scripts/validate_mm.py --mode fim``)."""
    flat = jnp.reshape(self.flatten_design(design), (int(self.design_dim()),))
    fraction, temperature, initial_A, initial_B = self._split(flat)
    kinetic = _to_zero_anchor(parameters)
    log_k0_cat = self._calibrate(kinetic, half_time)
    clean, _ = self._batch(fraction, temperature, initial_A, initial_B, kinetic, log_k0_cat)
    return clean

  def _event(self, fraction, temperature, initial_A, initial_B, event_index):
    """One event: draw an enzyme, calibrate its turnover, and run the batch. The design arguments are
    ``(n_experiments,)``; every draw uses ``event_index`` only."""
    key_parameters, key_half_time, key_noise = jax.random.split(jax.random.PRNGKey(event_index), 3)
    parameters = self._draw_parameters(key_parameters)
    low, high = self.half_time_bounds
    half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=math.log(low), maxval=math.log(high)))

    kinetic = _to_zero_anchor(parameters)
    log_k0_cat = self._calibrate(kinetic, half_time)
    clean, error = self._batch(fraction, temperature, initial_A, initial_B, kinetic, log_k0_cat)
    measurements = clean + self.measurement_sigma(clean) * jax.random.normal(key_noise, clean.shape)

    packed = jnp.stack([parameters[name] for name in PARAMETER_NAMES])
    coefficients = jnp.stack([parameters[name] for name in self.target_parameters])
    return measurements, coefficients, packed, half_time, error

  def _draw_parameters(self, key):
    """One enzyme: every parameter uniform within its own prior range."""
    keys = jax.random.split(key, len(self.parameter_ranges))
    return {
      name: jax.random.uniform(k, (), minval=low, maxval=high)
      for (name, (low, high)), k in zip(self.parameter_ranges, keys)
    }

  def _calibrate(self, kinetic, half_time):
    """``log_k0_cat`` putting the REFERENCE assay's half-conversion time exactly at ``half_time``.

    Turnover enters the rate as one multiplicative factor, so at ``log_k0_cat = 0`` the reference
    mixture's half-conversion time is

        tau = int_0^{x_half} dx / rate(x, calibration_temperature),

    a plain quadrature over the extent -- no time stepping -- because the whole state is a function of
    it; ``exp(log_k0_cat) = tau / half_time`` then rescales it onto ``half_time``.

    The reference mixture and its temperature are FIXED config constants, deliberately not the design's:
    a design-dependent reference would make the drawn ``half_time`` -- and hence the enzyme -- depend on
    the design, which would break the common-random-numbers contract the whole benchmark rests on. And
    unlike the melt model there is no search for the fastest temperature: below the melt the rate is
    monotone in ``T``, so the fastest temperature is just the top of the design box, which would tie the
    calibration to the box rather than to the enzyme."""
    A0 = self.calibration_concentration_A
    B0 = self.calibration_concentration_B
    E0 = self.concentration_E * self.calibration_fraction
    extent = self._quadrature_nodes * (0.5 * min(A0, B0))
    rate = self._rate(extent, A0, B0, E0, self.calibration_temperature, kinetic, 0.0)
    return jnp.log(jnp.trapezoid(1.0 / rate, extent)) - jnp.log(half_time)

  def _rate(self, extent, A0, B0, E0, temperature, kinetic, log_k0_cat):
    """The reaction rate at extent ``x``: ``A = A0 - x``, ``B = B0 - x``, ``C = D = x``, E constant.
    ``log_k0_cat`` is passed apart from ``kinetic`` because the calibration evaluates the rate before it
    is known."""
    return kinetics(A0 - extent, B0 - extent, extent, extent, E0, temperature, dict(kinetic, log_k0_cat=log_k0_cat))

  def _batch(self, fraction, temperature, initial_A, initial_B, kinetic, log_k0_cat):
    """The batch's CLEAN readout: [A] at every measurement time of every experiment, and the worst
    integration error over the batch."""
    def run(enzyme_fraction, experiment_temperature, A0, B0):
      extent, error = self._integrate(
        A0, B0, self.concentration_E * enzyme_fraction, experiment_temperature, kinetic, log_k0_cat
      )
      return A0 - extent, error

    concentration, errors = jax.vmap(run)(fraction, temperature, initial_A, initial_B)
    return concentration, jnp.max(errors)

  def _integrate(self, A0, B0, E0, temperature, kinetic, log_k0_cat):
    """RKC2 on the extent of reaction over ``duration``, split into ``n_measurements`` equal intervals.
    Returns the extent at the END of every interval (the read-out times) and the integration error,
    ESTIMATED INSIDE THE SOLVE.

    Two chains run over the same interval grid -- one at ``dt``, one at ``dt/2`` -- and the error is the
    largest ``|fine - coarse|`` over the interval ends. That is a GLOBAL error estimate on the quantity
    actually consumed downstream (the [A] the regressor sees), not a per-step local truncation estimate,
    which can stay small while the accumulated trajectory drifts. The ``dt`` chain is what is RETURNED,
    paired with that error -- the solution reported is the one the error was measured on. A jitted
    kernel cannot raise, so the assertion against ``integration_tolerance`` lives in :meth:`__call__`.

    Note that this estimate is not a rigorous bound; for a second-order scheme Richardson puts the
    returned chain's true error at ``~4/3`` of it, which is what it measures (median ratio 1.331
    against a fine reference, ``output/screen/audit_rkc2.py --section estimator``).

    It USED TO BE dominated by float32 roundoff in the Chebyshev recursion rather than by truncation,
    so that refining ``dt`` past ~160 steps per read-out made the guard WORSE. That was an artefact of
    the recursion's algebraic arrangement and was fixed on 2026-08-13 -- see
    :func:`~detopt.detector.enzyme.rkc2_increment`. The guard now falls like ``dt^2`` out to at least
    640 steps per read-out and reads the truncation error it is supposed to read."""
    dt = self.duration / (self.n_measurements * self.n_steps_per_measurement)
    rate = lambda x: self._rate(x, A0, B0, E0, temperature, kinetic, log_k0_cat)

    start = jnp.zeros(())  # the extent of reaction, from zero
    coarse = rkc2_chain(self._rkc2, rate, start, dt=dt, n_steps=self.n_steps_per_measurement,
                        n_intervals=self.n_measurements)
    fine = rkc2_chain(self._rkc2, rate, start, dt=0.5 * dt, n_steps=2 * self.n_steps_per_measurement,
                      n_intervals=self.n_measurements)
    return coarse, jnp.max(jnp.abs(fine - coarse))
