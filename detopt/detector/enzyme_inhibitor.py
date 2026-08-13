"""Inhibitor MECHANISM screening: which of three regimes does a compound act by?

The same wet-lab batch-design contract as :mod:`detopt.detector.enzyme` and the same chemistry
(hexokinase: glucose ``A`` + ATP ``B`` -> G6P ``C`` + ADP ``D``), with ONE species added: an
inhibitor ``I`` whose concentration is a design coordinate. It is not consumed, so it enters the rate
law as a parameter -- no new ODE state, no solver change.

* **design** -- a batch of ``n_experiments`` initial experiments. Per experiment the dilution factor
  of the enzyme stock, the initial ATP concentration ``B0``, the inhibitor concentration ``I`` and
  the temperature. Glucose is held FIXED and saturating, so it neither confounds the substrate
  contrast nor spends a design dimension. ``B0`` and ``I`` span decades and are therefore LOG-scaled
  into the unit cube (the read-out depends on ``log(B0/K_B)`` and ``log(I*k)``, so the objective is
  smooth in log concentration and violently curved in linear concentration -- the coordinate a
  stationary GP kernel needs).

  NO MIXING CONSTRAINT (waived by the user, 2026-08-12). Unlike the baseline enzyme detector, where
  the enzyme stock takes a volume fraction and the A and B stocks split the rest
  (``A0 = concentration_A * (1 - fraction) / 2``), every initial concentration here is INDEPENDENT:
  stocks can be made up at whatever strength an experiment needs, so the mixing ratio is bookkeeping
  rather than physics. What remains physical is each coordinate's own limit -- solubility at the top,
  the read-out noise floor at the bottom for the measured species, the cap ``min(A0, B0)`` on total
  conversion for the co-substrate, and pipetting practicality for the enzyme dilution -- and those
  are the rules the design bounds are stated from.
* **event** -- one enzyme drawn from a wide "unspecified hexokinase" prior AND one compound drawn
  from a library, plus the read-out noise: ``n_measurements`` noisy samples of [A] per experiment.
  Everything is seeded from ``event_index`` ALONE, so the same enzyme/compound pair is re-measured
  under every design (common random numbers) and the target is a property of the event, never of the
  design.
* **target** -- the compound's MECHANISM CLASS, one-hot over the classes the library contains:
  mostly-competitive (binds free E >> ES), mostly-noncompetitive (binds both within 1.4x),
  mostly-uncompetitive (binds ES >> E). ``mechanism_classes`` selects which of the three are drawn --
  all of them by default, the two TYPED ones for the binary-extremes task -- and the one-hot index is
  the SLOT in that tuple, so it is ``n_classes`` wide, not always 3.
* **loss** -- softmax cross-entropy divided by ``ln(n_classes)``, so the no-information level is
  exactly 1.0 and chance accuracy is ``1 / n_classes``.

The rate law
------------
::

    K_A_app = K_A * (1 + C/Ki_C)                        product inhibition, unchanged
    K_B_app = K_B * (1 + D/Ki_D + I*k1)                 ADP and the inhibitor COMPETE at the ATP site

                       k_cat * E * A * B
    v = -----------------------------------------
        (A + K_A_app) * (B + K_B_app) * (1 + I*k2)      uncompetitive part: caps V_max

``k1 = 1/Ki(E)`` is the competitive strength (the inhibitor binds free enzyme, so it raises the
apparent Michaelis constant of the varied substrate) and ``k2 = 1/Ki(ES)`` the uncompetitive one (it
binds the ES complex, so it caps the turnover and cannot be outrun by substrate). Two ligands
competing for the SAME site add inside one bracket rather than multiplying -- multiplying would count
the doubly-occupied state twice -- and at ``I = 0`` the law reduces exactly to the baseline enzyme's.

WHICH SITE, AND WHY THE ATP ONE. For a two-substrate enzyme "competitive" is always "competitive with
respect to WHICH substrate", so the model must name one, and two arguments agree on the nucleotide
site: glucose-6-phosphate, hexokinase's own product inhibitor, competes there, so a G6P-like compound
is a real class of HK inhibitor; and the read-out noise is ABSOLUTE (0.05 mM on [A]), so only
``K_M``(ATP) = 0.1-1 mM keeps a sub-``K_M`` arm above the floor -- ``K_M``(glucose) = 0.02-0.2 mM
would put it below. Applying the competitive factor to both sites would be a different, unphysical
compound and is deliberately not done.

Why the batch is a 2x2
----------------------
Writing ``R_B`` for the ratio of rates between a strong and a weak inhibitor dose at ATP level ``B``::

    B0 >> K_B_app :  R_high = (1 + I_s k2) / (1 + I_w k2)              -> k2 alone
    B0 << K_B_app :  R_low  = R_high * (1 + I_s k1) / (1 + I_w k1)     -> k1, once R_high is known

The inhibitor contrast cancels the UNKNOWN turnover (a global rate factor is exactly what ``k2``
produces, and the half-conversion time is drawn over a factor of 8, so without two doses ``k2`` is
degenerate with a slow enzyme); the substrate contrast separates ``k1`` from ``k2``, because
competitive inhibition is overcome by saturating substrate and uncompetitive inhibition is not.
Neither arm can be read without the other, so no cell of the 2x2 is redundant -- which is what makes
``n_experiments = 4`` a structural requirement rather than a knob.

Numerics
--------
Unchanged from the baseline enzyme detector: the 1:1:1:1 stoichiometry leaves the extent of reaction
``x`` as the only state (``A = A0 - x``, ``B = B0 - x``, ``C = D = x``), integrated with RKC2, and
every integration runs a ``dt`` and a ``dt/2`` chain over the same read-out times, RETURNS the ``dt``
one, and reports ``max|fine - coarse|`` there. :meth:`EnzymeInhibitorDetector.__call__` asserts on it
against the configurable ``integration_tolerance`` host-side, every call (a jitted kernel cannot
raise).

Thermal unfolding is NOT simulated. The temperature box sits below the ~41.9 C melting point of yeast
hexokinase, where the folded fraction of a real enzyme moves by ~4% across the whole box -- under the
read-out noise, monotone in T, and absorbed by the turnover calibration -- so simulating it would add
three thermodynamic parameters that change nothing measurable. Temperature is still a lever: it moves
``K_B`` by up to 5.7x across the box, hence where a fixed ``B0`` sits relative to it.
"""

import math
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .common import Detector
from .enzyme import vant_hoff, rkc2_chain, rkc2_coefficients, rkc2_step, GOLDEN_SECTION
from ..utils import tensor

__all__ = [
  'kinetics', 'PARAMETER_NAMES', 'CLASS_NAMES', 'N_CLASSES', 'EnzymeInhibitorDetector',
  'EnzymeInhibitorDesign', 'EnzymeInhibitorEvent', 'EnzymeInhibitorTarget', 'EnzymeInhibitorGroundTruth'
]

# The three mechanism REGIMES, in the "mostly" form an enzymologist reports. Their order is the
# one-hot order of the target.
CLASS_NAMES = ('mostly_competitive', 'mostly_noncompetitive', 'mostly_uncompetitive')
N_CLASSES = len(CLASS_NAMES)
COMPETITIVE, NONCOMPETITIVE, UNCOMPETITIVE = 0, 1, 2

# The kinetic parameters DRAWN per event, in the order they are packed into the ground truth.
# `log_k0_cat` is absent by construction: the turnover scale is not drawn but CALIBRATED, so that
# every enzyme reacts on a measurable timescale (see EnzymeInhibitorDetector._calibrate). The
# baseline detector's `delta_H`, `delta_C` and `T_melting` are absent too -- no unfolding is
# simulated inside this temperature box (see the module docstring).
PARAMETER_NAMES = (
  'log_K0_A', 'Q10_A',      # Michaelis constant of glucose: log-value at 0 C (mM) + per-10C factor
  'log_K0_B', 'Q10_B',      # ... of ATP
  'log_K0i_C', 'Q10_C',     # competitive inhibition constant of the product G6P
  'log_K0i_D', 'Q10_D',     # ... of the product ADP
  'Q10_cat',                # turnover: only its temperature slope is drawn
)


def kinetics(A, B, C, D, E, inhibitor, temperature, parameters):
  """The inhibited rate law (module docstring). ``inhibitor`` is a concentration (mM); ``parameters``
  carries the drawn enzyme, the calibrated ``log_k0_cat`` and the compound's ``k1``/``k2`` (1/mM)."""
  K_A = vant_hoff(temperature, parameters['log_K0_A'], parameters['Q10_A'])
  K_B = vant_hoff(temperature, parameters['log_K0_B'], parameters['Q10_B'])
  Ki_C = vant_hoff(temperature, parameters['log_K0i_C'], parameters['Q10_C'])
  Ki_D = vant_hoff(temperature, parameters['log_K0i_D'], parameters['Q10_D'])

  Kapp_A = K_A * (1 + C / Ki_C)
  # ADP and the inhibitor compete for the SAME (nucleotide) site, so their terms ADD.
  Kapp_B = K_B * (1 + D / Ki_D + inhibitor * parameters['k1'])

  ### actually Arrhenius, but the expression is the same
  k_cat = vant_hoff(temperature, parameters['log_k0_cat'], parameters['Q10_cat'])

  rate = k_cat * E * A * B / (A + Kapp_A) / (B + Kapp_B) / (1 + inhibitor * parameters['k2'])

  return rate


def _scale(values, bounds):
  """NOMINAL ``[low, high]`` -> SCALED ``[0, 1]``."""
  low, high = bounds
  return (values - low) / (high - low)


def _unscale(values, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL ``[low, high]`` (inverse of :func:`_scale`)."""
  low, high = bounds
  return values * (high - low) + low


def _log_scale(values, bounds):
  """NOMINAL concentration -> SCALED ``[0, 1]``, LOG10-affine: ``x = (log c - log c_min) / (log c_max
  - log c_min)``. Both bounds are strictly positive by construction (checked in ``__init__``)."""
  low, high = bounds
  return (jnp.log10(values) - math.log10(low)) / (math.log10(high) - math.log10(low))


def _log_unscale(values, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL concentration (inverse of :func:`_log_scale`)."""
  low, high = bounds
  return jnp.power(10.0, values * (math.log10(high) - math.log10(low)) + math.log10(low))


class EnzymeInhibitorDesign(NamedTuple):
  """One batch of initial experiments: per experiment, the volume fraction taken by the enzyme stock,
  the initial ATP concentration (mM), the inhibitor concentration (mM) and the temperature (C).
  Glucose is a config constant, held saturating for every experiment."""
  enzyme_fraction: jax.Array  # (n_experiments,)
  substrate_B: jax.Array      # (n_experiments,)
  inhibitor: jax.Array        # (n_experiments,)
  temperature: jax.Array      # (n_experiments,)


class EnzymeInhibitorEvent(NamedTuple):
  """The batch's read-out: the noisy [A] samples (mM) of every experiment."""
  measurements: jax.Array  # (n_experiments, n_measurements)


class EnzymeInhibitorTarget(NamedTuple):
  """What the regressor predicts: the drawn compound's mechanism class, one-hot in
  ``mechanism_classes`` order (== ``CLASS_NAMES`` for the default three-class library). A DRAWN
  property of the compound -- no design can move its own label."""
  mechanism: jax.Array  # (n_classes,)


class EnzymeInhibitorGroundTruth(NamedTuple):
  """The drawn enzyme and compound (== conditioning): the kinetic parameters in ``PARAMETER_NAMES``
  order, the half-conversion time the turnover was calibrated to (h), and the compound's two
  branch strengths as ``log10 k`` (1/mM). The class itself is the target, and is ``k1`` vs ``k2``."""
  parameters: jax.Array  # (len(PARAMETER_NAMES),)
  half_time: jax.Array   # (1,)
  log10_k1: jax.Array    # (1,) competitive branch, 1/Ki(E)
  log10_k2: jax.Array    # (1,) uncompetitive branch, 1/Ki(ES)


class EnzymeInhibitorDetector(Detector):
  """Single-batch design of ``n_experiments`` inhibition experiments (see the module docstring).

  Every constant is a constructor argument, i.e. lives in the yaml config: the stock solutions, the
  experiment length and its read-out, the enzyme and compound priors, the design bounds, and the
  resolution of each numerical procedure. Concentrations are mM and times hours throughout;
  temperatures are degrees Celsius (``kinetics`` converts internally).
  """

  # NOTE: `n_classes` is set per INSTANCE in __init__ (it follows `mechanism_classes`), not here.
  # `detopt.bo.gbdt` reads it to decide that the proxy is a softmax classifier scored by
  # cross-entropy / ln K -- the same quantity `loss` below computes. It is an explicit declaration
  # rather than something inferred from the target's width, because a one-hot 3-vector and a
  # 3-component regression target are the same array shape.

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
    substrate_B_bounds: tuple,
    inhibitor_bounds: tuple,
    temperature_bounds: tuple,
    inhibitor_potency_bounds: tuple,
    preference_bounds: tuple,
    preference_bounds_noncompetitive: tuple,
    calibration_fraction: float,
    half_time_bounds: tuple,
    n_temperature_steps: int,
    n_quadrature: int,
    n_steps_per_measurement: int,
    n_stages: int,
    damping: float,
    integration_tolerance: float,
    mechanism_classes: tuple = CLASS_NAMES
  ):
    # WHICH mechanism classes the compound library contains. The default is all three. Naming only
    # the two TYPED ones -- ('mostly-competitive', 'mostly-uncompetitive') -- gives the BINARY
    # EXTREMES task, and that is not an arbitrary simplification: those two are the pair the 2x2
    # factorial exists to separate, because competitive inhibition is visible ONLY below K_B and
    # uncompetitive ONLY at saturation. A design that puts every experiment at one ATP level
    # therefore cannot tell them apart at all, while a factorial one can -- maximal, physically
    # grounded design dependence. MEASURED on the proxy at TWO designs (the best and the worst of the
    # three-way campaign output/campaign-inhib/126382657/from_scratch/results.json): the binary task's
    # design-dependent span is 0.9767 -> 0.4576, i.e. 52% of its ceiling 1.0
    # (output/screen/ext-worst.json, ext-good.json), with accuracy running 0.546 (chance 0.5) at the bad
    # design to 0.827 at the good one; scored instead as REGRESSION on (log10 k1, log10 k2) the same two
    # designs span 0.1236 -> 0.0589 against a ceiling of 0.1955, i.e. 33% (output/screen/branch-*.json).
    # NO FIGURE IS QUOTED FOR THE THREE-WAY TASK at those designs: nothing on disk measures it there.
    unknown = [name for name in mechanism_classes if name not in CLASS_NAMES]
    if len(unknown) > 0:
      raise ValueError(f'unknown mechanism class(es) {unknown}; choose from {CLASS_NAMES}')
    if len(mechanism_classes) < 2:
      raise ValueError(f'need at least two mechanism classes to have something to discriminate, got {mechanism_classes}')
    self.mechanism_classes = tuple(mechanism_classes)
    # The GLOBAL class id of each drawn slot; the label is the SLOT, so the one-hot is `n_classes`
    # wide however many of the three are in play.
    self.class_indices = jnp.asarray([CLASS_NAMES.index(name) for name in self.mechanism_classes], jnp.int32)
    self.n_classes = len(self.mechanism_classes)

    self.n_experiments = int(n_experiments)
    self.n_measurements = int(n_measurements)
    self.concentration_A = float(concentration_A)
    self.concentration_B = float(concentration_B)
    self.concentration_E = float(concentration_E)
    self.duration = float(duration)
    self.measurement_noise = float(measurement_noise)
    self.enzyme_fraction_bounds = (float(enzyme_fraction_bounds[0]), float(enzyme_fraction_bounds[1]))
    self.substrate_B_bounds = (float(substrate_B_bounds[0]), float(substrate_B_bounds[1]))
    self.inhibitor_bounds = (float(inhibitor_bounds[0]), float(inhibitor_bounds[1]))
    self.temperature_bounds = (float(temperature_bounds[0]), float(temperature_bounds[1]))
    self.inhibitor_potency_bounds = (float(inhibitor_potency_bounds[0]), float(inhibitor_potency_bounds[1]))
    self.preference_bounds = (float(preference_bounds[0]), float(preference_bounds[1]))
    self.preference_bounds_noncompetitive = (
      float(preference_bounds_noncompetitive[0]), float(preference_bounds_noncompetitive[1])
    )
    self.calibration_fraction = float(calibration_fraction)
    self.half_time_bounds = (float(half_time_bounds[0]), float(half_time_bounds[1]))
    self.n_temperature_steps = int(n_temperature_steps)
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
        f'{sorted(unknown)} (`log_k0_cat` is calibrated, not drawn; no unfolding is simulated)'
      )
    # Drawn uniformly within each range, in PARAMETER_NAMES order; a `log_*` range is therefore
    # log-uniform on the quantity itself.
    self._parameter_ranges = tuple(
      (name, (float(parameters[name][0]), float(parameters[name][1]))) for name in PARAMETER_NAMES
    )
    for name, (low, high) in self._parameter_ranges:
      if not low < high:
        raise ValueError(f'the prior range of {name} must be an increasing (low, high), got ({low}, {high})')

    for name, bounds in (
      ('substrate_B_bounds', self.substrate_B_bounds), ('inhibitor_bounds', self.inhibitor_bounds),
      ('inhibitor_potency_bounds', self.inhibitor_potency_bounds), ('half_time_bounds', self.half_time_bounds)
    ):
      # LOG-scaled / log-uniform quantities: zero is not merely a bad experiment here, it is outside
      # the parameterisation. Excluding I = 0 is also what removes the exact ceiling -- information
      # is nonzero for every I > 0, so the ceiling is approached asymptotically instead of hit.
      if not 0.0 < bounds[0] < bounds[1]:
        raise ValueError(f'{name} must be an increasing pair of strictly positive values, got {bounds}')
    for name, bounds in (
      ('enzyme_fraction_bounds', self.enzyme_fraction_bounds), ('temperature_bounds', self.temperature_bounds),
      ('preference_bounds', self.preference_bounds),
      ('preference_bounds_noncompetitive', self.preference_bounds_noncompetitive)
    ):
      if not bounds[0] < bounds[1]:
        raise ValueError(f'{name} must be an increasing pair, got {bounds}')
    for index, bound in enumerate(self.enzyme_fraction_bounds):
      if not 0.0 < bound <= 1.0:
        raise ValueError(
          f'enzyme_fraction_bounds[{index}] is a volume fraction and must lie within (0, 1], got {bound}'
        )
    if not 0.0 < self.calibration_fraction < 1.0:
      raise ValueError(
        f'calibration_fraction is a volume fraction and must lie strictly within (0, 1), got {self.calibration_fraction}'
      )
    # THE GAP IS STRUCTURAL, not a convenience: compounds whose preference falls between the two
    # bands are genuinely "mixed" and no enzymologist would assign them a class, so without the gap
    # the Bayes error would be dominated by boundary-ambiguous compounds -- a floor NO design can
    # reduce, which is exactly the design-independent floor this benchmark exists to avoid.
    if not self.preference_bounds_noncompetitive[1] < self.preference_bounds[0]:
      raise ValueError(
        f'the class bands must leave a GAP: preference_bounds_noncompetitive {self.preference_bounds_noncompetitive} '
        f'must end strictly below preference_bounds {self.preference_bounds}'
      )

    # The compound's DOMINANT branch strength, log10 k = -log10 Ki (1/mM), drawn log-uniformly over
    # the potency prior; the weak branch sits `preference` decades below it.
    self.log10_k_bounds = (
      -math.log10(self.inhibitor_potency_bounds[1]), -math.log10(self.inhibitor_potency_bounds[0])
    )
    # Both branches together therefore live in [log10 k_min - preference_max, log10 k_max]; the
    # ground-truth normalisation uses that range.
    self.log10_branch_bounds = (self.log10_k_bounds[0] - self.preference_bounds[1], self.log10_k_bounds[1])

    # The quadrature nodes of the calibration (as a fraction of the half-conversion extent). Built
    # here, eagerly -- never lazily inside a jitted method.
    self._quadrature_nodes = jnp.linspace(0.0, 1.0, self.n_quadrature, dtype=jnp.float32)
    # The read-out times of one experiment: evenly spaced, ending at `duration` (t = 0 carries no
    # information -- [A] there is the glucose stock, known).
    self.measurement_times = (self.duration / self.n_measurements) * jnp.arange(
      1, self.n_measurements + 1, dtype=jnp.float32
    )
    self.measurement_dt = self.duration / (self.n_measurements * self.n_steps_per_measurement)
    self._generate = jax.jit(jax.vmap(self._event))

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return EnzymeInhibitorEvent(
      measurements=jax.ShapeDtypeStruct((self.n_experiments, self.n_measurements), np.float32)
    )

  def target_spec(self):
    return EnzymeInhibitorTarget(mechanism=jax.ShapeDtypeStruct((self.n_classes,), np.float32))

  def ground_truth_spec(self):
    f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
    return EnzymeInhibitorGroundTruth(
      parameters=f(len(PARAMETER_NAMES)), half_time=f(1), log10_k1=f(1), log10_k2=f(1)
    )

  def design_shape(self):
    return (4 * self.n_experiments,)

  def design_spec(self):
    f = jax.ShapeDtypeStruct((self.n_experiments,), np.float32)
    return EnzymeInhibitorDesign(enzyme_fraction=f, substrate_B=f, inhibitor=f, temperature=f)

  def design_bounds(self):
    return {
      'enzyme_fraction': self.enzyme_fraction_bounds, 'substrate_B': self.substrate_B_bounds,
      'inhibitor': self.inhibitor_bounds, 'temperature': self.temperature_bounds
    }

  def combined_event_shape(self):
    # element == experiment; its features are its own measurements + its own 4 design coordinates
    return (self.n_experiments, self.n_measurements + 4)

  def size(self):
    return None  # an analytic source: every index is a fresh (enzyme, compound) pair

  # ------------------------------------------------------------------ #
  # Design scaling: linear for the order-1 coordinates, LOG10 for the concentrations
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    d = jnp.asarray(design, jnp.float32)
    n = self.n_experiments
    return jnp.concatenate([
      _scale(d[..., :n], self.enzyme_fraction_bounds),
      _log_scale(d[..., n:2 * n], self.substrate_B_bounds),
      _log_scale(d[..., 2 * n:3 * n], self.inhibitor_bounds),
      _scale(d[..., 3 * n:4 * n], self.temperature_bounds)
    ], axis=-1)

  def _to_nominal_flat(self, design_scaled):
    e = jnp.asarray(design_scaled, jnp.float32)
    n = self.n_experiments
    return jnp.concatenate([
      _unscale(e[..., :n], self.enzyme_fraction_bounds),
      _log_unscale(e[..., n:2 * n], self.substrate_B_bounds),
      _log_unscale(e[..., 2 * n:3 * n], self.inhibitor_bounds),
      _unscale(e[..., 3 * n:4 * n], self.temperature_bounds)
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled, mask=None):
    """``features (..., n_experiments, n_measurements + 4)``: each experiment's [A] samples followed
    by its own four design values, taken STRAIGHT from the scaled design -- already each coordinate
    on its own range in [0, 1] (log-scaled for the two concentrations), which is what the network
    wants. The MEASUREMENTS stay LINEAR: the read-out noise is additive and homoscedastic in linear
    space, and a noisy [A] near zero can be negative, where a log is undefined. ``mask`` is unused:
    every experiment of the batch is real (the element axis is the design's, not a hit count)."""
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    if design_scaled.ndim == 1:  # one design for the whole event batch
      design_scaled = jnp.broadcast_to(design_scaled[None, :], event.measurements.shape[:-2] + design_scaled.shape)
    n = self.n_experiments
    # [A] runs from the (fixed, saturating) glucose stock down to zero.
    measurements = self._to_unit(event.measurements, (0.0, self.concentration_A))
    return jnp.concatenate([
      measurements,
      design_scaled[..., :n, None], design_scaled[..., n:2 * n, None],
      design_scaled[..., 2 * n:3 * n, None], design_scaled[..., 3 * n:4 * n, None]
    ], axis=-1)

  def element_mask(self, event, mask):
    return mask  # element == experiment

  @staticmethod
  def _to_unit(values, bounds):
    """Map ``[low, high]`` linearly onto ``[-1, 1]``."""
    low, high = bounds
    return (2.0 * values - (low + high)) / (high - low)

  def normalize_target(self, target):
    """IDENTITY on the one-hot label -- there is no scale to remove. The normalisation lives in the
    LOSS instead: softmax cross-entropy divided by ``ln(n_classes)``, whose no-information level is
    exactly 1.0 for any class prior that the predictor cannot beat, at ``1 / n_classes`` chance
    accuracy (``ln 3`` and 1/3 with the default three-class library, ``ln 2`` and 1/2 for the binary
    extremes)."""
    flat, _ = tensor.flatten(target)
    return flat

  def denormalize_predictions(self, normalised):
    """Logits -> class PROBABILITIES, the physical reading of a mechanism call."""
    probabilities = jax.nn.softmax(jnp.asarray(normalised, jnp.float32), axis=-1)
    return tensor.unflatten(tensor.structure(self.target_spec()), probabilities)

  def normalize_ground_truth(self, ground_truth):
    """Physical ``EnzymeInhibitorGroundTruth`` -> standardised flat ``(..., len(PARAMETER_NAMES) +
    3)``: every kinetic parameter by its own prior range, the half-conversion time by its log range,
    and each branch strength by the range the two class bands can reach."""
    low = jnp.asarray([r[0] for _, r in self._parameter_ranges], jnp.float32)
    high = jnp.asarray([r[1] for _, r in self._parameter_ranges], jnp.float32)
    log_half_time_bounds = (math.log(self.half_time_bounds[0]), math.log(self.half_time_bounds[1]))
    return jnp.concatenate([
      (2.0 * ground_truth.parameters - (low + high)) / (high - low),
      self._to_unit(jnp.log(ground_truth.half_time), log_half_time_bounds),
      self._to_unit(ground_truth.log10_k1, self.log10_branch_bounds),
      self._to_unit(ground_truth.log10_k2, self.log10_branch_bounds)
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    return f'cross-entropy / ln {self.n_classes} (inhibition mechanism)'

  def metric_labels(self):
    # `mechanism_classes`, NOT the global CLASS_NAMES: the label is the SLOT of the drawn class, so a
    # library of two classes has a 2x2 confusion whose slot 1 is whatever `mechanism_classes` names
    # there -- `mostly_uncompetitive` for the binary-extremes task, not `mostly_noncompetitive`.
    return ('loss', 'accuracy') + tuple(
      f'confusion_{truth}_{guess}' for truth in self.mechanism_classes for guess in self.mechanism_classes
    )

  def loss(self, predicted, target):
    """Per-sample softmax cross-entropy over the ``N_CLASSES`` logits, divided by ``ln N_CLASSES``.

    The uniform prediction scores exactly 1.0 on EVERY sample -- which is the point: an uninformative
    design's per-sample loss has ~zero variance, so its standard error collapses at once and the
    design stops being the expensive one. Under a squared-error target the same design costs the most
    (measured on the melting-point benchmark: 68266 calls to conclude nothing)."""
    return -jnp.sum(target * jax.nn.log_softmax(predicted, axis=-1), axis=-1) / math.log(self.n_classes)

  def metric(self, predicted, target):
    """Per-sample loss, accuracy and the ``n_classes x n_classes`` confusion matrix (as indicators, so
    their sample means are the JOINT distribution of truth x guess; a row divided by its sum is that
    class's recall). Rows and columns run over ``mechanism_classes``, in the same order as
    :meth:`metric_labels`, because the target's one-hot index is the SLOT in that tuple."""
    guess = jnp.argmax(predicted, axis=-1)
    truth = jnp.argmax(target, axis=-1)
    metrics = {'loss': self.loss(predicted, target), 'accuracy': (guess == truth).astype(jnp.float32)}
    for i, truth_name in enumerate(self.mechanism_classes):
      for j, guess_name in enumerate(self.mechanism_classes):
        metrics[f'confusion_{truth_name}_{guess_name}'] = ((truth == i) & (guess == j)).astype(jnp.float32)
    return metrics

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """Draw the (enzyme, compound) pairs at ``event_index`` and run each one's batch of experiments
    under ``design`` (one design broadcast over the batch, or one design per event). DETERMINISTIC:
    the enzyme, the compound and the read-out noise are seeded from ``event_index`` alone, so the
    same event under two designs is the same pair -- the target is a property of the event only."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    fraction, substrate, inhibitor, temperature = self._resolve_design(design, n)
    measurements, mechanism, parameters, half_time, log10_k1, log10_k2, error = self._generate(
      fraction, substrate, inhibitor, temperature, jnp.asarray(event_index, jnp.int32)
    )

    error = float(jnp.max(error))
    if not math.isfinite(error) or error > self.integration_tolerance:
      raise RuntimeError(
        f'integration error {error:.3g} mM over {n} events exceeds integration_tolerance '
        f'{self.integration_tolerance:.3g} mM -- the largest disagreement between the dt and dt/2 '
        f'chains on the extent of reaction at a read-out time; raise n_steps_per_measurement '
        f'(or check dt against scripts/check_stability.py).'
      )

    ground_truth = EnzymeInhibitorGroundTruth(
      parameters=parameters, half_time=half_time[:, None], log10_k1=log10_k1[:, None], log10_k2=log10_k2[:, None]
    )
    mask = jnp.ones((n, self.n_experiments), jnp.int32)
    return ground_truth, EnzymeInhibitorEvent(measurements=measurements), mask, EnzymeInhibitorTarget(mechanism=mechanism)

  def _resolve_design(self, design, n):
    """``EnzymeInhibitorDesign`` / config ``Mapping`` / flat array -> the four ``(n, n_experiments)``
    coordinate blocks (a single design is broadcast over the event batch)."""
    width = 4 * self.n_experiments
    flat = jnp.broadcast_to(jnp.reshape(self.flatten_design(design), (-1, width)), (n, width))
    m = self.n_experiments
    return flat[:, :m], flat[:, m:2 * m], flat[:, 2 * m:3 * m], flat[:, 3 * m:]

  def _initial_state(self, enzyme_fraction, substrate_B):
    """The concentrations at t = 0. All three are INDEPENDENT (no mixing constraint, see the module
    docstring): glucose is the fixed saturating stock, ATP comes from the design, and the enzyme
    stock is diluted by ``enzyme_fraction`` -- which dilutes NOTHING else."""
    return self.concentration_A, substrate_B, self.concentration_E * enzyme_fraction

  def _rate(self, extent, A0, B0, E0, inhibitor, temperature, parameters, log_k0_cat, k1, k2):
    """The reaction rate at extent ``x``: ``A = A0 - x``, ``B = B0 - x``, ``C = D = x``, E constant.
    ``log_k0_cat``, ``k1`` and ``k2`` are passed apart from ``parameters`` because the calibration
    evaluates the rate before the turnover is known and with no compound present."""
    return kinetics(
      A0 - extent, B0 - extent, extent, extent, E0, inhibitor, temperature,
      dict(parameters, log_k0_cat=log_k0_cat, k1=k1, k2=k2)
    )

  def _event(self, fraction, substrate, inhibitor, temperature, event_index):
    """One event: draw an enzyme and a compound, calibrate the enzyme's turnover, and run the batch.
    The design blocks are ``(n_experiments,)``; every draw uses ``event_index`` only."""
    key_parameters, key_half_time, key_compound, key_noise = jax.random.split(jax.random.PRNGKey(event_index), 4)
    parameters = self._draw_parameters(key_parameters)
    low, high = self.half_time_bounds
    half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=math.log(low), maxval=math.log(high)))
    mechanism, log10_k1, log10_k2 = self._draw_compound(key_compound)

    # The turnover is a property of the ENZYME: calibrated with no compound present, so the label
    # cannot leak into the timescale.
    log_k0_cat = self._calibrate(parameters, half_time)
    measurements, batch_error = self._run_batch(
      fraction, substrate, inhibitor, temperature, parameters, log_k0_cat,
      jnp.power(10.0, log10_k1), jnp.power(10.0, log10_k2), key_noise
    )

    packed = jnp.stack([parameters[name] for name in PARAMETER_NAMES])
    one_hot = jax.nn.one_hot(mechanism, self.n_classes, dtype=jnp.float32)
    return measurements, one_hot, packed, half_time, log10_k1, log10_k2, batch_error

  def _draw_parameters(self, key):
    """One enzyme: every parameter uniform within its own prior range."""
    keys = jax.random.split(key, len(self._parameter_ranges))
    return {
      name: jax.random.uniform(k, (), minval=low, maxval=high)
      for (name, (low, high)), k in zip(self._parameter_ranges, keys)
    }

  def _draw_compound(self, key):
    """One compound from the library: its mechanism CLASS (uniform over the three), the DOMINANT
    branch's potency, and the preference ``|d| = |log10(k1/k2)|`` from that class's band.

    The dominant branch is drawn first (rather than a geometric mean), which guarantees it is inside
    the detectable range; the weak branch may fall below detection, and that is not a defect -- a
    compound whose uncompetitive branch is invisible IS what "mostly competitive" means."""
    key_class, key_potency, key_preference, key_sign = jax.random.split(key, 4)
    # Draw a SLOT into `mechanism_classes` (the label, and the one-hot index); `actual` is the
    # global class id, which the sign/band logic below is written against.
    mechanism = jax.random.randint(key_class, (), 0, self.n_classes)
    actual = self.class_indices[mechanism]
    log10_k_strong = jax.random.uniform(
      key_potency, (), minval=self.log10_k_bounds[0], maxval=self.log10_k_bounds[1]
    )
    # Within 1.4x for the noncompetitive class, 10x-300x for the two typed ones; the band between
    # them is NOT sampled (checked in __init__).
    noncompetitive = (actual == NONCOMPETITIVE)
    low = jnp.where(noncompetitive, self.preference_bounds_noncompetitive[0], self.preference_bounds[0])
    high = jnp.where(noncompetitive, self.preference_bounds_noncompetitive[1], self.preference_bounds[1])
    preference = jax.random.uniform(key_preference, (), minval=0.0, maxval=1.0) * (high - low) + low
    # d = log10(k1/k2): positive means the compound prefers the free enzyme (competitive). The typed
    # classes fix the sign; a noncompetitive compound leans either way, at random.
    coin = jnp.where(jax.random.bernoulli(key_sign), 1.0, -1.0)
    sign = jnp.where(actual == COMPETITIVE, 1.0, jnp.where(actual == UNCOMPETITIVE, -1.0, coin))
    d = sign * preference
    # The dominant branch sits at k_strong, the other `preference` decades below it.
    return mechanism, log10_k_strong + jnp.minimum(d, 0.0), log10_k_strong - jnp.maximum(d, 0.0)

  def _calibrate(self, parameters, half_time):
    """``log_k0_cat`` putting the enzyme's FASTEST half-conversion time exactly at ``half_time``, on
    the reference (uninhibited) mixture.

    Turnover enters the rate as one multiplicative factor, so scaling ``k_cat`` scales every time by
    the inverse; in particular, which temperature reacts fastest does not depend on that scale. At
    ``log_k0_cat = 0`` the half-conversion time of the calibration mixture is

        tau(T) = int_0^{x_half} dx / rate(x, T),

    a plain quadrature over the extent -- no time stepping -- because the whole state is a function
    of it. Setting ``exp(log_k0_cat) = min_T tau(T) / half_time`` then rescales the fastest
    temperature onto ``half_time``. Without it a uniform prior over ``log_k0_cat`` would put most
    enzymes either finished or untouched within the hour, carrying no information either way."""
    A0, B0, E0 = self._initial_state(self.calibration_fraction, self.concentration_B)
    extent = self._quadrature_nodes * (0.5 * jnp.minimum(A0, B0))

    def half_conversion_time(temperature):
      return jnp.trapezoid(
        1.0 / self._rate(extent, A0, B0, E0, 0.0, temperature, parameters, 0.0, 0.0, 0.0), extent
      )

    fastest = self._fastest_temperature(half_conversion_time)
    return jnp.log(half_conversion_time(fastest)) - jnp.log(half_time)

  def _fastest_temperature(self, half_conversion_time):
    """The temperature within ``temperature_bounds`` minimising ``half_conversion_time``, by GOLDEN-
    SECTION search: ``n_temperature_steps`` bracket contractions, each costing ONE evaluation (the
    other interior point is reused) and shrinking the bracket by 0.618.

    ``log rate`` is a linear Arrhenius / van't Hoff term plus the concave ``-softplus`` of the
    Michaelis saturation, so the rate is log-concave in ``T`` and ``tau(T) = int dx / rate`` is
    log-convex with a single minimum -- which is what the search needs. Inside this box (below the
    melt) that minimum usually sits AT the warm edge; golden section handles a monotone function
    exactly as it handles an interior one, converging onto the edge."""
    low, high = (jnp.asarray(bound, jnp.float32) for bound in self.temperature_bounds)
    span = high - low
    # Interior points at the golden ratios, left < right.
    left, right = high - GOLDEN_SECTION * span, low + GOLDEN_SECTION * span
    f_left, f_right = half_conversion_time(left), half_conversion_time(right)

    def contract(carry, _):
      low, high, left, right, f_left, f_right = carry
      keep_left = f_left <= f_right  # better on the left -> the minimum lies in [low, right]
      low, high = jnp.where(keep_left, low, left), jnp.where(keep_left, right, high)
      # One interior point of the new bracket is an interior point of the old one: reuse it, and
      # evaluate only the point it does not cover.
      reused, f_reused = jnp.where(keep_left, left, right), jnp.where(keep_left, f_left, f_right)
      span = high - low
      fresh = jnp.where(keep_left, high - GOLDEN_SECTION * span, low + GOLDEN_SECTION * span)
      f_fresh = half_conversion_time(fresh)
      left, right = jnp.where(keep_left, fresh, reused), jnp.where(keep_left, reused, fresh)
      f_left, f_right = jnp.where(keep_left, f_fresh, f_reused), jnp.where(keep_left, f_reused, f_fresh)
      return (low, high, left, right, f_left, f_right), None

    start = (low, high, left, right, f_left, f_right)
    (low, high, left, right, f_left, f_right), _ = jax.lax.scan(
      contract, start, None, length=self.n_temperature_steps
    )
    return jnp.where(f_left <= f_right, left, right)

  def _run_batch(self, fraction, substrate, inhibitor, temperature, parameters, log_k0_cat, k1, k2, key):
    """The batch's read-out: [A] at every measurement time of every experiment, plus independent
    ``N(0, measurement_noise)`` read-out noise."""
    def run(enzyme_fraction, substrate_B, experiment_inhibitor, experiment_temperature):
      A0, B0, E0 = self._initial_state(enzyme_fraction, substrate_B)
      extent, error = self._integrate(
        A0, B0, E0, experiment_inhibitor, experiment_temperature, parameters, log_k0_cat, k1, k2,
        n_steps=self.n_steps_per_measurement, n_intervals=self.n_measurements
      )
      return A0 - extent, error

    concentration, errors = jax.vmap(run)(fraction, substrate, inhibitor, temperature)
    measurements = concentration + self.measurement_noise * jax.random.normal(key, concentration.shape)
    return measurements, jnp.max(errors)

  def rkc2_step(self, rate, extent, dt):
    """One RKC2 step of size ``dt`` on ``dx/dt = rate(x)`` from ``extent``, with this detector's own
    coefficients: :func:`detopt.detector.enzyme.rkc2_step`, which is what the bare name below resolves
    to (a method body does not see class scope). Kept as a method because the stability probes reach
    for it through the built detector."""
    return rkc2_step(self._rkc2, rate, extent, dt)

  def _chain(self, rate, *, dt, n_steps, n_intervals):
    """One RKC2 chain: ``n_intervals`` intervals of ``n_steps`` steps of ``dt``, returning the extent
    of reaction at the END of every interval (the measurement times). The extent starts at zero."""
    return rkc2_chain(self._rkc2, rate, jnp.zeros(()), dt=dt, n_steps=n_steps, n_intervals=n_intervals)

  def _integrate(self, A0, B0, E0, inhibitor, temperature, parameters, log_k0_cat, k1, k2, *, n_steps, n_intervals):
    """RKC2 on the extent of reaction over ``duration``, split into ``n_intervals`` equal intervals.
    Returns the extent at the END of every interval (the measurement times) and the integration
    error, ESTIMATED INSIDE THE SOLVE.

    Two chains run over the same interval grid -- one at ``dt``, one at ``dt/2`` -- and the error is
    the largest ``|fine - coarse|`` over the interval ends. That is a GLOBAL error estimate on the
    quantity actually consumed downstream (the [A] the network reads), not a per-step local
    truncation estimate: a local estimate can stay small while the accumulated trajectory drifts.

    The ``dt`` chain is what is RETURNED, paired with that error -- the solution reported is the one
    the error was measured on, rather than a finer solution whose own error is only inferred. A
    jitted kernel cannot raise, so the assertion against ``integration_tolerance`` lives outside the
    solve, in :meth:`__call__`, and runs on every call.

    ``dt`` must keep ``dt * |df/dx|`` inside the scheme's measured stability boundary everywhere the
    prior can reach; the rate law here is GENTLER than the baseline enzyme's (the uncompetitive
    factor only ever slows it down), which is re-measured rather than inherited."""
    dt = self.duration / (n_intervals * n_steps)
    rate = lambda x: self._rate(x, A0, B0, E0, inhibitor, temperature, parameters, log_k0_cat, k1, k2)

    coarse = self._chain(rate, dt=dt, n_steps=n_steps, n_intervals=n_intervals)
    fine = self._chain(rate, dt=0.5 * dt, n_steps=2 * n_steps, n_intervals=n_intervals)
    return coarse, jnp.max(jnp.abs(fine - coarse))
