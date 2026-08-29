"""Two-substrate Michaelis-Menten depletion: A + B -> C + D via E, three kinetic parameters.

    v = q [A] [B] / ((K_A + [A]) (K_B + [B])),     d[A]/dt = d[B]/dt = -v

The reaction is hexokinase's -- glucose (A) + ATP (B) -> glucose-6-phosphate + ADP -- and the
parameter priors and assay scales come from ``config/detector/enzyme.yaml``, which models the same
reaction and carries their provenance. FAITHFUL: the reaction, its 1:1 stoichiometry, the priors
and the scales. SIMPLIFICATION: the deliberate absence of every nuisance that file models -- no
temperature dependence, no product inhibition, no enzyme unfolding, no turnover calibration. The
absence of nuisance is the point of this task.

The PRODUCT form is rapid-equilibrium random binding of two substrates: either substrate may bind
first, the enzyme turns over only from the ternary complex, and the two binding steps are
independent. It carries exactly the three parameters the target names -- a maximum velocity and one
Michaelis constant per substrate -- and NO INHIBITION TERM, none implied and none to be added.

⚠️ THIS IS NOT THE PING-PONG BI-BI LAW ``q A B / (K_A B + K_B A + A B)``, which this task used until
its denominator was found to be missing the constant term ``K_A K_B``. The two are NOT the same
function and the difference is not a detail:

* ping-pong is SINGULAR at ``A = B = 0`` (both numerator and denominator vanish) and returned NaN on
  184 of 400 diagonal designs; the product form's denominator is ``K_A K_B`` there, so it is finite
  everywhere and the reciprocal-form workaround that once papered over this is DELETED;
* ping-pong makes the diagonal ``A0 == B0`` an EXACT degeneracy in which only ``K_A + K_B`` is
  identifiable. Under the product form that degeneracy does not exist -- equal-sum parameter pairs
  separate by 0.7-1.8x the read-out noise on the diagonal -- so the old reason to build the design
  box AWAY from the diagonal is gone.

Correcting it did NOT add a parameter: both laws have exactly ``(q, K_A, K_B)``. The old
justification, that the alternative "adds a fourth constant and makes the target four-dimensional",
was simply false, and hexokinase forms a ternary complex, so the product form is also the right
mechanism for this enzyme. Everything calibrated under the old law is quarantined in
``output/enzyme-depletion-bi/stale-ping-pong-law/``.

The stoichiometry is 1:1, so ``[B] = [A] + (B0 - A0)`` for all time and the system is ONE ODE. It is
integrated on the EXTENT of reaction ``x = A0 - [A] = B0 - [B] = [C] = [D]``, which is symmetric in
the two substrates.

* **design** -- the initial concentrations ``(A0, B0)`` of every experiment, and nothing else, so a
  batch of ``m`` experiments is ``2 m`` coordinates. The batch is a SET: permuting the experiments
  names the same design.
* **event** -- one unknown enzyme variant drawn from a wide prior, read out as ``n_measurements``
  noisy samples of the extent on an even grid ending at ``duration``. The draw depends on
  ``event_index`` ALONE, so the same variant is re-measured under every design (common random
  numbers) and the target is a property of the event, never of the design.
* **target** -- ``(ln q, ln K_A, ln K_B)``, logs because the measurement says so and not by taste;
  ``scripts/calibrate_enzyme_depletion_bi.py --section parameterisation`` reports the table that
  decides it.
* **loss** -- mean squared error of the three components mapped affinely onto ``[-1, 1]`` over their
  log prior ranges, so a prior-mean guess scores exactly ``1/3``
  (:meth:`EnzymeDepletionBiDetector.no_information_loss`) whatever the bounds are.

Reading the extent rather than ``[A]`` costs nothing and hides nothing: ``A0`` is part of the design
and therefore known exactly, so ``[A] = A0 - x`` is the same measurement up to a known constant. It
is preferred only because it treats the two substrates alike -- an absolute read-out noise on ``[A]``
would make an A-in-excess experiment look uninformative for a reason that is a choice of observable
rather than chemistry.

THE DESIGN PROBLEM. Pushing ``B0`` far above ``A0`` saturates B and leaves single-substrate kinetics
in A, which identifies ``K_A`` and abandons ``K_B``; the mirror design does the mirror thing. A batch
therefore has to break the A/B symmetry and to break it in both directions, while still depleting far
enough for the constants to bite -- and the two constants sit about five-fold apart, so the two knees
are at different concentration scales and no single pair of concentrations is informative for the
whole prior.

The diagonal ``A0 == B0`` is WEAK BUT NOT DEGENERATE. The two substrates track each other exactly and
the rate becomes ``q a^2 / ((K_A + a)(K_B + a))``, which still depends on the two constants
separately through their product, so equal-sum pairs separate; measured, that separation is 0.7-1.8x
the read-out noise on the diagonal, against 1.3-4.6x THAT for the off-diagonal designs. Symmetry
breaking is therefore still worth designing for, but the diagonal is not a hole in the box.

Numerics
--------
The integrator is **RKC2** (:func:`detopt.detector.enzyme.rkc2_chain`), shared with the other
enzyme-family detectors as a free function. The rate has POLES where the denominator vanishes,
outside the physical range but reachable by a step that overshoots a nearly exhausted substrate past
zero, so the step is set by that rather than by the linear stability boundary. The rate is
`q A B / ((K_A + A)(K_B + B))` -- rapid-equilibrium random binding of two substrates, three parameters
`(q, K_A, K_B)`, no inhibitor and none implied. Its denominator carries the constant term `K_A K_B`,
so it is finite everywhere including `A = B = 0`; there is NO diagonal singularity and no reciprocal
rewrite is needed. Every integration
runs a ``dt`` and a ``dt/2`` chain over the same read-out times, RETURNS THE ``dt`` ONE, and reports
the largest ``|fine - coarse|`` there; :meth:`EnzymeDepletionBiDetector.__call__` and
:meth:`EnzymeDepletionBiDetector.estimate` assert on it against ``integration_tolerance`` host-side,
because a jitted kernel cannot raise.

The reference the integrator is checked against is an implicit relation inverted by root-finding
(bisection + Newton, float64) -- a closed-form QUADRATURE, not a closed-form solution: separation
gives t(x) explicitly, but x(t) is not elementary. It is an independent numerical route with a
different error mode, NOT an error-free twin.
Separating variables in the single collapsed ODE gives

    q t = x + K_A ln(A0/(A0 - x)) + K_B ln(B0/(B0 - x))
            + (K_A K_B / D) ln( A0 (B0 - x) / (B0 (A0 - x)) ),        D = B0 - A0,

with the finite D -> 0 branch on the diagonal

    q t = x + (K_A + K_B) ln(A0/(A0 - x)) + K_A K_B (1/(A0 - x) - 1/A0),

an implicit, strictly monotone relation solved for ``x`` by the calibration script and the tests.

Every constant this detector takes is calibrated by `scripts/calibrate_enzyme_depletion_bi.py`,
which writes its numbers to json beside a figure per decision drawn by
`scripts/plot_enzyme_depletion_bi.py`; `config/detector/enzyme_depletion_bi_m2.yaml` carries the
resulting evidence inline.

The estimator
-------------
:meth:`EnzymeDepletionBiDetector.estimate` is the posterior mean of ``(ln q, ln K_A, ln K_B)`` on an
even lattice over the prior box under the Gaussian read-out likelihood -- the Bayes estimator for
squared error, so it cannot score worse than the prior and its landscape is monotone in information.

THE LATTICE IS ANISOTROPIC, and that is the whole design of it. The velocity is determined far more
sharply than either Michaelis constant -- the Cramer-Rao standard deviation of ``ln q`` is a few
parts in a thousand where the constants' are tenths -- so an ISOTROPIC lattice fine enough for the
constants is far coarser than the posterior's ridge in ``q``. The softmax then puts essentially all
weight on the single node nearest that ridge, the marginalisation over ``q`` is lost, and the
resulting estimate of the CONSTANTS is both biased and over-confident: it scores ABOVE the prior,
which a Bayes estimator cannot do. That is a sharp, self-contained test of whether the lattice is
adequate, needing no reference value, and it is what ``n_grid_velocity`` exists to pass.
"""

import math
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .common import Detector
from .enzyme import rkc2_chain, rkc2_coefficients
from ..utils import tensor

__all__ = [
  'EnzymeDepletionBiDetector', 'EnzymeDepletionBiDesign', 'EnzymeDepletionBiEvent', 'EnzymeDepletionBiTarget',
  'EnzymeDepletionBiGroundTruth', 'PARAMETER_NAMES',
]

PARAMETER_NAMES = ('velocity', 'michaelis_a', 'michaelis_b')


class EnzymeDepletionBiDesign(NamedTuple):
  """The whole design: both initial concentrations of every experiment, NOMINAL, in mM."""
  initial_a: jax.Array  # (n_experiments,)
  initial_b: jax.Array  # (n_experiments,)


class EnzymeDepletionBiEvent(NamedTuple):
  """The read-out: the noisy extent of reaction of every experiment at every sampling time."""
  extent: jax.Array  # (n_experiments, n_measurements)


class EnzymeDepletionBiTarget(NamedTuple):
  """What is estimated: the variant's own kinetics, PHYSICAL, ``(q, K_A, K_B)`` in (mM/s, mM, mM)."""
  kinetics: jax.Array  # (3,)


class EnzymeDepletionBiGroundTruth(NamedTuple):
  """The drawn variant (== conditioning). The target is the whole of it here."""
  kinetics: jax.Array  # (3,)


def _log_scale(values, bounds):
  """NOMINAL -> SCALED ``[0, 1]``, affine on the LOG of a strictly positive range."""
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return (jnp.log(jnp.asarray(values, jnp.float32)) - low) / (high - low)


def _log_unscale(scaled, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL (inverse of :func:`_log_scale`)."""
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return jnp.exp(jnp.asarray(scaled, jnp.float32) * (high - low) + low)


class EnzymeDepletionBiDetector(Detector):
  """``n_experiments`` two-substrate depletion curves under one unknown enzyme variant.

  Every constant is a constructor argument, i.e. lives in the yaml config: the batch size, the assay
  (duration, sampling count, read-out noise), the three parameter priors, the design box, the
  integrator settings and the estimator grid.
  """

  def __init__(
    self, *, n_experiments: int = 2, n_measurements: int = 10, duration: float = 3600.0,
    velocity_bounds: tuple = (9.0e-4, 9.0e-3), michaelis_a_bounds: tuple = (0.02, 0.2), michaelis_b_bounds: tuple = (0.1, 1.0),
    concentration_a_bounds: tuple = (0.8, 10.0), concentration_b_bounds: tuple = (1.0, 25.0), measurement_noise: float = 0.02,
    n_stages: int = 5, steps_per_measurement: int = 64, damping: float = 2.0 / 13.0, integration_tolerance: float = 5.0e-3,
    n_grid: int = 15, n_grid_velocity: int = 241
  ):
    self.n_experiments = int(n_experiments)
    self.n_measurements = int(n_measurements)
    self.duration = float(duration)
    self.velocity_bounds = (float(velocity_bounds[0]), float(velocity_bounds[1]))
    self.michaelis_a_bounds = (float(michaelis_a_bounds[0]), float(michaelis_a_bounds[1]))
    self.michaelis_b_bounds = (float(michaelis_b_bounds[0]), float(michaelis_b_bounds[1]))
    self.concentration_a_bounds = (float(concentration_a_bounds[0]), float(concentration_a_bounds[1]))
    self.concentration_b_bounds = (float(concentration_b_bounds[0]), float(concentration_b_bounds[1]))
    self.measurement_noise = float(measurement_noise)
    self.n_stages = int(n_stages)
    self.steps_per_measurement = int(steps_per_measurement)
    self.damping = float(damping)
    self.integration_tolerance = float(integration_tolerance)
    self.n_grid = int(n_grid)
    self.n_grid_velocity = int(n_grid_velocity)

    if self.n_experiments < 1:
      raise ValueError(f'n_experiments must be at least 1, got {n_experiments}')
    if self.n_measurements < 2:
      raise ValueError(f'n_measurements must be at least 2 to see a curve, got {n_measurements}')
    if not self.duration > 0.0:
      raise ValueError(f'duration must be strictly positive, got {duration}')
    if not self.measurement_noise > 0.0:
      raise ValueError(f'measurement_noise is a standard deviation and must be strictly positive, '
                       f'got {measurement_noise}')
    if self.n_grid < 2:
      raise ValueError(f'n_grid must be at least 2, got {n_grid}')
    if self.n_grid_velocity < 2:
      raise ValueError(f'n_grid_velocity must be at least 2, got {n_grid_velocity}')
    for name, bounds in (('velocity_bounds', self.velocity_bounds), ('michaelis_a_bounds', self.michaelis_a_bounds),
                         ('michaelis_b_bounds', self.michaelis_b_bounds),
                         ('concentration_a_bounds', self.concentration_a_bounds), ('concentration_b_bounds',
                                                                                   self.concentration_b_bounds)):
      if not 0.0 < bounds[0] < bounds[1]:
        raise ValueError(f'{name} must be a strictly positive increasing (low, high), got {bounds}')

    self._rkc2 = rkc2_coefficients(self.n_stages, self.damping)
    self._parameter_bounds = (self.velocity_bounds, self.michaelis_a_bounds, self.michaelis_b_bounds)
    axes = (
      np.linspace(-1.0, 1.0, self.n_grid_velocity), np.linspace(-1.0, 1.0, self.n_grid), np.linspace(-1.0, 1.0, self.n_grid)
    )
    grid_z = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    self._grid_scaled = jnp.asarray(grid_z, jnp.float32)
    self._grid_kinetics = jnp.stack([
      _log_unscale(0.5 * (self._grid_scaled[:, i] + 1.0), bounds) for i, bounds in enumerate(self._parameter_bounds)
    ], axis=-1)
    self._generate = jax.jit(jax.vmap(self._event, in_axes=(0, 0, 0)))
    self._posterior = jax.jit(self._posterior_mean)

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return EnzymeDepletionBiEvent(extent=jax.ShapeDtypeStruct((self.n_experiments, self.n_measurements), np.float32))

  def target_spec(self):
    return EnzymeDepletionBiTarget(kinetics=jax.ShapeDtypeStruct((3, ), np.float32))

  def ground_truth_spec(self):
    return EnzymeDepletionBiGroundTruth(kinetics=jax.ShapeDtypeStruct((3, ), np.float32))

  def design_shape(self):
    return (2 * self.n_experiments, )

  def design_spec(self):
    return EnzymeDepletionBiDesign(
      initial_a=jax.ShapeDtypeStruct((self.n_experiments, ), np.float32), initial_b=jax.ShapeDtypeStruct((self.n_experiments, ),
                                                                                                         np.float32)
    )

  def design_bounds(self):
    return {'initial_a': self.concentration_a_bounds, 'initial_b': self.concentration_b_bounds}

  def combined_event_shape(self, design: bool = True):
    return (self.n_experiments, self.n_measurements + (2 if design else 0))

  def size(self):
    return None  # an analytic source: every index is a fresh variant

  # ------------------------------------------------------------------ #
  # Design scaling: affine on the LOG concentration, with its OWN box per substrate.
  #
  # What an experiment resolves is the RATIO of a concentration to a Michaelis constant, and the
  # ratio B0/A0 that breaks the A/B degeneracy, so equal steps of the scaled coordinate are equal
  # steps of what is resolved; a linear map would spend most of the cube on the top of the box. The
  # two substrates get separate boxes because their priors differ by an order of magnitude, and a
  # shared box would put one substrate's informative window in a corner of the cube.
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    design = jnp.asarray(design, jnp.float32)
    return jnp.concatenate([
      _log_scale(design[..., :self.n_experiments], self.concentration_a_bounds),
      _log_scale(design[..., self.n_experiments:], self.concentration_b_bounds),
    ], axis=-1)

  def _to_nominal_flat(self, design_scaled):
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    return jnp.concatenate([
      _log_unscale(design_scaled[..., :self.n_experiments], self.concentration_a_bounds),
      _log_unscale(design_scaled[..., self.n_experiments:], self.concentration_b_bounds),
    ], axis=-1)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
    """``features (..., n_experiments, n_measurements + 2)``: each experiment's extent readings on a
    FIXED scale, followed by its two scaled concentrations.
    ⚠️ THE READINGS ARE SCALED BY A FIXED REFERENCE -- the TOP OF THE DESIGN BOX -- and NOT by each
    experiment's own limiting concentration min(A0, B0). Dividing by the experiment's own value was the old rule
    and it is refuted: the read-out noise is ABSOLUTE (a standard deviation in concentration units,
    added to the clean signal), so dividing by the design multiplies that noise by its reciprocal and
    the FEATURE noise then varies across the design box by the box's own ratio. The information is not
    lost -- the limiting concentration min(A0, B0) is appended as its own feature -- but the conditioning is, and a
    design at the bottom of the box arrives with the noisiest features for no physical reason. This is
    the same rule, and the same reason, as :mod:`detopt.detector.enzyme_mm`.

    Withholding the design drops the two trailing columns and nothing else: the readings never
    depended on them.

    ``mask`` is unused -- every experiment of the design is real (the element axis is the design's,
    not a hit count)."""
    extent = jnp.asarray(event.extent, jnp.float32)
    readings = extent
    if design_scaled is None or not reveal_design:
      return readings
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    if design_scaled.ndim == 1:  # one design for the whole event batch
      design_scaled = jnp.broadcast_to(design_scaled, extent.shape[:-2] + design_scaled.shape)
    scaled_a, scaled_b = jnp.split(design_scaled, 2, axis=-1)
    return jnp.concatenate([readings, scaled_a[..., None], scaled_b[..., None]], axis=-1)

  def element_mask(self, event, mask):
    return mask  # element == experiment

  def normalize_target(self, target):
    kinetics = jnp.asarray(target.kinetics, jnp.float32)
    return jnp.stack([2.0 * _log_scale(kinetics[..., i], bounds) - 1.0 for i, bounds in enumerate(self._parameter_bounds)],
                     axis=-1)

  def denormalize_predictions(self, normalised):
    normalised = jnp.asarray(normalised, jnp.float32)
    kinetics = jnp.stack([
      _log_unscale(0.5 * (normalised[..., i] + 1.0), bounds) for i, bounds in enumerate(self._parameter_bounds)
    ], axis=-1)
    return tensor.unflatten(tensor.structure(self.target_spec()), kinetics)

  def normalize_ground_truth(self, ground_truth):
    return self.normalize_target(EnzymeDepletionBiTarget(kinetics=ground_truth.kinetics))

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    return 'MSE (ln q, ln K_A, ln K_B; log-uniform prior on [-1, 1])'

  def metric_labels(self):
    return ('loss', ) + PARAMETER_NAMES

  def loss(self, predicted, target):
    return jnp.mean(jnp.square(predicted - target), axis=-1)

  def metric(self, predicted, target):
    squared = jnp.square(predicted - target)
    return {
      'loss': jnp.mean(squared, axis=-1),
      'velocity': squared[..., 0],
      'michaelis_a': squared[..., 1],
      'michaelis_b': squared[..., 2]
    }

  def metric_real_rmse(self, metric_means):
    """Sample-averaged per-parameter metric -> RMSE as a natural-log factor on the physical value."""
    spans = tuple(math.log(bounds[1] / bounds[0]) for bounds in self._parameter_bounds)
    return {name: (float(np.sqrt(metric_means[name]) * 0.5 * span), 'ln') for name, span in zip(PARAMETER_NAMES, spans)}

  def no_information_loss(self):
    """The loss of a GUESS: the best constant prediction under the prior, in closed form.

    All three targets are uniform on ``[-1, 1]`` after :meth:`normalize_target` (the priors are
    log-uniform and the map is affine in the log), the best constant is the mean, and the variance of
    ``U[-1, 1]`` is ``1/3`` -- so this is ``1/3`` exactly, whatever the bounds are. It is the
    denominator of the campaign criterion and the ceiling every estimate is measured against."""
    return 1.0 / 3.0

  # ------------------------------------------------------------------ #
  # Integration
  # ------------------------------------------------------------------ #
  def _integrate(self, initial_a, initial_b, velocity, michaelis_a, michaelis_b):
    """The extent of reaction at every read-out time for one experiment, plus the largest
    ``|dt/2 - dt|`` there.

    Returns the ``dt`` chain -- the solution the error was measured ON -- and its error monitor. All
    arguments are scalars; the caller vmaps.

    THE RATE IS WRITTEN AS THE PRODUCT FORM ``q A B / ((K_A + A)(K_B + B))`` AND NEEDS NO REWRITE.
    Its denominator is ``K_A K_B > 0`` at ``A = B = 0``, so it is finite everywhere in the physical
    range including the diagonal, where both substrates exhaust simultaneously. The reciprocal
    rewrite ``q / (K_A/A + K_B/B + 1)`` that used to live here belonged to the PING-PONG law, whose
    denominator vanishes with its numerator there and returned NaN on 184 of 400 diagonal cases; it
    is not an alternative spelling of this rate and must not be reintroduced."""

    def rate(extent):
      left_a = initial_a - extent
      left_b = initial_b - extent
      return velocity * left_a * left_b / ((michaelis_a + left_a) * (michaelis_b + left_b))

    dt = self.duration / (self.n_measurements * self.steps_per_measurement)
    start = jnp.zeros((), jnp.float32)
    coarse = rkc2_chain(self._rkc2, rate, start, dt=dt, n_steps=self.steps_per_measurement, n_intervals=self.n_measurements)
    fine = rkc2_chain(
      self._rkc2, rate, start, dt=0.5 * dt, n_steps=2 * self.steps_per_measurement, n_intervals=self.n_measurements
    )
    return coarse, jnp.max(jnp.abs(fine - coarse))

  def _check(self, error, where):
    error = float(jnp.max(error))
    if not math.isfinite(error) or error > self.integration_tolerance:
      raise RuntimeError(
        f'integration error {error:.3g} mM in {where} exceeds integration_tolerance '
        f'{self.integration_tolerance:.3g} mM -- the largest disagreement between the dt and dt/2 '
        f'chains on the extent of reaction at a read-out time; raise steps_per_measurement (the rate '
        f'has poles just outside the physical range, so a step that overshoots an exhausted '
        f'substrate runs away rather than merely losing accuracy).'
      )
    return error

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """Draw the variants at ``event_index`` and run each one's batch of experiments under ``design``
    (one design broadcast over the batch, or one design per event). DETERMINISTIC: the variant and
    its read-out noise are seeded from ``event_index`` alone, so the same event under two designs is
    the same enzyme and the target is a property of the event only."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    initial_a, initial_b = self._resolve_design(design, n)
    extent, kinetics, error = self._generate(initial_a, initial_b, jnp.asarray(event_index, jnp.int32))
    self._check(error, 'the event batch')
    return (
      EnzymeDepletionBiGroundTruth(kinetics=kinetics), EnzymeDepletionBiEvent(extent=extent),
      jnp.ones((n, self.n_experiments), jnp.int32), EnzymeDepletionBiTarget(kinetics=kinetics)
    )

  def _resolve_design(self, design, n):
    """Any accepted NOMINAL design form -> two ``(n, n_experiments)`` initial-concentration arrays."""
    flat = jnp.reshape(jnp.asarray(self.flatten_design(design), jnp.float32), (-1, self.design_dim()))
    broadcast = jnp.broadcast_to(flat, (n, self.design_dim()))
    return broadcast[:, :self.n_experiments], broadcast[:, self.n_experiments:]

  def _event(self, initial_a, initial_b, event_index):
    """One variant: draw ``(q, K_A, K_B)`` from the log-uniform prior and read every experiment out.

    The read-out noise rows are handed out by the RANK of the experiment in the batch's own
    lexicographic order, not by its position in the design vector. The design is a SET, so the
    objective has to be exactly invariant under permuting it; with positional noise the same batch
    written in two orders draws two different realisations and the two evaluations disagree, which is
    a split on an identical experiment rather than a real difference. Ranking makes each physical
    experiment keep its own noise however the batch is written, while two experiments at the SAME
    concentrations still get independent rows, so a replicate still buys information. The tie locus
    is a discontinuity of measure zero -- the same trade ``SortingRBF`` documents."""
    key_parameters, key_noise = jax.random.split(jax.random.PRNGKey(event_index), 2)
    unit = jax.random.uniform(key_parameters, (3, ), jnp.float32)
    kinetics = jnp.stack([_log_unscale(unit[i], bounds) for i, bounds in enumerate(self._parameter_bounds)])
    clean, error = jax.vmap(self._integrate,
                            in_axes=(0, 0, None, None, None))(initial_a, initial_b, kinetics[0], kinetics[1], kinetics[2])
    rank = jnp.argsort(jnp.lexsort((initial_b, initial_a)))
    noise = self.measurement_noise * jax.random.normal(key_noise, clean.shape, jnp.float32)[rank]
    return clean + noise, kinetics, jnp.max(error)

  # ------------------------------------------------------------------ #
  # The estimator
  # ------------------------------------------------------------------ #
  def _posterior_mean(self, initial_a, initial_b, extent):
    """Posterior mean of the SCALED ``(ln q, ln K_A, ln K_B)`` on the prior grid, for a batch of
    read-outs.

    ``initial_a`` / ``initial_b`` are ``(n_experiments,)`` NOMINAL, ``extent`` is ``(B,
    n_experiments, n_measurements)``. The model curves are integrated once for the whole batch, since
    they depend on the design and not on the event."""
    model, error = jax.vmap(
      lambda kinetics: jax.vmap(self._integrate, in_axes=(0, 0, None, None, None))
      (initial_a, initial_b, kinetics[0], kinetics[1], kinetics[2])
    )(self._grid_kinetics)
    flat_model = jnp.reshape(model, (self._grid_kinetics.shape[0], -1))
    flat_data = jnp.reshape(extent, (extent.shape[0], -1))
    # -|y - m|^2 / 2 sigma^2 up to a term constant in the grid, as one matmul.
    log_likelihood = (flat_data @ flat_model.T - 0.5 * jnp.sum(jnp.square(flat_model), axis=-1)[None, :])
    log_likelihood = log_likelihood / (self.measurement_noise**2)
    weight = jax.nn.softmax(log_likelihood, axis=-1)
    return weight @ self._grid_scaled, jnp.max(error)

  def estimate(self, design, event):
    """The instrument: NOMINAL ``design`` + an ``EnzymeDepletionBiEvent`` batch -> predictions in the
    NORMALISED target space, ``(B, 3)``, directly comparable with :meth:`normalize_target`.

    The posterior mean of the grid (see the module docstring); asserts the integration error of the
    model curves the same way :meth:`__call__` does for the events."""
    extent = jnp.asarray(event.extent, jnp.float32)
    flat = jnp.reshape(jnp.asarray(self.flatten_design(design), jnp.float32), (self.design_dim(), ))
    predicted, error = self._posterior(flat[:self.n_experiments], flat[self.n_experiments:], extent)
    self._check(error, 'the estimator grid')
    return predicted
