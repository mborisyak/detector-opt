"""Michaelis-Menten depletion of a single substrate: the simplest real parameter-estimation design.

    dA/dt = -q A / (A + K),      A(0) = A0

One substrate, one enzyme, nothing else -- no temperature, no pH, no inhibitor, no co-substrate.

* **design** -- the initial concentrations ``A0``, one per experiment, and nothing else. The batch
  is a SET: permuting the experiments names the same design.
* **event** -- one unknown enzyme variant drawn from a deliberately wide prior, read out as
  ``n_measurements`` noisy samples of ``[A]`` on an even grid ending at ``duration``. The draw
  depends on ``event_index`` ALONE, so the same variant is re-measured under every design (common
  random numbers) and the target is a property of the event, never of the design.
* **target** -- ``(ln q, ln K)``. Both are LOG parameters because the measurement says so, not by
  taste: binned by the true value, the estimator's ABSOLUTE error tracks the parameter while its LOG
  error does not, so the log is the coordinate on which the error is scale-free.
  ``scripts/calibrate_enzyme_depletion.py --section parameterisation`` reports that table.
* **loss** -- mean squared error of the two components mapped affinely onto ``[-1, 1]`` over their
  log prior range, so a prior-mean guess scores exactly ``1/3``
  (:meth:`EnzymeDepletionDetector.no_information_loss`) whatever the bounds are.

The design problem is the width of the ``K`` prior: it spans more than three decades, and an
initial concentration that puts the knee ``A ~ K`` inside the time window for a low-K variant
leaves a high-K one in the un-depleted regime, where only ``q/K`` is identifiable. No single
concentration is informative for the whole prior, so a batch has to spread.

Numerics
--------
The integrator is **RKC2** (:func:`detopt.detector.enzyme.rkc2_chain`), shared with the other
enzyme-family detectors. The rate has a POLE at ``A = -K``: a step that overshoots a nearly
depleted substrate past zero lands beyond it and the trajectory runs away, so the step is set by
that, not by the linear stability boundary. Every integration runs a ``dt`` and a ``dt/2`` chain
over the same read-out times, RETURNS THE ``dt`` ONE, and reports the largest ``|fine - coarse|``
there; :meth:`EnzymeDepletionDetector.__call__` and :meth:`EnzymeDepletionDetector.estimate` assert
on it against ``integration_tolerance`` host-side, because a jitted kernel cannot raise.

The estimator
-------------
:meth:`EnzymeDepletionDetector.estimate` is the posterior mean of ``(ln q, ln K)`` on an even grid
over the prior box under the Gaussian read-out likelihood -- the Bayes estimator for squared error,
so it cannot score worse than the prior and its landscape is monotone in information. The closed
form of the integrated Michaelis-Menten equation was implemented and MEASURED first
(``scripts/calibrate_enzyme_depletion.py --section estimator``) and rejected: the linearisation
puts the read-out noise in the regressor as well as the response, and the resulting
errors-in-variables bias makes it score ABOVE the no-information level and grow WORSE with more
measurements. Reweighting by the correct residual variance and re-substituting the fitted curve
both failed to remove it.
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
  'EnzymeDepletionDetector', 'EnzymeDepletionDesign', 'EnzymeDepletionEvent', 'EnzymeDepletionTarget',
  'EnzymeDepletionGroundTruth', 'PARAMETER_NAMES',
]

PARAMETER_NAMES = ('velocity', 'michaelis')


class EnzymeDepletionDesign(NamedTuple):
  """The whole design: the initial substrate concentration of every experiment, NOMINAL, in mM."""
  initial_concentration: jax.Array  # (n_experiments,)


class EnzymeDepletionEvent(NamedTuple):
  """The read-out: the noisy substrate concentration of every experiment at every sampling time."""
  concentration: jax.Array  # (n_experiments, n_measurements)


class EnzymeDepletionTarget(NamedTuple):
  """What is estimated: the variant's own kinetics, PHYSICAL, ``(q, K)`` in (mM/s, mM)."""
  kinetics: jax.Array  # (2,)


class EnzymeDepletionGroundTruth(NamedTuple):
  """The drawn variant (== conditioning). The target is the whole of it here."""
  kinetics: jax.Array  # (2,)


def _log_scale(values, bounds):
  """NOMINAL -> SCALED ``[0, 1]``, affine on the LOG of a strictly positive range."""
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return (jnp.log(jnp.asarray(values, jnp.float32)) - low) / (high - low)


def _log_unscale(scaled, bounds):
  """SCALED ``[0, 1]`` -> NOMINAL (inverse of :func:`_log_scale`)."""
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return jnp.exp(jnp.asarray(scaled, jnp.float32) * (high - low) + low)


class EnzymeDepletionDetector(Detector):
  """``n_experiments`` depletion curves under one unknown Michaelis-Menten variant.

  Every constant is a constructor argument, i.e. lives in the yaml config: the batch size, the
  assay (duration, sampling count, read-out noise), the two parameter priors, the design box, the
  integrator settings and the estimator grid.
  """

  def __init__(
    self, *, n_experiments: int = 1, n_measurements: int = 8, duration: float = 21600.0,
    velocity_bounds: tuple = (1.0e-4, 1.0e-3), michaelis_bounds: tuple = (0.01, 20.0),
    concentration_bounds: tuple = (0.45, 4.0), measurement_noise: float = 0.02, n_stages: int = 5,
    steps_per_measurement: int = 64, damping: float = 2.0 / 13.0, integration_tolerance: float = 2.0e-3, n_grid: int = 65
  ):
    self.n_experiments = int(n_experiments)
    self.n_measurements = int(n_measurements)
    self.duration = float(duration)
    self.velocity_bounds = (float(velocity_bounds[0]), float(velocity_bounds[1]))
    self.michaelis_bounds = (float(michaelis_bounds[0]), float(michaelis_bounds[1]))
    self.concentration_bounds = (float(concentration_bounds[0]), float(concentration_bounds[1]))
    self.measurement_noise = float(measurement_noise)
    self.n_stages = int(n_stages)
    self.steps_per_measurement = int(steps_per_measurement)
    self.damping = float(damping)
    self.integration_tolerance = float(integration_tolerance)
    self.n_grid = int(n_grid)

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
    for name, bounds in (('velocity_bounds', self.velocity_bounds), ('michaelis_bounds', self.michaelis_bounds),
                         ('concentration_bounds', self.concentration_bounds)):
      if not 0.0 < bounds[0] < bounds[1]:
        raise ValueError(f'{name} must be a strictly positive increasing (low, high), got {bounds}')

    self._rkc2 = rkc2_coefficients(self.n_stages, self.damping)
    axis = np.linspace(-1.0, 1.0, self.n_grid)
    grid_z = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    self._grid_scaled = jnp.asarray(grid_z, jnp.float32)
    self._grid_kinetics = jnp.stack([
      _log_unscale(0.5 * (self._grid_scaled[:, 0] + 1.0), self.velocity_bounds),
      _log_unscale(0.5 * (self._grid_scaled[:, 1] + 1.0), self.michaelis_bounds),
    ], axis=-1)
    self._generate = jax.jit(jax.vmap(self._event, in_axes=(0, 0)))
    self._posterior = jax.jit(self._posterior_mean)

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return EnzymeDepletionEvent(concentration=jax.ShapeDtypeStruct((self.n_experiments, self.n_measurements), np.float32))

  def target_spec(self):
    return EnzymeDepletionTarget(kinetics=jax.ShapeDtypeStruct((2, ), np.float32))

  def ground_truth_spec(self):
    return EnzymeDepletionGroundTruth(kinetics=jax.ShapeDtypeStruct((2, ), np.float32))

  def design_shape(self):
    return (self.n_experiments, )

  def design_spec(self):
    return EnzymeDepletionDesign(initial_concentration=jax.ShapeDtypeStruct((self.n_experiments, ), np.float32))

  def design_bounds(self):
    return {'initial_concentration': self.concentration_bounds}

  def combined_event_shape(self, design: bool = True):
    return (self.n_experiments, self.n_measurements + (1 if design else 0))

  def size(self):
    return None  # an analytic source: every index is a fresh variant

  # ------------------------------------------------------------------ #
  # Design scaling: affine on the LOG concentration.
  #
  # The box spans a decade and the informative quantity is the RATIO A0/K, so equal steps of the
  # scaled coordinate are equal steps of what the experiment resolves; a linear map would spend
  # most of the cube on the top half of the box.
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    return _log_scale(design, self.concentration_bounds)

  def _to_nominal_flat(self, design_scaled):
    return _log_unscale(design_scaled, self.concentration_bounds)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
    """``features (..., n_experiments, n_measurements + 1)``: each experiment's readings on a FIXED
    scale, followed by its own initial concentration on the scaled cube.
    ⚠️ THE READINGS ARE SCALED BY A FIXED REFERENCE -- the TOP OF THE DESIGN BOX -- and NOT by each
    experiment's own initial concentration. Dividing by the experiment's own value was the old rule
    and it is refuted: the read-out noise is ABSOLUTE (a standard deviation in concentration units,
    added to the clean signal), so dividing by the design multiplies that noise by its reciprocal and
    the FEATURE noise then varies across the design box by the box's own ratio. The information is not
    lost -- the initial concentration is appended as its own feature -- but the conditioning is, and a
    design at the bottom of the box arrives with the noisiest features for no physical reason. This is
    the same rule, and the same reason, as :mod:`detopt.detector.enzyme_mm`.

    Withholding the design drops the trailing column and nothing else: the readings never depended on
    it.

    ``mask`` is unused -- every experiment of the design is real (the element axis is the design's,
    not a hit count)."""
    concentration = jnp.asarray(event.concentration, jnp.float32)
    readings = concentration
    if design_scaled is None or not reveal_design:
      return readings
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    if design_scaled.ndim == 1:  # one design for the whole event batch
      design_scaled = jnp.broadcast_to(design_scaled, concentration.shape[:-2] + design_scaled.shape)
    return jnp.concatenate([readings, design_scaled[..., None]], axis=-1)

  def element_mask(self, event, mask):
    return mask  # element == experiment

  def normalize_target(self, target):
    kinetics = jnp.asarray(target.kinetics, jnp.float32)
    return jnp.stack([
      2.0 * _log_scale(kinetics[..., 0], self.velocity_bounds) - 1.0,
      2.0 * _log_scale(kinetics[..., 1], self.michaelis_bounds) - 1.0,
    ], axis=-1)

  def denormalize_predictions(self, normalised):
    normalised = jnp.asarray(normalised, jnp.float32)
    kinetics = jnp.stack([
      _log_unscale(0.5 * (normalised[..., 0] + 1.0), self.velocity_bounds),
      _log_unscale(0.5 * (normalised[..., 1] + 1.0), self.michaelis_bounds),
    ], axis=-1)
    return tensor.unflatten(tensor.structure(self.target_spec()), kinetics)

  def normalize_ground_truth(self, ground_truth):
    return self.normalize_target(EnzymeDepletionTarget(kinetics=ground_truth.kinetics))

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    return 'MSE (ln q, ln K; log-uniform prior on [-1, 1])'

  def metric_labels(self):
    return ('loss', ) + PARAMETER_NAMES

  def loss(self, predicted, target):
    return jnp.mean(jnp.square(predicted - target), axis=-1)

  def metric(self, predicted, target):
    squared = jnp.square(predicted - target)
    return {'loss': jnp.mean(squared, axis=-1), 'velocity': squared[..., 0], 'michaelis': squared[..., 1]}

  def metric_real_rmse(self, metric_means):
    """Sample-averaged per-parameter metric -> RMSE as a natural-log factor on the physical value."""
    spans = (
      math.log(self.velocity_bounds[1] / self.velocity_bounds[0]),
      math.log(self.michaelis_bounds[1] / self.michaelis_bounds[0])
    )
    return {name: (float(np.sqrt(metric_means[name]) * 0.5 * span), 'ln') for name, span in zip(PARAMETER_NAMES, spans)}

  def no_information_loss(self):
    """The loss of a GUESS: the best constant prediction under the prior, in closed form.

    Both targets are uniform on ``[-1, 1]`` after :meth:`normalize_target` (the priors are
    log-uniform and the map is affine in the log), the best constant is the mean, and the variance
    of ``U[-1, 1]`` is ``1/3`` -- so this is ``1/3`` exactly, whatever the bounds are. It is the
    denominator of the campaign criterion and the ceiling every estimate is measured against."""
    return 1.0 / 3.0

  # ------------------------------------------------------------------ #
  # Integration
  # ------------------------------------------------------------------ #
  def _integrate(self, initial, velocity, michaelis):
    """``[A]`` at every read-out time for one experiment, plus the largest ``|dt/2 - dt|`` there.

    Returns the ``dt`` chain -- the solution the error was measured ON -- and its error monitor.
    All arguments are scalars; the caller vmaps."""
    rate = lambda concentration: -velocity * concentration / (concentration + michaelis)
    dt = self.duration / (self.n_measurements * self.steps_per_measurement)
    coarse = rkc2_chain(
      self._rkc2, rate, jnp.asarray(initial, jnp.float32), dt=dt, n_steps=self.steps_per_measurement,
      n_intervals=self.n_measurements
    )
    fine = rkc2_chain(
      self._rkc2, rate, jnp.asarray(initial, jnp.float32), dt=0.5 * dt, n_steps=2 * self.steps_per_measurement,
      n_intervals=self.n_measurements
    )
    return coarse, jnp.max(jnp.abs(fine - coarse))

  def _check(self, error, where):
    error = float(jnp.max(error))
    if not math.isfinite(error) or error > self.integration_tolerance:
      raise RuntimeError(
        f'integration error {error:.3g} mM in {where} exceeds integration_tolerance '
        f'{self.integration_tolerance:.3g} mM -- the largest disagreement between the dt and dt/2 '
        f'chains on [A] at a read-out time; raise steps_per_measurement (the rate has a pole at '
        f'A = -K, so an overshoot past zero runs away rather than merely losing accuracy).'
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
    initial = self._resolve_design(design, n)
    concentration, kinetics, error = self._generate(initial, jnp.asarray(event_index, jnp.int32))
    self._check(error, 'the event batch')
    return (
      EnzymeDepletionGroundTruth(kinetics=kinetics), EnzymeDepletionEvent(concentration=concentration),
      jnp.ones((n, self.n_experiments), jnp.int32), EnzymeDepletionTarget(kinetics=kinetics)
    )

  def _resolve_design(self, design, n):
    """Any accepted NOMINAL design form -> ``(n, n_experiments)`` initial concentrations."""
    flat = jnp.reshape(jnp.asarray(self.flatten_design(design), jnp.float32), (-1, self.design_dim()))
    return jnp.broadcast_to(flat, (n, self.n_experiments))

  def _event(self, initial, event_index):
    """One variant: draw ``(q, K)`` from the log-uniform prior and read every experiment out.

    The read-out noise rows are handed out by the RANK of the initial concentration, not by the
    experiment's position in the design vector. The design is a SET, so the objective has to be
    exactly invariant under permuting it; with positional noise the same batch written in two
    orders draws two different realisations and the two evaluations disagree, which is a split on an
    identical experiment rather than a real difference. Ranking makes each physical experiment keep
    its own noise however the batch is written, while two experiments at the SAME concentration
    still get independent rows, so a replicate still buys information. The tie locus is a
    discontinuity of measure zero -- the same trade ``SortingRBF`` documents."""
    key_parameters, key_noise = jax.random.split(jax.random.PRNGKey(event_index), 2)
    unit = jax.random.uniform(key_parameters, (2, ), jnp.float32)
    velocity = _log_unscale(unit[0], self.velocity_bounds)
    michaelis = _log_unscale(unit[1], self.michaelis_bounds)
    clean, error = jax.vmap(self._integrate, in_axes=(0, None, None))(initial, velocity, michaelis)
    rank = jnp.argsort(jnp.argsort(initial))
    noise = self.measurement_noise * jax.random.normal(key_noise, clean.shape, jnp.float32)[rank]
    return clean + noise, jnp.stack([velocity, michaelis]), jnp.max(error)

  # ------------------------------------------------------------------ #
  # The estimator
  # ------------------------------------------------------------------ #
  def _posterior_mean(self, initial, concentration):
    """Posterior mean of the SCALED ``(ln q, ln K)`` on the prior grid, for a batch of read-outs.

    ``initial`` is ``(n_experiments,)`` NOMINAL, ``concentration`` is ``(B, n_experiments,
    n_measurements)``. The model curves are integrated once for the whole batch, since they depend
    on the design and not on the event."""
    model, error = jax.vmap(
      lambda kinetics: jax.vmap(self._integrate, in_axes=(0, None, None))(initial, kinetics[0], kinetics[1])
    )(self._grid_kinetics)
    flat_model = jnp.reshape(model, (self._grid_kinetics.shape[0], -1))
    flat_data = jnp.reshape(concentration, (concentration.shape[0], -1))
    # -|y - m|^2 / 2 sigma^2 up to a term constant in the grid, as one matmul.
    log_likelihood = (flat_data @ flat_model.T - 0.5 * jnp.sum(jnp.square(flat_model), axis=-1)[None, :])
    log_likelihood = log_likelihood / (self.measurement_noise**2)
    weight = jax.nn.softmax(log_likelihood, axis=-1)
    return weight @ self._grid_scaled, jnp.max(error)

  def estimate(self, design, event):
    """The instrument: NOMINAL ``design`` + an ``EnzymeDepletionEvent`` batch -> predictions in the
    NORMALISED target space, ``(B, 2)``, directly comparable with :meth:`normalize_target`.

    The posterior mean of the grid (see the module docstring); asserts the integration error of the
    model curves the same way :meth:`__call__` does for the events."""
    concentration = jnp.asarray(event.concentration, jnp.float32)
    initial = jnp.reshape(jnp.asarray(self.flatten_design(design), jnp.float32), (self.design_dim(), ))
    predicted, error = self._posterior(initial, concentration)
    self._check(error, 'the estimator grid')
    return predicted
