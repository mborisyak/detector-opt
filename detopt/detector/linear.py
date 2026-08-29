"""Linear-response probes: the smallest task in this repo that is a REAL optimal-design problem.

This is the DEBUG detector -- it runs on a CPU in milliseconds, has no ODE, no data file and no
physics, and exists so that the machinery around a detector (the trainers, the BO driver, the resume
protocol, the tests) can be exercised end to end without waiting on a simulator. What makes it worth
having rather than a random-number stub is that its answer is KNOWN IN CLOSED FORM, so a driver that
finds the wrong optimum on it is wrong, not unlucky.

* **design** -- ``n_probes`` probe POSITIONS ``x_i`` in ``[-1, 1]^d``, ``d = n_dimensions``.
* **event** -- one linear response drawn per event: ``w ~ N(0, I_d)`` (slope vector), ``b ~ N(0, 1)``
  (intercept). It is read out at each probe as ``y_i = w . x_i + b + eps_i``, ``eps_i ~ N(0, sigma^2)``
  with ``sigma = noise`` (0.1 by default). The draw depends on ``event_index`` ALONE, so the same
  response is re-measured under every design (common random numbers) and no design can move its own
  label.
* **target** -- ``(w, b)``, ``d + 1`` components. Already standard normal, so the loss needs no
  rescaling and reads on a fixed absolute scale.
* **loss** -- mean squared error over the ``d + 1`` components. Predicting the prior mean scores 1.0
  exactly, which is the no-information level, at every ``d``.

THE OPTIMUM, and why it is a design problem at all
--------------------------------------------------
With ``X`` the ``(n_probes, d + 1)`` matrix of rows ``(x_i, 1)``, the posterior covariance of
``(w, b)`` under the standard-normal prior is ``Sigma = (X^T X / sigma^2 + I)^-1`` and the achievable
loss is ``tr(Sigma) / (d + 1)`` -- a Bayes risk, so it is what a perfect regressor reaches and the
network's floor. At ``d = 1`` with two probes ``det(X^T X) = (x_1 - x_2)^2``, so the information is
driven ENTIRELY by how far apart the probes are: coincident probes measure ``w + b`` twice and never
separate the two coefficients. At ``n_probes <= d`` the design is RANK DEFICIENT by construction and
the unmeasured directions sit at their prior variance.

At ``sigma = 0.1`` and two probes that gives 0.00498 at the corners ``{-1, +1}`` against 0.501 with
both probes together -- a hundredfold span with the optimum on the boundary of the box, in BOTH
orders (the design is a set, so ``(-1, 1)`` and ``(1, -1)`` are the same experiment and an
exchangeable kernel should see them as one point).

:meth:`bayes_risk` computes that floor for any design, so a test can assert what a run should have
found instead of asserting that it found something.
"""

import collections
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .common import Detector
from ..utils import tensor

__all__ = ['LinearDetector', 'LinearDesign', 'LinearEvent', 'LinearTarget', 'LinearGroundTruth']


class LinearDesign(NamedTuple):
  """Where the response is read out, in ONE dimension: one position per probe, NOMINAL."""
  probe: jax.Array  # (n_probes,)


def design_type(n_dimensions):
  """The design record for ``n_dimensions``: ONE FIELD PER AXIS, each holding that axis' coordinate
    for every probe.

    Not one ``(n_probes, d)`` field, because `detopt.bo._exchangeable_blocks` builds one exchangeable
    block per design field whose width matches the group size -- so d fields of width ``n_probes`` are
    what let the invariant kernel permute PROBES (the design is a set of probes) rather than
    coordinates. It is the enzyme batch's layout, where `enzyme_fraction` and `temperature` are two
    blocks permuted together. At d = 1 it is :class:`LinearDesign` unchanged."""
  if n_dimensions == 1:
    return LinearDesign
  return collections.namedtuple("LinearDesignND", [f"probe_{i}" for i in range(n_dimensions)])


class LinearEvent(NamedTuple):
  """The read-out: the noisy response at every probe."""
  response: jax.Array  # (n_probes,)


class LinearTarget(NamedTuple):
  """What the regressor predicts: the drawn slope vector and intercept."""
  coefficients: jax.Array  # (n_dimensions + 1,) == (w, b)


class LinearGroundTruth(NamedTuple):
  """The drawn response itself (== conditioning). The target is the whole of it here."""
  coefficients: jax.Array  # (n_dimensions + 1,) == (w, b)


class LinearDetector(Detector):
  """``n_probes`` probes on a linear response (see the module docstring).

  Every constant is a constructor argument, i.e. lives in the yaml config: the number of probes, the
  read-out noise and the probe box.
  """

  def __init__(self, *, n_probes: int = 2, n_dimensions: int = 1, noise: float = 0.1,
               probe_bounds: tuple = (-1.0, 1.0)):
    self.n_probes = int(n_probes)
    self.n_dimensions = int(n_dimensions)
    self.noise = float(noise)
    self.probe_bounds = (float(probe_bounds[0]), float(probe_bounds[1]))
    if self.n_probes < 1:
      raise ValueError(f'n_probes must be at least 1, got {n_probes}')
    if self.n_dimensions < 1:
      raise ValueError(f'n_dimensions must be at least 1, got {n_dimensions}')
    self._design_type = design_type(self.n_dimensions)
    if not self.noise > 0.0:
      raise ValueError(f'noise is a standard deviation and must be strictly positive, got {noise}')
    if not self.probe_bounds[0] < self.probe_bounds[1]:
      raise ValueError(f'probe_bounds must be an increasing (low, high), got {probe_bounds}')
    self._generate = jax.jit(jax.vmap(self._event))

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return LinearEvent(response=jax.ShapeDtypeStruct((self.n_probes, ), np.float32))

  def target_spec(self):
    return LinearTarget(coefficients=jax.ShapeDtypeStruct((self.n_dimensions + 1, ), np.float32))

  def ground_truth_spec(self):
    return LinearGroundTruth(coefficients=jax.ShapeDtypeStruct((self.n_dimensions + 1, ), np.float32))

  def design_shape(self):
    return (self.n_dimensions, self.n_probes)  # field-major: one field per axis, `n_probes` wide

  def design_spec(self):
    leaf = jax.ShapeDtypeStruct((self.n_probes, ), np.float32)
    return self._design_type(*([leaf] * self.n_dimensions))

  def design_bounds(self):
    return {name: self.probe_bounds for name in self._design_type._fields}

  def _probe_positions(self, flat):
    """Flat design (..., d * n_probes), field-major, -> probe positions (..., n_probes, d)."""
    axes = jnp.reshape(flat, flat.shape[:-1] + (self.n_dimensions, self.n_probes))
    return jnp.swapaxes(axes, -1, -2)

  def combined_event_shape(self, design: bool = True):
    # element == probe: its reading, and either its POSITION (design revealed) or a one-hot of its
    # own INDEX (design withheld) -- see `combine_scaled` for why the withheld case is not empty
    return (self.n_probes, 1 + (self.n_dimensions if design else self.n_probes))

  def size(self):
    return None  # an analytic source: every index is a fresh response

  # ------------------------------------------------------------------ #
  # Design scaling: one affine map, the probe box onto [0, 1].
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    low, high = self.probe_bounds
    return (jnp.asarray(design, jnp.float32) - low) / (high - low)

  def _to_nominal_flat(self, design_scaled):
    low, high = self.probe_bounds
    return jnp.asarray(design_scaled, jnp.float32) * (high - low) + low

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled=None, mask=None, reveal_design: bool = True):
    """``features (..., n_probes, 1 + n_dimensions)``: each probe's RAW reading beside its own position.

    The reading is passed through UNCHANGED -- there is no rescaling. The position comes STRAIGHT from
    the scaled design, already on ``[0, 1]``. ``mask`` is unused -- every probe of the design is real
    (the element axis is the design's, not a hit count).

    WITH THE DESIGN WITHHELD -- ``reveal_design=False`` or ``design_scaled=None`` -- the position
    columns are replaced by a ONE-HOT OF THE PROBE'S OWN INDEX, ``(..., n_probes, 1 + n_probes)``. The
    reading is unchanged: the response is drawn from ``event_index`` alone and read out at the true
    probe positions either way, so withholding costs the knowledge of WHERE each reading was taken,
    never the reading.

    ⚠️ WHY THE WITHHELD CASE IS NOT SIMPLY THE READING ALONE. The set regressor is permutation-
    INVARIANT over the element axis, so a bare column of readings is an unordered MULTISET of
    ``w . x_i + b`` and ``w`` is not identifiable from it -- a withheld arm would plateau because the
    problem is underdetermined, which is a statement about the regressor's symmetry and not about the
    design. The one-hot restores each probe's IDENTITY without restoring its POSITION, so the two are
    separable: an arm that recovers with it was limited by exchangeability, and an arm that does not
    was limited by the design information itself."""
    reading = jnp.asarray(event.response, jnp.float32)[..., None]
    if design_scaled is None or not reveal_design:
      identity = jnp.broadcast_to(jnp.eye(self.n_probes, dtype=jnp.float32), reading.shape[:-1] + (self.n_probes, ))
      return jnp.concatenate([reading, identity], axis=-1)
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    positions = self._probe_positions(jnp.reshape(design_scaled, design_scaled.shape[:-1] + (-1, )))
    if positions.ndim == 2:  # one design for the whole event batch
      positions = jnp.broadcast_to(positions, event.response.shape[:-1] + positions.shape)
    return jnp.concatenate([reading, positions], axis=-1)

  def element_mask(self, event, mask):
    return mask  # element == probe

  def normalize_target(self, target):
    """A no-op rescale: ``w`` and ``b`` are drawn standard normal, so the target is already O(1) and
    the loss below is on an absolute scale -- 1.0 is exactly the no-information level."""
    flat, _ = tensor.flatten(target)
    return flat

  def denormalize_predictions(self, normalised):
    return tensor.unflatten(tensor.structure(self.target_spec()), jnp.asarray(normalised, jnp.float32))

  def normalize_ground_truth(self, ground_truth):
    return jnp.asarray(ground_truth.coefficients, jnp.float32)

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    return 'MSE (slope, intercept; prior N(0, 1))'

  def metric_labels(self):
    return ('loss', ) + tuple(f'slope_{i}' for i in range(self.n_dimensions)) + ('intercept', )

  def loss(self, predicted, target):
    return jnp.mean(jnp.square(predicted - target), axis=-1)

  def metric(self, predicted, target):
    squared = jnp.square(predicted - target)
    metrics = {'loss': jnp.mean(squared, axis=-1), 'intercept': squared[..., self.n_dimensions]}
    for i in range(self.n_dimensions):
      metrics[f'slope_{i}'] = squared[..., i]
    return metrics

  def metric_real_rmse(self, metric_means):
    named = [f'slope_{i}' for i in range(self.n_dimensions)] + ['intercept']
    return {name: (float(np.sqrt(metric_means[name])), 'prior sd') for name in named}

  # ------------------------------------------------------------------ #
  # The closed-form answer
  # ------------------------------------------------------------------ #
  def bayes_risk(self, design):
    """The BEST loss any regressor can reach at this design: ``tr[(X^T X / sigma^2 + I)^-1] / 2``.

    The posterior of ``(w, b)`` under a standard-normal prior and Gaussian read-out noise is Gaussian
    with that covariance, and :meth:`loss` is the mean squared error over the two components, so this
    is the floor the network is trying to reach -- not an approximation to it. ``design`` is NOMINAL
    (a ``LinearDesign``, a config mapping or a flat array), matching :meth:`__call__`.
    """
    flat = np.asarray(self.flatten_design(design), np.float64).reshape(-1)
    probe = flat.reshape(self.n_dimensions, self.n_probes).T
    rows = np.concatenate([probe, np.ones((self.n_probes, 1), np.float64)], axis=-1)
    precision = rows.T @ rows / self.noise**2 + np.eye(self.n_dimensions + 1)
    return float(np.trace(np.linalg.inv(precision)) / (self.n_dimensions + 1))

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """Draw the responses at ``event_index`` and read each one out at ``design`` (one design broadcast
    over the batch, or one design per event). DETERMINISTIC: the response and its read-out noise are
    seeded from ``event_index`` alone, so the same event under two designs is the same response and
    the target is a property of the event only."""
    event_index = np.asarray(event_index, np.int64)
    n = event_index.shape[0]
    flat = jnp.reshape(jnp.asarray(self.flatten_design(design), jnp.float32), (-1, self.design_dim()))
    probe = jnp.broadcast_to(self._probe_positions(flat), (n, self.n_probes, self.n_dimensions))
    response, coefficients = self._generate(probe, jnp.asarray(event_index, jnp.int32))
    mask = jnp.ones((n, self.n_probes), jnp.int32)
    return (
      LinearGroundTruth(coefficients=coefficients), LinearEvent(response=response), mask,
      LinearTarget(coefficients=coefficients)
    )

  def _event(self, probe, event_index):
    """One event: draw ``(w, b)`` and read the plane out at every probe. ``probe`` is
    ``(n_probes, n_dimensions)``; every draw uses ``event_index`` only."""
    key_line, key_noise = jax.random.split(jax.random.PRNGKey(event_index), 2)
    coefficients = jax.random.normal(key_line, (self.n_dimensions + 1, ), jnp.float32)
    clean = probe @ coefficients[:self.n_dimensions] + coefficients[self.n_dimensions]
    return clean + self.noise * jax.random.normal(key_noise, clean.shape, jnp.float32), coefficients
