"""Bayes-classifier instrument for the binary inhibitor-MECHANISM task (`enzyme_inhib` restricted to
``mechanism_classes: [mostly_competitive, mostly_uncompetitive]``).

The quantity estimated is the detector's own objective -- softmax cross-entropy over the two classes
divided by ``ln 2``, whose no-information level is exactly 1.0 -- for the classifier that reads the
posterior straight off the generative model instead of learning it.

WHAT IS CLOSED FORM AND WHAT IS NUMERICAL
    closed form   the likelihood of one batch of read-outs given one (enzyme, compound) draw. The
                  read-out noise is independent ``N(0, sigma)`` on every sample, so
                  ``log p(y | draw) = -||y - f(draw)||^2 / (2 sigma^2) + const`` exactly, and the
                  constant cancels in the posterior. The posterior over the two classes is then the
                  ratio of the two marginals, and the loss is ``-log2`` of it.
    numerical     (a) ``f(draw)`` itself -- the detector's own RKC2 integration, error-controlled by
                  its dt / dt-half guard; (b) the marginal over the prior, a Monte-Carlo average over
                  ``n_library`` draws from the SAME prior the events come from. The importance weights
                  are the closed-form likelihoods, so the only approximation is the sample average,
                  and :class:`InstrumentResult` reports its effective sample size.

The library draws and the events are DISJOINT index ranges: an event that is also a library node
contributes its own noise-matched likelihood to the marginal and the estimate collapses onto it.

``hedge`` mixes the estimated posterior with the class prior. It makes the instrument an explicit,
achievable classifier (hence an upper bound on the Bayes risk) with a finite worst-case loss, rather
than an estimator that a single mis-weighted event can send to infinity.
"""

import math
from typing import NamedTuple

import numpy as np

import jax

__all__ = ["MechanismInstrument", "InstrumentResult", "build_detector", "sobol_designs"]


def build_detector(config, **overrides):
  """``{detector_name: arguments}`` config dict -> detector, with ``overrides`` applied to the
  arguments. The config is not mutated."""
  import detopt.detector

  (name, ) = config.keys()
  return detopt.detector.from_config({name: dict(config[name], **overrides)})


def sobol_designs(dimension, n, *, seed):
  """``(n, dimension)`` scrambled-Sobol points in the unit cube."""
  from scipy.stats import qmc

  return np.asarray(qmc.Sobol(d=dimension, scramble=True, seed=seed).random(n), np.float64)


class InstrumentResult(NamedTuple):
  """One design's estimate. ``loss`` is the objective (cross-entropy / ln 2, no-information 1.0);
  ``standard_error`` is over events. ``effective_sample_size`` is the median over events of the
  importance weights' ``(sum w)^2 / sum w^2`` -- how many of the ``n_library`` prior draws the
  marginal actually rests on. ``hedge_fraction`` is the share of events whose loss is set by the
  hedge rather than by the weights."""
  loss: float
  standard_error: float
  accuracy: float
  effective_sample_size: float
  hedge_fraction: float


class MechanismInstrument:
  """Marginal-likelihood classifier for one detector configuration.

  The detector passed in must be NOISELESS (``measurement_noise: 0.0``); the read-out noise is
  applied here, so that one event can be scored under several noise draws and so that the same clean
  curves serve as both the events and the library. ``noise`` is the standard deviation that
  configuration is calibrated at.

  NO STATE FROM THE DESIGN is kept: :meth:`evaluate` takes the design and returns the estimate.
  """

  def __init__(
    self, detector, *, noise: float, n_events: int, n_library: int, n_noise: int = 1, hedge: float = 1.0e-3, seed: int = 0
  ):
    if detector.n_classes != 2:
      raise ValueError(f'the mechanism instrument scores the BINARY task; the detector has {detector.n_classes} classes')
    if not detector.measurement_noise == 0.0:
      raise ValueError(
        f'the detector must be the NOISELESS twin (measurement_noise 0.0), got {detector.measurement_noise}; '
        f'the read-out noise is applied by the instrument'
      )
    if not noise > 0.0:
      raise ValueError(f'noise must be positive, got {noise}')
    if not 0.0 <= hedge < 1.0:
      raise ValueError(f'hedge is a mixing weight and must lie within [0, 1), got {hedge}')
    self.detector = detector
    self.noise = float(noise)
    self.n_events = int(n_events)
    self.n_library = int(n_library)
    self.n_noise = int(n_noise)
    self.hedge = float(hedge)
    self.seed = int(seed)
    stream = np.random.SeedSequence(self.seed).generate_state(2)
    self._generator = np.random.default_rng(stream[0])
    offset = int(stream[1]) % (1 << 30)
    self.library_indices = offset + np.arange(self.n_library, dtype=np.int64)
    self.event_indices = offset + self.n_library + np.arange(self.n_events, dtype=np.int64)
    self._noise_draw = self._generator.standard_normal(
      (self.n_events, self.n_noise, detector.n_experiments * detector.n_measurements)
    ).astype(np.float32)

  def curves(self, design_scaled, indices):
    """Noise-free read-outs ``(len(indices), n_experiments * n_measurements)`` and the class label
    ``(len(indices),)`` of every drawn compound, at a design given in the unit cube."""
    nominal = np.asarray(self.detector._to_nominal_flat(np.asarray(design_scaled, np.float32)))
    _, event, _, target = self.detector(nominal, np.asarray(indices, np.int64))
    flat = np.asarray(event.measurements, np.float32).reshape(len(indices), -1)
    return flat, np.asarray(np.argmax(np.asarray(target.mechanism), axis=-1), np.int64)

  def posterior(self, design_scaled):
    """Per-event posterior probability of class 1 and the importance weights' effective sample size.
    Exposed so a caller can inspect the calibration behind :meth:`evaluate`."""
    library, library_class = self.curves(design_scaled, self.library_indices)
    events, event_class = self.curves(design_scaled, self.event_indices)
    observed = events[:, None, :] + self.noise * self._noise_draw
    observed = observed.reshape(self.n_events * self.n_noise, -1)
    squared = (
      np.sum(observed * observed, axis=1, dtype=np.float64)[:, None] - 2.0 * np.asarray(observed @ library.T, np.float64) +
      np.sum(library * library, axis=1, dtype=np.float64)[None, :]
    )
    log_weight = -0.5 * squared / (self.noise * self.noise)
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(log_weight)
    total = np.sum(weight, axis=1)
    probability = np.sum(weight * (library_class == 1)[None, :], axis=1) / total
    effective = total * total / np.sum(weight * weight, axis=1)
    return probability, effective, np.repeat(event_class, self.n_noise)

  def evaluate(self, design_scaled):
    """The design's estimated objective and its diagnostics."""
    probability, effective, truth = self.posterior(design_scaled)
    hedged = (1.0 - self.hedge) * probability + 0.5 * self.hedge
    predicted = np.where(truth == 1, hedged, 1.0 - hedged)
    loss = -np.log(predicted) / math.log(2.0)
    per_event = loss.reshape(self.n_events, self.n_noise).mean(axis=1)
    bound = -math.log(0.5 * self.hedge) / math.log(2.0) if self.hedge > 0.0 else math.inf
    return InstrumentResult(
      loss=float(np.mean(per_event)), standard_error=float(np.std(per_event, ddof=1) / math.sqrt(self.n_events)),
      accuracy=float(np.mean((hedged > 0.5) == (truth == 1))), effective_sample_size=float(np.median(effective)),
      hedge_fraction=float(np.mean(loss > 0.9 * bound)) if self.hedge > 0.0 else 0.0
    )
