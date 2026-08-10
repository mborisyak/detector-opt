"""Tests for the GBDT stand-in objective (:mod:`detopt.bo.gbdt`) and the config-override helper.

The properties the landscape study rests on: the score is DETERMINISTIC under common random numbers
(which is what makes the retuned benchmark noise-free for BO), an uninformative design scores the
target's own variance while an informative one scores well below it, and the reported stopping point
is a real early-stopping point rather than the last learner.
"""

import numpy as np
import pytest

import detopt
from detopt.bo.gbdt import sample_design, score_design
from detopt.utils.config import load_config, override

# The target is normalised onto [-1, 1] over its own uniform prior, so knowing nothing scores 1/3.
NO_INFORMATION = 1.0 / 3.0
# Small enough to keep the test quick, large enough for the two regimes to separate unambiguously.
N_EVENTS = 4096


def _detector(**assignments):
  config = load_config('config/detector/enzyme.yaml')
  return detopt.detector.from_config(override(config, [f'{k}={v}' for k, v in assignments.items()]))


def _batch(detector, fraction, temperature):
  """A flat physical design putting every experiment at the same fraction and temperature."""
  n = detector.n_experiments
  return np.concatenate([np.full(n, fraction), np.full(n, temperature)]).astype(np.float32)


@pytest.fixture(scope='module')
def detector():
  return _detector()


def test_sample_shapes(detector):
  features, target = sample_design(detector, _batch(detector, 1.0 / 3.0, 50.0), np.arange(64))
  assert features.shape == (64, detector.n_experiments * detector.n_measurements)
  assert target.shape == (64, )
  # normalize_target maps the melting prior onto [-1, 1]; the draw is uniform inside it.
  assert np.all(np.abs(target) <= 1.0 + 1e-5)


def test_common_random_numbers_are_deterministic(detector):
  """The whole point of a fixed event block: the objective becomes a function of the design alone.

  The detector seeds an enzyme (and its readout noise) from the event index only, and the GBDT does
  no subsampling, so re-scoring the same design on the same block must reproduce the score exactly --
  including under a different estimator seed.
  """
  design = _batch(detector, 1.0 / 3.0, 53.0)
  first = score_design(detector, design, n_events=N_EVENTS, event_offset=0, seed=0)
  again = score_design(detector, design, n_events=N_EVENTS, event_offset=0, seed=7)
  assert first.loss == pytest.approx(again.loss, abs=0.0)
  assert first.n_learners == again.n_learners


def test_uninformative_design_scores_the_prior_variance(detector):
  """Every experiment far above the melting prior kills every enzyme, so the readout is pure noise
  and no estimator can beat predicting the prior mean."""
  score = score_design(detector, _batch(detector, 1.0 / 3.0, 95.0), n_events=N_EVENTS, seed=0)
  assert score.loss == pytest.approx(NO_INFORMATION, rel=0.1)
  # Nothing to fit: boosting stops almost immediately rather than running to max_learners.
  assert score.n_learners < 10


def test_informative_design_beats_the_prior(detector):
  """A batch sitting on the melting edge identifies the target well enough to halve the variance."""
  score = score_design(detector, _batch(detector, 1.0 / 3.0, 53.0), n_events=N_EVENTS, seed=0)
  assert score.loss < 0.5 * NO_INFORMATION
  assert score.sem > 0.0
  # The stopping point is chosen by the validation minimum, so it must be interior, not the cap.
  assert 1 < score.n_learners < 400


def test_loss_is_the_mean_of_the_two_reported_halves(detector):
  score = score_design(detector, _batch(detector, 1.0 / 3.0, 53.0), n_events=N_EVENTS, seed=0)
  assert score.loss == pytest.approx(0.5 * (score.train + score.val))
  assert score.train <= score.val  # the fit is never worse on the data it was fitted to


def test_override_rejects_unknown_keys():
  config = load_config('config/detector/enzyme.yaml')
  assert override(config, ['enzyme.n_measurements=16'])['enzyme']['n_measurements'] == 16
  assert override(config, ['enzyme.parameters.T_melting=[25.0, 75.0]'])['enzyme']['parameters']['T_melting'] == [25.0, 75.0]
  with pytest.raises(KeyError):
    override(config, ['enzyme.no_such_key=1'])
  with pytest.raises(ValueError):
    override(config, ['enzyme.n_measurements'])
