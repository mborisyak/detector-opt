"""`LinearDetector` in d dimensions, and that d = 1 is unchanged."""

import numpy as np
import pytest

from detopt.detector import LinearDetector

# Closed-form risks measured on the 1-dimensional detector BEFORE it was generalised. They are the
# regression guard: d = 1 must remain bit-identical.
ONE_DIMENSIONAL = {(-1.0, 1.0): 0.0049751244, (0.0, 0.0): 0.5024875622, (-1.0, 0.5): 0.0071628844}


@pytest.mark.parametrize("design,expected", list(ONE_DIMENSIONAL.items()))
def test_one_dimensional_risk_is_unchanged(design, expected):
  assert LinearDetector(n_probes=2, noise=0.1).bayes_risk(list(design)) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("dimensions,probes", [(1, 2), (3, 4), (4, 5)])
def test_shapes_follow_the_dimension(dimensions, probes):
  detector = LinearDetector(n_probes=probes, n_dimensions=dimensions, noise=0.5)
  assert detector.design_dim() == probes * dimensions
  assert detector.combined_event_shape() == (probes, 1 + dimensions)
  assert detector.target_spec().coefficients.shape == (dimensions + 1, )
  # ONE FIELD PER AXIS, each `n_probes` wide -- what the invariant kernel needs to permute probes.
  spec = detector.design_spec()
  assert len(spec._fields) == dimensions
  assert all(leaf.shape == (probes, ) for leaf in spec)


@pytest.mark.parametrize("dimensions,probes", [(1, 2), (3, 4), (4, 5)])
def test_the_kernel_finds_one_exchangeable_block_per_axis(dimensions, probes):
  """`exchangeable: n_probes` must resolve, and give d blocks of that width -- otherwise the design is
    not treated as a SET of probes."""
  from detopt.bo import _exchangeable_blocks

  detector = LinearDetector(n_probes=probes, n_dimensions=dimensions, noise=0.5)
  blocks = _exchangeable_blocks(detector, probes)
  assert len(blocks) == dimensions
  assert all(len(block) == probes for block in blocks)
  assert sorted(i for block in blocks for i in block) == list(range(probes * dimensions))


@pytest.mark.parametrize("dimensions", [1, 3])
def test_no_information_level_is_one(dimensions):
  """Predicting the prior mean scores 1.0 at every d -- the loss reads on a fixed absolute scale."""
  detector = LinearDetector(n_probes=dimensions + 1, n_dimensions=dimensions, noise=0.5)
  _, _, _, target = detector(np.zeros(detector.design_dim(), np.float32), np.arange(4096))
  assert float(np.mean(np.square(np.asarray(target.coefficients)))) == pytest.approx(1.0, abs=0.05)


def test_rank_deficient_design_leaves_directions_at_the_prior():
  """With fewer probes than coefficients the unmeasured directions stay at prior variance 1, so the
    risk cannot fall below their share."""
  detector = LinearDetector(n_probes=2, n_dimensions=4, noise=0.01)
  design = np.eye(4, dtype=np.float32)[:2].T.reshape(-1)
  unmeasured = (detector.n_dimensions + 1 - detector.n_probes) / (detector.n_dimensions + 1)
  assert detector.bayes_risk(design) > unmeasured * 0.99


def test_more_spread_is_never_worse_than_coincident_probes():
  detector = LinearDetector(n_probes=4, n_dimensions=3, noise=0.5)
  corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float32).T.reshape(-1)
  assert detector.bayes_risk(corners) < detector.bayes_risk(np.zeros(12, np.float32))


def test_the_design_round_trips_through_the_named_record():
  """A named design and its flat form must describe the same probes."""
  detector = LinearDetector(n_probes=4, n_dimensions=3, noise=0.5)
  positions = np.random.default_rng(0).uniform(-1.0, 1.0, (4, 3)).astype(np.float32)
  named = detector.design_spec().__class__(*positions.T)
  assert np.allclose(np.asarray(detector.flatten_design(named)), positions.T.reshape(-1))
  assert detector.bayes_risk(named) == pytest.approx(detector.bayes_risk(positions.T.reshape(-1)))


def test_the_response_is_the_plane_read_at_the_probes():
  """Ground truth, not assertion: the noiseless read-out must equal `w . x + b` for the drawn `(w, b)`."""
  detector = LinearDetector(n_probes=4, n_dimensions=3, noise=1e-6)
  design = np.random.default_rng(1).uniform(-1.0, 1.0, (4, 3)).astype(np.float32)
  truth, event, _, _ = detector(design.T.reshape(-1), np.arange(256))
  coefficients = np.asarray(truth.coefficients)
  expected = design @ coefficients[:, :3].T + coefficients[:, 3]
  assert np.allclose(np.asarray(event.response), expected.T, atol=1e-4)
