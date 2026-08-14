"""Design / results serialisation across the encoded->scaled migration.

Files written before the migration hold N(0,1) vectors under names that now mean something else.
Read as scaled they are not merely wrong, they are silently wrong -- so both readers must refuse
them rather than convert. These tests pin that refusal.
"""

import json
import os

import numpy as np
import pytest

import detopt
from detopt.utils import io
from detopt.utils.config import load_config


def _detector():
  return detopt.detector.EnzymeDetector(**load_config('config/detector/enzyme.yaml')['enzyme'])


def test_design_round_trips_through_the_file_in_nominal_units(tmp_path):
  """save_design writes NOMINAL units and load_design returns SCALED -- the file is the only
  parameterisation-independent form, so a re-parameterisation cannot invalidate it."""
  detector = _detector()
  scaled = np.array([0.25, 0.5, 0.75, 1.0, 0.0, 0.3, 0.6, 0.9], dtype=np.float32)
  path = str(tmp_path / "design" / "d.json")
  io.save_design(detector, path, scaled)

  payload = json.load(open(path))
  assert payload["space"] == "nominal"  # the tag is what makes a stale file detectable
  on_disk = np.asarray(payload["design"], dtype=np.float32)
  nominal = np.asarray(detector.flatten_design(detector.to_nominal(scaled)), dtype=np.float32)
  np.testing.assert_allclose(on_disk, nominal, rtol=1e-6)
  # the temperatures really are degrees C, not [0, 1] numbers
  low, high = detector.temperature_bounds
  assert np.all((low <= on_disk[4:]) & (on_disk[4:] <= high))

  np.testing.assert_allclose(io.load_design(detector, path), scaled, atol=1e-6)


def test_load_design_rejects_a_pre_migration_encoded_file(tmp_path):
  """A pre-migration design.json is a BARE LIST holding an ENCODED N(0,1) vector (~0 = the centre of
  every range). It must be refused on the missing tag.

  A bounds check would NOT catch it, which is the whole reason for the tag: the enzyme bounds start
  at zero (`enzyme_fraction` [0, 1], `temperature` [0, 100]), so scaling ~0 gives ~0 -- a perfectly
  in-range scaled vector that reads as the lower corner of every range. This test asserts exactly
  that, so the tag cannot be dropped in favour of a range check later."""
  detector = _detector()
  path = str(tmp_path / "design" / "old.json")
  encoded = [-6.7e-05, -5.7e-05, -1.0e-04, 7.8e-05, -6.7e-05, 8.0e-05, 4.5e-05, -9.7e-05]
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(encoded, f)

  # Why the tag exists rather than a bounds check: whether an encoded file is DETECTABLE by its
  # range depends on the bounds. `enzyme_fraction_bounds` starts at 0, so those coordinates scale to
  # ~0 -- a perfectly in-range "scaled" value reading as the lower corner, with nothing out of
  # place. (`temperature_bounds` does not start at 0, so that half happens to fall outside; a guard
  # that relied on it would therefore work here and fail on the next detector.)
  scaled = np.asarray(detector.to_scaled(np.asarray(encoded, np.float32)))
  n = detector.n_experiments
  assert np.all(np.abs(scaled[:n]) < 0.01), 'fraction coordinates should look like a valid design'

  with pytest.raises(ValueError, match="not a tagged NOMINAL design file"):
    io.load_design(detector, path)


def test_check_bo_results_rejects_pre_migration_runs():
  """results.json from before the migration keys the searched design as ``x_encoded`` in N(0,1)
  units. Those do not convert to scaled, so the reader refuses the file."""
  current = [{"iteration": 0, "x_scaled": [0.1, 0.2], "design": [1.0, 2.0], "loss": 0.5}]
  # EQUAL, not identical: the reader now goes through `complete_results`, which drops the `incomplete`
  # row a killed run records and therefore returns a new list even when it drops nothing.
  assert io.check_bo_results(current, "results.json") == current
  assert io.check_bo_results([], "results.json") == []  # an empty run is not a stale one

  incomplete = current + [{"iteration": 1, "x_scaled": [0.3, 0.4], "design": [3.0, 4.0], "loss": None,
                           "status": "incomplete"}]
  assert io.check_bo_results(incomplete, "results.json") == current  # the unscored row never reaches arithmetic

  stale = [{"iteration": 0, "x_encoded": [0.1, 0.2], "design": [1.0, 2.0], "loss": 0.5}]
  with pytest.raises(ValueError, match="pre-migration BO results"):
    io.check_bo_results(stale, "results.json")
