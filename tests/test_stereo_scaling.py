"""The stereo family's nominal<->scaled contract, over the whole scaled cube.

`stereo_straw` is the one detector where "each coordinate affinely on its own range" does NOT hold
literally: the stations are ordered and magnet-excluding, so station k's window is coupled to
station k-1 and u_k is the fraction of the REMAINING admissible window. That coupling is what the
geometry constraints are enforced by, so it is checked over the cube rather than at a point --
including the corners, which are reachable now (they were theta -> +-inf under the old quantile
encoding) and are exactly where a sign error in a window would show up.

On tolerances. The coupling makes windows arbitrarily narrow: if station k-1 sits against its own
maximum, station k's admissible window can shrink to a fraction of a cm on z-values of ~9000, and
inverting `(z - lo) / (hi - lo)` there is float32 cancellation. Measured worst case over the cube:
1.4e-3 in u, and a 10 MICRON overshoot of the magnet face at the u=1 corner (8817.002 against a
face at 8817.0, on a 100 cm station). Those are precision, not geometry, so the tolerances below are
set just above them and the checks stay meaningful -- a real sign or ordering error is off by
whole centimetres, not microns.
"""

import glob
import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import detopt
from detopt.detector.straw import StrawEvent
from detopt.utils.config import load_config

CONFIGS = sorted(glob.glob("config/detector/stereo_*.yaml"))
MICRONS = 1e-2  # cm; comfortably above the measured float32 edge effects, far below any real error


def _detector(path):
  """Build the detector from its canonical config, forced onto the analytic engine so the test needs
  no MC on disk (some stereo configs point at a data directory)."""
  config = load_config(path)
  name = next(iter(config))
  arguments = dict(config[name])
  if "engine" in arguments:
    arguments["engine"] = "simplified"
  if "data_dir" in arguments:
    arguments["data_dir"] = None
  detector = detopt.detector.from_config({name: arguments})
  if not hasattr(detector, "n_stations"):
    pytest.skip(f"{name} is not a stereo-geometry detector")
  return detector


def _cube(detector, n, seed):
  """A uniform sample of the scaled cube WITH both corners appended -- the corners are reachable
  under affine scaling and are where a coupled window degenerates."""
  rng = np.random.default_rng(seed)
  u = rng.uniform(0.0, 1.0, (n, detector.design_dim())).astype(np.float32)
  corners = np.stack([np.zeros(detector.design_dim(), np.float32), np.ones(detector.design_dim(), np.float32)])
  return jnp.asarray(np.concatenate([u, corners], axis=0))


def _fab_event(detector, batch, seed):
  rng = np.random.default_rng(seed)
  m = detector.max_hits_per_event
  ri = lambda hi: rng.integers(0, hi, (batch, m)).astype(np.int32)
  return StrawEvent(
    station=ri(detector.n_stations), view=ri(detector.n_views_per_station), layer=ri(detector.n_layers_per_view),
    straw=ri(detector.n_straws), tdc=(440.0 + 30.0 * rng.standard_normal((batch, m))).astype(np.float32)
  )


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p)[:-5] for p in CONFIGS])
def test_round_trip(path):
  """Both directions invert. NOMINAL is the well-conditioned one (plain cm) and is held tight; the
  scaled direction carries the near-collapsed-window cancellation described above."""
  detector = _detector(path)
  cube = _cube(detector, 128, 0)

  nominal = jax.vmap(detector._to_nominal_flat)(cube)
  back = np.asarray(jax.vmap(detector._to_scaled_flat)(nominal))
  assert np.max(np.abs(back - np.asarray(cube))) < 5e-3

  # nominal -> scaled -> nominal, in centimetres
  again = np.asarray(jax.vmap(detector._to_nominal_flat)(jnp.asarray(back)))
  assert np.max(np.abs(again - np.asarray(nominal))) < MICRONS


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p)[:-5] for p in CONFIGS])
def test_whole_cube_is_admissible_geometry(path):
  """Every point of the cube -- corners included -- is an ORDERED, non-overlapping,
  magnet-excluding station layout inside `layer_bounds`. This is the property the sequential window
  exists to guarantee, and the affine map has to preserve it everywhere, not on average."""
  detector = _detector(path)
  nominal = np.asarray(jax.vmap(detector._to_nominal_flat)(_cube(detector, 512, 1)))
  assert np.all(np.isfinite(nominal))

  ns = detector.n_stations
  z, angle = nominal[:, :ns], nominal[:, ns]
  half = 0.5 * detector.station_width
  pitch = detector.station_width + detector.station_clearance

  gaps = np.diff(z, axis=1)
  assert np.all(gaps > 0), f"stations out of order (min gap {gaps.min()})"
  assert np.all(gaps >= pitch - MICRONS), f"stations overlap (min gap {gaps.min()} < pitch {pitch})"

  magnet_lo, magnet_hi = detector.z0 - detector.magnet_half_cm, detector.z0 + detector.magnet_half_cm
  intrusion = np.maximum(np.minimum(z + half, magnet_hi) - np.maximum(z - half, magnet_lo), 0.0)
  assert np.max(intrusion) < MICRONS, f"a station intrudes {np.max(intrusion)} cm into the magnet"

  low, high = detector.layer_bounds
  assert np.all(z - half >= low - MICRONS) and np.all(z + half <= high + MICRONS)

  angle_low, angle_high = detector.stereo_bound
  assert np.all(angle >= angle_low - 1e-4) and np.all(angle <= angle_high + 1e-4)
  # the cube's corners reach the angle bounds exactly -- no saturation
  assert np.isclose(angle.min(), angle_low, atol=1e-4) and np.isclose(angle.max(), angle_high, atol=1e-4)


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p)[:-5] for p in CONFIGS])
def test_combine_agrees_between_the_two_spaces(path):
  """`combine(event, nominal)` is defined as `combine_scaled(event, to_scaled(nominal))`; at a design
  whose round trip is exact the two must agree BIT FOR BIT, not approximately."""
  detector = _detector(path)
  event = _fab_event(detector, 4, 2)
  mask = np.ones(event.station.shape, np.int32)
  scaled = jnp.asarray(np.full(detector.design_dim(), 0.5, np.float32))

  from_scaled = np.asarray(detector.combine_scaled(event, scaled, mask=mask))
  from_nominal = np.asarray(detector.combine(event, detector.to_nominal(scaled), mask=mask))
  np.testing.assert_array_equal(from_nominal, from_scaled)

  gradient = np.asarray(jax.grad(lambda d: jnp.sum(detector.combine_scaled(event, d, mask=mask)))(scaled))
  assert np.all(np.isfinite(gradient)) and np.any(np.abs(gradient) > 0)
