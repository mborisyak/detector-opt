#!/usr/bin/env python3
"""`scripts/bo.py` with the regressor BLINDED to the design: a constant scaled design is substituted
inside ``combine_scaled`` while events keep being simulated at the TRUE design.

    python scripts/bo_blind_design.py constant_design=mid =bo_rw025 output=... seed=... \\
        nn_init_strategy=meta_ratio

WHY A LAUNCHER AND NOT A DETECTOR OPTION. The detector holds NO design -- that rule is absolute. This
installs the substitution on the detector INSTANCE that ``bo.py`` builds, exactly as
``scripts/probe_continue_adaptation.py`` does for its probe cells, by wrapping
``detopt.detector.from_config`` for the duration of the run. Nothing under ``detopt/detector/`` is
touched, no class gains a stored design, and ``detector(design, event_index)`` still simulates at the
design the optimiser proposed.

WHAT BLINDING MEANS HERE. SHiP has no separate design input: the design reaches the network ONLY
through ``_scaled_to_layer_geometry`` -- layer ``positions`` become ``norm_z``, ``angles`` become
``wire_y_left``/``wire_y_right`` -- while ``TDC`` is design-independent. Substituting a constant scaled
design therefore freezes the geometry the network is told about, while the hits it sees still come from
the real geometry. The network is blind to WHICH design produced the event.

⛔️ NOT ``zeros``. ``stereo_bound = (0.0, 0.2)`` puts ``view_angle`` at EXACTLY 0 at the lower corner,
making ``wire_y_left == wire_y_right`` for every hit -- the stereo channel collapses and "design
withheld" becomes confounded with "a feature destroyed". Measured: stereo span 0 under zeros against
0.115 on the true path. ``mid`` is the default and is a real geometry with a non-zero angle.

THE OVERRIDE IS CHECKED BEFORE THE RUN STARTS, not assumed, on ONE event set with only the design
varying: two different designs must produce IDENTICAL features under the substitution while
DIFFERING on the true path, the substitution must MOVE the features away from that true path,
``TDC`` must be untouched, and the stereo channel must survive. Any failure aborts.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

import detopt.detector

TDC_COLUMN = 0
GEOMETRY_COLUMNS = (1, 2, 3)


def constant_scaled_design(detector, choice):
  """The SCALED design substituted for whatever the optimiser proposed."""
  dimension = detector.design_dim()
  if choice == 'mid':
    return np.full((dimension, ), 0.5, dtype=np.float32)
  if choice == 'zeros':
    return np.zeros((dimension, ), dtype=np.float32)
  import json

  values = np.asarray(json.loads(choice), dtype=np.float32)
  if values.shape != (dimension, ):
    raise SystemExit(f'constant_design must hold {dimension} values, got {tuple(values.shape)}')
  return values


def install(detector, constant):
  """Route ``combine_scaled`` through a fixed design. ``combine`` is final and calls
  ``self.combine_scaled``, so wrapping the instance attribute is enough."""
  original = detector.combine_scaled
  fixed = jnp.asarray(constant, dtype=jnp.float32)

  def combine_scaled(event, design_scaled, mask=None):
    given = jnp.asarray(design_scaled, dtype=jnp.float32)
    return original(event, jnp.broadcast_to(fixed, given.shape), mask=mask)

  detector.combine_scaled = combine_scaled
  return original


def verify(detector, original, constant, n_events=256):
  """Abort unless the substitution demonstrably bites and leaves the measurement intact.

  ONE event set, simulated once at design ``a``; only the scaled design handed to the combine varies.
  The claim under test is about ``combine_scaled`` alone, so re-simulating per design would compare
  two different hit patterns and no correct substitution could ever come out identical -- measured,
  the geometry columns then differ by 1.8 whatever the substitution does. ``probe_continue_adaptation``
  checks the same override the same way.

  The true-path contrast is reported too: ``frozen == 0`` says nothing unless the two probe designs
  are demonstrably distinguishable when the design IS passed through.
  """
  rng = np.random.default_rng(0)
  a, b = rng.uniform(size=detector.design_dim()), rng.uniform(size=detector.design_dim())
  index = np.arange(int(n_events), dtype=np.int32)
  nominal = detector.to_nominal(np.asarray(a, np.float32))
  batched = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (index.shape[0], ) + jnp.asarray(x).shape), nominal)
  _truth, event, mask, _target = detector(batched, index)
  live = np.asarray(mask).astype(bool)

  def features(design_scaled, combine):
    scaled = jnp.broadcast_to(jnp.asarray(design_scaled, jnp.float32), (index.shape[0], detector.design_dim()))
    return combine(event, scaled, mask=mask)

  fa, fb = features(a, detector.combine_scaled), features(b, detector.combine_scaled)
  ta, tb = features(a, original), features(b, original)

  def spread(x, y, column):
    u, v = np.asarray(x)[..., column], np.asarray(y)[..., column]
    return float(np.max(np.abs(u - v)[live])) if live.any() else float('nan')

  frozen = max(spread(fa, fb, c) for c in GEOMETRY_COLUMNS)
  contrast = max(spread(ta, tb, c) for c in GEOMETRY_COLUMNS)
  moved = max(spread(fa, ta, c) for c in GEOMETRY_COLUMNS)
  tdc = spread(fa, ta, TDC_COLUMN)
  stereo = float(np.max(np.abs(np.asarray(fa)[..., 3] - np.asarray(fa)[..., 2])[live]))
  print(
    f'[blind] constant={np.asarray(constant).tolist()}  hits={int(live.sum())}/{live.size}  '
    f'geometry frozen across designs: {frozen:.3e}  (same two designs on the true path: {contrast:.3e})  '
    f'moved from true: {moved:.3e}  TDC delta: {tdc:.3e}  stereo span: {stereo:.4f}', flush=True
  )
  if contrast < 1e-6:
    raise SystemExit(f'[blind] FAILED: the two probe designs differ by {contrast:.3e} even unblinded -- nothing to freeze')
  if frozen > 1e-6:
    raise SystemExit(f'[blind] FAILED: two designs still differ by {frozen:.3e} -- the override does not bite')
  if moved < 1e-6:
    raise SystemExit('[blind] FAILED: the override changes nothing against the true-design path')
  if tdc > 1e-6:
    raise SystemExit(f'[blind] FAILED: TDC moved by {tdc:.3e} -- the override must not touch the measurement')
  if stereo < 1e-4:
    raise SystemExit(f'[blind] FAILED: stereo span {stereo:.3e} -- the two wire columns collapsed')


def main():
  arguments = list(sys.argv[1:])
  choice = 'mid'
  for token in list(arguments):
    if token.startswith('constant_design='):
      choice = token.split('=', 1)[1]
      arguments.remove(token)

  original_from_config = detopt.detector.from_config

  def from_config(spec):
    detector = original_from_config(spec)
    constant = constant_scaled_design(detector, choice)
    original = install(detector, constant)
    verify(detector, original, constant)
    return detector

  detopt.detector.from_config = from_config

  import gearup

  import bo as bo_module

  gearup.gearup(bo_module.bo).with_config('config/bo.yaml')(arguments)


if __name__ == '__main__':
  sys.path.insert(0, __file__.rsplit('/', 1)[0])
  main()
