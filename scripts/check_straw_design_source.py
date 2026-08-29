#!/usr/bin/env python3
"""Does ``design_features='straw-geometry'`` hand the gate the DESIGN and nothing else?

The four-feature straw combine emits ``[TDC, norm_z, wire_y_left, wire_y_right]`` and appends no
design vector, so an alpha-hypernetwork on this task has to recover the design from the measurement.
This asserts the recovery rather than assuming it:

* every live hit's ``norm_z`` is one of the design's own layer z's;
* every live hit's ``wire_y_right - wire_y_left`` is one of ``2 * layer_width * tan(angle) / y_scale``,
  i.e. the fired straw CANCELS -- checked by re-simulating the SAME design on DISJOINT events, where
  the raw wire column moves and the span set does not;
* two different designs give different spans, so the source is not degenerate;
* the model is capacity-matched to its ``zero_design`` baseline, is the identity at initialisation for
  both, and diverges from it once the gates leave zero -- which is the gate reading the design.

Exit code 0 means every assertion held.
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from gearup.config import read_config

import detopt.detector
import detopt.nn
from detopt.nn.alpha_hyper_set_regressor import design_from_features

TOLERANCE = 1e-5


def features_at(detector, design_scaled, lo, hi):
  index = np.arange(lo, hi, dtype=np.int32)
  design = detector.to_nominal(np.asarray(design_scaled, np.float32))
  batched = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (index.shape[0], ) + jnp.asarray(x).shape), design)
  _truth, event, mask, _target = detector(batched, index)
  combined = np.asarray(detector.combine_scaled(event, np.asarray(design_scaled, np.float32), mask=mask))
  return combined, np.asarray(mask).astype(bool)


def design_levels(detector, design_scaled):
  """The two design coordinates the source claims to recover, computed from the GEOMETRY MAP."""
  positions, angles, _field = detector._scaled_to_layer_geometry(jnp.asarray(design_scaled, jnp.float32)[None, :])
  positions, angles = np.asarray(positions)[0], np.asarray(angles)[0]
  a_max = max(abs(detector.angle_bounds[0]), abs(detector.angle_bounds[1]))
  y_scale = max(detector.layer_height + detector.layer_width * float(np.tan(a_max)), 1e-6)
  z_mid = 0.5 * (detector.layer_bounds[0] + detector.layer_bounds[1])
  z_half = max(0.5 * (detector.layer_bounds[1] - detector.layer_bounds[0]), 1e-6)
  return (positions - z_mid) / z_half, 2.0 * detector.layer_width * np.tan(angles) / y_scale


def nearest(values, levels):
  return float(np.max(np.min(np.abs(np.asarray(values)[:, None] - np.asarray(levels)[None, :]), axis=1)))


def main(config_path='config/bo_prec1e2_alphahyper.yaml', n_events: int = 64):
  _, config = read_config([], config_path)
  detector = detopt.detector.from_config(config['detector'])
  dimension = detector.design_dim()
  rng = np.random.default_rng(7)
  a, b = rng.uniform(size=dimension).astype(np.float32), rng.uniform(size=dimension).astype(np.float32)

  extracted = {}
  for name, design in (('A', a), ('B', b)):
    combined, mask = features_at(detector, design, 0, n_events)
    source = np.asarray(design_from_features(jnp.asarray(combined), 'straw-geometry', 2))[mask]
    z_levels, span_levels = design_levels(detector, design)
    z_error, span_error = nearest(source[:, 0], z_levels), nearest(source[:, 1], span_levels)
    print(
      f'[{name}] live hits {int(mask.sum())}  norm_z to nearest design layer z: {z_error:.3e}  '
      f'span to nearest design angle span: {span_error:.3e}  distinct spans {len(np.unique(np.round(source[:, 1], 6)))}',
      flush=True
    )
    assert z_error < TOLERANCE, f'{name}: norm_z is not a design layer z ({z_error:.3e})'
    assert span_error < TOLERANCE, f'{name}: span is not a design angle span ({span_error:.3e})'
    extracted[name] = (combined, mask, source)

  # The straw must cancel: disjoint events at the SAME design move the raw wire column, not the span.
  other, other_mask = features_at(detector, a, 4096, 4096 + n_events)
  other_source = np.asarray(design_from_features(jnp.asarray(other), 'straw-geometry', 2))[other_mask]
  first = extracted['A'][2]
  wire_shift = abs(float(extracted['A'][0][extracted['A'][1]][:, 2].mean()) - float(other[other_mask][:, 2].mean()))
  span_a = np.sort(np.unique(np.round(first[:, 1], 6)))
  span_other = np.sort(np.unique(np.round(other_source[:, 1], 6)))
  assert span_a.shape == span_other.shape, f'span level count changed with the events: {span_a} vs {span_other}'
  span_drift = float(np.max(np.abs(span_a - span_other)))
  print(
    f'[straw] disjoint events, same design: wire_y_left mean moves {wire_shift:.4e}, span set moves {span_drift:.3e}',
    flush=True
  )
  assert span_drift < TOLERANCE, f'span depends on which straw fired ({span_drift:.3e})'
  assert wire_shift > TOLERANCE, 'the raw wire column did not move, so this is not a real straw contrast'

  separation = abs(float(first[:, 1].mean()) - float(extracted['B'][2][:, 1].mean()))
  print(
    f'[design] span mean A {first[:, 1].mean():.5f} vs B {extracted["B"][2][:, 1].mean():.5f}  separation {separation:.5f}',
    flush=True
  )
  assert separation > 1e-3, f'two designs give the same span ({separation:.3e}), so the source carries no design'

  # The model, and its capacity-matched blind twin.
  block = config['regressor']['alpha-hyper-set-regressor']
  sighted = detopt.nn.from_config(detector, {'alpha-hyper-set-regressor': dict(block)}, rngs=nnx.Rngs(0))
  blind = detopt.nn.from_config(detector, {'alpha-hyper-set-regressor': dict(block, zero_design=True)}, rngs=nnx.Rngs(0))
  counts = [sum(int(np.prod(v.shape)) for v in jax.tree.leaves(nnx.state(m, nnx.Param))) for m in (sighted, blind)]
  leaves = [len(jax.tree.leaves(nnx.state(m, nnx.Param))) for m in (sighted, blind)]
  print(f'[model] params {counts[0]} vs {counts[1]} (zero_design), leaves {leaves[0]} vs {leaves[1]}', flush=True)
  assert counts[0] == counts[1] and leaves[0] == leaves[1], 'the blind twin is not capacity-matched'

  combined_a, mask_a = extracted['A'][0], extracted['A'][1]
  x, m = jnp.asarray(combined_a), jnp.asarray(mask_a)
  out_sighted, out_blind = np.asarray(sighted(x, m)), np.asarray(blind(x, m))
  identity = float(np.max(np.abs(out_sighted - out_blind)))
  print(f'[init] alpha == 0, so sighted and blind agree to {identity:.3e}', flush=True)
  assert identity < 1e-6, f'the gates are not zero at initialisation ({identity:.3e})'

  moved = 0
  for model in (sighted, blind):
    state = nnx.state(model, nnx.Param)
    for path, leaf in state.flat_state():
      if 'gate' in [str(p) for p in path] and str(path[-1]) == 'kernel':
        leaf.value = leaf.value + 0.3
        moved += 1
    nnx.update(model, state)
  gated = float(np.max(np.abs(np.asarray(sighted(x, m)) - np.asarray(blind(x, m)))))
  print(f'[gate] {moved} gate kernels moved off zero; sighted now differs from blind by {gated:.5f}', flush=True)
  assert gated > 1e-4, f'the gate does not read the design ({gated:.3e})'

  first_pass = np.asarray(sighted(x, m, deterministic=False, rngs=nnx.Rngs(dropconnect=1)))
  second_pass = np.asarray(sighted(x, m, deterministic=False, rngs=nnx.Rngs(dropconnect=2)))
  spread = float(np.max(np.abs(first_pass - second_pass)))
  configured = block.get('dropconnect', None)
  print(f'[dropconnect] configured {configured}; two stochastic passes differ by {spread:.5f}', flush=True)
  if configured is not None and configured > 0:
    assert spread > 0, 'dropconnect is configured but inert'

  print('ALL CHECKS PASSED', flush=True)


if __name__ == '__main__':
  import sys
  import gearup

  gearup.gearup(main)(sys.argv[1:])
