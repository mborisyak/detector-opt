#!/usr/bin/env python3
"""Does a PARAMETER-FREE layer norm after each hidden dense layer change what verification reports?

    python scripts/probe_layernorm.py =intersect \
        trajectory=output/final-bo/intersect/select/<seed>/<strategy>/<regime>/results.json \
        seed=<seed> verify.n_points=1

Runs `scripts.verify_trajectory`'s own verification TWICE on the same trajectory point -- once as the
campaign runs it, once with a layer norm spliced into every set block -- and reports wall time and the
held-out test loss for each.

WHY PARAMETER-FREE. Verification restores the network the run REPORTED each design with, from that
design's checkpoint. A layer norm with learnable scale/bias would add leaves the checkpoint does not
have, so the restore could not describe the same network and the comparison would be between a trained
network and a partly fresh one. Dropping the affine makes the normalisation a pure function -- no
`nnx.Param`, nothing in the parameter pytree -- so THE SAME WEIGHTS load into both arms and the only
difference is the forward pass. That is the comparison worth making.

WHY IT LIVES HERE AND NOT IN `detopt.nn`. This is a probe, not a feature. `set_regressor.py` is used by
three live campaigns; an architecture flag added for an experiment is a flag someone later has to prove
is off. The splice below reaches into `EnsembleSetBlock.shared` at runtime and touches nothing on disk.

WHAT IT DOES NOT ESTABLISH. One trajectory point on one cell. Verification trains for `verify.epochs`
on a freshly sampled split, so the two arms see identical data, but a single point cannot separate an
architecture effect from that point's own noise -- run several before believing an ordering.
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import gearup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ParameterFreeLayerNorm(nnx.Module):
  """``(x - mean) / sqrt(var + eps)`` over the trailing feature axis, no affine.

    Contributes NOTHING to the parameter pytree -- that is the point: the checkpoint's weights load
    into a model containing these exactly as they load into one without them. Statistics are per row
    over the last axis, so the ``M`` (hit) axis never enters and a masked slot cannot reach a live one.
    """

  def __init__(self, epsilon: float = 1e-6):
    self.epsilon = float(epsilon)

  def __call__(self, x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + self.epsilon)


def splice_layer_norm(model):
  """Insert a :class:`ParameterFreeLayerNorm` after every hidden ``EnsembleLinear`` in every block.

    Returns the number of layer norms inserted. Mutates ``model`` in place; call it AFTER the
    checkpoint has been restored, so the restore sees the architecture it was saved from.
    """
  from detopt.nn.set_regressor import EnsembleLinear

  inserted = 0
  for block in model.blocks:
    rebuilt = []
    for layer in block.shared:
      rebuilt.append(layer)
      if isinstance(layer, EnsembleLinear):
        rebuilt.append(ParameterFreeLayerNorm())
        inserted += 1
    block.shared = nnx.List(rebuilt)
  return inserted


def parameter_count(model):
  return sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def probe(trajectory: str, seed: int = 0, arm: str = "both", **config):
  """Run verification with and without the spliced layer norm and print a comparison."""
  import verify_trajectory as vt

  results = {}
  for name in (("baseline", "layernorm") if arm == "both" else (arm, )):
    original = vt.detopt.nn.from_config

    def patched(detector, config, *, rngs, design=True, _name=name, _orig=original):
      model = _orig(detector, config=config, rngs=rngs, design=design)
      if _name == "layernorm":
        n = splice_layer_norm(model)
        print(f"[{_name}] spliced {n} parameter-free layer norms; "
              f"parameters {parameter_count(model)} (unchanged by construction)", flush=True)
      return model

    vt.detopt.nn.from_config = patched
    started = time.time()
    try:
      vt.verify(trajectory=trajectory, seed=seed, progress="plain", force=True, **config)
    finally:
      vt.detopt.nn.from_config = original
    elapsed = time.time() - started

    run_dir = os.path.dirname(trajectory)
    payload = json.load(open(os.path.join(run_dir, "verification.json")))
    point = sorted(payload["points"], key=lambda p: p["detector_calls"])[-1]
    results[name] = {"wall_s": elapsed, "test_loss": point["test_loss"], "test_sem": point["test_sem"],
                     "reported": point["reported_loss"], "best_epoch": point["best_epoch"]}
    os.replace(os.path.join(run_dir, "verification.json"),
               os.path.join(run_dir, f"verification_{name}.json"))

  print("\n=== layer-norm probe ===")
  for name, r in results.items():
    print(f"  {name:<10} test_loss={r['test_loss']:.4f} +/- {r['test_sem']:.4f}  "
          f"best_epoch={r['best_epoch']:<3} wall={r['wall_s']/60:.1f} min  (reported {r['reported']:.4f})")
  if len(results) == 2:
    b, l = results["baseline"], results["layernorm"]
    d = l["test_loss"] - b["test_loss"]
    pooled = float(np.hypot(b["test_sem"], l["test_sem"]))
    print(f"  delta (layernorm - baseline): {d:+.4f} +/- {pooled:.4f}  "
          f"({'layer norm better' if d < 0 else 'baseline better'}, {abs(d)/pooled:.1f} sigma)")
    print(f"  speed: {l['wall_s']/b['wall_s']:.2f}x baseline wall time")


if __name__ == "__main__":
  gearup.gearup(probe).with_config("config/root.yaml")(sys.argv[1:])
