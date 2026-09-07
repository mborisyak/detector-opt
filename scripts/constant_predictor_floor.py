"""No-information floor of a SHiP task: the loss of the best CONSTANT predictor under the task's own loss.

    JAX_PLATFORMS=cpu python scripts/constant_predictor_floor.py config/angle.yaml [n_events] [theta ...]

Draws ``n_events`` events at each scaled design ``theta`` (every coordinate set to that value; default 0.5),
standardises the targets exactly as training does (``normalize_target``), and reports (a) the loss of the target
MEAN as predictor and (b) the loss of the constant that minimises the task loss, found by Adam from several starts
(the pairing minimum in the daughter term makes the optimum a PAIR of constants, not the mean). Also prints the
per-component variance of the standardised target (1.0 when the reference sigma equals the sample's) and the
per-component correlation of the two daughters' momenta, which sets the HNL-sum term. Read-only, CPU, no output
files.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

import detopt.detector


def draw(det, theta, n, batch=1024):
  rows = []
  for o in range(0, n, batch):
    idx = np.arange(o, min(o + batch, n), dtype=np.int64)
    phys = det.to_nominal(jnp.broadcast_to(jnp.full((det.design_dim(), ), theta, jnp.float32)[None, :], (idx.shape[0], det.design_dim())))
    _gt, _event, _mask, target = det(phys, idx)
    rows.append(np.asarray(det.normalize_target(target)))
  return np.concatenate(rows, axis=0)


def terms(det, c, t):
  c = jnp.broadcast_to(jnp.asarray(c, jnp.float32)[None, :], t.shape)
  vp, vt = c[:, :3], t[:, :3]
  p1p, p2p, p1t, p2t = c[:, 3:6], c[:, 6:9], t[:, 3:6], t[:, 6:9]
  vertex = 0.5 * jnp.mean(jnp.square(vp - vt), axis=-1)
  daughters = 0.25 * jnp.minimum(
    jnp.mean(jnp.square(p1p - p1t) + jnp.square(p2p - p2t), axis=-1), jnp.mean(jnp.square(p2p - p1t) + jnp.square(p1p - p2t), axis=-1)
  )
  hnl = 0.25 * jnp.mean(jnp.square(0.5 * ((p1p + p2p) - (p1t + p2t))), axis=-1)
  return float(jnp.mean(det.loss(c, t))), float(jnp.mean(vertex)), float(jnp.mean(daughters)), float(jnp.mean(hnl))


def optimise(det, t, c0, steps=3000, lr=0.02):
  t = jnp.asarray(t, jnp.float32)
  f = lambda c: jnp.mean(det.loss(jnp.broadcast_to(c[None, :], t.shape), t))
  opt = optax.adam(lr)
  c = jnp.asarray(c0, jnp.float32)
  state = opt.init(c)

  @jax.jit
  def step(c, state):
    val, g = jax.value_and_grad(f)(c)
    upd, state = opt.update(g, state, c)
    return optax.apply_updates(c, upd), state, val

  for _ in range(steps):
    c, state, _ = step(c, state)
  return np.asarray(c), float(f(c))


def main():
  cfg = yaml.safe_load(open(sys.argv[1]))
  n = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
  thetas = [float(x) for x in sys.argv[3:]] or [0.5]
  det = detopt.detector.from_config(cfg["detector"])
  print(f"task {sys.argv[1]}: design_dim {det.design_dim()}, target_dim {det.target_dim()}, n_events {n}")
  for theta in thetas:
    t = draw(det, theta, n)
    mean = t.mean(0)
    var = t.var(0)
    rho = [float(np.corrcoef(t[:, 3 + k], t[:, 6 + k])[0, 1]) for k in range(3)]
    print(f"--- scaled design theta={theta}: standardised target variance per component {np.round(var, 3).tolist()}")
    print(f"    daughter means p1 {np.round(mean[3:6], 3).tolist()} p2 {np.round(mean[6:9], 3).tolist()}; corr(p1_c, p2_c) {np.round(rho, 3).tolist()} (mean {np.mean(rho):.3f})")
    l, v, d, h = terms(det, mean, jnp.asarray(t))
    print(f"    MEAN predictor: loss {l:.4f} = vertex {v:.4f} + daughters {d:.4f} + hnl {h:.4f}")
    best = (None, np.inf)
    starts = [mean]
    for k in range(3):
      for s in (0.5, 1.0, 2.0):
        c = mean.copy()
        c[3 + k] += s * np.sqrt(var[3 + k])
        c[6 + k] -= s * np.sqrt(var[6 + k])
        starts.append(c)
    rng = np.random.default_rng(0)
    starts += [mean + rng.normal(0.0, 0.7, size=mean.shape) * np.sqrt(var) for _ in range(6)]
    for c0 in starts:
      c, val = optimise(det, t, c0)
      if val < best[1]:
        best = (c, val)
    c = best[0]
    l, v, d, h = terms(det, c, jnp.asarray(t))
    print(f"    OPTIMAL constant: loss {l:.4f} = vertex {v:.4f} + daughters {d:.4f} + hnl {h:.4f}")
    print(f"    optimal constant (standardised): vertex {np.round(c[:3], 3).tolist()} p1 {np.round(c[3:6], 3).tolist()} p2 {np.round(c[6:9], 3).tolist()}")


if __name__ == "__main__":
  main()
