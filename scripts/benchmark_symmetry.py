#!/usr/bin/env python3
"""How should a BATCH's permutation symmetry be handled -- in the kernel, in the parameterisation,
or not at all?

The design is the shared-fraction one: a single enzyme stock fraction for the whole batch plus
``n_experiments`` temperatures, so the symmetric group acts on SCALARS and the quotient is clean.
Four arms, identical objective, identical budget, differing only in how the symmetry is treated:

``ard``       plain ARD-RBF on (fraction, T_1..T_m). Ignores the symmetry: the surrogate models m!
              copies of every optimum and spends one lengthscale per SLOT on a distinction the
              problem cannot make.
``sort``      ARD-RBF on (fraction, sort(T)). Exactly invariant and constant-diagonal, because
              sorting realises the quotient metric for scalars (||sort a - sort b|| = min_pi ||a - pi b||).
              Not smooth across a tie: folding a non-symmetric function creases it.
``reparam``   ARD-RBF on (fraction, p_1..p_m), where p maps the cube onto the ORDERED simplex by
              stick-breaking. Never leaves the fundamental domain, so there is no fold to cross and
              no boundary condition to impose; smooth, invariant by construction.
``random``    the null. Uniform temperatures, which is the same physical distribution all three
              searched spaces induce -- so one null is valid for every arm.

    python scripts/benchmark_symmetry.py --seeds 5 --iterations 30 --output output/symmetry.json
"""

import argparse
import json
import os
import time

import numpy as np

import detopt
import detopt.utils.config
from detopt.bo import BayesianOptimizer, PermutationInvariantRBF
from detopt.bo.gbdt import score_design

CONFIG = "config/detector/enzyme.yaml"
GP = {"n_folds": 5, "n_restarts": 5, "n_steps": 40,
      "log_lengthscale_prior_bounds": [-6.0, 4.0], "log_amplitude_prior_bounds": [-6.0, 1.5]}
EI = {"n_restarts": 32, "n_steps": 100}


class SortingRBF(PermutationInvariantRBF):
  """ARD-RBF applied to the design with its temperature block SORTED.

  Invariance by folding rather than by averaging: ``k(x, y) = k_ard(sigma x, sigma y)``. PSD for
  free (composition with any map preserves it), exactly invariant since ``sigma(pi x) = sigma(x)``,
  and its diagonal is CONSTANT -- so, unlike the group average, its prior variance does not peak on
  the degenerate all-tied designs and EI has no reason to chase them.

  The price is a crease: the one-sided derivatives across a tie differ by ``c E (w_1 - w_0)/l^2``,
  driven by the spread of the OTHER point, so it survives even with a shared lengthscale.
  ``grad_diag`` is exactly zero because the diagonal is constant."""

  # The signature must name every hyperparameter explicitly: sklearn's `get_params` (and hence
  # `clone`, which the GP calls on every fit) reads them off the __init__ signature, so **kwargs
  # would hide them and `theta` would come back empty.
  def __init__(self, d, block=(), constant_value=1.0, constant_value_bounds=(1e-5, 1e5),
               length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
    super().__init__(d=d, blocks=(), constant_value=constant_value,
                     constant_value_bounds=constant_value_bounds, length_scale=length_scale,
                     length_scale_bounds=length_scale_bounds)
    self.block = block
    # As an ARRAY: `x[(1,2,3,4)]` on a 1-D vector is multi-dimensional indexing, not a gather.
    self._index = np.asarray(block, dtype=int)

  def _sorted(self, X):
    X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
    X[:, self._index] = np.sort(X[:, self._index], axis=1)
    return X

  def __call__(self, X, Y=None, eval_gradient=False):
    return super().__call__(self._sorted(X), None if Y is None else self._sorted(Y), eval_gradient)

  def diag(self, X):
    return super().diag(self._sorted(X))

  def grad_diag(self, x):
    return np.zeros(self.d)  # k(x, x) is the amplitude everywhere

  def k_and_grad_x(self, X_train, x):
    """Chain rule through the sort: its Jacobian is a permutation matrix almost everywhere, so the
    sorted-space gradient is scattered back through the inverse permutation."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x[self._index])
    k, jac = super().k_and_grad_x(self._sorted(X_train), self._sorted(x))
    scattered = np.array(jac, copy=True)
    scattered[:, self._index[order]] = jac[:, self._index]
    return k, scattered


class NormalisedInvariantRBF(PermutationInvariantRBF):
  """The group-averaged kernel, normalised to a unit diagonal: ``k(x,y) / sqrt(k(x,x) k(y,y))``.

  Keeps everything that makes the group average principled -- exact invariance, and smoothness at
  ties, which sorting gives up -- while removing the one property that misdirects the acquisition:
  its prior variance is ``|G|`` times larger on the fully-tied stratum than at a generic design, so
  EI's exploration term is maximal exactly on the all-identical batch. (This docstring used to say
  "measured 11.2x", which is not reproducible in this setting; over 2000 tied against 2000 generic
  designs on the full 8-D layout it is 6.3x at the config's initial lengthscale and 23.0x at
  l = 0.05, rising toward |G| = 24 as the lengthscale shortens.)

  ⚠️ THE IMPLEMENTATION BELOW IS BROKEN and its benchmark row is retracted: normalising as the bare
  k/sqrt(k(x,x)k(y,y)) cancels the amplitude, so k(x,x) == 1 for every `constant_value` and
  d k/d log(constant_value) == 0 exactly. The marginal likelihood is flat in that hyperparameter and
  the GP cannot scale its prior to the data. Use `detopt.bo.NormalisedInvariantRBF`, which keeps the
  amplitude outside the normalisation. Left here unfixed only so the retracted row remains
  reproducible.

  Still PSD: this is ``D^{-1/2} K D^{-1/2}``, a congruence. Still exactly invariant, since both the
  kernel and the diagonal are. What it gives up is the interpretation -- the image-method variance
  IS the Neumann prior for the fundamental domain, so normalising it means no longer modelling that.
  Whether that trade pays is the question this benchmark exists to answer."""

  def __call__(self, X, Y=None, eval_gradient=False):
    if not eval_gradient:
      K = super().__call__(X, Y, False)
      a = super().diag(np.atleast_2d(X))
      b = a if Y is None else super().diag(np.atleast_2d(Y))
      return K / np.sqrt(np.outer(a, b))
    # sklearn only asks for the hyperparameter gradient on the symmetric call, so a == b here.
    K, dK = super().__call__(X, None, True)
    a = np.diag(K).copy()
    a_grad = np.einsum("iit->it", dK)                       # d diag / d theta, (n, n_theta)
    norm = np.sqrt(np.outer(a, a))
    value = K / norm
    # d(K/N)/dt = dK/dt / N - (K/N) * ( a'_i/(2 a_i) + a'_j/(2 a_j) )
    share = a_grad / (2.0 * a[:, None])                      # (n, n_theta)
    gradient = dK / norm[:, :, None] - value[:, :, None] * (share[:, None, :] + share[None, :, :])
    return value, gradient

  def diag(self, X):
    return np.ones(np.atleast_2d(X).shape[0])

  def grad_diag(self, x):
    return np.zeros(self.d)

  def k_and_grad_x(self, X_train, x):
    x = np.asarray(x, dtype=float).ravel()
    k, jac = super().k_and_grad_x(X_train, x)
    kxx = float(super().diag(x[None, :])[0])
    a = super().diag(np.atleast_2d(X_train))
    norm = np.sqrt(kxx * a)
    value = k / norm
    # only k(x,x) depends on x, so the correction is one term
    return value, jac / norm[:, None] - value[:, None] * super().grad_diag(x)[None, :] / (2.0 * kxx)


def stick_breaking(p, low, high):
  """Cube -> ORDERED temperatures, measure-preserving.

  ``T_k = T_{k-1} + (high - T_{k-1}) (1 - (1 - p_k)^{1/(m-k)})`` is the inverse CDF of the k-th
  spacing, so a uniform draw in the cube maps to the order statistics of a uniform draw. The naive
  ``T_k = T_{k-1} + p_k (high - T_{k-1})`` is the same map without the exponent and is badly
  non-uniform -- it crowds every point against the upper bound (measured mean order statistics
  [52.5, 66.2, 73.1, 76.6] against the uniform [36.0, 47.0, 58.0, 69.0]), which would hand the
  random-search null a systematically worse design distribution than the one BO searches."""
  m = len(p)
  temperatures, previous = [], low
  for k in range(m):
    previous = previous + (high - previous) * (1.0 - (1.0 - p[k]) ** (1.0 / (m - k)))
    temperatures.append(previous)
  return np.asarray(temperatures)


def run(arm, seed, iterations, n_events, detector, low, high):
  m = detector.n_experiments
  d = 1 + m
  block = list(range(1, d))

  def to_design(x):
    fraction = float(x[0])
    temperatures = (stick_breaking(np.asarray(x[1:], float), low, high) if arm == "reparam"
                    else low + (high - low) * np.asarray(x[1:], float))
    return np.concatenate([np.full(m, fraction), temperatures]).astype(np.float32)

  kernel = None
  if arm in ("ard", "sort", "reparam", "sym", "symnorm"):
    ls_low, ls_high = GP["log_lengthscale_prior_bounds"]
    amp_low, amp_high = GP["log_amplitude_prior_bounds"]
    common = dict(constant_value=float(np.exp(amp_low + amp_high)),
                  constant_value_bounds=(float(np.exp(2 * amp_low)), float(np.exp(2 * amp_high))),
                  length_scale=float(np.exp(0.5 * (ls_low + ls_high))),
                  length_scale_bounds=(float(np.exp(ls_low)), float(np.exp(ls_high))))
    if arm == "sort":
      kernel = SortingRBF(d=d, block=tuple(block), **common)
    elif arm == "sym":
      kernel = PermutationInvariantRBF(d=d, blocks=(tuple(block),), **common)
    elif arm == "symnorm":
      kernel = NormalisedInvariantRBF(d=d, blocks=(tuple(block),), **common)
    else:
      kernel = PermutationInvariantRBF(d=d, blocks=(), **common)
    if arm in ("sym", "symnorm"):
      kernel.length_scale = np.full(kernel.n_length_scales, float(np.ravel(kernel.length_scale)[0]))

  optimiser = BayesianOptimizer(d, gp=GP, ei=EI, kernel=kernel, n_init=GP["n_folds"], seed=seed)
  rng = np.random.default_rng(seed)
  results, best = [], np.inf
  for iteration in range(iterations):
    start = time.time()
    x = (rng.uniform(0.0, 1.0, d).astype(np.float32) if arm == "random"
         else np.asarray(optimiser.propose(), dtype=np.float32))
    design = to_design(x)
    score = score_design(detector, design, n_events=n_events, event_offset=0, seed=0)
    if arm != "random":
      optimiser.append(x, score.loss, noise=max(score.sem, 1e-6))
    best = min(best, score.loss)
    results.append({"iteration": iteration, "loss": score.loss, "best": best,
                    "design": design.tolist(), "seconds": time.time() - start})
    print(f"[{arm} seed {seed}] {iteration + 1:3d}/{iterations} loss={score.loss:.5f} best={best:.5f}",
          flush=True)
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--seeds", type=int, default=5)
  parser.add_argument("--iterations", type=int, default=30)
  parser.add_argument("--n-events", type=int, default=4096)
  parser.add_argument("--output", default="output/symmetry.json")
  arguments = parser.parse_args()

  detector = detopt.detector.from_config(detopt.utils.config.load_config(CONFIG))
  low, high = detector.temperature_bounds
  celsius = 0.5 * (detector.melting_bounds[1] - detector.melting_bounds[0])

  payload = {"arms": {}, "settings": {"n_events": arguments.n_events, "iterations": arguments.iterations,
                                      "seeds": arguments.seeds, "temperature_bounds": [low, high],
                                      "n_experiments": int(detector.n_experiments)}}
  for arm in ("ard", "sort", "reparam", "sym", "symnorm", "random"):
    payload["arms"][arm] = [run(arm, s, arguments.iterations, arguments.n_events, detector, low, high)
                            for s in range(arguments.seeds)]

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump(payload, f, indent=2, default=float)

  print(f"\n{'arm':<9} {'@10':>8} {'@20':>8} {'final':>8} {'C':>6}   spread over seeds")
  for arm, runs in payload["arms"].items():
    curves = np.array([[r["best"] for r in run_] for run_ in runs])
    median = np.median(curves, axis=0)
    at = lambda k: median[min(k, median.size) - 1]
    print(f"{arm:<9} {at(10):8.4f} {at(20):8.4f} {median[-1]:8.4f} {np.sqrt(median[-1]) * celsius:6.2f}   "
          f"[{curves[:, -1].min():.4f}, {curves[:, -1].max():.4f}]")
  print(f"wrote {arguments.output}")


if __name__ == "__main__":
  main()
