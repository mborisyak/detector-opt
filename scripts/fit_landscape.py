#!/usr/bin/env python3
"""Is the enzyme objective reproducible in closed form?

    python scripts/fit_landscape.py --directory output/landscape

Takes the Sobol samples from `scripts/sample_landscape.py` and asks, at each batch size, how well
the loss is predicted by (a) a GBDT, (b) a GP -- the two surrogates BO actually uses, as a ceiling
on what any model can extract -- and (c) a handful of ANALYTIC forms.

The analytic forms are not arbitrary. The physics says `T_melting` enters through a saturating
folded fraction, so a single experiment is informative only near the transition; a batch should
therefore be an AGGREGATION of a per-experiment response. That makes m = 1 special: it measures the
single-experiment response `g` directly, with no set structure at all. The interesting test is then
whether m = 2..6 are predicted by aggregating the g fitted at m = 1 --

    product   y = y0 * prod_i (1 - g(T_i))      independent experiments, diminishing returns in m
    best      y = y0 * (1 - max_i g(T_i))       only the best-placed experiment matters
    mean      y = y0 * (1 - mean_i g(T_i))      linear pooling

-- because that distinguishes "the batch is m independent probes" from "the batch is one probe plus
noise", which is exactly what decides whether extra experiments (or extra BO iterations) can pay.
"""
import argparse
import glob
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split

NO_INFORMATION = 1.0 / 3.0


def bump(T, centre, width):
  """Per-experiment informativeness: a Gaussian in temperature, the shape the saturating folded
  fraction implies once averaged over the T_melting prior."""
  return np.exp(-0.5 * ((T - centre) / width) ** 2)


def fit_single(temperature, fraction, loss):
  """m = 1: loss = y0 * (1 - amplitude * bump(T)) * (1 + slope * f). Returns the fitted parameters
  and the fit quality. The fraction enters multiplicatively because it scales the reaction rate
  rather than the information content."""
  def residual(p):
    y0, amplitude, centre, width, slope = p
    model = y0 * (1.0 - amplitude * bump(temperature, centre, width)) * (1.0 + slope * fraction)
    return model - loss

  start = [NO_INFORMATION, 0.8, 0.5 * (temperature.min() + temperature.max()),
           0.1 * (temperature.max() - temperature.min()), 0.0]
  bounds = ([0.05, 0.0, temperature.min(), 0.5, -2.0], [1.0, 1.0, temperature.max(), 60.0, 2.0])
  solution = least_squares(residual, start, bounds=bounds)
  return solution.x


def aggregate(kind, g):
  if kind == "product":
    return np.prod(1.0 - g, axis=1)
  if kind == "best":
    return 1.0 - np.max(g, axis=1)
  if kind == "mean":
    return 1.0 - np.mean(g, axis=1)
  raise ValueError(kind)


def quality(prediction, truth):
  ss_res = float(np.sum((prediction - truth) ** 2))
  ss_tot = float(np.sum((truth - truth.mean()) ** 2))
  return 1.0 - ss_res / ss_tot, float(spearmanr(prediction, truth).statistic)


QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


def quantile_agreement(prediction, truth, quantiles=QUANTILES):
  """Compare the two DISTRIBUTIONS by their quantile functions: q-th quantile of the model against
  the q-th quantile of the objective.

  This is the shape metric that matters for BO. What an optimiser meets is the distribution of
  objective values over the design space -- how much mass sits at the no-information plateau, how
  deep the good tail runs -- and a proxy can reproduce that while mispredicting individual designs
  (or predict pointwise well while getting the tails wrong, which is worse: EI lives in the tail).
  Returns the paired quantiles and the largest absolute gap over the grid."""
  model_q = np.quantile(prediction, quantiles)
  truth_q = np.quantile(truth, quantiles)
  return model_q, truth_q, float(np.max(np.abs(model_q - truth_q)))


def surrogate_ceiling(X, y, seed=0):
  """What a GBDT and a GP extract from the same samples, held out. The analytic forms are judged
  against this rather than against 1.0: no closed form should be expected to beat the surrogates
  that see every coordinate."""
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
  gbdt = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                                       min_samples_leaf=20, random_state=seed).fit(X_train, y_train)
  kernel = (ConstantKernel(np.var(y_train), (1e-6, 1e2)) * RBF(np.full(X.shape[1], 0.3), (1e-2, 1e2))
            + WhiteKernel(1e-4, (1e-8, 1e-1)))
  gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1,
                                random_state=seed).fit(X_train[:512], y_train[:512])
  return quality(gbdt.predict(X_test), y_test), quality(gp.predict(X_test), y_test)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", default="output/landscape")
  arguments = parser.parse_args()

  files = sorted(glob.glob(os.path.join(arguments.directory, "m*.npz")),
                 key=lambda p: int(os.path.basename(p)[1:-4]))
  data = {int(np.load(f)["n_experiments"]): np.load(f) for f in files}
  if 1 not in data:
    print("m = 1 is missing; it is what the per-experiment response is fitted from")

  print(f"{'m':>2s} {'dim':>4s} {'n':>5s} {'loss range':>17s} {'GBDT R2':>8s} {'GP R2':>7s} "
        f"{'GBDT rho':>9s} {'GP rho':>7s}")
  ceilings = {}
  for m, d in data.items():
    X, y = np.asarray(d["scaled"]), np.asarray(d["loss"])
    (gbdt_r2, gbdt_rho), (gp_r2, gp_rho) = surrogate_ceiling(X, y)
    ceilings[m] = (gbdt_r2, gp_r2)
    print(f"{m:2d} {X.shape[1]:4d} {len(y):5d} {y.min():7.4f} .. {y.max():.4f} "
          f"{gbdt_r2:8.3f} {gp_r2:7.3f} {gbdt_rho:9.3f} {gp_rho:7.3f}")

  if 1 not in data:
    return
  single = data[1]
  design = np.asarray(single["design"])           # [fraction, temperature]
  parameters = fit_single(design[:, 1], design[:, 0], np.asarray(single["loss"]))
  y0, amplitude, centre, width, slope = parameters
  print(f"\nm = 1 fit:  loss = {y0:.4f} * (1 - {amplitude:.3f} * exp(-((T - {centre:.1f})/{width:.1f})^2 / 2))"
        f" * (1 + {slope:+.3f} f)")
  predicted = y0 * (1.0 - amplitude * bump(design[:, 1], centre, width)) * (1.0 + slope * design[:, 0])
  r2, rho = quality(predicted, np.asarray(single["loss"]))
  print(f"            R2 {r2:.3f}, Spearman {rho:.3f}   (GBDT ceiling R2 {ceilings[1][0]:.3f})")

  quantile_rows = []
  print(f"\nDoes aggregating that single-experiment response predict the batch?")
  print(f"{'m':>2s} " + " ".join(f"{k:>18s}" for k in ("product", "best", "mean")) + f"{'GBDT ceiling':>14s}")
  for m, d in sorted(data.items()):
    if m == 1:
      continue
    design = np.asarray(d["design"])
    fractions, temperatures = design[:, :m], design[:, m:]
    g = amplitude * bump(temperatures, centre, width)
    scale = y0 * (1.0 + slope * fractions.mean(axis=1))
    row, best_kind, best_gap = [], None, np.inf
    for kind in ("product", "best", "mean"):
      prediction = scale * aggregate(kind, g)
      r2, rho = quality(prediction, np.asarray(d["loss"]))
      _, _, gap = quantile_agreement(prediction, np.asarray(d["loss"]))
      row.append(f"R2 {r2:6.3f} dQ {gap:5.3f}")
      if gap < best_gap:
        best_kind, best_gap = kind, gap
    print(f"{m:2d} " + " ".join(f"{cell:>18s}" for cell in row) + f"{ceilings[m][0]:14.3f}")
    quantile_rows.append((m, best_kind, *quantile_agreement(
        scale * aggregate(best_kind, g), np.asarray(d["loss"]))[:2]))

  print(f"\nQUANTILE COMPARISON -- the q-th quantile of the best analytic form against the "
        f"objective's own")
  print(f"{'m':>2s} {'form':>8s} " + " ".join(f"{q:>13.2f}" for q in QUANTILES))
  for m, kind, model_q, truth_q in quantile_rows:
    print(f"{m:2d} {kind:>8s} " + " ".join(f"{a:6.4f}/{b:6.4f}" for a, b in zip(model_q, truth_q)))
  print("   (model / objective; equal pairs mean the shape of the landscape is reproduced)")


if __name__ == "__main__":
  main()
