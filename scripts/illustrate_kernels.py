#!/usr/bin/env python3
"""What each GP kernel BELIEVES about a symmetric function, on a design small enough to draw.

The design is a batch of TWO interchangeable experiments, ``(x_1, x_2)``, and the objective is

    f(x) = x_1^2 + x_2^2 + noise

which is symmetric: swapping the two experiments cannot change it, exactly as the enzyme batch's
loss cannot depend on the order the experiments are listed in. Every training point is therefore a
statement about BOTH ``(x_1, x_2)`` and ``(x_2, x_1)``, and the question each kernel answers
differently is whether the surrogate knows that.

Rows are the posterior MEAN, the PRIOR SD (before any data), and the posterior SD, on identical
data with identical hyperparameter bounds. Read them against the dashed diagonal ``x_1 = x_2``, the
mirror line:

  ard-rbf                   no symmetry. The mean is NOT mirror-symmetric -- the GP has to learn the
                            other half of the plane from data it was never given, and its SD stays
                            wide wherever no point happens to sit.
  permutation-invariant     the group average. Exactly symmetric, smooth across the diagonal, but
                            its PRIOR variance peaks on the diagonal -- k(x,x) is up to |G| times
                            larger on the tied stratum -- so EI's exploration term is maximal exactly
                            on the degenerate all-identical batch.
  normalised-invariant      the same average rescaled to a constant diagonal. Same symmetry, same
                            smoothness, no ridge.
  sorting-rbf               invariance by folding. Exactly symmetric and constant-diagonal, but the
                            fold leaves a CREASE on the mirror line -- the mean is continuous there
                            and its gradient is not.

The PRIOR row is the one that isolates the mechanism, and it is worth having because the posterior
SD does NOT: the three invariant kernels all show an elevated posterior SD near the diagonal simply
because the training points were drawn off it (ard-rbf, which folds nothing onto it, sits at 0.917 --
below 1), and sorting's posterior ridge is as large as the group average's
(1.41 against 1.38) even though its diagonal is exactly constant. Measured on the prior, the ridge
belongs to the group average alone (1.222 against 1.000, 1.000, 1.000).

    python scripts/illustrate_kernels.py --output output/kernel_illustration.png
"""

import argparse

import numpy as np
from matplotlib.figure import Figure
from sklearn.gaussian_process import GaussianProcessRegressor

import detopt.bo

# One panel per kernel, in the order they are argued about: no symmetry, then the three that have it.
KERNELS = [
  ("ard-rbf", "ard-rbf\n(no symmetry)", {}),
  ("permutation-invariant-rbf", "permutation-invariant-rbf\n$k(x,y)$ averaged over the group", {"blocks": ((0, 1),)}),
  ("normalised-invariant-rbf", "normalised-invariant-rbf\n$c\\,k(x,y)/\\sqrt{k(x,x)k(y,y)}$", {"blocks": ((0, 1),)}),
  ("sorting-rbf", "sorting-rbf\n$k(\\sigma x, \\sigma y)$, sorted", {"sort_blocks": ((0, 1),), "key": -1}),
]


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--n-points", type=int, default=7, help="training designs drawn from the square")
  parser.add_argument("--noise", type=float, default=0.05, help="std of the observation noise")
  parser.add_argument("--seed", type=int, default=3)
  parser.add_argument("--output", default="output/kernel_illustration.png")
  arguments = parser.parse_args()

  rng = np.random.default_rng(arguments.seed)
  # Deliberately drawn from ONE SIDE of the mirror line, which is what makes the comparison legible:
  # a symmetric kernel gets the other side for free, an asymmetric one does not. This is not a
  # contrivance -- it is the generic case, since a random batch is on one side or the other and only
  # the measure-zero tie locus is on neither.
  X = rng.uniform(-1.0, 1.0, size=(arguments.n_points, 2))
  X = np.where((X[:, [0]] > X[:, [1]]), X, X[:, ::-1])
  y = (X**2).sum(axis=1) + rng.normal(0.0, arguments.noise, size=X.shape[0])

  grid = np.linspace(-1.05, 1.05, 121)
  mesh_x, mesh_y = np.meshgrid(grid, grid)
  query = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

  figure = Figure(figsize=(4.0 * len(KERNELS), 11.2), dpi=130)
  axes = figure.subplots(3, len(KERNELS))
  means, sds, priors = [], [], []
  for name, _, layout in KERNELS:
    kernel = detopt.bo.__kernels__[name](
      d=2, constant_value=1.0, constant_value_bounds=(1e-3, 1e3),
      length_scale=np.full(1 if len(layout) > 0 else 2, 0.5), length_scale_bounds=(1e-2, 1e2), **layout
    )
    # The PRIOR sd of the kernel as configured, before any fit -- the mechanism, isolated from the
    # data. Taken at the shared initial hyperparameters so the four are directly comparable.
    priors.append(np.sqrt(np.ravel(kernel.diag(query))).reshape(mesh_x.shape))
    model = GaussianProcessRegressor(kernel=kernel, alpha=arguments.noise**2, normalize_y=True,
                                     n_restarts_optimizer=4, random_state=0).fit(X, y)
    mean, sd = model.predict(query, return_std=True)
    means.append(mean.reshape(mesh_x.shape))
    sds.append(sd.reshape(mesh_x.shape))
    print(f"{name:28s} fitted {model.kernel_}")

  mean_low, mean_high = min(m.min() for m in means), max(m.max() for m in means)
  sd_high = max(s.max() for s in sds)
  prior_high = max(p.max() for p in priors)
  for column, ((name, title, _), mean, prior, sd) in enumerate(zip(KERNELS, means, priors, sds)):
    for row, (field, high, low, cmap, label) in enumerate(
        [(mean, mean_high, mean_low, "viridis", "posterior mean"),
         (prior, prior_high, 0.0, "cividis", "PRIOR SD (no data)"),
         (sd, sd_high, 0.0, "magma", "posterior SD")]):
      axis = axes[row, column]
      image = axis.pcolormesh(mesh_x, mesh_y, field, vmin=low, vmax=high, cmap=cmap, shading="auto")
      axis.contour(mesh_x, mesh_y, field, levels=10, colors="white", linewidths=0.4, alpha=0.5)
      axis.plot([-1.05, 1.05], [-1.05, 1.05], color="white", ls="--", lw=1.2, alpha=0.9)
      axis.scatter(X[:, 0], X[:, 1], c="white", s=42, edgecolor="black", zorder=3, lw=0.8)
      # The mirror images of the training points: data a symmetric kernel effectively HAS and an
      # asymmetric one does not.
      axis.scatter(X[:, 1], X[:, 0], facecolor="none", s=42, edgecolor="white", zorder=3, lw=1.0, ls=":")
      axis.set_xlim(-1.05, 1.05)
      axis.set_ylim(-1.05, 1.05)
      axis.set_aspect("equal")
      if row == 0:
        axis.set_title(title, fontsize=10)
      if column == 0:
        axis.set_ylabel(f"{label}\n\n$x_2$", fontsize=9)
      axis.set_xlabel("$x_1$", fontsize=9)
      figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
  figure.suptitle("GP posterior for $f(x)=x_1^2+x_2^2+\\varepsilon$, a SYMMETRIC objective on a batch of two "
                  "interchangeable experiments\nfilled = observed designs, dotted = their mirror images "
                  "(never observed), dashed = the mirror line $x_1=x_2$", fontsize=11)
  figure.tight_layout(rect=(0, 0, 1, 0.93))
  figure.savefig(arguments.output)
  print(f"wrote {arguments.output}")

  # The numbers behind the picture: how far each posterior is from being symmetric.
  print("\nsymmetry defect  max |mu(x1,x2) - mu(x2,x1)| over the grid:")
  for (name, _, _), mean in zip(KERNELS, means):
    print(f"  {name:28s} {np.abs(mean - mean.T).max():.3e}")
  print("\nSD on the mirror line vs off it -- PRIOR (the mechanism) then POSTERIOR (data too):")
  for (name, _, _), prior, sd in zip(KERNELS, priors, sds):
    print(f"  {name:28s} prior {np.diag(prior).mean() / prior.mean():.3f}   "
          f"posterior {np.diag(sd).mean() / sd.mean():.3f}")


if __name__ == "__main__":
  main()
