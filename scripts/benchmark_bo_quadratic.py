"""Does the BO driver actually work? Validate it on analytic ||A x + b||^2.

Each problem is a convex quadratic whose global minimiser ``x*`` and minimum
value are known in closed form, so we can measure *true* simple regret rather
than a best-so-far proxy. The minimiser is planted strictly inside the box:

    f(x) = ||A x + b||^2,   A: (m, d),   b = -A x* + n,   n  in  left-null(A).

Because ``n`` is orthogonal to ``range(A)`` we have ``A^T n = 0``, so the
normal equations give ``A^T A x = A^T A x*`` -> the minimiser is exactly ``x*``
for any conditioning of ``A``, and ``f(x*) = ||n||^2``. That lets us hold the
minimiser fixed inside the box while independently dialling:

  * the singular spectrum of ``A`` (well-conditioned / tiny singular values /
    strong column correlations), and
  * the irreducible residual ``||n||`` (a "substantial b", |b| ~ 1).

We run the real :class:`detopt.bo.BayesianOptimizer` (config/bo.yaml gp+ei
settings) against a uniform random-search baseline and report, per problem,
the simple regret ``best_f - f_min`` and the distance ``||x_best - x*||``.

    python scripts/benchmark_bo_quadratic.py [--iters 60] [--seeds 3]

Runs on CPU (tiny GPs; avoids contending with any GPU training job).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detopt.bo import BayesianOptimizer

# The gp/ei settings from config/bo.yaml -- the configuration we actually ship.
_GP_CFG = dict(
    n_folds=5,
    n_restarts=5,
    n_steps=40,
    log_lengthscale_prior_bounds=(-6.0, 1.5),
    log_amplitude_prior_bounds=(-6.0, 1.5),
)
_EI_CFG = dict(n_restarts=32, n_steps=100)

# Analytic objectives have no measurement noise; give the GP a small jitter floor.
_NOISE = 1e-3


def _random_orthogonal(rng, n):
    """A Haar-random n x n orthogonal matrix (QR of a Gaussian)."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))  # fix QR sign ambiguity -> Haar measure


def make_problem(rng, singular_values, residual_norm, *, m, bound):
    """A planted-minimiser quadratic with the given singular spectrum.

    ``singular_values`` has length ``d``; ``A`` is ``(m, d)`` with ``m >= d`` so
    a left-null-space exists to host the residual. Returns ``A, b, x_star,
    f_min``.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    d = s.shape[0]
    U = _random_orthogonal(rng, m)
    V = _random_orthogonal(rng, d)
    A = (U[:, :d] * s) @ V.T  # = U_d diag(s) V^T, shape (m, d)

    # Minimiser planted well inside the box so both BO and random search can reach it.
    x_star = rng.uniform(-0.6 * bound, 0.6 * bound, size=d)

    # Residual n in the left-null-space of A = span(U[:, d:]); orthogonal to range(A).
    if m > d and residual_norm > 0.0:
        coeff = rng.standard_normal(m - d)
        n = U[:, d:] @ coeff
        n *= residual_norm / np.linalg.norm(n)
    else:
        n = np.zeros(m)
    b = -A @ x_star + n
    f_min = float(residual_norm**2)
    return A, b, x_star, f_min


def objective(A, b, x):
    r = A @ np.asarray(x, dtype=np.float64) + b
    return float(r @ r)


def run_bo(A, b, bounds, *, iters, seed):
    """Sequential BO; returns the best-so-far regret curve and the best x."""
    bo = BayesianOptimizer(bounds, gp=_GP_CFG, ei=_EI_CFG, seed=seed)
    best_f, best_x, curve = np.inf, None, []
    for _ in range(iters):
        x = bo.propose()
        f = objective(A, b, x)
        bo.append(x, f, noise=_NOISE)
        if f < best_f:
            best_f, best_x = f, np.asarray(x, dtype=np.float64)
        curve.append(best_f)
    return np.asarray(curve), best_x


def run_random(A, b, bounds, *, iters, seed):
    rng = np.random.default_rng(seed)
    low, high = bounds[:, 0], bounds[:, 1]
    best_f, best_x, curve = np.inf, None, []
    for _ in range(iters):
        x = rng.uniform(low, high)
        f = objective(A, b, x)
        if f < best_f:
            best_f, best_x = f, x
        curve.append(best_f)
    return np.asarray(curve), best_x


# (name, singular_values (len d), residual_norm). m and bound are shared below.
PROBLEMS = [
    ("well_conditioned", [1.0, 0.9, 1.1, 0.8], 0.0),
    ("tiny_singular_vals + |b|~1", [1.0, 1.0, 0.05, 0.02], 1.0),
    ("strong_correlations", [2.0, 1.0, 0.10, 0.03], 0.5),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--bound", type=float, default=3.0)
    parser.add_argument("--m", type=int, default=6, help="rows of A (m >= d for a left-null-space)")
    parser.add_argument("--output", default="output/bo_quadratic")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(5 * len(PROBLEMS), 4), squeeze=False)

    print(f"BO vs random search on ||Ax+b||^2  (iters={args.iters}, seeds={args.seeds}, m={args.m}, box=+-{args.bound})\n")
    for ax, (name, s, residual) in zip(axes[0], PROBLEMS):
        d = len(s)
        bounds = np.stack([np.full(d, -args.bound), np.full(d, args.bound)], axis=1).astype(np.float32)
        cond = max(s) / min(s)

        bo_curves, rand_curves = [], []
        bo_regret, rand_regret, bo_dist = [], [], []
        f_min = None
        for seed in range(args.seeds):
            # Same problem instance for BO and random within a seed; varies across seeds.
            rng = np.random.default_rng(1000 + seed)
            A, b, x_star, f_min = make_problem(rng, s, residual, m=args.m, bound=args.bound)

            bo_c, bo_x = run_bo(A, b, bounds, iters=args.iters, seed=seed)
            rand_c, _ = run_random(A, b, bounds, iters=args.iters, seed=seed)
            bo_curves.append(bo_c - f_min)
            rand_curves.append(rand_c - f_min)
            bo_regret.append(bo_c[-1] - f_min)
            rand_regret.append(rand_c[-1] - f_min)
            bo_dist.append(np.linalg.norm(bo_x - x_star))

        bo_curves = np.stack(bo_curves)
        rand_curves = np.stack(rand_curves)
        bo_med = np.median(bo_curves, axis=0)
        rand_med = np.median(rand_curves, axis=0)

        it = np.arange(1, args.iters + 1)
        ax.plot(it, bo_med, label="BO", color="C0")
        ax.fill_between(it, bo_curves.min(0), bo_curves.max(0), alpha=0.2, color="C0")
        ax.plot(it, rand_med, label="random", color="C1")
        ax.fill_between(it, rand_curves.min(0), rand_curves.max(0), alpha=0.2, color="C1")
        ax.set_yscale("log")
        ax.set_title(f"{name}\ncond(A)={cond:.0f}, f_min={f_min:.3g}")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("simple regret  (best f - f_min)")
        ax.legend()

        print(f"== {name} ==", flush=True)
        print(f"   cond(A)={cond:.1f}  |b|={residual:.2f}  f_min={f_min:.4g}", flush=True)
        print(
            f"   final regret  BO={np.median(bo_regret):.4g}  random={np.median(rand_regret):.4g}"
            f"  speedup={np.median(rand_regret)/max(np.median(bo_regret), 1e-12):.1f}x",
            flush=True,
        )
        print(f"   ||x_best - x*||  BO median={np.median(bo_dist):.4g}\n", flush=True)

    fig.tight_layout()
    path = os.path.join(args.output, "regret.png")
    fig.savefig(path, dpi=120)
    print(f"Regret curves -> {path}")


if __name__ == "__main__":
    main()
