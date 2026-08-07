"""Bayesian optimisation driver (scikit-learn GP) that owns X-normalisation + y-centering.

The public API works entirely in the *nominal* (physical) design space:
:meth:`append` takes nominal designs + objective values and :meth:`propose`
returns one nominal design. Internally:

* **X** is mapped to the unit cube ``[0, 1]^d`` (per-dimension min/max from
  ``bounds``); the GP and the Expected-Improvement search run there.
* **y** is **centered** (``y - mean(y)``) before fitting -- the GP has a zero
  prior mean, so an un-centered constant offset would force a long lengthscale /
  large amplitude. y is *not scaled* (the detector returns losses in a reasonable
  range); only the mean is removed. Centering is a constant shift, so EI's
  argmax -- and hence the proposed design -- is unchanged; it only makes the
  fitted lengthscale/amplitude meaningful.

Objectives are **minimised** -- append the value to minimise (e.g. the loss
directly); EI targets the largest expected reduction below the best so far.

The surrogate
-------------
The GP is scikit-learn's :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
with an ARD-RBF kernel (``ConstantKernel * RBF``, one lengthscale per dimension),
which is the same kernel family as :mod:`detopt.bo.gp`. Two things differ from that
JAX implementation, deliberately:

* **Objective.** sklearn tunes the hyperparameters by maximising the log MARGINAL
  likelihood; :mod:`detopt.bo.gp` minimises a predictive k-fold CV NLL. Different
  criteria pick different hyperparameters, so the two are not interchangeable
  fits -- BO trajectories will differ.
* **No JIT.** Nothing here is traced or compiled, which is the point: the jitted
  fit compiled once per distinct dataset size and retained every executable, so a
  run leaked tens of MB per BO iteration. Plain NumPy has no such failure mode and
  ``n`` may vary freely, so no padding, capacity or masking is needed.

:mod:`detopt.bo.gp` and :mod:`detopt.bo.acquisition` are untouched and still used by
``scripts/benchmark_gp.py`` and the GP tests; this driver simply no longer calls them.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

__all__ = ["BayesianOptimizer"]


class BayesianOptimizer:
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        bounds = config.pop("bounds")
        return cls(bounds, **config)

    def __init__(self, bounds, *, gp, ei, n_init=None, seed=0):
        bounds = np.asarray(bounds, dtype=np.float32)  # (d, 2): [low, high]
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (d, 2): per-dimension [low, high]")
        self.low = bounds[:, 0]
        self.high = bounds[:, 1]
        if np.any(self.high <= self.low):
            raise ValueError("each bound must satisfy high > low")
        self.d = int(bounds.shape[0])

        self.gp_cfg = dict(gp)
        self.ei_cfg = dict(ei)
        # Random proposals until there are enough points to fit anything sensible.
        self.n_init = int(n_init) if n_init is not None else int(self.gp_cfg["n_folds"])
        self._rng = np.random.default_rng(int(seed))
        self._seed = int(seed)

        # The config states the prior box in LOG space (that is how detopt.bo.gp
        # parameterises it); sklearn wants the natural-scale bounds. The kernel is
        # amplitude^2 * exp(-|dx/l|^2 / 2), so ConstantKernel carries amplitude^2 --
        # hence exp(2 * log_amplitude) -- and RBF carries l = exp(log_lengthscale).
        # `n_folds` and `n_steps` from the config do not apply to this backend (there
        # is no CV split and L-BFGS-B runs to its own convergence); `n_folds` is still
        # read above as the historical default for `n_init`.
        ls_low, ls_high = self.gp_cfg["log_lengthscale_prior_bounds"]
        amp_low, amp_high = self.gp_cfg["log_amplitude_prior_bounds"]
        self._length_scale_bounds = (float(np.exp(ls_low)), float(np.exp(ls_high)))
        self._amplitude_bounds = (float(np.exp(2.0 * amp_low)), float(np.exp(2.0 * amp_high)))
        self._length_scale_init = float(np.exp(0.5 * (ls_low + ls_high)))
        self._amplitude_init = float(np.exp(amp_low + amp_high))  # exp(2 * midpoint)
        self.n_gp_restarts = int(self.gp_cfg.get("n_restarts", 5))

        # X stored in the normalised unit cube [0, 1]; y / noise in nominal units.
        self.X = np.empty((0, self.d), dtype=np.float32)
        self.y = np.empty((0,), dtype=np.float32)
        self.noise = np.empty((0,), dtype=np.float32)
        # Diagnostics from the most recent GP-driven proposal (None during init).
        self.last_info = None

    # ------------------------------------------------------------------ #
    # Nominal <-> normalised unit cube [0, 1] conversion (per dimension)
    # ------------------------------------------------------------------ #
    def to_unit(self, X):
        """Nominal design(s) -> normalised unit cube ``[0, 1]``."""
        X = np.asarray(X, dtype=np.float32)
        return (X - self.low) / (self.high - self.low)

    def to_nominal(self, U):
        """Normalised unit cube ``[0, 1]`` -> nominal design(s)."""
        U = np.asarray(U, dtype=np.float32)
        return self.low + U * (self.high - self.low)

    # ------------------------------------------------------------------ #
    def append(self, X, y, noise):
        """Record observation(s) in nominal space.

        ``X`` is ``(d,)`` or ``(n, d)`` nominal designs; ``y`` the matching
        objective value(s) to minimise; ``noise`` the per-observation standard
        deviation -- **mandatory** (the GP is heteroscedastic; every observation
        carries its own measured uncertainty).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y = np.atleast_1d(np.asarray(y, dtype=np.float32))
        noise = np.broadcast_to(np.asarray(noise, dtype=np.float32), y.shape).copy()
        self.X = np.vstack([self.X, self.to_unit(X)])
        self.y = np.concatenate([self.y, y])
        self.noise = np.concatenate([self.noise, noise])

    # ------------------------------------------------------------------ #
    # Surrogate + acquisition
    # ------------------------------------------------------------------ #
    def _fit(self, y_centered):
        """Fit the ARD-RBF GP on the observations so far.

        ``alpha`` is the per-observation noise VARIANCE placed on the kernel diagonal,
        so the caller's standard deviations are squared here -- that is what carries the
        heteroscedastic per-design SEM into the fit. ``normalize_y`` stays off because
        this class already centers y.
        """
        kernel = ConstantKernel(self._amplitude_init, self._amplitude_bounds) * RBF(
            np.full(self.d, self._length_scale_init), self._length_scale_bounds
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=np.maximum(self.noise.astype(np.float64) ** 2, 1e-12),
            n_restarts_optimizer=self.n_gp_restarts,
            normalize_y=False,
            random_state=self._seed,
        )
        return model.fit(self.X.astype(np.float64), y_centered.astype(np.float64))

    @staticmethod
    def _expected_improvement(model, X, y_best):
        """Analytic EI at ``X`` ``(m, d)`` for MINIMISATION -- improvement is ``max(y_best - y, 0)``."""
        mean, std = model.predict(np.atleast_2d(X), return_std=True)
        std = np.maximum(std, 1e-12)
        z = (y_best - mean) / std
        return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)

    def _optimise_ei(self, model, y_best):
        """Maximise EI over the unit cube: coarse random sweep, then local polish.

        sklearn's ``predict`` exposes no gradient with respect to X (the JAX path
        differentiated straight through the GP), so the search is derivative-free at the
        top level: one BATCHED evaluation over many candidates finds the basins cheaply,
        and L-BFGS-B -- with its own finite differences -- refines the best few. The
        sweep is the important half; the polish only sharpens an already-chosen basin.
        """
        n_restarts = int(self.ei_cfg.get("n_restarts", 32))
        n_steps = int(self.ei_cfg.get("n_steps", 100))
        n_candidates = max(4096, 512 * self.d)

        candidates = self._rng.random((n_candidates, self.d))
        ei = self._expected_improvement(model, candidates, y_best)
        order = np.argsort(-ei)
        best_x, best_ei = candidates[order[0]], float(ei[order[0]])

        # Polishing every candidate would cost n_restarts x n_steps single-point
        # predicts; the sweep has already ranked them, so refine only the leaders.
        for start in candidates[order[:max(1, min(n_restarts, 8))]]:
            result = minimize(
                lambda x: -float(self._expected_improvement(model, x[None, :], y_best)[0]),
                start,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * self.d,
                options={"maxiter": n_steps},
            )
            if -float(result.fun) > best_ei:
                best_ei, best_x = -float(result.fun), np.clip(result.x, 0.0, 1.0)
        return best_x, best_ei

    # ------------------------------------------------------------------ #
    def propose(self):
        """Return the next design to evaluate, in nominal space."""
        if self.X.shape[0] < self.n_init:
            self.last_info = None
            return self.to_nominal(self._rng.random(self.d))

        # Center y (the GP has a zero prior mean); not scaled. The incumbent for EI is
        # the best CENTERED observation, so it is consistent with what the GP was fit on.
        y_mean = float(np.mean(self.y))
        y_centered = self.y - y_mean
        model = self._fit(y_centered)
        x_best, ei = self._optimise_ei(model, float(np.min(y_centered)))

        amplitude2, length_scale = model.kernel_.k1.constant_value, model.kernel_.k2.length_scale
        self.last_info = {
            "ei": float(ei),
            "log_lengthscale_mean": float(np.mean(np.log(np.atleast_1d(length_scale)))),
            "log_amplitude": float(0.5 * np.log(amplitude2)),
            "y_mean": y_mean,
        }
        return self.to_nominal(x_best)
