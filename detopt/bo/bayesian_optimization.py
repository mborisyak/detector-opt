"""Bayesian optimisation driver (scikit-learn GP) that owns X-normalisation + y-centering.

A design lives in one of TWO spaces:

* **nominal** -- the physical design (mM, degrees C, metres, tesla). Read from the
  config as a starting design and written to ``results.json``; never searched.
* **scaled** -- ``[0, 1]^d``, each coordinate affinely on its own design range. This
  is what every optimiser searches and what every network is conditioned on, and it
  is the ONLY space this class ever sees: :meth:`append` takes scaled designs and
  :meth:`propose` returns one. There is no transform here at all -- the detector owns
  nominal <-> scaled (``to_scaled`` / ``to_nominal``), so the GP is fitted, and EI
  maximised, directly on what the caller passes.

Internally:

* **X** is stored exactly as given, in ``[0, 1]^d``. Because the map onto that box is
  affine per coordinate, uniform in it is a UNIFORM NOMINAL DESIGN -- which is what
  the initial points and the EI candidate sweep must sample -- and mixed units (a
  volume fraction against a temperature in C) are already commensurate.
Objectives are **minimised** -- append the value to minimise (e.g. the loss
directly); EI targets the largest expected reduction below the best so far.

The surrogate
-------------
The GP is scikit-learn's :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
with an ARD-RBF kernel (``ConstantKernel * RBF``, one lengthscale per dimension),
which is the same kernel family as :mod:`detopt.bo.jax_gp`. Two things differ from that
JAX implementation, deliberately:

* **Objective.** sklearn tunes the hyperparameters by maximising the log MARGINAL
  likelihood; :mod:`detopt.bo.jax_gp` minimises a predictive k-fold CV NLL. Different
  criteria pick different hyperparameters, so the two are not interchangeable
  fits -- BO trajectories will differ.
* **No JIT.** Nothing here is traced or compiled, which is the point: the jitted
  fit compiled once per distinct dataset size and retained every executable, so a
  run leaked tens of MB per BO iteration. Plain NumPy has no such failure mode and
  ``n`` may vary freely, so no padding, capacity or masking is needed.

:mod:`detopt.bo.jax_gp` is retained as a reference implementation, used only by
``scripts/benchmark_gp.py`` and the GP tests; this driver never calls it.
"""

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor

from .kernels import ARDRBF

__all__ = ["BayesianOptimizer"]



class BayesianOptimizer:
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        return cls(config.pop("d"), **config)

    def __init__(self, d, *, gp, ei, kernel=None, n_init=None, seed=0):
        """``d`` is the design dimension. There are no bounds to pass: the caller works in the scaled
        box ``[0, 1]^d`` already, so the box IS the search space.

        ``kernel`` is the GP prior, built by :func:`detopt.bo.kernel_from_config` from the config's
        ``gp.kernel`` entry -- the caller owns it because choosing it needs the DETECTOR (the design
        dimension, and which of its coordinates are exchangeable), which this class never sees.
        ``None`` falls back to the plain ARD-RBF built from the ``gp`` prior bounds below."""
        self.d = int(d)
        if self.d < 1:
            raise ValueError(f"d must be a positive design dimension, got {d!r}")

        self.gp_cfg = dict(gp)
        self.ei_cfg = dict(ei)
        # Random proposals until there are enough points to fit anything sensible.
        self.n_init = int(n_init) if n_init is not None else int(self.gp_cfg["n_folds"])
        self._rng = np.random.default_rng(int(seed))
        self._seed = int(seed)
        # Initial design: a SCRAMBLED SOBOL sequence, not independent uniform draws. On a landscape
        # whose informative region is a small fraction of the box, i.i.d. points clump and leave
        # holes. Scrambling keeps it randomised per seed (so seeds remain independent replicates)
        # while guaranteeing the coverage.
        self._sobol = qmc.Sobol(self.d, scramble=True, seed=int(seed))

        # The config states the prior box in LOG space (that is how detopt.bo.jax_gp
        # parameterises it); sklearn wants the natural-scale bounds. The kernel is
        # amplitude^2 * exp(-|dx/l|^2 / 2), so ConstantKernel carries amplitude^2 --
        # hence exp(2 * log_amplitude) -- and RBF carries l = exp(log_lengthscale).
        # `n_folds` and `n_steps` from the config do not apply to this backend (there
        # is no CV split and L-BFGS-B runs to its own convergence); `n_folds` is still
        # read above as the historical default for `n_init`.
        #
        # The lengthscale bounds are read against the SCALED box [0, 1], so a lengthscale of 1
        # already spans the whole domain and anything much larger switches a dimension off. The
        # ceiling therefore has to sit well above 1 for ARD to be able to declare a design
        # dimension irrelevant -- see config/bo.yaml.
        ls_low, ls_high = self.gp_cfg["log_lengthscale_prior_bounds"]
        amp_low, amp_high = self.gp_cfg["log_amplitude_prior_bounds"]
        self._length_scale_bounds = (float(np.exp(ls_low)), float(np.exp(ls_high)))
        self._amplitude_bounds = (float(np.exp(2.0 * amp_low)), float(np.exp(2.0 * amp_high)))
        self._length_scale_init = float(np.exp(0.5 * (ls_low + ls_high)))
        self._amplitude_init = float(np.exp(amp_low + amp_high))  # exp(2 * midpoint)
        self.n_gp_restarts = int(self.gp_cfg.get("n_restarts", 5))

        # The GP prior. Supplied by the caller when the design has structure worth modelling (see
        # detopt.bo.kernels); otherwise the historical ARD-RBF, one lengthscale per coordinate.
        self.kernel = kernel if kernel is not None else ARDRBF(
            d=self.d,
            constant_value=self._amplitude_init,
            constant_value_bounds=self._amplitude_bounds,
            length_scale=np.full(self.d, self._length_scale_init),
            length_scale_bounds=self._length_scale_bounds,
        )

        # X in the SCALED box [0, 1]; y / noise in the objective's own units.
        # A kernel built for a different dimension fails only on the FIRST surrogate fit -- after
        # n_init designs have already been scored -- with an opaque IndexError from inside the
        # kernel. `kernel_from_config` sizes itself from the detector's full design, so any reduced
        # search (a fixed or shared coordinate) must build its own; say so here rather than there.
        # `k_and_grad_x` and `grad_diag` are hard requirements of the EI gradient, and `d` is what
        # makes the kernel's coordinate layout checkable -- so demand all three rather than letting
        # an object without them through to fail later, deeper, and less legibly.
        if kernel is not None:
            missing = [n for n in ("d", "k_and_grad_x", "grad_diag") if not hasattr(kernel, n)]
            if len(missing) > 0:
                raise ValueError(f"kernel {type(kernel).__name__} is missing {missing}; the analytic EI "
                                 f"gradient needs all of them")
            if int(kernel.d) != self.d:
                raise ValueError(f"kernel is {int(kernel.d)}-dimensional but the search is {self.d}-"
                                 f"dimensional; build the kernel at the SEARCHED dimension")
        self.X = np.empty((0, self.d), dtype=np.float32)
        self.y = np.empty((0,), dtype=np.float32)
        self.noise = np.empty((0,), dtype=np.float32)
        # Diagnostics from the most recent GP-driven proposal (None during init).
        self.last_info = None

    # ------------------------------------------------------------------ #
    def append(self, X, y, noise):
        """Record observation(s) in the SCALED box ``[0, 1]``.

        ``X`` is ``(d,)`` or ``(n, d)`` scaled designs; ``y`` the matching
        objective value(s) to minimise; ``noise`` the per-observation standard
        deviation -- **mandatory** (the GP is heteroscedastic; every observation
        carries its own measured uncertainty).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y = np.atleast_1d(np.asarray(y, dtype=np.float32))
        noise = np.broadcast_to(np.asarray(noise, dtype=np.float32), y.shape).copy()
        self.X = np.vstack([self.X, X])
        self.y = np.concatenate([self.y, y])
        self.noise = np.concatenate([self.noise, noise])

    # ------------------------------------------------------------------ #
    # Surrogate + acquisition
    # ------------------------------------------------------------------ #
    def _fit(self, X, y_centered, noise):
        """Fit the ARD-RBF GP on the given observations.

        ``alpha`` is the per-observation noise VARIANCE placed on the kernel diagonal,
        so the caller's standard deviations are squared here -- that is what carries the
        heteroscedastic per-design SEM into the fit. ``normalize_y`` stays off because
        this class already centers y.
        """
        model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=np.maximum(noise.astype(np.float64) ** 2, 1e-12),
            n_restarts_optimizer=self.n_gp_restarts,
            normalize_y=False,
            random_state=self._seed,
        )
        return model.fit(X.astype(np.float64), y_centered.astype(np.float64))

    @staticmethod
    def _expected_improvement(model, X, y_best):
        """Analytic EI at ``X`` ``(m, d)`` for MINIMISATION -- improvement is ``max(y_best - y, 0)``."""
        mean, std = model.predict(np.atleast_2d(X), return_std=True)
        std = np.maximum(std, 1e-12)
        z = (y_best - mean) / std
        return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _ei_and_grad(model, x, y_best):
        """EI and its EXACT gradient at a single point ``x`` ``(d,)``.

        With ``u = y_best - mu`` and ``z = u / sigma`` the outer derivatives of
        ``EI = u*Phi(z) + sigma*phi(z)`` collapse to ``dEI/du = Phi(z)`` and
        ``dEI/dsigma = phi(z)`` (the ``phi'(z) = -z phi(z)`` terms cancel), so

            grad EI = -Phi(z) * grad mu + phi(z) * grad sigma.

        For the ARD-RBF kernel ``k_i = c exp(-|x - X_i|^2_l / 2)`` the Jacobian is
        ``J_ij = dk_i/dx_j = k_i (X_ij - x_j) / l_j^2``, giving ``grad mu = J^T alpha``
        and, from ``sigma^2 = c - k^T K^-1 k``, ``grad sigma = -J^T v / sigma`` with
        ``v = K^-1 k``. Everything comes off the fitted model, so no finite differences
        are needed -- which matters because EI underflows to ~1e-15 on a flat surrogate,
        where differencing is pure cancellation noise.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        k, jac = model.kernel_.k_and_grad_x(model.X_train_, x)  # (n,), (n, d) = dk_i/dx_j
        # k(x, x), NOT the amplitude: a group-averaged kernel's diagonal is not constant (it
        # measures how far x is from its own permutations), so the prior variance depends on x.
        prior_variance = float(model.kernel_.diag(x[None, :])[0])
        v = cho_solve((model.L_, True), k)  # K^-1 k

        alpha = np.ravel(model.alpha_)
        mean = float(k @ alpha)
        var = max(prior_variance - float(k @ v), 1e-12)
        sigma = np.sqrt(var)

        u = y_best - mean
        z = u / sigma
        cdf, pdf = norm.cdf(z), norm.pdf(z)
        ei = u * cdf + sigma * pdf

        grad_mean = jac.T @ alpha
        # sigma^2 = diag(x) - k^T K^-1 k, so BOTH terms move with x. The second alone is the
        # stationary-kernel case; for a group-averaged kernel diag(x) depends on how far x sits from
        # its own permutations, and dropping its derivative can flip the gradient's direction.
        grad_prior = (model.kernel_.grad_diag(x) if hasattr(model.kernel_, "grad_diag")
                      else np.zeros_like(grad_mean))
        grad_sigma = (grad_prior - 2.0 * (jac.T @ v)) / (2.0 * sigma)
        return float(ei), -cdf * grad_mean + pdf * grad_sigma

    def _optimise_ei(self, model, y_best, low, high):
        """Maximise EI over the box ``[low, high]`` (the scaled cube):
        coarse random sweep, then local polish.

        One BATCHED evaluation over many candidates finds the basins cheaply, then
        L-BFGS-B refines the ``n_restarts`` best of them using the EXACT gradient from
        :meth:`_ei_and_grad` -- so every restart in the config is honoured, and the
        polish stays trustworthy where EI is small (finite differences there are
        cancellation noise).
        """
        n_restarts = int(self.ei_cfg.get("n_restarts", 32))
        n_steps = int(self.ei_cfg.get("n_steps", 100))
        n_candidates = max(4096, 512 * self.d)

        # Uniform in the box, which IS the cube, so this is a uniform DESIGN.
        candidates = low + self._rng.random((n_candidates, self.d)) * (high - low)
        ei = self._expected_improvement(model, candidates, y_best)
        order = np.argsort(-ei)
        best_x, best_ei = candidates[order[0]], float(ei[order[0]])

        def negative_ei(x):
            value, grad = self._ei_and_grad(model, x, y_best)
            return -value, -grad

        for start in candidates[order[:max(1, n_restarts)]]:
            result = minimize(
                negative_ei,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(low, high)),
                options={"maxiter": n_steps},
            )
            if -float(result.fun) > best_ei:
                best_ei, best_x = -float(result.fun), np.clip(result.x, low, high)
        return best_x, best_ei

    # ------------------------------------------------------------------ #
    def propose(self):
        """Return the next design to evaluate, in the SCALED box ``[0, 1]``."""
        # Space-filling initial design, before there is anything worth fitting.
        if self.X.shape[0] < self.n_init:
            self.last_info = None
            return np.clip(self._sobol.random(1)[0], 0.0, 1.0).astype(np.float32)

        # Center y (the GP has a zero prior mean); never scaled. A constant shift leaves EI's argmax
        # -- and hence the proposal -- unchanged; it only makes the fitted lengthscale/amplitude
        # meaningful. The incumbent for EI is the best CENTERED observation, so EI and the surrogate
        # agree about what "improvement" means.
        y_mean = float(np.mean(self.y))
        y_centered = self.y - y_mean
        model = self._fit(self.X, y_centered, self.noise)
        x_best, ei = self._optimise_ei(model, float(np.min(y_centered)),
                                       np.zeros(self.d), np.ones(self.d))

        amplitude2, length_scale = model.kernel_.constant_value, model.kernel_.length_scale
        self.last_info = {
            "ei": float(ei),
            "log_lengthscale_mean": float(np.mean(np.log(np.atleast_1d(length_scale)))),
            "log_amplitude": float(0.5 * np.log(amplitude2)),
            "y_mean": y_mean,
        }
        return np.asarray(x_best, dtype=np.float32)
