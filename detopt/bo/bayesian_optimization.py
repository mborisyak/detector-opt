"""Minimal Bayesian optimisation driver that owns X-normalisation + y-centering.

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

The incumbent for EI is read off the (centered) GP state, so no ``y_best`` is
threaded. Objectives are **minimised** -- append the value to minimise (e.g. the
loss directly); EI targets the largest expected reduction below the best so far.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .gp import fit_gp
from .acquisition import optimise_ei

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
        # Random proposals until the GP has enough points for k-fold CV.
        self.n_init = int(n_init) if n_init is not None else int(self.gp_cfg["n_folds"])
        self._key = jax.random.PRNGKey(int(seed))

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
        objective value(s) to maximise; ``noise`` the per-observation standard
        deviation -- **mandatory** (the GP is heteroscedastic; every observation
        carries its own measured uncertainty).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y = np.atleast_1d(np.asarray(y, dtype=np.float32))
        noise = np.broadcast_to(np.asarray(noise, dtype=np.float32), y.shape).copy()
        self.X = np.vstack([self.X, self.to_unit(X)])
        self.y = np.concatenate([self.y, y])
        self.noise = np.concatenate([self.noise, noise])

    def propose(self):
        """Return the next design to evaluate, in nominal space."""
        self._key, key = jax.random.split(self._key)
        if self.X.shape[0] < self.n_init:
            self.last_info = None
            u = jax.random.uniform(key, (self.d,), minval=0.0, maxval=1.0, dtype=jnp.float32)
            return self.to_nominal(np.asarray(u))

        key_gp, key_ei = jax.random.split(key)
        # Center y (the GP has a zero prior mean); not scaled. The incumbent for
        # EI is read off the GP state's (centered) y, so it is consistent.
        y_mean = float(np.mean(self.y))
        gp_state = fit_gp(
            key_gp,
            jnp.asarray(self.X),
            jnp.asarray(self.y - y_mean),
            jnp.asarray(self.noise),
            **self.gp_cfg,
        )
        # GP + EI live in the normalised unit cube [0, 1] (optimise_ei default).
        x_best, ei = optimise_ei(key_ei, gp_state, **self.ei_cfg)
        hp = gp_state.hyper_parameters
        self.last_info = {
            "ei": float(ei),
            "log_lengthscale_mean": float(jnp.mean(hp.log_lengthscale)),
            "log_amplitude": float(hp.log_amplitude),
            "y_mean": y_mean,
        }
        return self.to_nominal(np.asarray(x_best))
