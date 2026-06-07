"""Bayesian optimisation utilities for design search.

Provides:
  * :mod:`detopt.bo.gp` — an ARD-RBF Gaussian process with 5-fold-CV
    hyperparameter fitting.
  * :mod:`detopt.bo.acquisition` — analytic Expected Improvement and a
    multi-restart projected-gradient optimiser over a bounded hyperbox.
"""

from . import gp
from . import acquisition
from .gp import (
    GPHParams,
    GPState,
    fit_gp,
    refit_gp,
    gp_factorize,
    gp_predict,
    gp_posterior,
    rbf_kernel,
)
from .acquisition import expected_improvement, optimise_ei
from .bayesian_optimization import BayesianOptimizer

__all__ = [
    "gp",
    "acquisition",
    "BayesianOptimizer",
    "GPHParams",
    "GPState",
    "fit_gp",
    "refit_gp",
    "gp_factorize",
    "gp_predict",
    "gp_posterior",
    "rbf_kernel",
    "expected_improvement",
    "optimise_ei",
]
