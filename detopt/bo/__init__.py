"""Bayesian optimisation utilities for design search.

Provides:
  * :class:`~detopt.bo.BayesianOptimizer` — the driver everything actually uses.
    Its surrogate is a scikit-learn ARD-RBF Gaussian process fitted by marginal
    likelihood, with an analytic Expected-Improvement acquisition maximised by a
    random sweep plus exact-gradient L-BFGS-B.
  * :mod:`detopt.bo.jax_gp` — a JAX ARD-RBF GP with predictive k-fold-CV
    hyperparameter fitting and its own EI optimiser. **Reference only, for tests
    and benchmarks**: it is an independent, differentiable second opinion (used to
    check the driver's analytic EI gradient against autodiff) and the subject of
    ``scripts/benchmark_gp.py``. It is jitted, so it compiles once per distinct
    dataset size and retains every executable — do not use it in a real run.
"""

from . import jax_gp
from .jax_gp import (
    GPHParams,
    GPState,
    expected_improvement,
    fit_gp,
    gp_factorize,
    gp_posterior,
    gp_predict,
    optimise_ei,
    rbf_kernel,
    refit_gp,
)
from .bayesian_optimization import BayesianOptimizer

__all__ = [
    "BayesianOptimizer",
    # Reference JAX GP -- tests and benchmarks only (see the module docstring).
    "jax_gp",
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
