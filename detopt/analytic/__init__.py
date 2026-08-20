"""Semi-analytic estimators of a detector's objective: cheap stand-ins for a trained regressor whose
value is STATISTICS, not fidelity. One neural evaluation of a design costs the better part of an hour,
so a single seed decides nothing; an estimator here is fast enough to answer the same question over
hundreds of seeds.

Every estimator states which of its parts are closed form and which are numerical, and exposes the
diagnostics that say whether the numerical parts have converged.
"""

from .extremes import MechanismInstrument, InstrumentResult, build_detector, sobol_designs

__all__ = ["MechanismInstrument", "InstrumentResult", "build_detector", "sobol_designs"]
