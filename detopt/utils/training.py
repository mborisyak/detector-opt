"""Training-loop utilities: one-epoch step and convergence checks.

These are pure functions (no closures over mutable state) shared by the design
training procedure in ``scripts/bo.py`` and other training scripts.
"""

import jax.numpy as jnp
import numpy as np

__all__ = [
    "is_plateaued",
    "extrapolated_floor",
    "masked_mean",
    "masked_mean_sem",
]


def masked_mean(values, n_valid):
    """Mean of the first ``n_valid`` entries of ``values`` (rest are padding)."""
    valid = jnp.arange(values.shape[0]) < n_valid
    count = jnp.maximum(jnp.sum(valid), 1.0)
    return jnp.sum(jnp.where(valid, values, 0.0)) / count


def masked_mean_sem(values, n_valid):
    """Masked mean and its standard error over the first ``n_valid`` entries."""
    valid = jnp.arange(values.shape[0]) < n_valid
    count = jnp.maximum(jnp.sum(valid), 1.0)
    mean = jnp.sum(jnp.where(valid, values, 0.0)) / count
    var = jnp.maximum(jnp.sum(jnp.where(valid, values**2, 0.0)) / count - mean**2, 0.0)
    return mean, jnp.sqrt(var / count)


def is_plateaued(loss_history, patience, flatness_tol, loss_precision):
    """True when the training loss is flat over the last ``patience`` epochs.

    Fits a line to the last ``patience`` losses and declares a plateau when the
    loss falls by less than ``flatness_tol * loss_precision`` across the window:
    ``slope`` is per epoch, so the window's drop is ``-slope * patience``, and the
    test is ``-slope * patience < flatness_tol * loss_precision``. Tying the
    tolerance to ``loss_precision`` (the BO target) instead of the loss level makes
    "flat" mean "no longer improving at a scale that matters" -- ``flatness_tol`` is
    then a dimensionless fraction of the precision per window. A flat or rising
    trend both count as settled (a rising val loss is overfitting, which the
    downstream gap check turns into a data addition). ``loss_history`` should cover
    the current round only; returns ``False`` until ``patience`` epochs exist, so
    the mandatory warmup is never short-circuited.
    """
    history = np.asarray(loss_history, dtype=np.float64)
    if history.shape[0] < patience:
        return False
    window = history[-patience:]
    slope = float(np.polyfit(np.arange(patience), window, 1)[0])
    return -slope * patience < flatness_tol * loss_precision


def extrapolated_floor(loss_history, alpha=0.5):
    """Asymptote ``L*`` of the power law ``L(t) = L* + C * t**(-alpha)`` fit to the
    loss curve (``t = 1..n`` over the supplied round).

    SGD-style training decays sub-linearly toward a floor (``alpha = 1/2`` is the
    canonical stochastic rate), so the bare slope keeps shrinking and never hits
    zero -- a plateau/slope test always stops while still ``Theta(C t**-alpha)``
    above the floor. Extrapolating ``L*`` exposes that remaining gap directly.

    With ``alpha`` fixed the model is linear in ``(L*, C)``, so this is a weighted
    least-squares fit with features ``[1, t**-alpha]``. Weights decay into the past
    with half-life ``n``: the most recent epoch has weight ``1``, the one before
    ``rho``, then ``rho**2`` ... where ``rho = 0.5 ** (1/n)`` (~0.99 for n~69), so
    the tail (where the asymptote is informative) dominates the fit. Returns the
    last loss when there are too few points (<3) to fit.
    """
    y = np.asarray(loss_history, dtype=np.float64)
    n = y.shape[0]
    if n < 3:
        return float(y[-1])
    t = np.arange(1, n + 1, dtype=np.float64)
    x = t ** (-alpha)
    rho = 0.5 ** (1.0 / n)
    w = rho ** (n - t)  # most recent (t = n) -> 1, decaying into the past
    Sw = w.sum()
    Sx = (w * x).sum()
    Sy = (w * y).sum()
    Sxx = (w * x * x).sum()
    Sxy = (w * x * y).sum()
    denom = Sw * Sxx - Sx * Sx
    if abs(denom) < 1e-12:
        return float(y[-1])
    return float((Sy * Sxx - Sx * Sxy) / denom)  # intercept = L*
