"""Training-loop utilities: one-epoch step and convergence checks.

These are pure functions (no closures over mutable state) shared by the design
training procedure in ``scripts/bo.py`` and other training scripts.
"""

import math

import jax.numpy as jnp
import numpy as np

__all__ = [
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
    """Masked mean and its standard error over the first ``n_valid`` entries.

    The error of the mean is ``std / sqrt(n - 1)``: ``var`` below is the POPULATION variance
    (divided by ``n``), so the unbiased error of the mean divides by ``n - 1`` rather than ``n``.
    At the evaluation sizes used here the two differ by well under a permille, but the estimate
    feeds the GP's observation noise and the convergence test, so it is the defined quantity
    rather than the convenient one.
    """
    valid = jnp.arange(values.shape[0]) < n_valid
    count = jnp.maximum(jnp.sum(valid), 1.0)
    mean = jnp.sum(jnp.where(valid, values, 0.0)) / count
    var = jnp.maximum(jnp.sum(jnp.where(valid, values**2, 0.0)) / count - mean**2, 0.0)
    return mean, jnp.sqrt(var / jnp.maximum(count - 1.0, 1.0))



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


def bayesian_trend(values, sems, prior_sigma, harmonic=True):
    """Posterior over ``(intercept, slope)`` of a line through ``values`` at ``t = 0, 1, ...``.

    Conjugate Bayesian linear regression with KNOWN per-point observation noise -- the trainer already
    measures a SEM for every epoch's loss, so the noise is not estimated from the residuals (which at
    5-15 points is where a naive fit gets its uncertainty badly wrong). Prior on both coefficients is
    ``N(0, prior_sigma^2)``, zero-mean and weakly informative.

    ``harmonic`` down-weights older epochs as ``1/k`` with ``k`` counted from the END (newest weighs
    1, the previous 1/2, and so on). A loss curve is convex-decreasing, so a line through the raw
    history is steeper than the current local slope; the weights pull the fit toward the recent,
    flatter section.

    WHY 1/k AND NOT A WINDOW OR GEOMETRIC DECAY -- both were measured and both break a property this
    rule needs:

    * A FIXED WINDOW pins the effective sample size, so ``sd(slope)`` never shrinks. On a loss that
      has GENUINELY CONVERGED (flat, pure measurement noise) the termination test then reads
      P = 0.79 / 0.72 / 0.86 at 12 / 100 / 400 epochs -- it never crosses 0.9, and the design trains
      forever in exactly the case the rule exists to catch.
    * GEOMETRIC weights cap the effective sample size at ``1/(1-lambda)`` and reinstate the same
      non-termination.
    * ``1/k`` sums like the harmonic series, so the effective sample size grows without bound (slowly:
      3.1 at n=12, 7.3 at n=800). ``sd(slope)`` collapses 3.6e-4 -> 4e-6 over that range and the
      termination test fires from n=30 on a converged loss.

    The residual bias is not a defect, and the ratio that makes it look like one is misleading: it is
    ``fitted / true_local``, whose denominator collapses as the curve flattens, so it diverges by
    construction. The rules ask an ABSOLUTE question and the predicted change falls monotonically
    (0.179 -> 0.021 -> 0.0024 at n = 20 / 80 / 250 on a realistic decay), so a fixed threshold is
    always eventually crossed. Weighting only moves WHEN: measured first firing at n=133 weighted
    against n=193 unweighted.

    Returns ``(mean, cov)`` for the 2-vector ``[intercept, slope]``.
    """
    y = np.asarray(values, dtype=np.float64)
    s = np.asarray(sems, dtype=np.float64)
    n = y.shape[0]
    design = np.stack([np.ones(n), np.arange(n, dtype=np.float64)], axis=1)
    precision = 1.0 / np.square(np.maximum(s, 1.0e-12))
    if harmonic:
        precision = precision / np.arange(n, 0, -1, dtype=np.float64)
    a = design.T @ (design * precision[:, None]) + np.eye(2) / (prior_sigma ** 2)
    cov = np.linalg.inv(a)
    return cov @ (design.T @ (precision * y)), cov


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def probability_above(mean, cov, horizon, threshold, n_points):
    """``P(line(t_last + horizon) > threshold)`` under the posterior. ``n_points`` is the history
    length, so the last observed abscissa is ``n_points - 1``."""
    x = np.array([1.0, float(n_points - 1 + horizon)])
    mu = float(x @ mean)
    sigma = math.sqrt(max(float(x @ cov @ x), 1.0e-24))
    return 1.0 - _normal_cdf((threshold - mu) / sigma)


def probability_change_below(mean, cov, horizon, bound):
    """``P(|slope| * horizon < bound)``: the chance the fitted line moves by less than ``bound`` over
    the next ``horizon`` epochs. A statement about the SLOPE posterior alone, so it does not inherit
    the intercept's uncertainty."""
    mu = float(mean[1]) * horizon
    sigma = math.sqrt(max(float(cov[1, 1]), 1.0e-24)) * horizon
    return _normal_cdf((bound - mu) / sigma) - _normal_cdf((-bound - mu) / sigma)
