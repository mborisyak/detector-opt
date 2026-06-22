"""NLL / MAP tracker: a PROPER generative mixture likelihood. Each hit is drawn from track 0, track 1
(Gaussian on the drift-circle residual, width = annealed ``sharpness``) or a UNIFORM noise band ``1/W``
with a fixed fraction ``pi_noise``. The objective is the negative mixture log-likelihood plus the
optional momentum prior (a genuine prior here, since this IS a likelihood)."""
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .base import Tracker
from .measurements import TubesMeasurement, DriftMeasurement, HitsMeasurement, TDCMeasurement

HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)  # Gaussian log-normalisation constant


class NLLTracker(Tracker):

    def objective(self, residual, valid, sharpness, states_phys):
        log_inlier = jnp.log(0.5 * (1.0 - self.pi_noise))  # each of the 2 tracks shares (1 - pi)/2
        log_track = -0.5 * (residual / sharpness) ** 2 - jnp.log(sharpness) - HALF_LOG_2PI + log_inlier  # (2, n)
        log_noise = jnp.full((1, residual.shape[1]), jnp.log(self.pi_noise) - jnp.log(self.noise_W))  # uniform floor
        loglik = logsumexp(jnp.concatenate([log_track, log_noise], axis=0), axis=0)  # (n,) per-hit mixture
        return -jnp.sum(loglik * valid) + self.momentum_penalty(states_phys)


class NLLTubes(TubesMeasurement, NLLTracker):
    pass


class NLLTDC(TDCMeasurement, NLLTracker):
    pass


class NLLDrift(DriftMeasurement, NLLTracker):
    pass


class NLLHits(HitsMeasurement, NLLTracker):
    pass
