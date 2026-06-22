"""Retina tracker: a bounded, redescending alignment SCORE (not a likelihood). Each hit contributes a
Gaussian bump on its nearest track; the objective maximises the summed response (minimises its negative)
plus the optional momentum prior. Composed with each measurement to form the three concrete trackers."""
import jax.numpy as jnp

from .base import Tracker
from .measurements import TubesMeasurement, DriftMeasurement, HitsMeasurement, TDCMeasurement


class RetinaTracker(Tracker):

    def objective(self, residual, valid, sharpness, states_phys):
        bump = jnp.exp(-self.temperature * (residual / sharpness) ** 2)  # P(hit|track)^temperature
        response = jnp.sum(jnp.max(bump, axis=0) * valid)  # each hit claimed by its nearest track
        return -response + self.momentum_penalty(states_phys)


class RetinaTubes(TubesMeasurement, RetinaTracker):
    pass


class RetinaTDC(TDCMeasurement, RetinaTracker):
    pass


class RetinaDrift(DriftMeasurement, RetinaTracker):
    pass


class RetinaHits(HitsMeasurement, RetinaTracker):
    pass
