"""Shared test stand-in for the removed ``DebugDetector``: a small no-data analytic stereo detector
(``Stereo4Feature`` with ``engine='simplified'`` -- HNL Gaussians + 2-body decay, propagated by the fast
analytic engine). ``size()`` is ``None`` (infinite), so tests request any ``event_index`` they like."""

import numpy as np

from detopt.detector import Stereo4Feature

# Stereo physical design: station centres (4) + the stereo view angle (rad).
DESIGN = np.array([8407.0, 8607.0, 9307.0, 9507.0, 0.0798], np.float32)


def analytic_detector(**kwargs):
    """A no-data analytic stereo detector for tests (override any default via kwargs)."""
    return Stereo4Feature(engine="simplified", **kwargs)
