"""Tracking library: 2-track momentum/vertex fits over raw ``StrawEvent`` records.

Two objective families -- ``RetinaTracker`` (robust alignment score) and ``NLLTracker`` (proper
2-track + uniform-noise mixture) -- each composed with the three measurements (tubes / drift / hits).
Build from a config dict bound to a detector + design via :func:`from_config`."""
from .base import Tracker
from .retina import RetinaTracker, RetinaTubes, RetinaTDC, RetinaDrift, RetinaHits
from .nll import NLLTracker, NLLTubes, NLLTDC, NLLDrift, NLLHits
from . import metrics
from ..detector import track_solver
from .. import utils

__trackers__ = {
    "retina-tubes": RetinaTubes,
    "retina-tdc": RetinaTDC,
    "retina-drift": RetinaDrift,
    "retina-hits": RetinaHits,
    "nll-tubes": NLLTubes,
    "nll-tdc": NLLTDC,
    "nll-drift": NLLDrift,
    "nll-hits": NLLHits,
}


def from_config(config, detector, design, **overrides):
    """``{"retina-hits": {hyperparams}}`` + detector + physical design -> the constructed tracker.
    ``overrides`` (e.g. ``shared_vertex=True``) take precedence over the config's hyperparams."""
    clazz, arguments = utils.config.extract(config, library=__trackers__)
    return clazz(detector, design, **{**arguments, **overrides})
