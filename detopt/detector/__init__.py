from typing import Any

from .common import Detector

from . import straw_detector
from . import straw

from .. import utils
from .enzyme import EnzymeDetector
from .straw import StrawDetector  # abstract base (no design scheme)
from .free_straw import FreeStrawDetector, free_design_array
from .stereo_straw import StereoStrawDetector
from .stereo_tracking import Stereo4Feature, StereoTrackerTruth
from .stereo_tracking_layerwise import StereoLayerWise
from .stereo_tracking_image import StereoImage
from .stereo_tracking_hits import StereoHits

__detectors__: dict[str, type[Detector]] = {
    # GEOMETRY bases (StrawDetector / StereoStrawDetector) are abstract-combine and NOT registered as
    # leaves: every entry below is a concrete combine LEAF. FairShip REPLAY is no longer a separate class
    # -- it is the combine leaf with `engine: relay` + `data_dir` (a special engine-less source handled in
    # StrawDetector, loads one digi file), so `ship_relay_*` are those leaves; their configs set the engine.
    # Not a particle detector at all: the enzymatic-reaction single-batch design of experiments
    # (the proposal's bioprocess work package) under the same contract.
    "enzyme": EnzymeDetector,
    "straw": FreeStrawDetector,  # free per-layer geometry + 4-feature combine
    "stereo_tracking": Stereo4Feature,  # stereo geometry + 4-feature combine (the default stereo detector)
    "stereo_tracker_truth": StereoTrackerTruth,  # + per-hit (x,y)/drift_r/tdc TRUTH for tracker experiments
    "stereo_layerwise": StereoLayerWise,  # set element = layer (TDC grid per layer)
    "stereo_image": StereoImage,  # (n_layers, n_straws, 6) image for the CNN regressor
    "stereo_hits": StereoHits,  # per-hit (M, 7) features for the continuous-conv regressor
    "ship_relay_4feat": Stereo4Feature,  # replay FairShip digi hits (engine: relay): 4-feature combine
    "ship_relay_layerwise": StereoLayerWise,  # replay FairShip digi hits (engine: relay): layer-wise combine
}


def from_config(config: dict[str, Any]):
    clazz, arguments = utils.config.extract(config, library=__detectors__)
    return clazz.from_config(config=arguments)
