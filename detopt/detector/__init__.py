from typing import Any

from .common import Detector

from . import straw_detector
from . import straw

from .. import utils
from .straw import StrawDetector  # abstract base (no design scheme)
from .free_straw import FreeStrawDetector, free_design_array
from .stereo_straw import StereoStrawDetector
from .stereo_tracking import StereoTracking
from .debug import DebugDetector

__detectors__: dict[str, type[Detector]] = {
    "straw": FreeStrawDetector,  # the base is abstract; "straw" = the free per-layer design
    "stereo_straw": StereoStrawDetector,
    "stereo_tracking": StereoTracking,  # stereo design, daughter-kinematics target
    "debug": DebugDetector,
}


def from_config(config: dict[str, Any]):
    clazz, arguments = utils.config.extract(config, library=__detectors__)
    return clazz.from_config(config=arguments)
