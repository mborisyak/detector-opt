from typing import Any

from .common import Detector

from . import straw_detector
from . import straw

from .. import utils
from .straw import StrawDetector
from .debug import DebugDetector

__detectors__: dict[str, type[Detector]] = {
    "straw": StrawDetector,
    "debug": DebugDetector,
}


def from_config(config: dict[str, Any]):
    clazz, arguments = utils.config.extract(config, library=__detectors__)
    return clazz.from_config(config=arguments)
