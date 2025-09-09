from typing import Any

from .common import Detector

from . import straw_detector
from . import straw

from .. import utils
from .straw import StrawDetector
from .simple_straw import SimpleStrawDetector

__detectors__: dict[str, type[Detector]] = {
  'straw': StrawDetector,
  'simple_straw': SimpleStrawDetector
}

def from_config(config: dict[str, Any]):
  clazz, arguments = utils.config.extract(config, library=__detectors__)
  return clazz.from_config(config=arguments)
