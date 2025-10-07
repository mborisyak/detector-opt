from typing import Any

from . import straw_signal

from .common import Detector

from . import straw_detector
from . import straw

from .. import utils
from .straw import StrawDetector
from .simple_straw import SimpleStrawDetector
from .sparse_straw import SparseStrawDetector

__detectors__: dict[str, type[Detector]] = {
  'straw': StrawDetector,
  'simple_straw': SimpleStrawDetector,
  'sparse_straw': SparseStrawDetector
}

def from_config(config: dict[str, Any]):
  clazz, arguments = utils.config.extract(config, library=__detectors__)
  return clazz.from_config(config=arguments)
