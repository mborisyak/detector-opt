from .subgradient import Subgradient

from flax import nnx

from .. import utils
from ..detector import Detector

__methods__ = {
  'subgradient': Subgradient
}

def from_config(detector: Detector, config, *, rngs: nnx.Rngs):
  method, arguments = utils.config.extract(config, library=__methods__)
  return method.from_config(detector, config=arguments, rngs=rngs)