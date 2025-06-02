from typing import Any, Sequence

from flax import nnx

from ..detector import Detector
from .. import utils
from .regressor import *
from .generator import *
from .discriminator import *

__models__: dict[str, type[Regressor]] = {
  'mlp': MLP,
  'resnet': AlphaResNet,
  'hyper-resnet': HyperResNet,
  'deep-set': DeepSet,

  'deep-set-vae': DeepSetVAE,
  'cvae': CVAE,
  'mlp-vae': MLPVAE,

  'deep-set-lfi': DeepSetLFI
}

def from_config(
  detector: Detector,
  config: dict[str, Any], *, rngs: nnx.Rngs
):
  model, arguments = utils.config.extract(config, library=__models__)
  return model.from_config(detector, config=arguments, rngs=rngs)