from typing import Any, Sequence

from flax import nnx

from .. import utils
from .regression import *

__models__: dict[str, type[Regressor]] = {
  'mlp': MLP,
  'resnet': AlphaResNet,
  'hyper-resnet': HyperResNet,
}

def from_config(
  input_shape: Sequence[int], design_shape: Sequence[int], target_shape: Sequence[int],
  config: dict[str, Any], *, rngs: nnx.Rngs
):
  model, arguments = utils.config.extract(config, library=__models__)
  return model.from_config(input_shape, design_shape, target_shape, config=arguments, rngs=rngs)