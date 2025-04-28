import numpy as np

import jax

from flax import nnx

from ..detector import Detector

__all__ = [
  'Optimizer'
]

class Optimizer(object):
  @classmethod
  def from_config(cls, detector: Detector, config, *, rngs: nnx.Rngs):
    return cls(detector, **config)

  def step(self, seed: int | np.random.SeedSequence, design: jax.Array):
    raise NotImplementedError()