import math
import numpy as np

__all__ = [
  'Detector'
]

class Detector(object):
  def design_shape(self):
    raise NotImplementedError()

  def design_dim(self):
    return math.prod(self.design_shape())

  def output_shape(self):
    raise NotImplementedError()

  def output_dim(self):
    return math.prod(self.output_shape())

  def __call__(self, seed: int, configurations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError()

  def loss(self, target, predicted):
    raise NotImplementedError()



