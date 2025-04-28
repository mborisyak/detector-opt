from typing import Any

import math
import numpy as np

__all__ = [
  'Detector'
]

class Detector(object):
  @classmethod
  def from_config(cls, config: dict[str, Any]) -> 'Detector':
    return cls(**config)

  def design_shape(self):
    raise NotImplementedError()

  def design_dim(self):
    return math.prod(self.design_shape())

  def output_shape(self):
    raise NotImplementedError()

  def output_dim(self):
    return math.prod(self.output_shape())

  def target_shape(self):
    raise NotImplementedError()

  def target_dim(self):
    return math.prod(self.target_shape())

  def __call__(self, seed: int, configurations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError()

  def loss(self, target, predicted):
    raise NotImplementedError()

  def metric(self, target, predicted):
    return self.loss(target, predicted)



