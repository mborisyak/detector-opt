import numpy as np

import jax
import jax.numpy as jnp

import optax
from flax import nnx

from ..detectors import Detector
from ..nn import Regressor

__all__ = [
  'Subgradient'
]

def loss(regressor, measurements, design, target, optimizer):
  pass

class Subgradient(object):
  def __init__(
    self, detector: Detector, regressor: Regressor,
    batch_size: int, n_steps_regressor: int | None,
    design_eps: float=0.05, optimizer = optax.adabelief(learning_rate=1.0e-3)
  ):
    self.detector = detector
    self.regressor = regressor

    self.optimizer = nnx.Optimizer(regressor, optimizer)

  def train_regressor(self, seed, design):

  def step(self, seed, design: np.ndarray):
    pass