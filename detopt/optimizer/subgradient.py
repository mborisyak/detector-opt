import numpy as np

import jax
import jax.numpy as jnp

import optax
from flax import nnx

from ..detector import Detector
from .. import nn
from .. import utils

from .common import Optimizer

__all__ = [
  'Subgradient'
]

class Subgradient(Optimizer):
  @classmethod
  def from_config(cls, detector: Detector, config, *, rngs: nnx.Rngs):
    if 'optimizer_regressor' in config:
      config['optimizer_regressor'] = utils.config.optimizer(config['optimizer_regressor'])

    if 'optimizer_design' in config:
      config['optimizer_design'] = utils.config.optimizer(config['optimizer_design'])

    config['regressor'] = nn.from_config(
      input_shape=detector.output_shape(), design_shape=detector.design_shape(),
      target_shape=detector.target_shape(),
      rngs=rngs, config=config['regressor'],
    )

    return cls(detector=detector, **config)

  def __init__(
    self, detector: Detector, regressor: nn.Regressor,
    batch_size: int, n_steps_regressor: int,
    design_eps: float=0.1,
    optimizer_regressor=optax.adabelief(learning_rate=1.0e-3),
    optimizer_design=optax.nadam(learning_rate=1.0e-2),
  ):
    self.detector = detector
    self.regressor = regressor

    self.optimizer_regressor = nnx.Optimizer(regressor, optimizer_regressor)
    self.optimizer_design = optimizer_design
    self.optimizer_design_state = None

    self.batch_size = batch_size
    self.design_eps = design_eps
    self.n_steps_regressor = n_steps_regressor

    @nnx.jit
    def loss_f(model, measurements, design, target):
      pred = model(measurements, design)
      return jnp.mean(
        detector.loss(target, pred)
      )

    self.loss_f = loss_f

    @nnx.jit
    def step_regressor(state, measurements, design, target):
      value, grad = nnx.value_and_grad(loss_f, argnums=0)(state.model, measurements, design, target)
      state.update(grad)
      return value

    self.step_regressor = step_regressor


    @nnx.jit
    def step_design(model, measurements, design, target, opt_state):
      value, grad = nnx.value_and_grad(loss_f, argnums=2)(model, measurements, design, target)
      updates, opt_state = self.optimizer_design.update(grad, opt_state)
      design = optax.apply_updates(design, updates)
      return value, design, opt_state

    self.step_design = step_design

    @nnx.jit
    def metric_f(model, measurements, design, target):
      pred = model(measurements, design)
      return detector.metric(target, pred)

    self.metric_f = metric_f

  def step(self, seed: int | np.random.SeedSequence, design: jax.Array):
    training_losses = list()

    for i in range(self.n_steps_regressor):
      ss_design, ss_generator, ss_step = np.random.SeedSequence((seed, i)).spawn(3)
      perturbed_design = design[None] + self.design_eps * np.random.default_rng(ss_design).normal(
        size=(self.batch_size, *design.shape),
      ).astype(design.dtype)

      measurements, target = self.detector(ss_generator.entropy, perturbed_design)
      step_losses = self.step_regressor(
        self.optimizer_regressor, measurements, perturbed_design, target
      )
      training_losses.append(step_losses)

    ss_generator = np.random.SeedSequence((seed, self.n_steps_regressor))
    measurements, target = self.detector(ss_generator.entropy, design[None])

    if self.optimizer_design_state is None:
      self.optimizer_design_state = self.optimizer_design.init(design)

    loss, updated_design, self.optimizer_design_state = self.step_design(
      self.regressor, measurements, design, target, self.optimizer_design_state
    )

    return loss, np.stack(training_losses, axis=0), updated_design

  def validate(self, seed: int | np.random.SeedSequence, design):
    ss_generator = np.random.SeedSequence(seed)
    measurements, target = self.detector(ss_generator.entropy, design[None])
    return self.metric_f(self.regressor, measurements, design[None], target)

