import numpy as np

import jax
import jax.numpy as jnp

import optax
from flax import nnx

from ..detector import Detector
from .. import nn
from .. import utils

from .common import Optimizer

__all__ = ["Subgradient"]


class Subgradient(Optimizer):
    @classmethod
    def from_config(cls, detector: Detector, config, *, rngs: nnx.Rngs):
        if "optimizer_regressor" in config:
            config["optimizer_regressor"] = utils.config.optimizer(config["optimizer_regressor"])

        if "optimizer_design" in config:
            config["optimizer_design"] = utils.config.optimizer(config["optimizer_design"])

        config["regressor"] = nn.from_config(detector, config["regressor"], rngs=rngs)

        return cls(detector=detector, **config)

    def __init__(
        self,
        detector: Detector,
        regressor: nn.Regressor,
        batch_size: int,
        n_steps_regressor: int,
        design_eps: float = 0.1,
        optimizer_regressor=optax.adabelief(learning_rate=1.0e-3),
        optimizer_design=optax.nadam(learning_rate=1.0e-2),
    ):
        self.detector = detector
        self.batch_size = batch_size
        self.design_eps = design_eps
        self.n_steps_regressor = n_steps_regressor

        # Functional handle on the regressor: a static graphdef plus a params
        # pytree we optimise with plain optax. We avoid nnx.Optimizer/nnx.jit and
        # merge the model back inside the jitted steps. The non-param state is fixed
        # (the regressor runs deterministically here), so we close over it.
        graphdef, params, nonparams = nnx.split(regressor, nnx.Param, ...)
        self.graphdef = graphdef
        self.params = params

        self.optimizer_regressor = optimizer_regressor
        self.opt_regressor_state = optimizer_regressor.init(params)
        self.optimizer_design = optimizer_design
        self.opt_design_state = None

        def loss_f(params, measurements, design, target):
            model = nnx.merge(graphdef, params, nonparams)
            pred = model(measurements, design)
            return jnp.mean(detector.loss(pred, target))

        @jax.jit
        def step_regressor(params, opt_state, measurements, design, target):
            value, grad = jax.value_and_grad(loss_f)(params, measurements, design, target)
            updates, opt_state = optimizer_regressor.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return value, params, opt_state

        self.step_regressor = step_regressor

        @jax.jit
        def step_design(params, opt_state, measurements, design, target):
            value, grad = jax.value_and_grad(loss_f, argnums=2)(params, measurements, design, target)
            updates, opt_state = optimizer_design.update(grad, opt_state, design)
            design = optax.apply_updates(design, updates)
            return value, design, opt_state

        self.step_design = step_design

        @jax.jit
        def metric_f(params, measurements, design, target):
            model = nnx.merge(graphdef, params, nonparams)
            pred = model(measurements, design)
            return detector.metric(pred, target)

        self.metric_f = metric_f

    def step(self, seed: int | np.random.SeedSequence, design: jax.Array):
        design = jnp.asarray(design)
        training_losses = list()

        for i in range(self.n_steps_regressor):
            ss_design, ss_generator, _ss_step = np.random.SeedSequence((seed, i)).spawn(3)
            noise = (
                np.random.default_rng(ss_design)
                .normal(
                    size=(self.batch_size, *design.shape),
                )
                .astype(np.float32)
            )
            perturbed_design = np.asarray(design)[None] + self.design_eps * noise

            _gt, measurements, _mask, target = self.detector(ss_generator.entropy, perturbed_design)
            value, self.params, self.opt_regressor_state = self.step_regressor(
                self.params,
                self.opt_regressor_state,
                measurements,
                perturbed_design,
                target,
            )
            training_losses.append(value)

        ss_generator = np.random.SeedSequence((seed, self.n_steps_regressor))
        _gt, measurements, _mask, target = self.detector(ss_generator.entropy, np.asarray(design)[None])

        if self.opt_design_state is None:
            self.opt_design_state = self.optimizer_design.init(design)

        loss, design, self.opt_design_state = self.step_design(self.params, self.opt_design_state, measurements, design, target)

        self.last_loss = loss
        self.last_training_losses = np.stack(training_losses, axis=0)
        return design

    def validate(self, seed: int | np.random.SeedSequence, design):
        ss_generator = np.random.SeedSequence(seed)
        _gt, measurements, _mask, target = self.detector(ss_generator.entropy, np.asarray(design)[None])
        return self.metric_f(self.params, measurements, np.asarray(design)[None], target)
