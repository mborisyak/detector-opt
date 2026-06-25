import numpy as np

import jax

from flax import nnx

from ..detector import Detector

__all__ = ["Optimizer"]


class Optimizer(object):
    @classmethod
    def from_config(cls, detector: Detector, config, *, rngs: nnx.Rngs):
        """The single optimizer factory: resolve the nested config sections (the optax optimizers + the
        regressor network) from their own configs, then construct. Sections absent from a given optimizer's
        config are skipped, so this stays the ONE concrete factory for every Optimizer (no subclass override)."""
        from .. import nn, utils  # local import avoids an import cycle at module load

        config = dict(config)
        for key in ("optimizer_regressor", "optimizer_design"):
            if key in config:
                config[key] = utils.config.optimizer(config[key])
        if "regressor" in config:
            config["regressor"] = nn.from_config(detector, config["regressor"], rngs=rngs)
        return cls(detector, **config)

    def step(self, seed: int | np.random.SeedSequence, design: jax.Array):
        raise NotImplementedError()
