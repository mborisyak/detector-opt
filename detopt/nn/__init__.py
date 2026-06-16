from typing import Any, Sequence

from flax import nnx

from ..detector import Detector
from .. import utils
from .regressor import *
from .generator import *
from .discriminator import *
from .set_regressor import SetRegressor

__models__: dict[str, type[Regressor]] = {
    "mlp": MLP,
    "resnet": AlphaResNet,
    "hyper-resnet": HyperResNet,
    "deep-set": DeepSet,
    "bayes-deep-set": BayesDeepSet,
    "set-regressor": SetRegressor,  # ensemble via n_models (None = single net)
    "deep-set-vae": DeepSetVAE,
    "cvae": CVAE,
    "mlp-vae": MLPVAE,
    "set-discriminator": SetDiscriminator,
}


def from_config(detector: Detector, config: dict[str, Any], *, rngs: nnx.Rngs):
    model, arguments = utils.config.extract(config, library=__models__)
    return model.from_config(detector, config=arguments, rngs=rngs)
