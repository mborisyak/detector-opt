from typing import Any, Sequence

from flax import nnx

from ..detector import Detector
from .. import utils
from .common import Model
# NOTE: detopt/nn/regressor.py (MLP/AlphaResNet/CNN/HyperResNet/DeepSet/BayesDeepSet) and
# detopt/nn/generator.py (CVAE/DeepSetVAE/MLPVAE) are BROKEN/quarantined (unported to the padded
# (B, M, F) contract) -- deliberately NOT imported or registered here.
from .discriminator import *
from .mlp import MLPRegressor
from .set_regressor import SetRegressor
from .alpha_set_regressor import AlphaSetRegressor
from .pair_set_regressor import PairSetRegressor
from .induced import InducedSetRegressor
from .predictive import PredictiveSetRegressor, PredictiveProbRegressor, PredictiveMixtureRegressor
from .conv_regressor import ConvRegressor
from .alpha_conv_regressor import AlphaConvRegressor
from .plain_conv_regressor import PlainConvRegressor
from .continuous_conv import ContinuousConvRegressor
from .masked_set import MaskedSetRegressor
from .hierarchical import DoubleSetRegressor, StructuredSetRegressor

__models__: dict[str, type[Model]] = {
    "mlp-regressor": MLPRegressor,  # flat (order-dependent) MLP over a fixed-length element set; enzyme ablation
    "set-regressor": SetRegressor,  # ensemble via n_models (None = single net)
    # Same deep set, but each per-element block is a residual stack with zero-initialised
    # per-unit alpha: identity at init, so depth is cheap to add where the plain stack at
    # widths 16-24 degrades. Capacity is width/depth here, not a list of hidden sizes.
    "alpha-set-regressor": AlphaSetRegressor,
    "pair-set-regressor": PairSetRegressor,  # deep set over all hit PAIRS (single net)
    "set-discriminator": SetDiscriminator,
    "induced-set-regressor": InducedSetRegressor,
    "predictive-set-regressor": PredictiveSetRegressor,  # causal cumulative; for stereo_layerwise
    "predictive-prob-regressor": PredictiveProbRegressor,  # + next-layer per-straw hit-probability head
    "predictive-mixture-regressor": PredictiveMixtureRegressor,  # + next-layer Gaussian-mixture head
    "conv-regressor": ConvRegressor,  # hierarchical CNN; for stereo_image
    # Tiny residual CNN over a DENSE channels-last image (zero-init per-channel alpha, no BatchNorm);
    # for the mnist window detector, whose combine emits (rows, columns, 2).
    "alpha-conv-regressor": AlphaConvRegressor,
    # The same stack with the skip connection and the alpha gate REMOVED: the architecture control
    # for the alpha-conv runs, identical in every other hyper-parameter and operator.
    "plain-conv-regressor": PlainConvRegressor,
    "continuous-conv-regressor": ContinuousConvRegressor,  # kernel message-passing; for stereo_hits
    "masked-set-regressor": MaskedSetRegressor,  # set regressor + self-supervised flip-detection; stereo_layerwise
    "double-set-regressor": DoubleSetRegressor,  # hierarchical: straw->layer->global
    "structured-set-regressor": StructuredSetRegressor,  # hierarchical: straw->layer->view->station->global
}


def from_config(detector: Detector, config: dict[str, Any], *, rngs: nnx.Rngs):
    model, arguments = utils.config.extract(config, library=__models__)
    return model.from_config(detector, config=arguments, rngs=rngs)
