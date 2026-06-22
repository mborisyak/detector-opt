"""Detector-agnostic LFI discriminator (ground-truth conditioned).

A :class:`~detopt.nn.set_regressor.SetRegressor` with a fixed single-unit output, squeezed
to a scalar logit per event. It consumes ``(features, mask, conditioning)`` where the
conditioning is the (normalized) HNL ground truth ``[mass, p(3)]``; the conditioning is
broadcast onto every hit and concatenated to the design-``combine``d per-hit features.

Used by ``scripts/lfi.py`` to separate the joint ``(X, theta) | gt`` from the product
``(X, theta_shuffled) | gt`` (the ground-truth conditioning stays matched to ``X`` in both;
only ``theta`` is shuffled). At the optimum its logit is
``log p(X | theta, gt) - log p(X | gt)``, so its theta-gradient estimates the conditional
score ``grad_theta log p(X | theta, gt)``. Conditioning on the HNL kinematics removes that
nuisance variation so the theta-signal is learnable.
"""

import jax.numpy as jnp
from flax import nnx

from .set_regressor import SetRegressor

__all__ = ["SetDiscriminator"]


class SetDiscriminator(SetRegressor):
    """SetRegressor with a single logit output, conditioned on the HNL ground truth.

    ``__call__(features, mask, conditioning)`` broadcasts ``conditioning`` ``(..., C)`` onto
    the ``M`` hits, concatenates it to ``features`` ``(..., M, F)``, and returns the
    SetRegressor output with the length-1 target axis dropped: ``(..., )``.
    """

    @classmethod
    def from_config(cls, detector, config, *, rngs: nnx.Rngs):
        n_in = int(detector.combined_feature_dim) + int(detector.ground_truth_dim())
        return cls(n_features_in=n_in, target_dim=1, rngs=rngs, **config)

    def __call__(self, features, mask, conditioning, *, deterministic: bool = True, rngs=None):
        cond = jnp.broadcast_to(conditioning[..., None, :], features.shape[:-1] + (conditioning.shape[-1],))
        feats = jnp.concatenate([features, cond], axis=-1)
        return super().__call__(feats, mask, deterministic=deterministic, rngs=rngs)[..., 0]
