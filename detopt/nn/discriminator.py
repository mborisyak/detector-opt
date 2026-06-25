"""Detector-agnostic LFI discriminator (ground-truth conditioned).

A :class:`~detopt.nn.set_regressor.SetRegressor` (held by COMPOSITION) with a fixed single-unit
output, squeezed to a scalar logit per event. It consumes ``(features, mask, conditioning)`` where
the conditioning is the (normalized) HNL ground truth ``[mass, p(3)]``; the conditioning is broadcast
onto every hit and concatenated to the design-``combine``d per-hit features.

Used by ``scripts/lfi.py`` to separate the joint ``(X, theta) | gt`` from the product
``(X, theta_shuffled) | gt`` (the ground-truth conditioning stays matched to ``X`` in both;
only ``theta`` is shuffled). At the optimum its logit is
``log p(X | theta, gt) - log p(X | gt)``, so its theta-gradient estimates the conditional
score ``grad_theta log p(X | theta, gt)``. Conditioning on the HNL kinematics removes that
nuisance variation so the theta-signal is learnable.
"""

import jax.numpy as jnp
from flax import nnx

from .common import Model
from .set_regressor import SetRegressor

__all__ = ["SetDiscriminator"]


class SetDiscriminator(Model):
    """A :class:`SetRegressor` with a single logit output, conditioned on the HNL ground truth.

    ``__call__(features, mask, conditioning)`` broadcasts ``conditioning`` ``(..., C)`` onto the ``M``
    hits, concatenates it to ``features`` ``(..., M, F)``, and returns the inner regressor's output with
    the length-1 target axis dropped: ``(..., )``. Holds the inner regressor by COMPOSITION (not
    subclassing), so it adds its own forward without overriding ``SetRegressor.__call__``.
    """

    def __init__(self, input_shape, target_shape, ground_truth_shape, features, n_models=None,
                 p_dropout=None, *, rngs: nnx.Rngs):
        super().__init__(input_shape, target_shape, ground_truth_shape, rngs=rngs)
        # The HNL conditioning is concatenated onto every hit, so the inner regressor's feature width is the
        # combine width PLUS the ground-truth width; its output is a single logit (target dim 1).
        feat_in = int(input_shape[-1]) + int(ground_truth_shape[0])
        self.regressor = SetRegressor((int(input_shape[0]), feat_in), (1,), ground_truth_shape, features,
                                      n_models=n_models, p_dropout=p_dropout, rngs=rngs)

    def ensemble(self) -> int | None:
        return self.regressor.ensemble()

    def __call__(self, features, mask, conditioning, *, deterministic: bool = True, rngs=None):
        cond = jnp.broadcast_to(conditioning[..., None, :], features.shape[:-1] + (conditioning.shape[-1],))
        feats = jnp.concatenate([features, cond], axis=-1)
        return self.regressor(feats, mask, deterministic=deterministic, rngs=rngs)[..., 0]