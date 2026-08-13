from typing import Sequence, TypeAlias
import inspect

import math

import jax
import jax.numpy as jnp

from flax import nnx

from ..detector import Detector

__all__ = [
    "CELu",
    "SiLU",
    "LeakyTanh",
    "Softplus",
    "gated_leaky_tanh",
    "apply_with_kwargs",
    "Block",
    "bayes_aggregate",
]

Shape: TypeAlias = Sequence[int]


class Model(nnx.Module):
    @classmethod
    def from_config(cls, detector: Detector, config, *, rngs: nnx.Rngs):
        """The single factory for every Model: derive the universal external SHAPES from the detector --
        the per-hit/-element feature shape ``input_shape``, the ``target_shape`` and ``ground_truth_shape``
        -- and pass them + the config hyper-parameters to ``__init__``. Each Model takes the SAME three
        shapes and derives its own specifics (``input_shape[-1]`` etc.). Models that genuinely need the
        detector's geometry FACTORIZATION beyond these shapes (the hierarchical + legacy regressors) keep
        their own detector-based factory."""
        return cls(
            detector.combined_event_shape(),
            (detector.target_dim(),),
            (detector.ground_truth_dim(),),
            rngs=rngs,
            **config,
        )

    def __init__(
        self,
        input_shape: Shape,
        target_shape: Shape,
        ground_truth_shape: Shape,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.ground_truth_shape = ground_truth_shape

        self.rngs = rngs

    def ensemble(self) -> int | None:
        """Number of independently-trained ensemble members, or ``None``.

        ``None`` (the default) is a single model: trainers feed it one minibatch
        per step. An ``int`` ``n`` declares an ensemble of ``n`` members; trainers
        then feed ``n`` independent minibatches per step (one per member, drawn
        independently from the same pool) and average member predictions at
        evaluation.
        """
        return None

    def loss(self, loss_fn, features, mask, target, *, deterministic=True, rngs=None):
        """Per-sample loss for this model -- the MODEL owns the forward, so a subclass can
        inject network-specific loss terms (e.g. deep supervision over a per-element output).

        ``loss_fn`` is the detector's per-sample loss ``(predicted, target) -> (...,)`` (it reduces
        only over the target's last axis and broadcasts leading axes). The default simply forwards
        once and applies it; the returned array is per-sample (NOT reduced), matching the leading
        axes of ``target`` -- so call sites keep owning the reduction (train means it, design-grad
        / validation keep it per-event), exactly as a bare ``loss_fn(self(...), target)`` would.
        """
        return loss_fn(self(features, mask, deterministic=deterministic, rngs=rngs), target)

    def regularization(self):
        """``-log p(parameters)`` for THIS model, up to a constant: a scalar the caller adds to the
        MEAN loss after dividing by the number of training rows.

        Each model implements its own, because which parameters carry a prior and at what scale is a
        property of the architecture, not something a generic sweep over the parameter tree can decide.
        A tree-walking default got both wrong: it penalised BIASES (offsets, to which no
        input-to-output variance argument applies) and LEARNED ACTIVATION GAINS (initialised at 1.0,
        so a pull toward 0 changes the shape of the nonlinearity rather than the size of the weights),
        and it used one unit-variance scale for every layer regardless of fan-in.

        THE SCALE. For a map ``y = W x`` to send ``x ~ N(0, I)`` to ``y ~ N(0, I)`` the prior must be
        ``W_ji ~ N(0, 1/in_dim)``, giving ``in_dim * ||W||^2 / 2``. At initialisation that equals half
        the kernel's parameter count, which is the invariant to test an implementation against.

        THE COEFFICIENT IS THE CALLER'S. With a mean loss, MAP is
        ``(1/N) sum_i loss_i + regularization() / N``: the prior's weight relative to the data falls as
        ``1/N``, so a FIXED coefficient is a prior whose strength drifts with the window.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define regularization()")


class LeakyTanh(nnx.Module):
    def __init__(self, *shape):
        self.positive = nnx.Param(
            jnp.ones(
                shape=shape,
            )
        )
        self.negative = nnx.Param(
            jnp.ones(
                shape=shape,
            )
        )

    def __call__(self, x):
        return jax.nn.tanh(x) + self.positive[...] * jax.nn.softplus(x) - self.negative * jax.nn.softplus(-x)


def gated_leaky_tanh(x, alpha, beta):
    return jax.nn.tanh(x) + alpha * jax.nn.softplus(x) - beta * jax.nn.softplus(-x)


class CELu(nnx.Module):
    def __init__(self, n):
        self.alpha = nnx.Param(
            jnp.ones(
                shape=(n,),
            )
        )

    def __call__(self, X):
        return jax.nn.celu(X, self.alpha[...])


class SiLU(nnx.Module):
    def __call__(self, X):
        return jax.nn.silu(X)


class Softplus(nnx.Module):
    def __call__(self, X):
        return jax.nn.softplus(X)


class LeakyReLU(nnx.Module):
    def __call__(self, X):
        return jax.nn.leaky_relu(X, 0.05)


class MaxPool(nnx.Module):
    def __call__(self, X):
        return jax.lax.reduce_window(
            X,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 2, 2, 1),
            window_strides=(1, 2, 2, 1),
            padding="SAME",
        )


def has_kwargs(f, name):
    signature = inspect.signature(f)

    return any(
        param_name == name or param.kind == inspect.Parameter.VAR_KEYWORD for param_name, param in signature.parameters.items()
    )


### I can't believe I have to write this...
def apply_with_kwargs(f, args, kwargs):
    signature = inspect.signature(f)
    has_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for _, param in signature.parameters.items())

    if has_var_kw:
        return f(*args, **kwargs)
    else:
        filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return f(*args, **filtered)


def eval_with_kwargs(module, args, kwargs):
    if isinstance(module, nnx.Module):
        return apply_with_kwargs(module, args, kwargs)

    elif isinstance(module, (tuple, list)):
        return [apply_with_kwargs(m, args, kwargs) for m in module]
    else:
        raise ValueError("a module should be either nnx.Module or list/tuple of them.")


class Block(nnx.Module):
    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, *args, **kwargs):
        result = args
        *first, last = self.modules
        for module in first:
            result = eval_with_kwargs(module, result, kwargs)

            if isinstance(result, jax.Array):
                result = (result,)

        result = eval_with_kwargs(last, result, kwargs)
        return result


def bayes_aggregate(mu, log_sigma, axis, keepdims=False):
    # inv_sigma_sqr = jnp.exp(-2 * log_sigma)
    inv_sigma_sqr = jax.nn.softplus(-log_sigma)

    mu_inv_sigma_sqr = jnp.sum(mu * inv_sigma_sqr, axis=axis, keepdims=keepdims)
    inv_sum_inv_sigma_sqr = 1 / (1 + jnp.sum(inv_sigma_sqr, axis=axis, keepdims=keepdims))

    mu_aggregated = mu_inv_sigma_sqr * inv_sum_inv_sigma_sqr
    sigma_aggregated = jnp.sqrt(inv_sum_inv_sigma_sqr)

    return mu_aggregated, sigma_aggregated
