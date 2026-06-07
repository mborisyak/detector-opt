"""Expected Improvement acquisition for Bayesian optimisation.

Both the analytic EI score and a multi-restart **L-BFGS** maximiser of EI over
the unit cube are exposed. The objective is *minimised*; the incumbent is the
best (lowest) observed value, read straight off the GP state
(``state.y_train.min()``). (EI itself is always maximised to pick the next point.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jax.scipy.stats import norm as jax_norm

from .gp import GPState, gp_predict

__all__ = [
    "expected_improvement",
    "optimise_ei",
]


def expected_improvement(X: jax.Array, state: GPState) -> jax.Array:
    """Analytic Expected Improvement at ``X`` for **minimisation**.

    ``X`` has shape ``(m, d)``. Returns a ``(m,)`` vector of EI values. The
    incumbent is the best (lowest) observed objective, ``state.y_train.min()``;
    improvement is ``max(y_best - y, 0)``.
    """
    mean, var = gp_predict(state, X)
    sigma = jnp.sqrt(var)
    y_best = jnp.min(state.y_train)
    z = (y_best - mean) / sigma
    return (y_best - mean) * jax_norm.cdf(z) + sigma * jax_norm.pdf(z)


def optimise_ei(
    key: jax.Array,
    state: GPState,
    *,
    n_restarts: int,
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Multi-restart L-BFGS maximisation of EI over the unit cube ``[0, 1]^d``.

    The GP/EI always live in the normalised design cube ``[0, 1]^d``. The search
    is parameterised in unconstrained space via a logistic map (``x =
    sigmoid(z)``), so L-BFGS (with a zoom/Wolfe line search) stays unconstrained
    while ``x`` is confined to ``(0, 1)``. Each of ``n_restarts`` independent
    starts runs ``n_steps`` iterations; the best-seen iterate is kept.

    Returns ``(x_best, ei_best)`` -- the maximiser in ``[0, 1]^d`` and its EI.
    """
    d = state.X_train.shape[1]
    x0 = jax.random.uniform(key, (n_restarts, d), minval=0.0, maxval=1.0, dtype=jnp.float32)
    x0 = jnp.clip(x0, 1e-4, 1.0 - 1e-4)
    z0 = jnp.log(x0) - jnp.log1p(-x0)  # logit

    def neg_ei(z):
        x = jax.nn.sigmoid(z)
        return -expected_improvement(x[None, :], state)[0]

    def run_one(z_init):
        opt = optax.lbfgs()  # default: zoom (Wolfe) line search
        value_and_grad = optax.value_and_grad_from_state(neg_ei)

        def step(carry, _):
            z, opt_state, best_z, best_v = carry
            v, g = value_and_grad(z, state=opt_state)
            improved = v < best_v
            best_z = jnp.where(improved, z, best_z)
            best_v = jnp.where(improved, v, best_v)
            updates, opt_state = opt.update(g, opt_state, z, value=v, grad=g, value_fn=neg_ei)
            z = optax.apply_updates(z, updates)
            return (z, opt_state, best_z, best_v), None

        init = (z_init, opt.init(z_init), z_init, jnp.inf)
        (_, _, best_z, best_v), _ = jax.lax.scan(step, init, None, length=n_steps)
        return jax.nn.sigmoid(best_z), -best_v

    x_finals, ei_finals = jax.jit(jax.vmap(run_one))(z0)  # (n_restarts, d), (n_restarts,)
    best_idx = jnp.argmax(ei_finals)
    return x_finals[best_idx], ei_finals[best_idx]
