"""Parameter averaging (Polyak-Ruppert) as an optax wrapper.

:func:`with_parameter_average` carries an exponential moving average of the PARAMETERS inside the state
of any optax transformation. The wrapped optimiser returns exactly the inner one's updates, so the
trained iterate is untouched and the average is a read-only side buffer; :func:`parameter_average` reads
it back out. Wrapping rather than changing the training kernel is what keeps the averaged and the
unaveraged path the same code.

The buffer is SEEDED AT THE PARAMETERS, so its weights sum to one at every step and no debiasing is
needed: the average is a real network from step zero, and at any parameter discontinuity -- a fresh
design, a rewind, a re-initialised optimiser -- it simply IS the new network and blends forward from
there over one horizon.

``optax.ema`` averages UPDATES, not parameters, and ``optax.polyak_sgd`` is the Polyak step-size rule --
neither is iterate averaging, which is why this exists.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

__all__ = ["ParameterAverageState", "decay_for_horizon", "with_parameter_average", "parameter_average"]


class ParameterAverageState(NamedTuple):
  inner: optax.OptState
  average: optax.Params


def decay_for_horizon(steps: float) -> float:
  """Per-step decay whose ``1/e`` horizon is ``steps`` update steps."""
  steps = float(steps)
  if steps <= 1.0:
    raise ValueError(f"an averaging horizon must span more than one step, got {steps}")
  return 1.0 - 1.0 / steps


def with_parameter_average(inner: optax.GradientTransformation, decay: float) -> optax.GradientTransformation:
  """``inner``, plus an EMA of the POST-UPDATE parameters carried in its state."""
  decay = float(decay)
  if not 0.0 < decay < 1.0:
    raise ValueError(f"decay must lie strictly inside (0, 1), got {decay}")

  def init(params):
    return ParameterAverageState(inner=inner.init(params), average=jax.tree.map(jnp.asarray, params))

  def update(updates, state, params=None):
    if params is None:
      raise ValueError("a parameter average needs the parameters it averages: call update(..., params=params)")
    inner_updates, inner_state = inner.update(updates, state.inner, params)
    stepped = optax.apply_updates(params, inner_updates)
    average = jax.tree.map(lambda a, p: decay * a + (1.0 - decay) * p, state.average, stepped)
    return inner_updates, ParameterAverageState(inner=inner_state, average=average)

  return optax.GradientTransformation(init, update)


def parameter_average(state: ParameterAverageState):
  """The average held in ``state``."""
  return state.average


def with_average(state: ParameterAverageState, average: optax.Params) -> ParameterAverageState:
  """``state`` carrying ``average`` in place of its own.

  For the one case the wrapper cannot see: the network is moved by a known amount from outside the
  update (a rewind), so the estimate of where it is must move with it, while the inner optimiser state
  is rebuilt from scratch. Re-``init`` alone would seed the average on the moved network and discard the
  history the move itself kept.
  """
  return state._replace(average=average)
