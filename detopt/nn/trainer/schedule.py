"""Cosine-to-a-floor learning-rate schedule keyed on DATA ADDITIONS, as an optax transform.

The convergence procedure decides the total step count, so a schedule over the whole run has no horizon.
This one decays over ``epochs * steps_per_epoch`` optimiser steps since the last data addition, from
``peak`` times the base rate down to the base rate, and holds there: past the decay a design trains at
exactly the constant rate it would have without the schedule.

A DATA ADDITION RESETS THE SCHEDULE, NOT THE OPTIMISER. The multiplier is ``optax.scale_by_schedule``
chained after the inner optimiser, and that transform keeps its own step counter; :func:`rephase` zeroes
that counter and nothing else, so the inner optimiser's moments and bias-correction count carry across
the addition untouched. The chained state holds exactly one schedule counter, which is what makes the
reset unambiguous; :func:`with_cosine_floor` refuses an inner optimiser that already carries one.
"""
import jax
import jax.numpy as jnp
import optax


def cosine_floor_schedule(steps_per_epoch: int, epochs: int, peak: float):
  """Multiplier on the inner update: ``peak`` at step 0, cosine to 1.0 at ``epochs * steps_per_epoch``, 1.0 after."""
  if int(steps_per_epoch) <= 0:
    raise ValueError(f'steps_per_epoch must be > 0, got {steps_per_epoch}')
  if int(epochs) <= 0:
    raise ValueError(f'epochs must be > 0, got {epochs}')
  if float(peak) < 1.0:
    raise ValueError(f'peak must be >= 1, got {peak}')
  return optax.cosine_decay_schedule(
    init_value=float(peak), decay_steps=int(epochs) * int(steps_per_epoch), alpha=1.0 / float(peak)
  )


def _is_schedule_state(node) -> bool:
  return isinstance(node, optax.ScaleByScheduleState)


def _schedule_states(opt_state):
  return [node for node in jax.tree_util.tree_leaves(opt_state, is_leaf=_is_schedule_state) if _is_schedule_state(node)]


def with_cosine_floor(inner: optax.GradientTransformation, schedule) -> optax.GradientTransformation:
  """``inner`` followed by the ``schedule`` multiplier; the chained state carries exactly one schedule counter."""
  if len(_schedule_states(inner.init({}))) > 0:
    raise ValueError('the inner optimiser already carries a schedule counter; a data addition could not reset one alone')
  return optax.chain(inner, optax.scale_by_schedule(schedule))


def schedule_count(opt_state) -> int:
  """Steps taken since the schedule was last (re)phased."""
  states = _schedule_states(opt_state)
  if len(states) != 1:
    raise ValueError(f'expected exactly one schedule counter in the optimiser state, found {len(states)}')
  return int(states[0].count)


def rephase(opt_state):
  """``opt_state`` with its schedule counter at zero; the inner optimiser's state is unchanged."""
  return jax.tree_util.tree_map(
    lambda node: node._replace(count=jnp.zeros_like(node.count))
    if _is_schedule_state(node) else node, opt_state, is_leaf=_is_schedule_state,
  )
