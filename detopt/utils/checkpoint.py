import os

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

__all__ = [
  'get_manager',
  'save_model',
  'load_model',
]

def get_manager(path):
  import absl.logging
  absl.logging.set_verbosity(absl.logging.ERROR)

  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)

  options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1, )
  manager = ocp.CheckpointManager(path, options=options)
  return manager

def save_model(manager, step: int, parameters, state, optimizer_state, aux):
  return manager.save(
    step, args=ocp.args.Composite(
      model=ocp.args.StandardSave({
        'parameters': jax.tree.leaves(parameters),
        'state': jax.tree.leaves(state),
        'optimizer_state': jax.tree.leaves(optimizer_state)
      }),
      aux=ocp.args.StandardSave(aux)
    )
  )

def load_model(manager: ocp.CheckpointManager):
  latest_step = manager.latest_step()
  if latest_step is None:
    return None, None

  device, = jax.devices()
  fallback_sharding = jax.sharding.SingleDeviceSharding(device)

  restored = manager.restore(
    latest_step,
    args=ocp.args.Composite(
      model=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
      aux=ocp.args.StandardRestore(fallback_sharding=fallback_sharding)
    )
  )
  return latest_step, restored