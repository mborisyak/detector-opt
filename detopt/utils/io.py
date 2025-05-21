import os

import pickle
import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp

__all__ = [
  'save_model',
  'load_model'
]

def save_model(path, model, aux):
  _, state = nnx.split(model)

  checkpointer = ocp.StandardCheckpointer()
  checkpointer.save(os.path.abspath(path), state)

  # with open(path, 'wb') as f:
  #   pickle.dump({'state' : state_dict, 'aux': aux}, f)

def load_model(path, model):
  with open(path, 'rb') as f:
    data = pickle.load(f)

  model_def, abs_state = nnx.split(model)

  state = jax.tree.map(
    lambda x: jnp.array(x, dtype=x.dtype),
    data['state']
  )
  abs_state.replace_by_pure_dict(state)
  model = nnx.merge(model_def, state)
  return model, data['aux']