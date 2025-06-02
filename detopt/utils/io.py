import os

import flax
import jax.tree
from flax import nnx

import orbax.checkpoint as ocp

from . import config as config_utils

__all__ = [
  'get_checkpointer',
  'save_model',
  'restore_model',
  'restore_aux',

  'load_design',
  'save_design',
  'restore_state',
  'save_state'
]

def get_checkpointer(path):
  import absl.logging
  absl.logging.set_verbosity(absl.logging.ERROR)

  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)
  options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1, )
  manager = ocp.CheckpointManager(path, options=options)
  return manager

def restore_aux(manager: ocp.CheckpointManager):
  data = manager.restore(
    manager.latest_step(),
    args=ocp.args.Composite(aux=ocp.args.PyTreeRestore())
  )

  return data['aux']

def load_design(detector, design_path):
  import json
  with open(design_path, 'r') as f:
    return detector.encode_design(
      json.load(f)
    )

def save_design(detector, design_path, design):
  import json
  os.makedirs(os.path.dirname(design_path), exist_ok=True)

  with open(design_path, 'w') as f:
    json.dump(detector.decode_design(design), f, indent=2)

def save_model(parameters, state, optimizer_state):
  if parameters is None:
    return dict(parameters=None, state=None, optimizer_state=None)

  optimizer_state = jax.tree.leaves(optimizer_state)
  return dict(parameters=parameters, state=state, optimizer_state=optimizer_state)

def save_state(
  step, manager: ocp.CheckpointManager,
  design, design_optimizer_state=None,
  regressor_parameters=None, regressor_state=None, regressor_optimizer_state=None,
  generator_parameters=None, generator_state=None, generator_optimizer_state=None,
  discriminator_parameters=None, discriminator_state=None, discriminator_optimizer_state=None,
  *, aux
):
  if design_optimizer_state is not None:
    design_optimizer_state =  jax.tree.leaves(design_optimizer_state)

  manager.save(step, args=ocp.args.Composite(
    ### because saving a standalone array is difficult
    design=ocp.args.PyTreeSave({'design' : design, 'optimizer_state': design_optimizer_state}),

    regressor=ocp.args.PyTreeSave(
      save_model(regressor_parameters, regressor_state, regressor_optimizer_state)
    ),
    generator=ocp.args.PyTreeSave(
      save_model(generator_parameters, generator_state, generator_optimizer_state)
    ),
    discriminator=ocp.args.PyTreeSave(
      save_model(discriminator_parameters, discriminator_state, discriminator_optimizer_state)
    ),

    aux=ocp.args.PyTreeSave(aux)
  ))

def restore_model(config, detector, restored, rngs):
  from .. import nn
  if config is None:
    return dict(
      model=None, optimizer=None,
      parameters=None, state=None, optimizer_state=None
    )

  model = nn.from_config(
    detector, config=config['model'], rngs=rngs
  )
  model_def, initial_model_parameters, initial_model_state = nnx.split(model, nnx.Param, nnx.Variable)
  optimizer = config_utils.optimizer(config['optimizer'])

  if restored is None:
    parameters = nnx.to_pure_dict(initial_model_parameters)
    model_state = nnx.to_pure_dict(initial_model_state)
    optimizer_state = optimizer.init(parameters)
  else:
    parameters = restored['parameters']
    model_state = restored['state']
    optimizer_state = jax.tree.unflatten(
      jax.tree.structure(optimizer.init(parameters)),
      restored['optimizer_state']
    )

  return dict(
    model=model_def, optimizer=optimizer,
    parameters=parameters, state=model_state, optimizer_state=optimizer_state
  )

def restore_state(manager, detector, config, *, rngs: nnx.Rngs, restore=True):
  if 'optimizer' in config:
    design_optimizer = config_utils.optimizer(config['optimizer'])
  else:
    design_optimizer = None

  last_epoch = manager.latest_step()
  if last_epoch is not None and restore:
    starting_epoch = last_epoch + 1

    data = manager.restore(
      manager.latest_step(),
      args=ocp.args.Composite(
        design=ocp.args.PyTreeRestore(),
        regressor=ocp.args.PyTreeRestore(),
        generator=ocp.args.PyTreeRestore(),
        discriminator=ocp.args.PyTreeRestore(),
        aux=ocp.args.PyTreeRestore()
      )
    )
    ### design -> design because it is a standalone array
    design, design_optimizer_state = data['design']['design'], data['design']['optimizer_state']

    if design_optimizer_state is not None:
      design_optimizer_state = jax.tree.unflatten(
        jax.tree.structure(design_optimizer.init(design)),
        design_optimizer_state
      )

  else:
    starting_epoch = 0
    data = {}

    ### design -> design because it is a standalone array
    design = load_design(detector, config['initial_design'])
    if design_optimizer is None:
      design_optimizer_state = None
    else:
      design_optimizer_state = design_optimizer.init(design)

  regressor = restore_model(
    config.get('regressor', None), detector, data.get('regressor', None), rngs=rngs
  )

  generator = restore_model(
    config.get('generator', None), detector, data.get('generator', None), rngs=rngs
  )

  discriminator = restore_model(
    config.get('discriminator', None), detector, data.get('discriminator', None), rngs=rngs
  )

  return dict(
    starting_epoch=starting_epoch,
    design=dict(design=design, optimizer=design_optimizer, optimizer_state=design_optimizer_state),
    regressor=regressor, generator=generator, discriminator=discriminator,
    aux=data.get('aux', None)
  )