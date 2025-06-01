import numpy as np
import jax
import jax.numpy as jnp

import flax
from flax import nnx
import optax

import detopt

def get_model(checkpointer, input_shape, design_shape, target_shape, config, *, rngs: nnx.Rngs, restore=True):
  regressor = detopt.nn.from_config(input_shape, design_shape, target_shape, config=config['model'], rngs=rngs)
  regressor_def, parameters, regressor_state = nnx.split(regressor, nnx.Param, nnx.Variable)

  optimizer = detopt.utils.config.optimizer(config['optimizer'])

  last_epoch = checkpointer.latest_step()
  print(last_epoch)
  if last_epoch is not None and restore:
    parameters, regressor_state, optimizer_state = detopt.utils.io.restore_model(checkpointer)
    state = optimizer.init(parameters)
    optimizer_state = flax.serialization.from_state_dict(state, optimizer_state)
    starting_epoch = last_epoch + 1
  else:
    optimizer_state = optimizer.init(parameters)
    starting_epoch = 0

  return starting_epoch, regressor_def, parameters, regressor_state, optimizer, optimizer_state

def get_design(detector, design_path):
  import json
  with open(design_path, 'r') as f:
    return detector.encode_design(
      json.load(f)
    )

def info(**config):
  import math

  rng = jax.random.PRNGKey(1)
  rngs = nnx.Rngs(1)

  detector = detopt.detector.from_config(config['detector'])
  regressor = detopt.nn.from_config(
    detector.output_shape(), detector.design_shape(), detector.target_shape(),
    config=config['regressor']['model'], rngs=rngs
  )
  _, parameters, _ = nnx.split(regressor, nnx.Param, nnx.Variable)
  total_number_of_parameters = sum(math.prod(p.shape) for p in jax.tree.leaves(parameters))

  print(f'Total number of parameters: {total_number_of_parameters}')

def regress(seed, output, progress=True, restore=True, **config):
  rng = jax.random.PRNGKey(seed)
  np_rng = np.random.default_rng(seed=(seed, 0))

  detector = detopt.detector.from_config(config['detector'])
  design = get_design(detector, config['initial_design'])
  design_eps = float(config['design_epsilon'])

  rng, key_init = jax.random.split(rng, num=2)

  epochs, steps = config['epochs'], config['steps']
  batch, validation_batches = config['batch'], config['validation_batches']

  rng, key_init = jax.random.split(rng, num=2)
  checkpointer = detopt.utils.io.get_checkpointer(output)

  starting_epoch, regressor_def, parameters, regressor_state, optimizer, optimizer_state = get_model(
    checkpointer,
    input_shape=detector.output_shape(), design_shape=detector.design_shape(), target_shape=detector.target_shape(),
    config=config['regressor'], rngs=nnx.Rngs(key_init), restore=restore
  )

  reg_coef = config['regressor'].get('regularization', 1.0e-2)

  @jax.jit
  def loss_f(x, c, t, params, r_state):
    regressor = nnx.merge(regressor_def, params, r_state)
    p = regressor(x, c, deterministic=False)
    reg = regressor.regularization()

    _, _, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)
    return jnp.mean(detector.loss(t, p)) + reg_coef * reg, r_state

  @jax.jit
  def metric_f(x, c, t, params, r_state):
    regressor = nnx.merge(regressor_def, params, r_state)
    p = regressor(x, c, deterministic=True)
    return jnp.mean(detector.metric(t, p))

  @jax.jit
  def step(x, c, t, parameters, r_state, state):
    (loss, r_state), grad = jax.value_and_grad(loss_f, argnums=3, has_aux=True)(x, c, t, parameters, r_state)
    updates, state = optimizer.update(grad, state)
    parameters = optax.apply_updates(parameters, updates)
    return loss, parameters, r_state, state

  training_losses = np.ndarray(shape=(epochs, steps))
  validation_losses = np.ndarray(shape=(epochs, validation_batches))

  if starting_epoch > 0:
    aux = detopt.utils.io.restore_aux(checkpointer)
    training_losses[:starting_epoch] = aux['training'][:starting_epoch]
    validation_losses[:starting_epoch] = aux['validation'][:starting_epoch]

  status = detopt.utils.progress.status_bar(disable=not progress)

  for i in status.epochs(starting_epoch, epochs):
    for j in status.training(steps):
      design_batch = design[None, :] + design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=(seed, i, j, 0), configurations=design_batch)

      rng, key_step = jax.random.split(rng, num=2)
      training_losses[i, j], parameters, regressor_state, optimizer_state = step(
        measurements, design_batch, target,
        parameters, regressor_state, optimizer_state
      )

    for j in status.validation(validation_batches):
      design_batch = design[None, :] + design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=(seed, i, j, 1), configurations=design_batch)

      validation_losses[i, j] = metric_f(measurements, design_batch, target, parameters, regressor_state)

    detopt.utils.io.save_state(
      i, checkpointer, design=design,
      regressor_parameters=parameters, regressor_state=regressor_state, regressor_optimizer_state=optimizer_state,
      aux={'training': training_losses, 'validation': validation_losses}
    )

  checkpointer.close()

def report(checkpoint, report):
  import matplotlib.pyplot as plt

  manager = detopt.utils.io.get_checkpointer(checkpoint)

  aux = detopt.utils.io.restore_aux(manager)

  training_losses = aux['training']
  validation_losses = aux['validation']

  fig = plt.figure(figsize=(9, 12))
  axes = fig.subplots(2, 1)

  detopt.utils.viz.losses.plot(training_losses, axes[0])
  axes[0].set_title('training')
  detopt.utils.viz.losses.plot(validation_losses, axes[1])
  axes[1].set_title('validation')

  fig.tight_layout()
  fig.savefig(report)
  plt.close(fig)

if __name__ == '__main__':
  import gearup

  gearup.gearup(train=regress, report=report, info=info).with_config('config/regress.yaml')()