
import os

import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx
import optax

import detopt

import matplotlib
matplotlib.use('AGG')

MAX_INT = 9223372036854775807

def optimize(seed, output, progress=True, restore=True, trace=None, report=None, **config):
  print(f'using {config.get("regressor")} as regressor')

  np_rng = np.random.default_rng(seed=seed)
  get_seed = lambda: np_rng.integers(low=0, high=MAX_INT)
  rng = jax.random.PRNGKey(get_seed())

  checkpointer = detopt.utils.io.get_checkpointer(output)
  if checkpointer.latest_step() is not None and checkpointer.latest_step() >= config['epochs']:
    return

  detector = detopt.detector.from_config(config['detector'])

  rng, key_init = jax.random.split(rng, num=2)
  restored = detopt.utils.io.restore_state(
    checkpointer, detector, config, rngs=nnx.Rngs(key_init), restore=restore
  )

  starting_epoch = restored['starting_epoch']

  design = restored['design']['design']
  print(f'using {design} as initial design')
  print(f'using {detector.decode_design(design)} as initial design')
  design_optimizer, design_optimizer_state = restored['design']['optimizer'], restored['design']['optimizer_state']

  regressor_def, regressor_optimizer = restored['regressor']['model'], restored['regressor']['optimizer']
  regressor_parameters, regressor_state =  restored['regressor']['parameters'],  restored['regressor']['state']
  regressor_optimizer_state = restored['regressor']['optimizer_state']

  design_eps = float(config['design_eps'])

  epochs, steps, substeps = config['epochs'], config['steps'], config['substeps']
  batch, validation_batches = config['batch'], config['validation_batches']

  reg_coef = config.get('regularization', 1.0e-4)

  @jax.jit
  def loss_f(x, c, t, r_params, r_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)
    p = regressor(x, c, deterministic=False)

    loss = jnp.mean(detector.loss(t, p)) + reg_coef * regressor.regularization() + 1.0e-2 * jnp.mean(jnp.square(p))

    _, _, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)
    return loss, r_state

  @jax.jit
  def metric_f(x, c, t, r_params, r_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)
    p = regressor(x, c, deterministic=True)

    metric = detector.metric(t, p)

    return metric

  @jax.jit
  def step_regressor(x, c, t, r_params, r_state, opt_state):
    (loss, r_state), grad = jax.value_and_grad(loss_f, argnums=3, has_aux=True)(x, c, t, r_params, r_state)
    updates, opt_state = regressor_optimizer.update(grad, opt_state)
    r_params = optax.apply_updates(r_params, updates)
    grad_check = jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grad)
    return loss, r_params, r_state, opt_state, grad_check

  @jax.jit
  def step_design(design, x, c, t, r_params, r_state, opt_state):
    (loss, _), grad = jax.value_and_grad(loss_f, argnums=1, has_aux=True)(x, c, t, r_params, r_state)

    grad = jax.tree.map(lambda g: jnp.mean(g, axis=0), grad)

    updates, opt_state = design_optimizer.update(grad, opt_state)
    design = optax.apply_updates(design, updates)
    return loss, design, opt_state

  regressor_losses = np.ndarray(shape=(epochs, steps, substeps))
  regressor_validation = np.ndarray(shape=(epochs, validation_batches, batch))

  aux: dict | None = restored['aux']
  if aux is not None:
    regressor_losses[:starting_epoch] = aux['regressor']['training'][:starting_epoch]
    regressor_validation[:starting_epoch] = aux['regressor']['validation'][:starting_epoch]

  status = detopt.utils.progress.status_bar(disable=not progress)

  @jax.jit
  def check(params, state):
    return jnp.all(
      jnp.array([
        jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)
      ])
    )

  for i in status.epochs(starting_epoch, epochs):
    for j in status.training(steps):
      for k in range(substeps):
        design_batch = design[None, :] + \
                            design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
        _, measurements, target = detector(seed=get_seed(), configurations=design_batch)

        regressor_losses[i, j, k], regressor_parameters, regressor_state, regressor_optimizer_state, grad_check = \
          step_regressor(
            measurements, design_batch, target,
            regressor_parameters, regressor_state, regressor_optimizer_state
          )

        if not check(regressor_parameters, regressor_state):
          print('measurements', np.min(measurements), np.max(measurements))
          print('target', np.min(target), np.max(target))
          print(jax.tree.map(
            lambda x: jnp.all(jnp.isfinite(x)),
            regressor_parameters
          ))
          print(grad_check)
          raise ValueError()

      design_batch = np.broadcast_to(design[None], shape=(batch, *detector.design_shape()))
      _, measurements, target = detector(seed=get_seed(), configurations=design_batch)

      _, design_updated, design_optimizer_state = step_design(
        design,
        measurements, design_batch, target,
        regressor_parameters, regressor_state,
        design_optimizer_state
      )

      if not np.all(np.isfinite(design_updated)):
        print('oriignal:', design)
        print('updated:', design_updated)
        print('measurements', np.all(np.isfinite(measurements)), np.max(measurements))
        print('target', np.all(np.isfinite(target)), np.max(target))
        raise ValueError()
      else:
        design = design_updated

    for j in status.validation(validation_batches):
      design_batch = design[None, :] + \
                     design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      _, measurements, target = detector(seed=get_seed(), configurations=design_batch)

      regressor_validation[i, j] = metric_f(measurements, design_batch, target, regressor_parameters, regressor_state)

    aux = {
      'regressor': {
        'training': regressor_losses[:i + 1],
        'validation': regressor_validation[:i + 1]
      }
    }

    detopt.utils.io.save_state(
      i, checkpointer, design=design, design_optimizer_state=design_optimizer_state,

      regressor_parameters=regressor_parameters, regressor_state=regressor_state,
      regressor_optimizer_state=regressor_optimizer_state,
      aux=aux
    )

    if trace is not None:
      detopt.utils.io.save_design(detector, os.path.join(trace, f'design-{i:05d}.json'), design)

    if report is not None:
      import matplotlib.pyplot as plt
      os.makedirs(report, exist_ok=True)

      fig = plot(aux)
      fig.savefig(os.path.join(report, 'losses.png'))
      plt.close(fig)

  checkpointer.close()

def plot(aux):
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(9, 12))
  axes = fig.subplots(2, 1)

  detopt.utils.viz.losses.plot(aux['regressor']['training'], axes[0])
  axes[0].set_title('Regressor losses')
  detopt.utils.viz.losses.plot(aux['regressor']['validation'], axes[1])
  axes[1].set_title('Regressor validation')

  fig.tight_layout()
  return fig


def report(seed, checkpoint, report, **config):
  import matplotlib.pyplot as plt
  os.makedirs(report, exist_ok=True)

  rng = jax.random.PRNGKey(seed)

  checkpointer = detopt.utils.io.get_checkpointer(checkpoint)
  detector = detopt.detector.from_config(config['detector'])

  restored = detopt.utils.io.restore_state(
    checkpointer, detector, config, rngs=nnx.Rngs(rng), restore=True
  )

  aux = restored['aux']

  fig = plot(aux)
  fig.savefig(os.path.join(report, 'losses.png'))
  plt.close(fig)



if __name__ == '__main__':
  import gearup
  gearup.gearup(optimize=optimize, report=report).with_config('config/subgradient.yaml')()