
import os

import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx
import optax

import matplotlib
matplotlib.use('AGG')

import detopt

MAX_INT = 9223372036854775807

def optimize(seed, regressor_checkpoint, discriminator_checkpoint, progress=True, trace=None, report=None, **config):
  print(f'using {config.get("regressor")} as regressor')
  print(f'using {config.get("discriminator")} as discriminator')

  np_rng = np.random.default_rng(seed=seed)
  get_seed = lambda: np_rng.integers(low=0, high=MAX_INT)
  rng = jax.random.PRNGKey(get_seed())

  rng, key_regressor, key_discriminator = jax.random.split(rng, num=3)

  detector = detopt.detector.from_config(config['detector'])

  regressor_def, regressor_parameters, regressor_state = detopt.utils.io.load_model(
    'regressor', regressor_checkpoint, detector, config, rngs=nnx.Rngs(key_regressor)
  )
  discriminator_def, discriminator_parameters, discriminator_state = detopt.utils.io.load_model(
    'discriminator', discriminator_checkpoint, detector, config, rngs=nnx.Rngs(key_discriminator)
  )

  with open(config['initial_design'], 'r') as f:
    import json
    design = detector.encode_design(json.load(f))

  print(f'using {design} as initial design')
  print(f'using {detector.decode_design(design)} as initial design')
  design_optimizer = detopt.utils.config.optimizer(config['optimizer'])
  design_optimizer_state = design_optimizer.init(design)

  epochs, steps, validation_batches = config['epochs'], config['steps'], config['validation_batches']
  batch = config['batch']

  reg_coef = config.get('regularization', 1.0e-4)

  metric_names = detector.metric_names()

  @jax.jit
  def loss_f(x, c, t, r_params, r_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)
    p = regressor(x, c, deterministic=False)

    loss = jnp.mean(detector.loss(t, p)) + reg_coef * regressor.regularization()

    _, _, r_state = nnx.split(regressor, nnx.Param, nnx.Variable)
    return loss, r_state

  @jax.jit
  def metric_f(x, c, t, r_params, r_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)
    p = regressor(x, c, deterministic=True)

    metric = detector.metric(t, p)

    return metric

  @jax.jit
  def log_P_x_given_c(x, c, y, d_params, d_state):
    ### log P(x, y | c) - log P(c)
    discriminator = nnx.merge(discriminator_def, d_params, d_state)
    return discriminator(x, c, y)

  @jax.jit
  def loss_design(x, c, y, r_params, r_state, d_params, d_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)

    p_t = regressor(x, c)
    ### loss due to change of the optimal regressor
    loss_reg = detector.loss(y, p_t)

    ### loss due to change of the distribution of x
    loss_gen = log_P_x_given_c(x, c, y, d_params, d_state) * jax.lax.stop_gradient(loss_reg - 0.5)

    return jnp.mean(loss_reg) + jnp.mean(loss_gen)

  @jax.jit
  def step_design(design, x, c, y, r_params, r_state, d_params, d_state, opt_state):
    loss, grad = jax.value_and_grad(loss_design, argnums=1)(
      x, c, y, r_params, r_state, d_params, d_state
    )

    grad = jax.tree.map(lambda g: jnp.mean(g, axis=0), grad)

    updates, opt_state = design_optimizer.update(grad, opt_state)
    design = optax.apply_updates(design, updates)
    return loss, design, opt_state

  design_losses = np.ndarray(shape=(epochs, steps, ))
  design_validation = {
    k: np.ndarray(shape=(epochs, validation_batches, batch))
    for k in metric_names
  }

  status_bar = detopt.utils.progress.status_bar(disable=not progress)

  for i in status_bar.epochs(epochs):
    for j in status_bar.training(steps):
      design_batch = np.broadcast_to(design[None], shape=(batch, *detector.design_shape()))
      measurements, target = detector(seed=get_seed(), configurations=design_batch)

      design_losses[i, j], design_updated, design_optimizer_state = step_design(
        design,
        measurements, design_batch, target,
        regressor_parameters, regressor_state,
        discriminator_parameters, discriminator_state,
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

    for j in status_bar.validation(validation_batches):
      design_batch = np.broadcast_to(design[None], shape=(batch, *detector.design_shape()))
      measurements, target = detector(seed=get_seed(), configurations=design_batch)

      metrics = metric_f(measurements, design_batch, target, regressor_parameters, regressor_state)
      for k in metric_names:
        design_validation[k][i, j] = metrics[k]

      aux = {
        'design': {
          'training': design_losses[:i + 1],
          'validation': {
            k: design_validation[k][:(i + 1)]
            for k in metric_names
          }
        },
      }

    if trace is not None:
      detopt.utils.io.save_design(detector, os.path.join(trace, f'design-{i:05d}.json'), design)

    if report is not None:
      import matplotlib.pyplot as plt
      os.makedirs(report, exist_ok=True)

      fig = plot(aux)
      fig.savefig(os.path.join(report, 'losses.png'))
      plt.close(fig)

def plot(aux):
  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(18, 12))
  axes = fig.subplots(2, 1, squeeze=False)

  detopt.utils.viz.losses.plot(aux['design']['training'], axes[0, 0])
  axes[0, 0].set_title('Design losses')

  detopt.utils.viz.losses.plot(aux['design']['validation'], axes[1, 0])
  axes[1, 0].set_title('Design validation')

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
  gearup.gearup(optimize=optimize, report=report).with_config('config/pre-lfi.yaml')()