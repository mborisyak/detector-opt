
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

def optimize(seed, output, progress=True, restore=True, trace=None, report=None, **config):
  print(f'using {config.get("regressor")} as regressor')
  print(f'using {config.get("discriminator")} as discriminator')

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
  warmup = config.get('warmup', 0)

  design = restored['design']['design']
  print(f'using {design} as initial design')
  print(f'using {detector.decode_design(design)} as initial design')
  design_optimizer, design_optimizer_state = restored['design']['optimizer'], restored['design']['optimizer_state']

  regressor_def, regressor_optimizer = restored['regressor']['model'], restored['regressor']['optimizer']
  regressor_parameters, regressor_state =  restored['regressor']['parameters'],  restored['regressor']['state']
  regressor_optimizer_state = restored['regressor']['optimizer_state']

  discriminator_def, discriminator_optimizer = restored['discriminator']['model'], restored['discriminator']['optimizer']
  discriminator_parameters, discriminator_state = restored['discriminator']['parameters'], restored['discriminator']['state']
  discriminator_optimizer_state = restored['discriminator']['optimizer_state']

  design_eps = float(config['design_eps'])

  epochs, steps, substeps = config['epochs'], config['steps'], config['substeps']
  batch, validation_batches = config['batch'], config['validation_batches']

  reg_coef = config.get('regularization', 1.0e-4)
  discr_reg_coef = config.get('discriminator_regularization', 1.0e-2)

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
    p = regressor(x, c, deterministic=False)

    metric = detector.metric(t, p)

    return metric

  @jax.jit
  def step_regressor(x, c, t, r_params, r_state, opt_state):
    (loss, r_state), grad = jax.value_and_grad(loss_f, argnums=3, has_aux=True)(x, c, t, r_params, r_state)
    updates, opt_state = regressor_optimizer.update(grad, opt_state)
    r_params = optax.apply_updates(r_params, updates)
    return loss, r_params, r_state, opt_state

  @jax.jit
  def combine(x_real, c_real, gt_real, x_gen, c_gen, gt_gen):
    n_real, *_ = x_real.shape
    n_gen, *_ = x_gen.shape

    x = jnp.concatenate([x_real, x_gen], axis=0)
    c = jnp.concatenate([c_real, c_gen], axis=0)
    gt = jnp.concatenate([gt_real, gt_gen], axis=0)

    y = jnp.concatenate([
      jnp.ones(shape=(n_real,), dtype=x_real.dtype),
      jnp.zeros(shape=(n_gen,), dtype=x_gen.dtype),
    ], axis=0)

    return x, c, gt, y

  @jax.jit
  def loss_discriminator_f(x_true, c_true, gt_true, x_gen, c_gen, gt_gen, d_params, d_state):
    x, c, gt, y = combine(x_true, c_true, gt_true, x_gen, c_gen, gt_gen)

    discriminator = nnx.merge(discriminator_def, d_params, d_state)

    p = discriminator(x, c, gt)

    cross_entropy = jnp.mean(y * jax.nn.softplus(-p) + (1 - y) * jax.nn.softplus(p))
    loss = cross_entropy + reg_coef * discriminator.regularization() + discr_reg_coef * jnp.mean(jnp.square(p))

    _, _, d_state = nnx.split(discriminator, nnx.Param, nnx.Variable)

    return loss, d_state

  @jax.jit
  def metric_discriminator_f(x_true, c_true, gt_true, x_gen, c_gen, gt_gen, d_params, d_state):
    x, c, gt, y = combine(x_true, c_true, gt_true, x_gen, c_gen, gt_gen)
    discriminator = nnx.merge(discriminator_def, d_params, d_state)
    p = discriminator(x, c, gt)
    metric = (p > 0.0) == (y > 0.5)

    return metric

  @jax.jit
  def step_discriminator(x_true, c_true, gt_true, x_gen, c_gen, gt_gen, d_params, d_state, opt_state):
    (loss, d_state), grad = jax.value_and_grad(loss_discriminator_f, argnums=6, has_aux=True)(
      x_true, c_true, gt_true, x_gen, c_gen, gt_gen, d_params, d_state
    )
    updates, opt_state = discriminator_optimizer.update(grad, opt_state)
    d_params = optax.apply_updates(d_params, updates)
    return loss, d_params, d_state, opt_state

  @jax.jit
  def log_P_x_given_c_gt(x, c, gt, d_params, d_state):
    ### log P(x | c, gt) - log P(x)
    discriminator = nnx.merge(discriminator_def, d_params, d_state)

    return discriminator(x, c, gt)

  @jax.jit
  def loss_design(x, c, gt, t, r_params, r_state, d_params, d_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)

    p_t = regressor(x, c)
    ### loss due to change of the optimal regressor
    loss_reg = detector.loss(t, p_t)

    ### loss due to change of the distribution of x
    loss_gen = log_P_x_given_c_gt(x, c, gt, d_params, d_state) * jax.lax.stop_gradient(loss_reg - 0.5)

    return jnp.mean(loss_reg) + jnp.mean(loss_gen)

  @jax.jit
  def step_design(design, x, c, gt, t, r_params, r_state, d_params, d_state, opt_state):
    loss, grad = jax.value_and_grad(loss_design, argnums=1)(
      x, c, gt, t, r_params, r_state, d_params, d_state
    )

    grad = jax.tree.map(lambda g: jnp.mean(g, axis=0), grad)

    updates, opt_state = design_optimizer.update(grad, opt_state)
    design = optax.apply_updates(design, updates)
    return loss, design, opt_state

  design_losses = np.ndarray(shape=(epochs, steps, substeps))

  regressor_losses = np.ndarray(shape=(epochs, steps, substeps))
  regressor_validation = np.ndarray(shape=(epochs, validation_batches, batch))

  discriminator_losses = np.ndarray(shape=(epochs, steps, substeps))
  discriminator_validation = np.ndarray(shape=(epochs, validation_batches, 2 * batch))

  aux: dict | None = restored['aux']
  if aux is not None:
    design_losses[:starting_epoch] = aux['design']['training'][:starting_epoch]

    regressor_losses[:starting_epoch] = aux['regressor']['training'][:starting_epoch]
    regressor_validation[:starting_epoch] = aux['regressor']['validation'][:starting_epoch]

    discriminator_losses[:starting_epoch] = aux['discriminator']['training'][:starting_epoch]
    discriminator_validation[:starting_epoch] = aux['discriminator']['validation'][:starting_epoch]

  status = detopt.utils.progress.status_bar(disable=not progress)

  @jax.jit
  def check(params):
    return jnp.all(
      jnp.array([
        jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)
      ])
    )

  def sample(rng_seed, c):
    design_noise = design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
    c_batch = c[None, :] + design_noise

    gt, x, t = detector(seed=rng_seed, configurations=c_batch)
    return gt, x, c_batch, t

  def sample_independent(rng_seed_1, rng_seed_2, c):
    _, x, _, _ = sample(rng_seed_1, c)
    gt, _, c_batch, _ = sample(rng_seed_2, c)

    return gt, x, c_batch


  for i in status.epochs(starting_epoch, epochs):
    for j in status.training(steps):
      for k in range(substeps):
        ground_truth, measurements, design_batch, target = sample(get_seed(), design)

        regressor_losses[i, j, k], regressor_parameters, regressor_state, regressor_optimizer_state = \
          step_regressor(
            measurements, design_batch, target,
            regressor_parameters, regressor_state, regressor_optimizer_state
          )
        if not check(regressor_parameters):
          raise ValueError()

        ground_truth_pseudo, measurements_pseudo, design_batch_pseudo = sample_independent(
          get_seed(), get_seed(), design
        )

        discriminator_losses[i, j, k], discriminator_parameters, discriminator_state, discriminator_optimizer_state = \
          step_discriminator(
            measurements, design_batch, ground_truth,
            measurements_pseudo, design_batch_pseudo, ground_truth_pseudo,
            discriminator_parameters, discriminator_state, discriminator_optimizer_state
          )
        if not check(discriminator_parameters):
          raise ValueError()

      design_batch = np.broadcast_to(design[None], shape=(batch, *detector.design_shape()))
      ground_truth, measurements, target = detector(seed=get_seed(), configurations=design_batch)

      if i >= warmup:
        design_losses[i, j], design_updated, design_optimizer_state = step_design(
          design,
          measurements, design_batch, ground_truth, target,
          regressor_parameters, regressor_state,
          discriminator_parameters, discriminator_state,
          design_optimizer_state
        )
      else:
        design_updated = design
        design_losses[i, j] = 0.0

      if not np.all(np.isfinite(design_updated)):
        print('oriignal:', design)
        print('updated:', design_updated)
        print('measurements', np.all(np.isfinite(measurements)), np.max(measurements))
        print('target', np.all(np.isfinite(target)), np.max(target))
        raise ValueError()
      else:
        design = design_updated

    for j in status.validation(validation_batches):
      ground_truth, measurements, design_batch, target = sample(get_seed(), design)
      regressor_validation[i, j] = metric_f(measurements, design_batch, target, regressor_parameters, regressor_state)

      ground_truth_pseudo, measurements_pseudo, design_batch_pseudo = sample_independent(
        get_seed(), get_seed(), design
      )

      discriminator_validation[i, j] = metric_discriminator_f(
        measurements, design_batch, ground_truth,
        measurements_pseudo, design_batch_pseudo, ground_truth_pseudo,
        discriminator_parameters, discriminator_state
      )

    aux = {
      'design': {'training': design_losses[:i + 1]},
      'regressor': {'training': regressor_losses[:i + 1], 'validation': regressor_validation[:i + 1]},
      'discriminator': {'training': discriminator_losses[:i + 1], 'validation': discriminator_validation[:i + 1]},
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
  fig = plt.figure(figsize=(18, 12))
  axes = fig.subplots(3, 2, squeeze=False)

  detopt.utils.viz.losses.plot(aux['design']['training'], axes[0, 0])
  axes[0, 0].set_title('Design losses')

  detopt.utils.viz.losses.plot(aux['regressor']['training'], axes[1, 0])
  axes[1, 0].set_title('Regressor losses')
  detopt.utils.viz.losses.plot(aux['regressor']['validation'], axes[1, 1])
  axes[1, 1].set_title('Regressor validation')

  detopt.utils.viz.losses.plot(aux['discriminator']['training'], axes[2, 0])
  axes[2, 0].set_title('Discriminator losses')
  detopt.utils.viz.losses.plot(aux['discriminator']['validation'], axes[2, 1])
  axes[2, 1].set_title('Discriminator validation')

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
  gearup.gearup(optimize=optimize, report=report).with_config('config/lfi.yaml')()