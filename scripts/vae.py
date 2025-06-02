
import os

import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx
import optax

import detopt

MAX_INT = 9223372036854775807

def generate(seed, output, progress=True, restore=True, **config):
  print(f'using {config.get("regressor")} as regressor')
  print(f'using {config.get("generator")} as generator')
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

  design = restored['design']['design']
  print(f'using {design} as initial design')
  print(f'using {detector.decode_design(design)} as initial design')
  design_optimizer, design_optimizer_state = restored['design']['optimizer'], restored['design']['optimizer_state']

  regressor_def, regressor_optimizer = restored['regressor']['model'], restored['regressor']['optimizer']
  regressor_parameters, regressor_state =  restored['regressor']['parameters'],  restored['regressor']['state']
  regressor_optimizer_state = restored['regressor']['optimizer_state']

  generator_def, generator_optimizer = restored['generator']['model'], restored['generator']['optimizer']
  generator_parameters, generator_state = restored['generator']['parameters'], restored['generator']['state']
  generator_optimizer_state = restored['generator']['optimizer_state']

  discriminator_def, discriminator_optimizer = restored['discriminator']['model'], restored['discriminator']['optimizer']
  discriminator_parameters, discriminator_state = restored['discriminator']['parameters'], restored['discriminator']['state']
  discriminator_optimizer_state = restored['discriminator']['optimizer_state']

  design_eps = float(config['eps'])
  likelihood_sigma = float(config['likelihood_sigma'])

  epochs, steps, substeps = config['epochs'], config['steps'], config['substeps']
  batch, validation_batches = config['batch'], config['validation_batches']

  @jax.jit
  def combine(x_real, c_real, x_gen, c_gen):
    n_real, *_ = x_real.shape
    n_gen, *_ = x_gen.shape

    x = jnp.concatenate([x_real, x_gen], axis=0)
    c = jnp.concatenate([c_real, c_gen], axis=0)
    y = jnp.concatenate([
      jnp.ones(shape=(n_real, ), dtype=x_real.dtype),
      jnp.zeros(shape=(n_gen,), dtype=x_gen.dtype),
    ], axis=0)

    return x, c, y

  @jax.jit
  def discriminator_loss_f(x_real, c_real, x_gen, c_gen, params, d_state):
    discriminator = nnx.merge(discriminator_def, params, d_state)
    x, c, y = combine(x_real, c_real, x_gen, c_gen)

    p = discriminator(x, c, deterministic=False)
    loss = jnp.mean(y * jax.nn.softplus(-p) + (1 - y) * jax.nn.softplus(p))
    loss += 1.0e-4 * discriminator.regularization()
    loss += 1.0e-1 * jnp.mean(jnp.square(p))

    _, _, d_state = nnx.split(discriminator, nnx.Param, nnx.Variable)

    return loss, d_state

  @jax.jit
  def discriminator_metric_f(x_real, c_real, x_gen, c_gen, params, d_state):
    discriminator = nnx.merge(discriminator_def, params, d_state)
    x, c, y = combine(x_real, c_real, x_gen, c_gen)
    p = discriminator(x, c, deterministic=True)

    return (p > 0.0) == (y > 0.5)

  @jax.jit
  def step_discriminator(x_real, c_real, x_gen, c_gen, parameters, d_state, opt_state):
    (loss, d_state), grad = jax.value_and_grad(discriminator_loss_f, argnums=4, has_aux=True)(
      x_real, c_real, x_gen, c_gen, parameters, d_state
    )
    updates, opt_state = discriminator_optimizer.update(grad, opt_state)
    parameters = optax.apply_updates(parameters, updates)
    return loss, parameters, d_state, opt_state

  @jax.jit
  def generator_loss_f(key, x, c, g_params, g_state, d_params, d_state):
    key_elbo, key_gan = jax.random.split(key, num=2)

    vae = nnx.merge(generator_def, g_params, g_state)
    discriminator = nnx.merge(discriminator_def, d_params, d_state)

    mean, sigma = vae.encode(x, c)
    latent = mean + sigma * jax.random.normal(key_elbo, shape=sigma.shape, dtype=sigma.dtype)
    x_reco = vae.decode(latent, c)
    loss = detopt.nn.elbo(x, x_reco, mean, sigma, sigma_reconstructed=likelihood_sigma)

    # mean, sigma = vae.encode(x, c)
    # x_reco = vae.decode(mean, c)
    # loss = jnp.mean(x * jax.nn.softplus(-x_reco) + (1 - x) * jax.nn.softplus(x_reco))

    # n_b, *_ = x.shape
    # latent = jax.random.normal(key_elbo, shape=(n_b, vae.latent_dim), dtype=x.dtype)
    # x_gen = vae.decode(latent, c)
    # p = discriminator(x_gen, c, deterministic=False)
    # loss = -jnp.mean(jax.nn.softplus(p))

    _, _, g_state = nnx.split(vae, nnx.Param, nnx.Variable)


    return jnp.mean(loss), g_state

  @jax.jit
  def step_generator(key, x, c, g_parameters, g_state, d_parameters, d_state, opt_state):
    (loss, g_state), grad = jax.value_and_grad(generator_loss_f, argnums=3, has_aux=True)(
      key, x, c, g_parameters, g_state, d_parameters, d_state,
    )
    updates, opt_state = generator_optimizer.update(grad, opt_state)
    parameters = optax.apply_updates(g_parameters, updates)
    return loss, parameters, g_state, opt_state

  @jax.jit
  def sample(key, design, parameters, g_state):
    vae = nnx.merge(generator_def, parameters, g_state)
    key_sample, key_noise = jax.random.split(key, num=2)
    clean_sample = vae.sample(key, design)
    eps = jax.random.normal(key_noise, shape=clean_sample.shape, dtype=clean_sample.dtype)
    sample = clean_sample + likelihood_sigma * eps
    return jax.nn.sigmoid(clean_sample)

  generator_losses = np.ndarray(shape=(epochs, steps))
  discriminator_losses = np.ndarray(shape=(epochs, steps, substeps))

  generator_validation = np.ndarray(shape=(epochs, validation_batches, 2 * batch))

  aux: dict | None = restored['aux']
  if aux is not None:
    generator_losses[:starting_epoch] = aux['generator_training'][:starting_epoch]
    discriminator_losses[:starting_epoch] = aux['discriminator_training'][:starting_epoch]

    generator_validation[:starting_epoch] = aux['generator_validation'][:starting_epoch]

  status = detopt.utils.progress.status_bar(disable=not progress)

  for i in status.epochs(starting_epoch, epochs):
    for j in status.training(steps):
      for k in range(substeps):
        design_batch_real = design[None, :] + \
                            design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
        measurements, target = detector(seed=get_seed(), configurations=design_batch_real)

        design_batch_gen = design[None, :] + \
                           design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
        rng, key_sample = jax.random.split(rng, num=2)
        generated = sample(key_sample, design_batch_gen, generator_parameters, generator_state)

        discriminator_losses[i, j, k], discriminator_parameters, discriminator_state, discriminator_optimizer_state = \
          step_discriminator(
            measurements, design_batch_real, generated, design_batch_gen,
            discriminator_parameters, discriminator_state, discriminator_optimizer_state
          )

      design_batch = design[None, :] + \
                         design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=get_seed(), configurations=design_batch)
      rng, key_step = jax.random.split(rng, num=2)

      generator_losses[i, j], generator_parameters, generator_state, generator_optimizer_state = step_generator(
        key_step, measurements, design_batch,
        generator_parameters, generator_state, discriminator_parameters, discriminator_state,
        generator_optimizer_state
      )

    for j in status.validation(validation_batches):
      design_batch_real = design[None, :] + \
                          design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=get_seed(), configurations=design_batch_real)

      design_batch_gen = design[None, :] + \
                         design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      rng, key_sample = jax.random.split(rng, num=2)
      generated = sample(key_sample, design_batch_gen, generator_parameters, generator_state)

      generator_validation[i, j] = discriminator_metric_f(
        measurements, design_batch_real, generated, design_batch_gen,
        discriminator_parameters, discriminator_state
      )

    detopt.utils.io.save_state(
      i, checkpointer, design=design, design_optimizer_state=design_optimizer_state,

      regressor_parameters=regressor_parameters, regressor_state=regressor_state,
      regressor_optimizer_state=regressor_optimizer_state,

      generator_parameters=generator_parameters, generator_state=generator_state,
      generator_optimizer_state=generator_optimizer_state,

      discriminator_parameters=discriminator_parameters, discriminator_state=discriminator_state,
      discriminator_optimizer_state=discriminator_optimizer_state,

      aux={
        'generator_training': generator_losses[:i + 1],
        'discriminator_training': discriminator_losses[:i + 1],
        'generator_validation': generator_validation[:i + 1]
      }
    )

  checkpointer.close()

def report(seed, checkpoint, report, **config):
  import matplotlib.pyplot as plt
  os.makedirs(report, exist_ok=True)

  rng = jax.random.PRNGKey(seed)
  np_rng = np.random.default_rng(seed + 1)
  rng, key_init = jax.random.split(rng, num=2)

  checkpointer = detopt.utils.io.get_checkpointer(checkpoint)
  detector = detopt.detector.from_config(config['detector'])

  rng, key_init = jax.random.split(rng, num=2)
  restored = detopt.utils.io.restore_state(
    checkpointer, detector, config, rngs=nnx.Rngs(key_init), restore=True
  )

  aux = restored['aux']

  fig = plt.figure(figsize=(9, 18))
  axes = fig.subplots(3, 1)

  detopt.utils.viz.losses.plot(aux['generator_training'], axes[0])
  axes[0].set_title('Generator losses')
  detopt.utils.viz.losses.plot(aux['discriminator_training'], axes[1])
  axes[1].set_title('Discriminator losses')
  detopt.utils.viz.losses.plot(aux['generator_validation'], axes[2])
  axes[2].set_title('Generator validation')

  fig.tight_layout()
  fig.savefig(os.path.join(report, 'losses.png'))
  plt.close(fig)

  design = restored['design']['design']
  design_eps = config['eps']

  generator_def, generator_optimizer = restored['generator']['model'], restored['generator']['optimizer']
  generator_parameters, generator_state = restored['generator']['parameters'], restored['generator']['state']

  n, m = 3, 5
  batch = n * m

  vae = nnx.merge(generator_def, generator_parameters, generator_state)
  key_sample, key_noise = jax.random.split(rng, num=2)

  eps = np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
  designs = design[None, :] + design_eps * eps
  measurements, _ = detector(seed=(seed, seed), configurations=designs)
  measurements = np.reshape(measurements, shape=(n, m, *measurements.shape[1:]))

  clean_sample = vae.sample(key_sample, designs)
  clean_sample = jnp.reshape(clean_sample, shape=(n, m, *clean_sample.shape[1:]))

  fig = plt.figure(figsize=(2 * n, 2 * m))
  axes = fig.subplots(2 * n, m, squeeze=False)

  for i in range(n):
    for j in range(m):
      im = axes[i, j].imshow(measurements[i, j].T)
      plt.colorbar(im, ax=axes[i, j])
      axes[i, j].set_title('real')
      im = axes[i + n, j].imshow(clean_sample[i, j].T)
      plt.colorbar(im, ax=axes[i + n, j])
      axes[i + n, j].set_title('generated')

  fig.tight_layout()
  fig.savefig(os.path.join(report, 'samples.png'))
  plt.close(fig)



if __name__ == '__main__':
  import gearup

  gearup.gearup(generate=generate, report=report).with_config('config/vae.yaml')()