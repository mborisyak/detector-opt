
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

def regressor(seed, output, dataset=None, progress=True, restore=True, report=None, **config):
  print(f'using {config.get("regressor")} as regressor')

  np_rng = np.random.default_rng(seed=seed)
  get_seed = lambda: np_rng.integers(low=0, high=MAX_INT)
  rng = jax.random.PRNGKey(get_seed())

  checkpointer = detopt.utils.io.get_checkpointer(output)
  if checkpointer.latest_step() is not None and checkpointer.latest_step() >= config['regression']['epochs']:
    return

  detector = detopt.detector.from_config(config['detector'])
  metric_names = detector.metric_names()

  rng, key_init = jax.random.split(rng, num=2)
  restored = detopt.utils.io.restore_state(
    checkpointer, detector, config, rngs=nnx.Rngs(key_init), restore=restore
  )

  starting_epoch = restored['starting_epoch']

  regressor_def, regressor_optimizer = restored['regressor']['model'], restored['regressor']['optimizer']
  regressor_parameters, regressor_state =  restored['regressor']['parameters'],  restored['regressor']['state']
  regressor_optimizer_state = restored['regressor']['optimizer_state']

  design_eps = float(config['design_eps'])

  epochs, steps = config['regression']['epochs'], config['regression']['steps']
  batch, validation_batches = config['regression']['batch'], config['regression']['validation_batches']

  reg_coef = config.get('regularization', 1.0e-4)

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
  def step_regressor(x, c, t, r_params, r_state, opt_state):
    (loss, r_state), grad = jax.value_and_grad(loss_f, argnums=3, has_aux=True)(x, c, t, r_params, r_state)
    updates, opt_state = regressor_optimizer.update(grad, opt_state)
    r_params = optax.apply_updates(r_params, updates)
    grad_check = jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grad)
    return loss, r_params, r_state, opt_state, grad_check

  regressor_losses = np.ndarray(shape=(epochs, steps))
  regressor_validation = {
    k: np.ndarray(shape=(epochs, validation_batches, batch))
    for k in metric_names
  }

  aux: dict | None = restored['aux']
  if aux is not None:
    regressor_losses[:starting_epoch] = aux['regressor']['training'][:starting_epoch]
    for k in metric_names:
      regressor_validation[k][:starting_epoch] = aux['regressor']['validation'][k][:starting_epoch]

  size = steps * batch
  val_size = validation_batches * batch

  if restore and dataset is not None:
    try:
      data = np.load(dataset)
    except FileNotFoundError:
      data = None
  else:
    data = None

  if progress:
    from tqdm import tqdm
    progress_bar = tqdm
  else:
    progress_bar = lambda x, *args, **kwargs: x

  if data is None:
    c = np.ndarray(shape=(size, *detector.design_shape()), dtype=np.float32)
    X = np.ndarray(shape=(size, *detector.output_shape()), dtype=np.float32)
    y = np.ndarray(shape=(size, *detector.target_shape()), dtype=np.float32)

    c_val = np.ndarray(shape=(val_size, *detector.design_shape()), dtype=np.float32)
    X_val = np.ndarray(shape=(val_size, *detector.output_shape()), dtype=np.float32)
    y_val = np.ndarray(shape=(val_size, *detector.target_shape()), dtype=np.float32)

    for i in progress_bar(range(steps), desc='sampling training set'):
      c_batch = design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      X_batch, y_batch = detector(get_seed(), configurations=c_batch)

      c[i * batch:(i + 1) * batch] = c_batch
      X[i * batch:(i + 1) * batch] = X_batch
      y[i * batch:(i + 1) * batch] = y_batch

    for i in progress_bar(range(validation_batches), desc='sampling test set'):
      c_batch = design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      X_batch, y_batch = detector(get_seed(), configurations=c_batch)

      c_val[i * batch:(i + 1) * batch] = c_batch
      X_val[i * batch:(i + 1) * batch] = X_batch
      y_val[i * batch:(i + 1) * batch] = y_batch

    if dataset is not None:
      np.savez(
        dataset,
        condition=c, samples=X, target=y,
        condition_validation=c_val, samples_validation=X_val, target_validation=y_val
      )

  else:
    c, X, y = data['condition'], data['samples'], data['target']
    c_val, X_val, y_val = data['condition_validation'], data['samples_validation'], data['target_validation']

  assert X.shape[0] == size
  assert X_val.shape[0] == val_size

  status = detopt.utils.progress.status_bar(disable=not progress)

  for i in status.epochs(starting_epoch, epochs):
    for j in status.training(steps):
      batch_index = np_rng.integers(0, X.shape[0], size=(batch, ))
      c_batch = c[batch_index]
      X_batch = X[batch_index]
      y_batch = y[batch_index]

      regressor_losses[i, j], regressor_parameters, regressor_state, regressor_optimizer_state, grad_check = \
        step_regressor(X_batch, c_batch, y_batch, regressor_parameters, regressor_state, regressor_optimizer_state)

    for j in status.validation(validation_batches):
      c_batch = c[j * batch:(j + 1) * batch]
      X_batch = X[j * batch:(j + 1) * batch]
      y_batch = y[j * batch:(j + 1) * batch]

      metrics = metric_f(X_batch, c_batch, y_batch, regressor_parameters, regressor_state)
      for k in metric_names:
        regressor_validation[k][i, j] = metrics[k]

    aux = {
      'regressor': {
        'training': regressor_losses[:i + 1],
        'validation': {
          k : regressor_validation[k][:i + 1] for k in metric_names
        }
      }
    }

    detopt.utils.io.save_state(
      i, checkpointer, design=restored['design']['design'], design_optimizer_state=None,

      regressor_parameters=regressor_parameters, regressor_state=regressor_state,
      regressor_optimizer_state=regressor_optimizer_state,
      aux=aux
    )

    if report is not None:
      import matplotlib.pyplot as plt
      os.makedirs(report, exist_ok=True)

      fig = plot(aux)
      fig.savefig(os.path.join(report, 'losses.png'))
      plt.close(fig)

  checkpointer.close()

def discriminator(seed, output, dataset=None, progress=True, restore=True, report=None, **config):
  print(f'using {config.get("discriminator")} as discriminator')

  np_rng = np.random.default_rng(seed=seed)
  get_seed = lambda: np_rng.integers(low=0, high=MAX_INT)
  rng = jax.random.PRNGKey(get_seed())

  checkpointer = detopt.utils.io.get_checkpointer(output)
  if checkpointer.latest_step() is not None and checkpointer.latest_step() >= config['training']['epochs']:
    return

  detector = detopt.detector.from_config(config['detector'])
  metric_names = detector.metric_names()

  rng, key_init = jax.random.split(rng, num=2)
  restored = detopt.utils.io.restore_state(
    checkpointer, detector, config, rngs=nnx.Rngs(key_init), restore=restore
  )

  starting_epoch = restored['starting_epoch']

  discriminator_def, discriminator_optimizer = restored['discriminator']['model'], restored['discriminator']['optimizer']
  discriminator_parameters, discriminator_state =  restored['discriminator']['parameters'],  restored['discriminator']['state']
  discriminator_optimizer_state = restored['discriminator']['optimizer_state']

  base_design = np.zeros(shape=detector.design_shape(), dtype=np.float32)
  design_eps = float(config['design_eps'])

  epochs, steps = config['training']['epochs'], config['training']['steps']
  batch, validation_batches = config['training']['batch'], config['training']['validation_batches']

  reg_coef = config.get('regularization', 1.0e-3)
  discr_reg_coef = config.get('regularization', 1.0e-3)

  @jax.jit
  def combine(x_real, c_real, y_real, x_gen, c_gen, y_gen):
    n_real, *_ = x_real.shape
    n_gen, *_ = x_gen.shape

    x = jnp.concatenate([x_real, x_gen], axis=0)
    c = jnp.concatenate([c_real, c_gen], axis=0)
    y = jnp.concatenate([y_real, y_gen], axis=0)

    labels = jnp.concatenate([
      jnp.ones(shape=(n_real,), dtype=x_real.dtype),
      jnp.zeros(shape=(n_gen,), dtype=x_gen.dtype),
    ], axis=0)

    return x, c, y, labels

  @jax.jit
  def loss_discriminator_f(x_true, c_true, y_true, x_gen, c_gen, y_gen, d_params, d_state):
    x_batch, c_batch, y_batch, labels = combine(x_true, c_true, y_true, x_gen, c_gen, y_gen)

    discriminator = nnx.merge(discriminator_def, d_params, d_state)

    p = discriminator(x_batch, c_batch, y_batch, deterministic=False)

    cross_entropy = jnp.mean(labels * jax.nn.softplus(-p) + (1 - labels) * jax.nn.softplus(p))
    loss = cross_entropy + reg_coef * discriminator.regularization() + discr_reg_coef * jnp.mean(jnp.square(p))

    _, _, d_state = nnx.split(discriminator, nnx.Param, nnx.Variable)

    return loss, d_state

  @jax.jit
  def metric_discriminator_f(x_true, c_true, y_true, x_gen, c_gen, y_gen, d_params, d_state):
    x_batch, c_batch, y_batch, labels = combine(x_true, c_true, y_true, x_gen, c_gen, y_gen)
    discriminator = nnx.merge(discriminator_def, d_params, d_state)
    p = discriminator(x_batch, c_batch, y_batch, deterministic=True)
    metric = (p > 0.0) == (labels > 0.5)

    return metric

  @jax.jit
  def step_discriminator(x_true, c_true, y_true, x_gen, c_gen, y_gen, d_params, d_state, opt_state):
    (loss, d_state), grad = jax.value_and_grad(loss_discriminator_f, argnums=6, has_aux=True)(
      x_true, c_true, y_true, x_gen, c_gen, y_gen, d_params, d_state
    )
    updates, opt_state = discriminator_optimizer.update(grad, opt_state)
    d_params = optax.apply_updates(d_params, updates)
    return loss, d_params, d_state, opt_state

  discriminator_losses = np.ndarray(shape=(epochs, steps))
  discriminator_validation = np.ndarray(shape=(epochs, validation_batches, 2 * batch))

  aux: dict | None = restored['aux']
  if aux is not None:
    discriminator_losses[:starting_epoch] = aux['discriminator']['training'][:starting_epoch]
    discriminator_validation[:starting_epoch] = aux['discriminator']['validation'][:starting_epoch]

  size = steps * batch
  val_size = validation_batches * batch

  if restore and dataset is not None:
    try:
      data = np.load(dataset)
    except FileNotFoundError:
      data = None
  else:
    data = None

  if progress:
    from tqdm import tqdm
    progress_bar = tqdm
  else:
    progress_bar = lambda x, *args, **kwargs: x

  if data is None:
    c = np.ndarray(shape=(size, *detector.design_shape()), dtype=np.float32)
    X = np.ndarray(shape=(size, *detector.output_shape()), dtype=np.float32)
    y = np.ndarray(shape=(size, *detector.target_shape()), dtype=np.float32)

    c_val = np.ndarray(shape=(val_size, *detector.design_shape()), dtype=np.float32)
    X_val = np.ndarray(shape=(val_size, *detector.output_shape()), dtype=np.float32)
    y_val = np.ndarray(shape=(val_size, *detector.target_shape()), dtype=np.float32)

    for i in progress_bar(range(steps), desc='sampling training set'):
      c_batch = design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      X_batch, y_batch = detector(get_seed(), configurations=c_batch)

      c[i * batch:(i + 1) * batch] = c_batch
      X[i * batch:(i + 1) * batch] = X_batch
      y[i * batch:(i + 1) * batch] = y_batch

    for i in progress_bar(range(validation_batches), desc='sampling test set'):
      c_batch = design_eps * np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      X_batch, y_batch = detector(get_seed(), configurations=c_batch)

      c_val[i * batch:(i + 1) * batch] = c_batch
      X_val[i * batch:(i + 1) * batch] = X_batch
      y_val[i * batch:(i + 1) * batch] = y_batch

    if dataset is not None:
      np.savez(
        dataset,
        condition=c, samples=X, target=y,
        condition_validation=c_val, samples_validation=X_val, target_validation=y_val
      )

  else:
    c, X, y = data['condition'], data['samples'], data['target']
    c_val, X_val, y_val = data['condition_validation'], data['samples_validation'], data['target_validation']

  # assert X.shape[0] == size
  # assert X_val.shape[0] == val_size

  status = detopt.utils.progress.status_bar(disable=not progress)

  def sample(X_dataset, c_dataset, y_dataset):
    n, *_ = X_dataset.shape

    indx = np_rng.integers(0, n, size=(batch,))
    c_real = c_dataset[indx]
    X_real = X_dataset[indx]
    y_real = y_dataset[indx]

    indx = np_rng.integers(0, n, size=(batch,))
    c_pseudo = c_dataset[indx]
    y_pseudo = y_dataset[indx]

    indx = np_rng.integers(0, n, size=(batch,))
    X_pseudo = X_dataset[indx]

    return X_real, c_real, y_real, X_pseudo, c_pseudo, y_pseudo

  for i in status.epochs(starting_epoch, epochs):
    for j in status.training(steps):
      X_real, c_real, y_real, X_pseudo, c_pseudo, y_pseudo = sample(X, c, y)

      discriminator_losses[i, j], discriminator_parameters, discriminator_state, discriminator_optimizer_state = \
        step_discriminator(
          X_real, c_real, y_real, X_pseudo, c_pseudo, y_pseudo,
          discriminator_parameters, discriminator_state, discriminator_optimizer_state
        )

    for j in status.validation(validation_batches):
      X_real, c_real, y_real, X_pseudo, c_pseudo, y_pseudo = sample(X_val, c_val, y_val)

      metrics = metric_discriminator_f(
        X_real, c_real, y_real, X_pseudo, c_pseudo, y_pseudo,
        discriminator_parameters, discriminator_state
      )
      discriminator_validation[i, j] = metrics

    aux = {
      'discriminator': {
        'training': discriminator_losses[:i + 1],
        'validation': discriminator_validation[:i + 1]
      }
    }

    detopt.utils.io.save_state(
      i, checkpointer, design=restored['design']['design'], design_optimizer_state=None,

      discriminator_parameters=discriminator_parameters, discriminator_state=discriminator_state,
      discriminator_optimizer_state=discriminator_optimizer_state,
      aux=aux
    )

    if report is not None:
      import matplotlib.pyplot as plt
      os.makedirs(report, exist_ok=True)

      fig = plot(aux)
      fig.savefig(os.path.join(report, 'losses.png'))
      plt.close(fig)

  checkpointer.close()

def plot(aux):
  import matplotlib.pyplot as plt

  present = ['regressor', 'discriminator']
  present = [t for t in present if t in aux]

  fig = plt.figure(figsize=(9, 12))
  axes = fig.subplots(2, len(present), squeeze=False)

  for i, name in enumerate(present):
    detopt.utils.viz.losses.plot(aux[name]['training'], axes[0, i])
    axes[0, i].set_title(f'{name} losses')
    detopt.utils.viz.losses.plot(aux[name]['validation'], axes[1, i])
    axes[1, i].set_title(f'{name} validation')

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
  gearup.gearup(
    regressor=regressor, discriminator=discriminator, report=report
  ).with_config('config/pretrain.yaml')()