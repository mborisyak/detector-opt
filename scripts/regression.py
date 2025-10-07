import os

import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx
import optax

import detopt

from tqdm import tqdm
import matplotlib
matplotlib.use('AGG')

MAX_INT = 9223372036854775807

def info(**config):
  import math

  rng = jax.random.PRNGKey(1)
  rngs = nnx.Rngs(1)

  detector = detopt.detector.from_config(config['detector'])
  regressor = detopt.nn.from_config(detector, config=config['regressor']['model'], rngs=rngs)
  _, parameters, _ = nnx.split(regressor, nnx.Param, nnx.Variable)
  total_number_of_parameters = sum(math.prod(p.shape) for p in jax.tree.leaves(parameters))

  print(f'Total number of parameters: {total_number_of_parameters}')

def regress(seed, output, dataset=None, progress=True, restore=True, report=None, **config):
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
  regressor_parameters, regressor_state = restored['regressor']['parameters'], restored['regressor']['state']
  regressor_optimizer_state = restored['regressor']['optimizer_state']

  epochs, steps = config['epochs'], config['steps']
  batch, validation_batches = config['batch'], config['validation_batches']

  @jax.jit
  def loss_f(x, c, t, r_params, r_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)
    p = regressor(x, c, deterministic=False)

    loss = jnp.mean(detector.loss(t, p))

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
    return loss, r_params, r_state, opt_state

  regressor_losses = np.ndarray(shape=(epochs, steps))
  metric_names = detector.metric_names()
  regressor_validation = {
    k: np.ndarray(shape=(epochs, validation_batches, batch))
    for k in metric_names
  }

  aux: dict | None = restored['aux']
  if aux is not None:
    regressor_losses[:starting_epoch] = aux['regressor']['training'][:starting_epoch]
    for k in metric_names:
      regressor_validation[k][:starting_epoch] = aux['regressor']['validation'][k][:starting_epoch]

  design_batch = np.broadcast_to(design[None], shape=(batch, *detector.design_shape()))

  if dataset is not None:
    try:
      f = np.load(dataset)
      X, y, X_val, y_val = f['X'], f['y'], f['X_val'], f['y_val']
      n_total, n_total_val = X.shape[0], X_val.shape[0]
    except FileNotFoundError:
      X, y, X_val, y_val = None, None, None, None
      n_total, n_total_val = None, None
  else:
    X, y, X_val, y_val = None, None, None, None
    n_total, n_total_val = None, None

  if X is None:
    n_total = batch * steps

    X = np.ndarray(shape=(n_total, *detector.output_shape()), dtype=np.float32)
    y = np.ndarray(shape=(n_total, *detector.target_shape()), dtype=np.float32)

    if progress:
      progress_bar = tqdm
    else:
      progress_bar = lambda x, *args, **kwargs: x

    for i in progress_bar(range(steps), desc='sampling training'):
      X[i * batch: (i + 1) * batch], y[i * batch: (i + 1) * batch] = detector(
        seed=get_seed(), configurations=design_batch
      )

    n_total_val = batch * steps
    X_val = np.ndarray(shape=(n_total_val, *detector.output_shape()), dtype=np.float32)
    y_val = np.ndarray(shape=(n_total_val, *detector.target_shape()), dtype=np.float32)

    for i in progress_bar(range(validation_batches), desc='sampling validation'):
      X_val[i * batch: (i + 1) * batch], y_val[i * batch: (i + 1) * batch] = detector(
        seed=get_seed(), configurations=design_batch
      )

    if dataset is not None:
      np.savez(dataset, X=X, y=y, X_val=X_val, y_val=y_val)

  status = detopt.utils.progress.status_bar(disable=not progress)

  for i in status.epochs(starting_epoch, epochs):
    for j in status.training(steps):
      indx = np_rng.integers(low=0, high=n_total, size=(batch, ))
      measurements, target = X[indx], y[indx]

      regressor_losses[i, j], regressor_parameters, regressor_state, regressor_optimizer_state = \
        step_regressor(
          measurements, design_batch, target,
          regressor_parameters, regressor_state, regressor_optimizer_state
        )

    for j in status.validation(validation_batches):
      measurements, target = X_val[i * batch: (i + 1) * batch], y_val[i * batch: (i + 1) * batch]

      metrics = metric_f(measurements, design_batch, target, regressor_parameters, regressor_state)
      for k, m in metrics.items():
        regressor_validation[k][i, j] = m

    aux = {
      'regressor': {
        'training': regressor_losses[:i + 1],
        'validation': {k: v[:i + 1] for k, v in regressor_validation.items()}
      }
    }

    detopt.utils.io.save_state(
      i, checkpointer, design=design,

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

  gearup.gearup(regress=regress, report=report).with_config('config/regression.yaml')()