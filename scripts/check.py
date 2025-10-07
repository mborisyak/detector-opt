
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

def check(seed, regressor_checkpoint, output, progress=True, report=None, **config):
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
  with open(config['design'], 'r') as f:
    import json
    design = detector.encode_design(json.load(f))

  print(f'using {design} as design')

  validation_batches = config['validation_batches']
  batch = config['batch']

  metric_names = detector.metric_names()

  @jax.jit
  def error_f(x, c, t, r_params, r_state):
    regressor = nnx.merge(regressor_def, r_params, r_state)
    p = regressor(x, c, deterministic=True)

    metric = detector.errors(t, p)

    return metric

  design_validation = {
    k: np.ndarray(shape=(validation_batches, batch))
    for k in metric_names
  }

  if progress:
    from tqdm import tqdm
    progress_bar = tqdm
  else:
    progress_bar = lambda x, *args, **kwargs: x

  for i in progress_bar(range(validation_batches)):
    design_batch = np.broadcast_to(design[None], shape=(batch, *detector.design_shape()))
    measurements, target = detector(seed=get_seed(), configurations=design_batch)

    errors = error_f(measurements, design_batch, target, regressor_parameters, regressor_state)
    for k in metric_names:
      design_validation[k][i] = errors[k]

  os.makedirs(os.path.basename(output), exist_ok=True)
  np.savez(output, **design_validation)

def report(output, **config):
  data = {}

  detector = detopt.detector.from_config(config['detector'])
  label_names = detector.labels()

  for name in config['compare']:
    f = np.load(config['compare'][name])
    data[name] = {
      k: f[k] for k in f.keys()
    }

  metrics = set([m for k in data for m in data[k]])
  for k in data:
    assert len(metrics) == len(data[k])

  first_name, *_ = data.keys()
  metrics = data[first_name].keys()

  n_rows = 2
  n_cols = len(metrics) // n_rows + (0 if len(metrics) % n_rows == 0 else 1)

  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(n_cols * 6, n_rows * 4))
  axes = fig.subplots(n_rows, n_cols, squeeze=False).ravel()

  statistics = {}
  for i, m in enumerate(metrics):
    statistics[m] = {}

    for k in data:
      xs = data[k][m].ravel()
      square_errors = np.square(xs)
      mean_sqr = np.mean(square_errors)

      mean = np.sqrt(mean_sqr)
      err = 1.96 * np.sqrt(
        np.var(square_errors, ddof=1) / 4 / mean_sqr / (xs.size - 1)
      )
      statistics[m][k] = (mean, err)

  for i, m in enumerate(metrics):
    losses = []
    labels = []
    for k in data:
      xs = data[k][m].ravel()

      mean, err = statistics[m][k]

      label = f'{k}, ${mean:.2f} \\pm {err:.2f}$'
      losses.append(xs)
      labels.append(label)

    axes[i].hist(losses, histtype='step', label=labels, bins=config.get('bins', 20))
    # axes[i].set_title(label_names[i], fontsize=14)
    axes[i].set_xlabel(label_names[i], fontsize=16)
    axes[i].legend(loc='lower center',  bbox_to_anchor=(0.5, 1.05), fontsize=14, ncols=2)
    # axes[i].legend(loc='upper left', fontsize=11, ncols=1)

  fig.tight_layout()
  fig.savefig(output)
  plt.close()

  header = ['metric'] + list(config['compare'])
  print(' & '.join(header), '\\\\', '\\hline')
  for label, metric in zip(label_names, statistics):
    row = [label]
    for k in statistics[metric]:
      mean, err = statistics[metric][k]
      row.append(f'${mean:.3f} \\pm {err:.3f}$')

    print(' & '.join(row), '\\\\', '\\hline')



if __name__ == '__main__':
  import gearup
  gearup.gearup(check=check, report=report).with_config('config/check.yaml')()