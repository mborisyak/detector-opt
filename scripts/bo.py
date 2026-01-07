import detopt

import os

import math
import numpy as np
from scipy import special as sps
import jax
import jax.numpy as jnp

from flax import nnx
import optax

import matplotlib.pyplot as plt

MAX_TDC = 2000

SQRT_2 = math.sqrt(2)
INV_SQRT_2 = math.sqrt(0.5)

def uniform_to_normal(xs):
  eps = np.finfo(xs.dtype).eps
  us = np.clip(xs, eps - 1, 1 - eps)

  return sps.erfinv(us) * SQRT_2

def normal_to_uniform(xs):
  us = sps.erf(xs * INV_SQRT_2)
  return us

def get_regressor():
  return detopt.nn.SparseDeepSet(
    inputs=5, conditions=1, outputs=3, hidden=32,
    depth=5, rngs=nnx.Rngs(123), dropout=0.2
  )

def get_data(path, seed, config):
  data = np.load(path)
  positions = data['positions']
  mask = data['mask']
  tdc = data['tdc']
  hnl_momenta = data['hnl_momenta']
  hnl_decay = data['hnl_decay']
  hnl_mass = data['hnl_mass']
  reconstructed_momenta = data['reconstructed_momenta']
  reconstructed = data['reconstructed']
  design = data['design']
  design_index = data['design_index']

  tdc_cutoff = config['tdc_cutoff']

  mask[(tdc > tdc_cutoff)[:, :, 0]] = False
  tdc[tdc > tdc_cutoff] = 0.0

  for f in data:
    assert np.all(np.isfinite(data[f])), f

  j = 0
  for i in range(design_index.shape[0]):
    if design_index[i] >= 0:
      j = i
    else:
      design_index[i] = design_index[j]
      design[i] = design[j]

  stations = config['stations']
  views = config['views']
  layers = config['layers']
  tubes = config['tubes']

  position_normalization = np.array([stations, views, layers, tubes], dtype=np.float32)
  momentum_normalization = np.array(config['normalization']['momentum'])
  decay_normalization = np.array(config['normalization']['decay'])
  tdc_normalization = np.array(config['normalization']['tdc'])
  design_lower = np.array(config['design']['lower'], dtype=np.float32)
  design_upper = np.array(config['design']['upper'], dtype=np.float32)
  design_center = (design_lower + design_upper) / 2
  design_delta = (design_upper - design_lower) / 2

  def encode_design(x):
    return uniform_to_normal((x - design_center) / design_delta)

  def decode_design(x):
    return normal_to_uniform(x) * design_delta + design_center

  design_encoded = encode_design(design)

  positions = positions[reconstructed]
  tdc = tdc[reconstructed]
  mask = mask[reconstructed]
  hnl_momenta = hnl_momenta[reconstructed]
  design_encoded = design_encoded[reconstructed]
  reconstructed_momenta = reconstructed_momenta[reconstructed]
  design_index = design_index[reconstructed]
  design = design[reconstructed]

  from sklearn.model_selection import train_test_split
  train_index, test_index = train_test_split(np.arange(positions.shape[0]), test_size=0.2, random_state=seed)

  X_train = np.concatenate(
    [positions[train_index] / position_normalization, tdc[train_index] / tdc_normalization],
    axis=-1
  )
  mask_train = mask[train_index]
  y_train = hnl_momenta[train_index] / momentum_normalization
  c_train = design_encoded[train_index, -1:]

  print('X yrain:', np.min(X_train[mask_train, :], axis=(0,)), np.max(X_train[mask_train, :], axis=(0,)))
  print('C train:', np.min(c_train, axis=(0,)), np.max(c_train, axis=(0,)))
  print('Y train:', np.min(y_train, axis=(0,)), np.max(y_train, axis=(0,)))

  X_test = np.concatenate(
    [positions[test_index] / position_normalization, tdc[test_index] / tdc_normalization],
    axis=-1
  )
  mask_test = mask[test_index]
  y_test = hnl_momenta[test_index] / momentum_normalization
  c_test = design_encoded[test_index, -1:]

  N = np.max(design_index) + 1
  errors = [[] for _ in range(N)]
  ns = np.zeros(shape=(N,), dtype=int)
  designs_flat = np.zeros(shape=(N, 3))

  for i in range(design.shape[0]):
    j = design_index[i]
    if j < 0:
      continue
    designs_flat[j] = design[i]
    error = np.sum(np.square(hnl_momenta[i, :] - reconstructed_momenta[i, :]))
    errors[j].append(error)

  print(np.var(hnl_momenta / momentum_normalization, axis=0))

  # print('0', np.sum(errors == 0.0), np.sum(ns == 0))

  # assert len(active_designs) == N, f'{len(active_designs)} / {N}'

  rng = np.random.default_rng(seed)

  def bootstrap(xs, n=32768):
    indx = rng.integers(size=(n, xs.shape[0]), low=0, high=xs.shape[0])
    means = np.mean(xs[indx], axis=0)

    low, high = np.quantile(means, q=(0.025, 0.975))
    med = np.median(means)

    return med, low, high

  n = 50
  ds = np.array([designs_flat[i, 2] for i, xs in enumerate(errors) if len(xs) > n])
  ys, ys_low, ys_high = zip(*[bootstrap(np.array(xs)) for xs in errors if len(xs) > n])
  ys, ys_low, ys_high = np.array(ys), np.array(ys_low), np.array(ys_high)

  fig = plt.figure(figsize=(16, 6))
  axes = fig.subplots(1, 2, squeeze=False)
  axes[0, 0].scatter(ds, ys, label='FairSHiP reconstruction',)
  axes[0, 0].errorbar(
    ds,
    ys,
    # yerr=[
    #   [np.quantile(xs, q=0.1) for xs in errors if len(xs) > n],
    #   [np.quantile(xs, q=0.9) for xs in errors if len(xs) > n],
    # ],
    yerr=[ys - ys_low, ys_high - ys],
    linestyle='',
    label='95% CI, bootstrap'
  )
  print(ys - ys_low)
  print(ys_high - ys)
  axes[0, 0].plot(
    [4.57, 4.57],
    [0, np.max(ys_high)],
    linestyle='--', color='black',
    label='default design'
  )
  axes[0, 0].legend()
  axes[0, 0].set_title('HNL momentum reconstruction MSE')
  axes[0, 0].set_ylabel('MSE')
  axes[0, 0].set_xlabel('view angle')
  axes[0, 0].set_xlim([np.min(designs_flat[:, 2]), np.max(designs_flat[:, 2])])

  axes[0, 1].bar(designs_flat[:, 2], [len(xs) for xs in errors], width=0.75 * 9.14 / 32)
  axes[0, 1].set_xlim([np.min(designs_flat[:, 2]), np.max(designs_flat[:, 2])])
  axes[0, 1].set_title('Number of events for each run')
  axes[0, 1].set_xlabel('view angle')

  fig.tight_layout()
  fig.savefig('angles.png')
  plt.close(fig)

  return X_train, mask_train, c_train, y_train, X_test, mask_test, c_test, y_test

def train(seed, dataset, checkpoint, **config):
  X_train, mask_train, c_train, y_train, X_test, mask_test, c_test, y_test = get_data(dataset, seed=seed, config=config)

  batch, batch_val = 16, 128
  epochs = 1024
  iterations = X_train.shape[0] // batch

  rng = np.random.default_rng(seed + 1)

  losses = np.zeros(shape=(epochs, iterations))
  val_losses = np.zeros(shape=(epochs, X_test.shape[0]))

  regressor = get_regressor()
  model_def, model_parameters, model_state = nnx.split(regressor, nnx.Param, nnx.Variable)
  optimizer = optax.adabelief(learning_rate=2e-4)
  optimizer_state = optimizer.init(model_parameters)

  print('Total parameters', sum(math.prod(p.shape) for p in jax.tree.leaves(model_parameters)))

  manager = detopt.utils.checkpoint.get_manager(checkpoint)
  latest_step, restored = detopt.utils.checkpoint.load_model(manager)

  if restored is not None:
    print(f'Restoring from {checkpoint} at step {latest_step}')

    model_parameters_def = jax.tree.structure(model_parameters)
    model_parameters = jax.tree.unflatten(model_parameters_def, restored['model']['parameters'])

    model_state_def = jax.tree.structure(model_state)
    model_state = jax.tree.unflatten(model_state_def, restored['model']['state'])

    optimizer_state_def = jax.tree.structure(optimizer_state)
    optimizer_state = jax.tree.unflatten(optimizer_state_def, restored['model']['optimizer_state'])
    losses[:latest_step + 1] = restored['aux']['train']
    val_losses[:latest_step + 1] = restored['aux']['test']
  else:
    latest_step = -1

  def apply(x, mask, c, m_params, m_state, deterministic=True):
    model = nnx.merge(model_def, m_params, m_state)
    p = model(x, c, mask=mask, deterministic=deterministic)
    _, _, m_state_updated = nnx.split(regressor, nnx.Param, nnx.Variable)
    return p, m_state_updated

  apply = jax.jit(apply, static_argnames=('deterministic', ))

  def regularization(m_params):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(m_params))

  print(regularization(model_parameters))

  @jax.jit
  def loss(x, mask, c, y, m_params, m_state):
    p, m_state_updated = apply(x, mask, c, m_params, m_state, deterministic=False)
    l2_reg = regularization(m_params)
    loss = jnp.mean(np.sum(jnp.square(y - p), axis=-1)) # + 1.0e-5 * l2_reg
    return loss, m_state_updated

  @jax.jit
  def metric(x, mask, c, y, m_params, m_state):
    p, _ = apply(x, mask, c, m_params, m_state, deterministic=True)

    return np.sum(jnp.square(y - p), axis=-1)

  @jax.jit
  def step(x, mask, c, y, m_params, m_state, o_state):
    (l, m_state_updated), grad = jax.value_and_grad(loss, argnums=4, has_aux=True)(x, mask, c, y, m_params, m_state)
    updates, o_state_updated = optimizer.update(grad, o_state)
    m_params_updated = optax.apply_updates(m_params, updates)

    return l, m_params_updated, m_state_updated, o_state_updated

  status = detopt.utils.status.bar(
    starting_epoch=latest_step + 1, epochs=epochs,
    train_iterations=iterations,
    validation_batches=detopt.utils.batched.number_of_batches(X_test.shape[0], batch_val, subbatch=True),
  )

  for i in range(latest_step + 1, epochs):
    for j in range(iterations):
      index = rng.integers(0, X_train.shape[0], size=(batch, ))
      X_batch, mask_batch, c_batch, y_batch = X_train[index], mask_train[index], c_train[index], y_train[index]

      losses[i, j], model_parameters, model_statem, optimizer_state = step(
        X_batch, mask_batch, c_batch[:, None, :], y_batch,
        model_parameters, model_state, optimizer_state
      )

      status.train_step()

    for index in detopt.utils.batched.range(X_test.shape[0], batch=batch_val, subbatch=True):
      val_losses[i, index] = metric(
        X_test[index], mask_test[index], c_test[index, None, :], y_test[index],
        model_parameters, model_state
      )
      status.validation_step()

    aux = {
      'train' : losses[:(i + 1)],
      'test': val_losses[:(i + 1)]
    }
    detopt.utils.checkpoint.save_model(manager, i, model_parameters, model_state, optimizer_state, aux)
    detopt.utils.plot.losses(
      os.path.join(checkpoint, 'losses.png'),
      dict(title='training loss', losses=losses[:(i + 1)]),
      dict(title='test nmetrics', losses=val_losses[:(i + 1)]),
    )

    status.epoch()

  manager.close()

def test(seed, dataset, checkpoint, **config):
  X_train, mask_train, c_train, y_train, X_test, mask_test, c_test, y_test = get_data(dataset, seed=seed, config=config)

  batch_val = 128

  regressor = get_regressor()
  model_def, model_parameters, model_state = nnx.split(regressor, nnx.Param, nnx.Variable)

  manager = detopt.utils.checkpoint.get_manager(checkpoint)
  latest_step, restored = detopt.utils.checkpoint.load_model(manager)

  print(f'Restoring from {checkpoint} at step {latest_step}')

  model_parameters_def = jax.tree.structure(model_parameters)
  model_parameters = jax.tree.unflatten(model_parameters_def, restored['model']['parameters'])

  model_state_def = jax.tree.structure(model_state)
  model_state = jax.tree.unflatten(model_state_def, restored['model']['state'])

  normalization = np.array(config['normalization']['momentum'], dtype=np.float32)

  manager.close()

  @jax.jit
  def apply(x, mask, c, m_params, m_state):
    model = nnx.merge(model_def, m_params, m_state)
    p = model(x, c, mask=mask, deterministic=True)
    return p

  predictions = np.concatenate([
    apply(X_test[index], mask_test[index], c_test[index, None, :], model_parameters, model_state)
    for index in detopt.utils.batched.range(X_test.shape[0], batch=batch_val, subbatch=True)
  ])

  low, high = 0.0, 9.14
  design = normal_to_uniform(c_test[:, 0]) * (high - low) / 2  +  (high + low) / 2

  bins = np.linspace(0.0, 9.14, num=32)
  dx = np.mean(np.diff(bins))
  bins = bins + dx
  index = np.searchsorted(bins, design)

  grouped = []
  for i in range(bins.shape[0]):
    group_index, = np.where(index == i)
    losses = np.sum(np.square(normalization * (predictions[group_index] - y_test[group_index])), axis=-1)
    grouped.append(losses)


  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import Matern
  from sklearn.model_selection import cross_val_predict

  xs = np.array([bins[i] for i, ls in enumerate(grouped) if len(ls) > 1])
  ys = np.array([np.mean(ls) for ls in grouped if len(ls) > 1])
  ys_var = np.array([np.var(ls) / (len(ls) - 1) for ls in grouped if len(ls) > 1])
  print(np.sqrt(ys_var))

  # C = 1000.0
  # gp = GaussianProcessRegressor(kernel=Matern(length_scale_bounds=(1e-2, 100)), alpha=ys_var / C / C, normalize_y=False)
  # gp.fit(xs[:, None], ys[:, None] / C)
  # m, s = gp.predict(xs[:, None], return_std=True)
  # m, s = m * C, s * C
  # print(m)
  # print(s)

  fig = plt.figure(figsize=(16, 6))
  axes = fig.subplots(1, 1, squeeze=False)
  axes[0, 0].scatter(
    [bins[i] for i, ls in enumerate(grouped) if len(ls) > 1],
    [np.mean(ls) for ls in grouped if len(ls) > 1], #width=np.mean(np.diff(bins))
  )
  axes[0, 0].errorbar(
    [bins[i] for i, ls in enumerate(grouped) if len(ls) > 1],
    [np.mean(ls) for ls in grouped if len(ls) > 1],
    yerr=[np.std(ls) / np.sqrt(len(ls) - 1) for ls in grouped if len(ls) > 1],
    linestyle=''
  )
  axes[0, 0].set_xlim([0, 9.14])
  fig.tight_layout()
  fig.savefig(os.path.join('deep-set.png'))
  plt.close(fig)

if __name__ == '__main__':
  import gearup

  gearup.gearup(
    train=train,
    test=test
  ).with_config('config/bo.yaml')()
