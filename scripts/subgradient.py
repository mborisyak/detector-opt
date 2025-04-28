import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx

import detopt

def optimize(seed, progress=False, **config):
  print(config)

  rngs = nnx.Rngs(seed)

  detector = detopt.detector.from_config(config['detector'])
  method = detopt.optimizer.from_config(detector, config['method'], rngs=rngs)

  design = np.zeros(shape=detector.design_shape())

  epochs, steps, validation_batches = config['epochs'], config['steps'], config['validation_batches']

  if progress:
    from tqdm import tqdm
    main_pb = tqdm(total=epochs, desc='epochs')
    iter_pb = tqdm(total=steps, desc='training')
    val_pb = tqdm(total=validation_batches, desc='validation')

    main_pb_inc = main_pb.update
    iter_pb_inc = iter_pb.update
    iter_pb_reset = iter_pb.reset
    val_pb_inc = val_pb.update
    val_pb_reset = val_pb.reset
  else:
    main_pb_inc = lambda : None
    iter_pb_inc = lambda : None
    iter_pb_reset = lambda: None
    val_pb_inc = lambda: None
    val_pb_reset = lambda: None

  design_training_losses = np.ndarray(shape=(epochs, steps))
  design_validation_losses = list()
  regressor_training_losses = list()

  for i in range(epochs):
    iter_pb_reset()
    val_pb_reset()

    regressor_training_losses_epoch = list()
    for j in range(steps):
      design_training_losses[i, j], regressor_training_loss, design = method.step(
        seed=(seed, i, j, 0), design=design
      )
      regressor_training_losses_epoch.append(regressor_training_loss)
      iter_pb_inc()

    regressor_training_losses.append(np.stack(regressor_training_losses_epoch, axis=0))

    design_validation_losses_epoch = list()
    for j in range(validation_batches):
      design_validation_losses_epoch.append(
        method.validate(seed=(seed, i, j, 1), design=design)
      )
      val_pb_inc()
    design_validation_losses.append(np.stack(design_validation_losses_epoch, axis=0))
    main_pb_inc()

  detopt.utils.viz.losses.plot_losses(design_training_losses, file='design-train.png')
  detopt.utils.viz.losses.plot_losses(design_validation_losses, file='design-val.png')
  detopt.utils.viz.losses.plot_losses(regressor_training_losses, file='reg-train.png')

  import json
  with open('design.json', 'w') as f:
    json.dump(design.tolist(), f)

def regress(seed, progress=False, output_dir=None, **config):
  print(config)

  rngs = nnx.Rngs(seed)
  np_rng = np.random.default_rng(seed=(seed, 0))

  detector = detopt.detector.from_config(config['detector'])
  regressor = detopt.nn.from_config(
    detector.output_shape(), detector.design_shape(), detector.target_shape(),
    config=config['model'], rngs=rngs
  )
  optimizer = nnx.Optimizer(regressor, detopt.utils.config.optimizer(config['optimizer']))

  epochs, steps = config['epochs'], config['steps']
  batch, validation_batches = config['batch'], config['validation_batches']

  @nnx.jit
  def loss_f(model, x, c, t):
    p = model(x, c)
    return jnp.mean(detector.loss(t, p))

  @nnx.jit
  def metric_f(model, x, c, t):
    p = model(x, c)
    return jnp.mean(detector.metric(t, p))

  @nnx.jit
  def step(state, x, c, t):
    x, c, t = jnp.array(x), jnp.array(c), jnp.array(t)
    loss, grad = nnx.value_and_grad(loss_f, argnums=0)(state.model, x, c, t)
    state.update(grad)
    return loss

  if progress:
    from tqdm import tqdm
    main_pb = tqdm(total=epochs, desc='epochs')
    iter_pb = tqdm(total=steps, desc='training')
    val_pb = tqdm(total=validation_batches, desc='validation')

    main_pb_inc = main_pb.update
    iter_pb_inc = iter_pb.update
    iter_pb_reset = iter_pb.reset
    val_pb_inc = val_pb.update
    val_pb_reset = val_pb.reset
  else:
    main_pb_inc = lambda : None
    iter_pb_inc = lambda : None
    iter_pb_reset = lambda: None
    val_pb_inc = lambda: None
    val_pb_reset = lambda: None

  training_losses = np.ndarray(shape=(epochs, steps))
  validation_losses = np.ndarray(shape=(epochs, validation_batches))

  for i in range(epochs):
    iter_pb_reset()
    val_pb_reset()

    for j in range(steps):
      design = np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=(seed, i, j, 0), configurations=design)

      training_losses[i, j] = step(optimizer, measurements, design, target)
      iter_pb_inc()

    for j in range(validation_batches):
      design = np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=(seed, i, j, 1), configurations=design)

      validation_losses[i, j] = metric_f(regressor, design, measurements, target)

      val_pb_inc()
    main_pb_inc()

  import os
  if output_dir is None:
    output_dir = '.'
  else:
    os.makedirs(output_dir, exist_ok=True)

  detopt.utils.viz.losses.plot_losses(training_losses, file=os.path.join(output_dir, 'regress-train.png'))
  detopt.utils.viz.losses.plot_losses(validation_losses, file=os.path.join(output_dir, 'regress-val.png'))

if __name__ == '__main__':
  import gearup

  gearup.gearup(
    optimize=optimize,
    regress=regress
  ).with_config('config/config.yaml')()