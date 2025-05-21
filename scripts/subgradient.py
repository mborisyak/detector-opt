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

if __name__ == '__main__':
  import gearup

  gearup.gearup(
    optimize=optimize,
    regress=regress
  ).with_config('config/config.yaml')()