import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx

import detopt

def regress(seed, output, progress=False, report=None, **config):
  print(config)

  rngs = nnx.Rngs(seed)
  np_rng = np.random.default_rng(seed=(seed, 0))

  detector = detopt.detector.from_config(config['detector'])
  regressor = detopt.nn.from_config(
    detector.output_shape(), detector.design_shape(), detector.target_shape(),
    config=config['regressor'], rngs=rngs
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

  training_losses = np.ndarray(shape=(epochs, steps))
  validation_losses = np.ndarray(shape=(epochs, validation_batches))

  status = detopt.utils.progress.status_bar(epochs, steps, validation_batches, disable=not progress)

  for i in status.epochs():
    for j in status.training():
      design = np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=(seed, i, j, 0), configurations=design)

      training_losses[i, j] = step(optimizer, measurements, design, target)

    for j in status.validation():
      design = np_rng.normal(size=(batch, *detector.design_shape())).astype(np.float32)
      measurements, target = detector(seed=(seed, i, j, 1), configurations=design)

      validation_losses[i, j] = metric_f(regressor, design, measurements, target)

  if report is not None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1)
    detopt.utils.viz.losses.plot_losses(training_losses, axes=axes[0], color=0)
    axes[0].set_title('Training losses')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel(detector.loss_name())
    axes[0].set_yscale('log')
    detopt.utils.viz.losses.plot_losses(validation_losses, axes=axes[1], color=1)
    axes[1].set_title('Validation losses')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel(detector.metric_name())
    axes[1].set_yscale('log')
    fig.tight_layout()
    fig.savefig(report)
    plt.close(fig)

  detopt.utils.io.save_model(output, regressor, aux=None)

if __name__ == '__main__':
  import gearup
  gearup.gearup(regress).with_config('config/config.yaml')()