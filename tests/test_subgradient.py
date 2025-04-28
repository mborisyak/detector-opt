import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx

import detopt

def test_subgradient(seed):
  rngs = nnx.Rngs(seed + 1)

  detector = detopt.detector.StrawDetector()
  regressor = detopt.nn.AlphaResNet(
    detector.output_shape(), detector.design_shape(), detector.target_shape(),
    n_hidden=32, depth=3, rngs=rngs
  )

  opt = detopt.optimizer.Subgradient(detector, regressor, batch_size=7, n_steps_regressor=3)

  design = np.zeros(shape=detector.design_shape())

  for i in range(100):
    design = opt.step(seed=seed + i, design=design)

  print(design)