import os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

import detopt

def test_solve(seed, plot_root):
  n = 64
  rng = jax.random.PRNGKey(seed)
  key_x, key_y, key_z, key_q, key_m = jax.random.split(rng, num=5)

  x0 = jnp.zeros(shape=(n, 3)) + jnp.array([0, 0, -5])
  v0 = 5 * jnp.array([0, 0, 1], dtype=jnp.float32) + \
    jnp.stack([
      jax.random.normal(key_x, shape=(n, )),
      jax.random.normal(key_y, shape=(n,)),
      jax.random.exponential(key_z, shape=(n,)),
    ], axis=-1)

  v0_norm = jnp.sqrt(jnp.sum(jnp.square(v0), axis=-1))
  v0 = v0 / v0_norm[..., None]

  B0 = jnp.array([0, 0, 0], dtype=jnp.float32)
  q = 2 * jax.random.randint(key_q, shape=(n, ), minval=0, maxval=2) - 1
  m = jax.random.lognormal(key_m, shape=(n, ), sigma=1.0)

  trajectory = detopt.detectors.straw.solve(x0, v0, q=q, m=m, J=0.0, K=0.2, L=3.0, B0=B0, dt=1.0e-2, steps=2000)

  print(trajectory.shape)

  fig = plt.figure(figsize=(6, 6))
  axes = fig.add_subplot(projection='3d')

  for i in range(n):
    axes.plot(
      trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2],
      color=plt.cm.tab10(0) if q[i] < 0 else plt.cm.tab10(1), alpha=0.2 + 0.8 * float(m[i] / jnp.max(m))
    )
  axes.set_xlabel('x')
  axes.set_ylabel('y')
  axes.set_zlabel('z')
  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'Bz.png'))
  plt.close(fig)

if __name__ == '__main__':
  x0 = np.zeros(shape=(32, 3))
  v0 = np.zeros(shape=(32, 3))
  q = np.ones(shape=(32, ))
  m = np.ones(shape=(32, ))
  J, K, L = 0.1, 0.1, 2.0

  output = np.zeros(shape=(32, 1000, 3))

  print(
    detopt.detectors.straw_detector.solve(
      x0, v0, q, m, J, K, L,
      0.1, output
    )
  )

  print(output)