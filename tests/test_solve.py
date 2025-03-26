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
  n, n_t = 11, 2 * 1024
  dt = 1.0e-2
  x0 = np.zeros(shape=(n, 3)) + np.array([0.0, 0.0, -5.0])
  v0 = np.array([0.0, 0.0, 1.0])[None, :] + 0.05 * np.random.normal(size=(n, 3))
  v0 = 0.99 * v0 / np.sqrt(np.sum(np.square(v0), axis=-1))[..., None]
  q = 2 * np.random.binomial(p=0.5, n=1, size=(n, )).astype(np.double) - 1
  m = np.ones(shape=(n, ))
  B, L = 1.0, 2.0

  output = np.zeros(shape=(n, n_t, 3))

  import time
  n_trials = 1024
  start_time = time.perf_counter()
  for i in range(n_trials):
    _ = detopt.detectors.straw_detector.solve(x0, v0, m, q, B, L, dt, output)
  runtime = time.perf_counter() - start_time
  print(f'iterations per second: {n_trials * n / runtime}')

  _ = detopt.detectors.straw_detector.solve(x0, v0, m, q, B, L, dt, output)

  n_l, n_s = 5, 23
  layers = np.linspace(-5, 5, num=n_l + 2)[1:-1]
  response = np.zeros(shape=(n, n_l, n_s))

  _ = detopt.detectors.straw_detector.detect(output, layers, response, 1.0, 2.0)

  print(output[:, 0, :])
  print(output[:, -1, :])

  print(response[..., 0])

  ax = plt.figure(figsize=(9, 9)).add_subplot(projection='3d')

  xs_ = np.array([np.min(output[..., 0]), np.max(output[..., 0])])
  ys_ = np.array([np.min(output[..., 1]), np.max(output[..., 1])])
  xs_grid, ys_grid = np.meshgrid(xs_, ys_, indexing='ij')
  for l_z in layers:
    ax.plot_surface(xs_grid, ys_grid, np.ones_like(xs_grid) * l_z, alpha=0.1, color=plt.cm.tab10(3))

  for i in range(n):
    ax.plot(
      output[i, :, 0], output[i, :, 1], output[i, :, 2],
      color=plt.cm.tab10(0) if q[i] < 0.0 else plt.cm.tab10(1)
    )

  zs = np.linspace(np.min(output[..., 2]), np.max(output[..., 2]), num=128)

  ax.plot(B * np.exp(-np.square(zs / L)), zs, zs=np.max(output[..., 1]), zdir='y', color='black', label='$B_y$')
  ax.legend()
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.tight_layout()
  plt.savefig('boris.png')
  plt.close()