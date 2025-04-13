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
  rng = np.random.RandomState(123)

  n, n_t = 7, 1500
  dt = 1.0e-2
  n_layers, n_straws = 7, 40
  x0 = rng.normal(size=(n, 3))
  x0[..., -1] = -5
  v0 = np.array([0.0, 0.0, 1.0])[None, :] + 0.1 * rng.normal(size=(n, 3))
  v0 = 0.99 * v0 / np.sqrt(np.sum(np.square(v0), axis=-1))[..., None]
  q = 2 * rng.binomial(p=0.5, n=1, size=(n, )).astype(np.double) - 1
  print(q)
  m = np.ones(shape=(n, ))
  B, L = 1.0, 2.0

  trajectories = np.zeros(shape=(n, n_t, 3))
  responses = np.zeros(shape=(n, n_layers, n_straws))

  layers = np.linspace(-4, 4, num=n_layers, dtype=np.float64)
  width = 5 * np.ones(shape=(n_layers, ), dtype=np.float64)
  heights = 5 * np.ones(shape=(n_layers,), dtype=np.float64)
  angles = np.linspace(0, np.pi, num=n_layers, dtype=np.float64)

  import time
  n_trials = 1024
  start_time = time.perf_counter()
  for i in range(n_trials):
    _ = detopt.detectors.straw_detector.solve(
      x0, v0, m, q, B, L, dt,
      layers, width, heights, angles,
      trajectories, responses
    )
  runtime = time.perf_counter() - start_time
  print(f'iterations per second: {n_trials * n / runtime}')

  responses[:] = 0.0
  _ = detopt.detectors.straw_detector.solve(
    x0, v0, m, q, B, L, dt,
    layers, width, heights, angles,
    trajectories, responses
  )

  print(trajectories[:, 0, :])
  print(trajectories[:, -1, :])

  print(responses[0])
  combined_response = np.sum(responses, axis=0)

  import pyvista as pv

  plotter = pv.Plotter()

  xs_ = np.array([np.min(trajectories[..., 0]), np.max(trajectories[..., 0])])
  ys_ = np.array([np.min(trajectories[..., 1]), np.max(trajectories[..., 1])])
  xs_grid, ys_grid = np.meshgrid(xs_, ys_, indexing='ij')
  for i, l_z in enumerate(layers):
    for k in range(n_straws):
      A = np.array([
        [np.cos(angles[i]), np.sin(angles[i])],
        [-np.sin(angles[i]), np.cos(angles[i])]
      ])

      h, w = heights[i], width[i]
      r = heights[i] / n_straws
      verts = np.array([
        [-w, 2 * r * k - h], [-w, 2 * r * k - h + 2 * r], [w, 2 * r * k - h + 2 * r], [w, 2 * r * k - h]
      ])
      verts = np.concatenate([np.dot(verts, A), l_z * np.ones(shape=(4, 1))], axis=-1)
      faces = np.array([[4, 0, 1, 2, 3]])
      mesh = pv.PolyData(verts, faces=faces)
      plotter.add_mesh(
        mesh, color=(1.0, 0.0, 0.0), show_edges=False,
        opacity=0.5 if combined_response[i, k] > 0.0 else 0.0
      )

      plotter.add_mesh(mesh, color='black', style='wireframe', opacity=0.25, line_width=0.1)

  for i in range(n):
    traj = pv.Spline(trajectories[i])#.tube(radius=0.05, )
    plotter.add_mesh(traj, color='red' if q[i] > 0 else 'blue', line_width=4, opacity=0.5)
    # ax.plot(
    #   trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2],
    #   color=plt.cm.tab10(0) if q[i] < 0.0 else plt.cm.tab10(1), zorder=10
    # )

  zs = np.linspace(np.min(trajectories[..., 2]), np.max(trajectories[..., 2]), num=128)
  plotter.show_grid()
  # plotter.enable_depth_peeling()
  plotter.show()