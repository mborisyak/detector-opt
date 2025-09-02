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

  trajectory = detopt.detector.straw.solve(x0, v0, q=q, m=m, J=0.0, K=0.2, L=3.0, B0=B0, dt=1.0e-2, steps=2000)

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

  n_batch = 3
  n_layers = 16
  generator = detopt.detector.StrawDetector(n_layers=n_layers)

  configs = np.ndarray(shape=(n_batch, 2 * n_layers + 1))

  configs[:, :n_layers] = np.linspace(-3, 3, num=n_layers)[None, :]
  configs[:, n_layers:-1] = np.linspace(-3, 3, num=n_layers)[None, :]
  configs[:, -1] = 5.0

  import time
  n_trials = 1024
  start_time = time.perf_counter()
  for i in range(n_trials):
    _ = generator(seed=1, configurations=configs)
  runtime = time.perf_counter() - start_time
  events_per_second_single_core = n_trials * n_batch / runtime
  print(f'events per second: {events_per_second_single_core}')

  import cProfile
  def generate():
    for _ in range(n_trials):
      generator(seed=1, configurations=configs)

  cProfile.run('generate()', 'generate.profile')

  from concurrent.futures import ThreadPoolExecutor
  import os

  n_cores = 5 # os.cpu_count()
  start_time = time.perf_counter()
  with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = [
      executor.submit(generator, seed=1, configurations=configs)
      for _ in range(n_trials)
    ]

    for future in futures:
      _ = future.result()

  runtime = time.perf_counter() - start_time
  events_per_second_multi_core = n_trials * n_batch / runtime
  print(
    f'events per second: {events_per_second_multi_core} '
    f'(eff. {events_per_second_multi_core / events_per_second_single_core / n_cores})'
  )

  v0s = list()
  ys = list()
  for i in range(1000):
    _, _, _, v0, trajectories, response, signal = generator.sample(seed=i, design=configs)
    v0s.append(v0)
    ys.append(signal)

  v0s = np.concatenate(v0s, axis=0)
  ys = np.concatenate(ys, axis=0)

  ns = np.sum(np.sum(np.square(v0s), axis=-1) > 1.0e-3, axis=-1)

  plt.hist([ns[ys > 0.5], ns[ys < 0.5]], bins=7, label=['signal', 'noise'], histtype='step')
  plt.title('Number of trajectories per event')
  plt.legend()
  plt.savefig('straw-events.png')
  plt.close()

  _, _, _, _, trajectories, response, signal = generator.sample(seed=1234567899, design=configs)
  layers, angles, widths, heights, Bs, Ls = generator.get_design(design=configs)

  detopt.utils.viz.straw.show(layers[0], angles[0], widths[0], heights[0], response[0], trajectories[0], signal[0])
