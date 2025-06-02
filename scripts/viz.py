import os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

import detopt

def viz(seed=123, design='data/design/default.json', **config):
  n_batch = 3
  n_layers = 16
  detector = detopt.detector.from_config(config['detector'])

  root = os.path.dirname(os.path.dirname(__file__))

  with open(os.path.join(root, design), 'r') as f:
    import json
    design = detector.encode_design(json.load(f))

  configs = np.broadcast_to(design[None], (n_batch, *design.shape))

  import time
  n_trials = 1024
  start_time = time.perf_counter()
  for i in range(n_trials):
    _ = detector(seed=1, configurations=configs)
  runtime = time.perf_counter() - start_time
  events_per_second_single_core = n_trials * n_batch / runtime
  print(f'events per second: {events_per_second_single_core}')

  from concurrent.futures import ThreadPoolExecutor

  n_cores = 5 # os.cpu_count()
  start_time = time.perf_counter()
  with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = [
      executor.submit(detector, seed=1, configurations=configs)
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
    _, _, _, v0, trajectories, response, signal = detector.sample(seed=i, design=configs)
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

  _, _, _, _, trajectories, response, signal = detector.sample(seed=seed, design=configs)
  layers, angles, widths, heights, Bs, Ls = detector.get_design(design=configs)

  print(response.shape)
  plt.matshow(response[0].T)
  plt.colorbar()
  plt.show()
  plt.close()

  detopt.utils.viz.straw.show(
    layers[0], angles[0], widths[0], heights[0], response[0], trajectories[0], signal[0],
    threshold=0.3
  )

if __name__ == '__main__':
  import gearup
  gearup.gearup(viz).with_config('config/config.yaml')()