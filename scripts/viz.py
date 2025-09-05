import os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

import detopt

def viz(seed=123, design='data/design/default.json', **config):
  n_batch = 32
  n_layers = 16
  detector = detopt.detector.from_config(config['detector'])

  root = os.path.dirname(os.path.dirname(__file__))

  with open(os.path.join(root, design), 'r') as f:
    import json
    design = detector.encode_design(json.load(f))

  configs = np.broadcast_to(design[None], (n_batch, *design.shape))

  import time
  n_trials = 128
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

  n, m, *_ = trajectories.shape

  trajectories = np.reshape(trajectories, shape=(n * m, *trajectories.shape[2:]))

  detopt.utils.viz.straw.show(
    layers[0], angles[0], widths[0], heights[0], response[0], trajectories, signal[0],
    threshold=0.3
  )

def compare(design, reference='data/design/default.json', report='designs.png', aux=None, seed=123456789, **config):
  import matplotlib.pyplot as plt
  import json

  detector: detopt.detector.StrawDetector = detopt.detector.from_config(config['detector'])
  L = detector.L
  W = detector.layer_width
  H = detector.layer_height

  with open(design, 'r') as f:
    design = json.load(f)

  rng = np.random.default_rng(seed)


  with open(reference, 'r') as f:
    reference = json.load(f)

  pos = np.array(design['positions'])
  pos_ref = np.array(reference['positions'])

  angles = np.array(design['angles'])
  angles_ref = np.array(reference['angles'])

  B = design['magnetic_strength']
  B_ref = reference['magnetic_strength']

  ys = np.array([-H, H])[:, None] + 0 * pos[None, :]
  xs = np.array([-W, W])[:, None] + 0 * pos[None, :]

  zs = np.linspace(-5, 5, num=128)
  Bs_ref = B_ref * np.exp(-np.square(zs / L))
  Bs = B * np.exp(-np.square(zs / L))
  max_B = max(np.max(Bs_ref), np.max(Bs))

  def sample(d, n):
    encoded = detector.encode_design(d)
    configs = np.broadcast_to(encoded[None], shape=(n, *encoded.shape))
    _, _, _, _, trajectories, response, signal = detector.sample(seed=seed, design=configs)
    return trajectories

  fig = plt.figure(figsize=(12, 6))
  axes = fig.subplots(2, 2)

  traj = sample(reference, 32)

  ax = axes[0, 0]
  ax.set_title('default design (side view)')
  ax.plot(np.stack([pos_ref, pos_ref]), ys, color=plt.cm.tab10(0))
  twin = ax.twinx()
  twin.plot(zs, Bs_ref, color='black')
  twin.set_ylim([-0.05, 1.05 * max_B])
  twin.set_ylabel('magnetic field, $B_x$')

  for i in range(traj.shape[0]):
    ax.plot(traj[i, 0, :, 2], traj[i, 0, :, 1], color=plt.cm.tab10(0), alpha=0.15)
    ax.plot(traj[i, 1, :, 2], traj[i, 1, :, 1], color=plt.cm.tab10(0), alpha=0.15)

  ax.set_ylim([-1.25 * H, 1.25 * H])
  ax.set_xlim([-6.0, 6.0])
  ax.set_ylabel('y-axis')
  ax.set_xlabel('z-axis')

  ax = axes[0, 1]
  ax.bar(pos_ref, angles_ref + np.pi / 3, width=0.05, color=plt.cm.tab10(0), bottom=-np.pi / 3)
  ax.plot([-6.0, 6.0], [0.0, 0.0], color='black', linestyle='--')
  ax.set_ylim([-np.pi / 3, np.pi / 3])
  ax.set_xlim([-6.0, 6.0])

  ax.set_ylabel("layer's angle")
  ax.set_xlabel('z-axis')

  ax = axes[1, 0]
  ax.set_title('optimized design (side view)')
  ax.plot(np.stack([pos, pos]), ys, color=plt.cm.tab10(1))
  twin = ax.twinx()
  twin.plot(zs, Bs, color='black')
  twin.set_ylim([-0.05, 1.05 * max_B])
  twin.set_ylabel('magnetic field, $B_x$')

  traj = sample(design, 32)
  for i in range(traj.shape[0]):
    ax.plot(traj[i, 0, :, 2], traj[i, 0, :, 1], color=plt.cm.tab10(1), alpha=0.15)
    ax.plot(traj[i, 1, :, 2], traj[i, 1, :, 1], color=plt.cm.tab10(1), alpha=0.15)

  ax.set_ylim([-1.25 * H, 1.25 * H])
  ax.set_xlim([-6.0, 6.0])
  ax.set_ylabel('y-axis')
  ax.set_xlabel('z-axis')

  ax = axes[1, 1]
  ax.bar(pos, angles + np.pi / 3, width=0.05, color=plt.cm.tab10(1), bottom=-np.pi / 3)
  ax.plot([-6.0, 6.0], [0.0, 0.0], color='black', linestyle='--')
  ax.set_ylim([-np.pi / 3, np.pi / 3])
  ax.set_xlim([-6.0, 6.0])

  ax.set_ylabel("layer's angle")
  ax.set_xlabel('z-axis')

  fig.tight_layout()
  fig.savefig(report)
  plt.close(fig)

  if aux is not None:
    checkpointer = detopt.utils.io.get_checkpointer(aux)
    aux = detopt.utils.io.restore_aux(checkpointer)

  losses = aux['regressor']['validation']
  mean = np.mean(losses[-1])
  std = np.std(losses[-1])
  error = std / np.sqrt(1 + np.prod(losses.shape[1:]))
  print(f'{mean:.3f} +- {error:.3f}')

if __name__ == '__main__':
  import gearup
  gearup.gearup(viz=viz, compare=compare).with_config('config/config.yaml')()