import os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

import detopt

def viz(seed=123, design='data/design/default/straw.json', n_events: int=10, **config):
    detector: detopt.detector.StrawDetector = detopt.detector.from_config(config['detector'])

    root = os.path.dirname(os.path.dirname(__file__))

    # Always load the design file for geometry
    with open(os.path.join(root, design), 'r') as f:
      import json
      design_vec = detector.encode_design(json.load(f))
      design_vec = np.repeat(design_vec[None], 100, axis=0)

      x, y = detector(seed=1, configurations=design_vec)
      print(x.shape, y.shape)

      print('x', np.mean(x), np.std(x))
      print('y', np.mean(y), np.std(y))

      ts, response, trajectories, hnl_positions, hnl_momenta = detector.sample(
        seed=1, design=design_vec, compute_trajectories=True
      )
      layer_positions, layer_widths, layer_heights, layer_angles, B, B_z0, B_sigma = detector.get_geometry(design=design_vec)
      for i in range(n_events):
        print(np.median(response[i]), np.max(response[i]))

        if np.max(response[i]) > 10 * detector.straw_noise_rate:
          detopt.utils.viz.straw.show(
            layer_positions[0], layer_widths[0], layer_heights[0], layer_angles[0],
            response[i], trajectories[i],
            threshold=0.1
          )
        else:
          print(i, 'HNL', hnl_positions[i], hnl_momenta[i])
          detopt.utils.viz.straw.show(
            layer_positions[0], layer_widths[0], layer_heights[0], layer_angles[0],
            response[i], trajectories[i],
            threshold=1.0
          )

def compare(design, reference='data/design/default.json', report='designs.png', aux=None, seed=123456789, **config):
  import matplotlib.pyplot as plt
  import json

  detector: detopt.detector.StrawDetector = detopt.detector.from_config(config['detector'])
  L = detector.B_sigma
  W = detector.layer_width
  H = detector.layer_height

  with open(design, 'r') as f:
    design = json.load(f)

  rng = np.random.default_rng(seed)

  with open(reference, 'r') as f:
    reference = json.load(f)

  pos = np.array(design['stations'])
  pos_ref = np.array(reference['stations'])

  z_min, z_max = min(np.min(pos), np.min(pos_ref)), max(np.max(pos), np.max(pos_ref))
  z_delta = (z_max - z_min)
  z_min, z_max = z_min - 0.1 * z_delta, z_max + 0.1 * z_delta

  station_w = 100
  view_w = station_w / detector.n_views_per_station / 2
  view_offsets = np.linspace(0, station_w, num=detector.n_views_per_station)
  view_pos = (pos[:, None] + view_offsets[None, :]).ravel()
  view_pos_ref = (pos_ref[:, None] + view_offsets[None, :]).ravel()

  angles = np.array(design['views'])
  angles_ref = np.array(reference['views'])

  a_delta = max(np.max(np.abs(angles)), np.max(np.abs(angles_ref)))
  a_delta = 1.1 * a_delta

  B0 = detector.B_z0
  B = design['magnetic_strength']
  B_ref = reference['magnetic_strength']

  ys = np.array([-H, H])[:, None] + 0 * view_pos[None, :]
  xs = np.array([-W, W])[:, None] + 0 * view_pos[None, :]

  zs = np.linspace(z_min, z_max, num=128)
  Bs_ref = B_ref * np.exp(-np.square((zs - B0) / L))
  Bs = B * np.exp(-np.square((zs - B0) / L))
  max_B = max(np.max(Bs_ref), np.max(Bs))

  def sample(d, n):
    encoded = detector.encode_design(d)
    configs = np.broadcast_to(encoded[None], shape=(n, *encoded.shape))
    response, signal, trajectories, _, _ = detector.sample(seed=seed, design=configs, compute_trajectories=True)
    return trajectories

  fig = plt.figure(figsize=(12, 6))
  axes = fig.subplots(2, 2)

  traj = sample(reference, 32)

  ax = axes[0, 0]
  ax.set_title('default design (side view)')
  ax.plot(np.stack([view_pos_ref, view_pos_ref]), ys, color=plt.cm.tab10(0))
  # twin = ax.twinx()
  # twin.plot(zs, Bs_ref, color='black')
  # twin.set_ylim([-0.05, 1.05 * max_B])
  # twin.set_ylabel('magnetic field, $B_x$')

  for i, event in enumerate(traj):
    for j in range(event.shape[0]):
      ax.plot(event[j, :, 2], event[j, :, 1], color=plt.cm.tab10(0), alpha=0.15)
      ax.plot(event[j, :, 2], event[j, :, 1], color=plt.cm.tab10(0), alpha=0.15)

  ax.set_ylim([-1.25 * H, 1.25 * H])
  ax.set_xlim([z_min, z_max])
  ax.set_ylabel('y-axis')
  ax.set_xlabel('z-axis')

  ax = axes[0, 1]
  ax.bar(view_pos_ref, angles_ref + a_delta, width=view_w, color=plt.cm.tab10(0), bottom=-a_delta)
  ax.plot([z_min, z_max], [0.0, 0.0], color='black', linestyle='--')
  ax.set_ylim([-a_delta, a_delta])
  ax.set_xlim([z_min, z_max])

  ax.set_ylabel("layer's angle")
  ax.set_xlabel('z-axis')

  ax = axes[1, 0]
  ax.set_title('optimized design (side view)')
  ax.plot(np.stack([view_pos, view_pos]), ys, color=plt.cm.tab10(1))
  # twin = ax.twinx()
  # twin.plot(zs, Bs, color='black')
  # twin.set_ylim([-0.05, 1.05 * max_B])
  # twin.set_ylabel('magnetic field, $B_x$')

  traj = sample(design, 32)
  for i, event in enumerate(traj):
    for j in range(event.shape[0]):
      ax.plot(event[j, :, 2], event[j, :, 1], color=plt.cm.tab10(1), alpha=0.15)
      ax.plot(event[j, :, 2], event[j, :, 1], color=plt.cm.tab10(1), alpha=0.15)

  ax.set_ylim([-1.25 * H, 1.25 * H])
  ax.set_xlim([z_min, z_max])
  ax.set_ylabel('y-axis')
  ax.set_xlabel('z-axis')

  ax = axes[1, 1]
  ax.bar(view_pos, angles + a_delta, width=view_w, color=plt.cm.tab10(0), bottom=-a_delta)
  ax.plot([z_min, z_max], [0.0, 0.0], color='black', linestyle='--')
  ax.set_ylim([-a_delta, a_delta])
  ax.set_xlim([z_min, z_max])

  ax.set_ylabel("layer's angle")
  ax.set_xlabel('z-axis')

  fig.tight_layout()
  fig.savefig(report)
  plt.close(fig)

  # if aux is not None:
  #   checkpointer = detopt.utils.io.get_checkpointer(aux)
  #   aux = detopt.utils.io.restore_aux(checkpointer)
  #
  # losses = aux['regressor']['validation']
  # mean = np.mean(losses[-1])
  # std = np.std(losses[-1])
  # error = std / np.sqrt(1 + np.prod(losses.shape[1:]))
  # print(f'{mean:.3f} +- {error:.3f}')

if __name__ == '__main__':
  import gearup
  gearup.gearup(viz=viz, compare=compare).with_config('config/config.yaml')()
