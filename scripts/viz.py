import os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

import detopt

def viz(seed=123, design='data/design/default.json', use_root_particles=True, **config):
    n_batch = 1
    n_layers = 64

    detector = detopt.detector.from_config(config['detector'])

    root = os.path.dirname(os.path.dirname(__file__))

    # Always load the design file for geometry
    with open(os.path.join(root, design), 'r') as f:
      import json
      design_vec = detector.encode_design(json.load(f))

    if use_root_particles:
      rootfile = "/Users/nikitagladin/SHiP/clean.root"
      print(f"Loading particles from ROOT file: {rootfile}")
      masses, charges, initial_positions, initial_momentum, trajectories, response, signal, waveforms, t0_arr, r_mm, fdigi_times = detector.simulate_from_root(
        rootfile,
        tree_name="mytree",
        px_name="px", py_name="py", pz_name="pz",
        x_name="vx", y_name="vy", z_name="vz",
        pid_name="pdgcode",
        design=design_vec[None]
      )
      layers, angles, widths, heights, Bs, Ls = detector.get_design(design=design_vec[None])
      detopt.utils.viz.straw.show(
        layers[0], angles[0], widths[0], heights[0], response[0], trajectories[0], signal[0] if hasattr(signal, '__getitem__') else signal,
        threshold=0.3
      )

      # --- Plot all waveforms on the same canvas without normalization ---
      import matplotlib.pyplot as plt
      import matplotlib.cm as cm
      import numpy as np
      import yaml

      # Draw a separate canvas for each station, visualizing only the layers belonging to that station
      import yaml
      with open("/Users/nikitagladin/SHiP/detector-opt/config/detector/straw.yaml") as f:
          config = yaml.safe_load(f)
      detector_cfg = config["straw"]
      n_stations = len(detector_cfg["station_z"])
      n_views_per_station = detector_cfg["n_views_per_station"]
      n_layers_per_view = detector_cfg["n_layers_per_view"]
      layers_per_station = n_views_per_station * n_layers_per_view

      for station in range(n_stations):
          plt.figure()
          for key, (t, s) in waveforms.items():
              layer = key[2]
              this_station = layer // layers_per_station
              if this_station == station:
                  plt.plot(t, s, label=f"event={key[0]}, particle={key[1]}, layer={layer}, straw={key[3]}")
          plt.xlabel("Time [ns]")
          plt.ylabel("Signal [Coulombs]")
          plt.title(f"Straw hit waveforms for Station {station+1} (not normalized)")
          # plt.legend(fontsize='x-small', ncol=2)
          plt.show()

      # --- Plot histogram of FairShip-style fdigi_times per station ---
      tdc_by_station = [[] for _ in range(n_stations)]
      for key, fdigi in fdigi_times.items():
          layer = key[2]
          station = layer // layers_per_station
          tdc_by_station[station].append(fdigi)
      plt.figure()
      for station in range(n_stations):
          if tdc_by_station[station]:
              plt.hist(tdc_by_station[station], bins=50, alpha=0.6, label=f"Station {station+1}")
      plt.xlabel("TDC time [ns]")
      plt.ylabel("Counts")
      plt.title("Histogram of FairShip-style TDC times per station")
      plt.legend()
      plt.show()

      # --- Compute and plot histogram of TDC times per station ---
      from detopt.detector.straw_signal import compute_tdc_times
      tdc_times = compute_tdc_times(
          waveforms, t0_arr, r_mm,
          straw_length=detector_cfg["straw_length"],
          v_wire=0.2,  # mm/ns, adjust as needed
          t0_event=0.0
      )
      tdc_by_station = [[] for _ in range(n_stations)]
      for key, fdigi in tdc_times.items():
          layer = key[2]
          station = layer // layers_per_station
          tdc_by_station[station].append(fdigi)
      plt.figure()
      # Only include stations with hits
      tdc_data = [tdc_by_station[station] for station in range(n_stations) if tdc_by_station[station]]
      labels = [f"Station {station+1}" for station in range(n_stations) if tdc_by_station[station]]
      plt.hist(tdc_data, bins=50, stacked=True, label=labels, alpha=0.7)
      plt.xlabel("TDC time [ns]")
      plt.ylabel("Counts")
      plt.title("Stacked histogram of TDC times per station")
      plt.legend()
      plt.show()



    configs = np.broadcast_to(design_vec[None], (n_batch, *design_vec.shape))

    import time
    n_trials = 1
    start_time = time.perf_counter()
    for i in range(n_trials):
      _ = detector(seed=1, configurations=configs)
    runtime = time.perf_counter() - start_time
    events_per_second_single_core = n_trials * n_batch / runtime
    print(f'events per second: {events_per_second_single_core}')

    from concurrent.futures import ThreadPoolExecutor

    n_cores = 1 # os.cpu_count()
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
    for i in range(1):
      _, _, _, v0, trajectories, response, signal = detector.sample(seed=i, design=configs)
      v0s.append(v0)
      ys.append(signal)
      # print(i, end='\n')
      # print( *response, *signal, sep='\n', end='\n\n')

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

    # print(layers, angles, widths, heights, Bs, Ls, sep='  \n\n')
    # Bs, Ls - easy
    # angles - easy
    #

    print(response.shape)
    plt.matshow(response[0].T)
    plt.colorbar()
    plt.show()
    plt.close()

    detopt.utils.viz.straw.show(
      layers[0], angles[0], widths[0], heights[0], response[0], trajectories[0], signal[0],
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
