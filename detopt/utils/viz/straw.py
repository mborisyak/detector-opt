import numpy as np
import pyvista as pv

# Visualization utility for straw tracker geometry.
# For fast layout checking, reduce n_layers and n_straws in your config,
# or limit the number of layers/straws rendered in the code below.

__all__ = [
  'show'
]

def show(layers, width, height, angles, response, trajectories=None, threshold=0.5):
  z_min, z_max = np.min(layers), np.max(layers)
  z_delta = z_max - z_min
  z_min, z_max = z_min - 0.1 * z_delta, z_max + 0.1 * z_delta

  plotter = pv.Plotter(off_screen=False)
  plotter.camera_position = [
    (-100, 0, 50),
    (0, 0, 0),
    (0, 1, 0),
  ]

  if response.ndim == 3:
    _, n_layers, n_straws = response.shape
    combined_response = np.sum(response, axis=0)
  elif response.ndim == 2:
    n_layers, n_straws = response.shape
    combined_response = response
  else:
    raise ValueError('response must be either 3D (per-particle) or 2D (combined) array')

  combined_response = combined_response / np.max(combined_response)

  for i, l_z in enumerate(layers):
    A = np.array([
      [np.cos(angles[i]), np.sin(angles[i]), 0],
      [-np.sin(angles[i]), np.cos(angles[i]), 0],
      [0, 0, 1],
    ])

    h, w = height[i], width[i]
    r = h / n_straws

    # Draw parallelogram for stereo views (turned), rectangle for straight
    skew = h * np.tan(angles[i])
    verts = np.array([
      [-w - skew, -h, l_z - z_min],   # bottom left
      [-w + skew,  h, l_z - z_min],   # top left
      [ w + skew,  h, l_z - z_min],   # top right
      [ w - skew, -h, l_z - z_min]    # bottom right
    ])
    verts = np.dot(verts, A)
    faces = np.array([[4, 0, 1, 2, 3]])
    mesh = pv.PolyData(verts, faces=faces)
    plotter.add_mesh(mesh, color='black', style='wireframe', opacity=0.5, line_width=1.0)

    for k in range(n_straws):
      verts = np.array([
        [-w, 2 * r * k - h + r, l_z - z_min],
        [w, 2 * r * k - h + r, l_z - z_min],
      ])
      verts = np.dot(verts, A)

      if combined_response[i, k] > threshold:
        mesh = pv.lines_from_points(verts).tube(radius=r, n_sides=5)
        plotter.add_mesh(
          mesh, color="red", show_edges=False, opacity=float(combined_response[i, k])
        )

  if trajectories is not None:
    n_particles, n_t, _ = trajectories.shape

    for i in range(n_particles):
      indices, = np.where(np.logical_and(trajectories[i, :, 2] < z_max, trajectories[i, :, 2] > z_min))
      trajectory = trajectories[i, indices, :]
      n_samples = min(512, trajectory.shape[0])
      indices = np.linspace(0, trajectory.shape[0] - 1, n_samples).astype(int)
      trajectory = trajectory[indices, :]
      trajectory = trajectory -np.array([0, 0, z_min])

      if trajectory.shape[0] > 0:
        traj = pv.Spline(trajectory)#.tube(radius=0.05)
        plotter.add_mesh(traj, color='blue', line_width=4, opacity=0.5)

  plotter.enable_depth_peeling()
  plotter.show_grid()
  plotter.reset_camera()
  plotter.show(screenshot='straw.png')
