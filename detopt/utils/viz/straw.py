import numpy as np
import pyvista as pv

# Visualization utility for straw tracker geometry.
# For fast layout checking, reduce n_layers and n_straws in your config,
# or limit the number of layers/straws rendered in the code below.

__all__ = [
  'show'
]

def show(layers, angles, width, height, response, trajectories=None, signal=None, threshold=1.0):
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

  # Limit to first 3 layers for fast layout checking
  max_layers = len(layers)
  max_straws = 10
  for i, l_z in enumerate(layers[:max_layers]):
    A = np.array([
      [np.cos(angles[i]), np.sin(angles[i]), 0],
      [-np.sin(angles[i]), np.cos(angles[i]), 0],
      [0, 0, 1],
    ])

    h, w = height[i], width[i]
    r = h / max_straws

    # Draw parallelogram for stereo views (turned), rectangle for straight
    skew = h * np.tan(angles[i])
    verts = np.array([
      [-w - skew, -h, l_z],   # bottom left
      [-w + skew,  h, l_z],   # top left
      [ w + skew,  h, l_z],   # top right
      [ w - skew, -h, l_z]    # bottom right
    ])
    verts = np.dot(verts, A)
    faces = np.array([[4, 0, 1, 2, 3]])
    mesh = pv.PolyData(verts, faces=faces)
    plotter.add_mesh(mesh, color='black', style='wireframe', opacity=0.5, line_width=1.0)

    for k in range(max_straws):
      verts = np.array([
        [-w, 2 * r * k - h + r, l_z], [w, 2 * r * k - h + r, l_z],
      ])
      verts = np.dot(verts, A)
      mesh = pv.lines_from_points(verts)
      plotter.add_mesh(
        mesh, color=(1.0, 0.0, 0.0), show_edges=False,
        opacity=1.0
      )

  # Comment out trajectory rendering for faster layout visualization
  # if trajectories is not None:
  #   n_particles, n_t, _ = trajectories.shape
  #   if signal is None:
  #     signal = 0.0

  #   for i in range(n_particles):
  #     traj = pv.Spline(trajectories[i])#.tube(radius=0.05, )
  #     plotter.add_mesh(traj, color='red' if signal > 0.5 and i < 2 else 'blue', line_width=4, opacity=0.5)


  plotter.show_grid()
  plotter.reset_camera()
  plotter.show(screenshot='straw.png')
