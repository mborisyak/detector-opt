import numpy as np

float64_array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

def solve(
  initial_positions: float64_array,
  initial_velocities: float64_array,
  masses: float64_array,
  charges: float64_array,

  B: float64_array,
  B_z0: float64_array,
  B_sigma: float64_array,

  n_steps: int, dt: float,
  layers: float64_array,
  width: float64_array,
  heights: float64_array,
  angles: float64_array,
  trajectories: float64_array | None,
  response: float64_array,

  e_dep: float64_array, r_mm: float64_array,
  t0: float64_array, hit_pos: float64_array
) -> int: ...