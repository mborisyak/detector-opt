import math

import jax
import jax.numpy as jnp

import numpy as np
import scipy as sp

from . import straw_detector

__all__ = [
  'StrawDetector'
]

INV_SQRT_2 = math.sqrt(0.5)
SQRT_2 = math.sqrt(2.0)
LOG_2 = math.log(2.0)

def uniform_to_normal(xs, low, high):
  """
  A transformation that converts a uniformly distributed variable U[low, high] into N(0, 1).
  """
  ### uniform = 1 - exp(-x / 2)
  ### normal = erfinv(2 * uniform - 1) * sqrt(2)
  if hasattr(xs, 'dtype'):
    eps = np.finfo(xs.dtype).eps
  else:
    eps = np.finfo(np.float64).eps

  c = (low + high) / 2
  d = (high - low) / 2

  us = np.clip((xs - c) / d, eps - 1, 1 - eps)

  return sp.special.erfinv(us) * SQRT_2

def normal_to_uniform(xs, low, high):
  """
  A transformation that converts a normally distributed variable N(0, 1) into U[low, high].
  """
  us = sp.special.erf(xs * INV_SQRT_2,)
  c = (low + high) / 2
  d = (high - low) / 2
  return us * d + c

class StrawDetector(object):
  def __init__(
    self,
    max_B: float=0.5, L=1.0,
    layer_bounds: tuple[float | int, float | int]=(-5.0, 5.0),
    layer_width: float=5, layer_height: float=5,
    dt: float=1.0e-2,
    n_layers: int=16, n_straws: int=32,
    max_particles=8,
    origin=(0.0, 0.0, -10.0), origin_sigma=(0.2, 0.2, 1.0),
    momentum=(0.0, 0.0, 5.0), momentum_sigma=(0.25, 0.25, 0.5),
    noise_origin=(0.0, 0.0, -10.0), noise_origin_sigma=(1.0, 1.0, 1.0),
    noise_momentum=(0.0, 0.0, 5.0), noise_momentum_sigma=(0.2, 0.2, 1.0),
    straw_signal_rate=200.0,
    straw_noise_rate=10.0,
  ):
    """
    :param max_B: maximal strength of the magnetic field;
    :param L: length parameter of the magnetic field;
    :param origin: the mean point of particles' origin;
    :param layer_bounds: restrictions on the layers' positions;
    :param dt: time increment for the ODE solver;
    """

    self.max_B = max_B
    self.L = L

    self.origin = np.array(origin, dtype=np.float32)
    self.origin_sigma = np.array(origin_sigma, np.float32)
    self.momentum = np.array(momentum, dtype=np.float32)
    self.momentum_sigma = np.array(momentum_sigma, dtype=np.float32)

    self.noise_origin = np.array(noise_origin, dtype=np.float32)
    self.noise_origin_sigma = np.array(noise_origin_sigma, np.float32)
    self.noise_momentum = np.array(noise_momentum, dtype=np.float32)
    self.noise_momentum_sigma = np.array(noise_momentum_sigma, dtype=np.float32)

    self.layer_bounds = layer_bounds
    self.dt = dt

    self.layer_width = layer_width
    self.layer_height = layer_height

    self.n_layers = n_layers
    self.n_straws = n_straws
    assert max_particles > 1, 'signal events produce at least 2 particles'
    self.max_particles = max_particles

    flight_distance = max(5 * L, layer_bounds[1]) - origin[2] - 3 * origin_sigma[2]

    self.n_t = int(2 * flight_distance / dt)
    self.straw_signal_rate = straw_signal_rate
    self.straw_noise_rate = straw_noise_rate

  def design_shape(self):
    ### positions + angles + magnetic field strength
    return (self.n_layers + self.n_layers + 1, )

  def design_dim(self):
    return math.prod(self.design_shape())

  def output_shape(self):
    ### positions + angles + magnetic field strength
    return (self.n_layers, self.n_straws)

  def output_dim(self):
    return math.prod(self.output_shape())

  def get_design(self, design: np.ndarray[tuple[int, int], np.dtype[np.float32]]):
    n, _ = design.shape
    m = self.n_layers

    positions_normalized = design[:, :m]
    angles_normalized = design[:, m:-1]
    magnetic_strength_normalized = design[:, -1]

    layers = normal_to_uniform(positions_normalized, *self.layer_bounds).astype(np.float32)
    ### overlapping angles for easier optimization
    angles = normal_to_uniform(angles_normalized, -np.pi, np.pi).astype(np.float32)

    widths = self.layer_width + np.zeros(shape=(n, m), dtype=np.float32)
    heights = self.layer_height + np.zeros(shape=(n, m), dtype=np.float32)

    Bs = normal_to_uniform(magnetic_strength_normalized, 0.0, self.max_B).astype(np.float32)
    Ls = self.L + np.zeros(shape=(n,), dtype=np.float32)

    return layers, angles, widths, heights, Bs, Ls

  def sample(self, seed: int, design: np.ndarray):
    ### the first two particles reserve for signal
    ### the rest is sampled as noise
    rng = np.random.RandomState(seed)

    n, _ = design.shape

    layers, angles, widths, heights, Bs, Ls = self.get_design(design)

    masses = np.ones(shape=(n ,self.max_particles, ), dtype=np.float32)
    charges = (2 * rng.binomial(n=1, p=0.5, size=(n, self.max_particles)) - 1).astype(np.float32)

    initial_positions = np.ndarray(shape=(n, self.max_particles, 3), dtype=np.float32)
    initial_momentum = np.ndarray(shape=(n, self.max_particles, 3), dtype=np.float32)

    u = rng.uniform(low=0.0, high=1.0, size=(n, ))
    included = (rng.uniform(low=0.0, high=1.0, size=(n, self.max_particles - 2)) < u[:, None]).astype(dtype=np.float32)

    noise_positions = self.noise_origin_sigma * rng.normal(size=(n, self.max_particles, 3)) + self.noise_origin
    ### the total momentum would be ~ N(m, sigma^2)
    noise_momentum = self.noise_momentum_sigma * rng.normal(size=(n, self.max_particles, 3)) + self.noise_momentum

    initial_positions[:, 2:, :] = included[:, :, None] * noise_positions[:, 2:]
    initial_momentum[:, 2:, :] = included[:, :, None] * noise_momentum[:, 2:]

    signal_positions = self.origin_sigma * rng.normal(size=(n, 3)) + self.origin
    ### the total momentum would be ~ N(m, sigma^2)
    signal_momentum = INV_SQRT_2 * self.momentum_sigma * rng.normal(size=(n, 2, 3)) + 0.5 * self.momentum

    signal = rng.binomial(n=1, p=0.5, size=(n,)).astype(dtype=np.float32)
    initial_positions[:, :2, :] = signal[:, None, None] * signal_positions[:, None, :] + \
                                  (1 - signal)[:, None, None] * noise_positions[:, :2, :]
    initial_momentum[:, :2, :] = signal[:, None, None] * signal_momentum + \
                                 (1 - signal)[:, None, None] * noise_momentum[:, :2, :]
    charges[:, 0] = -signal * charges[:, 1] + (1 - signal) * charges[:, 0]

    momentum_norm_sqr = np.sum(np.square(initial_momentum), axis=-1)
    initial_velocities = initial_momentum / np.sqrt(np.square(masses) + momentum_norm_sqr)[..., None]

    trajectories = np.zeros(shape=(n, self.max_particles, self.n_t, 3), dtype=np.float32)
    response = np.zeros(shape=(n, self.max_particles, self.n_layers, self.n_straws), dtype=np.float32)

    straw_detector.solve(
      initial_positions, initial_velocities,
      masses, charges, Bs, Ls,
      self.n_t, self.dt, layers, widths, heights,
      angles, trajectories, response
    )

    combined_response = np.sum(response, axis=1) * self.straw_signal_rate + self.straw_noise_rate
    measurements = rng.poisson(combined_response, size=combined_response.shape) / self.straw_signal_rate

    return masses, charges, initial_positions, initial_momentum, trajectories, measurements, signal

  def __call__(self, seed: int, configurations: np.ndarray):
    _, _, _, _, _, measurements, signal = self.sample(seed, configurations)
    return measurements, signal



