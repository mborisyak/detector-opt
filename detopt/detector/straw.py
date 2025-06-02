import math
import numpy as np

from ..utils.encoding import uniform_to_normal, normal_to_uniform
from .common import Detector
from . import straw_detector

__all__ = [
  'StrawDetector'
]

INV_SQRT_2 = math.sqrt(0.5)

class StrawDetector(Detector):
  def __init__(
    self,
    max_B: float=0.5, L=1.0,
    layer_bounds: tuple[float | int, float | int]=(-5.0, 5.0),
    layer_width: float=5, layer_height: float=5,
    angles_bounds: tuple[float | int, float | int]=(-1.0471975511965976, 1.0471975511965976),
    dt: float=1.0e-2,
    n_layers: int=16, n_straws: int=32,
    max_particles=2,
    origin=(0.0, 0.0, -10.0), origin_sigma=(1.0, 1.0, 1.0),
    momentum=(0.0, 0.0, 5.0), momentum_sigma=(0.25, 0.25, 0.5),
    noise_origin=(0.0, 0.0, -10.0), noise_origin_sigma=(1.0, 1.0, 1.0),
    noise_momentum=(0.0, 0.0, 5.0), noise_momentum_sigma=(0.25, 0.25, 0.5),
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
    self.angle_bounds = angles_bounds
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

  def output_shape(self):
    ### positions + angles + magnetic field strength
    return (self.n_layers, self.n_straws)

  def target_shape(self):
    ### charges + momenta + initial positions
    return (self.max_particles + 3 * self.max_particles + 3 * self.max_particles, )

  def ground_truth_shape(self):
    return ()

  def get_design(self, design: np.ndarray[tuple[int, int], np.dtype[np.float32]]):
    n, _ = design.shape
    m = self.n_layers

    design_decoded = self._decode_design(design)

    #layers = normal_to_uniform(positions_normalized, *self.layer_bounds).astype(np.float32)
    layers = design_decoded['positions']
    assert layers.shape == (n, m), layers.shape
    assert np.all(np.isfinite(layers)), f'NaN decoding positions, {layers}'
    ### overlapping angles for easier optimization
    # angles = normal_to_uniform(angles_normalized, *self.angle_bounds).astype(np.float32)
    angles = design_decoded['angles']
    assert np.all(np.isfinite(angles)), f'NaN decoding angles, {angles}'

    widths = self.layer_width + np.zeros(shape=(n, m), dtype=np.float32)
    heights = self.layer_height + np.zeros(shape=(n, m), dtype=np.float32)

    #Bs = normal_to_uniform(magnetic_strength_normalized, 0.0, self.max_B).astype(np.float32)
    Bs = design_decoded['magnetic_strength']
    assert np.all(np.isfinite(Bs)), f'NaN decoding B, {Bs}'
    Ls = self.L + np.zeros(shape=(n,), dtype=np.float32)

    return layers, angles, widths, heights, Bs, Ls

  def sample(self, seed: int, design: np.ndarray):
    ### the first two particles reserve for signal
    ### the rest is sampled as noise
    rng = np.random.default_rng(seed)

    n, _ = design.shape

    layers, angles, widths, heights, Bs, Ls = self.get_design(design)

    masses = np.ones(shape=(n ,self.max_particles, ), dtype=np.float32)
    charges = (2 * rng.binomial(n=1, p=0.5, size=(n, self.max_particles)) - 1).astype(np.float32)

    initial_positions = np.ndarray(shape=(n, self.max_particles, 3), dtype=np.float32)
    initial_momentum = np.ndarray(shape=(n, self.max_particles, 3), dtype=np.float32)

    u = rng.uniform(low=0.0, high=1.0, size=(n, ))
    # included = (rng.uniform(low=0.0, high=1.0, size=(n, self.max_particles - 2)) < u[:, None]).astype(dtype=np.float32)
    included = np.zeros(shape=(n, self.max_particles - 2), dtype=np.float32)

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
    # initial_momentum[:, :2, :] = signal[:, None, None] * signal_momentum + \
    #                              (1 - signal)[:, None, None] * noise_momentum[:, :2, :]
    initial_momentum[:, :2, :] = signal_momentum
    charges[:, 0] = -charges[:, 1]

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
    measurements = rng.poisson(combined_response, size=combined_response.shape) / (self.straw_signal_rate + self.straw_noise_rate)

    # measurements = np.where(np.sum(response, axis=1) > 0.5, 1.0, 0.0)

    return masses, charges, initial_positions, initial_momentum, trajectories, measurements, signal

  def encode_ground_truth(self, masses, charges, initial_positions, initial_momentum):
    n, *_ = initial_positions.shape
    normalized_positions = (initial_positions - self.origin) / self.origin_sigma
    normalized_positions = np.reshape(normalized_positions, shape=(n, -1))
    normalized_momenta = np.reshape(initial_momentum, shape=(n, -1))
    ground_truth = np.concatenate([charges, normalized_positions, normalized_momenta], axis=-1)
    return ground_truth

  def __call__(self, seed: int, configurations: np.ndarray):
    masses, charges, initial_positions, initial_momentum, _, measurements, signal = self.sample(seed, configurations)
    ground_truth = self.encode_ground_truth(masses, charges, initial_positions, initial_momentum)
    return ground_truth, measurements, signal

  def loss(self, target, predicted):
    import optax
    return optax.sigmoid_binary_cross_entropy(predicted, target)

  def encode_design(self, design):
    positions = np.array(design['positions'], dtype=np.float32)
    positions = uniform_to_normal(positions, *self.layer_bounds)
    angles = np.array(design['angles'], dtype=np.float32)
    angles = uniform_to_normal(angles, *self.angle_bounds)
    magnetic_strength = np.array(design['magnetic_strength'], dtype=np.float32)
    magnetic_strength = uniform_to_normal(magnetic_strength, 0.0, self.max_B)

    return np.concatenate([positions, angles, magnetic_strength[None]], axis=0)

  def _decode_design(self, encoded_design):
    n = self.n_layers
    positions = normal_to_uniform(encoded_design[..., :n], *self.layer_bounds)
    angles = normal_to_uniform(encoded_design[..., n:-1], *self.angle_bounds)
    magnetic_strength = normal_to_uniform(encoded_design[..., -1], 0.0, self.max_B)

    return dict(
      positions=positions,
      angles=angles,
      magnetic_strength=magnetic_strength
    )

  def decode_design(self, encoded_design):
    decoded = self._decode_design(encoded_design)

    return dict(
      positions=[float(p) for p in decoded['positions']],
      angles=[float(a) for a in decoded['angles']],
      magnetic_strength=float(decoded['magnetic_strength'])
    )


