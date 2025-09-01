import math
import numpy as np
import subprocess
import os
# import ROOT

# For reading ROOT files
try:
    import uproot
except ImportError:
    uproot = None

from ..utils.encoding import uniform_to_normal, normal_to_uniform
from .common import Detector
from . import straw_detector

__all__ = [
  'StrawDetector'
]

INV_SQRT_2 = math.sqrt(0.5)

NAME2PID = {
    "mu-": 13,   "mu+": -13,
    "e-": 11,    "e+": -11,
    "pi-": -211, "pi+": 211,
    "gamma": 22,
    "proton": 2212, "antiproton": -2212,
    "neutron": 2112, "antineutron": -2112,
    "nu_e": 12, "nu_e_bar": -12,
    "nu_mu": 14, "nu_mu_bar": -14,
}

class StrawDetector(Detector):
  def __init__(
    self,
    # Geometry hierarchy
    # Real detector geometry
    station_z: list = [2598.0, 2698.0, 3498.0, 3538.0],
    n_views_per_station: int = 4,
    n_layers_per_view: int = 4,
    n_straws_per_layer: int = 200,
    straw_pitch: float = 2.0,
    straw_length: float = 200.0,
    layer_x_offset: float = 1.0,
    view_angles: tuple = (0.0, 0.0798,  -0.0798, 0.0),  # X, U, X', V
    layer_z_gap: float = 1.732,
    view_z_gap: float = 5.0,
    # Physics parameters
    max_B: float=0.5, L=1.0,
    z0: float=None, B_sigma: float=None,
    layer_bounds: tuple[float | int, float | int]=(-5.0, 5.0),
    dt: float=1.0e-2,
    max_particles=2,
    origin=(-100.0, -100.0, 1700.0), origin_sigma=(1.0, 1.0, 1.0),
    momentum=(0.0, 0.0, 5.0), momentum_sigma=(0.25, 0.25, 0.5),
    noise_origin=(0.0, 0.0, -10.0), noise_origin_sigma=(1.0, 1.0, 1.0),
    noise_momentum=(0.0, 0.0, 5.0), noise_momentum_sigma=(0.25, 0.25, 0.5),
    straw_signal_rate=200.0,
    straw_noise_rate=10.0,
    angles_bounds=None,  # <-- ADD THIS LINE
    layer_width=None,
    layer_height=None,
    n_layers=None,
    n_straws=None,

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
    self.z0 = z0
    self.B_sigma = B_sigma

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

    # Real detector geometry
    self.station_z = station_z
    self.n_stations = len(station_z)
    self.n_views_per_station = n_views_per_station
    self.n_layers_per_view = n_layers_per_view
    self.n_straws = n_straws_per_layer
    self.straw_pitch = straw_pitch
    self.straw_length = straw_length
    self.layer_x_offset = layer_x_offset
    self.view_angles = view_angles
    self.layer_z_gap = layer_z_gap
    self.view_z_gap = view_z_gap

    self.n_layers = self.n_stations * self.n_views_per_station * self.n_layers_per_view

    # Layer height/width for visualization/hit logic
    self.layer_height = self.straw_pitch * self.n_straws / 2.0  # half-length for +/- y
    self.layer_width = self.straw_length / 2.0  # half-length for +/- x

    # Angle bounds: use legacy if provided, else default
    if angles_bounds is not None:
        self.angle_bounds = angles_bounds
    else:
        self.angle_bounds = (-0.1, 0.1)

    assert max_particles > 1, 'signal events produce at least 2 particles'
    self.max_particles = max_particles
    assert self.max_particles == 2, 'this is temporary'

    flight_distance = layer_bounds[1] - layer_bounds[0]

    self.n_t = int(flight_distance / dt)
    self.straw_signal_rate = straw_signal_rate
    self.straw_noise_rate = straw_noise_rate
    print("\n\n\nSet up\n\n\n")

  def design_shape(self):
    ### positions + angles + magnetic field strength
    return (self.n_layers + self.n_layers + 1, )

  def output_shape(self):
    ### positions + angles + magnetic field strength
    return (self.n_layers, self.n_straws)

  def target_shape(self):
    ### charges + momenta + initial positions
    return ()

  def ground_truth_shape(self):
    return (self.max_particles + 3 * self.max_particles + 3 * self.max_particles, )

  def get_design(self, design: np.ndarray[tuple[int, int], np.dtype[np.float32]]):
    n, _ = design.shape
    m = self.n_layers

    print(n, m, end='\n\n\n')

    design_decoded = self._decode_design(design)

    # Real detector geometry calculation
    layer_positions = []
    layer_angles = []
    layer_x_offsets = []
    for s, station_z in enumerate(self.station_z):
        for v in range(self.n_views_per_station):
            view_angle = self.view_angles[v]
            view_z = station_z + v * self.view_z_gap
            for l in range(self.n_layers_per_view):
                layer_z = view_z + l * self.layer_z_gap
                x_offset = l * self.layer_x_offset
                layer_positions.append(layer_z)
                layer_angles.append(view_angle)
                layer_x_offsets.append(x_offset)
    # Repeat for batch dimension
    layers = np.array(layer_positions, dtype=np.float32)[None, :].repeat(n, axis=0)
    angles = np.array(layer_angles, dtype=np.float32)[None, :].repeat(n, axis=0)
    widths = self.layer_width + np.zeros(shape=(n, m), dtype=np.float32)
    heights = self.layer_height + np.zeros(shape=(n, m), dtype=np.float32)
    # Optionally: x_offsets = np.array(layer_x_offsets, dtype=np.float32)[None, :].repeat(n, axis=0)
    # ... Bs, Ls as before ...
    # Return arrays as before

    Bs = design_decoded['magnetic_strength']
    assert np.all(np.isfinite(Bs)), f'NaN decoding B, {Bs}'
    Ls = self.L + np.zeros(shape=(n,), dtype=np.float32)

    return layers, angles, widths, heights, Bs, Ls

  def sample(self, seed: int, design:  np.ndarray):
    ### the first two particles reserve for signal
    ### the rest is sampled as noise
    rng = np.random.default_rng(seed)

    n, _ = design.shape
    m = 2

    layers, angles, widths, heights, Bs, Ls = self.get_design(design)

    masses = np.ones(shape=(n, m, ), dtype=np.float32)
    charges = (2 * rng.binomial(n=1, p=0.5, size=(n, m)) - 1).astype(np.float32)

    # initial_positions = np.ndarray(shape=(n, m, 3), dtype=np.float32)
    # initial_momentum = np.ndarray(shape=(n, m, 3), dtype=np.float32)

    noise_positions = self.noise_origin_sigma * rng.normal(size=(n, m, 3)) + self.noise_origin
    signal_positions = self.origin_sigma * rng.normal(size=(n, 3)) + self.origin
    ### the total momentum would be ~ N(m, sigma^2)
    initial_momentum = INV_SQRT_2 * self.momentum_sigma * rng.normal(size=(n, m, 3)) + 0.5 * self.momentum

    signal = rng.binomial(n=1, p=0.5, size=(n,)).astype(dtype=np.float32)

    initial_positions = signal[:, None, None] * signal_positions[:, None, :] + (1 - signal)[:, None, None] * noise_positions
    initial_positions = initial_positions.astype(np.float32)
    initial_momentum = initial_momentum.astype(np.float32)

    # initial_positions[:, 2:, :] = included[:, :, None] * noise_positions[:, 2:]
    # initial_momentum[:, 2:, :] = included[:, :, None] * noise_momentum[:, 2:]
    #
    # initial_positions[:, :2, :] = signal[:, None, None] * signal_positions[:, None, :] + \
    #                               (1 - signal)[:, None, None] * noise_positions[:, :2, :]
    # # initial_momentum[:, :2, :] = signal[:, None, None] * signal_momentum + \
    # #                              (1 - signal)[:, None, None] * noise_momentum[:, :2, :]
    # initial_momentum[:, :2, :] = signal_momentum
    charges[:, 0] = -charges[:, 1]

    momentum_norm_sqr = np.sum(np.square(initial_momentum), axis=-1)
    initial_velocities = initial_momentum / np.sqrt(np.square(masses) + momentum_norm_sqr)[..., None]

    trajectories = np.zeros(shape=(n, m, self.n_t, 3), dtype=np.float32)
    response = np.zeros(shape=(n, m, self.n_layers, self.n_straws), dtype=np.float32)

    # Magnetic field parameters for the batch
    if self.z0 is not None:
        z0_arr = np.full((n,), self.z0, dtype=np.float32)
    else:
        z0_arr = np.mean(layers, axis=1).astype(np.float32)
    if self.B_sigma is not None:
        B_sigma_arr = np.full((n,), self.B_sigma, dtype=np.float32)
    else:
        B_sigma_arr = Ls.astype(np.float32)

    straw_detector.solve(
      initial_positions,  initial_momentum,
      masses, charges, Bs, Ls,
      z0_arr, B_sigma_arr,
      self.n_t, self.dt, layers, widths, heights,
      angles, trajectories, response
    )

    combined_response = np.sum(response, axis=1) * self.straw_signal_rate + self.straw_noise_rate
    measurements = rng.poisson(combined_response, size=combined_response.shape) / (self.straw_signal_rate + self.straw_noise_rate)

    # measurements = np.where(np.sum(response, axis=1) > 0.5, 1.0, 0.0)
    print( 'ip',initial_positions, 'iv',initial_velocities,
    'm',masses,'c',  charges,  'b',Bs, Ls,
    self.n_t, self.dt, layers, widths, heights,
    angles, trajectories, response,sep='\n')

    return masses, charges, initial_positions, initial_momentum, trajectories, measurements, signal




  def encode_ground_truth(self, masses, charges, initial_positions, initial_momentum):
    n, *_ = initial_positions.shape
    normalized_positions = (initial_positions - self.origin) / self.origin_sigma
    normalized_positions = np.reshape(normalized_positions, shape=(n, -1))
    normalized_momenta = np.reshape(initial_momentum, shape=(n, -1))
    ground_truth = np.concatenate([charges, normalized_positions, normalized_momenta], axis=-1)
    return ground_truth

  def simulate_from_root(self, rootfile, tree_name="Events", px_name="px", py_name="py", pz_name="pz",
                        x_name="x", y_name="y", z_name="z", pid_name="pid", batch_size=None, design=None):
    """
    Load particle properties from a ROOT file and run the detector simulation.

    Args:
        rootfile: Path to the ROOT file.
        tree_name: Name of the TTree.
        px_name, py_name, pz_name: Branch names for momentum.
        x_name, y_name, z_name: Branch names for position.
        pid_name: Branch name for particle ID.
        batch_size: Number of events to process at once (None = all).
        design: Detector design/configuration (if None, use default).

    Returns:
        Output of the detector simulation (trajectories, response, etc.)
    """
    if uproot is None:
        raise ImportError("uproot is required to read ROOT files. Please install with `pip install uproot awkward`.")

    n_viz = 10
    with uproot.open(rootfile) as file:
        tree = file[tree_name]
        px = tree[px_name].array(library="np")[:n_viz]
        py = tree[py_name].array(library="np")[:n_viz]
        pz = tree[pz_name].array(library="np")[:n_viz]
        x = tree[x_name].array(library="np")[:n_viz]
        y = tree[y_name].array(library="np")[:n_viz]
        z = tree[z_name].array(library="np")[:n_viz]
        pid = tree[pid_name].array(library="np")[:n_viz]

    # Treat all loaded particles as a single event with n_particles
    n_particles = px.shape[0]
    n_events = 1

    initial_positions = np.stack([x, y, z], axis=-1).reshape((1, n_particles, 3)).astype(np.float32)
    initial_momentum = np.stack([px, py, pz], axis=-1).reshape((1, n_particles, 3)).astype(np.float32)
    masses = np.ones((1, n_particles), dtype=np.float32)

    def pid_to_charge(pid: int) -> float:
        CHARGE = {
            11: -1.0,  -11: +1.0,    # e-, e+
            13: -1.0,  -13: +1.0,    # mu-, mu+
            211: +1.0, -211: -1.0,   # pi+, pi-
            2212: +1.0, -2212: -1.0, # p, pbar
            2112: 0.0, -2112: 0.0,   # n, nbar
            22: 0.0,                 # gamma
            12: 0.0,  -12: 0.0,      # nu_e, anti
            14: 0.0,  -14: 0.0,      # nu_mu, anti
        }
        return CHARGE.get(pid, 0.0)

    # --- Rest masses in GeV (use abs(pid) for particle/antiparticle) ---
    def pid_to_mass_GeV(pid: int) -> float:
        MASS = {
            11: 0.000510999,   # e±
            13: 0.1056583755,  # mu±
            211: 0.13957039,   # pi±
            2212: 0.9382720813,# proton/antiproton
            2112: 0.9395654133,# neutron/antineutron
            22: 0.0,           # gamma
            12: 0.0,  # ν_e (set ~0 for toy; tiny real mass irrelevant here)
            14: 0.0,  # ν_μ
        }
        return MASS.get(abs(pid), 0.0)

    # --- Example: build pid / mass / charge arrays for your particles ---
    # Suppose you already have a (1, n_particles) array of pids:
    # pids = np.array([[211, -211, 13, -13, 11, -11, 2212, 2112, 22, 12, -12, 14, -14]], dtype=np.int32)

    def build_masses_and_charges(pids: np.ndarray):
        pid_vec = np.vectorize  # convenience
        masses = pid_vec(pid_to_mass_GeV)(pids).astype(np.float32)   # shape (1, n_particles)
        charges = pid_vec(pid_to_charge)(pids).astype(np.float32)    # shape (1, n_particles)
        if masses.ndim == 1:
                masses = masses.reshape((1, -1))
                charges = charges.reshape((1, -1))
        return masses, charges




    masses, charges = build_masses_and_charges(pid)
    print("masses shape:", masses.shape, "dtype:", masses.dtype)
    print("charges shape:", charges.shape, "dtype:", charges.dtype)

    # momentum_norm_sqr = np.sum(np.square(initial_momentum), axis=-1)
    # initial_velocities = initial_momentum / np.sqrt(np.square(masses) + momentum_norm_sqr)[..., None]


    if design is None:
        # Use a default design (single batch)
        design = np.zeros((1, self.design_dim()), dtype=np.float32)
    layers, angles, widths, heights, Bs, Ls = self.get_design(design)


    # Allocate output arrays
    trajectories = np.zeros((n_events, n_particles, self.n_t, 3), dtype=np.float32)
    response = np.zeros((n_events, n_particles, self.n_layers, self.n_straws), dtype=np.float32)

    print("response shape:", response.shape, "dtype:", response.dtype)
    print("trajectories shape:", trajectories.shape, "dtype:", trajectories.dtype)
    print("initial_positions shape:", initial_positions.shape, "dtype:", initial_positions.dtype)

    # Call the C extension directly
    from . import straw_detector

    # Magnetic field parameters for the batch
    if self.z0 is not None:
        z0_arr = np.full((n_events,), self.z0, dtype=np.float32)
    else:
        z0_arr = np.mean(layers, axis=1).astype(np.float32)
    if self.B_sigma is not None:
        B_sigma_arr = np.full((n_events,), self.B_sigma, dtype=np.float32)
    else:
        B_sigma_arr = Ls.astype(np.float32)

    straw_detector.solve(
        initial_positions, initial_momentum, masses, charges, Bs, Ls,
        z0_arr, B_sigma_arr,
        self.n_t, self.dt, layers, widths, heights, angles, trajectories, response
    )

    signal = np.ones((n_events,), dtype=np.float32)
    return masses, charges, initial_positions, initial_momentum, trajectories, response, signal

  def __call__(self, seed: int, configurations: np.ndarray):
    masses, charges, initial_positions, initial_momentum, _, measurements, signal = self.sample(seed, configurations)
    ground_truth = self.encode_ground_truth(masses, charges, initial_positions, initial_momentum)
    return ground_truth, measurements, signal

  def loss(self, target, predicted):
    import optax
    return optax.sigmoid_binary_cross_entropy(predicted, target)

  def metric(self, target, predicted):
    return (target > 0.5) == (predicted > 0.0)

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
