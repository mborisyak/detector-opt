from typing import Sequence

import math
import numpy as np

from ..utils.encoding import uniform_to_normal, normal_to_uniform
from .common import Detector
from . import straw_detector
from .utils import load_events, pid_to_charge, pid_to_mass_MeV

__all__ = [
  'SparseStrawDetector'
]

SPEED_OF_LIGHT = 30 # cm / ns
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

ACTIVE_STATION = 0
ACTIVE_VIEW = 1
ACTIVE_LAYER = 2
ACTIVE_TDC = 3
ACTIVE_RESPONSE = 4

class SparseStrawDetector(Detector):
  def __init__(
    self,
    event_path: str,
    # Geometry hierarchy
    # Real detector geometry
    max_size: int=None,
    n_stations: int=4,
    n_views_per_station: int = 4,
    n_layers_per_view: int = 4,
    n_straws_per_layer: int = 200,
    straw_pitch: float = 2.0,
    straw_length: float = 200.0,
    layer_x_offset: float = 1.0,
    layer_z_gap: float = 1.732,
    view_z_gap: float = 5.0,
    # Physics parameters
    max_B: float=0.5,
    B_z0: float=None, B_sigma: float=None,
    station_z_bounds: tuple[float | int, float | int]=(2000.0, 4000.0),
    dt: float=1.0e-2,
    straw_signal_rate=1.0e+7,
    straw_noise_rate: float=1.0e+5,
    straw_time_normalization: float=300.0,
    view_angle_bounds=(-0.2, 0.2),
    charge_threshold: float=0.1,
    min_event_size: int | None=2,
    max_event_size: int | None=None,
    hnl_position_offset: Sequence[float | int] = (0.0, 0.0, 2000.0),
    hnl_position_scale: Sequence[float | int] = (1.0, 1.0, 2000.0),
    hnl_momentum_offset: Sequence[float | int] = (0.0, 0.0, 0.0),
    hnl_momentum_scale: Sequence[float | int] = (0.0, 0.0, 0.0),
  ):
    """
    :param event_path: path to the event file;
    :param max_B: maximal strength of the magnetic field;
    :param L: length parameter of the magnetic field;
    :param origin: the mean point of particles' origin;
    :param station_z_bounds: restrictions on the stations' positions;
    :param dt: time increment for the ODE solver;
    """

    self.min_event_size = min_event_size
    self.max_event_size = max_event_size
    self.events = load_events(event_path, min_event_size=min_event_size, max_event_size=max_event_size)

    self.max_B = max_B
    self.B_z0 = B_z0
    self.B_sigma = B_sigma

    self.station_z_bounds = station_z_bounds
    self.dt = dt

    # Real detector geometry
    self.n_stations = n_stations
    self.n_views_per_station = n_views_per_station
    self.n_layers_per_view = n_layers_per_view
    self.n_straws = n_straws_per_layer
    self.straw_pitch = straw_pitch
    self.straw_length = straw_length
    self.layer_x_offset = layer_x_offset
    self.layer_z_gap = layer_z_gap
    self.view_z_gap = view_z_gap

    self.n_layers = self.n_stations * self.n_views_per_station * self.n_layers_per_view

    # Layer height/width for visualization/hit logic
    self.layer_height = self.straw_pitch * self.n_straws / 2.0  # half-length for +/- y
    self.layer_width = self.straw_length / 2.0  # half-length for +/- x

    self.angle_bounds = view_angle_bounds

    flight_distance = station_z_bounds[1] - station_z_bounds[0]

    self.n_t = int(2 * flight_distance / dt / SPEED_OF_LIGHT) # cm/ns
    self.straw_signal_rate = straw_signal_rate
    self.straw_noise_rate = straw_noise_rate
    self.straw_time_normalization = straw_time_normalization

    self.hnl_position_offset = np.array(hnl_position_offset, dtype=np.float32)
    self.hnl_position_scale = np.array(hnl_position_scale, dtype=np.float32)
    self.hnl_momentum_offset = np.array(hnl_momentum_offset, dtype=np.float32)
    self.hnl_momentum_scale = np.array(hnl_momentum_scale, dtype=np.float32)

    self.view_angle_bounds = view_angle_bounds

    self.view_offsets = (np.arange(self.n_views_per_station)) * (self.view_z_gap + self.layer_z_gap)
    self.layer_offsets = (np.arange(self.n_layers_per_view)) * self.layer_z_gap

    if max_size is None:
      self.max_size = 4 * self.n_stations * self.n_views_per_station * self.n_layers_per_view
    else:
      self.max_size = max_size

  def design_shape(self):
    ### positions + angles + magnetic field strength
    shape = (self.n_stations + self.n_stations * self.n_views_per_station + 1, )
    return shape

  def output_shape(self):
    ### station, view, layer, tdc, signal
    shape = (self.max_size, 5)
    return shape

  def target_shape(self):
    ### HNL's position and momentum
    return (3 + 3, )

  def ground_truth_shape(self):
    ### HNL's position and momentum
    return (3 + 3, )

  def encode_design(self, design):
    stations = np.array(design['stations'], dtype=np.float32)
    stations = uniform_to_normal(stations, *self.station_z_bounds)
    views = np.array(design['views'], dtype=np.float32)
    views = uniform_to_normal(views, *self.view_angle_bounds)
    magnetic_strength = np.array(design['magnetic_strength'], dtype=np.float32)
    magnetic_strength = uniform_to_normal(magnetic_strength, 0.0, self.max_B)

    return np.concatenate([stations, views, magnetic_strength[None]], axis=0)

  def _decode_design(self, encoded_design):
    n = self.n_stations

    stations = normal_to_uniform(encoded_design[..., :n], *self.station_z_bounds)
    views = normal_to_uniform(encoded_design[..., n:-1], *self.view_angle_bounds)
    magnetic_strength = normal_to_uniform(encoded_design[..., -1], 0.0, self.max_B)

    return dict(
      stations=stations,
      views=views,
      magnetic_strength=magnetic_strength
    )

  def decode_design(self, encoded_design):
    decoded = self._decode_design(encoded_design)

    return dict(
      stations=[float(p) for p in decoded['stations']],
      views=[float(a) for a in decoded['views']],
      magnetic_strength=float(decoded['magnetic_strength'])
    )

  def get_geometry(self, design: np.ndarray[tuple[int, int], np.dtype[np.float32]]):
    n, _ = design.shape
    m = self.n_layers

    design_decoded = self._decode_design(design)

    station_positions = design_decoded['stations']
    layer_positions = station_positions[..., None, None] + self.view_offsets[:, None] + self.layer_offsets
    layer_positions = np.reshape(layer_positions, shape=(n, m))
    layer_angles = np.repeat(design_decoded['views'], self.n_layers_per_view, axis=1)

    layer_widths = self.layer_width + np.zeros(shape=(n, m), dtype=np.float32)
    layer_heights = self.layer_height + np.zeros(shape=(n, m), dtype=np.float32)

    B = design_decoded['magnetic_strength']
    assert np.all(np.isfinite(B)), f'NaN decoding B, {B}'
    B_z0 = self.B_z0 + np.zeros(shape=(n,), dtype=np.float32)
    B_sigma = self.B_sigma + np.zeros(shape=(n,), dtype=np.float32)

    return layer_positions, layer_widths, layer_heights, layer_angles, B, B_z0, B_sigma

  def sample(self, seed, design, compute_trajectories=False):
    layer_positions, layer_widths, layer_heights, layer_angles, B, B_z0, B_sigma = self.get_geometry(design)
    rng = np.random.default_rng(seed)
    n = layer_positions.shape[0]

    batch_index = rng.integers(0, self.events['index'].shape[0] - 1, size=(n, ))
    sizes = [self.events['index'][k + 1] - self.events['index'][k] for k in batch_index]
    total_size = sum(sizes)

    ### flattening batch + particles dimensions
    ### as trajectories are for vizualization only they stay as a list
    responses = np.ndarray(shape=(total_size, self.n_layers, self.n_straws), dtype=np.float32)
    edep = np.zeros_like(responses, dtype=np.float32)
    r_mm = np.zeros_like(responses, dtype=np.float32)
    t0_arr = np.zeros_like(responses, dtype=np.float32)
    hit_pos = np.zeros(responses.shape + (3,), dtype=np.float32)

    trajectories = list()
    offset = 0

    for i, size in enumerate(sizes):
      # initial positions in CM
      # Masses, energies, momenta - in MeV
      j = batch_index[i]

      k_start, k_end = self.events['index'][j], self.events['index'][j + 1]
      positions = self.events['particle_positions'][k_start:k_end]
      momenta = self.events['particle_momenta'][k_start:k_end]
      masses = self.events['particle_masses'][k_start:k_end]
      charges = self.events['particle_charges'][k_start:k_end]

      n_particles = k_end - k_start

      # Allocate output arrays
      if compute_trajectories:
        traj = np.zeros((1, n_particles, self.n_t, 3), dtype=np.float32)
      else:
        traj = None

      positions_event = positions[None].astype(np.float32)
      momenta_event = momenta[None].astype(np.float32)
      masses_event = masses[None].astype(np.float32)
      charges_event = charges[None].astype(np.float32)

      straw_detector.solve(
        positions_event, momenta_event, masses_event, charges_event,
        B[i:i+1], B_z0[i:i+1], B_sigma[i:i+1],
        self.n_t, self.dt,
        layer_positions[i:i+1].astype(np.float32), layer_widths[i:i+1].astype(np.float32),
        layer_heights[i:i+1].astype(np.float32), layer_angles[i:i+1].astype(np.float32),
        traj, responses[None, offset:offset + size],
        edep[None, offset:offset + size], r_mm[None, offset:offset + size], t0_arr[None, offset:offset + size],
        hit_pos[None, offset:offset + size]
      )

      if compute_trajectories:
        trajectories.append(traj[0])

      offset += size

    ### no hits -> noisy signal

    activations = np.ndarray(shape=(n, self.max_size, 5))

    offset = 0
    for i, size in enumerate(sizes):
      from .straw_signal import simplified_straw_response, simpified_TDC
      ### very simplified TDC
      ### angles are small, wire time ~= width - hit_x
      tdc = simpified_TDC(rng, t0_arr, r_mm, hit_pos[..., 0], self.layer_width)

      true_hits = np.where(t0_arr[offset:offset + size] > self.dt)

      edep_true_hits = edep[offset:offset + size][true_hits]
      straw_response = simplified_straw_response(rng, edep_true_hits, dark_current=self.straw_noise_rate)

      tdc_true_hits = tdc[offset:offset + size][true_hits]
      noise_ts = rng.uniform(low=tdc_first_hit - self.dt, high=tdc_last_hit + self.dt, size=self.max_size)

      tdc_ = np.where(true_hits, tdc[offset:offset + size] - tdc_first_hit, noise_ts)

      ### taking maximal signal in the tube
      indx = np.argmax(straw_response[offset:offset + size], axis=0, keepdims=True)
      rs[i] = np.take_along_axis(straw_response, offset + indx, axis=0).squeeze(axis=0)
      ts[i] = np.take_along_axis(tdc_, indx, axis=0).squeeze(axis=0)
      offset += size

    hnl_positions = self.events['HNL_positions'][batch_index]
    hnl_momenta = self.events['HNL_momenta'][batch_index]

    return ts, rs, trajectories, hnl_positions, hnl_momenta

  def encode_sample(self, ts, rs, hnl_positions, hnl_momenta):
    n, *_ = ts.shape
    rs = np.reshape(rs, shape=(n, self.n_stations, self.n_views_per_station, self.n_layers_per_view, self.n_straws))
    ts = np.reshape(ts, shape=(n, self.n_stations, self.n_views_per_station, self.n_layers_per_view, self.n_straws))

    ts /= self.straw_time_normalization
    rs /= self.straw_signal_rate
    rs = np.where(rs > 0.1, np.ones_like(rs), np.zeros_like(rs))
    x = np.stack([rs, ts], axis=-1)

    ps = (hnl_positions - self.hnl_position_offset) / self.hnl_position_scale
    ms = (hnl_momenta - self.hnl_momentum_offset) / self.hnl_momentum_scale

    y = np.concatenate([ps, ms], axis=-1)
    return x, y

  def __call__(self, seed: int, configurations: np.ndarray):
    ts, rs, _, hnl_positions, hnl_momenta = self.sample(seed, configurations)
    x, y = self.encode_sample(ts, rs, hnl_positions, hnl_momenta)
    return x, y

  def loss(self, target, predicted):
    import jax.numpy as jnp
    return jnp.mean(jnp.square(target - predicted))

  def metric(self, target, predicted):
    import jax.numpy as jnp
    px, py, pz = predicted[..., 0], predicted[..., 1], predicted[..., 2]
    tx, ty, tz = target[..., 0], target[..., 1], target[..., 2]

    ppx, ppy, ppz = predicted[..., 3], predicted[..., 4], predicted[..., 5]
    tpx, tpy, tpz = target[..., 3], target[..., 4], target[..., 5]

    sx, sy, sz = self.hnl_position_scale
    spx, spy, spz = self.hnl_momentum_scale

    f = lambda delta: jnp.mean(jnp.abs(delta), axis=range(1, tx.ndim - 1))

    return {
      'dx': f((px - tx) * sx),
      'dy': f((py - ty) * sy),
      'dz': f((pz - tz) * sz),

      'dpx': f((ppx - tpx) * spx),
      'dpy': f((ppy - tpy) * spy),
      'dpz': f((ppz - tpz) * spz),
    }

  def metric_names(self):
    return ('dx', 'dy', 'dz', 'dpx', 'dpy', 'dpz')

  def errors(self, target, predicted):
    px, py, pz = predicted[..., 0], predicted[..., 1], predicted[..., 2]
    tx, ty, tz = target[..., 0], target[..., 1], target[..., 2]

    ppx, ppy, ppz = predicted[..., 3], predicted[..., 4], predicted[..., 5]
    tpx, tpy, tpz = target[..., 3], target[..., 4], target[..., 5]

    sx, sy, sz = self.hnl_position_scale
    spx, spy, spz = self.hnl_momentum_scale

    return {
      'dx': (px - tx) * sx,
      'dy': (py - ty) * sy,
      'dz': (pz - tz) * sz,

      'dpx': (ppx - tpx) * spx,
      'dpy': (ppy - tpy) * spy,
      'dpz': (ppz - tpz) * spz,
    }

  def labels(self):
    return (
      'HNL decay x error',
      'HNL decay y error',
      'HNL decay z error',
      'HNL momentum x error',
      'HNL momentum y error',
      'HNL momentum z error',
    )