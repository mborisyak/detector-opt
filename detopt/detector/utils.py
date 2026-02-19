import math
import numpy as np

__all__ = [
  'load_events',
  'pid_to_charge',
  'pid_to_mass_MeV'
]

MeV = 1000.0

PID_TO_CHARGE = {
  11: -1.0, -11: +1.0,  # e-, e+
  13: -1.0, -13: +1.0,  # mu-, mu+
  211: +1.0, -211: -1.0,  # pi+, pi-
  2212: +1.0, -2212: -1.0,  # p, pbar
  2112: 0.0, -2112: 0.0,  # n, nbar
  22: 0.0,  # gamma
  12: 0.0, -12: 0.0,  # nu_e, anti
  14: 0.0, -14: 0.0,  # nu_mu, anti
}

PID_TO_MASS = {
  11: 0.510999,  # e±
  13: 105.6583755,  # mu±
  211: 139.57039,  # pi±
  2212: 938.2720813,  # proton/antiproton
  2112: 939.5654133,  # neutron/antineutron
  22: 0.0,  # gamma
  12: 1.0e-6,  # ν_e (set ~0 for toy; tiny real mass irrelevant here)
  14: 1.0e-6,  # ν_μ
}

def get_index(sizes):
  indices = np.ndarray(shape=(sizes.shape[0] + 1, ), dtype=sizes.dtype)
  indices[0] = 0
  np.cumsum(sizes, dtype=sizes.dtype, out=indices[1:])

  return indices

def load_events(path: str, min_event_size: int | None=None, max_event_size: int | None=None, dtype=np.float32):
  f = np.load(path)

  event_sizes = f['event_sizes']

  HNL_positions = f['HNL_positions']
  HNL_momenta = f['HNL_momenta']
  HNL_masses = f['HNL_masses']

  particle_positions = f['particle_positions']
  particle_momenta = f['particle_momenta']
  particle_pdg = f['particle_pdg']

  if min_event_size is None and max_event_size is None:
    return {
      'index': get_index(event_sizes),

      'HNL_positions': np.astype(HNL_positions, dtype=dtype),
      'HNL_momenta': np.astype(HNL_momenta, dtype=dtype) * MeV,
      'HNL_masses': np.astype(HNL_masses, dtype=dtype),

      'particle_positions': np.astype(particle_positions, dtype=dtype),
      'particle_momenta': np.astype(particle_momenta, dtype=dtype),
      'particle_masses': pids_to_mass_MeV(particle_pdg, dtype=dtype),
      'particle_charges': pids_to_charge(particle_pdg, dtype=dtype),
    }

  min_event_size = 0 if min_event_size is None else min_event_size
  max_event_size = math.inf if max_event_size is None else max_event_size

  n_events = 0
  n_particles = 0

  for i in range(event_sizes.shape[0]):
    size = event_sizes[i]
    decay_z = HNL_positions[i, 2]

    if min_event_size <= size <= max_event_size:
      n_events += 1
      n_particles += size

  HNL_positions_ = np.ndarray(shape=(n_events, 3), dtype=dtype)
  HNL_momenta_ = np.ndarray(shape=(n_events, 3), dtype=dtype)
  HNL_masses_ = np.ndarray(shape=(n_events,), dtype=dtype)

  event_sizes_ = np.ndarray(shape=(n_events,), dtype=event_sizes.dtype)

  particle_positions_ = np.ndarray(shape=(n_particles, 3), dtype=dtype)
  particle_momenta_ = np.ndarray(shape=(n_particles, 3), dtype=dtype)
  particle_pdg_ = np.ndarray(shape=(n_particles,), dtype=particle_pdg.dtype)


  i_, j_ = 0, 0
  j = 0

  for i in range(event_sizes.shape[0]):
    size = event_sizes[i]
    decay_z = HNL_positions[i, 2]

    if min_event_size <= size <= max_event_size:
      event_sizes_[i_] = size

      HNL_positions_[i_] = HNL_positions[i]
      HNL_momenta_[i_] = HNL_momenta[i]
      HNL_masses_[i_] = HNL_masses[i]

      particle_positions_[j_:j_ + size] = particle_positions[j:j + size]
      particle_momenta_[j_:j_ + size] = particle_momenta[j:j + size]
      particle_pdg_[j_:j_ + size] = particle_pdg[j:j + size]

      i_ += 1
      j_ += size

    j += size

  return {
    'index': get_index(event_sizes_),

    'HNL_positions': HNL_positions_,
    'HNL_momenta': HNL_momenta_,
    'HNL_masses': HNL_masses_,

    'particle_positions': particle_positions_,
    'particle_momenta': particle_momenta_ * MeV,
    'particle_masses': pids_to_mass_MeV(particle_pdg_, dtype=dtype),
    'particle_charges': pids_to_charge(particle_pdg_, dtype=dtype),
  }

def pid_to_charge(pid: int) -> float:
  return PID_TO_CHARGE.get(pid, 0.0)

def pids_to_charge(pids: np.ndarray, dtype=np.float32):
  charges = np.ndarray(shape=pids.size, dtype=dtype)
  pids_ = pids.ravel()

  for i in range(pids_.size):
    pid = pids_[i]
    if pid in PID_TO_CHARGE:
      charges[i] = PID_TO_CHARGE[pid]
    else:
      raise ValueError(f'unknown pdg {pid}')

  return np.reshape(charges, shape=pids.shape)

# --- Rest masses in MeV (use abs(pid) for particle/antiparticle) ---
def pid_to_mass_MeV(pid: int) -> float:
  return PID_TO_MASS.get(abs(pid), 0.0)

def pids_to_mass_MeV(pids: np.ndarray, dtype=np.float32):
  masses = np.ndarray(shape=pids.size, dtype=dtype)
  pids_ = pids.ravel()

  for i in range(pids_.size):
    pid = abs(pids_[i])

    if pid in PID_TO_MASS:
      masses[i] = PID_TO_MASS[pid]
    else:
      raise ValueError(f'unknown pdg {pid}')

  return np.reshape(masses, shape=pids.shape)