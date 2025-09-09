import math
import numpy as np

__all__ = [
  'load_events'
]

def get_index(sizes):
  indices = np.ndarray(shape=(sizes.shape[0] + 1, ), dtype=sizes.dtype)
  indices[0] = 0
  np.cumsum(sizes, dtype=sizes.dtype, out=indices[1:])

  return indices

def load_events(path: str, min_event_size: int | None=None, max_event_size: int | None=None):
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

      'HNL_positions': HNL_positions,
      'HNL_momenta': HNL_momenta,
      'HNL_masses': HNL_masses,

      'particle_positions': particle_positions,
      'particle_momenta': particle_momenta,
      'particle_pdg': particle_pdg
    }

  min_event_size = 0 if min_event_size is None else min_event_size
  max_event_size = math.inf if max_event_size is None else max_event_size

  n_events = 0
  n_particles = 0

  for i in range(event_sizes.shape[0]):
    size = event_sizes[i]

    if min_event_size <= size <= max_event_size:
      n_events += 1
      n_particles += size

  HNL_positions_ = np.ndarray(shape=(n_events, 3), dtype=HNL_positions.dtype)
  HNL_momenta_ = np.ndarray(shape=(n_events, 3), dtype=HNL_momenta.dtype)
  HNL_masses_ = np.ndarray(shape=(n_events,), dtype=HNL_masses.dtype)

  event_sizes_ = np.ndarray(shape=(n_events,), dtype=event_sizes.dtype)

  particle_positions_ = np.ndarray(shape=(n_particles, 3), dtype=particle_positions.dtype)
  particle_momenta_ = np.ndarray(shape=(n_particles, 3), dtype=particle_momenta.dtype)
  particle_pdg_ = np.ndarray(shape=(n_particles,), dtype=particle_pdg.dtype)


  i_, j_ = 0, 0
  j = 0

  for i in range(event_sizes.shape[0]):
    size = event_sizes[i]

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
    'index': get_index(event_sizes),

    'HNL_positions': HNL_positions_,
    'HNL_momenta': HNL_momenta_,
    'HNL_masses': HNL_masses_,

    'particle_positions': particle_positions_,
    'particle_momenta': particle_momenta_,
    'particle_pdg': particle_pdg_
  }