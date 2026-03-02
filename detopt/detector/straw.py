import math
import os
import subprocess

import numpy as np

# For reading ROOT files
try:
    import uproot
except ImportError:
    uproot = None

from ..utils.encoding import normal_to_uniform, uniform_to_normal
from . import straw_detector
from .common import Detector

__all__ = ["StrawDetector", "SparseHits", "sparse_to_dense"]


class SparseHits:
    """Container for sparse hit representation."""

    def __init__(self, events, particles, layers, straws, values, r_mm, t0, hit_pos):
        self.events = np.asarray(events, dtype=np.int32)
        self.particles = np.asarray(particles, dtype=np.int32)
        self.layers = np.asarray(layers, dtype=np.int32)
        self.straws = np.asarray(straws, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float32)
        self.r_mm = np.asarray(r_mm, dtype=np.float32)
        self.t0 = np.asarray(t0, dtype=np.float32)
        self.hit_pos = np.asarray(hit_pos, dtype=np.float32)
        if self.hit_pos.ndim == 1:
            self.hit_pos = self.hit_pos.reshape(-1, 3)

    def __len__(self):
        return len(self.events)

    def to_dense(self, n_events, n_particles, n_layers, n_straws):
        """Convert sparse hits to dense arrays."""
        return sparse_to_dense(
            self.events,
            self.particles,
            self.layers,
            self.straws,
            self.values,
            self.r_mm,
            self.t0,
            self.hit_pos,
            n_events,
            n_particles,
            n_layers,
            n_straws,
        )


def sparse_to_dense(
    events,
    particles,
    layers,
    straws,
    values,
    r_mm,
    t0,
    hit_pos,
    n_events,
    n_particles,
    n_layers,
    n_straws,
):
    """
    Convert sparse hit representation to dense arrays.

    Returns:
        response: (n_events, n_particles, n_layers, n_straws) array
        edep: (n_events, n_particles, n_layers, n_straws) array
        r_mm: (n_events, n_particles, n_layers, n_straws) array
        t0: (n_events, n_particles, n_layers, n_straws) array
        hit_pos: (n_events, n_particles, n_layers, n_straws, 3) array
    """
    response = np.zeros((n_events, n_particles, n_layers, n_straws), dtype=np.float32)
    edep_dense = np.zeros((n_events, n_particles, n_layers, n_straws), dtype=np.float32)
    r_mm_dense = np.zeros((n_events, n_particles, n_layers, n_straws), dtype=np.float32)
    t0_dense = np.zeros((n_events, n_particles, n_layers, n_straws), dtype=np.float32)
    hit_pos_dense = np.zeros(
        (n_events, n_particles, n_layers, n_straws, 3), dtype=np.float32
    )

    events = np.asarray(events, dtype=np.int32)
    particles = np.asarray(particles, dtype=np.int32)
    layers = np.asarray(layers, dtype=np.int32)
    straws = np.asarray(straws, dtype=np.int32)
    values = np.asarray(values, dtype=np.float32)
    r_mm = np.asarray(r_mm, dtype=np.float32)
    t0 = np.asarray(t0, dtype=np.float32)
    hit_pos = np.asarray(hit_pos, dtype=np.float32)
    if hit_pos.ndim == 1:
        hit_pos = hit_pos.reshape(-1, 3)

    # Filter valid indices
    valid = (
        (events >= 0)
        & (events < n_events)
        & (particles >= 0)
        & (particles < n_particles)
        & (layers >= 0)
        & (layers < n_layers)
        & (straws >= 0)
        & (straws < n_straws)
    )

    if np.any(valid):
        response[events[valid], particles[valid], layers[valid], straws[valid]] = (
            values[valid]
        )
        r_mm_dense[events[valid], particles[valid], layers[valid], straws[valid]] = (
            r_mm[valid]
        )
        t0_dense[events[valid], particles[valid], layers[valid], straws[valid]] = t0[
            valid
        ]
        hit_pos_dense[events[valid], particles[valid], layers[valid], straws[valid]] = (
            hit_pos[valid]
        )

    return response, r_mm_dense, t0_dense, hit_pos_dense


class StrawDetector(Detector):
    def __init__(
        self,
        # Geometry hierarchy
        station_z: list = [8407.0, 8607.0, 9307.0, 9507.0],
        n_views_per_station: int = 4,
        n_layers_per_view: int = 2,
        n_straws_per_layer: int = 200,
        straw_pitch: float = 2.0,
        straw_length: float = 400.0,
        layer_x_offset: float = 1.0,
        view_angles: tuple = (0.0, 0.0798, -0.0798, 0.0),  # X, U, X', V
        layer_z_gap: float = 1.732,
        view_z_gap: float = 5.0,
        # Physics parameters
        max_B: float = 0.0005,
        z0: float = 8957.0,
        B_sigma: float = 300.0,
        layer_bounds: tuple[float | int, float | int] = (8000.0, 10000.0),
        dt: float = 0.1,
        max_particles=50,
        secondary_multiplier=5,
        angles_bounds=None,
        layer_width=None,
        layer_height=None,
        n_layers=None,
        n_straws=None,
        data_dir="combined_tof",
        loss=None,
        p_spawn_single=0.0,
        p_spawn_pair=0.0,
        E_sec_MeV=0.01,
        origin=(-2.4590519e01, 4.7251717e01, 5.5158970e03),
        origin_sigma=(1.2907193e02, 1.3767969e02, 1.6168958e03),
        momentum=(-2.0980914e-01, 2.1826500e-01, 2.6931271e01),
        momentum_sigma=(7.2272283e-01, 4.9576724e-01, 1.2292634e01),
    ):
        """
        :param max_B: maximal strength of the magnetic field;
        :param L: length parameter of the magnetic field;
        :param origin: the mean point of particles' origin;
        :param layer_bounds: restrictions on the layers' positions;
        :param dt: time increment for the ODE solver;
        """

        self.max_B = max_B
        self.z0 = z0
        self.B_sigma = B_sigma

        self.origin = np.array(origin, dtype=np.float32)
        self.origin_sigma = np.array(origin_sigma, np.float32)
        self.momentum = np.array(momentum, dtype=np.float32)
        self.momentum_sigma = np.array(momentum_sigma, dtype=np.float32)

        # Secondary particle parameters
        self.p_spawn_single = p_spawn_single  # Probability of single e- emission
        self.p_spawn_pair = p_spawn_pair  # Probability of e+e- pair production
        self.E_sec_MeV = E_sec_MeV

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

        self.n_layers = (
            self.n_stations * self.n_views_per_station * self.n_layers_per_view
        )

        # Layer height/width for visualization/hit logic
        self.layer_height = (
            self.straw_pitch * self.n_straws / 2.0
        )  # half-length for +/- y
        self.layer_width = self.straw_length / 2.0  # half-length for +/- x

        # Angle bounds
        if angles_bounds is not None:
            self.angle_bounds = angles_bounds
        else:
            self.angle_bounds = (-0.1, 0.1)

        # primary/secondary bookkeeping
        assert max_particles > 1, "signal events produce at least 2 particles"
        self.max_particles = int(max_particles)
        self.secondary_multiplier = int(secondary_multiplier)
        if self.secondary_multiplier < 1:
            self.secondary_multiplier = 1

        flight_distance = layer_bounds[1] - layer_bounds[0]
        self.n_t = int(flight_distance / (dt * 29.9792))  #
        # self.n_t = 500
        # print("para", flight_distance, self.n_t, self.dt)

        # NPZ data loading support
        self._numpyfile_cache = None
        self._numpyfile_path = None
        self.data_dir = data_dir
        self.max_particles_real = 10
        self._data_loader = None

        # Loss function parameters for position and momentum prediction
        if loss is None:
            loss = {}
        self.position_scale = loss.get("position_scale", 100.0)  # cm
        self.momentum_scale = loss.get("momentum_scale", 10.0)  # GeV/c
        self.position_weight = loss.get("position_weight", 1.0)
        self.momentum_weight = loss.get("momentum_weight", 1.0)

        # Target normalization: calculated from actual data (combined_all_100)
        # Units: positions in cm, momenta in GeV/c
        self.target_mean = np.array(
            [
                7.9546314e-01,
                -7.2685266e-01,
                6.3946289e03,
                4.1835890e-03,
                -1.1129675e-02,
                5.0176453e01,
            ],
            dtype=np.float32,
        )
        self.target_std = np.array(
            [
                7.4502007e01,
                7.6979668e01,
                1.3506592e03,
                5.1641846e-01,
                5.5257869e-01,
                2.8627583e01,
            ],
            dtype=np.float32,
        )

    def design_shape(self):
        # positions + angles + magnetic field strength
        return (self.n_layers + self.n_layers + 1,)

    def output_shape(self):
        return (self.n_layers, self.n_straws)

    def target_shape(self):
        # Return shape for decay vertex reconstruction: [x, y, z, px, py, pz]
        return (6,)

    def ground_truth_shape(self):
        # charges + positions + momenta (flattened)
        return (
            2 * self.max_particles + 3 * self.max_particles + 3 * self.max_particles,
        )

    def encode_ground_truth(self, masses, charges, initial_positions, initial_momentum):
        n, *_ = initial_positions.shape
        normalized_positions = (initial_positions - self.origin) / self.origin_sigma
        normalized_positions = np.reshape(normalized_positions, shape=(n, -1))
        normalized_momenta = np.reshape(initial_momentum, shape=(n, -1))
        ground_truth = np.concatenate(
            [masses, charges, normalized_positions, normalized_momenta], axis=-1
        )
        return ground_truth

    def decode_ground_truth(self, ground_truth):
        # print(ground_truth.shape)

        masses = ground_truth[:, : self.max_particles]
        batch = masses.shape[0]
        # print(masses.shape)
        charges = ground_truth[
            :, self.max_particles : self.max_particles + self.max_particles
        ]
        pos_flat = ground_truth[
            :,
            self.max_particles + self.max_particles : self.max_particles
            + self.max_particles
            + self.max_particles * 3,
        ]
        mom_flat = ground_truth[
            :, self.max_particles + self.max_particles + self.max_particles * 3 :
        ]

        normalized_positions = pos_flat.reshape((batch, self.max_particles, 3))
        initial_momentum = mom_flat.reshape((batch, self.max_particles, 3))

        initial_positions = normalized_positions * self.origin_sigma + self.origin

        return masses, charges, initial_positions, initial_momentum

    def get_design(self, design: np.ndarray):
        n = design.shape[0]
        m = self.n_layers

        design_decoded = self._decode_design(design)

        # Use decoded positions and angles from the design
        layers = design_decoded["positions"]
        layers = layers.astype(np.float32)

        angles = design_decoded["angles"]
        angles = angles.astype(np.float32)

        # Ensure layers and angles have correct shape (n, m)
        if layers.ndim == 1:
            layers = np.tile(layers[None, :], (n, 1))
        if angles.ndim == 1:
            angles = np.tile(angles[None, :], (n, 1))

        # If angles has fewer columns than layers, pad with zeros
        if angles.shape[1] < m:
            angles = np.pad(angles, ((0, 0), (0, m - angles.shape[1])), mode="edge")
        elif angles.shape[1] > m:
            angles = angles[:, :m]

        widths = np.full((n, m), self.layer_width, dtype=np.float32)
        heights = np.full((n, m), self.layer_height, dtype=np.float32)

        # Ensure Bs is array of shape (n,) for consistency with C code expectations
        Bs = np.asarray(design_decoded["magnetic_strength"], dtype=np.float32)
        if Bs.ndim == 0:
            Bs = np.full((n,), Bs, dtype=np.float32)

        return layers, angles, widths, heights, Bs

    def simulate(self, seed, configurations, use_sparse=True):
        # print("\n\n\n\nSimulation started\n\n\n\n")

        n_events = configurations.shape[0]
        # print(configurations.shape)
        rng = np.random.default_rng(seed)

        # Load real daughter particle data and HNL targets
        daughter_data, hnl_targets = self._load_real_data(n_events, rng)

        # print("\n\n\n\ndata loaded\n\n\n\n")
        # print(daughter_data)
        # Extract daughter particle arrays
        masses = daughter_data["masses"]  # (batch, max_particles)
        charges = daughter_data["charges"]  # (batch, max_particles)
        initial_positions = daughter_data["positions"]  # (batch, max_particles, 3)
        initial_momentum = daughter_data["momenta"]  # (batch, max_particles, 3)
        initial_times = daughter_data["times"]  # (batch, max_particles)
        n_particles_per_event = daughter_data["n_particles"]

        # Print loaded particle information
        # print(f"\n{'=' * 80}")
        # print(f"Loaded Particle Information")
        # print(f"{'=' * 80}")
        # print(f"Number of events: {masses.shape[0]}")
        # print(f"Max particles per event: {masses.shape[1]}")

        # for event_idx in range(min(3, masses.shape[0])):  # Show first 3 events
        #     print(f"\n--- Event {event_idx} ---")
        #     # Count non-zero mass particles (actual particles)
        #     n_particles = np.sum(masses[event_idx] > 0)
        #     print(f"Number of particles: {n_particles}")

        #     if n_particles > 0:
        #         print(
        #             f"\n{'Idx':<4} {'Mass (MeV)':<12} {'Charge':<8} {'Position (x,y,z) [cm]':<35} {'Momentum (px,py,pz) [GeV/c]'}"
        #         )
        #         print("-" * 110)
        #         for i in range(min(int(n_particles), 10)):  # Show first 10 particles
        #             mass = masses[event_idx, i]
        #             charge = charges[event_idx, i]
        #             pos = initial_positions[event_idx, i]
        #             mom = initial_momentum[event_idx, i]

        #             print(
        #                 f"{i:<4} {mass:<12.2f} {charge:+.1f}     ({pos[0] / 10:8.2f},{pos[1] / 10:8.2f},{pos[2] / 10:8.2f})  ({mom[0]:7.4f},{mom[1]:7.4f},{mom[2]:7.4f})"
        #             )

        #         if n_particles > 10:
        #             print(f"... and {n_particles - 10} more particles")

        # print(f"\n{'=' * 80}\n")

        # Get detector design parameters
        layers, angles, widths, heights, Bs = self.get_design(configurations)

        # Prepare arrays for solve_sparse

        n_events = configurations.shape[0]
        p_slots = initial_momentum.shape[1]
        # print(initial_positions.shape)
        # print(configurations.shape)
        # input("Wait")

        # print("\n" * 5)
        # print("Design obtained")
        # print(f"layers.shape: {layers.shape}, dtype: {layers.dtype}")
        # print(f"angles.shape: {angles.shape}, dtype: {angles.dtype}")
        # print(f"widths.shape: {widths.shape}, dtype: {widths.dtype}")
        # print(f"heights.shape: {heights.shape}, dtype: {heights.dtype}")
        # print(f"Bs type: {type(Bs)}, value: {Bs}")
        # print(f"n_batch: {n_events}, n_particles: {p_slots}")
        # print("\n" * 5)

        # Allocate output arrays with p_slots

        # Magnetic field parameters for the batch
        Bs_arr = Bs.astype(np.float32)
        z0_arr = np.full((n_events,), self.z0, dtype=np.float32)
        B_sigma_arr = np.full((n_events,), self.B_sigma, dtype=np.float32)

        # Estimate max hits for sparse arrays
        max_hits = 200 * n_events * self.max_particles * self.n_layers

        # print("max hits", max_hits, n_events, p_slots, self.n_t, self.n_layers)
        trajectories = np.zeros(
            (n_events, self.max_particles, self.n_t, 3), dtype=np.float32
        )
        mask = np.zeros((n_events, self.max_particles), dtype=np.float32)

        # Pre-allocate sparse output arrays
        sparse_events = np.zeros(max_hits, dtype=np.int32)
        sparse_particles = np.zeros(max_hits, dtype=np.int32)
        sparse_layers = np.zeros(max_hits, dtype=np.int32)
        sparse_straws = np.zeros(max_hits, dtype=np.int32)
        sparse_values = np.zeros(max_hits, dtype=np.float32)
        sparse_r_mm = np.zeros(max_hits, dtype=np.float32)
        sparse_t0 = np.zeros(max_hits, dtype=np.float32)
        sparse_hit_pos = np.zeros((max_hits, 3), dtype=np.float32)
        sparse_count = np.zeros(1, dtype=np.int32)

        # Call solve with sparse arrays - dimensions passed directly
        straw_detector.solve(
            initial_positions,
            initial_momentum,
            masses,
            charges,
            initial_times,
            Bs_arr,
            z0_arr,
            B_sigma_arr,
            self.n_t,
            self.dt,
            n_events,
            p_slots,
            self.n_layers,
            self.n_straws,
            layers,
            widths,
            heights,
            angles,
            trajectories,
            sparse_events,
            sparse_particles,
            sparse_layers,
            sparse_straws,
            sparse_values,
            sparse_r_mm,
            sparse_t0,
            sparse_hit_pos,
            sparse_count,
            self.p_spawn_single,
            self.p_spawn_pair,
            self.E_sec_MeV,
            self.max_particles,
        )
        # print("\n" * 5)
        # print(
        #     f"DEBUG: p_spawn_single = {self.p_spawn_single}, p_spawn_pair = {self.p_spawn_pair}, E_sec_MeV = {self.E_sec_MeV}"
        # )

        # print("Sim done")
        # print(trajectories.shape)
        # print(trajectories)
        # print("\n" * 5)
        # print(sparse_hit_pos.shape)
        # print(sparse_hit_pos)

        # Check for sparse array overflow

        n_hits = sparse_count[0]
        n_hits = int(sparse_count[0])
        # print("n_hits", n_hits, "max_hits", max_hits)
        assert 0 <= n_hits <= max_hits, (n_hits, max_hits)
        # if n_hits >= max_hits:
        #     print("\n" + "!" * 80)
        #     print("WARNING: Sparse array overflow detected!")
        #     print(f"Allocated space: {max_hits} hits")
        #     print(f"Hits recorded: {n_hits} (may be truncated)")
        #     print("Consider increasing max_hits calculation in simulate()")
        #     print("!" * 80 + "\n")

        # Trim sparse arrays to actual hit count
        events = sparse_events[:n_hits]
        particles = sparse_particles[:n_hits]
        layers_arr = sparse_layers[:n_hits]
        straws = sparse_straws[:n_hits]
        values = sparse_values[:n_hits]
        r_mm_sparse = sparse_r_mm[:n_hits]
        t0_sparse = sparse_t0[:n_hits]
        hit_pos_sparse = sparse_hit_pos[:n_hits]

        # Convert to SparseHits object
        sparse_hits = SparseHits(
            events,
            particles,
            layers_arr,
            straws,
            values,
            r_mm_sparse,
            t0_sparse,
            hit_pos_sparse,
        )

        # PRINT
        # print(f"\n{'=' * 80}")
        # print(f"Detector Hits Summary")
        # print(f"{'=' * 80}")
        # print(f"Total hits recorded: {len(sparse_hits)}")
        # print(f"Sparse array capacity: {max_hits} hits")
        # print(f"Utilization: {100.0 * len(sparse_hits) / max_hits:.2f}%")
        # print(
        #     f"Dense array size: {n_events * p_slots * self.n_layers * self.n_straws} elements"
        # )

        # if len(sparse_hits) > 0:  # Calculate spawn statistics
        #     primary_particles = sparse_hits.particles[
        #         sparse_hits.particles < self.max_particles
        #     ]
        #     secondary_particles = sparse_hits.particles[
        #         sparse_hits.particles >= self.max_particles
        #     ]
        #     n_primary_hits = len(primary_particles)
        #     n_secondary_hits = len(secondary_particles)
        #     n_unique_secondaries = (
        #         len(np.unique(secondary_particles))
        #         if len(secondary_particles) > 0
        #         else 0
        #     )

        #     # Show ALL unique particle indices that have hits
        #     all_unique_particles = np.unique(sparse_hits.particles)
        #     print(f"\n--- Particle Index Analysis ---")
        #     print(f"max_particles setting: {self.max_particles}")
        #     print(
        #         f"Total unique particle indices with hits: {len(all_unique_particles)}"
        #     )
        #     print(f"Particle indices: {all_unique_particles}")
        #     print(f"\nExpected particles from data: 2 (indices 0 and 1)")
        #     print(
        #         f"Unexpected particles (indices >= 2): {np.sum(all_unique_particles >= 2)}"
        #     )

        #     print(f"\n--- Hit Distribution ---")
        #     for pidx in all_unique_particles[:20]:  # Show first 20
        #         n_hits_for_particle = np.sum(sparse_hits.particles == pidx)
        #         particle_type = "Expected" if pidx < 2 else "UNEXPECTED"
        #         print(
        #             f"  Particle {pidx:3d} ({particle_type}): {n_hits_for_particle:4d} hits"
        #         )
        #         if len(all_unique_particles) > 20:
        #             print(f"  ... and {len(all_unique_particles) - 20} more particles")

        #         # Analyze multiple hits per straw
        #     print(f"\n--- Multiple Hits Per Straw Statistics ---")
        #     # Create unique keys for each (event, particle, layer, straw) combination
        #     unique_keys = np.column_stack(
        #         [
        #             sparse_hits.events,
        #             sparse_hits.particles,
        #             sparse_hits.layers,
        #             sparse_hits.straws,
        #         ]
        #     )
        #     # Count occurrences of each combination
        #     from collections import Counter

        #     key_tuples = [tuple(key) for key in unique_keys]
        #     hit_counts = Counter(key_tuples)

        #     # Statistics on hit multiplicity
        #     multiplicities = list(hit_counts.values())
        #     max_hits_per_straw = max(multiplicities)
        #     avg_hits_per_straw = np.mean(multiplicities)
        #     straws_with_multiple_hits = sum(1 for count in multiplicities if count > 1)

        #     print(
        #         f"Total unique (event,particle,layer,straw) combinations: {len(hit_counts)}"
        #     )
        #     print(f"Total hits recorded: {len(sparse_hits)}")
        #     print(f"Average hits per unique straw: {avg_hits_per_straw:.2f}")
        #     print(f"Max hits in a single straw: {max_hits_per_straw}")
        #     print(
        #         f"Straws with multiple hits: {straws_with_multiple_hits} ({100.0 * straws_with_multiple_hits / len(hit_counts):.1f}%)"
        #     )

        #     # Show histogram of hit multiplicities
        #     mult_histogram = Counter(multiplicities)
        #     print(f"\nHit multiplicity histogram:")
        #     for mult in sorted(mult_histogram.keys())[
        #         :10
        #     ]:  # Show first 10 multiplicities
        #         count = mult_histogram[mult]
        #         print(
        #             f"  {mult} hits: {count} straws ({100.0 * count / len(hit_counts):.1f}%)"
        #         )
        #     if len(mult_histogram) > 10:
        #         print(f"  ... and {len(mult_histogram) - 10} more multiplicity values")

        #     print(f"\n--- Spawn Statistics ---")
        #     print(f"Particles < max_particles threshold: {n_primary_hits} hits")
        #     print(f"Particles >= max_particles threshold: {n_secondary_hits} hits")
        #     # print(f"Spawn probability (p_spawn): {self.p_spawn}")
        #     # if self.p_spawn > 0 and n_primary_hits > 0:
        #     #     expected_pairs = n_primary_hits * self.p_spawn
        #     #     actual_pairs = n_unique_secondaries / 2
        #     #     print(f"Expected pairs: ~{expected_pairs:.1f}")
        #     #     print(f"Actual pairs: {actual_pairs:.0f}")

        #     # Group hits by event and particle
        #     print(f"\n--- Hits by Event and Particle ---")
        #     for evt_idx in range(min(3, n_events)):  # Show first 3 events
        #         evt_mask = sparse_hits.events == evt_idx
        #         n_hits_evt = np.sum(evt_mask)
        #         if n_hits_evt == 0:
        #             continue

        #         print(f"\nEvent {evt_idx}: {n_hits_evt} hits")

        #         # Get unique particles that hit in this event
        #         particles_in_evt = np.unique(sparse_hits.particles[evt_mask])
        #         for part_idx in particles_in_evt[:10]:  # Show first 10 particles
        #             part_mask = evt_mask & (sparse_hits.particles == part_idx)
        #             n_hits_part = np.sum(part_mask)

        #             # Determine if primary or secondary
        #             particle_type = (
        #                 "Primary" if part_idx < self.max_particles else "Secondary"
        #             )
        #             print(
        #                 f"  Particle {part_idx} ({particle_type}): {n_hits_part} hits"
        #             )

        # END OF PRINT

        # Detailed hit table
        # print(f"\n--- First 20 Hits (Detailed) ---")
        # print(
        #     f"{'Event':<7} {'Particle':<10} {'Type':<10} {'Layer':<7} {'Straw':<7} "
        #     f"{'Value(ns)':<11}  {'r_mm':<9} {'t0(ns)':<11} {'Position (x,y,z) [mm]'}"
        # )
        # # print("-" * 130)
        # for i in range(min(20, len(sparse_hits))):
        #     evt = sparse_hits.events[i]
        #     part = sparse_hits.particles[i]
        #     part_type = "Primary" if part < self.max_particles else "Secondary"
        #     lay = sparse_hits.layers[i]
        #     stw = sparse_hits.straws[i]
        #     val = sparse_hits.values[i]
        #     r = sparse_hits.r_mm[i]
        #     t = sparse_hits.t0[i]
        #     pos = sparse_hits.hit_pos[i]

        #     print(
        #         f"{evt:<7} {part:<10} {part_type:<10} {lay:<7} {stw:<7} "
        #         f"{val:<11.4f} {r:<9.4f} {t:<11.4f} "
        #         f"({pos[0]:7.2f},{pos[1]:7.2f},{pos[2]:7.2f})"
        #     )

        # if len(sparse_hits) > 20:
        #     print(f"... and {len(sparse_hits) - 20} more hits")

        # print(f"{'=' * 80}\n")
        #     print("=" * 95 + "\n")
        # print("mask shape:", mask.shape, "dtype:", mask.dtype)
        # print(mask)
        # print(trajectories)

        # Calculate fdigi from sparse hits (FairShip-style TDC)
        fdigi_times = {}
        times = []
        if sparse_hits is not None and len(sparse_hits) > 0:
            v_drift = 0.0033  # cm/ns (drift velocity)
            sigma_spatial = 0.012  # cm (spatial resolution)
            c = 29.9792  # cm/ns (speed of light)

            def get_straw_endpoints(
                layer, straw, layers, angles, widths, heights, n_straws
            ):
                l_z = layers[layer]
                angle = angles[layer]
                w = widths[layer]
                h = heights[layer]
                r = h / n_straws
                y_local = 2 * r * straw - h + r
                p0_local = np.array([-w, y_local, l_z])
                p1_local = np.array([w, y_local, l_z])
                A = np.array(
                    [
                        [np.cos(angle), np.sin(angle), 0],
                        [-np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1],
                    ]
                )
                p0 = np.dot(p0_local, A)
                p1 = np.dot(p1_local, A)
                return p0, p1

            n_straws = widths.shape[1] if len(widths.shape) > 1 else widths.shape[0]

            for i in range(len(sparse_hits)):
                event = int(sparse_hits.events[i])
                particle = int(sparse_hits.particles[i])
                layer = int(sparse_hits.layers[i])
                straw = int(sparse_hits.straws[i])

                # Get hit information
                r_mm_val = float(sparse_hits.r_mm[i])  # distance to wire in mm
                t_MC_val = float(sparse_hits.t0[i])  # MC time in ns
                hit_xyz = sparse_hits.hit_pos[i]  # hit position (x, y, z)

                # Get straw endpoints
                p0, p1 = get_straw_endpoints(
                    layer, straw, layers[0], angles[0], widths[0], heights[0], n_straws
                )

                # Calculate drift time with Gaussian smearing
                # t_drift = |Gaus(dist2Wire, sigma_spatial)| / v_drift
                dist_cm = r_mm_val / 10.0  # convert mm to cm
                dist_smeared = abs(np.random.normal(dist_cm, sigma_spatial))
                t_drift = dist_smeared / v_drift

                # Calculate signal propagation time along wire
                propagation_time = (p1[0] - hit_xyz[0]) / c

                # FairShip formula: fdigi = t_MC + t_drift + propagation_time
                # t_MC already includes t_initial (from C code)
                # t_drift: drift time from hit point to wire
                # propagation_time: signal propagation time along the wire
                fdigi = t_MC_val + t_drift + propagation_time

                key = (event, particle, layer, straw)
                fdigi_times[key] = fdigi
                times.append(fdigi)
        times = np.array(times)
        # print(times.mean())
        # print(times.std())
        # # input("wait for times")
        # # Print TDC component breakdown for first few hits
        # if len(fdigi_times) > 0:
        #     print("\n" + "=" * 80)
        #     print("TDC (fdigi) Component Breakdown - First 5 Hits")
        #     print("=" * 80)
        #     print(
        #         f"{'Event':<7} {'Particle':<10} {'Layer':<7} {'Straw':<7} {'t_MC(ns)':<12} {'t_drift(ns)':<14} {'r_mm':<14} {'prop_time(ns)':<15} {'fdigi(ns)':<12}"
        #     )
        #     print("-" * 120)

        #     for idx, (key, fdigi_val) in enumerate(list(fdigi_times.items())[:5]):
        #         event, particle, layer, straw = key
        #         # Recalculate components for display
        #         hit_idx = None
        #         for i in range(len(sparse_hits)):
        #             if (
        #                 sparse_hits.events[i] == event
        #                 and sparse_hits.particles[i] == particle
        #                 and sparse_hits.layers[i] == layer
        #                 and sparse_hits.straws[i] == straw
        #             ):
        #                 hit_idx = i
        #                 break

        #         if hit_idx is not None:
        #             t_MC = sparse_hits.t0[hit_idx]
        #             r_mm = sparse_hits.r_mm[hit_idx]
        #             hit_xyz = sparse_hits.hit_pos[hit_idx]

        #             p0, p1 = get_straw_endpoints(
        #                 layer,
        #                 straw,
        #                 layers[0],
        #                 angles[0],
        #                 widths[0],
        #                 heights[0],
        #                 n_straws,
        #             )

        #             dist_cm = r_mm / 10.0
        #             t_drift = abs(np.random.normal(dist_cm, sigma_spatial)) / v_drift
        #             propagation_time = (p1[0] - hit_xyz[0]) / c

        #             print(
        #                 f"{event:<7} {particle:<10} {layer:<7} {straw:<7} {t_MC:<12.4f} {t_drift:<14.6f}  {r_mm:<14.6f} {propagation_time:<15.6f} {fdigi_val:<12.4f}"
        #             )

        #     print("=" * 80)
        #     print(f"\nTotal TDC values stored: {len(fdigi_times)}")
        #     print(f"\nPhysics parameters:")
        #     print(f"  v_drift = {v_drift} cm/ns (drift velocity)")
        #     print(f"  sigma_spatial = {sigma_spatial} cm (spatial resolution)")
        #     print(f"  c = {c} cm/ns (speed of light)")
        #     print(f"\nNote: r_mm is drift distance to wire in mm")
        #     print(f"      Expected t_drift ≈ (r_mm/10) / v_drift")
        #     print()

        # print("\n" * 5)
        # print("Fdigi")
        # print(len(fdigi_times))
        # print(fdigi_times)
        # print(initial_times)
        # print(sparse_hits)

        # Return sparse_hits directly
        return (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            trajectories,
            sparse_hits,
            fdigi_times,
            times,
            mask,
            hnl_targets,  # (batch, 6) = [dx, dy, dz, px, py, pz] in cm and GeV/c
        )

    def _load_real_data(self, n_events, rng):
        """
        Load real daughter particle data from NPZ files.

        Args:
            n_events: Number of events to load
            rng: Random number generator

        Returns:
            daughter_data: Dict with particle info
                - masses: MeV (for C code)
                - charges: e (elementary charge)
                - positions: cm
                - momenta: MeV/c (converted from GeV/c for C code compatibility)
                - times: ns (time of flight to prestraw detector)
            hnl_targets: (n_events, 6) HNL decay vertex targets [dx, dy, dz in cm, px, py, pz in GeV/c]
        """
        # Initialize data loader if not already done
        if self._data_loader is None:
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from load_hnl_data import HNLDataLoader

            # Pass max_particles to loader so it loads data with correct padding
            # max_particles = getattr(self, "max_particles_real", 50)
            self._data_loader = HNLDataLoader(
                self.data_dir, max_particles=self.max_particles
            )
            # print(f"Initialized HNL data loader from {self.data_dir}")
        # Load batch of events (data is already in memory - fast!)
        daughter_data, hnl_targets = self._data_loader.get_batch(
            batch_size=n_events, rng=rng
        )

        return daughter_data, hnl_targets

    def __call__(self, seed: int, configurations: np.ndarray):
        """
        returns ground_truth, measurements, target

        Returns:
            ground_truth: (batch, 14) encoded daughter particle info
            measurements: SparseHits object or dense array
            target: (batch, 6) HNL decay vertex [x, y, z, px, py, pz]
        """
        (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            traj,  # traj
            sparse_hits,  # sparse_hits
            fdigi_times,  # fdigi_times (dict)
            times,
            _,  # mask
            target,
        ) = self.simulate(seed, configurations)
        ground_truth = self.encode_ground_truth(
            masses, charges, initial_positions, initial_momentum
        )
        events = sparse_hits.events
        # offset = events[0]
        # for i in range(len(events)):
        #     events[i] -= offset
        info = (
            events,
            sparse_hits.layers,
            sparse_hits.straws,
            times,
        )
        # print(events)
        # input("wait for events")
        # target is already (batch, 6) from simulate()
        # print(ground_truth)
        return ground_truth, info, target, traj, fdigi_times

    def loss(self, target, predicted):
        # MSE on normalized targets
        # Network outputs normalized values, targets need to be normalized
        # target shape: (batch, 6) where [:3] is position (x,y,z) and [3:] is momentum (px,py,pz)
        import jax
        import jax.numpy as jnp

        # print(target, target.shape)
        # input("wait tar")
        from jax import config

        config.update("jax_disable_jit", True)
        # Normalize targets (predictions are already normalized from network)
        target_norm = (target - self.target_mean) / self.target_std
        diff = target_norm - predicted
        # mse_per_dim = jnp.mean(diff**2, axis=0)
        # print("iii")
        # print(diff**2)

        # # print(target_norm, target_norm.shape)

        # jax.debug.print("mse_per_dim {}", diff)
        # input("wait tar")
        # print(predicted, predicted.shape)
        # input("wait pr")
        # MSE on normalized values (all dimensions have equal weight now)
        mse = jnp.mean(jnp.square(diff), axis=-1)
        # pos_mse = jnp.mean(diff[:, :3] ** 2)
        # mom_mse = jnp.mean(diff[:, 3:] ** 2)
        # jax.debug.print("pos_mse {} mom_mse {}", pos_mse, mom_mse)

        return mse  # Shape: (batch,)

    def metric(self, target, predicted):
        # RMSE on normalized targets (same as loss but with sqrt)
        # Network outputs normalized values, targets need to be normalized
        import jax.numpy as jnp

        # Normalize targets (predictions are already normalized from network)
        target_norm = (target - self.target_mean) / self.target_std

        # RMSE on normalized values
        rmse = jnp.sqrt(jnp.mean(jnp.square(target_norm - predicted), axis=-1))

        return rmse  # Shape: (batch,)

    def encode_design(self, design):
        positions = np.array(design["positions"], dtype=np.float32)
        positions = uniform_to_normal(positions, *self.layer_bounds)
        angles = np.array(design["angles"], dtype=np.float32)
        angles = uniform_to_normal(angles, *self.angle_bounds)
        magnetic_strength = np.float32(design["magnetic_strength"])
        magnetic_strength = uniform_to_normal(magnetic_strength, 0.0, self.max_B)

        return np.concatenate([positions, angles, [magnetic_strength]], axis=0)

    def _decode_design(self, encoded_design):
        n = self.n_layers
        positions = normal_to_uniform(encoded_design[..., :n], *self.layer_bounds)
        angles = normal_to_uniform(encoded_design[..., n:-1], *self.angle_bounds)
        magnetic_strength = normal_to_uniform(encoded_design[..., -1], 0.0, self.max_B)

        # Ensure float32 for consistency
        positions = np.asarray(positions, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)
        # magnetic_strength is already a scalar, just ensure it's float32
        magnetic_strength = np.float32(magnetic_strength)

        return dict(
            positions=positions, angles=angles, magnetic_strength=magnetic_strength
        )

    def decode_design(self, encoded_design):
        decoded = self._decode_design(encoded_design)
        # Handle both scalar and array magnetic_strength
        mag_strength = decoded["magnetic_strength"]
        mag_strength_val = float(mag_strength)

        return dict(
            positions=[float(p) for p in decoded["positions"]],
            angles=[float(a) for a in decoded["angles"]],
            magnetic_strength=mag_strength_val,
        )

    def get_current_design(self):
        positions = []
        angles = []

        # layer ordering: station -> view -> layer-within-view
        for z_station in self.station_z:
            for v in range(self.n_views_per_station):
                view_base_z = z_station + v * self.view_z_gap
                ang = (
                    self.view_angles[v]
                    if v < len(self.view_angles)
                    else self.view_angles[-1]
                )
                for l in range(self.n_layers_per_view):
                    positions.append(view_base_z + l * self.layer_z_gap)
                    angles.append(ang)

        if len(positions) != self.n_layers:
            raise RuntimeError(
                f"current_design_dict produced {len(positions)} layers, expected {self.n_layers}"
            )

        return {
            "positions": positions,
            "angles": angles,
            "magnetic_strength": float(self.max_B),
        }

    def get_encoded_current_design(self):
        d = self.get_current_design()
        B = float(d["magnetic_strength"])
        B = max(0.0, min(B, float(self.max_B)))
        d["magnetic_strength"] = B
        enc = self.encode_design(d)
        enc = np.asarray(enc, dtype=np.float32)
        return enc
