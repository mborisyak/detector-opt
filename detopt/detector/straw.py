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

    def __init__(
        self, events, particles, layers, straws, values, edep, r_mm, t0, hit_pos
    ):
        self.events = np.asarray(events, dtype=np.int32)
        self.particles = np.asarray(particles, dtype=np.int32)
        self.layers = np.asarray(layers, dtype=np.int32)
        self.straws = np.asarray(straws, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float32)
        self.edep = np.asarray(edep, dtype=np.float32)
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
            self.edep,
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
    edep,
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
    edep = np.asarray(edep, dtype=np.float32)
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
        edep_dense[events[valid], particles[valid], layers[valid], straws[valid]] = (
            edep[valid]
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

    return response, edep_dense, r_mm_dense, t0_dense, hit_pos_dense


INV_SQRT_2 = math.sqrt(0.5)

NAME2PID = {
    "mu-": 13,
    "mu+": -13,
    "e-": 11,
    "e+": -11,
    "pi-": -211,
    "pi+": 211,
    "gamma": 22,
    "proton": 2212,
    "antiproton": -2212,
    "neutron": 2112,
    "antineutron": -2112,
    "nu_e": 12,
    "nu_e_bar": -12,
    "nu_mu": 14,
    "nu_mu_bar": -14,
}


class StrawDetector(Detector):
    def __init__(
        self,
        # Geometry hierarchy
        station_z: list = [2598.0, 2698.0, 3498.0, 3538.0],
        n_views_per_station: int = 4,
        n_layers_per_view: int = 4,
        n_straws_per_layer: int = 200,
        straw_pitch: float = 2.0,
        straw_length: float = 200.0,
        layer_x_offset: float = 1.0,
        view_angles: tuple = (0.0, 0.0798, -0.0798, 0.0),  # X, U, X', V
        layer_z_gap: float = 1.732,
        view_z_gap: float = 5.0,
        # Physics parameters
        max_B: float = 0.5,
        L=1.0,
        z0: float = None,
        B_sigma: float = None,
        layer_bounds: tuple[float | int, float | int] = (-5.0, 5.0),
        dt: float = 1.0,
        max_particles=2,
        secondary_multiplier=5,
        origin=(-100.0, -100.0, 1700.0),
        origin_sigma=(1.0, 1.0, 1.0),
        momentum=(0.0, 0.0, 5.0),
        momentum_sigma=(0.25, 0.25, 0.5),
        noise_origin=(0.0, 0.0, -10.0),
        noise_origin_sigma=(1.0, 1.0, 1.0),
        noise_momentum=(0.0, 0.0, 5.0),
        noise_momentum_sigma=(0.25, 0.25, 0.5),
        straw_signal_rate=200.0,
        straw_noise_rate=10.0,
        angles_bounds=None,
        layer_width=None,
        layer_height=None,
        n_layers=None,
        n_straws=None,
        data_dir="combined_all_100",
        loss=None,
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
        self.noise_origin_sigma = np.array(noise_origin_sigma, dtype=np.float32)
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
        self.n_t = int(flight_distance / (dt * 29.9792)) * 10  #

        self.straw_signal_rate = straw_signal_rate
        self.straw_noise_rate = straw_noise_rate

        # NPZ data loading support
        self._numpyfile_cache = None
        self._numpyfile_path = None
        self.data_dir = data_dir
        self.max_particles_real = 50
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
            [-26.74728, 6.876283, 6029.46, -0.11996946, 0.050958868, 35.228668],
            dtype=np.float32,
        )
        self.target_std = np.array(
            [149.0265, 142.9359, 1384.0433, 0.8211833, 0.7007604, 21.184158],
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
        return (self.max_particles + 3 * self.max_particles + 3 * self.max_particles,)

    def encode_ground_truth(self, masses, charges, initial_positions, initial_momentum):
        n, *_ = initial_positions.shape
        normalized_positions = (initial_positions - self.origin) / self.origin_sigma
        normalized_positions = np.reshape(normalized_positions, shape=(n, -1))
        normalized_momenta = np.reshape(initial_momentum, shape=(n, -1))
        ground_truth = np.concatenate(
            [charges, normalized_positions, normalized_momenta], axis=-1
        )
        return ground_truth

    def get_design(self, design: np.ndarray):
        n, _ = design.shape
        m = self.n_layers

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

        layers = np.array(layer_positions, dtype=np.float32)[None, :].repeat(n, axis=0)
        angles = np.array(layer_angles, dtype=np.float32)[None, :].repeat(n, axis=0)
        widths = self.layer_width + np.zeros(shape=(n, m), dtype=np.float32)
        heights = self.layer_height + np.zeros(shape=(n, m), dtype=np.float32)

        Bs = design_decoded["magnetic_strength"]
        assert np.all(np.isfinite(Bs)), f"NaN decoding B, {Bs}"
        Ls = self.L + np.zeros(shape=(n,), dtype=np.float32)

        return layers, angles, widths, heights, Bs, Ls

    def simulate(self, seed, configurations, use_sparse=True):
        """
        Load real daughter particles from NPZ files and run detector simulation.

        Args:
            seed: Random seed for event sampling
            configurations: (batch, 65) detector design parameters
            use_sparse: If True, return sparse hit format (only option now)

        Returns:
            Tuple of (masses, charges, initial_positions, initial_momentum,
                     trajectories, sparse_hits, signal, waveforms, fdigi_times, mask, target)
        """
        n_events = configurations.shape[0]
        rng = np.random.default_rng(seed)

        # Load real daughter particle data and HNL targets
        daughter_data, hnl_targets = self._load_real_data(n_events, rng)

        # Extract daughter particle arrays
        masses = daughter_data["masses"]  # (batch, max_particles)
        charges = daughter_data["charges"]  # (batch, max_particles)
        initial_positions = daughter_data["positions"]  # (batch, max_particles, 3)
        initial_momentum = daughter_data["momenta"]  # (batch, max_particles, 3)
        n_particles_per_event = daughter_data["n_particles"]

        # Get detector design parameters
        layers, angles, widths, heights, Bs, Ls = self.get_design(configurations)

        # Prepare arrays for solve_sparse
        p_slots = initial_positions.shape[1]
        n_events = configurations.shape[0]

        # Allocate output arrays with p_slots
        trajectories = np.zeros((n_events, p_slots, self.n_t, 3), dtype=np.float32)
        mask = np.zeros((n_events, p_slots), dtype=np.float32)

        # Magnetic field parameters for the batch
        if self.z0 is not None:
            z0_arr = np.full((n_events,), self.z0, dtype=np.float32)
        else:
            z0_arr = np.mean(layers, axis=1).astype(np.float32)
        if self.B_sigma is not None:
            B_sigma_arr = np.full((n_events,), self.B_sigma, dtype=np.float32)
        else:
            B_sigma_arr = Ls.astype(np.float32)

        sparse_hits = None
        if use_sparse:
            # Sparse mode: call solve_sparse
            sparse_result = straw_detector.solve_sparse(
                initial_positions,
                initial_momentum,
                masses,
                charges,
                Bs,
                Ls,
                z0_arr,
                B_sigma_arr,
                self.n_t,
                self.dt,
                layers,
                widths,
                heights,
                angles,
                trajectories,
                mask,
            )
            # Unpack sparse results
            (
                events,
                particles,
                layers_arr,
                straws,
                values,
                edep_sparse,
                r_mm_sparse,
                t0_sparse,
                hit_pos_sparse,
            ) = sparse_result

            # Convert to SparseHits object
            sparse_hits = SparseHits(
                events,
                particles,
                layers_arr,
                straws,
                values,
                edep_sparse,
                r_mm_sparse,
                t0_sparse,
                hit_pos_sparse,
            )

            # Print sparse output information
        #     print(f"\n=== Sparse Hits Summary ===")
        #     print(f"Total hits: {len(sparse_hits)}")
        #     print(
        #         f"Events: {n_events}, Particles: {p_slots}, Layers: {self.n_layers}, Straws: {self.n_straws}"
        #     )
        #     print(
        #         f"Sparse representation: {len(sparse_hits)} hits vs {n_events * p_slots * self.n_layers * self.n_straws} dense array elements"
        #     )
        #     print(
        #         f"Memory savings: {100.0 * (1.0 - len(sparse_hits) / (n_events * p_slots * self.n_layers * self.n_straws)):.2f}%"
        #     )

        #     if len(sparse_hits) > 0:
        #         print(f"\nFirst 10 hits:")
        #         print(
        #             f"{'Event':<8} {'Particle':<10} {'Layer':<8} {'Straw':<8} {'Value':<12} {'Edep (MeV)':<15} {'r_mm':<10} {'t0 (ns)':<12}"
        #         )
        #         print("-" * 95)
        #         for i in range(min(10, len(sparse_hits))):
        #             print(
        #                 f"{sparse_hits.events[i]:<8} {sparse_hits.particles[i]:<10} "
        #                 f"{sparse_hits.layers[i]:<8} {sparse_hits.straws[i]:<8} "
        #                 f"{sparse_hits.values[i]:<12.6f} {sparse_hits.edep[i]:<15.6e} "
        #                 f"{sparse_hits.r_mm[i]:<10.4f} {sparse_hits.t0[i]:<12.6f}"
        #             )
        #         if len(sparse_hits) > 10:
        #             print(f"... and {len(sparse_hits) - 10} more hits")
        #     print("=" * 95 + "\n")
        # print("mask shape:", mask.shape, "dtype:", mask.dtype)
        # print(mask)
        # print(trajectories)

        # waveform modeling
        from .straw_signal import straw_response

        waveforms = {}
        if sparse_hits is not None:
            # Use sparse data directly for waveform generation
            for i in range(len(sparse_hits)):
                event = int(sparse_hits.events[i])
                particle = int(sparse_hits.particles[i])
                layer = int(sparse_hits.layers[i])
                straw = int(sparse_hits.straws[i])
                Edep_mev = float(sparse_hits.edep[i])
                r_mm_val = float(sparse_hits.r_mm[i])
                t0 = float(sparse_hits.t0[i])
                if Edep_mev > 0:
                    t, s = straw_response(Edep_mev, r_mm_val, t0)
                    waveforms[(event, particle, layer, straw)] = (t, s)

        signal = np.ones((n_events,), dtype=np.float32)

        # --- FairShip-style TDC calculation using real geometry ---
        from detopt.detector.straw_signal import fairship_fdigi

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
        fdigi_times = {}
        st = [0, 0, 0, 0]
        ns = [0, 0, 0, 0]

        # Create lookup dict for sparse hits if available
        sparse_lookup = None
        if use_sparse and sparse_hits is not None:
            sparse_lookup = {}
            for i in range(len(sparse_hits)):
                key = (
                    int(sparse_hits.events[i]),
                    int(sparse_hits.particles[i]),
                    int(sparse_hits.layers[i]),
                    int(sparse_hits.straws[i]),
                )
                sparse_lookup[key] = i

        for key in waveforms:
            event, particle, layer, straw = key
            if sparse_lookup is not None and key in sparse_lookup:
                # Use sparse data
                hit_idx = sparse_lookup[key]
                hit_xyz = sparse_hits.hit_pos[hit_idx]
                t_MC_val = sparse_hits.t0[hit_idx]
                r_mm_val = sparse_hits.r_mm[hit_idx]

            p0, p1 = get_straw_endpoints(
                layer, straw, layers[0], angles[0], widths[0], heights[0], n_straws
            )
            wire_vec = p1 - p0
            wire_len = np.linalg.norm(wire_vec)
            wire_dir = wire_vec / wire_len if wire_len > 0 else np.zeros(3)
            proj = np.dot(hit_xyz - p0, wire_dir)
            x_hit = proj
            x_readout = wire_len
            fdigi = fairship_fdigi(
                layer,
                t0_event=0.0,
                t_MC=t_MC_val,
                r_mm=r_mm_val,
                x_hit=x_hit,
                x_readout=x_readout,
                sigma_spatial=0.12,
                v_drift=0.033,
                c=29.9792,
            )
            st[layer // 8] += fdigi
            ns[layer // 8] += 1
            fdigi_times[key] = fdigi

        signal = np.ones((n_events,), dtype=np.float32)

        # Return sparse_hits directly
        return (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            trajectories,
            sparse_hits,
            signal,
            waveforms,
            fdigi_times,
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
            hnl_targets: (n_events, 6) HNL decay vertex targets [dx, dy, dz in cm, px, py, pz in GeV/c]
        """
        # Initialize data loader if not already done
        if self._data_loader is None:
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from load_hnl_data import HNLDataLoader

            self._data_loader = HNLDataLoader(self.data_dir)
            print(f"Initialized HNL data loader from {self.data_dir}")
        # Set max_particles based on detector config
        max_particles = getattr(self, "max_particles_real", 50)

        # Load batch of events
        daughter_data, hnl_targets = self._data_loader.get_batch(
            batch_size=n_events, max_particles=max_particles, rng=rng
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
            _,
            measurements,
            signal,
            _,
            _,
            _,
            target,
        ) = self.simulate(seed, configurations)
        ground_truth = self.encode_ground_truth(
            masses, charges, initial_positions, initial_momentum
        )

        # target is already (batch, 6) from simulate() - it's the real HNL decay vertex
        return ground_truth, measurements, target

    def loss(self, target, predicted):
        # MSE on normalized targets
        # Network outputs normalized values, targets need to be normalized
        # target shape: (batch, 6) where [:3] is position (x,y,z) and [3:] is momentum (px,py,pz)
        import jax.numpy as jnp

        # Normalize targets (predictions are already normalized from network)
        target_norm = (target - self.target_mean) / self.target_std

        # MSE on normalized values (all dimensions have equal weight now)
        mse = jnp.mean(jnp.square(target_norm - predicted), axis=-1)

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
        magnetic_strength = np.array(design["magnetic_strength"], dtype=np.float32)
        magnetic_strength = uniform_to_normal(magnetic_strength, 0.0, self.max_B)

        return np.concatenate([positions, angles, magnetic_strength[None]], axis=0)

    def _decode_design(self, encoded_design):
        n = self.n_layers
        positions = normal_to_uniform(encoded_design[..., :n], *self.layer_bounds)
        angles = normal_to_uniform(encoded_design[..., n:-1], *self.angle_bounds)
        magnetic_strength = normal_to_uniform(encoded_design[..., -1], 0.0, self.max_B)

        return dict(
            positions=positions, angles=angles, magnetic_strength=magnetic_strength
        )

    def decode_design(self, encoded_design):
        decoded = self._decode_design(encoded_design)
        return dict(
            positions=[float(p) for p in decoded["positions"]],
            angles=[float(a) for a in decoded["angles"]],
            magnetic_strength=float(decoded["magnetic_strength"]),
        )
