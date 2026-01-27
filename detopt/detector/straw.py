import math
import os
import subprocess

import jax.numpy as jnp
import numpy as np

# For reading ROOT files
try:
    import uproot
except ImportError:
    uproot = None

from ..utils.encoding import normal_to_uniform, uniform_to_normal
from . import straw_detector
from .common import Detector

__all__ = ["StrawDetector"]


class StrawDetector(Detector):
    def __init__(
        self,
        # Geometry hierarchy
        station_z: list = [2598.0, 2698.0, 3498.0, 3538.0],
        n_views_per_station: int = 4,
        n_layers_per_view: int = 2,
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
        z0: float | None = None,
        B_sigma: float | None = None,
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
        # Loss function parameters
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
        self.n_t = int(flight_distance / (dt * 29.9792))  #
        print(self.n_t)
        self.straw_signal_rate = straw_signal_rate
        self.straw_noise_rate = straw_noise_rate

        # Cache for numpy file data to avoid reloading on each call
        self._numpyfile_cache = None
        self._numpyfile_path = None

        self.data_dir = "combined_all"
        self.max_particles_real = 50
        self._data_loader = None

        # Loss function parameters for position and momentum prediction
        if loss is None:
            loss = {}
        self.position_scale = loss.get("position_scale", 100.0)  # cm
        self.momentum_scale = loss.get("momentum_scale", 10.0)  # GeV/c
        self.position_weight = loss.get("position_weight", 1.0)
        self.momentum_weight = loss.get("momentum_weight", 1.0)

        print("\n\n\nSet up\n\n\n")

    def design_shape(self):
        # positions + angles + magnetic field strength
        return (self.n_layers + self.n_layers + 1,)

    def output_shape(self):
        # Returns shape per sample; actual batch dimension added at runtime
        return (self.secondary_multiplier,)

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

        print(design.shape, n, m, end="\n\n\n")
        print(design)

        design_decoded = self._decode_design(design)
        print(design_decoded.keys())
        print(design_decoded["positions"].shape)
        print(design_decoded["angles"].shape)
        print(design_decoded["magnetic_strength"].shape)

        # Use decoded design positions and angles instead of fixed geometry
        # This allows detector design optimization via subgradient method
        layers = design_decoded["positions"]
        angles = design_decoded["angles"]
        widths = self.layer_width + np.zeros(shape=(n, m), dtype=np.float32)
        heights = self.layer_height + np.zeros(shape=(n, m), dtype=np.float32)

        Bs = design_decoded["magnetic_strength"]
        assert np.all(np.isfinite(Bs)), f"NaN decoding B, {Bs}"
        Ls = self.L + np.zeros(shape=(n,), dtype=np.float32)

        return layers, angles, widths, heights, Bs, Ls

    def simulate(self, seed, configurations, use_sparse=True):
        """
        Load real daughter particles and simulate through detector.

        Args:
            seed: Random seed for event sampling
            configurations: (batch, 65) detector design parameters
            use_sparse: If True, return sparse hit format

        Returns:
            Tuple of (masses, charges, initial_positions, initial_momentum,
                     trajectories, sparse_response, signal, fdigi_times, mask, target)
            where target is (batch, 6) = [dx, dy, dz, px, py, pz] HNL decay vertex
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

        print("/n/n/n/nMMMMMM/n/n/n/n/n/n/n", initial_positions)

        # Get detector design parameters
        layers, angles, widths, heights, Bs, Ls = self.get_design(configurations)

        # Prepare arrays for solve_sparse
        max_particles = initial_positions.shape[1]
        trajectories = np.zeros(
            (n_events, max_particles, self.n_t, 3), dtype=np.float32
        )

        # Create mask for valid particles
        mask = np.zeros((n_events, max_particles), dtype=np.float32)
        for i in range(n_events):
            n_parts = min(n_particles_per_event[i], max_particles)
            mask[i, :n_parts] = 1.0

        # Magnetic field parameters
        z0_arr = (
            np.mean(layers, axis=1).astype(np.float32)
            if self.z0 is None
            else np.full((n_events,), self.z0, dtype=np.float32)
        )
        B_sigma_arr = (
            Ls.astype(np.float32)
            if self.B_sigma is None
            else np.full((n_events,), self.B_sigma, dtype=np.float32)
        )

        # Run detector simulation using solve_sparse
        print(f"Running solve_sparse for {n_events} events...")

        sparse_response = None
        fdigi_times = {}

        from . import straw_detector

        # Run detector simulation
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

        print(f"Generated {len(events)} hits from solve_sparse")

        # Helper function to get straw endpoints
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

        # Compute fdigi values for all hits
        fdigi_values = np.zeros(len(events), dtype=np.float32)

        from .straw_signal import fairship_fdigi

        n_straws = widths.shape[1] if len(widths.shape) > 1 else widths.shape[0]

        for i in range(len(events)):
            event_idx = events[i]
            layer_idx = layers_arr[i]
            straw_idx = straws[i]
            r_mm = r_mm_sparse[i]
            t0 = t0_sparse[i]
            hit_xyz = hit_pos_sparse[i]
            print(event_idx, layer_idx, straw_idx, r_mm, t0, hit_xyz)

            # Get straw endpoints to calculate wire position
            p0, p1 = get_straw_endpoints(
                layer_idx,
                straw_idx,
                layers[event_idx],
                angles[event_idx],
                widths[event_idx],
                heights[event_idx],
                n_straws,
            )

            # Calculate wire direction and hit position along wire
            wire_vec = p1 - p0
            wire_len = np.linalg.norm(wire_vec)
            wire_dir = wire_vec / wire_len if wire_len > 0 else np.zeros(3)
            proj = np.dot(hit_xyz - p0, wire_dir)
            x_hit = proj / 10.0  # Convert to cm
            x_readout = wire_len / 10.0  # Convert to cm

            # Calculate fdigi using fairship_fdigi
            print(layer_idx, 0, t0, r_mm, x_hit, x_readout, 0.12, 0.033, 29.9792)
            fdigi = fairship_fdigi(
                layer=layer_idx,
                t0_event=0.0,
                t_MC=t0,
                r_mm=r_mm,
                x_hit=x_hit,
                x_readout=x_readout,
                sigma_spatial=0.12,
                v_drift=0.033,
                c=29.9792,
            )
            print(t0, fdigi)
            fdigi_values[i] = fdigi

            # Also store in fdigi_times dict for diagnostics
            station = layer_idx // (self.n_layers_per_view * self.n_views_per_station)
            view = (
                layer_idx % (self.n_layers_per_view * self.n_views_per_station)
            ) // self.n_layers_per_view
            layer_local = layer_idx % self.n_layers_per_view + 1
            key = (station, view, layer_local, straw_idx)
            fdigi_times[key] = fdigi

        # Package sparse response with fdigi VALUES (not raw values)
        sparse_response = {
            "events": events,
            "particles": particles,
            "layers": layers_arr,
            "straws": straws,
            "values": fdigi_values,  # USE FDIGI, not raw values!
        }

        # Signal array (all real data is signal)
        signal = np.ones((n_events,), dtype=np.float32)

        # Return with HNL targets
        return (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            trajectories,
            sparse_response,
            signal,
            fdigi_times,
            mask,
            hnl_targets,  # (batch, 6) = [dx, dy, dz, px, py, pz]
        )

    def _load_real_data(self, n_events, rng):
        """
        Load real daughter particle data from NPZ files.

        This should be added as a method to StrawDetector class.

        Args:
            n_events: Number of events to load
            rng: Random number generator

        Returns:
            daughter_data: Dict with particle info
            hnl_targets: (n_events, 6) HNL decay vertex targets
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

    def __call__(self, seed: int | tuple, configurations: np.ndarray):
        """
        returns ground_truth, measurements, target

        Returns:
            ground_truth: (batch, 14) encoded daughter particle info
            measurements: dict with sparse hits or None
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
            fdigi_times,
            mask,
            target,
        ) = self.simulate(seed, configurations)
        ground_truth = self.encode_ground_truth(
            masses, charges, initial_positions, initial_momentum
        )

        # target is already (batch, 6) from simulate() - it's the real HNL decay vertex
        return ground_truth, measurements, target

    def loss(self, target, predicted):
        # Weighted and normalized MSE for position (cm) and momentum (GeV/c)
        # target shape: (batch, 6) where [:3] is position (x,y,z) and [3:] is momentum (px,py,pz)

        # Separate position and momentum
        pos_target = target[..., :3]
        pos_pred = predicted[..., :3]
        mom_target = target[..., 3:]
        mom_pred = predicted[..., 3:]

        # Normalized MSE for each component using configured scales
        pos_loss = jnp.mean(
            jnp.square((pos_target - pos_pred) / self.position_scale), axis=-1
        )
        mom_loss = jnp.mean(
            jnp.square((mom_target - mom_pred) / self.momentum_scale), axis=-1
        )

        # Weighted combination using configured weights
        return (
            self.position_weight * pos_loss + self.momentum_weight * mom_loss
        )  # Shape: (batch,)

    def metric(self, target, predicted):
        # Separate RMSE for position and momentum, then average
        # This gives interpretable error metrics in original units

        pos_target = target[..., :3]
        pos_pred = predicted[..., :3]
        mom_target = target[..., 3:]
        mom_pred = predicted[..., 3:]

        # RMSE in original units
        pos_rmse = jnp.sqrt(jnp.mean(jnp.square(pos_target - pos_pred), axis=-1))  # cm
        mom_rmse = jnp.sqrt(
            jnp.mean(jnp.square(mom_target - mom_pred), axis=-1)
        )  # GeV/c

        # Return average of normalized errors for fair comparison using configured scales
        return 0.5 * (
            pos_rmse / self.position_scale + mom_rmse / self.momentum_scale
        )  # Shape: (batch,)

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
