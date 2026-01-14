import math
import numpy as np
import subprocess
import os

# For reading ROOT files
try:
    import uproot
except ImportError:
    uproot = None

from ..utils.encoding import uniform_to_normal, normal_to_uniform
from .common import Detector
from . import straw_detector

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
        
        print("\n\n\nSet up\n\n\n")

    def design_shape(self):
        # positions + angles + magnetic field strength
        return (self.n_layers + self.n_layers + 1,)

    def output_shape(self, batch_size):
        return (batch_size * self.secondary_multiplier, )

    def target_shape(self):
        return ()

    def ground_truth_shape(self):
        # charges + positions + momenta (flattened)
        return (self.max_particles + 3 * self.max_particles + 3 * self.max_particles,)
    
    def encode_ground_truth(self, masses, charges, initial_positions, initial_momentum):
        n, *_ = initial_positions.shape
        normalized_positions = (initial_positions - self.origin) / self.origin_sigma
        normalized_positions = np.reshape(normalized_positions, shape=(n, -1))
        normalized_momenta = np.reshape(initial_momentum, shape=(n, -1))
        ground_truth = np.concatenate([charges, normalized_positions, normalized_momenta], axis=-1)
        return ground_truth

    def get_design(self, design: np.ndarray):
        n, _ = design.shape
        m = self.n_layers

        print(design.shape,n, m, end="\n\n\n")
        print(design)

        design_decoded = self._decode_design(design)
        print(design_decoded.keys())
        print(design_decoded["positions"].shape)
        print(design_decoded["angles"].shape)
        print(design_decoded["magnetic_strength"].shape)
      

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

    def simulate(
        self,
        numpyfile,
        batch_size=None,
        design=None,
        use_sparse=True,
    ):
        """
        Load particle properties from a numpy file and run the detector simulation.
        """
        n_viz = 10
        
        # Cache the loaded data to avoid reloading on each call
        if self._numpyfile_path != numpyfile or self._numpyfile_cache is None:
            with np.load(numpyfile) as data:
                # Store a copy of the data arrays (npz files are lazy-loaded)
                self._numpyfile_cache = {
                    "px": data["px"][:n_viz] * 1e3,  # GeV -> MeV
                    "py": data["py"][:n_viz] * 1e3,
                    "pz": data["pz"][:n_viz] * 1e3,
                    "x": data["x"][:n_viz],
                    "y": data["y"][:n_viz],
                    "z": data["z"][:n_viz],
                    "pid": data["pid"][:n_viz],
                }
                self._numpyfile_path = numpyfile
        
        # Use cached data
        px = self._numpyfile_cache["px"]
        py = self._numpyfile_cache["py"]
        pz = self._numpyfile_cache["pz"]
        x = self._numpyfile_cache["x"]
        y = self._numpyfile_cache["y"]
        z = self._numpyfile_cache["z"]
        pid = self._numpyfile_cache["pid"]

        # Treat all loaded particles as a single batch
        n_particles = px.shape[0]
        n_events = 1

        # reserve extra particle slots for spawned secondaries
        p_slots = max(n_particles, self.max_particles) * self.secondary_multiplier

        # positions (cm)
        initial_positions = np.zeros((n_events, p_slots, 3), dtype=np.float32)
        initial_positions[:, :n_particles, :] = (
            np.stack([x, y, z], axis=-1).reshape((1, n_particles, 3)).astype(np.float32)
        )

        # momenta (MeV)
        initial_momentum = np.zeros((n_events, p_slots, 3), dtype=np.float32)
        initial_momentum[:, :n_particles, :] = (
            np.stack([px, py, pz], axis=-1)
            .reshape((1, n_particles, 3))
            .astype(np.float32)
        )

        def pid_to_charge(pid: int) -> float:
            CHARGE = {
                11: -1.0,
                -11: +1.0,
                13: -1.0,
                -13: +1.0,
                211: +1.0,
                -211: -1.0,
                2212: +1.0,
                -2212: -1.0,
                2112: 0.0,
                -2112: 0.0,
                22: 0.0,
                12: 0.0,
                -12: 0.0,
                14: 0.0,
                -14: 0.0,
            }
            return CHARGE.get(pid, 0.0)

        def pid_to_mass_MeV(pid: int) -> float:
            MASS = {
                11: 0.510999,
                13: 105.6583755,
                211: 139.57039,
                2212: 938.2720813,
                2112: 939.5654133,
                22: 0.0,
                12: 0.0,
                14: 0.0,
            }
            return MASS.get(abs(pid), 0.0)

        def build_masses_and_charges(pids: np.ndarray):
            pid_vec = np.vectorize
            masses = pid_vec(pid_to_mass_MeV)(pids).astype(np.float32)
            charges = pid_vec(pid_to_charge)(pids).astype(np.float32)
            if masses.ndim == 1:
                masses = masses.reshape((1, -1))
                charges = charges.reshape((1, -1))
            return masses, charges

        masses_in, charges_in = build_masses_and_charges(pid)
        print("masses shape:", masses_in.shape, "dtype:", masses_in.dtype)
        print("charges shape:", charges_in.shape, "dtype:", charges_in.dtype)

        # expand to reserved slots
        masses = np.ones((n_events, p_slots), dtype=np.float32)
        charges = np.zeros((n_events, p_slots), dtype=np.float32)
        masses[:, :n_particles] = masses_in
        charges[:, :n_particles] = charges_in

        if design is None:
            design = np.zeros((1, self.design_dim()), dtype=np.float32)
        layers, angles, widths, heights, Bs, Ls = self.get_design(design)

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
            # Sparse mode
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

            events, particles, layers_arr, straws, values, edep_sparse, r_mm_sparse, t0_sparse, hit_pos_sparse = sparse_result
            print(sparse_result)

        
        print("mask shape:", mask.shape, "dtype:", mask.dtype)

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
        

        
        for i in range(len(events)):
            event, particle, layer, straw, value, edep, r_mm, t0, hit_xyz = events[i], particles[i], layers_arr[i], straws[i], values[i], edep_sparse[i], r_mm_sparse[i], t0_sparse[i], hit_pos_sparse[i]

            t_MC_val = t0
            r_mm_val = r_mm
            
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
            station = layer // (self.n_layers_per_view * self.n_views_per_station)
            view = (layer % (self.n_layers_per_view * self.n_views_per_station)) // self.n_layers_per_view
            layer = layer % self.n_layers_per_view + 1
            key = (station, view, layer, straw)
            fdigi_times[key] = fdigi

        print("\n\n\n\n\n\n\n")
        print("fdigi_times:")
        for k, v in fdigi_times.items():
            print(f"{k}: {float(v)}")
        print("\n\n\n\n\n\n\n")
        
        # Return sparse data as dict for visualization
        if use_sparse:
            sparse_response = {
                'events': events,
                'particles': particles,
                'layers': layers_arr,
                'straws': straws,
                'values': values,
            }
        else:
            sparse_response = None
        
        return (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            trajectories,
            sparse_response,
            signal,
            fdigi_times,
            mask,  # <- new: which slots were secondaries
        )

    def __call__(self, seed: int, configurations: np.ndarray):
        """
        returns normalized tdc, normalized target
        """
        (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            _,
            measurements,
            signal,
        ) = self.simulate(seed, configurations)
        ground_truth = self.encode_ground_truth(
            masses, charges, initial_positions, initial_momentum
        )
        return ground_truth, measurements, signal
        ### measurements = (indx, tdc)

    def loss(self, target, predicted):
        import optax

        return optax.sigmoid_binary_cross_entropy(predicted, target)

    def metric(self, target, predicted):
        return (target > 0.5) == (predicted > 0.0)

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
