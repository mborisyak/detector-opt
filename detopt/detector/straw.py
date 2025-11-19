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

__all__ = ["StrawDetector", "sparse_to_dense", "SparseHits"]


class SparseHits:
    """Container for sparse hit representation."""
    def __init__(self, events, particles, layers, straws, values, edep, r_mm, t0, hit_pos):
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
            self.events, self.particles, self.layers, self.straws, self.values,
            self.edep, self.r_mm, self.t0, self.hit_pos,
            n_events, n_particles, n_layers, n_straws
        )


def sparse_to_dense(events, particles, layers, straws, values, edep, r_mm, t0, hit_pos,
                    n_events, n_particles, n_layers, n_straws):
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
    hit_pos_dense = np.zeros((n_events, n_particles, n_layers, n_straws, 3), dtype=np.float32)
    
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
    valid = (events >= 0) & (events < n_events) & \
            (particles >= 0) & (particles < n_particles) & \
            (layers >= 0) & (layers < n_layers) & \
            (straws >= 0) & (straws < n_straws)
    
    if np.any(valid):
        response[events[valid], particles[valid], layers[valid], straws[valid]] = values[valid]
        edep_dense[events[valid], particles[valid], layers[valid], straws[valid]] = edep[valid]
        r_mm_dense[events[valid], particles[valid], layers[valid], straws[valid]] = r_mm[valid]
        t0_dense[events[valid], particles[valid], layers[valid], straws[valid]] = t0[valid]
        hit_pos_dense[events[valid], particles[valid], layers[valid], straws[valid]] = hit_pos[valid]
    
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
        print("\n\n\nSet up\n\n\n")

    def design_shape(self):
        # positions + angles + magnetic field strength
        return (self.n_layers + self.n_layers + 1,)

    def output_shape(self):
        return (self.n_layers, self.n_straws)

    def target_shape(self):
        return ()

    def ground_truth_shape(self):
        # charges + positions + momenta (flattened)
        return (self.max_particles + 3 * self.max_particles + 3 * self.max_particles,)

    def get_design(self, design: np.ndarray):
        n, _ = design.shape
        m = self.n_layers

        print(n, m, end="\n\n\n")

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

    def simulate_from_root(
        self,
        rootfile,
        tree_name="Events",
        px_name="px",
        py_name="py",
        pz_name="pz",
        x_name="x",
        y_name="y",
        z_name="z",
        pid_name="pid",
        batch_size=None,
        design=None,
        use_sparse=True,
    ):
        """
        Load particle properties from a ROOT file and run the detector simulation.
        """
        if uproot is None:
            raise ImportError(
                "uproot is required to read ROOT files. Please install with `pip install uproot awkward`."
            )

        # how many particles to actually read for this test
        n_viz = 20
        with uproot.open(rootfile) as file:
            tree = file[tree_name]
            px = tree[px_name].array(library="np")[:n_viz] * 1e3  # GeV -> MeV
            py = tree[py_name].array(library="np")[:n_viz] * 1e3
            pz = tree[pz_name].array(library="np")[:n_viz] * 1e3
            x = tree[x_name].array(library="np")[:n_viz]
            y = tree[y_name].array(library="np")[:n_viz]
            z = tree[z_name].array(library="np")[:n_viz]
            pid = tree[pid_name].array(library="np")[:n_viz]

        # Treat all loaded particles as a single event
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
            events, particles, layers_arr, straws, values, edep_sparse, r_mm_sparse, t0_sparse, hit_pos_sparse = sparse_result
            print(sparse_result)
            # Convert to SparseHits object
            sparse_hits = SparseHits(events, particles, layers_arr, straws, values,
                                     edep_sparse, r_mm_sparse, t0_sparse, hit_pos_sparse)
            
            # Print sparse output information
            print(f"\n=== Sparse Hits Summary ===")
            print(f"Total hits: {len(sparse_hits)}")
            print(f"Events: {n_events}, Particles: {p_slots}, Layers: {self.n_layers}, Straws: {self.n_straws}")
            print(f"Sparse representation: {len(sparse_hits)} hits vs {n_events * p_slots * self.n_layers * self.n_straws} dense array elements")
            print(f"Memory savings: {100.0 * (1.0 - len(sparse_hits) / (n_events * p_slots * self.n_layers * self.n_straws)):.2f}%")
            
            if len(sparse_hits) > 0:
                print(f"\nFirst 10 hits:")
                print(f"{'Event':<8} {'Particle':<10} {'Layer':<8} {'Straw':<8} {'Value':<12} {'Edep (MeV)':<15} {'r_mm':<10} {'t0 (ns)':<12}")
                print("-" * 95)
                for i in range(min(10, len(sparse_hits))):
                    print(f"{sparse_hits.events[i]:<8} {sparse_hits.particles[i]:<10} "
                          f"{sparse_hits.layers[i]:<8} {sparse_hits.straws[i]:<8} "
                          f"{sparse_hits.values[i]:<12.6f} {sparse_hits.edep[i]:<15.6e} "
                          f"{sparse_hits.r_mm[i]:<10.4f} {sparse_hits.t0[i]:<12.6f}")
                if len(sparse_hits) > 10:
                    print(f"... and {len(sparse_hits) - 10} more hits")
            print("=" * 95 + "\n")
            
            # Convert to dense for compatibility with rest of code
            response, edep, r_mm, t0_arr, hit_pos = sparse_hits.to_dense(
                n_events, p_slots, self.n_layers, self.n_straws
            )
        else:
            # Dense mode: allocate dense arrays
            response = np.zeros(
                (n_events, p_slots, self.n_layers, self.n_straws), dtype=np.float32
            )
            edep = np.zeros_like(response, dtype=np.float32)
            r_mm = np.zeros_like(response, dtype=np.float32)
            t0_arr = np.zeros_like(response, dtype=np.float32)
            hit_pos = np.zeros(response.shape + (3,), dtype=np.float32)
            
            print("response shape:", response.shape, "dtype:", response.dtype)
            print("trajectories shape:", trajectories.shape, "dtype:", trajectories.dtype)
            print(
                "initial_positions shape:",
                initial_positions.shape,
                "dtype:",
                initial_positions.dtype,
            )

            # call C extension (dense mode)
            straw_detector.solve(
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
                response,
                edep,
                r_mm,
                t0_arr,
                hit_pos,
                mask,
            )
        print("mask shape:", mask.shape, "dtype:", mask.dtype)
        print(mask)
        print(trajectories)

        # waveform modeling
        from .straw_signal import straw_response

        waveforms = {}
        if use_sparse and sparse_hits is not None:
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
        else:
            # Use dense arrays
            for event in range(edep.shape[0]):
                for particle in range(edep.shape[1]):
                    for layer in range(edep.shape[2]):
                        for straw in range(edep.shape[3]):
                            Edep_mev = edep[event, particle, layer, straw]
                            r_mm_val = r_mm[event, particle, layer, straw]
                            t0 = t0_arr[event, particle, layer, straw]
                            if Edep_mev > 0:
                                t, s = straw_response(Edep_mev, r_mm_val, t0)
                                waveforms[(event, particle, layer, straw)] = (t, s)

        print("edep shape:", edep.shape, "dtype:", edep.dtype)
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
                key = (int(sparse_hits.events[i]), int(sparse_hits.particles[i]),
                       int(sparse_hits.layers[i]), int(sparse_hits.straws[i]))
                sparse_lookup[key] = i
        
        for key in waveforms:
            event, particle, layer, straw = key
            if sparse_lookup is not None and key in sparse_lookup:
                # Use sparse data
                hit_idx = sparse_lookup[key]
                hit_xyz = sparse_hits.hit_pos[hit_idx]
                t_MC_val = sparse_hits.t0[hit_idx]
                r_mm_val = sparse_hits.r_mm[hit_idx]
            else:
                # Use dense arrays
                hit_xyz = hit_pos[event, particle, layer, straw]
                t_MC_val = t0_arr[event, particle, layer, straw]
                r_mm_val = r_mm[event, particle, layer, straw]
            
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
        for kk in range(len(st)):
            if ns[kk] == 0:
                print(0)
            else:
                print(st[kk] / ns[kk])

        return (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            trajectories,
            response,
            signal,
            waveforms,
            t0_arr,
            r_mm,
            fdigi_times,
            mask,  # <- new: which slots were secondaries
        )

    def __call__(self, seed: int, configurations: np.ndarray):
        (
            masses,
            charges,
            initial_positions,
            initial_momentum,
            _,
            measurements,
            signal,
        ) = self.sample(seed, configurations)
        ground_truth = self.encode_ground_truth(
            masses, charges, initial_positions, initial_momentum
        )
        return ground_truth, measurements, signal

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
