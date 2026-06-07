import math
import os
import subprocess

import numpy as np

# For reading ROOT files
try:
    import uproot
except ImportError:
    uproot = None

from ..utils.encoding import (
    normal_to_uniform,
    uniform_to_normal,
    normal_to_uniform_jax,
    uniform_to_normal_jax,
)
from ..data import HNLDataLoader
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
    hit_pos_dense = np.zeros((n_events, n_particles, n_layers, n_straws, 3), dtype=np.float32)

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
        response[events[valid], particles[valid], layers[valid], straws[valid]] = values[valid]
        r_mm_dense[events[valid], particles[valid], layers[valid], straws[valid]] = r_mm[valid]
        t0_dense[events[valid], particles[valid], layers[valid], straws[valid]] = t0[valid]
        hit_pos_dense[events[valid], particles[valid], layers[valid], straws[valid]] = hit_pos[valid]

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
        max_particles=5,
        secondary_multiplier=5,
        angles_bounds=None,
        layer_width=None,
        layer_height=None,
        n_layers=None,
        n_straws=None,
        loss=None,
        p_spawn_single=0.0,
        p_spawn_pair=0.0,
        E_sec_MeV=0.01,
        origin=(-2.4590519e01, 4.7251717e01, 5.5158970e03),
        origin_sigma=(1.2907193e02, 1.3767969e02, 1.6168958e03),
        momentum=(-2.0980914e-01, 2.1826500e-01, 2.6931271e01),
        momentum_sigma=(7.2272283e-01, 4.9576724e-01, 1.2292634e01),
        constrain_stereo_angles: bool = False,  # If True, optimize single angle: [0, +α, -α, 0]
        optimize_stations: bool = True,  # If False, freeze station positions
        optimize_gaps: bool = True,  # If False, freeze layer_z_gap and view_z_gap
        optimize_bfield: bool = True,  # If False, freeze max_B, B_sigma, z0
        data_dir=None,  # event-source location (detector config; the only data the detector owns)
        val_fraction: float = 0.2,  # held-out fraction of the finite event source
        split_seed: int = 42,  # seed defining the train/val partition
    ):
        """
        :param max_B: maximal strength of the magnetic field;
        :param origin: the mean point of particles' origin;
        :param layer_bounds: restrictions on the layers' positions;
        :param dt: time increment for the ODE solver;
        :param data: optional data-source config; the detector owns its event
            source (see :meth:`__call__`). Lazily constructs an
            :class:`HNLDataLoader` on first use.
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
        self.constrain_stereo_angles = constrain_stereo_angles
        self.optimize_stations = optimize_stations
        self.optimize_gaps = optimize_gaps
        self.optimize_bfield = optimize_bfield

        self.n_layers = self.n_stations * self.n_views_per_station * self.n_layers_per_view

        # Layer height/width for visualization/hit logic
        self.layer_height = self.straw_pitch * self.n_straws / 2.0  # half-length for +/- y
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
        self.n_t = int(flight_distance / (dt * 29.9792))

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

        # Owned event source: only the *location*. The train/val split below is
        # part of the detector's own event-source config (not injected by the
        # optimiser): the loader partitions the finite dataset on first use.
        self._data_dir = data_dir
        self._val_fraction = float(val_fraction)
        self._split_seed = int(split_seed)
        self._loader = None

    # ------------------------------------------------------------------ #
    # Event source
    # ------------------------------------------------------------------ #
    @property
    def loader(self) -> HNLDataLoader:
        if self._loader is None:
            if self._data_dir is None:
                raise RuntimeError(
                    "StrawDetector has no event source configured; pass `data_dir` "
                    "in the detector config to generate events."
                )
            self._loader = HNLDataLoader(
                data_dir=self._data_dir,
                max_particles=self.max_particles,
                val_fraction=self._val_fraction,
                split_seed=self._split_seed,
            )
        return self._loader

    @property
    def n_events(self) -> int:
        return int(self.loader.n_events)

    # ------------------------------------------------------------------ #
    # Shapes
    # ------------------------------------------------------------------ #
    @property
    def max_hits_per_event(self) -> int:
        """``M`` in the per-event padded layout."""
        return 2 * self.max_particles * self.n_layers

    def design_shape(self):
        # positions + angles + magnetic field strength
        return (self.n_layers + self.n_layers + 1,)

    def output_shape(self):
        """Legacy dense output grid shape (kept for older code)."""
        return (self.n_layers, self.n_straws)

    def target_shape(self):
        # decay vertex + HNL momentum: [x, y, z, px, py, pz]
        return (6,)

    def event_shape(self):
        # per-hit raw features: [station, view, layer_in_view, straw, time]
        return (self.max_hits_per_event, 5)

    def combined_event_shape(self):
        return (self.max_hits_per_event, self.event_shape()[-1] + self.design_dim())

    def ground_truth_shape(self):
        # charges + positions + momenta (flattened)
        return (2 * self.max_particles + 3 * self.max_particles + 3 * self.max_particles,)

    # ------------------------------------------------------------------ #
    # Ground-truth encoding (daughter particles)
    # ------------------------------------------------------------------ #
    def encode_ground_truth(self, masses, charges, initial_positions, initial_momentum):
        n, *_ = initial_positions.shape
        normalized_positions = (initial_positions - self.origin) / self.origin_sigma
        normalized_positions = np.reshape(normalized_positions, shape=(n, -1))
        normalized_momenta = np.reshape(initial_momentum, shape=(n, -1))
        ground_truth = np.concatenate([masses, charges, normalized_positions, normalized_momenta], axis=-1)
        return ground_truth

    def decode_ground_truth(self, ground_truth):
        masses = ground_truth[:, : self.max_particles]
        batch = masses.shape[0]
        charges = ground_truth[:, self.max_particles : self.max_particles + self.max_particles]
        pos_flat = ground_truth[
            :,
            self.max_particles + self.max_particles : self.max_particles + self.max_particles + self.max_particles * 3,
        ]
        mom_flat = ground_truth[:, self.max_particles + self.max_particles + self.max_particles * 3 :]

        normalized_positions = pos_flat.reshape((batch, self.max_particles, 3))
        initial_momentum = mom_flat.reshape((batch, self.max_particles, 3))

        initial_positions = normalized_positions * self.origin_sigma + self.origin

        return masses, charges, initial_positions, initial_momentum

    # ------------------------------------------------------------------ #
    # Geometry from a physical (un-encoded) design array
    # ------------------------------------------------------------------ #
    def _design_to_geometry(self, design):
        """Split a *physical* design ``[positions(n), angles(n), B]`` into the
        per-layer geometry arrays the C solver expects.

        Returns ``(layers, angles, widths, heights, Bs)`` with batch dim ``n``.
        """
        design = np.asarray(design, dtype=np.float32)
        if design.ndim == 1:
            design = design[None, :]
        n_batch = design.shape[0]
        m = self.n_layers

        layers = design[:, :m].astype(np.float32)
        angles = design[:, m : 2 * m].astype(np.float32)
        Bs = design[:, 2 * m].astype(np.float32)

        widths = np.full((n_batch, m), self.layer_width, dtype=np.float32)
        heights = np.full((n_batch, m), self.layer_height, dtype=np.float32)
        return layers, angles, widths, heights, Bs

    def layer_design_to_array(self, layer_design):
        """Flatten a ``{positions, angles, magnetic_strength}`` dict into the
        physical design array ``[positions(n), angles(n), B]``."""
        positions = np.asarray(layer_design["positions"], dtype=np.float32)
        angles = np.asarray(layer_design["angles"], dtype=np.float32)
        B = np.float32(layer_design["magnetic_strength"])
        return np.concatenate([positions, angles, np.array([B], np.float32)]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Event generation
    # ------------------------------------------------------------------ #
    def _sample_daughters(self, seed, n, split):
        rng = np.random.default_rng(seed)
        daughter_data, targets = self.loader.get_batch(batch_size=int(n), rng=rng, split=(split or "all"))
        return daughter_data, np.asarray(targets, dtype=np.float32)

    def _run_solver(self, daughter_data, design):
        """Run the C straw solver for a physical ``design`` and ``daughter_data``.

        Returns ``(sparse_hits, fdigi_times, n_hits, trajectories)``.
        """
        masses = daughter_data["masses"]
        charges = daughter_data["charges"]
        initial_positions = daughter_data["positions"]
        initial_momentum = daughter_data["momenta"]
        initial_times = daughter_data["times"]

        layers, angles, widths, heights, Bs = self._design_to_geometry(design)

        n_events = layers.shape[0]
        p_slots = initial_momentum.shape[1]

        Bs_arr = Bs.astype(np.float32)
        z0_arr = np.full((n_events,), self.z0, dtype=np.float32)
        B_sigma_arr = np.full((n_events,), self.B_sigma, dtype=np.float32)

        max_hits = 2 * n_events * self.max_particles * self.n_layers
        trajectories = np.zeros((n_events, self.max_particles, self.n_t, 3), dtype=np.float32)

        sparse_events = np.zeros(max_hits, dtype=np.int32)
        sparse_particles = np.zeros(max_hits, dtype=np.int32)
        sparse_layers = np.zeros(max_hits, dtype=np.int32)
        sparse_straws = np.zeros(max_hits, dtype=np.int32)
        sparse_values = np.zeros(max_hits, dtype=np.float32)
        sparse_r_mm = np.zeros(max_hits, dtype=np.float32)
        sparse_t0 = np.zeros(max_hits, dtype=np.float32)
        sparse_hit_pos = np.zeros((max_hits, 3), dtype=np.float32)
        sparse_count = np.zeros(1, dtype=np.int32)

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

        n_hits = int(sparse_count[0])
        assert 0 <= n_hits <= max_hits, (n_hits, max_hits)

        sparse_hits = SparseHits(
            sparse_events,
            sparse_particles,
            sparse_layers,
            sparse_straws,
            sparse_values,
            sparse_r_mm,
            sparse_t0,
            sparse_hit_pos,
        )

        # FairShip-style TDC (fdigi) times per hit.
        fdigi_times = np.zeros(max_hits, dtype=np.float32)
        if n_hits > 0:
            v_drift = 0.0033  # cm/ns
            sigma_spatial = 0.012  # cm
            c = 29.9792  # cm/ns

            for i in range(n_hits):
                r_mm_val = float(sparse_hits.r_mm[i])
                t_MC_val = float(sparse_hits.t0[i])
                hit_xyz = sparse_hits.hit_pos[i]

                dist_cm = r_mm_val / 10.0
                dist_smeared = abs(np.random.normal(dist_cm, sigma_spatial))
                t_drift = dist_smeared / v_drift

                # signal propagation along the (x-oriented) wire to its +x end
                propagation_time = (self.layer_width - hit_xyz[0]) / c
                fdigi_times[i] = t_MC_val + t_drift + propagation_time

        return sparse_hits, fdigi_times, n_hits, trajectories

    def _bucket_hits(self, sparse_hits, fdigi_times, n_hits, n_events):
        """Bucket flat hits into per-event padded ``X (B, M, 5)`` + ``mask (B, M)``.

        Raw per-hit features: ``[station, view, layer_in_view, straw, time]``.
        """
        M = self.max_hits_per_event
        per_station = self.n_views_per_station * self.n_layers_per_view

        X = np.zeros((n_events, M, 5), dtype=np.float32)
        mask = np.zeros((n_events, M), dtype=np.int32)

        if n_hits > 0:
            ev = sparse_hits.events[:n_hits].astype(np.int32)
            la = sparse_hits.layers[:n_hits].astype(np.int32)
            st = sparse_hits.straws[:n_hits].astype(np.int32)
            ti = fdigi_times[:n_hits].astype(np.float32)

            valid = (ev >= 0) & (ev < n_events)
            ev, la, st, ti = ev[valid], la[valid], st[valid], ti[valid]

            order = np.argsort(ev, kind="stable")
            ev, la, st, ti = ev[order], la[order], st[order], ti[order]

            starts = np.searchsorted(ev, np.arange(n_events), side="left")
            ends = np.searchsorted(ev, np.arange(n_events), side="right")
            for e in range(n_events):
                s, t = int(starts[e]), int(ends[e])
                k = min(t - s, M)
                if k <= 0:
                    continue
                layer = la[s : s + k]
                station = layer // per_station
                rem = layer % per_station
                view = rem // self.n_layers_per_view
                layer_in_view = rem % self.n_layers_per_view
                X[e, :k, 0] = station
                X[e, :k, 1] = view
                X[e, :k, 2] = layer_in_view
                X[e, :k, 3] = st[s : s + k]
                X[e, :k, 4] = ti[s : s + k]
                mask[e, :k] = 1

        return X, mask

    def sample_events(self, seed, design, split=None):
        """Generate events and return a rich dict.

        ``design`` is the *physical* (un-encoded) design ``(B, design_dim)`` (or
        ``(design_dim,)`` for a single event). Batch size ``B`` is inferred.
        Used by visualisation / likelihood-free / generative scripts that need
        the daughter ground truth or trajectories.
        """
        design = np.asarray(design, dtype=np.float32)
        if design.ndim == 1:
            design = design[None, :]
        n_events = design.shape[0]

        daughter_data, targets = self._sample_daughters(seed, n_events, split)
        sparse_hits, fdigi_times, n_hits, trajectories = self._run_solver(daughter_data, design)
        X, mask = self._bucket_hits(sparse_hits, fdigi_times, n_hits, n_events)

        ground_truth = self.encode_ground_truth(
            daughter_data["masses"],
            daughter_data["charges"],
            daughter_data["positions"],
            daughter_data["momenta"],
        )
        return {
            "X": X,
            "mask": mask,
            "targets": targets,
            "ground_truth": ground_truth,
            "trajectories": trajectories,
            "sparse_hits": sparse_hits,
        }

    def __call__(self, seed, design, split=None):
        """Generate events for an un-encoded ``design`` ``(B, design_dim)``.

        Returns ``(ground_truth (B, G), measurements (B, M, 5), mask (B, M),
        target (B, 6))``.
        """
        out = self.sample_events(seed, design, split=split)
        return out["ground_truth"], out["X"], out["mask"], out["targets"]

    # ------------------------------------------------------------------ #
    # Design encoding (constrained <-> unconstrained), differentiable
    # ------------------------------------------------------------------ #
    def encode_design(self, design):
        """Physical design ``[positions(n), angles(n), B]`` -> ``N(0,1)`` space.

        JAX/jittable; accepts ``(design_dim,)`` or ``(B, design_dim)``.
        """
        import jax.numpy as jnp

        design = jnp.asarray(design, dtype=jnp.float32)
        n = self.n_layers
        pos = design[..., :n]
        ang = design[..., n : 2 * n]
        B = design[..., 2 * n : 2 * n + 1]

        pos_e = uniform_to_normal_jax(pos, *self.layer_bounds)
        ang_e = uniform_to_normal_jax(ang, *self.angle_bounds)
        B_e = uniform_to_normal_jax(B, 0.0, self.max_B)
        return jnp.concatenate([pos_e, ang_e, B_e], axis=-1)

    def decode_design(self, encoded_design):
        """Inverse of :meth:`encode_design` (JAX/jittable)."""
        import jax.numpy as jnp

        enc = jnp.asarray(encoded_design, dtype=jnp.float32)
        n = self.n_layers
        pos = enc[..., :n]
        ang = enc[..., n : 2 * n]
        B = enc[..., 2 * n : 2 * n + 1]

        pos_d = normal_to_uniform_jax(pos, *self.layer_bounds)
        ang_d = normal_to_uniform_jax(ang, *self.angle_bounds)
        B_d = normal_to_uniform_jax(B, 0.0, self.max_B)
        return jnp.concatenate([pos_d, ang_d, B_d], axis=-1)

    # ------------------------------------------------------------------ #
    # Event normalisation
    # ------------------------------------------------------------------ #
    # Continuous-feature standardisation constants.
    _TDC_MEAN = 440.0
    _TDC_STD = 80.0

    def _feature_scales(self):
        """Per-column ``(mean, std)`` for the 5 raw event features."""
        means = np.array([0.0, 0.0, 0.0, 0.0, self._TDC_MEAN], dtype=np.float32)
        stds = np.array(
            [
                max(self.n_stations - 1, 1),
                max(self.n_views_per_station - 1, 1),
                max(self.n_layers_per_view - 1, 1),
                max(self.n_straws - 1, 1),
                self._TDC_STD,
            ],
            dtype=np.float32,
        )
        return means, stds

    def normalize(self, X):
        """``(B, M, 5)`` raw event features -> standardised ~[-1, 1]."""
        import jax.numpy as jnp

        means, stds = self._feature_scales()
        return (jnp.asarray(X, dtype=jnp.float32) - means) / stds

    def denormalize(self, X_norm):
        import jax.numpy as jnp

        means, stds = self._feature_scales()
        return jnp.asarray(X_norm, dtype=jnp.float32) * stds + means

    # ------------------------------------------------------------------ #
    # Combine
    # ------------------------------------------------------------------ #
    def combine(self, X_norm, encoded_design):
        """Broadcast-concatenate an encoded design onto each normalised hit.

        ``X_norm (B, M, 5)`` + ``encoded_design (design_dim,) | (B, design_dim)``
        -> ``features (B, M, 5 + design_dim)``. Differentiable w.r.t. both.
        The hit ``mask`` is applied downstream by the model.
        """
        import jax.numpy as jnp

        X_norm = jnp.asarray(X_norm, dtype=jnp.float32)
        B, M, _ = X_norm.shape

        d_enc = jnp.asarray(encoded_design, dtype=jnp.float32)
        if d_enc.ndim == 1:
            d_enc = jnp.broadcast_to(d_enc[None, :], (B, d_enc.shape[0]))
        d_per_hit = jnp.broadcast_to(d_enc[:, None, :], (B, M, d_enc.shape[-1]))

        return jnp.concatenate([X_norm, d_per_hit], axis=-1)

    # ------------------------------------------------------------------ #
    # Current design helpers
    # ------------------------------------------------------------------ #
    def get_current_design(self):
        positions = []
        angles = []

        # layer ordering: station -> view -> layer-within-view
        for z_station in self.station_z:
            for v in range(self.n_views_per_station):
                view_base_z = z_station + v * self.view_z_gap
                ang = self.view_angles[v] if v < len(self.view_angles) else self.view_angles[-1]
                for l in range(self.n_layers_per_view):
                    positions.append(view_base_z + l * self.layer_z_gap)
                    angles.append(ang)

        if len(positions) != self.n_layers:
            raise RuntimeError(f"current_design_dict produced {len(positions)} layers, expected {self.n_layers}")

        return {
            "positions": positions,
            "angles": angles,
            "magnetic_strength": float(self.max_B),
        }

    def get_current_design_array(self):
        return self.layer_design_to_array(self.get_current_design())

    def get_encoded_current_design(self):
        return np.asarray(self.encode_design(self.get_current_design_array()), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # YAML (high-level) design space -- used by the BO outer loop
    # ------------------------------------------------------------------ #
    def encode_yaml_design(self, yaml_params):
        """Encode YAML parameters into the normalized BO search space."""
        params = []

        station_z = np.array(yaml_params.get("station_z", self.station_z), dtype=np.float32)

        if self.constrain_stereo_angles:
            if "stereo_angle" in yaml_params:
                stereo_angle = np.float32(yaml_params["stereo_angle"])
            elif "view_angles" in yaml_params:
                angles = yaml_params["view_angles"]
                stereo_angle = np.float32(angles[1]) if len(angles) > 1 else 0.0
            else:
                stereo_angle = np.float32(self.view_angles[1])
        else:
            view_angles = np.array(yaml_params.get("view_angles", self.view_angles), dtype=np.float32)
        layer_z_gap = np.float32(yaml_params.get("layer_z_gap", self.layer_z_gap))
        view_z_gap = np.float32(yaml_params.get("view_z_gap", self.view_z_gap))
        max_B = np.float32(yaml_params.get("max_B", self.max_B))
        B_sigma = np.float32(yaml_params.get("B_sigma", self.B_sigma))
        z0 = np.float32(yaml_params.get("z0", self.z0))

        if self.optimize_stations:
            station_z_norm = uniform_to_normal(station_z, *self.layer_bounds)
            params.append(station_z_norm)

        if self.constrain_stereo_angles:
            stereo_angle_norm = uniform_to_normal(stereo_angle, 0.0, 0.2)
            params.append([stereo_angle_norm])
        else:
            view_angles_norm = uniform_to_normal(view_angles, -0.2, 0.2)
            params.append(view_angles_norm)

        if self.optimize_gaps:
            layer_z_gap_norm = uniform_to_normal(layer_z_gap, 0.5, 10.0)
            view_z_gap_norm = uniform_to_normal(view_z_gap, 1.0, 20.0)
            params.append([layer_z_gap_norm])
            params.append([view_z_gap_norm])

        if self.optimize_bfield:
            max_B_norm = uniform_to_normal(max_B, 0.0, 1.0)
            B_sigma_norm = uniform_to_normal(B_sigma, 50.0, 1000.0)
            z0_norm = uniform_to_normal(z0, *self.layer_bounds)
            params.append([max_B_norm])
            params.append([B_sigma_norm])
            params.append([z0_norm])

        return np.concatenate(params, axis=0)

    def decode_yaml_design(self, encoded):
        """Decode normalized BO parameters back to YAML physical parameters."""
        idx = 0

        if self.optimize_stations:
            n_stations = len(self.station_z)
            station_z = normal_to_uniform(encoded[idx : idx + n_stations], *self.layer_bounds)
            idx += n_stations
        else:
            station_z = np.array(self.station_z, dtype=np.float32)

        if self.constrain_stereo_angles:
            stereo_angle = normal_to_uniform(encoded[idx], 0.0, 0.2)
            idx += 1
            view_angles = np.array([0.0, stereo_angle, -stereo_angle, 0.0], dtype=np.float32)
        else:
            n_angles = len(self.view_angles)
            view_angles = normal_to_uniform(encoded[idx : idx + n_angles], -0.2, 0.2)
            idx += n_angles

        if self.optimize_gaps:
            layer_z_gap = normal_to_uniform(encoded[idx], 0.5, 10.0)
            idx += 1
            view_z_gap = normal_to_uniform(encoded[idx], 1.0, 20.0)
            idx += 1
        else:
            layer_z_gap = np.float32(self.layer_z_gap)
            view_z_gap = np.float32(self.view_z_gap)

        if self.optimize_bfield:
            max_B = normal_to_uniform(encoded[idx], 0.0, 1.0)
            idx += 1
            B_sigma = normal_to_uniform(encoded[idx], 50.0, 1000.0)
            idx += 1
            z0 = normal_to_uniform(encoded[idx], *self.layer_bounds)
        else:
            max_B = np.float32(self.max_B)
            B_sigma = np.float32(self.B_sigma)
            z0 = np.float32(self.z0)

        result = {
            "station_z": [float(z) for z in station_z],
            "view_angles": [float(a) for a in view_angles],
            "layer_z_gap": float(layer_z_gap),
            "view_z_gap": float(view_z_gap),
            "max_B": float(max_B),
            "B_sigma": float(B_sigma),
            "z0": float(z0),
        }
        if self.constrain_stereo_angles:
            result["stereo_angle"] = float(view_angles[1])
        return result

    def yaml_to_layer_design(self, yaml_params):
        """Expand YAML parameters into a per-layer ``{positions, angles, B}`` dict."""
        positions = []
        angles = []

        station_z = yaml_params["station_z"]
        view_angles = yaml_params["view_angles"]
        layer_z_gap = yaml_params["layer_z_gap"]
        view_z_gap = yaml_params["view_z_gap"]

        for z_station in station_z:
            for v in range(self.n_views_per_station):
                view_base_z = z_station + v * view_z_gap
                ang = view_angles[v] if v < len(view_angles) else view_angles[-1]
                for l in range(self.n_layers_per_view):
                    positions.append(view_base_z + l * layer_z_gap)
                    angles.append(ang)

        if len(positions) != self.n_layers:
            raise RuntimeError(f"yaml_to_layer_design produced {len(positions)} layers, " f"expected {self.n_layers}")

        return {
            "positions": positions,
            "angles": angles,
            "magnetic_strength": yaml_params["max_B"],
        }

    def yaml_to_design_array(self, yaml_params):
        """YAML parameters -> physical design array ``[positions(n), angles(n), B]``."""
        return self.layer_design_to_array(self.yaml_to_layer_design(yaml_params))

    def get_current_yaml_design(self):
        return {
            "station_z": list(self.station_z),
            "view_angles": list(self.view_angles),
            "layer_z_gap": float(self.layer_z_gap),
            "view_z_gap": float(self.view_z_gap),
            "max_B": float(self.max_B),
            "B_sigma": float(self.B_sigma),
            "z0": float(self.z0),
        }

    def get_encoded_current_yaml_design(self):
        yaml_params = self.get_current_yaml_design()
        enc = self.encode_yaml_design(yaml_params)
        return np.asarray(enc, dtype=np.float32)

    def yaml_design_shape(self):
        """Shape of the YAML (BO) design parameter vector."""
        n_params = 0
        if self.optimize_stations:
            n_params += len(self.station_z)
        if self.constrain_stereo_angles:
            n_params += 1
        else:
            n_params += len(self.view_angles)
        if self.optimize_gaps:
            n_params += 2
        if self.optimize_bfield:
            n_params += 3
        return (n_params,)
