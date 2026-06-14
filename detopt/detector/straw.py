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
    normal_to_uniform_jax,
    uniform_to_normal_jax,
)
from ..data import load_ship2numpy_events
from . import straw_detector
from .common import Detector

__all__ = ["StrawDetector", "MATERIALS", "material_constants"]


# Energy-independent material constants for secondary production (PDG values).
# Per material: (Z/A, density [g/cm^3], radiation length X0 [cm]). The straw wall
# in FairShip is Kapton (strawtubes media), so it is the default; the support
# frame is aluminium. From these:
#   delta_const = (K/2)(Z/A) rho  [MeV/cm]   -- material part of the delta-ray rate
#   lambda_conv = (9/7) X0        [cm]       -- photon / pair conversion length
# with K = 0.307075 MeV mol^-1 cm^2. The per-hit probabilities (energy-dependent
# for delta-rays via z^2/beta^2, energy-independent for pair) are built in the C
# solver; here we only precompute the constants.
# Backstop on the C physics-loop iteration count. The loop really terminates on the
# max_time flight-time break (every step advances time by dt_step > 0); this is just a
# guard against a pathological non-advancing step, decoupled from n_t.
_MAX_SOLVER_STEPS = 200_000

_K_HALF = 0.307075 / 2.0  # MeV mol^-1 cm^2
MATERIALS = {
    "kapton": (0.51264, 1.42, 28.56),  # polyimide film -- FairShip straw wall (default)
    "mylar": (0.52037, 1.40, 28.54),  # polyethylene terephthalate
    "aluminium": (0.48181, 2.699, 8.897),  # support frame
}


def material_constants(material):
    """Return ``(delta_const [MeV/cm], lambda_conv [cm])`` for a named material."""
    z_over_a, rho, x0_cm = MATERIALS[material]
    return _K_HALF * z_over_a * rho, (9.0 / 7.0) * x0_cm


class StrawDetector(Detector):
    def __init__(
        self,
        # Geometry hierarchy (structural counts + fixed hardware; NO nominal design)
        n_stations: int = 4,
        n_views_per_station: int = 4,
        n_layers_per_view: int = 2,
        n_straws_per_layer: int = 200,
        straw_pitch: float = 2.0,  # cm; centre-to-centre straw spacing (FairShip strawtubes_config.yaml)
        straw_length: float = 400.0,
        layer_y_offset: float = 1.0,  # cm; half-pitch y-stagger between the two layers in a view
        # Physics parameters
        max_B: float = 0.20,  # T; peak |Bx| of FairShip V13_3500 spectrometer field map (on-axis)
        z0: float = 8957.0,  # cm; magnet centre (FairShip geometry_config c.z = 89.57 m, map z_local=0)
        B_sigma: float = 286.0,  # cm; our-form width from a Gaussian fit to the V13_3500 Bx(z) on-axis profile
        # Design-space bounds (the design itself is supplied per-call, not stored)
        layer_bounds: tuple[float | int, float | int] = (8000.0, 10000.0),
        angles_bounds=None,
        dt=None,  # fixed step (ns); None -> adaptive per-step dt (sagitta-bounded), capped by max_dt
        max_dt=1.0,  # upper clamp (ns) on the adaptive step; bounds the field-free Bx->0 region
        max_time=200.0,  # total integration-time cap (ns) per particle; trapped/curling tracks stop here
        max_particles=5,
        material="kapton",  # straw-wall material (FairShip default); sets delta_const + lambda_conv
        wall_thickness=0.0036,  # straw wall thickness (cm); FairShip strawtubes_config.yaml
        delta_Tcut=0.5,  # delta-ray tracking threshold (MeV): production cut for knock-on electrons
        lambda_conv_cm=None,  # photon/pair conversion length (cm); None -> 9/7 * X0 of `material`
        enable_decay=False,  # charged pi/K decay-in-flight
        noise_rate=0.0,  # mean uncorrelated noise hits per event; <=0 disables
        scatter_xX0=0.0,  # Highland multiple-scattering material budget x/X0 per layer crossing; <=0 disables
        # Target (6-vec) normalization: decay vertex (cm) + HNL momentum (GeV).
        # Defaults computed from data/mc (180,672 events); None -> derive from the
        # loaded `targets` at construction.
        decay_mean=(-0.2423, 0.2345, 6205.60),
        decay_sigma=(76.11, 101.06, 1365.96),
        momentum_mean=(-0.0027, 0.0021, 46.686),
        momentum_sigma=(0.5421, 0.6479, 28.203),
        data_dir=None,  # ship2numpy event file, preloaded once (None = no event source)
        boundary_z=None,  # crossing-plane z (cm); None -> read from the file
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

        # Target-normalization constants (resolved after the event pool is loaded;
        # any left None are derived from the data's `targets`). See _resolve_target_norm.
        self._decay_mean_arg = decay_mean
        self._decay_sigma_arg = decay_sigma
        self._momentum_mean_arg = momentum_mean
        self._momentum_sigma_arg = momentum_sigma

        # Secondary production is material-driven: the C solver builds the per-hit
        # delta-ray / pair probabilities from these energy-independent constants and
        # the track's (z, beta, gamma). The wall material sets delta_const and the
        # photon/pair conversion length; the track crosses 2 walls per hit.
        self.material = material
        self.wall_thickness = float(wall_thickness)
        self.delta_Tcut = float(delta_Tcut)
        self.delta_const, _lambda_default = material_constants(material)
        self.lambda_conv_cm = float(lambda_conv_cm) if lambda_conv_cm is not None else _lambda_default
        self.enable_decay = int(bool(enable_decay))
        self.noise_rate = float(noise_rate)
        # Highland multiple scattering: angular kick per layer crossing, scaled by
        # the layer material budget x/X0 (straw walls + gas + frame). 0 -> clean tracks.
        self.scatter_xX0 = float(scatter_xX0)

        self.layer_bounds = layer_bounds

        # Real detector geometry (structural counts + fixed hardware only)
        self.n_stations = int(n_stations)
        self.n_views_per_station = int(n_views_per_station)
        self.n_layers_per_view = int(n_layers_per_view)
        self.n_straws = int(n_straws_per_layer)
        self.straw_pitch = float(straw_pitch)
        self.straw_length = float(straw_length)
        self.layer_y_offset = float(layer_y_offset)

        self.n_layers = self.n_stations * self.n_views_per_station * self.n_layers_per_view

        # Layer height/width for visualization/hit logic
        self.layer_height = self.straw_pitch * self.n_straws / 2.0  # half-length for +/- y
        self.layer_width = self.straw_length / 2.0  # half-length for +/- x

        # Angle bounds
        if angles_bounds is not None:
            self.angle_bounds = angles_bounds
        else:
            self.angle_bounds = (-0.1, 0.1)

        assert max_particles > 1, "signal events produce at least 2 particles"
        self.max_particles = int(max_particles)

        # Integration step is adaptive per-step in the C solver: each step is 0.9x
        # the sagitta-bounded ceiling (chord-vs-arc error < straw resolution),
        # clamped to max_dt. A fixed `dt` (not None) overrides it. The trajectory
        # buffer is sampled evenly in time, decoupled from the physics step: n_t
        # samples over [0, max_time], so step_t = max_time / n_t ~ max_dt.
        self.dt_fixed = float(dt) if dt is not None else 0.0  # 0 -> adaptive
        self.max_dt = float(max_dt)
        self.max_time = float(max_time)
        self.n_t = max(int(self.max_time / self.max_dt) + 1, 2)  # trajectory viz samples
        # Physics-loop iteration backstop passed to the C solver. The loop's real
        # termination is the `max_time` flight-time break (every step advances time
        # by dt_step > 0); this is just a guard, decoupled from n_t.
        self.max_steps = _MAX_SOLVER_STEPS

        # Event source: preload the file once into immutable per-event arrays and
        # sample from them by seed. No train/val split, no mutable loader state --
        # that does not belong to the detector. Mirrors DebugDetector, which
        # generates events on demand from a seed (here drawn from a preloaded pool).
        self._events = None
        self._n_events = 0
        self.boundary_z = float(boundary_z) if boundary_z is not None else None
        if data_dir is not None:
            self._events = load_ship2numpy_events(data_dir, self.max_particles, boundary_z=boundary_z)
            self._n_events = self._events["n_events"]
            self.boundary_z = self._events["boundary_z"]

        # Resolve target-normalization constants (derive any left None from data).
        self._resolve_target_norm()

    @classmethod
    def from_config(cls, config):
        """Build the detector from a (yaml-parsed) ``config`` dict, passing its
        entries straight into ``__init__``. Unlike the base blind-splat, this
        validates the keys against the constructor signature(s) so an unknown or
        mistyped key raises instead of being silently ignored."""
        import inspect

        allowed = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, p in inspect.signature(init).parameters.items():
                if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                allowed.add(name)
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown {cls.__name__} config key(s): {sorted(unknown)}")
        return cls(**config)

    # ------------------------------------------------------------------ #
    # Event source
    # ------------------------------------------------------------------ #
    @property
    def n_events(self) -> int:
        return int(self._n_events)

    # ------------------------------------------------------------------ #
    # Shapes
    # ------------------------------------------------------------------ #
    @property
    def max_hits_per_event(self) -> int:
        """``M`` in the per-event padded layout."""
        return 2 * self.max_particles * self.n_layers

    def design_shape(self):
        # The base carries no design scheme -- subclasses define the design space.
        raise NotImplementedError("design_shape is defined by the design subclass")

    def target_shape(self):
        # decay vertex + HNL momentum: [x, y, z, px, py, pz]
        return (6,)

    def event_shape(self):
        # per-hit raw features: [station, view, layer_in_view, straw, time]
        return (self.max_hits_per_event, 5)

    def combined_event_shape(self):
        # combine() -> [time, norm_layer_z, norm_straw_y, norm_angle, norm_B]
        return (self.max_hits_per_event, 5)

    def ground_truth_shape(self):
        # charges + positions + momenta (flattened)
        return (2 * self.max_particles + 3 * self.max_particles + 3 * self.max_particles,)

    # ------------------------------------------------------------------ #
    # Ground-truth encoding (daughter particles) -- unimplemented for the sparse
    # pool. Only generators/LFI consume it (as conditioning); to be reworked.
    # ------------------------------------------------------------------ #
    def encode_ground_truth(self, masses, charges, initial_positions, initial_momentum):
        raise NotImplementedError("ground-truth encoding is not implemented for the sparse pool (to be reworked)")

    def decode_ground_truth(self, ground_truth):
        raise NotImplementedError("ground-truth decoding is not implemented for the sparse pool (to be reworked)")

    # ------------------------------------------------------------------ #
    # Target normalization (6-vec [decay_vertex(3, cm), HNL_momentum(3, GeV)])
    # ------------------------------------------------------------------ #
    def _resolve_target_norm(self):
        """Set ``decay_mean/sigma`` + ``momentum_mean/sigma``, deriving any passed
        as ``None`` from the loaded data's ``targets`` (vertex ``[:, :3]``,
        momentum ``[:, 3:]``). Sigmas are floored to avoid divide-by-zero."""
        args = (self._decay_mean_arg, self._decay_sigma_arg, self._momentum_mean_arg, self._momentum_sigma_arg)
        if any(a is None for a in args):
            if self._events is None:
                raise ValueError(
                    "target normalization has None entries but no data source to derive them from; "
                    "pass `data_dir` or supply decay_mean/decay_sigma/momentum_mean/momentum_sigma."
                )
            t = self._events["targets"]  # (N, 6)
            d_mean, d_std, m_mean, m_std = t[:, :3].mean(0), t[:, :3].std(0), t[:, 3:].mean(0), t[:, 3:].std(0)
        else:
            d_mean = d_std = m_mean = m_std = None

        pick = lambda arg, derived: np.asarray(derived if arg is None else arg, dtype=np.float32)
        self.decay_mean = pick(self._decay_mean_arg, d_mean)
        self.decay_sigma = np.maximum(pick(self._decay_sigma_arg, d_std), 1e-3)
        self.momentum_mean = pick(self._momentum_mean_arg, m_mean)
        self.momentum_sigma = np.maximum(pick(self._momentum_sigma_arg, m_std), 1e-3)

    def _target_norm_arrays(self):
        import jax.numpy as jnp

        mean = jnp.concatenate([jnp.asarray(self.decay_mean), jnp.asarray(self.momentum_mean)])
        std = jnp.concatenate([jnp.asarray(self.decay_sigma), jnp.asarray(self.momentum_sigma)])
        return mean, std

    def normalize_target(self, target):
        """Physical 6-vec target -> standardised; vertex by decay_*, momentum by momentum_*."""
        import jax.numpy as jnp

        mean, std = self._target_norm_arrays()
        return (jnp.asarray(target, dtype=jnp.float32) - mean) / std

    def denormalize_predictions(self, normalised):
        """Inverse of :meth:`normalize_target`: back to physical units (cm, GeV)."""
        import jax.numpy as jnp

        mean, std = self._target_norm_arrays()
        return jnp.asarray(normalised, dtype=jnp.float32) * std + mean

    # ------------------------------------------------------------------ #
    # Design -> geometry (abstract; the design scheme lives in subclasses)
    # ------------------------------------------------------------------ #
    def _design_to_geometry(self, design):
        """Map a *physical* design into per-layer ``(layers, angles, widths,
        heights, Bs)`` for the C solver. Defined by the design subclass."""
        raise NotImplementedError("_design_to_geometry is defined by the design subclass")

    # ------------------------------------------------------------------ #
    # Event generation
    # ------------------------------------------------------------------ #
    def generate_events(self, rng, n):
        """Sample ``n`` events (with replacement) from the preloaded sparse pool.

        Returns ``(daughter_data, targets)``: the flat per-particle arrays + CSR
        ``offsets`` the C solver consumes and the 6-vector regression targets.
        Mirrors :meth:`DebugDetector.generate_events` (seed -> events), but events
        are drawn (ragged-gathered) from the preloaded file's flat pool.
        """
        if self._events is None:
            raise RuntimeError("StrawDetector has no event source; pass `data_dir` in the detector config.")
        ev = self._events
        offsets = ev["offsets"]  # (n_events+1,)
        idx = rng.choice(self._n_events, size=int(n), replace=True)

        counts = (offsets[1:] - offsets[:-1])  # particles per event
        sel_counts = counts[idx].astype(np.int64)  # (n,)
        batch_offsets = np.zeros(int(n) + 1, dtype=np.int32)
        batch_offsets[1:] = np.cumsum(sel_counts)
        total = int(batch_offsets[-1])

        # Ragged gather: map each batch slot back to its source flat row.
        src_start = np.repeat(offsets[idx].astype(np.int64), sel_counts)  # (total,)
        dst_start = np.repeat(batch_offsets[:-1].astype(np.int64), sel_counts)  # (total,)
        gather = src_start + (np.arange(total, dtype=np.int64) - dst_start)

        daughter_data = {
            "masses": ev["masses"][gather],
            "charges": ev["charges"][gather],
            "positions": ev["positions"][gather],
            "momenta": ev["momenta"][gather],
            "times": ev["times"][gather],
            "offsets": batch_offsets,
        }
        return daughter_data, ev["targets"][idx]

    def _run_solver(self, daughter_data, design, rng):
        """Run the C straw solver for a physical ``design`` and ``daughter_data``.

        The C solver writes the padded dense ``X (B, M, 5)`` + ``mask (B, M)``
        directly (no sparse intermediate); ``counts`` is the per-event write
        cursor. Returns ``(X, mask, trajectories)``.
        """
        masses = daughter_data["masses"]
        charges = daughter_data["charges"]
        initial_positions = daughter_data["positions"]
        initial_momentum = daughter_data["momenta"]
        initial_times = daughter_data["times"]
        offsets = daughter_data["offsets"]

        layers, angles, widths, heights, Bs = self._design_to_geometry(design)

        n_events = layers.shape[0]

        Bs_arr = Bs.astype(np.float32)
        z0_arr = np.full((n_events,), self.z0, dtype=np.float32)
        B_sigma_arr = np.full((n_events,), self.B_sigma, dtype=np.float32)

        trajectories = np.zeros((n_events, self.max_particles, self.n_t, 3), dtype=np.float32)

        # Dense output: the C solver writes the padded (B, M, 5) hit array + mask
        # directly; ``counts`` is the per-event write cursor (M is the hard cap).
        M = self.max_hits_per_event
        X = np.zeros((n_events, M, 5), dtype=np.float32)
        mask = np.zeros((n_events, M), dtype=np.int32)
        counts = np.zeros((n_events,), dtype=np.int32)

        straw_detector.solve(
            initial_positions,
            initial_momentum,
            masses,
            charges,
            initial_times,
            Bs_arr,
            z0_arr,
            B_sigma_arr,
            self.max_steps,
            self.dt_fixed,
            n_events,
            offsets,
            self.n_layers,
            self.n_straws,
            layers,
            widths,
            heights,
            angles,
            trajectories,
            self.delta_const,
            self.wall_thickness,
            self.delta_Tcut,
            self.max_particles,
            self.lambda_conv_cm,
            self.enable_decay,
            self.noise_rate,
            X,
            mask,
            counts,
            self.n_views_per_station,
            self.n_layers_per_view,
            int(rng.integers(1, 2**32)),  # propagate the caller's seed into the C RNG
            self.max_dt,
            self.max_time,
            self.scatter_xX0,
            self.layer_y_offset,
        )

        return X, mask, trajectories

    def sample_events(self, seed, design):
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

        rng = np.random.default_rng(seed)
        daughter_data, targets = self.generate_events(rng, n_events)
        X, mask, trajectories = self._run_solver(daughter_data, design, rng)

        # ground_truth (daughter conditioning for generators/LFI) is unimplemented
        # for the sparse pool -- to be reworked. The regression/BO path uses only
        # X, mask, targets, so it is unaffected.
        return {
            "X": X,
            "mask": mask,
            "targets": targets,
            "ground_truth": None,
            "trajectories": trajectories,
        }

    def __call__(self, seed, design):
        """Generate events for an un-encoded ``design`` ``(B, design_dim)``.

        Returns ``(ground_truth (B, G), measurements (B, M, 5), mask (B, M),
        target (B, 6))``.
        """
        out = self.sample_events(seed, design)
        return out["ground_truth"], out["X"], out["mask"], out["targets"]

    # ------------------------------------------------------------------ #
    # Design encoding (constrained <-> unconstrained) -- abstract; the design
    # subclass owns the bounds and the encode/decode to N(0,1).
    # ------------------------------------------------------------------ #
    def encode_design(self, design):
        """Physical design -> ``N(0,1)`` space (defined by the design subclass)."""
        raise NotImplementedError("encode_design is defined by the design subclass")

    def decode_design(self, encoded_design):
        """``N(0,1)`` -> physical design (defined by the design subclass)."""
        raise NotImplementedError("decode_design is defined by the design subclass")

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
        """Per-hit design-informed features (all *normalised*, not encoded):

            [time, norm(layer z), norm(straw y), norm(layer angle), norm(B)]

        Each hit's layer z and view angle are gathered from the **decoded**
        design at the hit's own (station, view, layer-in-view) -> global layer
        index; straw y is the hit straw's transverse position. Mirrors
        :meth:`DebugDetector.combine`: a compact fixed width, differentiable
        w.r.t. the encoded design through the decode + gather. The hit ``mask``
        is threaded separately by the caller.
        """
        import jax.numpy as jnp

        X_norm = jnp.asarray(X_norm, dtype=jnp.float32)
        B, M, _ = X_norm.shape

        # Recover the raw integer indices that normalize() standardised.
        means, stds = self._feature_scales()
        raw = X_norm * jnp.asarray(stds) + jnp.asarray(means)  # [station, view, layer_in_view, straw, time]
        per_station = self.n_views_per_station * self.n_layers_per_view
        station = jnp.clip(jnp.round(raw[..., 0]).astype(jnp.int32), 0, self.n_stations - 1)
        view = jnp.clip(jnp.round(raw[..., 1]).astype(jnp.int32), 0, self.n_views_per_station - 1)
        layer_in_view = jnp.clip(jnp.round(raw[..., 2]).astype(jnp.int32), 0, self.n_layers_per_view - 1)
        straw = raw[..., 3]
        time = X_norm[..., 4]  # the (already standardised) measurement feature
        layer = station * per_station + view * self.n_layers_per_view + layer_in_view  # (B, M) global layer idx

        # Decode design and gather each hit's own layer geometry.
        d_enc = jnp.asarray(encoded_design, dtype=jnp.float32)
        if d_enc.ndim == 1:
            d_enc = jnp.broadcast_to(d_enc[None, :], (B, d_enc.shape[0]))
        positions, angles, B_field = self._decode_to_layer_geometry(d_enc)  # (B,n),(B,n),(B,)

        z_hit = jnp.take_along_axis(positions, layer, axis=1)  # (B, M)
        angle_hit = jnp.take_along_axis(angles, layer, axis=1)  # (B, M)
        # Half-pitch stagger: even layer-in-view -> -h, odd -> +h (matches the C solver).
        y_stagger = jnp.where((layer_in_view & 1) == 1, 0.5 * self.layer_y_offset, -0.5 * self.layer_y_offset)
        straw_y = (straw + 0.5) * self.straw_pitch - self.layer_height + y_stagger

        z_mid = 0.5 * (self.layer_bounds[0] + self.layer_bounds[1])
        z_half = max(0.5 * (self.layer_bounds[1] - self.layer_bounds[0]), 1e-6)
        a_mid = 0.5 * (self.angle_bounds[0] + self.angle_bounds[1])
        a_half = max(0.5 * (self.angle_bounds[1] - self.angle_bounds[0]), 1e-6)
        b_mid = 0.5 * self.max_B
        b_half = max(0.5 * self.max_B, 1e-6)

        norm_z = (z_hit - z_mid) / z_half
        norm_y = straw_y / self.layer_height
        norm_angle = (angle_hit - a_mid) / a_half
        norm_B = jnp.broadcast_to(((B_field - b_mid) / b_half)[:, None], (B, M))

        return jnp.stack([time, norm_z, norm_y, norm_angle, norm_B], axis=-1)

    def _decode_to_layer_geometry(self, d_enc):
        """Encoded design ``(B, design_dim)`` -> per-layer ``(positions(B,n),
        angles(B,n), B_field(B))``, used by :meth:`combine`. Defined by the
        design subclass (it owns how the design expands into per-layer geometry)."""
        raise NotImplementedError("_decode_to_layer_geometry is defined by the design subclass")
