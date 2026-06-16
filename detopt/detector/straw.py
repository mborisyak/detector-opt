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
# Per-straw scratch empty-sentinel (must match STRAW_TDC_EMPTY in straw_detector.c):
# an unfired straw holds this; a real TDC is < it, so record_hit is a branchless min.
_STRAW_TDC_EMPTY = 1.0e30
_SOLVER_MAX_DEPTH = 64  # secondary-cascade safety bound (faithful cascade self-terminates)

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
        n_straws_per_layer: int = 316,  # FairShip SST V2023 straws per layer (digi_straw index 1..316); 2 cm pitch -> |y| <= 316 cm
        straw_pitch: float = 2.0,  # cm; centre-to-centre straw spacing (FairShip strawtubes_config.yaml)
        straw_length: float = 400.0,  # cm; FairShip SST aperture width 200 cm (half) -> 400 cm straws (x)
        # Physics parameters
        max_B: float = -0.151,  # T; on-axis peak Bx of FairShip MainSpectrometerField.root (negative polarity)
        z0: float = 8957.0,  # cm; magnet centre (FairShip geometry_config c.z = 89.57 m, map z_local=0)
        B_sigma: float = 283.0,  # cm; proper-Gaussian std-dev fit to MainSpectrometerField on-axis Bx(z) (int Bx dz ~ -1.07 T.m)
        # Design-space bounds (the design itself is supplied per-call, not stored)
        layer_bounds: tuple[float | int, float | int] = (8000.0, 10000.0),
        angles_bounds=None,
        dt=None,  # fixed step (ns); None -> adaptive per-step dt (sagitta-bounded), capped by max_dt
        max_dt=1.0,  # upper clamp (ns) on the adaptive step; bounds the field-free Bx->0 region
        max_time=200.0,  # total integration-time cap (ns) per particle; trapped/curling tracks stop here
        max_particles=5,  # DEBUG-only now: sizes the optional trajectory/tree slots; does NOT gate hits
        max_hits_per_event=384,  # M: NN-facing per-event hit cap; solver emits the M earliest-TDC straws (p99 of MC ~328)
        material="kapton",  # straw-wall material (FairShip default); sets delta_const + lambda_conv
        wall_thickness=0.0036,  # straw wall thickness (cm); FairShip strawtubes_config.yaml
        # Defaults below are tuned so the per-hit process composition approximately
        # matches the FairShip MC truth (data/mc): real-track hits ~83% primary,
        # ~5% pair, ~3% decay + ~17.5% untracked soft secondaries. See physics_match notes.
        delta_Tcut=0.28,  # delta-ray tracking threshold (MeV); tuned (faithful cascade) so delta hits ~ MC's 17.8% soft-secondary band
        lambda_conv_cm=95.0,  # EFFECTIVE pair/conversion length (cm); tuned so pair hits ~ MC 4.2% (the faithful cascade over-produces at the 36.7 cm kapton value)
        enable_decay=True,  # charged pi/K decay-in-flight (Geant4 does this; ~matches MC decay hits)
        noise_rate=0.0,  # mean uncorrelated noise hits/event; 0 = match FairShip (straw digi injects NO electronic noise)
        scatter_xX0=2.5e-4,  # Highland multiple scattering per layer crossing = 2 kapton straw walls / X0 (matches Geant4 angles)
        enable_eloss=True,  # mean ionisation (Bethe-Bloch) energy loss per wire-plane crossing; ranges out soft delta-rays
        eloss_min_ke=0.05,  # stop a track once its kinetic energy drops below this (MeV) -- ranged out
        eloss_scale=0.45,  # tuning multiplier on the ionisation dE/dx; 0.45 lands station-0 delta-ray occupancy on the V2023 MC (ratio ~0.99)
        # Target (6-vec) normalization: decay vertex (cm) + HNL momentum (GeV). These are
        # FIXED constants (computed once from data/mc, 180,672 events) -- baked in so a
        # change of dataset can't silently re-scale targets/predictions and wreck the loss.
        # Pass any as None to recompute it from the loaded `targets` (a convenience).
        decay_mean=(-0.2423, 0.2345, 6205.60),
        decay_sigma=(76.11, 101.06, 1365.96),
        momentum_mean=(-0.0027, 0.0021, 46.686),
        momentum_sigma=(0.5421, 0.6479, 28.203),
        # Daughter-momentum normalization (GeV), for the daughter-tracking target
        # ([vertex, p1, p2]); the vertex reuses decay_* (it IS the HNL decay vertex).
        # ONE constant SHARED by both daughters (pooled over them), so the
        # permutation-invariant daughter loss's p1<->p2 swap is exact. None -> recompute.
        daughter_momentum_mean=(-0.0014, 0.0010, 23.3433),
        daughter_momentum_sigma=(0.4243, 0.4740, 20.5800),
        # HNL conditioning mass normalization [mass GeV]; bolted to 0.5 / 0.5 for now
        # (single-mass sample). None -> recompute from the loaded `conditioning`.
        mass_mean=0.5,
        mass_sigma=0.5,
        data_dir=None,  # ship2numpy event file, preloaded once (None = no event source)
        boundary_z=None,  # crossing-plane z (cm); None -> read from the file
        pool_split=None,  # Sequence|Mapping of normalized fractions -> disjoint event pools (None = one pool)
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
        self._mass_mean_arg = mass_mean
        self._mass_sigma_arg = mass_sigma
        self._daughter_momentum_mean_arg = daughter_momentum_mean
        self._daughter_momentum_sigma_arg = daughter_momentum_sigma

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

        # Mean ionisation energy loss (Bethe-Bloch), applied once per wire-plane crossing
        # to range out soft delta-rays. Per crossing the C solver computes
        #   dE = (eloss_wall_coef + eloss_gas_const * gas_path) * (z^2/beta^2) * Bethe_log
        # with eloss_wall_coef = K(Z/A)rho_wall * 2*t_wall [MeV] (fixed 2-wall path folded
        # in) and eloss_gas_const = K(Z/A)rho_gas [MeV/cm] (gas_path = slab chord, in C).
        self.enable_eloss = int(bool(enable_eloss))
        self.eloss_min_ke = float(eloss_min_ke)
        _K = 0.307075  # MeV mol^-1 cm^2
        _zA_wall, _rho_wall, _ = MATERIALS[material]
        _zA_gas, _rho_gas = 0.489, 1.78e-3  # straw gas ~ Ar/CO2 (80/20) at ~1 atm: Z/A, g/cm^3
        _es = float(eloss_scale)
        self.eloss_wall_coef = _K * _zA_wall * _rho_wall * (2.0 * self.wall_thickness) * _es  # MeV (2 walls)
        self.eloss_gas_const = _K * _zA_gas * _rho_gas * _es  # MeV/cm (gas; path computed per crossing in C)
        self.eloss_I = 79.6e-6  # Kapton mean excitation energy [MeV] (walls dominate the loss)

        self.layer_bounds = layer_bounds

        # Real detector geometry (structural counts + fixed hardware only)
        self.n_stations = int(n_stations)
        self.n_views_per_station = int(n_views_per_station)
        self.n_layers_per_view = int(n_layers_per_view)
        self.n_straws = int(n_straws_per_layer)
        self.straw_pitch = float(straw_pitch)
        self.straw_length = float(straw_length)
        # Half-pitch (= straw radius) stagger between the two layers in a view; derived,
        # not a free parameter: each layer's straw centres sit at +/-layer_y_offset/2.
        self.layer_y_offset = self.straw_pitch / 2.0

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
        self.max_hits_per_event = int(max_hits_per_event)  # M: NN output cap (decoupled from max_particles)

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
            self._events = load_ship2numpy_events(data_dir, boundary_z=boundary_z)
            self._n_events = self._events["n_events"]
            self.boundary_z = self._events["boundary_z"]

        # Disjoint event pools (e.g. train/val): partition the loaded events into index
        # subsets by normalized fraction, after a deterministic shuffle (so pools are random
        # and not biased by file/run order). `pool=` on generate_events/sample_events/__call__
        # selects one. The first key is the default pool.
        self.pool_split = self.resolve_pool_split(pool_split)
        self._pools = {}
        if self._n_events > 0:
            perm = np.random.default_rng(0xC0FFEE).permutation(self._n_events)
            start = 0
            for k, frac in self.pool_split.items():
                end = self._n_events if k == list(self.pool_split)[-1] else start + int(round(frac * self._n_events))
                self._pools[k] = perm[start:end]
                start = end
        self._default_pool = next(iter(self.pool_split))

        # Init-once C objects (constants + structural counts) + the reused per-straw
        # scratch (sparse set; tdc starts at the empty sentinel). Built once, passed to
        # every solve(); the design + outputs vary per call.
        self._sim_params = straw_detector.SimParams(
            self.max_dt,
            self.max_time,
            self.dt_fixed,
            self.max_steps,
            _SOLVER_MAX_DEPTH,
            self.scatter_xX0,
            self.lambda_conv_cm,
            self.noise_rate,
            self.enable_decay,
            self.delta_const,
            self.wall_thickness,
            self.delta_Tcut,
            self.enable_eloss,
            self.eloss_wall_coef,
            self.eloss_gas_const,
            self.eloss_I,
            self.eloss_min_ke,
        )
        self._layout = straw_detector.Layout(
            self.n_views_per_station,
            self.n_layers_per_view,
            self.n_straws,
            self.n_layers,
            self.max_hits_per_event,
            self.layer_y_offset,
            self.layer_width,
            self.layer_height,
            self.z0,
            self.B_sigma,  # fixed geometry/field constants (not optimised)
        )
        # Reused per-straw sparse-set scratch, wrapped in the C Scratch object (tdc starts
        # at the empty sentinel; the numpy arrays are held alive by self._scratch).
        n_cells = self.n_layers * self.n_straws
        self._tdc = np.full(n_cells, _STRAW_TDC_EMPTY, dtype=np.float32)
        self._proc = np.zeros(n_cells, dtype=np.int32)
        self._fired_idx = np.zeros(n_cells, dtype=np.int32)
        self._scratch = straw_detector.Scratch(self._tdc, self._proc, self._fired_idx)
        # The validated, ref-held particle pool (only when an event source is loaded).
        self._input_events = self._make_input_events(self._events) if self._events is not None else None

        # Resolve target-normalization constants (derive any left None from data).
        self._resolve_target_norm()

    @staticmethod
    def _make_input_events(ev):
        """Wrap a pool dict (masses/charges/positions/momenta/times) in the C InputEvents
        object (validates + holds refs once). Used for the loaded pool and by single-event
        callers that build their own per-event arrays."""
        return straw_detector.InputEvents(ev["masses"], ev["charges"], ev["positions"], ev["momenta"], ev["times"])

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
    # max_hits_per_event (``M``, the NN-facing per-event hit cap) is a plain
    # constructor attribute now -- decoupled from max_particles. The solver fills a
    # per-straw scratch and emits the M earliest-TDC straws, so M sizes only the
    # output X/mask, not the simulation.

    def design_shape(self):
        # The base carries no design scheme -- subclasses define the design space.
        raise NotImplementedError("design_shape is defined by the design subclass")

    def target_shape(self):
        # Network regression target, selected by `_targets_field`: the HNL 6-vec
        # [vertex, momentum] or the daughter 9-vec [vertex, p1, p2].
        return (9,) if getattr(self, "_targets_field", "targets") == "daughter_targets" else (6,)

    def event_shape(self):
        # per-hit raw features: [station, view, layer_in_view, straw, time]
        return (self.max_hits_per_event, 5)

    def combined_event_shape(self):
        # combine() -> [TDC, norm_layer_z, wire_y_left, wire_y_right]  (field fixed -> not a feature)
        return (self.max_hits_per_event, 4)

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
        """Set ``decay_mean/sigma`` + ``momentum_mean/sigma`` (+ HNL ``mass_mean/sigma``),
        deriving any constructor arg passed as ``None`` from the loaded data's ``targets``
        (vertex ``[:, :3]``, momentum ``[:, 3:]``). HNL mass stats come from the data's
        ``conditioning`` when present. Sigmas are floored to avoid divide-by-zero."""
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

        # HNL conditioning norm [mass, p(3)]: momentum reuses the target momentum stats
        # above; mass is FIXED (bolted to the constructor default, 0.5/0.5). Pass
        # mass_mean/mass_sigma as None to recompute the mass stats from the data.
        if self._mass_mean_arg is None or self._mass_sigma_arg is None:
            if self._events is None or self._events.get("conditioning") is None:
                raise ValueError("mass normalization is None but no `conditioning` data to derive it from.")
            mass = self._events["conditioning"][:, 0]
            mm = mass.mean() if self._mass_mean_arg is None else self._mass_mean_arg
            ms = mass.std() if self._mass_sigma_arg is None else self._mass_sigma_arg
        else:
            mm, ms = self._mass_mean_arg, self._mass_sigma_arg
        self.mass_mean = np.float32(mm)
        self.mass_sigma = np.float32(max(ms, 1e-3))

        # Daughter-momentum norm (shared by both daughters). None -> pool over both
        # daughters' momenta in the data's `daughter_targets` ([:, 3:6] and [:, 6:9]).
        if self._daughter_momentum_mean_arg is None or self._daughter_momentum_sigma_arg is None:
            if self._events is None or self._events.get("daughter_targets") is None:
                raise ValueError("daughter-momentum normalization is None but no `daughter_targets` data to derive it from.")
            dt = self._events["daughter_targets"]
            p = np.concatenate([dt[:, 3:6], dt[:, 6:9]], axis=0)  # pooled -> symmetric over daughters
            dm_mean = p.mean(0) if self._daughter_momentum_mean_arg is None else self._daughter_momentum_mean_arg
            dm_std = p.std(0) if self._daughter_momentum_sigma_arg is None else self._daughter_momentum_sigma_arg
        else:
            dm_mean, dm_std = self._daughter_momentum_mean_arg, self._daughter_momentum_sigma_arg
        self.daughter_momentum_mean = np.asarray(dm_mean, dtype=np.float32)
        self.daughter_momentum_sigma = np.maximum(np.asarray(dm_std, dtype=np.float32), 1e-3)

    def _target_norm_parts(self):
        """``(mean_parts, std_parts)`` for the network regression target, selected by
        ``_targets_field`` (a subclass sets it to pick which sampled ground truth the net
        predicts); concatenated to standardize. Default ``"targets"`` is the HNL
        ``[vertex(3), momentum(3)]``; ``"daughter_targets"`` is ``[vertex(3), p1(3), p2(3)]`` (the
        vertex reuses ``decay_*``; both daughter momenta share ``daughter_momentum_*``)."""
        if getattr(self, "_targets_field", "targets") == "daughter_targets":
            return (
                (self.decay_mean, self.daughter_momentum_mean, self.daughter_momentum_mean),
                (self.decay_sigma, self.daughter_momentum_sigma, self.daughter_momentum_sigma),
            )
        return (
            (self.decay_mean, self.momentum_mean),
            (self.decay_sigma, self.momentum_sigma),
        )

    def _target_norm_arrays(self):
        import jax.numpy as jnp

        mean_parts, std_parts = self._target_norm_parts()
        mean = jnp.concatenate([jnp.asarray(p) for p in mean_parts])
        std = jnp.concatenate([jnp.asarray(p) for p in std_parts])
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
    # HNL conditioning [mass, p(3)] (for the LFI discriminator, etc.)
    # ------------------------------------------------------------------ #
    def conditioning_shape(self):
        return (4,)  # [HNL mass, px, py, pz]

    def conditioning_dim(self):
        return 4

    def _conditioning_norm_arrays(self):
        import jax.numpy as jnp

        mean = jnp.concatenate([jnp.asarray(self.mass_mean)[None], jnp.asarray(self.momentum_mean)])
        std = jnp.concatenate([jnp.asarray(self.mass_sigma)[None], jnp.asarray(self.momentum_sigma)])
        return mean, std

    def normalize_conditioning(self, conditioning):
        """Physical HNL ``[mass, p(3)]`` -> standardised (mass by mass_*, p by momentum_*)."""
        import jax.numpy as jnp

        mean, std = self._conditioning_norm_arrays()
        return (jnp.asarray(conditioning, dtype=jnp.float32) - mean) / std

    def loss_label(self):
        return f"MSE (normalized {self.target_shape()[0]}-vec target)"

    def metric_labels(self):
        """Keys of the metric() dict, in display order."""
        return ("loss", "vertex_x", "vertex_y", "vertex_z", "p_x", "p_y", "p_z")

    def loss(self, predicted, target):
        """Per-sample ``(...,)`` mean-squared error between predictions and the ALREADY-NORMALIZED
        6-vec target ``[vertex(3), momentum(3)]`` (the caller standardises via
        :meth:`normalize_target`; predictions are in that same space). Broadcasts over any leading
        axes (e.g. ``(members, B)``)."""
        import jax.numpy as jnp

        predicted = jnp.asarray(predicted, jnp.float32)
        target = jnp.asarray(target, jnp.float32)
        return jnp.mean(jnp.square(predicted - target), axis=-1)

    def metric(self, predicted, target):
        """Per-sample diagnostics dict on the normalized target: overall ``loss`` plus per-component
        squared error -- ``vertex_{x,y,z}`` and the momentum ``p_{x,y,z}``."""
        import jax.numpy as jnp

        predicted = jnp.asarray(predicted, jnp.float32)
        target = jnp.asarray(target, jnp.float32)
        se = jnp.square(predicted - target)
        return {
            "loss": self.loss(predicted, target),
            "vertex_x": se[..., 0],
            "vertex_y": se[..., 1],
            "vertex_z": se[..., 2],
            "p_x": se[..., 3],
            "p_y": se[..., 4],
            "p_z": se[..., 5],
        }

    # ------------------------------------------------------------------ #
    # Design -> geometry (abstract; the design scheme lives in subclasses)
    # ------------------------------------------------------------------ #
    def b_bounds(self):
        """Ordered ``(low, high)`` encode/decode bound on the field design dof.

        ``max_B`` carries the magnet polarity: ``(0, max_B)`` for a positive
        field, ``(max_B, 0)`` for a negative one (e.g. MainSpectrometerField,
        whose on-axis Bx is negative). Keeps ``low < high`` for either sign.
        """
        return (0.0, self.max_B) if self.max_B > 0 else (self.max_B, 0.0)

    def _design_to_geometry(self, design):
        """Map a *physical* design into per-layer ``(layers, angles, widths,
        heights, Bs)`` for the C solver. Defined by the design subclass."""
        raise NotImplementedError("_design_to_geometry is defined by the design subclass")

    # ------------------------------------------------------------------ #
    # Event generation
    # ------------------------------------------------------------------ #
    def generate_events(self, rng, n, pool=None):
        """Sample ``n`` events (with replacement) from the requested disjoint event pool
        (``pool`` int|str, default = first pool key).

        Returns ``(boundaries, targets, conditioning)``: a ``(n, 2)`` int32 array of
        ``[start, end)`` spans into the shared, uncopied pool (``self._input_events``) --
        no ragged gather -- the 6-vector regression targets, and the ``(n, 4)`` HNL
        conditioning ``[mass, p(3)]`` for the sampled events.
        """
        if self._events is None:
            raise RuntimeError("StrawDetector has no event source; pass `data_dir` in the detector config.")
        ev = self._events
        offsets = ev["offsets"]  # (n_events+1,)
        pool_idx = self._pools[self._default_pool if pool is None else pool]  # event indices in this pool
        idx = pool_idx[rng.integers(0, len(pool_idx), size=int(n))]  # with replacement, within the pool
        boundaries = np.stack([offsets[idx], offsets[idx + 1]], axis=1).astype(np.int32)
        conditioning = (
            ev["conditioning"][idx] if "conditioning" in ev else np.zeros((len(idx), self.conditioning_dim()), dtype=np.float32)
        )
        # `_targets_field` lets a subclass pick a different regression target (e.g. the
        # tracking detector uses "daughter_targets"); defaults to the HNL 6-vec "targets".
        targets = ev[getattr(self, "_targets_field", "targets")][idx]
        return boundaries, targets, conditioning

    def _run_solver(self, boundaries, design, rng, input_events=None, trajectories=None, process_ids=None, tree=None):
        """Run the C straw solver for ``design`` over the events selected by
        ``boundaries`` ((n, 2) ``[start, end)`` spans) into the shared particle pool
        ``input_events`` (the cached ``self._input_events`` if None). The solver writes
        the dense ``X (n, M, 5)`` + ``mask (n, M)`` directly. Returns ``(X, mask,
        trajectories)``.

        The optional debug buffers (``trajectories (n, mp, n_t, 3)``, ``process_ids
        (n, M)``, ``tree`` dict) are caller-allocated numpy arrays, filled in place via
        a ``DebugBuffers`` object; pass ``None`` (default) to skip -- they do not affect
        ``X``/``mask``.
        """
        ie = input_events if input_events is not None else self._input_events
        if ie is None:
            raise RuntimeError("StrawDetector has no event source; pass input_events or `data_dir`.")
        boundaries = np.ascontiguousarray(boundaries, dtype=np.int32)
        layers, angles, Bs = self._design_to_geometry(design)  # widths/heights/z0/B_sigma are fixed (Layout)
        n_events = boundaries.shape[0]

        Bs_arr = Bs.astype(np.float32)
        # One independent uint32 seed per event -> reproducible + parallel-safe.
        seeds = rng.integers(1, 2**32, size=n_events, dtype=np.uint32)

        M = self.max_hits_per_event
        X = np.zeros((n_events, M, 5), dtype=np.float32)
        mask = np.zeros((n_events, M), dtype=np.int32)

        debug = None
        if trajectories is not None or process_ids is not None or tree is not None:
            debug = straw_detector.DebugBuffers(
                trajectories,
                process_ids,
                None if tree is None else tree["int"],
                None if tree is None else tree["float"],
                None if tree is None else tree["event"],
                None if tree is None else tree["count"],
            )

        straw_detector.solve(
            self._sim_params,
            self._layout,
            ie,
            self._scratch,
            seeds,
            boundaries,
            layers,
            angles,
            Bs_arr,
            X,
            mask,
            debug,
        )
        return X, mask, trajectories

    def simulate_debug(self, daughter_data, design, rng):
        """Run the solver with all debug outputs on (for sim-vs-MC matching).

        On top of the production ``X / mask / trajectories`` this also returns the
        per-hit ``process_ids`` (the TMCProcess code of the particle that caused
        each hit, parallel to ``mask``) and a flat/sparse MC-particle ``tree`` --
        one row per tracked particle (primaries + secondaries) across the whole
        batch, located by ``event_index`` (CSR-style). Mirrors the FairShip
        ``mc_info`` / ``mc_particles`` layout so the two are directly comparable.

        Returns a dict with: ``X, mask, trajectories, process_ids`` and the tree
        arrays ``pdg, process_id, parent_id, n_hits`` (int) + ``momentum (P, 3),
        position (P, 3), t0 (P,)`` (float) + ``event_index (P,)``.
        """
        design = np.asarray(design, dtype=np.float32)
        design = design[None, :] if design.ndim == 1 else design
        n_events = design.shape[0]

        # Build a transient InputEvents from the given per-event pool; boundaries select
        # the events (default: a single event spanning the whole pool).
        ie = self._make_input_events(daughter_data)
        boundaries = daughter_data.get("boundaries")
        if boundaries is None:
            n_part = len(np.asarray(daughter_data["masses"]))
            boundaries = np.tile(np.array([[0, n_part]], dtype=np.int32), (n_events, 1))

        trajectories = np.zeros((n_events, self.max_particles, self.n_t, 3), dtype=np.float32)
        process_ids = np.zeros((n_events, self.max_hits_per_event), dtype=np.int32)
        # Debug MC tree, flat across the batch; cap = n_events * max_particles (max_particles
        # is the DEBUG slot count -- make it large enough for the faithful cascade; the C
        # side raises on overflow).
        cap = int(n_events) * int(self.max_particles)
        tree = {
            "int": np.zeros((cap, 4), dtype=np.int32),  # [pdg, process_id, parent_id, n_hits]
            "float": np.zeros((cap, 7), dtype=np.float32),  # [px, py, pz, x, y, z, t0]
            "event": np.zeros((cap,), dtype=np.int32),
            "count": np.zeros((1,), dtype=np.int32),  # shared write cursor / final count
        }
        X, mask, trajectories = self._run_solver(
            boundaries,
            design,
            rng,
            input_events=ie,
            trajectories=trajectories,
            process_ids=process_ids,
            tree=tree,
        )

        p = int(tree["count"][0])  # number of particles actually recorded
        ti, tf = tree["int"][:p], tree["float"][:p]
        return {
            "X": X,
            "mask": mask,
            "trajectories": trajectories,
            "process_ids": process_ids,
            "pdg": ti[:, 0],
            "process_id": ti[:, 1],
            "parent_id": ti[:, 2],
            "n_hits": ti[:, 3],
            "momentum": tf[:, 0:3],
            "position": tf[:, 3:6],
            "t0": tf[:, 6],
            "event_index": tree["event"][:p],
        }

    def sample_events(self, seed, design, pool=None):
        """Generate events and return a rich dict.

        ``design`` is the *physical* (un-encoded) design ``(B, design_dim)`` (or
        ``(design_dim,)`` for a single event). Batch size ``B`` is inferred. ``pool``
        (int|str) selects the disjoint event pool to sample from (default = first key).
        Used by visualisation / likelihood-free / generative scripts that need
        the daughter ground truth or trajectories.
        """
        if isinstance(design, dict):
            design = self.flatten_design(design)  # named physical design -> flat array
        design = np.asarray(design, dtype=np.float32)
        if design.ndim == 1:
            design = design[None, :]
        n_events = design.shape[0]

        rng = np.random.default_rng(seed)
        boundaries, targets, conditioning = self.generate_events(rng, n_events, pool=pool)
        X, mask, _ = self._run_solver(boundaries, design, rng)  # operational: no debug buffers

        # ground_truth (daughter conditioning for generators/LFI) is unimplemented
        # for the sparse pool -- to be reworked. The regression/BO path uses only
        # X, mask, targets, so it is unaffected. trajectories are a debug-only output
        # (use simulate_debug / pass a trajectory buffer to _run_solver) -> None here.
        return {
            "X": X,
            "mask": mask,
            "targets": targets,
            "conditioning": conditioning,  # (B, 4) HNL [mass, p(3)]
            "ground_truth": None,
            "trajectories": None,
        }

    def __call__(self, seed, design, pool=None):
        """Generate events for an un-encoded ``design`` ``(B, design_dim)``.

        Returns ``(ground_truth (B, G), measurements (B, M, 5), mask (B, M),
        target (B, 6))``. ``pool`` selects the disjoint event pool (default = first key).
        """
        out = self.sample_events(seed, design, pool=pool)
        return out["ground_truth"], out["X"], out["mask"], out["targets"]

    # Design encoding: the base Detector wraps the subclass `_encode_flat`/`_decode_flat`
    # (flat physical <-> encoded) with the dict<->flat conversion; the design subclass owns
    # the bounds, `design_spec`, `design_bounds`, and the flat encode/decode to N(0,1).

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

            [TDC, norm(layer z), wire_y_left, wire_y_right]

        Each hit's layer z and stereo angle are gathered from the **decoded** design at the
        hit's own (station, view, layer-in-view) -> global layer index. Rather than handing
        the network ``(straw_y, angle)`` separately, we encode the (tilted) sense wire as its
        two y-endpoints at the FIXED x-ends of the parallelogram (``x = +/- layer_width``;
        sheared geometry ``Y = y - x*tan(angle)`` is constant along the wire). These two y's
        give the network the wire as a 3D line segment -- the natural input for stereo
        triangulation -> track fit -> curvature -> momentum -- and are differentiable w.r.t.
        the design through the decode+gather. The hit ``mask`` is threaded separately.
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
        tdc = X_norm[..., 4]  # the (already standardised) drift/propagation TDC measurement
        layer = station * per_station + view * self.n_layers_per_view + layer_in_view  # (B, M) global layer idx

        # Decode design and gather each hit's own layer geometry.
        d_enc = jnp.asarray(encoded_design, dtype=jnp.float32)
        if d_enc.ndim == 1:
            d_enc = jnp.broadcast_to(d_enc[None, :], (B, d_enc.shape[0]))
        positions, angles, _B = self._decode_to_layer_geometry(d_enc)  # (B,n),(B,n),(B,); field unused (fixed)

        z_hit = jnp.take_along_axis(positions, layer, axis=1)  # (B, M)
        angle_hit = jnp.take_along_axis(angles, layer, axis=1)  # (B, M)
        # Half-pitch stagger: even layer-in-view -> -h, odd -> +h (matches the C solver).
        y_stagger = jnp.where((layer_in_view & 1) == 1, 0.5 * self.layer_y_offset, -0.5 * self.layer_y_offset)
        straw_y = (straw + 0.5) * self.straw_pitch - self.layer_height + y_stagger  # wire centre y (at x=0)

        z_mid = 0.5 * (self.layer_bounds[0] + self.layer_bounds[1])
        z_half = max(0.5 * (self.layer_bounds[1] - self.layer_bounds[0]), 1e-6)
        norm_z = (z_hit - z_mid) / z_half

        # Wire y at the fixed x-ends (+/- layer_width); the x's are constant so omitted. Y-scale
        # bounds |y| over all straws and the steepest stereo angle so the features stay ~[-1,1].
        dy = self.layer_width * jnp.tan(angle_hit)  # (B, M) y-offset from x=0 to x=+width
        a_max = max(abs(self.angle_bounds[0]), abs(self.angle_bounds[1]))
        y_scale = max(self.layer_height + self.layer_width * float(np.tan(a_max)), 1e-6)
        wire_y_left = (straw_y - dy) / y_scale
        wire_y_right = (straw_y + dy) / y_scale

        return jnp.stack([tdc, norm_z, wire_y_left, wire_y_right], axis=-1)

    def _decode_to_layer_geometry(self, d_enc):
        """Encoded design ``(B, design_dim)`` -> per-layer ``(positions(B,n),
        angles(B,n), B_field(B))``, used by :meth:`combine`. Defined by the
        design subclass (it owns how the design expands into per-layer geometry)."""
        raise NotImplementedError("_decode_to_layer_geometry is defined by the design subclass")
