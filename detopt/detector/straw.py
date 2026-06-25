import os
import warnings
from typing import NamedTuple

import jax
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
from ..utils import tensor
from ..data import load_ship2numpy_events
from . import straw_detector
from .common import Detector
from .propagation import RealisticEngine, SimplifiedEngine
from ..utils.det_random import det_uniforms, box_muller

__all__ = ["StrawDetector", "StrawEvent", "HNLTarget", "DaughterTarget", "StrawGroundTruth", "Pool",
           "MATERIALS", "material_constants", "four_feature_combine", "four_feature_shape"]


class Pool(NamedTuple):
    """A particle pool the detector hands to a :class:`PropagationEngine` -- just the numpy arrays. The
    realistic engine packs them into the C ``InputEvents`` inside its ``solve`` (a few cheap header checks
    in C, no caching); the analytic engine reads ``positions``/``momenta``/``charges`` directly."""
    masses: np.ndarray
    charges: np.ndarray
    positions: np.ndarray
    momenta: np.ndarray
    times: np.ndarray


class StrawEvent(NamedTuple):
    """Raw per-hit straw event (leaves carry a leading batch axis ``(B, M)`` once sampled).

    The hit address is honest integers; only the TDC is a continuous measurement -- so no
    rounding is needed to recover indices (unlike the old all-float ``(M, 5)`` tensor)."""

    station: jax.Array  # int32
    view: jax.Array  # int32 (view-within-station)
    layer: jax.Array  # int32 (layer-within-view)
    straw: jax.Array  # int32
    tdc: jax.Array  # float32 (drift/propagation TDC, ns)
    # Optional per-hit truth measurements for the tracking fits, produced only under the
    # __call__ flags (None otherwise): x, y = the exact (x, y) crossing of the layer plane
    # (hits_xy); drift_r = perpendicular crossing->wire distance, re-derived host-side from
    # (x, y) (drift_r). All float32, parallel to the address leaves.
    x: jax.Array | None = None
    y: jax.Array | None = None
    drift_r: jax.Array | None = None


class HNLTarget(NamedTuple):
    """HNL regression target: decay vertex (cm) + HNL momentum (GeV). (Superseded by DaughterTarget as the
    unified regression target; kept for the conditioning/ground-truth packing + back-compat.)"""

    vertex: jax.Array  # (..., 3)
    momentum: jax.Array  # (..., 3)


class DaughterTarget(NamedTuple):
    """The UNIFIED regression target for every detector: shared decay vertex (cm) + the two daughters'
    momenta (GeV). The HNL momentum is the sum p1 + p2 (a derived prediction)."""

    vertex: jax.Array  # (..., 3)
    p1: jax.Array  # (..., 3)
    p2: jax.Array  # (..., 3)


class StrawGroundTruth(NamedTuple):
    """Generator ground truth == discriminator conditioning: HNL mass, momentum, decay vertex."""

    mass: jax.Array  # (..., 1) GeV
    momentum: jax.Array  # (..., 3) GeV
    vertex: jax.Array  # (..., 3) cm


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

# HNL -> 2-body decay daughters (the no-data analytic input-event source; generic, not geometry).
_MUON_MASS = 0.105658  # GeV
_PION_MASS = 0.139570  # GeV


def _boost(p_rest, E_rest, P_hnl, E_hnl, M):
    """Boost rest-frame 3-momenta (`p_rest`, energy `E_rest`) into the lab where the HNL has momentum
    `P_hnl` (n,3) and energy `E_hnl` (n,). Batched; used by the no-data analytic event source."""
    beta = P_hnl / E_hnl[:, None]
    b2 = np.sum(beta**2, axis=1)
    gamma = E_hnl / M
    bp = np.sum(beta * p_rest, axis=1)
    coeff = np.where(b2 > 1e-12, (gamma - 1.0) * bp / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return (p_rest + beta * (coeff + gamma * E_rest)[:, None]).astype(np.float32)


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
    # The loaded ground-truth field used as the (unified) regression target: the daughter 9-vec
    # [vertex, p1, p2]. One objective for every straw detector -- never overridden.
    _targets_field = "daughter_targets"

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
        data_dir=None,  # ship2numpy event file, preloaded once (None -> the no-data analytic source)
        boundary_z=None,  # crossing-plane z (cm); None -> read from the file
        engine="realistic",  # propagation engine: 'realistic' (C ODE solver) | 'simplified' (analytic)
        # No-data analytic INPUT-EVENT source (data_dir=None): HNL momentum/vertex Gaussians -> 2-body
        # decay -> 2 daughters. Generic HNL physics (NOT a geometry concern) -- the base owns it. The
        # `hnl_` prefix keeps these GENERATION distributions distinct from the target-NORMALIZATION
        # `momentum_mean`/`decay_mean`/... above.
        hnl_momentum_mean: tuple = (0.0, 0.0, 46.7),
        hnl_momentum_sigma: tuple = (0.54, 0.65, 28.2),
        hnl_vertex_mean: tuple = (0.0, 0.0, 6206.0),
        hnl_vertex_sigma: tuple = (76.0, 101.0, 1366.0),
        hnl_mass: float = 1.0,
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

        # Event source: preload the file once into immutable per-event arrays, addressed by
        # event_index. No train/val split, no rng, no mutable loader state -- that belongs to the
        # scripts, not the detector (the analytic subclass generates events from the index instead).
        self._events = None
        self._n_events = 0
        self.boundary_z = float(boundary_z) if boundary_z is not None else None
        if data_dir is not None and engine != "relay":
            self._events = load_ship2numpy_events(data_dir, boundary_z=boundary_z)
            self._n_events = self._events["n_events"]
            self.boundary_z = self._events["boundary_z"]

        # Loaded particle pool (just the numpy arrays); None for a no-data (analytic) detector.
        ev = self._events
        self._pool = (Pool(ev["masses"], ev["charges"], ev["positions"], ev["momenta"], ev["times"])
                      if ev is not None else None)

        # The propagation engine -- identical across all straw detectors, so it is built here in the base.
        # The REALISTIC engine owns the C SimParams/Layout/Scratch; the simplified one is pure numpy. The
        # detector holds none of that C state.
        geometry = dict(
            n_views_per_station=self.n_views_per_station, n_layers_per_view=self.n_layers_per_view,
            n_straws=self.n_straws, n_layers=self.n_layers, max_hits_per_event=self.max_hits_per_event,
            layer_width=self.layer_width, layer_height=self.layer_height, straw_pitch=self.straw_pitch,
            layer_y_offset=self.layer_y_offset, B_sigma=self.B_sigma, z0=self.z0,
        )
        if engine == "realistic":
            sim_params = straw_detector.SimParams(
                self.max_dt, self.max_time, self.dt_fixed, self.max_steps, _SOLVER_MAX_DEPTH,
                self.scatter_xX0, self.lambda_conv_cm, self.noise_rate, self.enable_decay, self.delta_const,
                self.wall_thickness, self.delta_Tcut, self.enable_eloss, self.eloss_wall_coef,
                self.eloss_gas_const, self.eloss_I, self.eloss_min_ke,
            )
            layout = straw_detector.Layout(
                self.n_views_per_station, self.n_layers_per_view, self.n_straws, self.n_layers,
                self.max_hits_per_event, self.layer_y_offset, self.layer_width, self.layer_height,
                self.z0, self.B_sigma,
            )
            self.engine = RealisticEngine(sim_params, layout, **geometry)
        elif engine == "simplified":
            self.engine = SimplifiedEngine(**geometry)
        elif engine == "relay":
            self.engine = None  # special ENGINE-LESS source: replay packed FairShip data (see _load_relay)
        else:
            raise ValueError(f"unknown engine {engine!r} (expected 'realistic', 'simplified' or 'relay')")

        # No-data analytic INPUT-EVENT source: with no loaded pool, _events_at SYNTHESIZES events from the
        # HNL distributions (generic physics; works through EITHER engine -- source x engine independent).
        self.hnl_momentum_mean = np.asarray(hnl_momentum_mean, dtype=np.float32)
        self.hnl_momentum_sigma = np.asarray(hnl_momentum_sigma, dtype=np.float32)
        self.hnl_vertex_mean = np.asarray(hnl_vertex_mean, dtype=np.float32)
        self.hnl_vertex_sigma = np.asarray(hnl_vertex_sigma, dtype=np.float32)
        self.hnl_mass = float(hnl_mass)
        M, m1, m2 = self.hnl_mass, _MUON_MASS, _PION_MASS  # 2-body decay kinematics, fixed by the masses
        self._decay_E1 = (M**2 + m1**2 - m2**2) / (2.0 * M)        # daughter-1 rest-frame energy
        self._decay_E2 = (M**2 + m2**2 - m1**2) / (2.0 * M)        # daughter-2 rest-frame energy
        self._decay_pstar = float(np.sqrt(max(self._decay_E1**2 - m1**2, 0.0)))  # rest-frame momentum
        self._analytic = self._events is None

        # Relay source: replay packed REAL FairShip hits at the fixed design (engine='relay'); no solver.
        self._relay = engine == "relay"
        if self._relay:
            self._load_relay(data_dir)

        # Resolve target-normalization constants (derive any left None from data; warns when it does).
        self._resolve_target_norm()

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

    def event_spec(self):
        # Raw per-hit features: int32 address [station, view, layer-in-view, straw] + float32 TDC.
        M = self.max_hits_per_event
        i = jax.ShapeDtypeStruct((M,), np.int32)
        return StrawEvent(station=i, view=i, layer=i, straw=i, tdc=jax.ShapeDtypeStruct((M,), np.float32))

    def target_spec(self):
        # The unified daughter 9-vec target: shared decay vertex + the two daughters' momenta.
        f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
        return DaughterTarget(vertex=f(3), p1=f(3), p2=f(3))

    def ground_truth_spec(self):
        # Ground truth == conditioning: HNL mass + momentum + decay vertex (retires the old
        # generator charges/positions/momenta truth, which was never implemented for the sparse pool).
        f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
        return StrawGroundTruth(mass=f(1), momentum=f(3), vertex=f(3))

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
            derived = [n for n, a in zip(("decay_mean", "decay_sigma", "momentum_mean", "momentum_sigma"), args)
                       if a is None]
            warnings.warn(f"StrawDetector: deriving target-norm {derived} from the loaded data -- a data swap "
                          "will silently re-scale the loss; pass them explicitly to pin the scale.", stacklevel=2)
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
            warnings.warn("StrawDetector: deriving HNL mass-norm from the loaded data's conditioning.", stacklevel=2)
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
            warnings.warn("StrawDetector: deriving daughter-momentum norm from the loaded data.", stacklevel=2)
            dt = self._events["daughter_targets"]
            p = np.concatenate([dt[:, 3:6], dt[:, 6:9]], axis=0)  # pooled -> symmetric over daughters
            dm_mean = p.mean(0) if self._daughter_momentum_mean_arg is None else self._daughter_momentum_mean_arg
            dm_std = p.std(0) if self._daughter_momentum_sigma_arg is None else self._daughter_momentum_sigma_arg
        else:
            dm_mean, dm_std = self._daughter_momentum_mean_arg, self._daughter_momentum_sigma_arg
        self.daughter_momentum_mean = np.asarray(dm_mean, dtype=np.float32)
        self.daughter_momentum_sigma = np.maximum(np.asarray(dm_std, dtype=np.float32), 1e-3)

    def _target_norm_parts(self):
        """``(mean_parts, std_parts)`` for the unified daughter target ``[vertex(3), p1(3), p2(3)]``,
        concatenated to standardize: the vertex by ``decay_*``, both daughter momenta by
        ``daughter_momentum_*``."""
        return (
            (self.decay_mean, self.daughter_momentum_mean, self.daughter_momentum_mean),
            (self.decay_sigma, self.daughter_momentum_sigma, self.daughter_momentum_sigma),
        )

    def _target_norm_arrays(self):
        import jax.numpy as jnp

        mean_parts, std_parts = self._target_norm_parts()
        mean = jnp.concatenate([jnp.asarray(p) for p in mean_parts])
        std = jnp.concatenate([jnp.asarray(p) for p in std_parts])
        return mean, std

    def normalize_target(self, target):
        """Physical ``Target`` -> standardised flat array; vertex by decay_*, momentum by momentum_*
        (daughter target: both momenta by daughter_momentum_*). Accepts a flat array too."""
        mean, std = self._target_norm_arrays()
        flat, _ = tensor.flatten(target)  # Target record (or flat (B, T) array) -> (B, T)
        return (flat - mean) / std

    def denormalize_predictions(self, normalised):
        """Inverse of :meth:`normalize_target`: flat normalised array -> physical ``Target`` namedtuple."""
        import jax.numpy as jnp

        mean, std = self._target_norm_arrays()
        phys = jnp.asarray(normalised, dtype=jnp.float32) * std + mean
        return tensor.unflatten(tensor.structure(self.target_spec()), phys)

    def prediction_errors(self, predicted_norm, target_norm):
        """Signed real-unit residuals ``predicted - true``, for error histograms. The two daughters are
        PERMUTATION-INVARIANT (matched by the loss-minimizing assignment, then POOLED into ``p_{x,y,z}``,
        ``(2N, 3)`` in GeV); the HNL momentum ``p_hnl_{x,y,z}`` is the daughter-SUM residual (``N``, GeV);
        the vertex is cm (``N``). Inputs are NORMALIZED ``(N, 9)``."""
        import numpy as np

        pred = self.denormalize_predictions(predicted_norm)  # DaughterTarget
        true = self.denormalize_predictions(target_norm)
        vertex = np.asarray(pred.vertex) - np.asarray(true.vertex)
        p1p, p2p = np.asarray(pred.p1), np.asarray(pred.p2)
        p1t, p2t = np.asarray(true.p1), np.asarray(true.p2)
        l_direct = np.sum((p1p - p1t) ** 2 + (p2p - p2t) ** 2, axis=-1)
        l_swap = np.sum((p2p - p1t) ** 2 + (p1p - p2t) ** 2, axis=-1)
        swap = (l_swap < l_direct)[:, None]
        err1 = np.where(swap, p2p - p1t, p1p - p1t)  # residual against true daughter 1
        err2 = np.where(swap, p1p - p2t, p2p - p2t)  # residual against true daughter 2
        mom = np.concatenate([err1, err2], axis=0)   # pooled over both daughters -> (2N, 3)
        hnl = (p1p + p2p) - (p1t + p2t)              # HNL momentum (daughter sum) residual -> (N, 3)
        return {
            "vertex_x": (vertex[:, 0], "cm"),
            "vertex_y": (vertex[:, 1], "cm"),
            "vertex_z": (vertex[:, 2], "cm"),
            "p_x": (mom[:, 0], "GeV"),
            "p_y": (mom[:, 1], "GeV"),
            "p_z": (mom[:, 2], "GeV"),
            "p_hnl_x": (hnl[:, 0], "GeV"),
            "p_hnl_y": (hnl[:, 1], "GeV"),
            "p_hnl_z": (hnl[:, 2], "GeV"),
        }

    # ------------------------------------------------------------------ #
    # Ground truth == conditioning [HNL mass, p(3), decay vertex(3)] (for the LFI discriminator).
    # ------------------------------------------------------------------ #
    def normalize_ground_truth(self, ground_truth):
        """Physical ``StrawGroundTruth`` (mass, momentum, decay vertex) -> standardised flat
        ``(..., 7)``: mass by ``mass_*``, momentum by ``momentum_*``, vertex by ``decay_*``.
        Field order is [mass, momentum, vertex] (matches ``StrawGroundTruth``). Accepts a flat array too."""
        import jax.numpy as jnp

        mean = jnp.concatenate(
            [jnp.asarray(self.mass_mean)[None], jnp.asarray(self.momentum_mean), jnp.asarray(self.decay_mean)]
        )
        std = jnp.concatenate(
            [jnp.asarray(self.mass_sigma)[None], jnp.asarray(self.momentum_sigma), jnp.asarray(self.decay_sigma)]
        )
        flat, _ = tensor.flatten(ground_truth)  # StrawGroundTruth record (or flat (B, 7) array) -> (B, 7)
        return (flat - mean) / std

    def loss_label(self):
        return f"MSE (normalized {self.target_dim()}-vec target)"

    def metric_labels(self):
        """Keys of the metric() dict, in display order: overall loss, vertex, the shared daughter
        momentum ``p_{x,y,z}``, and the HNL momentum ``p_hnl_{x,y,z}`` (the daughter sum)."""
        return ("loss", "vertex_x", "vertex_y", "vertex_z",
                "p_x", "p_y", "p_z", "p_hnl_x", "p_hnl_y", "p_hnl_z")

    def loss(self, predicted, target):
        """The UNIFIED per-sample objective on the NORMALIZED daughter 9-vec ``[vertex, p1, p2]``: vertex
        MSE + PERMUTATION-INVARIANT daughter-momenta MSE (the cheaper of the two daughter assignments) +
        HNL-momentum MSE on the daughter sum, scaled by 0.5 (i.e. the MEAN ``0.5(p1+p2)``, so the sum sits
        on the daughter normalization scale before its weight). Weighted 0.5 / 0.25 / 0.25 (sums to 1,
        ~6-vec scale). The caller standardises via :meth:`normalize_target`; predictions live in that space.
        Broadcasts over any leading axes (e.g. ``(members, B)``)."""
        import jax.numpy as jnp

        vp, vt = predicted[..., :3], target[..., :3]
        p1p, p2p = predicted[..., 3:6], predicted[..., 6:9]
        p1t, p2t = target[..., 3:6], target[..., 6:9]
        loss_vertex = 0.5 * jnp.mean(jnp.square(vp - vt), axis=-1)
        loss_daughters = 0.25 * jnp.minimum(
            jnp.mean(jnp.square(p1p - p1t) + jnp.square(p2p - p2t), axis=-1),
            jnp.mean(jnp.square(p2p - p1t) + jnp.square(p1p - p2t), axis=-1),
        )
        loss_hnl = 0.25 * jnp.mean(jnp.square(0.5 * ((p1p + p2p) - (p1t + p2t))), axis=-1)
        return loss_vertex + loss_daughters + loss_hnl

    def metric(self, predicted, target):
        """Per-sample diagnostics on the normalized daughter target: overall ``loss`` + per-component
        squared error -- ``vertex_{x,y,z}``, the SHARED daughter momentum ``p_{x,y,z}`` (summed over the
        two daughters under the best assignment), and the HNL momentum ``p_hnl_{x,y,z}`` (daughter sum)."""
        import jax.numpy as jnp

        vp, vt = predicted[..., :3], target[..., :3]
        p1p, p2p = predicted[..., 3:6], predicted[..., 6:9]
        p1t, p2t = target[..., 3:6], target[..., 6:9]
        vertex = jnp.square(vp - vt)
        l1 = jnp.square(p1p - p1t) + jnp.square(p2p - p2t)
        l2 = jnp.square(p2p - p1t) + jnp.square(p1p - p2t)
        swap = jnp.sum(l1, axis=-1) > jnp.sum(l2, axis=-1)
        momenta = 0.5 * (
          (1 - swap)[..., None] * l1 + swap[..., None] * l2
        )
        hnl = 0.5 * jnp.square((p1p + p2p) - (p1t + p2t))
        return {
            "loss": self.loss(predicted, target),
            "vertex_x": vertex[..., 0], "vertex_y": vertex[..., 1], "vertex_z": vertex[..., 2],
            "p_x": momenta[..., 0], "p_y": momenta[..., 1], "p_z": momenta[..., 2],
            "p_hnl_x": hnl[..., 0], "p_hnl_y": hnl[..., 1], "p_hnl_z": hnl[..., 2],
        }

    def metric_real_rmse(self, metric_means, metric_errors=None):
        """Sample-averaged normalized per-component metric (from :meth:`metric`) -> real-unit RMSE.
        ``vertex_{x,y,z}`` are single normalized squared errors -> RMSE = ``sqrt(mse) * decay_sigma`` (cm).
        ``p_{x,y,z}`` are SUMMED over the two daughters -> per-daughter RMSE = ``sqrt(mse/2) *
        daughter_momentum_sigma`` (GeV). ``p_hnl_{x,y,z}`` (the daughter sum, one quantity) ->
        ``sqrt(mse) * daughter_momentum_sigma`` (GeV). ``loss`` has no single unit and is omitted. If
        ``metric_errors`` (the SEM on each mean MSE) is given, propagate to the RMSE error."""
        import numpy as np

        ds = np.asarray(self.decay_sigma, np.float64)
        dps = np.asarray(self.daughter_momentum_sigma, np.float64)
        spec = {  # key: (sigma, n_daughters_summed, unit)
            "vertex_x": (ds[0], 1, "cm"), "vertex_y": (ds[1], 1, "cm"), "vertex_z": (ds[2], 1, "cm"),
            "p_x": (dps[0], 2, "GeV"), "p_y": (dps[1], 2, "GeV"), "p_z": (dps[2], 2, "GeV"),
            "p_hnl_x": (dps[0], 1, "GeV"), "p_hnl_y": (dps[1], 1, "GeV"), "p_hnl_z": (dps[2], 1, "GeV"),
        }
        out = {}
        for k, (sigma, n, unit) in spec.items():
            if k not in metric_means:
                continue
            mse = float(metric_means[k])
            entry = {"rmse": float(np.sqrt(mse / n) * sigma), "unit": unit}
            if metric_errors is not None and k in metric_errors:
                sem = float(metric_errors[k])
                entry["error"] = float(sigma * sem / (2.0 * np.sqrt(n * mse))) if mse > 0 else float("inf")
            out[k] = entry
        return out

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
        """Map a *physical* design (the Design NAMEDTUPLE, or a flat array as a conversion intermediate)
        into per-layer ``(layers, angles, Bs)`` for the C solver. Defined by the design subclass; it reads
        the design's fields BY NAME (never positional-slices a carried flat)."""
        raise NotImplementedError("_design_to_geometry is defined by the design subclass")

    def _resolve_design(self, design, n):
        """Normalize ``design`` (a Design namedtuple / config Mapping / flat physical array) to the Design
        NAMEDTUPLE with every field broadcast to a leading ``(n, ...)`` event axis. The flat<->namedtuple
        conversion happens ONCE here; downstream (:meth:`_design_to_geometry`) carries the namedtuple, so
        no raw flat unencoded design is ever stored or threaded through the simulation."""
        nt = self.unflatten_design(self.flatten_design(design))  # -> Design namedtuple (fields 1-D or (b, ...))

        def batch(x):
            x = np.asarray(x, np.float32)
            x = x[None] if x.ndim == 1 else x  # ensure a leading event axis
            return np.broadcast_to(x, (n,) + x.shape[1:])

        return jax.tree.map(batch, nt)

    # ------------------------------------------------------------------ #
    # Event generation
    # ------------------------------------------------------------------ #
    def size(self):
        """Number of available events: the loaded FairShip replay rows (engine='relay'), the finite
        data-backed count, or ``None`` (infinite -- the analytic source). Scripts use it to build a
        shuffled event_index (and to decide whether `n` oversamples)."""
        if self._relay:
            return self._relay_n
        return self._n_events if self._events is not None else None

    # ------------------------------------------------------------------ #
    # Relay source (engine='relay'): replay packed REAL FairShip hits, no solver.
    # ------------------------------------------------------------------ #
    def _load_relay(self, data_dir):
        """Load + pack ONE FairShip digi file (a small replay sample) from ``data_dir`` -- engine='relay'.
        No design is stored: combine receives the geometry the data was recorded at (the caller passes it);
        event SELECTION is the caller's (external ``event_index``)."""
        import os
        from ..data.fairship_loader import load_fairship_digi, pack_fairship_events

        if data_dir is None:
            raise ValueError("engine='relay' requires data_dir (the FairShip digi directory to replay)")
        data, _nf = load_fairship_digi(n_files=1, data_glob=os.path.join(data_dir, "*.npz"))
        event, mask, _rows, truth = pack_fairship_events(
            data, n_stations=self.n_stations, n_views_per_station=self.n_views_per_station,
            n_layers_per_view=self.n_layers_per_view, n_straws=self.n_straws,
            max_hits=self.max_hits_per_event, tdc_clip=getattr(self, "tdc_clip", 1.0e4))
        self._relay_event, self._relay_mask, self._relay_truth = event, mask, truth
        # daughter target 9-vec [vertex(3), p1(3), p2(3)] from the FairShip truth (15-vec).
        self._relay_daughter = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], axis=1).astype(np.float32)
        self._relay_n = int(mask.shape[0])

    def _gather_relay(self, idx):
        """Replay the packed events at ``idx`` -> ``(ground_truth, event, mask, target)`` (produce order)."""
        import jax
        event = jax.tree.map(lambda a: a[idx], self._relay_event)  # StrawEvent (B, M)
        target = self._pack_target(self._relay_daughter[idx])  # DaughterTarget
        truth = self._relay_truth[idx]
        # StrawGroundTruth (== conditioning): [mass, momentum(3)] + decay vertex (truth[:, 4:7]).
        ground_truth = self._pack_ground_truth(truth[:, 4:], truth[:, 0:4])
        return ground_truth, event, np.asarray(self._relay_mask[idx], np.int32), target

    def events_for_rows(self, rows):
        """Deterministic ordered relay access (for tracking_nn-style eval): the packed ``(event, mask,
        target)`` for ``rows`` (truth row indices), in order."""
        _gt, event, mask, target = self._gather_relay(np.asarray(rows))
        return event, mask, target

    def _events_at(self, event_index):
        """Return ``(pool, boundaries, targets, conditioning)`` for the integer ``event_index``: GATHER from
        the loaded data pool, or -- with no data source -- SYNTHESIZE the events (generic HNL physics). The
        event SOURCE (data or synthetic) is a base concern; the GEOMETRY parametrization is the subclass's.
        No detector state, no pools."""
        idx = np.asarray(event_index, np.int64).reshape(-1)
        if self._events is None:
            return self._synthesize(idx)
        ev = self._events
        offsets = ev["offsets"]  # (n_events+1,)
        boundaries = np.stack([offsets[idx], offsets[idx + 1]], axis=1).astype(np.int32)
        conditioning = (
            ev["conditioning"][idx] if "conditioning" in ev else np.zeros((len(idx), 4), dtype=np.float32)
        )  # raw HNL [mass, p(3)]; the decay vertex (from `targets`) completes the GroundTruth
        targets = ev[self._targets_field][idx]  # "targets" (HNL 6-vec) or "daughter_targets" (subclass)
        return self._pool, boundaries, targets, conditioning

    def _synthesize(self, event_index):
        """No-data analytic INPUT-EVENT source: synthesize the events at ``event_index`` -- HNL
        momentum/vertex Gaussians -> 2-body decay (muon + pion) -> 2 daughters -- per index. Returns a
        fresh ``(pool, boundaries, targets, conditioning)`` with ``boundaries[i] = [2i, 2i+2)``. GENERIC
        HNL physics, independent of the geometry parametrization."""
        n = event_index.shape[0]
        u = det_uniforms(self._seeds(event_index), 9)                          # (n, 9) in [0,1)
        g = box_muller(u[:, :6]).reshape(n, 2, 3)                              # (n,2,3) standard normals
        vertex = (self.hnl_vertex_mean + self.hnl_vertex_sigma * g[:, 0]).astype(np.float32)
        P = (self.hnl_momentum_mean + self.hnl_momentum_sigma * g[:, 1]).astype(np.float64)  # HNL lab momentum
        M, m1, m2 = self.hnl_mass, _MUON_MASS, _PION_MASS
        E = np.sqrt(np.sum(P**2, axis=1) + M**2)
        E1, E2, p_star = self._decay_E1, self._decay_E2, self._decay_pstar
        cos_t = 2.0 * u[:, 6] - 1.0                                            # isotropic decay direction
        phi = 2.0 * np.pi * u[:, 7]
        sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
        d = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
        p1 = _boost(p_star * d, E1, P, E, M)                                   # (n,3) daughter 1
        p2 = _boost(-p_star * d, E2, P, E, M)                                  # (n,3) daughter 2
        q1 = np.where(u[:, 8] < 0.5, -1.0, 1.0).astype(np.float32)             # opposite charges
        q2 = -q1
        masses = np.tile(np.array([m1, m2], np.float32), n)
        charges = np.stack([q1, q2], axis=1).reshape(-1).astype(np.float32)
        positions = np.repeat(vertex, 2, axis=0).astype(np.float32)
        momenta = np.stack([p1, p2], axis=1).reshape(-1, 3).astype(np.float32)
        pool = Pool(masses, charges, positions, momenta, np.zeros(2 * n, np.float32))
        boundaries = np.stack([np.arange(0, 2 * n, 2), np.arange(2, 2 * n + 1, 2)], axis=1).astype(np.int32)
        Pf = P.astype(np.float32)
        # The regression target follows _targets_field: daughters [vertex, p1, p2] (9) or HNL [vertex, P] (6).
        if self._targets_field == "daughter_targets":
            targets = np.concatenate([vertex, p1, p2], axis=1).astype(np.float32)
        else:
            targets = np.concatenate([vertex, Pf], axis=1).astype(np.float32)
        conditioning = np.concatenate([np.full((n, 1), M, np.float32), Pf], axis=1).astype(np.float32)
        return pool, boundaries, targets, conditioning

    def _run_solver(self, pool, boundaries, layers, angles, Bs, seeds, *,
                    z_planes=None, traj=None, n_cross=None, part_idx=None, primaries=False,
                    process_ids=None, tree=None):
        """Allocate the output buffers and run the propagation engine over `pool` (a :class:`Pool`) at the
        events `boundaries` ((n,2) [start,end) particle spans), with the per-event geometry (`layers`,
        `angles`, `Bs`) + per-event `seeds`. Returns `(hits_idx, tdc, traj)`: `hits_idx (n,M,4) uint32`
        [station,view,layer,straw] + `tdc (n,M) f32` (< 0 = no hit; init to -1). The engine fills them in
        place + handles the optional trajectory (`z_planes`/`traj`/...) and debug (`process_ids`/`tree`)."""
        n_events = boundaries.shape[0]
        M = self.max_hits_per_event
        hits_idx = np.zeros((n_events, M, 4), dtype=np.uint32)
        tdc = np.full((n_events, M), -1.0, dtype=np.float32)
        self.engine.solve(
            pool, np.ascontiguousarray(boundaries, np.int32), layers, angles, Bs, hits_idx, tdc,
            seeds=seeds, z_planes=z_planes, traj=traj, n_cross=n_cross, part_idx=part_idx,
            primaries=primaries, process_ids=process_ids, tree=tree,
        )
        return hits_idx, tdc, traj

    def simulate_debug(self, daughter_data, design, rng):
        """Run the solver with all debug outputs on (for sim-vs-MC matching).

        On top of the production ``X / mask`` this also returns the per-hit ``process_ids``
        (the TMCProcess code of the particle that caused each hit, parallel to ``mask``) and
        a flat/sparse MC-particle ``tree`` -- one row per tracked particle (primaries +
        secondaries) across the whole batch, located by ``event_index`` (CSR-style). Mirrors
        the FairShip ``mc_info`` / ``mc_particles`` layout so the two are directly comparable.

        Returns a dict with: ``X, mask, process_ids`` and the tree arrays ``pdg, process_id,
        parent_id, n_hits`` (int) + ``momentum (P, 3), position (P, 3), t0 (P,)`` (float) +
        ``event_index (P,)``.
        """
        flat = np.asarray(self.flatten_design(design), np.float32)
        n_events = 1 if flat.ndim == 1 else flat.shape[0]
        design = self._resolve_design(design, n_events)  # -> batched Design namedtuple (no carried flat)

        # Build a transient Pool from the given per-event particle arrays; boundaries select
        # the events (default: a single event spanning the whole pool). The engine packs the C
        # InputEvents internally.
        pool = Pool(daughter_data["masses"], daughter_data["charges"], daughter_data["positions"],
                    daughter_data["momenta"], daughter_data["times"])
        boundaries = daughter_data.get("boundaries")
        if boundaries is None:
            n_part = len(np.asarray(daughter_data["masses"]))
            boundaries = np.tile(np.array([[0, n_part]], dtype=np.int32), (n_events, 1))
        layers, angles, Bs = self._design_to_geometry(design)
        seeds = rng.integers(1, 2**32, size=n_events, dtype=np.uint32)

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
        hits_idx, tdc, _ = self._run_solver(
            pool, boundaries, layers, angles, Bs, seeds,
            primaries=True, process_ids=process_ids, tree=tree,
        )

        p = int(tree["count"][0])  # number of particles actually recorded
        ti, tf = tree["int"][:p], tree["float"][:p]
        return {
            "X": self._pack_event(hits_idx, tdc),
            "mask": (tdc >= 0).astype(np.int32),
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

    @staticmethod
    def _seeds(event_index):
        """Per-event uint32 seeds = a hash of ``event_index`` -- DESIGN-INDEPENDENT, so perturbing the
        design at a fixed index gives common random numbers (low-variance design gradient) and a repeated
        index reproduces the identical event. Never 0 (the C RNG wants a nonzero seed)."""
        h = (np.asarray(event_index, np.uint64) + np.uint64(1)) * np.uint64(2654435761)
        h ^= h >> np.uint64(16)
        s = (h & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        return np.where(s == np.uint32(0), np.uint32(1), s)

    def _simulate(self, design, event_index, *, z_planes=None, n_tracks=2, primaries=False):
        """Core simulation, shared by :meth:`__call__` and the tracker subclass: resolve ``design`` to the
        per-event geometry, gather/generate the events at ``event_index`` (:meth:`_events_at`), run the
        engine, and pack. Returns a dict with the fired-straw ``StrawEvent`` (``"X"``), the ``mask``
        (``tdc >= 0``), ``target``, ``ground_truth`` (+ the optional per-track trajectory when ``z_planes``
        is given). DETERMINISTIC in ``(design, event_index)``. ``design`` is one design (broadcast across
        ``event_index``) or a batched ``(n, design_dim)`` design (one per event)."""
        event_index = np.asarray(event_index, np.int64).reshape(-1)
        n = event_index.shape[0]
        design = self._resolve_design(design, n)  # -> batched Design namedtuple (no carried flat)
        pool, boundaries, targets, conditioning = self._events_at(event_index)
        layers, angles, Bs = self._design_to_geometry(design)

        traj = n_cross = part_idx = zp = None
        if z_planes is not None:
            zp = np.ascontiguousarray(z_planes, np.float32)
            m = zp.shape[0]
            traj = np.zeros((n, int(n_tracks), m, 3), np.float32)
            n_cross = np.zeros((n, int(n_tracks)), np.int32)
            part_idx = np.full((n, int(n_tracks)), -1, np.int32)
        hits_idx, tdc, _ = self._run_solver(
            pool, boundaries, layers, angles, Bs, self._seeds(event_index),
            z_planes=zp, traj=traj, n_cross=n_cross, part_idx=part_idx, primaries=primaries,
        )
        return {
            "X": self._pack_event(hits_idx, tdc),  # StrawEvent (int address + float TDC)
            "mask": (tdc >= 0).astype(np.int32),   # derived from tdc (< 0 = padding)
            "target": self._pack_target(targets),  # HNLTarget / DaughterTarget
            "ground_truth": self._pack_ground_truth(targets, conditioning),  # StrawGroundTruth
            "traj": traj, "n_cross": n_cross, "part_idx": part_idx,
        }

    def _pack_event(self, hits_idx, tdc):
        """Pack the engine's ``hits_idx (n, M, 4) uint32`` [station, view, layer, straw] + ``tdc (n, M)``
        float32 into a ``StrawEvent`` -- addresses to int32 (already integers), TDC stays float32 (a
        ``tdc < 0`` row is padding -- the network mask is ``tdc >= 0``)."""
        hi = np.asarray(hits_idx)
        return StrawEvent(station=hi[..., 0].astype(np.int32), view=hi[..., 1].astype(np.int32),
                          layer=hi[..., 2].astype(np.int32), straw=hi[..., 3].astype(np.int32),
                          tdc=np.asarray(tdc, np.float32))

    def _pack_target(self, targets):
        """Pack the daughter target array ``(n, 9)`` [vertex, p1, p2] into a ``DaughterTarget``."""
        t = np.asarray(targets)
        return DaughterTarget(vertex=t[..., :3], p1=t[..., 3:6], p2=t[..., 6:9])

    def _pack_ground_truth(self, targets, conditioning):
        """``StrawGroundTruth`` (== conditioning): HNL mass + momentum (from the data conditioning
        ``[mass, p(3)]``) and the decay vertex (the target's first 3 components)."""
        c, t = np.asarray(conditioning), np.asarray(targets)
        return StrawGroundTruth(mass=c[..., 0:1], momentum=c[..., 1:4], vertex=t[..., :3])

    def __call__(self, design, event_index):
        """Simulate the events at the integer ``event_index`` ``(n,)`` for ``design``, returning the
        ``(ground_truth, event, mask, target)`` namedtuple records (leaves carry the leading batch axis).

        DETERMINISTIC: the same ``(design, event_index)`` always yields the same output (the per-event RNG
        is seeded from ``event_index``). ``design`` is one design (broadcast across ``event_index``) or a
        batched ``(n, design_dim)`` design (one design per event). ``event`` is the fired-straw
        ``StrawEvent`` (address + TDC); the per-hit ``(x, y)``/``drift_r`` truth lives in the separate
        tracker detector. NEVER overridden -- the production is the :meth:`_produce` hook."""
        return self._produce(design, event_index)

    def _produce(self, design, event_index):
        """Produce the ``(ground_truth, event, mask, target)`` records for ``event_index``. Default: the
        SIMULATED path (run the C/numpy solver via :meth:`_simulate`). The design-blind RELAY source
        (engine='relay') instead GATHERS the packed FairShip events (no solver) -- handled here so there is
        no replay subclass. The single variation point of ``__call__``."""
        if self._relay:
            return self._gather_relay(np.asarray(event_index, np.int64))
        out = self._simulate(design, event_index)
        return out["ground_truth"], out["X"], out["mask"], out["target"]

    # Design encoding: the base Detector wraps the subclass `_encode_flat`/`_decode_flat`
    # (flat physical <-> encoded) with the dict<->flat conversion; the design subclass owns
    # the bounds, `design_spec`, `design_bounds`, and the flat encode/decode to N(0,1).

    # ------------------------------------------------------------------ #
    # Combine: ABSTRACT here (Detector declares combine_encoded/combined_event_shape/element_mask). Each
    # combine LEAF builds its own per-hit/-layer features; the shared 4-feature logic is the module
    # function `four_feature_combine` below. The TDC standardisation constants live here (event
    # normalisation is straw-wide, used by that function).
    # ------------------------------------------------------------------ #
    _TDC_MEAN = 440.0
    _TDC_STD = 80.0

    def _decode_to_layer_geometry(self, d_enc):
        """Encoded design ``(B, design_dim)`` -> per-layer ``(positions(B,n),
        angles(B,n), B_field(B))``, used by :func:`four_feature_combine`. Defined by the
        design subclass (it owns how the design expands into per-layer geometry)."""
        raise NotImplementedError("_decode_to_layer_geometry is defined by the design subclass")


def four_feature_combine(det, event, encoded_design, mask=None):
    """The shared 4-feature combine: raw ``StrawEvent`` + ENCODED design -> per-hit features (normalised)
    ``[TDC, norm(layer z), wire_y_left, wire_y_right]``. Geometry-AGNOSTIC: it gathers each hit's own layer
    geometry through ``det._decode_to_layer_geometry`` (the design subclass owns that decode), so every
    4-feature combine leaf (``FreeStrawDetector``, ``Stereo4Feature``, incl. its ``engine='relay'`` mode) calls this.
    ``mask`` is accepted for the uniform signature but ignored (the element axis IS the hit axis; padded
    hits are zeroed downstream by the regressor mask).

    The sense wire is encoded as its two y-endpoints at the FIXED x-ends of the parallelogram (the sheared
    geometry ``Y = y - x*tan(angle)`` is constant along the wire) -- the natural input for stereo
    triangulation -> track fit -> curvature -> momentum, differentiable w.r.t. the design through the
    decode+gather."""
    import jax.numpy as jnp

    station = jnp.asarray(event.station, jnp.int32)
    view = jnp.asarray(event.view, jnp.int32)
    layer_in_view = jnp.asarray(event.layer, jnp.int32)
    straw = jnp.asarray(event.straw, jnp.float32)  # float: indexes the continuous wire-centre y
    tdc = (jnp.asarray(event.tdc, jnp.float32) - det._TDC_MEAN) / det._TDC_STD

    B, M = station.shape
    per_station = det.n_views_per_station * det.n_layers_per_view
    layer = station * per_station + view * det.n_layers_per_view + layer_in_view  # (B, M) global layer idx

    # Decode design and gather each hit's own layer geometry.
    d_enc = jnp.asarray(encoded_design, dtype=jnp.float32)
    if d_enc.ndim == 1:
        d_enc = jnp.broadcast_to(d_enc[None, :], (B, d_enc.shape[0]))
    positions, angles, _B = det._decode_to_layer_geometry(d_enc)  # (B,n),(B,n),(B,); field unused (fixed)

    z_hit = jnp.take_along_axis(positions, layer, axis=1)  # (B, M)
    angle_hit = jnp.take_along_axis(angles, layer, axis=1)  # (B, M)
    # Half-pitch stagger: even layer-in-view -> -h, odd -> +h (matches the C solver).
    y_stagger = jnp.where((layer_in_view & 1) == 1, 0.5 * det.layer_y_offset, -0.5 * det.layer_y_offset)
    straw_y = (straw + 0.5) * det.straw_pitch - det.layer_height + y_stagger  # wire centre y (at x=0)

    z_mid = 0.5 * (det.layer_bounds[0] + det.layer_bounds[1])
    z_half = max(0.5 * (det.layer_bounds[1] - det.layer_bounds[0]), 1e-6)
    norm_z = (z_hit - z_mid) / z_half

    # Wire y at the fixed x-ends (+/- layer_width); the x's are constant so omitted. Y-scale bounds |y|
    # over all straws and the steepest stereo angle so the features stay ~[-1, 1].
    dy = det.layer_width * jnp.tan(angle_hit)  # (B, M) y-offset from x=0 to x=+width
    a_max = max(abs(det.angle_bounds[0]), abs(det.angle_bounds[1]))
    y_scale = max(det.layer_height + det.layer_width * float(np.tan(a_max)), 1e-6)
    wire_y_left = (straw_y - dy) / y_scale
    wire_y_right = (straw_y + dy) / y_scale

    return jnp.stack([tdc, norm_z, wire_y_left, wire_y_right], axis=-1)


def four_feature_shape(det):
    """The 4-feature combine's ``combined_event_shape``: one element per hit, 4 features."""
    return (det.max_hits_per_event, 4)
