"""Debug / test detector (see ``debug-detector-spec.md``).

A simplified, self-contained straw tracker used to exercise the BO pipeline
without GEANT/Pythia data. Everything is analytic NumPy (no compiled solver, no
ODE); the detector samples its own HNL -> muon + pion events and propagates the
daughters (straight -> circular arc in the magnet -> straight).

Geometry (cm, z = beam axis):
  * decay volume 0-200, two stations in 200-275, magnet 275-325, two in 325-400;
  * 4 stations x 4 views x 1 layer of 256 straws; views are 200 cm wide in x.

Design (un-encoded, physical), ``design_dim = 9``::

    [station_z (4), view_tilt (4, shared across stations), B (1 Tesla)]

Per-hit raw features (``raw_feature_dim = 4``): ``[station, view, straw, edep]``.
Target (``target_dim = 6``): ``[vtx_x, vtx_y, vtx_z, HNL_px, HNL_py, HNL_pz]``.
"""

from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from ..utils.encoding import uniform_to_normal_jax, normal_to_uniform_jax
from ..utils import tensor
from .common import Detector

__all__ = ["DebugDetector", "DebugEvent", "DebugTarget", "DebugDesign"]

_MUON_MASS = 0.105658  # GeV
_PION_MASS = 0.139570  # GeV
_C_CM_PER_NS = 29.9792


class DebugEvent(NamedTuple):
    """Raw per-hit debug event: int (station, view, straw) address + float energy deposit."""

    station: jax.Array  # int32
    view: jax.Array  # int32
    straw: jax.Array  # int32
    energy: jax.Array  # float32 (edep)


class DebugTarget(NamedTuple):
    """Debug target == discriminator conditioning: HNL decay vertex (cm) + HNL momentum (GeV)."""

    vertex: jax.Array  # (..., 3)
    momentum: jax.Array  # (..., 3)


class DebugDesign(NamedTuple):
    """Debug design: per-station z, per-view tilt (shared across stations), field strength."""

    stations: jax.Array  # (..., n_stations)
    tilts: jax.Array  # (..., n_views_per_station)
    field_strength: jax.Array  # (..., 1)


class DebugDetector(Detector):
    def __init__(
        self,
        # geometry
        n_stations: int = 4,
        n_views_per_station: int = 4,
        n_straws: int = 256,
        view_half_width: float = 100.0,  # x half-extent (straw length / 2)
        view_half_height: float = 100.0,  # y half-extent (straw stack)
        magnet_z: tuple = (275.0, 325.0),
        decay_volume_z: tuple = (0.0, 200.0),
        station_z: tuple = (225.0, 250.0, 350.0, 375.0),
        view_tilt: tuple = (-12.0, -4.0, 4.0, 12.0),  # cm, shared across stations
        tilt_bound: float = 20.0,
        # magnet
        B: float = 0.3,  # Tesla (nominal / design centre)
        B_bounds: tuple = (0.0, 1.0),
        # physics (simple distributions)
        hnl_mass: float = 1.0,  # GeV
        # Tuned so ~94% of events have both daughters cross all 4 stations
        # at the nominal design (see debug-detector-spec.md, acceptance target).
        momentum_mean: tuple = (0.0, 0.0, 10.0),  # GeV/c
        momentum_sigma: tuple = (0.15, 0.15, 1.0),
        vertex_sigma_xy: float = 1.5,  # cm
        # measurement (noisy ionization)
        edep_mean: float = 1.0,
        edep_sigma: float = 0.2,
        # multiple scattering: per-station-plane RMS deflection angle (radians).
        # Small Gaussian angular kicks accumulate down the track (random walk).
        scatter_sigma: float = 1e-3,
        loss=None,
        pool_split=None,  # Sequence|Mapping of fractions -> disjoint event STREAMS (keys define pools)
    ):
        self.n_stations = int(n_stations)
        self.n_views_per_station = int(n_views_per_station)
        self.n_straws = int(n_straws)
        self.n_layers_per_view = 1
        self.view_half_width = float(view_half_width)
        self.view_half_height = float(view_half_height)
        self.magnet_z = (float(magnet_z[0]), float(magnet_z[1]))
        self.decay_volume_z = (float(decay_volume_z[0]), float(decay_volume_z[1]))

        self.station_z = np.asarray(station_z, dtype=np.float32)
        self.view_tilt = np.asarray(view_tilt, dtype=np.float32)
        self.tilt_bound = float(tilt_bound)
        self.B = float(B)
        self.B_bounds = (float(B_bounds[0]), float(B_bounds[1]))

        self.hnl_mass = float(hnl_mass)
        self.momentum_mean = np.asarray(momentum_mean, dtype=np.float32)
        self.momentum_sigma = np.asarray(momentum_sigma, dtype=np.float32)
        self.vertex_sigma_xy = float(vertex_sigma_xy)
        self.edep_mean = float(edep_mean)
        self.edep_sigma = float(edep_sigma)
        self.scatter_sigma = float(scatter_sigma)

        # straw pitch in the (sheared) straw coordinate c = y - (tilt/half_width)*x
        self.straw_pitch = 2.0 * self.view_half_height / self.n_straws

        # Disjoint event pools: DebugDetector synthesizes events, so a pool is an independent
        # RNG STREAM keyed by the pool key (fractions are informational only here).
        self.pool_split = self.resolve_pool_split(pool_split)
        self._default_pool = next(iter(self.pool_split))

        # Station-position bounds: stations before the magnet live in
        # [decay_end, magnet_start], those after in [magnet_end, z_max].
        z_before = (self.decay_volume_z[1], self.magnet_z[0])
        z_after = (self.magnet_z[1], 400.0)
        n_before = self.n_stations // 2
        lows, highs = [], []
        for s in range(self.n_stations):
            lo, hi = z_before if s < n_before else z_after
            lows.append(lo)
            highs.append(hi)
        lows += [-self.tilt_bound] * self.n_views_per_station + [self.B_bounds[0]]
        highs += [self.tilt_bound] * self.n_views_per_station + [self.B_bounds[1]]
        self._design_low = np.asarray(lows, dtype=np.float32)
        self._design_high = np.asarray(highs, dtype=np.float32)

        # Target normalisation (analytic from the sampling distributions).
        dvz0, dvz1 = self.decay_volume_z
        self.target_mean = np.array(
            [
                0.0,
                0.0,
                0.5 * (dvz0 + dvz1),
                self.momentum_mean[0],
                self.momentum_mean[1],
                self.momentum_mean[2],
            ],
            dtype=np.float32,
        )
        self.target_std = np.array(
            [
                self.vertex_sigma_xy,
                self.vertex_sigma_xy,
                (dvz1 - dvz0) / np.sqrt(12.0),
                max(self.momentum_sigma[0], 1e-3),
                max(self.momentum_sigma[1], 1e-3),
                max(self.momentum_sigma[2], 1e-3),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Shapes
    # ------------------------------------------------------------------ #
    @property
    def max_hits_per_event(self) -> int:
        # 2 daughters x stations x views (1 layer each, co-located at station z)
        return 2 * self.n_stations * self.n_views_per_station

    def design_shape(self):
        # station_z (n_stations) + view_tilt (n_views, shared) + B (1)
        return (self.n_stations + self.n_views_per_station + 1,)

    def event_spec(self):
        # Raw per-hit features: int32 [station, view, straw] + float32 energy deposit.
        M = self.max_hits_per_event
        i = jax.ShapeDtypeStruct((M,), np.int32)
        return DebugEvent(station=i, view=i, straw=i, energy=jax.ShapeDtypeStruct((M,), np.float32))

    def combined_event_shape(self):
        # combine() -> [energy, norm_station_z, wire_y_left, wire_y_right, field_strength]
        return (self.max_hits_per_event, 5)

    def target_spec(self):
        f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
        return DebugTarget(vertex=f(3), momentum=f(3))

    def ground_truth_spec(self):
        # Ground truth == conditioning: the full 6-vec target stands in (stereo uses [mass, p, vtx]).
        f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
        return DebugTarget(vertex=f(3), momentum=f(3))

    def metric_labels(self):
        """Keys of the metric() dict, in display order."""
        return ("loss", "vertex_x", "vertex_y", "vertex_z", "p_x", "p_y", "p_z")

    def loss(self, predicted, target):
        """Per-sample ``(...,)`` MSE between predictions and the ALREADY-NORMALIZED 6-vec target
        ``[vertex(3), momentum(3)]`` (caller standardises via :meth:`normalize_target`)."""
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
    # Design slicing
    # ------------------------------------------------------------------ #
    def _split_design(self, design):
        """``(B, 9)`` physical design (``DebugDesign`` / Mapping / flat array) -> (station_z (B,S),
        tilt (B,V), B (B,))."""
        design = np.asarray(self.flatten_design(design), dtype=np.float32)
        if design.ndim == 1:
            design = design[None, :]
        s = self.n_stations
        v = self.n_views_per_station
        return design[:, :s], design[:, s : s + v], design[:, s + v]

    # ------------------------------------------------------------------ #
    # Event generation (numpy)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pool_int(pool):
        """Stable (run-independent) int from a pool key (int or str) for seed-folding."""
        import zlib

        return pool if isinstance(pool, int) else int(zlib.crc32(str(pool).encode())) & 0x7FFFFFFF

    def generate_events(self, rng, n, pool=None):
        """Sample ``n`` HNL -> muon + pion events. Returns a dict of numpy arrays.

        ``pool`` (int|str) selects a disjoint event stream: the kinematics rng is folded
        with the pool key, so different pools yield independent (non-overlapping) events.
        """
        if pool is not None and pool != self._default_pool:
            rng = np.random.default_rng([int(rng.integers(0, 2**31)), self._pool_int(pool)])
        dvz0, dvz1 = self.decay_volume_z
        z0 = rng.uniform(dvz0, dvz1, size=n)
        x0 = rng.normal(0.0, self.vertex_sigma_xy, size=n)
        y0 = rng.normal(0.0, self.vertex_sigma_xy, size=n)
        vertex = np.stack([x0, y0, z0], axis=1).astype(np.float32)

        # HNL momentum: all three components normal.
        P = rng.normal(self.momentum_mean, self.momentum_sigma, size=(n, 3))
        M = self.hnl_mass
        E = np.sqrt(np.sum(P**2, axis=1) + M**2)

        # 2-body decay kinematics (rest frame), masses fixed.
        m1, m2 = _MUON_MASS, _PION_MASS
        E1 = (M**2 + m1**2 - m2**2) / (2.0 * M)
        E2 = (M**2 + m2**2 - m1**2) / (2.0 * M)
        p_star = np.sqrt(max(E1**2 - m1**2, 0.0))

        # isotropic direction in the rest frame
        cos_t = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        sin_t = np.sqrt(1.0 - cos_t**2)
        d = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
        p1_rf = p_star * d
        p2_rf = -p1_rf

        mu_p = _boost(p1_rf, E1, P, E, M)
        pi_p = _boost(p2_rf, E2, P, E, M)

        # opposite charges; randomise which daughter is positive.
        q_mu = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
        q_pi = -q_mu

        momenta = np.stack([mu_p, pi_p], axis=1).astype(np.float32)  # (n, 2, 3)
        charges = np.stack([q_mu, q_pi], axis=1).astype(np.float32)  # (n, 2)
        target = np.concatenate([vertex, P.astype(np.float32)], axis=1)  # (n, 6)
        ground_truth = np.concatenate([charges, mu_p, pi_p], axis=1).astype(np.float32)  # (n, 8)
        return {
            "vertex": vertex,
            "momenta": momenta,
            "charges": charges,
            "target": target,
            "ground_truth": ground_truth,
        }

    # ------------------------------------------------------------------ #
    # Propagation (numpy): straight -> circular arc in magnet -> straight.
    # x is never bent (B || x); y bends in the y-z plane inside the magnet.
    # ------------------------------------------------------------------ #
    def _scatter_offsets(self, rng, station_z):
        """Accumulated multiple-scattering offsets at each station plane.

        A small Gaussian angular kick ``~N(0, scatter_sigma)`` is sampled at every
        station plane (independently per daughter and per transverse axis). Treating
        the planes as a random walk in angle, the transverse displacement that a
        kick at plane ``j`` induces at a downstream plane ``k > j`` is
        ``dtheta_j * (z_k - z_j)``. Returns ``(off_x, off_y)`` of shape ``(n, 2, S)``.
        """
        n, S = station_z.shape
        if self.scatter_sigma <= 0.0:
            z = np.zeros((n, 2, S), dtype=np.float32)
            return z, z.copy()
        dtheta_x = rng.normal(0.0, self.scatter_sigma, size=(n, 2, S)).astype(np.float32)
        dtheta_y = rng.normal(0.0, self.scatter_sigma, size=(n, 2, S)).astype(np.float32)
        Z = station_z[:, None, :]  # (n,1,S)
        dz = Z[..., :, None] - Z[..., None, :]  # (n,1,S,S): z_k - z_j
        tri = np.tril(np.ones((S, S), np.float32), k=-1)  # [k,j] = 1 iff j < k
        lever = dz * tri  # (n,1,S,S)
        off_x = np.sum(lever * dtheta_x[:, :, None, :], axis=-1)  # (n,2,S)
        off_y = np.sum(lever * dtheta_y[:, :, None, :], axis=-1)
        return off_x.astype(np.float32), off_y.astype(np.float32)

    def _propagate(self, vertex, momenta, charges, station_z, B, rng=None):
        x0 = vertex[:, 0][:, None]  # (n,1) shared across daughters
        y0 = vertex[:, 1][:, None]
        z0 = vertex[:, 2][:, None]
        px, py, pz = momenta[..., 0], momenta[..., 1], momenta[..., 2]  # (n,2)
        pz = np.where(np.abs(pz) < 1e-6, 1e-6, pz)

        zs = station_z[:, None, :]  # (n,1,S)
        x = x0[:, :, None] + (px / pz)[:, :, None] * (zs - z0[:, :, None])  # (n,2,S)
        y_before = y0[:, :, None] + (py / pz)[:, :, None] * (zs - z0[:, :, None])

        # --- magnet arc -> exit state at z = magnet_z[1] ---
        zm0, zm1 = self.magnet_z
        p_yz = np.sqrt(py**2 + pz**2)
        y_e = y0 + (py / pz) * (zm0 - z0)  # y at magnet entry (n,2)
        Bsafe = max(abs(float(B)) if np.ndim(B) == 0 else 1.0, 0.0)
        B_arr = np.broadcast_to(np.asarray(B, np.float32)[:, None], py.shape)  # (n,2)

        # signed bend radius (cm); R = p_yz / (0.3 B q) [m] -> *100 [cm]
        denom = 0.3 * np.abs(B_arr) * np.maximum(np.abs(charges), 1e-6)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        R = p_yz / denom * 100.0
        sgn = np.sign(charges) * np.sign(np.where(B_arr == 0, 1.0, B_arr))
        # centre of the circle in the (z, y) plane: C = entry + R * n_hat,
        # n_hat = sgn * (-py, pz)/p_yz  (toward centre, from F ~ q B (-py, pz))
        Cz = zm0 + R * sgn * (-py / p_yz)
        Cy = y_e + R * sgn * (pz / p_yz)
        disc = R**2 - (zm1 - Cz) ** 2
        sq = np.sqrt(np.maximum(disc, 0.0))
        y_lin = y_e + (py / pz) * (zm1 - zm0)
        yp, ym = Cy + sq, Cy - sq
        y_exit = np.where(np.abs(yp - y_lin) <= np.abs(ym - y_lin), yp, ym)
        # exit tangent: perpendicular to the radius, moving forward (dz > 0)
        rz, ry = (zm1 - Cz), (y_exit - Cy)
        tz_a, ty_a = -ry, rz
        use_a = tz_a > 0
        tz = np.where(use_a, tz_a, ry)
        ty = np.where(use_a, ty_a, -rz)
        slope_exit = ty / np.where(np.abs(tz) < 1e-9, 1e-9, tz)
        # near-zero field (huge R) -> straight passage
        straight = R > 1e5
        y_exit = np.where(straight, y_e + (py / pz) * (zm1 - zm0), y_exit)
        slope_exit = np.where(straight, py / pz, slope_exit)

        y_after = y_exit[:, :, None] + slope_exit[:, :, None] * (zs - zm1)
        y = np.where(zs < zm0, y_before, y_after)

        # Small multiple scattering: add the accumulated random-walk offsets at
        # each station plane (no-op when scatter_sigma == 0 or rng is None).
        if rng is not None and self.scatter_sigma > 0.0:
            off_x, off_y = self._scatter_offsets(rng, station_z)
            x = x + off_x
            y = y + off_y
        return x.astype(np.float32), y.astype(np.float32)

    def _hits(self, vertex, x, y, tilt):
        """Build per-(daughter, station, view) hit features + validity + acceptance.

        ``x``, ``y``: ``(n, 2, S)``. ``tilt``: ``(n, V)`` shared across stations.
        Returns ``X (n, M, 4)``, ``mask (n, M)``, ``station_face_ok (n, 2, S)``.
        """
        n, _, S = x.shape
        V = self.n_views_per_station
        slope = (tilt / self.view_half_width)[:, None, None, :]  # (n,1,1,V)

        x4 = x[:, :, :, None]  # (n,2,S,1)
        y4 = y[:, :, :, None]
        c = y4 - slope * x4  # sheared straw coordinate (n,2,S,V)
        straw_f = (c + self.view_half_height) / self.straw_pitch
        # non-crossing tracks can yield nan/inf/huge values here; clip to a safe
        # out-of-range sentinel before casting to int32.
        straw_f = np.nan_to_num(straw_f, nan=-1.0, posinf=-1.0, neginf=-1.0)
        straw_f = np.clip(straw_f, -1.0, self.n_straws + 1.0)
        straw = np.floor(straw_f).astype(np.int32)

        in_x = np.abs(x4) <= self.view_half_width
        in_straw = (straw >= 0) & (straw < self.n_straws)
        valid = in_x & in_straw  # (n,2,S,V)

        station_idx = np.broadcast_to(np.arange(S)[None, None, :, None], (n, 2, S, V))
        view_idx = np.broadcast_to(np.arange(V)[None, None, None, :], (n, 2, S, V))

        # face acceptance for the 90%-both-daughters criterion (geometry only)
        in_y_face = np.abs(y) <= self.view_half_height
        station_face_ok = (np.abs(x) <= self.view_half_width) & in_y_face  # (n,2,S)

        return station_idx, view_idx, straw, valid, station_face_ok

    # ------------------------------------------------------------------ #
    # Measurement assembly (shared by __call__ and sample_events)
    # ------------------------------------------------------------------ #
    def _measure(self, rng, ev, station_z, tilt, B):
        """Propagate, register straw hits, and pack the padded ``(X, mask)``.

        Returns ``(X (n,M,4), mask (n,M), sidx, vidx, straw, valid (n,2,S,V))``.
        """
        n = station_z.shape[0]
        x, y = self._propagate(ev["vertex"], ev["momenta"], ev["charges"], station_z, B, rng=rng)
        sidx, vidx, straw, valid, _face = self._hits(ev["vertex"], x, y, tilt)

        # deposited energy with this call's RNG stream
        edep = np.abs(rng.normal(self.edep_mean, self.edep_sigma, size=valid.shape)).astype(np.float32)

        feats = np.stack(
            [
                sidx.astype(np.float32),
                vidx.astype(np.float32),
                straw.astype(np.float32),
                edep,
            ],
            axis=-1,
        )  # (n,2,S,V,4)
        X = feats.reshape(n, self.max_hits_per_event, 4)
        mask = valid.reshape(n, self.max_hits_per_event).astype(np.int32)
        X = X * mask[..., None]  # zero padded slots
        return X, mask, sidx, vidx, straw, valid

    # ------------------------------------------------------------------ #
    # Detector contract
    # ------------------------------------------------------------------ #
    def __call__(self, seed, design, pool=None):
        """Generate one event per design row; returns ``(gt, X, mask, target)``.

        Events are generated on demand from ``seed``; ``pool`` (int|str) selects a disjoint
        event stream (folded into the kinematics rng), so train/val pools never overlap.
        """
        rng = np.random.default_rng(seed)
        station_z, tilt, B = self._split_design(design)
        ev = self.generate_events(rng, station_z.shape[0], pool=pool)
        X, mask, *_ = self._measure(rng, ev, station_z, tilt, B)
        return self._pack_target(ev["target"]), self._pack_event(X), mask, self._pack_target(ev["target"])

    def _pack_event(self, X):
        """Pack the ``(n, M, 4)`` float buffer [station, view, straw, edep] into a ``DebugEvent``."""
        X = np.asarray(X)
        idx = lambda c: np.rint(X[..., c]).astype(np.int32)
        return DebugEvent(station=idx(0), view=idx(1), straw=idx(2), energy=X[..., 3].astype(np.float32))

    def _pack_target(self, target):
        """Pack the ``(n, 6)`` target [vertex, momentum] into a ``DebugTarget`` (== conditioning)."""
        t = np.asarray(target)
        return DebugTarget(vertex=t[..., :3], momentum=t[..., 3:6])

    def sample_events(self, seed, design, n_traj_steps=4, pool=None):
        """Generate events and return a rich dict for visualisation / inspection.

        ``design`` is the *physical* (un-encoded) design ``(B, design_dim)`` (or a
        single ``(design_dim,)`` vector). In addition to the contract outputs it
        returns the daughter ``trajectories`` (a continuous polyline per daughter:
        vertex -> magnet entry -> magnet exit -> far plane; scattering is omitted
        from the clean polyline so its effect shows as a hit/track mismatch) and the
        per-(daughter, station, view) hit indices.
        """
        rng = np.random.default_rng(seed)
        station_z, tilt, B = self._split_design(design)
        ev = self.generate_events(rng, station_z.shape[0], pool=pool)
        X, mask, sidx, vidx, straw, valid = self._measure(rng, ev, station_z, tilt, B)
        trajectories = self._trajectory(ev["vertex"], ev["momenta"], ev["charges"], B)
        return {
            "vertex": ev["vertex"],
            "momenta": ev["momenta"],
            "charges": ev["charges"],
            "target": self._pack_target(ev["target"]),  # DebugTarget
            "ground_truth": self._pack_target(ev["target"]),  # == conditioning (full target stands in)
            "X": self._pack_event(X),  # DebugEvent
            "mask": mask,
            "station_idx": sidx,
            "view_idx": vidx,
            "straw": straw,
            "valid": valid,
            "station_z": station_z,
            "tilt": tilt,
            "B": B,
            "trajectories": trajectories,
        }

    def _trajectory(self, vertex, momenta, charges, B, z_max=400.0):
        """Continuous daughter polylines ``(n, 2, 4, 3)``.

        Anchors: decay vertex, magnet entry (``magnet_z[0]``), magnet exit
        (``magnet_z[1]``), and the far plane ``z_max``. The straight in/out segments
        are exact; the magnet is drawn as the chord between entry and exit, so the
        deflection appears as the slope change across the magnet.
        """
        x0, y0, z0 = vertex[:, 0], vertex[:, 1], vertex[:, 2]
        px, py, pz = momenta[..., 0], momenta[..., 1], momenta[..., 2]  # (n,2)
        pz = np.where(np.abs(pz) < 1e-6, 1e-6, pz)
        n = vertex.shape[0]
        zm0, zm1 = self.magnet_z

        # exit / far anchors: both in the after-magnet straight region (continuous).
        planes = np.broadcast_to(np.array([zm1, z_max], np.float32)[None, :], (n, 2))
        xa, ya = self._propagate(vertex, momenta, charges, planes, B, rng=None)  # (n,2,2)

        sx, sy = px / pz, py / pz  # before-magnet slopes (n,2)
        x0b = np.broadcast_to(x0[:, None], (n, 2))
        y0b = np.broadcast_to(y0[:, None], (n, 2))
        z0b = np.broadcast_to(z0[:, None], (n, 2))

        def before(z):
            return x0b + sx * (z - z0b), y0b + sy * (z - z0b)

        xe, ye = before(zm0)
        P0 = np.stack([x0b, y0b, z0b], axis=-1)
        P1 = np.stack([xe, ye, np.full((n, 2), zm0, np.float32)], axis=-1)
        P2 = np.stack([xa[..., 0], ya[..., 0], np.full((n, 2), zm1, np.float32)], axis=-1)
        P3 = np.stack([xa[..., 1], ya[..., 1], np.full((n, 2), z_max, np.float32)], axis=-1)
        return np.stack([P0, P1, P2, P3], axis=2).astype(np.float32)  # (n,2,4,3)

    def acceptance_fraction(self, seed, n, design=None):
        """Fraction of events where BOTH daughters cross all stations (face acceptance)."""
        if design is None:
            design = self.get_current_design_array()
        design = np.broadcast_to(np.asarray(design, np.float32)[None, :], (n, self.design_dim()))
        rng = np.random.default_rng(seed)
        station_z, tilt, B = self._split_design(design)
        ev = self.generate_events(rng, n)
        x, y = self._propagate(ev["vertex"], ev["momenta"], ev["charges"], station_z, B, rng=rng)
        _, _, _, _, face_ok = self._hits(ev["vertex"], x, y, tilt)
        both_all = face_ok.all(axis=(1, 2))  # all daughters, all stations
        return float(np.mean(both_all))

    # ------------------------------------------------------------------ #
    # Design encode / decode (JAX, differentiable) + current design
    # ------------------------------------------------------------------ #
    def design_spec(self):
        # DebugDesign filled with ShapeDtypeStruct; flatten/unflatten are generic (base, via tensor).
        f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
        return DebugDesign(stations=f(self.n_stations), tilts=f(self.n_views_per_station), field_strength=f(1))

    def design_bounds(self):
        return {
            "stations": (self.decay_volume_z[1], 400.0),
            "tilts": (-self.tilt_bound, self.tilt_bound),
            "field_strength": (self.B_bounds[0], self.B_bounds[1]),
        }

    def _encode_flat(self, design):
        d = jnp.asarray(design, dtype=jnp.float32)
        low = jnp.asarray(self._design_low)
        high = jnp.asarray(self._design_high)
        return uniform_to_normal_jax(d, low, high)

    def _decode_flat(self, encoded_design):
        e = jnp.asarray(encoded_design, dtype=jnp.float32)
        low = jnp.asarray(self._design_low)
        high = jnp.asarray(self._design_high)
        return normal_to_uniform_jax(e, low, high)

    def get_current_design_array(self):
        return np.concatenate([self.station_z, self.view_tilt, np.array([self.B], np.float32)]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Target / ground-truth normalisation + combine
    # combine_encoded owns event normalisation (the per-hit energy standardisation).
    # ------------------------------------------------------------------ #
    # Station-z normalisation constants (physical cm -> ~[-1, 1]).
    _Z_MEAN = 300.0
    _Z_STD = 75.0

    def normalize_target(self, target):
        """Physical ``DebugTarget`` ``[vertex(3), momentum(3)]`` -> standardised by ``target_mean``/``target_std``."""
        import jax.numpy as jnp

        flat, _ = tensor.flatten(target)  # DebugTarget record (or flat (B, 6) array) -> (B, 6)
        return (flat - jnp.asarray(self.target_mean)) / jnp.asarray(self.target_std)

    def denormalize_predictions(self, normalised):
        """Inverse of :meth:`normalize_target`: flat normalised array -> ``DebugTarget``."""
        import jax.numpy as jnp

        phys = jnp.asarray(normalised, jnp.float32) * jnp.asarray(self.target_std) + jnp.asarray(self.target_mean)
        return tensor.unflatten(tensor.structure(self.target_spec()), phys)

    def normalize_ground_truth(self, ground_truth):
        """Debug ground truth == conditioning == the 6-vec target; standardised by ``target_mean``/``target_std``."""
        return self.normalize_target(ground_truth)

    def combine_encoded(self, event, encoded_design):
        """Raw ``DebugEvent`` + ENCODED design -> per-hit features (all *normalised*):

            [energy, norm(station z), wire_y_left, wire_y_right, field_strength]

        The energy is standardised here (this method owns event normalisation). Mirrors
        :meth:`StrawDetector.combine_encoded`: the sheared sense wire is encoded by its two
        y-endpoints at the fixed x-ends (``x = +/- view_half_width``), where the shear slope is
        ``tilt / view_half_width`` so ``y(+/-width) = straw_y +/- tilt``. station z and tilt are
        gathered from the decoded design; differentiable through decode + gather.
        """
        import jax.numpy as jnp

        station_idx = jnp.asarray(event.station, jnp.int32)
        view_idx = jnp.asarray(event.view, jnp.int32)
        straw_idx = jnp.asarray(event.straw, jnp.float32)  # float: indexes the continuous wire-centre y
        energy = (jnp.asarray(event.energy, jnp.float32) - self.edep_mean) / max(self.edep_sigma, 1e-3)
        B, M = station_idx.shape

        d_enc = jnp.asarray(encoded_design, dtype=jnp.float32)
        if d_enc.ndim == 1:
            d_enc = jnp.broadcast_to(d_enc[None, :], (B, d_enc.shape[0]))
        phys = self._decode_flat(d_enc)  # (B, design_dim) flat physical
        station_z = phys[:, : self.n_stations]  # (B, S)
        tilts = phys[:, self.n_stations : self.n_stations + self.n_views_per_station]
        B_field = phys[:, -1]  # (B,)

        z_hit = jnp.take_along_axis(station_z, station_idx, axis=1)  # (B, M)
        tilt_hit = jnp.take_along_axis(tilts, view_idx, axis=1)  # (B, M) y-offset at x=+width (cm)
        straw_y = (straw_idx + 0.5) * self.straw_pitch - self.view_half_height  # wire centre y (at x=0)

        norm_z = (z_hit - self._Z_MEAN) / self._Z_STD
        # Wire y at the fixed x-ends (+/- view_half_width); y-scale bounds |y| over all straws
        # and the steepest tilt so the features stay ~[-1, 1].
        y_scale = max(self.view_half_height + self.tilt_bound, 1e-6)
        wire_y_left = (straw_y - tilt_hit) / y_scale
        wire_y_right = (straw_y + tilt_hit) / y_scale
        # field_strength = fraction of the field range, mapped to ~[-1, 1].
        b_lo, b_hi = self.B_bounds
        field_strength = jnp.broadcast_to((2.0 * (B_field - b_lo) / max(b_hi - b_lo, 1e-6) - 1.0)[:, None], (B, M))

        return jnp.stack([energy, norm_z, wire_y_left, wire_y_right, field_strength], axis=-1)


def _boost(p_rest, E_rest, P_hnl, E_hnl, M):
    """Boost a rest-frame 3-momentum (``p_rest``, energy ``E_rest``) into the lab,
    where the HNL has momentum ``P_hnl`` (n,3) and energy ``E_hnl`` (n,)."""
    beta = P_hnl / E_hnl[:, None]  # (n,3)
    b2 = np.sum(beta**2, axis=1)  # (n,)
    gamma = E_hnl / M  # (n,)
    bp = np.sum(beta * p_rest, axis=1)  # (n,)
    coeff = np.where(b2 > 1e-12, (gamma - 1.0) * bp / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    p_lab = p_rest + beta * (coeff + gamma * E_rest)[:, None]
    return p_lab.astype(np.float32)
