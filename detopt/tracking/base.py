"""``Tracker`` -- 2-track momentum/vertex fit over a raw ``StrawEvent``.

The base owns only what every tracker genuinely shares: the geometry it reads off the detector, the
opposite-charge multistart, the annealed Boris MAP fit, and the closest-approach prediction. The two axes
that differ are left to concrete classes (no branching in the base):

    _measurement(event, mask) -> (hit_layer, hit_Y, hit_r, valid)   # the MEASUREMENT (tubes/drift/hits)
    s_lo                                                            # that measurement's resolution (cm)
    objective(resid, valid, s, states_phys) -> scalar to MINIMISE   # retina score vs NLL mixture

A track's internal state is ``(x0, y0, tx, ty, q/p)`` at ``z_start``. ``shared_vertex`` adds a
closest-approach penalty so the two tracks prefer a common decay point.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

from ..detector import track_solver

MEV_PER_GEV = 1000.0
SEED_MOMENTUM_GEV = 3.0  # |p| of the multistart seeds (the fit refines magnitude + sign)
SEED_N_INTERCEPT = 4  # intercept grid points spanning the occupied sheared-Y range (per event)
Z_START_MARGIN_CM = 5.0  # reference plane sits this far upstream of the first layer
GRAD_CLIP_NORM = 1.0  # global-norm gradient clip
QUANTIZATION_SIGMA = 1.0 / np.sqrt(12.0)  # uniform-bin RMS; wire-centre resolution = pitch * this
INLIER_NSIGMA = 5.0  # a hit is "owned" by its nearest track (counts toward its hits/chi2) within this x s_lo


class Tracker(object):

    def __init__(self, detector, design, *, n_iters=1000, lr=0.05, s_hi=30.0, drift_sigma=0.05, coef=0.0,
               pi_noise=0.02, temperature=1.0, shared_vertex=False, vertex_penalty=0.001, dt=0.4, n_steps=160):
        self.n_iters, self.lr, self.s_hi = int(n_iters), float(lr), float(s_hi)
        self.drift_sigma, self.coef, self.pi_noise = float(drift_sigma), float(coef), float(pi_noise)
        self.temperature = float(temperature)  # retina likelihood power exp(-temperature*(r/s)^2); <1 = robust
        self.shared_vertex, self.vertex_penalty = bool(shared_vertex), float(vertex_penalty)
        self.dt, self.n_steps = float(dt), int(n_steps)
        # Geometry read off the detector at the (uniform) physical design.
        flat = np.asarray(detector.flatten_design(design), np.float32)
        flat = flat[None] if flat.ndim == 1 else flat
        positions, angles, _ = detector._design_to_geometry(flat)
        self.layer_z = np.asarray(positions[0], np.float64)
        self.layer_tan = np.tan(np.asarray(angles[0], np.float64))
        self.z_start = float(self.layer_z.min() - Z_START_MARGIN_CM)
        self.field = (float(detector.max_B), float(detector.z0), float(detector.B_sigma))
        self.pitch, self.height = float(detector.straw_pitch), float(detector.layer_height)
        self.layer_width = float(detector.layer_width)  # frame half-width (cm); +x readout end for TDC propagation
        self.layer_y_offset, self.n_straws = float(detector.layer_y_offset), int(detector.n_straws)
        self.per_station = detector.n_views_per_station * detector.n_layers_per_view
        self.n_layers_per_view, self.z0_field = detector.n_layers_per_view, float(detector.z0)
        self.momentum_mean = np.asarray(detector.daughter_momentum_mean, np.float64)
        self.momentum_sigma = np.asarray(detector.daughter_momentum_sigma, np.float64)
        self.sigma_hit = self.pitch * QUANTIZATION_SIGMA  # wire-centre (pos-only) resolution
        self.noise_W = float(self.n_straws * self.pitch)  # uniform-noise band width (cm)
        self.n_stations = self.layer_z.shape[0] // self.per_station
        self.residual = self._make_residual()
        self._fit = self._build_fit(lam_vertex=self.vertex_penalty if self.shared_vertex else 0.0)
        self._assign = jax.jit(jax.vmap(self._nearest_track))

    def _make_residual(self):
        """The per-(track, hit) residual builder. Default = the drift-circle (cm) residual shared by the
        tubes/drift/hits measurements; the TDC measurement overrides this with a time-domain residual."""
        return track_solver.make_residual(self.layer_z, self.layer_tan, self.z_start, self.field, self.dt, self.n_steps)

    # ---- the two axes that differ (concrete classes implement these) ------- #
    @property
    def s_lo(self):
        """The measurement resolution (cm) -- the anneal target / Gaussian width."""
        raise NotImplementedError()

    def _measurement(self, event, mask):
        """Raw ``StrawEvent`` + mask -> ``(hit_layer (B,M) int, hit_Y (B,M), hit_r (B,M), valid (B,M))``."""
        raise NotImplementedError()

    def objective(self, residual, valid, sharpness, states_phys):
        """Per-event scalar to MINIMISE from the ``(2, n_hits)`` drift-circle residual + the ``(2,5)``
        physical states (e.g. ``-retina_response`` or ``-mixture_loglik``, plus :meth:`momentum_penalty`)."""
        raise NotImplementedError()

    # ---- geometry helpers the measurements share --------------------------- #
    def global_layer(self, event):
        """Per-hit global layer index (station/view/layer-in-view -> 0..n_layers-1), clipped int32."""
        layer = event.station * self.per_station + event.view * self.n_layers_per_view + event.layer
        return np.clip(layer, 0, self.layer_z.shape[0] - 1).astype(np.int32)

    def wire_sheared_y(self, event):
        stagger = np.where((event.layer & 1) == 1, 0.5 * self.layer_y_offset, -0.5 * self.layer_y_offset)
        return (event.straw + 0.5) * self.pitch - self.height + stagger  # wire-centre sheared y

    def momentum_penalty(self, states_phys):
        """``coef`` * Gaussian prior on each daughter momentum (offered to the objective)."""
        z = (track_solver.p_vec(states_phys, jnp) - jnp.asarray(self.momentum_mean)) / jnp.asarray(self.momentum_sigma)
        return self.coef * 0.5 * jnp.sum(z**2)

    # ---- opposite-charge multistart ---------------------------------------- #
    def _seed_bank(self, hit_Y, valid):
        """Per event: a ``(K, 2, 5)`` internal seed bank for the parallel multistart. The two tracks are
        seeded at every distinct pair of intercepts on a grid spanning the occupied sheared-Y range, with
        slope 0 (the wide-sharpness anneal recovers the slope) and OPPOSITE charges. Fully vectorised over
        events -- no per-event search. ``K = C(SEED_N_INTERCEPT, 2) * 2``."""
        qop = 1.0 / (SEED_MOMENTUM_GEV * MEV_PER_GEV)  # 1/MeV; the fit refines magnitude + sign
        any_valid = (valid > 0).any(axis=1)  # (B,) events with at least one hit
        lo = np.where(any_valid, np.where(valid > 0, hit_Y, np.inf).min(axis=1), 0.0)  # (B,) sheared-Y span
        hi = np.where(any_valid, np.where(valid > 0, hit_Y, -np.inf).max(axis=1), 0.0)
        fraction = np.linspace(0.0, 1.0, SEED_N_INTERCEPT)
        intercept = lo[:, None] + (hi - lo)[:, None] * fraction  # (B, N_INTERCEPT)
        zeros = np.zeros(intercept.shape[0])

        def track(b, charge):  # (B,) intercept + charge sign -> (B, 5) physical state (slope 0)
            return np.stack([zeros, b, zeros, zeros, np.full(zeros.shape, charge * qop)], axis=-1)

        seeds = [np.stack([track(intercept[:, i], s0), track(intercept[:, j], s1)], axis=1)
                 for i in range(SEED_N_INTERCEPT) for j in range(i + 1, SEED_N_INTERCEPT)
                 for s0, s1 in ((+1.0, -1.0), (-1.0, +1.0))]  # opposite charges only
        bank = np.stack(seeds, axis=1)  # (B, K, 2, 5) physical
        return (bank / track_solver.PARAM_SCALE).astype(np.float32)

    # ---- the annealed Boris MAP fit ---------------------------------------- #
    def _build_fit(self, lam_vertex=0.0):
        residual = self.residual
        scale = jnp.asarray(track_solver.PARAM_SCALE)
        s_hi, s_lo, n_iters, lr = self.s_hi, self.s_lo, self.n_iters, self.lr
        s_rank = max(s_lo, self.sigma_hit)  # coarse multistart-ranking scale (tight s_lo collapses the score)
        z_start, objective = self.z_start, self.objective

        def loss(params, hit_layer, hit_Y, hit_r, valid, sharpness):
            states_phys = params * scale
            obj = objective(residual(params, hit_layer, hit_Y, hit_r, valid), valid, sharpness, states_phys)
            if lam_vertex > 0.0:
                _vertex, doca_sq = track_solver.doca_sq_jax(states_phys, z_start)
                obj = obj + lam_vertex * doca_sq  # common-vertex constraint (shared-vertex mode)
            return obj

        def fit_one(seed, hit_layer, hit_Y, hit_r, valid):
            opt = optax.chain(optax.clip_by_global_norm(GRAD_CLIP_NORM), optax.adam(optax.cosine_decay_schedule(lr, n_iters)))

            def step(carry, i):
                params, state = carry
                sharpness = s_hi * (s_lo / s_hi) ** (i / n_iters)  # anneal wide -> true resolution
                _value, grad = jax.value_and_grad(loss)(params, hit_layer, hit_Y, hit_r, valid, sharpness)
                update, state = opt.update(grad, state, params)
                return (optax.apply_updates(params, update), state), None

            (params, _), _ = jax.lax.scan(step, (seed, opt.init(seed)), jnp.arange(n_iters))
            return params, -loss(params, hit_layer, hit_Y, hit_r, valid, s_rank)  # rank at the coarse scale

        def fit_event(seed_bank, hit_layer, hit_Y, hit_r, valid):
            candidates, score = jax.vmap(lambda seed: fit_one(seed, hit_layer, hit_Y, hit_r, valid))(seed_bank)
            return candidates[jnp.argmax(score)]  # multistart winner (2,5)

        return jax.jit(jax.vmap(fit_event))

    def _predict(self, params):
        """Fitted ``(B, 2, 5)`` internal params -> ([vertex(3), p1(3), p2(3)] (B, 9), doca (B,)), numpy."""
        states_phys = np.asarray(params) * track_solver.PARAM_SCALE
        vertex, doca = track_solver.vertex_doca(states_phys, self.z_start)
        momenta = track_solver.p_vec(states_phys, np)  # (B, 2, 3)
        return np.concatenate([vertex, momenta[:, 0], momenta[:, 1]], axis=1), doca

    def _nearest_track(self, params, hit_layer, hit_Y, hit_r, valid):
        """Per hit: the nearest fitted track (argmin |residual|) + that residual (cm for the drift-circle
        measurements, ns for TDC). ``valid`` informs the TDC residual's per-event time-origin profiling."""
        absr = jnp.abs(self.residual(params, hit_layer, hit_Y, hit_r, valid))  # (2, M)
        return jnp.argmin(absr, axis=0), jnp.min(absr, axis=0)

    def _quality(self, assign, nearest, hit_layer, valid):
        """Per reconstructed track: hit count, stations crossed, and chi2/Ndf (Ndf = nhits - 5,
        measurement error = s_lo) over the hits each track OWNS -- its nearest hits within INLIER_NSIGMA
        (so secondary/noise hits a track does not actually fit are excluded, like FairShip's pattern
        recognition). Returns three ``(B, 2)`` numpy arrays."""
        inlier = (valid > 0) & (nearest < INLIER_NSIGMA * self.s_lo)  # (B, M) hits close enough to be a track's own
        station = hit_layer // self.per_station  # (B, M)
        nhits = np.zeros((assign.shape[0], 2))
        n_stations, chi2 = np.zeros_like(nhits), np.zeros_like(nhits)
        for t in (0, 1):
            owned = (assign == t) & inlier
            n = owned.sum(1)
            nhits[:, t] = n
            chi2[:, t] = (nearest**2 * owned).sum(1) / (self.s_lo**2) / np.maximum(n - 5, 1)
            n_stations[:, t] = sum(((station == s) & owned).any(1) for s in range(self.n_stations))
        return nhits, n_stations, chi2

    def evaluate(self, event, mask):
        """Like :meth:`fit`, plus the per-track FairShip-selection quantities: returns ``(pred9 (B,9),
        doca (B,), nhits (B,2), n_stations (B,2), chi2_ndf (B,2))`` so the caller can apply the cuts."""
        hit_layer, hit_Y, hit_r, valid = self._measurement(event, mask)
        bank = jnp.asarray(self._seed_bank(hit_Y, valid))
        hl, hY, hr = jnp.asarray(hit_layer), jnp.asarray(hit_Y), jnp.asarray(hit_r)
        v = jnp.asarray(valid)
        params = self._fit(bank, hl, hY, hr, v)
        pred9, doca = self._predict(params)
        assign, nearest = self._assign(params, hl, hY, hr, v)
        nhits, n_stations, chi2 = self._quality(np.asarray(assign), np.asarray(nearest), hit_layer, valid)
        return pred9, doca, nhits, n_stations, chi2

    def fit(self, event, mask):
        """Raw ``StrawEvent`` + ``mask`` (B, M) -> (predictions ``[vertex(3), p1(3), p2(3)]`` (B, 9),
        ``doca`` (B,) = the two fitted tracks' closest approach)."""
        hit_layer, hit_Y, hit_r, valid = self._measurement(event, mask)
        bank = jnp.asarray(self._seed_bank(hit_Y, valid))  # (B, K, 2, 5)
        params = self._fit(bank, jnp.asarray(hit_layer), jnp.asarray(hit_Y), jnp.asarray(hit_r), jnp.asarray(valid))
        return self._predict(params)
