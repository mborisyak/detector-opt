"""JAX track-fit solver utilities (free functions, NOT a Detector): the Boris propagator, the momentum
readout, the per-(track, hit) drift-circle residual, the closest-approach vertex, and the coarse retina
seed lines. A differentiable JAX counterpart to the C straw solver, consumed by ``detopt.tracking``.

A track's internal state is ``(x0, y0, tx, ty, q/p)`` at a reference plane ``z_start``: transverse
position (cm), the two slopes ``dx/dz, dy/dz``, and the signed inverse momentum (1/MeV); rescaled to O(1)
by ``PARAM_SCALE`` for the optimiser. The field bend is linear in ``q/p`` (well-conditioned)."""
import numpy as np
import jax
import jax.numpy as jnp

# FairShip-matched constants (see detopt/detector/straw_detector.c).
K_BORIS = 44.937759
C_CM_PER_NS = 29.9792458
STRAW_VDRIFT = 0.0033333333  # drift speed (cm/ns), mirrors straw_detector.c
STRAW_SIGMA_SPATIAL = 0.012  # single-hit spatial resolution (cm), mirrors straw_detector.c

# Per-parameter scale: (x0, y0 [cm], tx, ty [slope], q/p [1/MeV]) -> O(1) internal params.
PARAM_SCALE = np.array([100.0, 100.0, 0.1, 0.1, 5.0e-5], np.float64)
QOP_MIN = 1.0e-6  # smooth |p| cap: |p| = 1/sqrt(qop^2 + QOP_MIN^2) -> <= 1 TeV, no 1/0 singularity.


def p_vec(track_params_phys, xp):
    """Momentum 3-vector (GeV) from physical params (x0, y0, tx, ty, q/p[1/MeV]). ``xp`` = np or jnp."""
    tx, ty, qop = track_params_phys[..., 2], track_params_phys[..., 3], track_params_phys[..., 4]
    nrm = xp.sqrt(tx * tx + ty * ty + 1.0)
    pmag = 1.0 / (xp.sqrt(qop * qop + QOP_MIN * QOP_MIN) * 1e3)  # |p| GeV (smoothly capped)
    return pmag[..., None] * xp.stack([tx, ty, xp.ones_like(tx)], -1) / nrm[..., None]


def boris_traj(x0, y0, z0, tx, ty, qop, field, dt, n_steps):
    """Fixed-dt, fixed-step Boris trajectory for an ultrarelativistic track (beta=1). ``qop`` = q/p in
    1/MeV (signed). Returns (z, x, y) each (n_steps+1,), including the start point so ``jnp.interp``
    covers the first plane. The bend is ``rot = K_BORIS*dt*Bx*qop`` (y-z rotation). ``field`` is either the
    Gaussian tuple ``(B, z0f, Bsig)`` -> ``Bx = B*exp(-0.5*((z-z0f)/Bsig)^2)`` (the sim's own field), or a
    callable ``bx_at(x, y, z) -> Bx`` (e.g. the interpolated FairShip field map, for fitting FairShip hits)."""
    if callable(field):
        bx_at = field
    else:
        B, z0f, Bsig = field
        bx_at = lambda x, y, z: B * jnp.exp(-0.5 * ((z - z0f) / Bsig) ** 2)
    n = jnp.sqrt(tx * tx + ty * ty + 1.0)
    vx0, vy0, vz0 = tx / n, ty / n, 1.0 / n  # unit velocity (beta=1)

    def step(carry, _):
        x, y, z, vx, vy, vz = carry
        Bx = bx_at(x, y, z)
        rot = K_BORIS * dt * Bx * qop
        vy_m = vy + vz * rot
        vz_m = vz - vy * rot
        s = 2.0 * rot / (1.0 + rot * rot)
        vy = vy + vz_m * s
        vz = vz - vy_m * s  # vx unchanged (B has no x-rotation component)
        x = x + vx * dt * C_CM_PER_NS
        y = y + vy * dt * C_CM_PER_NS
        z = z + vz * dt * C_CM_PER_NS
        return (x, y, z, vx, vy, vz), (z, x, y)

    _, (zt, xt, yt) = jax.lax.scan(step, (x0, y0, z0, vx0, vy0, vz0), None, length=n_steps)
    start = lambda v, t: jnp.concatenate([jnp.asarray(v, t.dtype).reshape(1), t])
    return start(z0, zt), start(x0, xt), start(y0, yt)


def make_residual(layer_z, layer_tan, z_start, field, dt, n_steps):
    """Per-(track, hit) drift-circle residual for a 2-track ``(2,5)`` internal-param set. Returns
    ``residual(states, hit_layer, hit_Y, hit_r, valid) -> (2, n_hits)`` in cm: ``|perpendicular
    track-to-wire distance| - hit_r`` (L/R ambiguity folded into the ``|.|``; ``hit_r = 0`` ->
    track-to-wire distance). ``hit_layer`` indexes ``layer_z``/``layer_tan``; ``hit_Y`` is the measured
    sheared-Y; ``valid`` is accepted for a uniform residual contract (the TDC residual needs it) but
    unused here -- the drift-circle residual is per-hit local, masked by the caller."""
    scale = jnp.asarray(PARAM_SCALE)
    layer_z, layer_tan = jnp.asarray(layer_z), jnp.asarray(layer_tan)
    cos_layer = 1.0 / jnp.sqrt(1.0 + layer_tan**2)  # per-layer sheared -> perpendicular factor

    def sheared_pred(tp):
        x0, y0, tx, ty, qop = tp * scale
        zt, xt, yt = boris_traj(x0, y0, z_start, tx, ty, qop, field, dt, n_steps)
        return jnp.interp(layer_z, zt, yt) - jnp.interp(layer_z, zt, xt) * layer_tan  # sheared-Y at every plane

    def residual(states, hit_layer, hit_Y, hit_r, valid=None):
        Yp = jnp.stack([sheared_pred(states[0]), sheared_pred(states[1])])[:, hit_layer]  # (2, n_hits) sheared
        wire_dist = (Yp - hit_Y[None]) * cos_layer[hit_layer][None]  # * cos(angle) -> true distance to wire
        return jnp.abs(wire_dist) - hit_r[None]

    return residual


def make_tdc_residual(layer_z, layer_tan, layer_width, z_start, field, dt, n_steps, v_drift):
    """Per-(track, hit) **TDC** residual for a 2-track ``(2,5)`` internal-param set -- the V2 target, the
    raw readout the network sees. Returns ``tdc_residual(states, hit_layer, hit_wireY, hit_tdc, valid) ->
    (2, n_hits)`` in **cm** (drift-radius space): ``track-to-wire distance - measured drift radius``, where
    the measured drift radius is recovered from the TDC by the FairShip readout model (straw_detector.c):

        TDC  = ToF + drift_time + propagation,  drift_radius = v_drift * (TDC - ToF - prop - origin)
        ToF  = arc-length(track, z_plane) / c           (time of flight, beta=1)
        prop = (layer_width - x_crossing) / (cos * c)   (signal propagation to the +x readout end)

    Returning cm (rather than ns) makes V2 hyperparameter-identical to V3 drift -- same s_hi / s_lo, same
    well-behaved objective for BOTH retina and NLL. The measured TDC is emitted relative to the event's
    earliest hit; the prediction is referenced the SAME way (to the predicted earliest hit, ``min`` of each
    hit's own-track predicted TDC) so the unknown event time origin cancels WITHOUT a free parameter (a
    free per-event ``t0`` is degenerate with the signal -- the fit would collapse onto unphysical
    near-constant-TDC tracks). ToF / propagation / origin are DETACHED (a fixed-point correction evaluated
    at the current track but carrying no gradient) so the steep ~1/v_drift slope can't trade against track
    length; the gradient flows only through the geometric track-to-wire distance, exactly like V3. Hits are
    assigned to tracks by the robust GEOMETRIC nearest-wire distance. ``hit_wireY`` is the fired wire's
    sheared-Y; ``hit_tdc`` the measured (relative) TDC."""
    scale = jnp.asarray(PARAM_SCALE)
    layer_z, layer_tan = jnp.asarray(layer_z), jnp.asarray(layer_tan)
    cos_layer = 1.0 / jnp.sqrt(1.0 + layer_tan**2)  # per-layer sheared -> perpendicular factor

    def per_track(tp, hit_layer, hit_wireY):
        x0, y0, tx, ty, qop = tp * scale
        zt, xt, yt = boris_traj(x0, y0, z_start, tx, ty, qop, field, dt, n_steps)
        seg = jnp.sqrt(jnp.diff(zt) ** 2 + jnp.diff(xt) ** 2 + jnp.diff(yt) ** 2)
        S = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])  # cumulative arc length (cm) along the track
        xp = jnp.interp(layer_z, zt, xt)[hit_layer]  # crossing x at each HIT's plane
        cp = (jnp.interp(layer_z, zt, yt) - jnp.interp(layer_z, zt, xt) * layer_tan)[hit_layer]  # sheared crossing
        tof = jnp.interp(layer_z, zt, S)[hit_layer] / C_CM_PER_NS  # time of flight (beta=1)
        cos_h = cos_layer[hit_layer]
        gdist = jnp.abs(cp - hit_wireY) * cos_h  # perpendicular crossing->wire distance (cm); drift = gdist/v
        nuisance = tof + (layer_width - xp) / (cos_h * C_CM_PER_NS)  # ToF + propagation (the slow corrections)
        return gdist, nuisance

    def tdc_residual(states, hit_layer, hit_wireY, hit_tdc, valid):
        g0, n0 = per_track(states[0], hit_layer, hit_wireY)
        g1, n1 = per_track(states[1], hit_layer, hit_wireY)
        gdist, nuisance = jnp.stack([g0, g1]), jnp.stack([n0, n1])  # (2, n_hits): track-wire distance (cm), ToF+prop (ns)
        vh = valid > 0
        pred = gdist / v_drift + nuisance  # full predicted absolute TDC (ns)
        assign = jnp.argmin(gdist, axis=0)  # geometric nearest track per hit
        own_pred = jnp.take_along_axis(pred, assign[None], 0)[0]  # (n_hits,) each hit's own-track pred
        min_pred = jnp.min(jnp.where(vh, own_pred, 1e20))  # predicted earliest hit (the event origin)
        ref = jax.lax.stop_gradient(nuisance - min_pred)  # detached: ToF + prop - origin
        measured_r = v_drift * (hit_tdc[None] - ref)  # measured drift radius (cm), TDC with corrections removed
        return gdist - measured_r  # cm, drift-circle residual (gradient flows only through gdist, like V3)

    return tdc_residual


def vertex_doca(ph, z_start):
    """Decay vertex + DoCA from fitted physical states ``ph`` (N,2,5) by back-extrapolating both tracks
    to their closest approach upstream (field ~ 0 there). Returns ``(vertex (N,3), doca (N,))``. numpy."""
    nrm = np.sqrt(ph[..., 2] ** 2 + ph[..., 3] ** 2 + 1.0)
    direc = np.stack([ph[..., 2], ph[..., 3], np.ones_like(nrm)], -1) / nrm[..., None]  # (N, 2, 3) unit dir
    a = np.stack([ph[..., 0], ph[..., 1], np.full_like(ph[..., 0], z_start)], -1)  # (N, 2, 3) state at z_start
    d0, d1, w0 = direc[:, 0], direc[:, 1], a[:, 0] - a[:, 1]
    b_ = (d0 * d1).sum(-1)
    denom = np.where(np.abs(1 - b_**2) < 1e-6, 1e-6, 1 - b_**2)
    s0 = (b_ * (d1 * w0).sum(-1) - (d0 * w0).sum(-1)) / denom
    s1 = ((d1 * w0).sum(-1) - b_ * (d0 * w0).sum(-1)) / denom
    P0, P1 = a[:, 0] + s0[:, None] * d0, a[:, 1] + s1[:, None] * d1  # closest-approach point on each track
    return 0.5 * (P0 + P1), np.linalg.norm(P0 - P1, axis=1)


def doca_sq_jax(states_phys, z_start):
    """Single-event jax closest approach of the two tracks ``states_phys`` (2,5): returns
    ``(vertex(3), doca_squared)`` by back-extrapolating both to their mutual closest point (field ~ 0
    upstream). The SQUARED separation is returned (not its sqrt) so the common-vertex penalty is
    differentiable at doca -> 0 (``d|x|/dx`` is 0/0 there). For the jitted fit only."""
    tx, ty = states_phys[:, 2], states_phys[:, 3]
    nrm = jnp.sqrt(tx * tx + ty * ty + 1.0)
    direc = jnp.stack([tx, ty, jnp.ones_like(tx)], -1) / nrm[:, None]  # (2, 3) unit dir
    a = jnp.stack([states_phys[:, 0], states_phys[:, 1], jnp.full_like(tx, z_start)], -1)  # (2, 3)
    d0, d1, w0 = direc[0], direc[1], a[0] - a[1]
    b_ = jnp.sum(d0 * d1)
    denom = jnp.where(jnp.abs(1 - b_ * b_) < 1e-6, 1e-6, 1 - b_ * b_)
    s0 = (b_ * jnp.sum(d1 * w0) - jnp.sum(d0 * w0)) / denom
    s1 = (jnp.sum(d1 * w0) - b_ * jnp.sum(d0 * w0)) / denom
    P0, P1 = a[0] + s0 * d0, a[1] + s1 * d1
    return 0.5 * (P0 + P1), jnp.sum((P0 - P1) ** 2)
