#!/usr/bin/env python3
"""RKC2 stability for the enzyme solver: measure the boundary, measure the prior, pick the step.

Three measurements, in order.

1. **The scheme's real-axis stability boundary, in float32, on the target device.** The implemented
   step (:meth:`EnzymeDetector.rkc2_step`, not a formula for it) is applied to ``y' = -lambda y``
   over a sweep of ``z = lambda * dt``; the boundary is the largest ``z`` for which the amplification
   ``|R(z)|`` has stayed ``<= 1`` throughout. Measuring rather than quoting ``0.81 s^2`` matters
   because the Chebyshev recursion is evaluated in float32 here, and roundoff in a degree-``s``
   recursion eats into the theoretical value.
2. **The stiffest the prior gets.** A random search with local refinement over the parameter box,
   crossed with the temperature scan and the extremes of the design's enzyme fraction, for the
   largest ``|df/dx|``, taken at the FASTEST calibrated half-conversion time.
3. **The step that follows.** ``dt_max = boundary / |df/dx|``, then the recommended step applies a
   safety factor and the corresponding ``n_objective_steps`` / ``n_steps_per_measurement``. Finally
   the worst corner is integrated at the CONFIGURED step to confirm the extent stays physical and the
   half-step error estimate stays inside ``integration_tolerance``.

Verification (``scripts/verify_trajectory.py``) re-runs the whole chain at HALF this step, so a
result that depended on the step size shows up as a gap between the self-evaluated and verified
curves rather than passing silently.

    python scripts/check_stability.py =enzyme
    python scripts/check_stability.py =enzyme n_stages_sweep=true
"""

import numpy as np

import jax
import jax.numpy as jnp

import detopt
from detopt.detector.enzyme import PARAMETER_NAMES


def measure_boundary(detector, *, z_max=200.0, n_z=200000):
    """Largest ``z = lambda * dt`` with ``|R(z)| <= 1``, for the IMPLEMENTED step in float32.

    Applies one step to ``y' = -lambda y`` from ``y = 1`` with ``lambda = 1`` and ``dt = z``, so the
    returned value IS the amplification factor ``R(z)``. The boundary is the end of the contiguous
    stable interval starting at 0 (not merely the largest stable ``z`` anywhere).
    """
    z = jnp.linspace(0.0, float(z_max), int(n_z), dtype=jnp.float32)
    amplification = jax.jit(jax.vmap(
        lambda dt: detector.rkc2_step(lambda y: -y, jnp.asarray(1.0, jnp.float32), dt)
    ))(z)
    # R(0) = 1 exactly in exact arithmetic, so the test needs a few ulps of slack: a degree-s
    # Chebyshev recursion in float32 lands a hair either side of 1 near z = 0, and without the
    # slack the contiguous-interval scan can terminate at the very first sample.
    tolerance = 1.0 + 8.0 * float(np.finfo(np.float32).eps)
    stable = np.abs(np.asarray(amplification)) <= tolerance
    unstable = np.flatnonzero(~stable)
    edge = int(unstable[0]) if unstable.size > 0 else stable.size
    return float(np.asarray(z)[max(edge - 1, 0)])


def _worst_corner(detector, ranges, fractions, half_time, *, n_draws, n_rounds, n_extent, temperatures, seed):
    """Random search (with local refinement) for the parameter draw maximising ``|df/dx|``.

    Returns ``(lambda_max, parameters, fraction)``. ``|df/dx|`` is maximised over the whole reachable
    extent and every temperature, so the bound does not depend on how far a particular trajectory
    actually travels.
    """
    d_rate = jax.grad(lambda x, A0, B0, E0, T, p, log_k0_cat: detector._rate(x, A0, B0, E0, T, p, log_k0_cat))

    def decay(values, fraction):
        parameters = {name: values[i] for i, name in enumerate(PARAMETER_NAMES)}
        log_k0_cat = detector._calibrate(parameters, jnp.asarray(half_time, jnp.float32))
        A0, B0, E0 = detector._initial_state(fraction)
        extent = jnp.minimum(A0, B0) * jnp.linspace(0.0, 1.0 - 1e-4, n_extent, dtype=jnp.float32)
        grid = jax.vmap(lambda T: jax.vmap(lambda x: d_rate(x, A0, B0, E0, T, parameters, log_k0_cat))(extent))(temperatures)
        return jnp.max(jnp.abs(grid))

    search = jax.jit(jax.vmap(decay))
    rng = np.random.default_rng(seed)
    best = (0.0, None, None)
    for fraction in fractions:
        for round_index in range(n_rounds):
            draws = rng.uniform(ranges[:, 0], ranges[:, 1], size=(n_draws, len(PARAMETER_NAMES))).astype(np.float32)
            if round_index > 0 and best[1] is not None:  # refine locally around the worst found so far
                span = 0.12 * (ranges[:, 1] - ranges[:, 0])
                draws = np.clip(best[1] + rng.uniform(-span, span, draws.shape), ranges[:, 0], ranges[:, 1]).astype(np.float32)
            values = np.asarray(search(jnp.asarray(draws), jnp.full(n_draws, fraction, jnp.float32)))
            finite = np.where(np.isfinite(values), values, -np.inf)
            index = int(np.argmax(finite))
            if finite[index] > best[0]:
                best = (float(finite[index]), draws[index], float(fraction))
    return best


def check_stability(n_draws: int = 2048, n_rounds: int = 5, n_extent: int = 128, n_temperatures: int = 64,
                    safety: float = 0.5, stage_sweep: bool = False, seed: int = 0, **config):
    detector = detopt.detector.from_config(config["detector"])
    settings = dict(config["detector"]["enzyme"])
    device = jax.devices()[0]
    print(f"device: {device.platform}:{device.device_kind}  |  dtype float32  |  RKC2, "
          f"{detector.n_stages} stages, damping {detector.damping:.6g}")

    # --- 1. the scheme's boundary, measured on the implemented step
    boundary = measure_boundary(detector)
    print(f"\n1. measured real-axis stability boundary: |R(z)| <= 1 up to z = {boundary:.3f}")
    print(f"   (explicit Euler is 2.0; the RKC2 rule of thumb ~0.81 s^2 = {0.81 * detector.n_stages ** 2:.1f})")
    print(f"   per rate evaluation: {boundary / detector.n_stages:.3f} (Euler 2.0)")
    if stage_sweep:
        print("   stage sweep:")
        for s in (2, 3, 4, 5, 6, 8, 10):
            other = detopt.detector.EnzymeDetector(**dict(settings, n_stages=s))
            b = measure_boundary(other)
            print(f"     s={s:<3} boundary {b:8.3f}   per evaluation {b / s:6.3f}   vs 0.81 s^2 = {0.81 * s * s:7.2f}")

    # --- 2. the stiffest the prior gets
    ranges = np.array([settings["parameters"][name] for name in PARAMETER_NAMES], np.float64)
    low, high = detector.enzyme_fraction_bounds
    fractions = (low, 0.25, 0.5, 0.75, high)
    temperatures = jnp.linspace(*detector.temperature_bounds, int(n_temperatures), dtype=jnp.float32)
    half_time = detector.half_time_bounds[0]  # the stiffest end of the prior
    print(f"\n2. searching the prior for the largest |df/dx| ({n_draws} draws x {n_rounds} rounds x "
          f"{len(fractions)} fractions x {n_temperatures} temperatures, half_time = {half_time} h)...")
    lambda_max, values, fraction = _worst_corner(
        detector, ranges, fractions, half_time,
        n_draws=int(n_draws), n_rounds=int(n_rounds), n_extent=int(n_extent),
        temperatures=temperatures, seed=int(seed),
    )
    parameters = {name: float(v) for name, v in zip(PARAMETER_NAMES, values)}
    print(f"   |df/dx| = {lambda_max:.4g} /h   (enzyme fraction {fraction})")
    for name, value in parameters.items():
        print(f"     {name:<12} {value:>12.4g}" + (f"  ({np.exp(value):.4g} mM)" if name.startswith("log_") else ""))

    # --- 3. the step this allows
    dt_max = boundary / lambda_max
    dt = float(safety) * dt_max
    print(f"\n3. dt_max = boundary / |df/dx| = {dt_max:.4e} h ({dt_max * 3600:.3g} s)")
    print(f"   recommended dt = {safety:g} x dt_max = {dt:.4e} h ({dt * 3600:.3g} s)")
    print(f"     n_objective_steps       >= {int(np.ceil(detector.duration / dt))}")
    print(f"     n_steps_per_measurement >= {int(np.ceil(detector.duration / (detector.n_measurements * dt)))}")
    print(f"   configured: objective dt = {detector.objective_dt:.4e} h "
          f"(z = {detector.objective_dt * lambda_max:.3f}, {detector.objective_dt / dt_max:.3f} x dt_max), "
          f"measurement dt = {detector.measurement_dt:.4e} h "
          f"(z = {detector.measurement_dt * lambda_max:.3f}, {detector.measurement_dt / dt_max:.3f} x dt_max)")

    # --- confirm at the configured step
    tree = {name: jnp.asarray(value, jnp.float32) for name, value in parameters.items()}
    log_k0_cat = detector._calibrate(tree, jnp.asarray(half_time, jnp.float32))
    A0, B0, E0 = detector._initial_state(fraction)
    extent_max = float(jnp.minimum(A0, B0))
    decay_at = jax.jit(jax.vmap(lambda T: jnp.abs(jax.grad(
        lambda x: detector._rate(x, A0, B0, E0, T, tree, log_k0_cat))(jnp.asarray(0.0, jnp.float32)))))
    worst_temperature = temperatures[int(np.argmax(np.asarray(decay_at(temperatures))))]
    print(f"\n   integrating that corner at {float(worst_temperature):.1f} C (extent_max = {extent_max:.5f} mM):")
    verdict = 0
    for label, n_steps, n_intervals in (("objective", detector.n_objective_steps, 1),
                                        ("measurement", detector.n_steps_per_measurement, detector.n_measurements)):
        extents, error = jax.jit(
            lambda: detector._integrate(A0, B0, E0, worst_temperature, tree, log_k0_cat,
                                        n_steps=n_steps, n_intervals=n_intervals)
        )()
        final, error = float(extents[-1]), float(error)
        physical = np.isfinite(final) and -1e-6 <= final <= extent_max * (1.0 + 1e-3)
        within = np.isfinite(error) and error <= detector.integration_tolerance
        verdict |= 0 if (physical and within) else 1
        print(f"     {label:<12} final extent {final:>11.5f} / {extent_max:.5f} mM "
              f"{'OK' if physical else 'DIVERGED'} | half-step error {error:.3e} mM "
              f"{'OK' if within else f'> tol {detector.integration_tolerance:.1e}'}")
    print("\n" + ("STABLE: the configured step is inside the measured boundary everywhere the prior reaches."
                  if verdict == 0 else
                  "UNSTABLE at the configured step -- raise the step counts (or the stage count) above."))
    return {"boundary": boundary, "lambda_max": lambda_max, "dt_max": dt_max, "dt": dt,
            "parameters": parameters, "enzyme_fraction": fraction, "stable": verdict == 0}


if __name__ == "__main__":
    import gearup

    gearup.gearup(check_stability).with_config("config/bo.yaml")()
