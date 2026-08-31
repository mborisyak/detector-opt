#!/usr/bin/env python3
"""Verify the enzyme detector's RKC2 integration against scipy's LSODA.

An independent reference for the solver: the SAME rate expression
(:func:`detopt.detector.enzyme.kinetics`, evaluated in float64 with jax's x64 mode on) integrated by
``scipy.integrate.solve_ivp(method="LSODA")`` at tight tolerances, compared against the detector's
own float32 RKC2 result at the measurement times. LSODA is an adaptive, stiffness-switching
Adams/BDF solver, so it agrees with the fixed-step Chebyshev scheme only if the step size really is
inside both the stability boundary and the accuracy the guard claims.

What is compared, per drawn enzyme and per temperature: the extent of reaction at each of the
``n_measurements`` read-out times, and the conversion the objective mixture reaches after
``duration`` -- the latter being what the target is an argmax of, so an error there moves the label.

    python scripts/verify_lsoda.py =enzyme
    python scripts/verify_lsoda.py =enzyme n_draws=32 rtol=1e-11
"""

import numpy as np

import jax

# float64 for the reference RHS -- before any other jax use.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import detopt
from detopt.detector.enzyme import PARAMETER_NAMES, kinetics


def _reference(detector, parameters, log_k0_cat, temperature, A0, B0, E0, *, rtol, atol, times):
    """LSODA solution of ``dx/dt = rate(x)`` at ``times``, in float64.

    The RHS calls the model's own ``kinetics`` with float64 arguments, so this checks the SOLVER, not
    a re-derivation of the chemistry.
    """
    from scipy.integrate import solve_ivp

    tree = dict({name: float(value) for name, value in parameters.items()}, log_k0_cat=float(log_k0_cat))

    def rhs(_t, state):
        x = float(state[0])
        return [float(kinetics(A0 - x, B0 - x, x, x, E0, float(temperature), tree))]

    solution = solve_ivp(rhs, (0.0, float(times[-1])), [0.0], method="LSODA",
                         t_eval=np.asarray(times, np.float64), rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"LSODA failed: {solution.message}")
    return np.asarray(solution.y[0], np.float64)


def verify_lsoda(n_draws: int = 16, n_temperatures: int = 5, rtol: float = 1e-10, atol: float = 1e-14,
                 **config):
    detector = detopt.detector.from_config(config["detector"])
    print(f"RKC2 s={detector.n_stages} vs scipy LSODA (rtol={rtol:g}, atol={atol:g})")
    print(f"  measurement dt = {detector.measurement_dt:.4e} h, objective dt = {detector.objective_dt:.4e} h")

    times = np.asarray(detector.measurement_times, np.float64)
    # A spread of temperatures over the range where the enzyme is actually alive.
    temperatures = np.linspace(detector.temperature_bounds[0] + 10.0, 55.0, int(n_temperatures))

    worst_extent, worst_conversion = 0.0, 0.0
    rows = []
    for index in range(int(n_draws)):
        key_parameters, key_half_time, _ = jax.random.split(jax.random.PRNGKey(index), 3)
        parameters = detector._draw_parameters(key_parameters)
        low, high = detector.half_time_bounds
        half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=np.log(low), maxval=np.log(high)))
        log_k0_cat = detector._calibrate(parameters, half_time)

        for fraction, label, n_steps, n_intervals in (
            (detector.calibration_fraction, "measurement", detector.n_steps_per_measurement, detector.n_measurements),
            (detector.objective_fraction, "objective", detector.n_objective_steps, 1),
        ):
            A0, B0, E0 = (float(v) for v in detector._initial_state(fraction))
            for temperature in temperatures:
                ours, _ = detector._integrate(A0, B0, E0, float(temperature), parameters, log_k0_cat,
                                              n_steps=n_steps, n_intervals=n_intervals)
                ours = np.asarray(ours, np.float64)
                grid = times if n_intervals == detector.n_measurements else np.array([detector.duration])
                theirs = _reference(detector, parameters, log_k0_cat, temperature, A0, B0, E0,
                                    rtol=rtol, atol=atol, times=grid)
                extent_error = float(np.max(np.abs(ours - theirs)))
                extent_max = min(A0, B0)
                conversion_error = float(abs(ours[-1] - theirs[-1]) / extent_max)
                worst_extent = max(worst_extent, extent_error)
                worst_conversion = max(worst_conversion, conversion_error)
                rows.append((index, label, float(temperature), extent_error, conversion_error))

    print(f"\n  {len(rows)} comparisons ({n_draws} enzymes x {n_temperatures} temperatures x 2 mixtures)")
    extent = np.array([r[3] for r in rows])
    conversion = np.array([r[4] for r in rows])
    print(f"  |extent_RKC2 - extent_LSODA|:  median {np.median(extent):.3e}  p95 {np.percentile(extent, 95):.3e}  "
          f"max {extent.max():.3e} mM")
    print(f"  conversion difference:         median {np.median(conversion):.3e}  p95 "
          f"{np.percentile(conversion, 95):.3e}  max {conversion.max():.3e}")
    print(f"  for scale: read-out noise {detector.measurement_noise:g} mM, "
          f"integration_tolerance {detector.integration_tolerance:g} mM")

    worst = max(rows, key=lambda r: r[3])
    print(f"  worst: enzyme {worst[0]}, {worst[1]} mixture, {worst[2]:.1f} C -> {worst[3]:.3e} mM")
    verdict = worst_extent <= detector.integration_tolerance
    print("\n" + ("AGREES: RKC2 matches LSODA to within integration_tolerance everywhere tested."
                  if verdict else
                  f"DISAGREES: worst {worst_extent:.3e} mM exceeds integration_tolerance "
                  f"{detector.integration_tolerance:g} mM -- raise the step counts."))
    return {"worst_extent": worst_extent, "worst_conversion": worst_conversion, "agrees": bool(verdict)}


if __name__ == "__main__":
    import gearup

    gearup.gearup(verify_lsoda).with_config("config/root.yaml")()
