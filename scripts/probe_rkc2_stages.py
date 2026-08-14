#!/usr/bin/env python3
"""Does REDUCING the RKC2 stage count reduce the integration error AT THE FAILING CORNER?

Only the upward direction was ever tried (7 stages instead of 5, which did not help). This probe
sweeps ``n_stages`` DOWN as well, at the corner that actually fails, and states first what it must be
able to separate.

THE THREE HYPOTHESES, and what distinguishes them
-------------------------------------------------
``max|fine - coarse|`` at the corner is measured over ``n_stages`` x ``n_steps_per_measurement``:

  (a) the rounding lives in the STAGE ARITHMETIC -- the Chebyshev recursion evaluates the rate
      ``n_stages`` times per step, so the error would scale with ``n_stages * n_steps``. Prediction:
      at fixed ``n_steps``, the error falls ~40% going 5 -> 3 stages and rises ~40% going 5 -> 7,
      i.e. a factor ~3.5 across the row 2 -> 7.
  (b) the rounding lives in the EXTENT ACCUMULATOR -- ONE addition per STEP into a growing total,
      which is where the earlier diagnosis pointed. Prediction: the error scales with the step count
      and is INDEPENDENT of ``n_stages``, i.e. every row is flat.
  (c) TRUNCATION still dominates at the corner. Prediction: the error falls monotonically with more
      steps. Note the step-count behaviour was only ever measured at the GOOD design -- never here.

CAN THIS PROBE SEPARATE THEM? Yes, and the reason is that the measurement carries NO SAMPLING NOISE:
the enzymes are seeded from the event index alone, so every cell of the table sees the identical
population and two cells differ only by the integration. (a) predicts a ~3.5x spread along each row,
(b) predicts a flat row, and (c) predicts a monotone decrease along each column; those are three
distinguishable shapes, and the smallest of the effects (a) predicts is ~40%, far above anything the
determinism of the measurement could fake.

THE CONFOUND, and how it is separated
-------------------------------------
RKC2's real-axis stability region grows like ``s^2``, so 5 -> 3 stages SHRINKS it ~2.7x and an
unstable solve also shows a large ``max|fine - coarse|``. Three separations are reported per cell
rather than one number that could be either:

  * the MEASURED float32 real-axis boundary ``z_max(s)`` against ``z = dt * max|d(rate)/d(extent)|``
    at this corner over the drawn enzymes -- an a-priori check;
  * the STEP DEPENDENCE, which runs OPPOSITE ways for the two causes: ``dt`` shrinks as steps grow,
    so instability DISAPPEARS with more steps while rounding accumulates and GROWS with them;
  * the health of the clean solution itself -- finite, inside ``[0, A0]``, and monotone in time,
    which an unstable chain is not.

    srun --cpus-per-task=4 -u python -u scripts/probe_rkc2_stages.py
"""
import argparse
import os


_allocated = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, _allocated)

import math

import numpy as np

import jax
import jax.numpy as jnp

from detopt.detector.enzyme_mm import _to_zero_anchor, rkc2_step
from validate_mm import build, corner_designs, guard_error

CONFIG = "config/detector/enzyme_mm_sym.yaml"
CORNER = "HHHH"  # enzyme_fraction 1.0, 35 C, [A]0 = [B]0 = 10 mM -- the design that fires


def stability_boundary(detector):
  """The MEASURED float32 real-axis stability boundary of the IMPLEMENTED step, for this detector's
  own ``n_stages``. ``y' = -y`` with ``dt = z``, so one step returns the amplification ``R(z)``."""
  z = np.linspace(1.0e-3, 120.0, 240000, dtype=np.float32)
  amplification = jax.vmap(
    lambda t: rkc2_step(detector._rkc2, lambda y: -y, jnp.ones((), jnp.float32), t)
  )(jnp.asarray(z))
  unstable = np.flatnonzero(np.abs(np.asarray(amplification)) > 1.0 + 1.0e-5)
  return float(z[unstable[0] - 1]) if len(unstable) > 0 else float(z[-1])


def _draw(detector, event_index):
  """The SAME enzyme ``_event`` draws at ``event_index``: parameters, half-time, calibrated turnover."""
  key_parameters, key_half_time, _ = jax.random.split(jax.random.PRNGKey(event_index), 3)
  parameters = detector._draw_parameters(key_parameters)
  low, high = detector.half_time_bounds
  half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=math.log(low), maxval=math.log(high)))
  return parameters, half_time


def corner_stiffness(detector, design, n_enzymes):
  """``max |d(rate)/d(extent)|`` at this corner over the drawn enzymes and the whole extent range.

  The stiffness the STABILITY boundary has to cover. A property of the corner and the population, so
  it does not depend on ``n_stages`` or on the step count -- only ``z = dt * this`` does."""
  fraction, temperature, initial_A, initial_B = (
    np.asarray(block, np.float64).reshape(-1) for block in detector._split(jnp.asarray(design))
  )
  # Every experiment of a pure corner design is the same experiment; take the first.
  f, t, A0, B0 = float(fraction[0]), float(temperature[0]), float(initial_A[0]), float(initial_B[0])

  def worst(event_index):
    parameters, half_time = _draw(detector, event_index)
    kinetic = _to_zero_anchor(parameters)
    log_k0_cat = detector._calibrate(kinetic, half_time)
    extent = jnp.linspace(0.0, min(A0, B0), 129)
    slope = jax.vmap(jax.grad(
      lambda x: detector._rate(x, A0, B0, detector.concentration_E * f, t, kinetic, log_k0_cat)
    ))(extent)
    return jnp.max(jnp.abs(slope))

  indices = jnp.arange(n_enzymes, dtype=jnp.int32)
  return float(jnp.max(jax.jit(jax.vmap(worst))(indices)))


def solution_health(detector, design, n_enzymes):
  """Is the CLEAN solve at this corner a solution at all? ``(finite, inside [0, A0], monotone)``.

  An unstable chain fails these; a chain limited by rounding passes them all while still disagreeing
  with its own ``dt/2`` twin at the 1e-2 level. This is the a-posteriori half of the instability
  separation."""
  initial_A = float(np.asarray(detector._split(jnp.asarray(design))[2], np.float64).reshape(-1)[0])

  def one(event_index):
    parameters, half_time = _draw(detector, event_index)
    return detector.readout(design, parameters, half_time)

  clean = np.asarray(jax.jit(jax.vmap(one))(jnp.arange(n_enzymes, dtype=jnp.int32)), np.float64)
  finite = np.isfinite(clean)
  # Guard the range/monotonicity tests against NaN, which compares False both ways.
  safe = np.where(finite, clean, 0.0)
  inside = np.logical_and(safe >= -1.0e-3, safe <= initial_A + 1.0e-3)
  monotone = np.diff(safe, axis=-1) <= 1.0e-4  # [A] falls as the extent grows
  return float(finite.mean()), float(np.logical_and(inside, finite).mean()), float(monotone.mean())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=CONFIG)
  parser.add_argument("--stages", type=int, nargs="*", default=[2, 3, 5, 7])
  parser.add_argument("--steps", type=int, nargs="*", default=[80, 160, 320, 640])
  parser.add_argument("--events", type=int, nargs="*", default=[16384, 65536])
  parser.add_argument("--deep-events", type=int, default=262144,
                      help="re-measure the cells that PASS at the largest sweep count at this count")
  parser.add_argument("--n-deep", type=int, default=5, help="how many passing cells to re-measure")
  parser.add_argument("--tolerance", type=float, nargs="*", default=[1.25e-2, 2.5e-2],
                      help="thresholds to report margins against (25%% of noise 0.05 and 0.10)")
  parser.add_argument("--n-diagnostic-enzymes", type=int, default=256)
  arguments = parser.parse_args()

  shipped = build(arguments.config)
  design = corner_designs(shipped)[CORNER]
  fraction, temperature, initial_A, initial_B = (
    np.asarray(block, np.float64).reshape(-1) for block in shipped._split(jnp.asarray(design))
  )
  print("=" * 110)
  print(f"RKC2 STAGE PROBE at the FAILING CORNER {CORNER}: "
        f"(fraction {fraction[0]:g}, {temperature[0]:g} C, [A]0 {initial_A[0]:g}, [B]0 {initial_B[0]:g}) mM")
  print("=" * 110)
  print(f"  shipped: n_stages {shipped.n_stages}, n_steps_per_measurement {shipped.n_steps_per_measurement}, "
        f"measurement_noise {shipped.measurement_noise}, integration_tolerance "
        f"{shipped.integration_tolerance:.3e}")
  print(f"  thresholds: " + "  ".join(f"{value:.3e}" for value in arguments.tolerance))
  print(f"  the enzymes are seeded from the event index alone, so every cell below sees the IDENTICAL")
  print(f"  population and two cells differ only by the integration -- there is no sampling noise.")

  # -------------------------------------------------------------------------------------------- #
  # The stability confound, measured BEFORE the sweep so the table can be read against it.
  # -------------------------------------------------------------------------------------------- #
  print(f"\n  STABILITY (the confound): boundary MEASURED in float32 on the implemented step; the")
  print(f"  stiffness is a property of the corner and the population, so only z = dt * |df/dx| moves.")
  stiffness = corner_stiffness(shipped, design, arguments.n_diagnostic_enzymes)
  print(f"    worst |d(rate)/d(extent)| over {arguments.n_diagnostic_enzymes} drawn enzymes: "
        f"{stiffness:.4g} /h")
  boundary = {}
  for n_stages in arguments.stages:
    probe = build(arguments.config, n_stages=int(n_stages), integration_tolerance=1.0e9)
    boundary[n_stages] = stability_boundary(probe)
    print(f"    n_stages {n_stages}: real-axis boundary z_max = {boundary[n_stages]:.3f}")
  print(f"\n    {'n_steps':>8s} {'dt (h)':>12s} " +
        "".join(f"{'z/z_max s=' + str(s):>14s}" for s in arguments.stages))
  for n_steps in arguments.steps:
    dt = shipped.duration / (shipped.n_measurements * int(n_steps))
    print(f"    {n_steps:8d} {dt:12.4e} " +
          "".join(f"{dt * stiffness / boundary[s]:14.4f}" for s in arguments.stages))
  print(f"    (> 1 would be UNSTABLE for the dt chain; the dt/2 chain sits at half of it)")

  # -------------------------------------------------------------------------------------------- #
  # The sweep.
  # -------------------------------------------------------------------------------------------- #
  errors, health = {}, {}
  for count in arguments.events:
    print(f"\n  max|fine - coarse| (mM) at {count} events" + " " * 6 +
          "rows = n_stages, columns = n_steps_per_measurement")
    print(f"    {'stages':>8s} " + "".join(f"{n:>14d}" for n in arguments.steps))
    for n_stages in arguments.stages:
      row = []
      for n_steps in arguments.steps:
        probe = build(arguments.config, n_stages=int(n_stages), n_steps_per_measurement=int(n_steps),
                      integration_tolerance=1.0e9)
        errors[(n_stages, n_steps, count)] = guard_error(probe, design, count)
        if (n_stages, n_steps) not in health:
          health[(n_stages, n_steps)] = solution_health(probe, design, arguments.n_diagnostic_enzymes)
        row.append(errors[(n_stages, n_steps, count)])
      print(f"    {n_stages:8d} " + "".join(f"{value:14.3e}" for value in row), flush=True)

  for value in arguments.tolerance:
    print(f"\n  margin = {value:.3e} / error at {arguments.events[-1]} events   (< 1 FIRES)")
    print(f"    {'stages':>8s} " + "".join(f"{n:>14d}" for n in arguments.steps))
    for n_stages in arguments.stages:
      print(f"    {n_stages:8d} " + "".join(
        f"{value / errors[(n_stages, n, arguments.events[-1])]:14.2f}" for n in arguments.steps))

  print(f"\n  SOLUTION HEALTH of the clean solve ({arguments.n_diagnostic_enzymes} enzymes): "
        f"fraction finite / inside [0, A0] / monotone in time")
  print(f"    {'stages':>8s} " + "".join(f"{n:>22d}" for n in arguments.steps))
  for n_stages in arguments.stages:
    print(f"    {n_stages:8d} " + "".join(
      "{:>22s}".format("{:.3f}/{:.3f}/{:.3f}".format(*health[(n_stages, n)])) for n in arguments.steps))

  # -------------------------------------------------------------------------------------------- #
  # Scaling, read off the table rather than eyeballed.
  # -------------------------------------------------------------------------------------------- #
  count = arguments.events[-1]
  print(f"\n  SCALING at {count} events, each row normalised to its own n_stages = "
        f"{shipped.n_stages} value:")
  print(f"    (a) stage arithmetic predicts these track n_stages / {shipped.n_stages};  "
        f"(b) extent accumulator predicts 1.000 everywhere)")
  print(f"    {'stages':>8s} {'n_stages ratio':>16s} " + "".join(f"{n:>14d}" for n in arguments.steps))
  for n_stages in arguments.stages:
    print(f"    {n_stages:8d} {n_stages / shipped.n_stages:16.3f} " + "".join(
      f"{errors[(n_stages, n, count)] / errors[(shipped.n_stages, n, count)]:14.3f}"
      for n in arguments.steps))

  print(f"\n  STEP DEPENDENCE at {count} events, each row normalised to its own "
        f"n_steps = {shipped.n_steps_per_measurement} value:")
  print(f"    (c) truncation predicts these FALL along the row; rounding accumulation predicts they RISE;")
  print(f"     instability predicts a LARGE value at the coarsest step that collapses as steps grow)")
  print(f"    {'stages':>8s} " + "".join(f"{n:>14d}" for n in arguments.steps))
  for n_stages in arguments.stages:
    reference = errors[(n_stages, shipped.n_steps_per_measurement, count)]
    print(f"    {n_stages:8d} " + "".join(
      f"{errors[(n_stages, n, count)] / reference:14.3f}" for n in arguments.steps))

  # -------------------------------------------------------------------------------------------- #
  # Anything that looks good is re-measured DEEPER: the guard is a max over events.
  # -------------------------------------------------------------------------------------------- #
  threshold = arguments.tolerance[0]
  passing = sorted(
    [(errors[(s, n, count)], s, n) for s in arguments.stages for n in arguments.steps
     if errors[(s, n, count)] <= threshold],
    key=lambda entry: entry[0]
  )[:max(0, arguments.n_deep)]
  shipped_cell = (shipped.n_stages, shipped.n_steps_per_measurement)
  if shipped_cell not in [(s, n) for _, s, n in passing]:
    passing.append((errors[(shipped_cell[0], shipped_cell[1], count)], *shipped_cell))
  print(f"\n  RE-MEASURED at {arguments.deep_events} events -- the cells that clear {threshold:.3e} at "
        f"{count}, plus the shipped one.")
  print(f"  A margin at {count} is not a margin at campaign scale (~2e6 events per run).")
  print(f"    {'stages':>8s} {'steps':>8s} {'@' + str(count):>14s} {'@' + str(arguments.deep_events):>14s} "
        + "".join(f"{'margin @' + format(v, '.2e'):>18s}" for v in arguments.tolerance))
  for _, n_stages, n_steps in passing:
    probe = build(arguments.config, n_stages=int(n_stages), n_steps_per_measurement=int(n_steps),
                  integration_tolerance=1.0e9)
    deep = guard_error(probe, design, arguments.deep_events)
    print(f"    {n_stages:8d} {n_steps:8d} {errors[(n_stages, n_steps, count)]:14.3e} {deep:14.3e} "
          + "".join(f"{value / deep:18.2f}" for value in arguments.tolerance), flush=True)


if __name__ == "__main__":
  main()
