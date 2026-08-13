#!/usr/bin/env python3
"""Validate the `enzyme_mm` candidate BEFORE screening it: build, integrator, identifiability, anchors.

Everything `docs/benchmark-acceptance.md` section 4 asks for up to (but not including) the landscape,
which `scripts/screen_task.py` owns. Each mode states what it must be able to separate and then
measures it; nothing here is a tick-box.

    build       builds, simulates, read-outs finite, targets inside their priors, the MEASURED
                no-information ceiling, the folded fraction over the box x the unfolding prior (the
                melt must never enter), and the calibrated k_cat distribution against the literature
                band -- the check that `concentration_E` carries a physical enzyme loading.
    stability   the RKC2 real-axis boundary of the IMPLEMENTED step, in float32 on this device, and the
                worst |df/dx| the design box x prior can reach -> the margin z = dt |df/dx|.
    steps       the step count, RE-MEASURED for this rate law. Two tables: the solve's own dt/dt-half
                guard, and the TRUE error of the shipped chain against an INDEPENDENT float64 numpy
                implementation of the same scheme. They disagree, and only the second is right.
    guard       the dt/dt-half assertion at SCREENING SCALE, over the designs a screen actually reaches:
                the known designs, EVERY CORNER of the box, and a random sample. The guard is a MAX over
                sampled events, so it GROWS with the event count and a margin measured at 4096 does not
                transfer to the 16384 a screen runs -- which is the whole reason this mode exists
                separately from `steps` (whose 8 uniform random designs sit nowhere near the worst).
    fim         identifiability. Fisher information of the measured [A](t) w.r.t. every drawn
                parameter, marginalised over the others with their own prior, on designs written down
                from the prior alone. Chooses the target from what is measurable rather than desirable.
    calib       KNOWN-GOOD vs KNOWN-BAD vs DEAD through the same proxy the screen uses. If the design
                the physics says is good does not score clearly better, the target is not in the data.

    srun --cpus-per-task=4 python -u scripts/validate_mm.py --mode all
"""
import argparse
import itertools
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# BLAS/OpenMP default to every core on the machine, which is wrong under a scheduler. Set before numpy.
_allocated = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, _allocated)

import math

import numpy as np

import jax
import jax.numpy as jnp

import detopt.detector
import detopt.utils.config
from detopt.detector.enzyme import ZERO_CELSIUS
from detopt.detector.enzyme_mm import ANCHOR_SHIFT, PARAMETER_NAMES, rkc2_step

CONFIG = "config/detector/enzyme_mm.yaml"


def build(config_path, n_experiments=None, **overrides):
  config = detopt.utils.config.load_config(config_path)
  (name,) = [k for k in config if isinstance(config[k], dict)]
  entry = dict(config[name], **overrides)
  if n_experiments is not None:
    entry["n_experiments"] = int(n_experiments)
  return detopt.detector.from_config({name: entry})


# --------------------------------------------------------------------------------------------- #
# Designs written down FROM THE PRIOR ALONE -- never from a landscape scan or a finished run.
# --------------------------------------------------------------------------------------------- #
def known_designs(detector):
  """`{name: flat NOMINAL design}` for the calibration checks.

  Four designs, all written down from the prior and the instrument alone; each is documented where it
  is built. They must separate as GOOD < FLAT < BAD ~ DEAD ~ ceiling -- if they do not, the target is
  not in the data.
  """
  m = detector.n_experiments
  ranges = dict(detector.parameter_ranges)
  K_A = (math.exp(ranges["log_K_A"][0]), math.exp(ranges["log_K_A"][1]))
  K_B = (math.exp(ranges["log_K_B"][0]), math.exp(ranges["log_K_B"][1]))
  temperature_low, temperature_high = detector.temperature_bounds
  fraction_low, fraction_high = detector.enzyme_fraction_bounds
  B_low, B_high = detector.concentration_B_bounds
  designs_glucose = detector.concentration_A_bounds is not None

  def pack(fraction, temperature, initial_A, initial_B):
    blocks = [fraction, temperature] + ([initial_A] if designs_glucose else []) + [initial_B]
    return np.concatenate(blocks).astype(np.float32)

  middle = math.sqrt(B_low * B_high)
  if designs_glucose:
    A_low, A_high = detector.concentration_A_bounds
    good_A = np.clip(np.geomspace(K_A[0], 3.0 * K_A[1], m), A_low, A_high)
    bad_A, dead_A, flat_A = np.full(m, A_low), np.full(m, A_high), np.full(m, math.sqrt(A_low * A_high))
  else:
    good_A = bad_A = dead_A = flat_A = np.full(m, float(detector.concentration_A))

  designs = {
    # GOOD: the classical prescription -- the varied substrate STRADDLES its own K_M prior (the whole
    # box, sub-saturating to saturating), temperatures spread over the whole box for the Arrhenius
    # slope, enzyme loadings spread so k_cat * E0 is not degenerate.
    "GOOD": pack(np.geomspace(0.05, 0.5, m), np.linspace(temperature_low, temperature_high, m), good_A,
                 np.geomspace(B_low, B_high, m)),
    # FLAT: every experiment IDENTICAL, at the geometric middle of every range. It reacts -- there is
    # signal -- but there is no contrast, so nothing separates K_M from the turnover.
    "FLAT": pack(np.full(m, math.sqrt(fraction_low * fraction_high)),
                 np.full(m, 0.5 * (temperature_low + temperature_high)), flat_A, np.full(m, middle)),
    # BAD: every experiment identical, the least substrate, the least enzyme, the coldest. No saturation
    # curve, no temperature lever, and almost nothing reacts.
    "BAD": pack(np.full(m, fraction_low), np.full(m, temperature_low), bad_A, np.full(m, B_low)),
    # DEAD: the deliberately uninformative design of acceptance section 4.3 -- the least enzyme on the
    # MOST substrate, coldest, so nothing measurable is converted within `duration` and every read-out
    # is just its own [A]0 plus noise.
    "DEAD": pack(np.full(m, fraction_low), np.full(m, temperature_low), dead_A, np.full(m, B_high)),
  }
  return designs


def random_designs(detector, n, seed):
  """`n` uniform designs in the SCALED cube, returned in NOMINAL units."""
  rng = np.random.default_rng(seed)
  scaled = rng.random((n, int(detector.design_dim()))).astype(np.float32)
  return np.asarray(jax.vmap(detector._to_nominal_flat)(jnp.asarray(scaled)), np.float32)


def corner_designs(detector):
  """`{name: flat NOMINAL design}` for EVERY CORNER of the design box -- all coordinates at a bound.

  The box has `n_fields * m` coordinates, so it has `2**(n_fields * m)` corners (4096 at the shipped
  4 fields x 3 experiments), which is not what is enumerated here and does not need to be. The guard
  is a max over the experiments of a PER-EXPERIMENT quantity -- `_batch` vmaps `_integrate` over the
  experiments and returns `jnp.max(errors)` -- and every design is scored against the same events, so

      max over all 2**(n_fields * m) corner designs  ==  max over the 2**n_fields per-experiment corners

  exactly. The `2**n_fields` designs built here put EVERY experiment at the same corner, so each is
  itself a corner of the full box and their max is that exact maximum. The name is one letter per
  field in `_design_layout` order, `L` at the lower bound and `H` at the upper.
  """
  m = detector.n_experiments
  bounds = [pair for _, pair, _ in detector._design_layout]
  designs = {}
  for choice in itertools.product((0, 1), repeat=len(bounds)):
    name = ''.join('H' if side == 1 else 'L' for side in choice)
    designs[name] = np.concatenate(
      [np.full(m, bounds[k][side], np.float32) for k, side in enumerate(choice)]
    ).astype(np.float32)
  return designs


# --------------------------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------------------------- #
def mode_build(detector, arguments):
  print("=" * 100)
  print(f"BUILD  m={detector.n_experiments}  design_dim={detector.design_dim()}  "
        f"target={detector.target_parameters}")
  print("=" * 100)

  designs = known_designs(detector)
  index = np.arange(arguments.n_events, dtype=np.int64)
  for name, design in designs.items():
    ground_truth, event, mask, target = detector(design, index)
    measurements = np.asarray(event.measurements)
    normalised = np.asarray(detector.normalize_target(target))
    finite = bool(np.all(np.isfinite(measurements)) and np.all(np.isfinite(normalised)))
    inside = bool(np.all(np.abs(normalised) <= 1.0 + 1e-5))
    print(f"  {name:5s}: readout finite {finite}, targets inside their priors {inside}, "
          f"|A| range [{measurements.min():.4f}, {measurements.max():.4f}] mM, "
          f"normalised target variance {float(np.mean(np.var(normalised, axis=0))):.4f}")
    if not finite or not inside:
      raise SystemExit(f"validate_mm: {name} produced a non-finite read-out or an out-of-prior target")

  # The MEASURED no-information ceiling: the variance of the normalised target over the population,
  # which is what the best design-independent predictor scores. Uniform prior -> 1/3.
  ground_truth, _, _, target = detector(designs["GOOD"], np.arange(arguments.n_events, dtype=np.int64))
  normalised = np.asarray(detector.normalize_target(target))
  ceiling = float(np.mean(np.square(normalised - normalised.mean(axis=0))))
  print(f"\n  MEASURED no-information ceiling (variance of the normalised target): {ceiling:.4f}   "
        f"(uniform prior -> 0.3333)")

  # The folded fraction over the box x the unfolding prior: the melt must NEVER enter.
  print(f"  worst folded fraction over temperature_bounds x prior: {detector._worst_folded_fraction():.4f}   "
        f"(config requires > {detector.min_folded_fraction})")

  # The calibrated turnover, in physical units: the check that `concentration_E` carries a realistic
  # enzyme loading rather than absorbing the whole k_cat * E0 product.
  parameters = np.asarray(ground_truth.parameters)
  half_time = np.asarray(ground_truth.half_time)[:, 0]
  q10_cat = parameters[:, PARAMETER_NAMES.index("Q10_cat")]

  def calibrate(row, tau):
    drawn = {name: jnp.asarray(row[k], jnp.float32) for k, name in enumerate(PARAMETER_NAMES)}
    from detopt.detector.enzyme_mm import _to_zero_anchor
    return detector._calibrate(_to_zero_anchor(drawn), jnp.asarray(tau, jnp.float32))

  log_k0_cat = np.asarray(jax.jit(jax.vmap(calibrate))(jnp.asarray(parameters), jnp.asarray(half_time)))
  # k_cat at the 25 C anchor, per hour -> per second.
  k_cat = np.exp(log_k0_cat + np.log(q10_cat) * ANCHOR_SHIFT) / 3600.0
  quantiles = np.percentile(k_cat, [5, 50, 95])
  print(f"  calibrated k_cat at 25 C: median {quantiles[1]:.1f} /s, 5-95% {quantiles[0]:.1f}-{quantiles[2]:.1f} /s"
        f"   (hexokinase 30-300 /s)")
  return {"ceiling": ceiling}


# --------------------------------------------------------------------------------------------- #
# stability
# --------------------------------------------------------------------------------------------- #
def mode_stability(detector, arguments):
  print("=" * 100)
  print("STABILITY  (the boundary is MEASURED on the implemented step, in float32, on this device)")
  print("=" * 100)

  # y' = -lambda y with lambda = 1 and dt = z, so one step returns the amplification R(z) directly. The
  # sweep starts strictly above 0 (at z = 0 the step is the identity and float32 rounding can put |R|
  # a hair above 1, which would report the sweep's own top as the boundary) and a 1e-5 tolerance keeps
  # roundoff from ending it early.
  z = np.linspace(1.0e-3, 60.0, 120000, dtype=np.float32)
  amplification = jax.vmap(lambda t: rkc2_step(detector._rkc2, lambda y: -y, jnp.ones((), jnp.float32), t))(
    jnp.asarray(z)
  )
  unstable = np.flatnonzero(np.abs(np.asarray(amplification)) > 1.0 + 1.0e-5)
  boundary = float(z[unstable[0] - 1]) if len(unstable) > 0 else float(z[-1])
  print(f"  RKC2 s={detector.n_stages} real-axis boundary, float32: z = {boundary:.3f}")

  # The stiffest the design box x prior gets: |d(rate)/d(extent)|, over random designs, random enzymes
  # and the whole extent range each one traverses.
  designs = random_designs(detector, arguments.n_stability_designs, arguments.seed)
  keys = jax.random.split(jax.random.PRNGKey(arguments.seed + 1), arguments.n_stability_enzymes)
  from detopt.detector.enzyme_mm import _to_zero_anchor

  def worst_slope(design, key):
    drawn = detector._draw_parameters(key)
    half_time = jnp.asarray(detector.half_time_bounds[0], jnp.float32)  # the fastest enzyme: stiffest
    kinetic = _to_zero_anchor(drawn)
    log_k0_cat = detector._calibrate(kinetic, half_time)
    fraction, temperature, initial_A, initial_B = detector._split(design)

    def per_experiment(f, t, A0, B0):
      extent = jnp.linspace(0.0, jnp.minimum(A0, B0), 65)
      slope = jax.vmap(jax.grad(
        lambda x: detector._rate(x, A0, B0, detector.concentration_E * f, t, kinetic, log_k0_cat)
      ))(extent)
      return jnp.max(jnp.abs(slope))

    return jnp.max(jax.vmap(per_experiment)(fraction, temperature, initial_A, initial_B))

  slopes = jax.jit(jax.vmap(jax.vmap(worst_slope, in_axes=(None, 0)), in_axes=(0, None)))(
    jnp.asarray(designs), keys
  )
  worst = float(jnp.max(slopes))
  dt = detector.measurement_dt
  print(f"  worst |df/dx| over {len(designs)} designs x {len(keys)} enzymes: {worst:.1f} /h")
  print(f"  dt = {dt:.4e} h  ->  z = dt |df/dx| = {dt * worst:.4f}   "
        f"margin {boundary / (dt * worst):.0f}x  (accuracy, not stability, sets the step)")
  return {"boundary": boundary, "worst_slope": worst}


# --------------------------------------------------------------------------------------------- #
# steps -- the INDEPENDENT float64 reference
# --------------------------------------------------------------------------------------------- #
def rkc2_coefficients_numpy(s, damping):
  """Verwer's RKC2 coefficients, re-derived in float64 numpy.

  Deliberately a SECOND transcription of the scheme rather than a call into
  `detopt.detector.enzyme.rkc2_coefficients`: the point of the reference is that a transcription error
  in the shipped version shows up as a mismatch instead of cancelling.
  """
  omega_0 = 1.0 + damping / (s * s)
  T = np.zeros(s + 1); dT = np.zeros(s + 1); ddT = np.zeros(s + 1)
  T[0], T[1] = 1.0, omega_0
  dT[0], dT[1] = 0.0, 1.0
  for j in range(2, s + 1):
    T[j] = 2.0 * omega_0 * T[j - 1] - T[j - 2]
    dT[j] = 2.0 * T[j - 1] + 2.0 * omega_0 * dT[j - 1] - dT[j - 2]
    ddT[j] = 4.0 * dT[j - 1] + 2.0 * omega_0 * ddT[j - 1] - ddT[j - 2]
  omega_1 = dT[s] / ddT[s]
  b = np.zeros(s + 1)
  for j in range(2, s + 1):
    b[j] = ddT[j] / (dT[j] * dT[j])
  b[0] = b[1] = b[2]
  a = 1.0 - b * T
  rows = [(2.0 * b[j] * omega_0 / b[j - 1], -b[j] / b[j - 2], 2.0 * b[j] * omega_1 / b[j - 1],
           -a[j - 1] * 2.0 * b[j] * omega_1 / b[j - 1]) for j in range(2, s + 1)]
  return b[1] * omega_1, np.asarray(rows, np.float64)


def mode_steps(detector, arguments):
  print("=" * 100)
  print("STEPS  (the step count is RE-MEASURED for this rate law, not inherited)")
  print("=" * 100)

  from detopt.detector.enzyme_mm import _to_zero_anchor

  # One flat population of independent (experiment, enzyme) problems, so the numpy reference runs
  # vectorised over all of them at once.
  designs = random_designs(detector, arguments.n_step_designs, arguments.seed + 7)
  keys = jax.random.split(jax.random.PRNGKey(arguments.seed + 8), arguments.n_step_enzymes)
  m = detector.n_experiments

  drawn = jax.jit(jax.vmap(detector._draw_parameters))(keys)
  half_time = jnp.exp(jax.random.uniform(
    jax.random.PRNGKey(arguments.seed + 9), (len(keys),),
    minval=math.log(detector.half_time_bounds[0]), maxval=math.log(detector.half_time_bounds[1])
  ))
  kinetic = _to_zero_anchor(drawn)
  log_k0_cat = np.asarray(jax.jit(jax.vmap(detector._calibrate))(kinetic, half_time), np.float64)
  kinetic = {name: np.asarray(value, np.float64) for name, value in kinetic.items()}

  # Cross every design-experiment with every enzyme -> a flat lane index.
  n_lanes = len(designs) * m * len(keys)
  split = [np.asarray(block, np.float64) for block in detector._split(jnp.asarray(designs))]
  fraction, temperature, initial_A, initial_B = (
    np.repeat(block.reshape(-1), len(keys)) for block in split
  )
  lane = {name: np.tile(value, len(designs) * m) for name, value in kinetic.items()}
  lane_log_k0_cat = np.tile(log_k0_cat, len(designs) * m)
  E0 = detector.concentration_E * fraction

  def vant_hoff_numpy(T, log_K_0, Q10):
    span = 1.0 / ZERO_CELSIUS - 1.0 / (10.0 + ZERO_CELSIUS)
    delta = 1.0 / ZERO_CELSIUS - 1.0 / (T + ZERO_CELSIUS)
    return np.exp(log_K_0 + np.log(Q10) * delta / span)

  def gibbs_numpy(T, delta_H, delta_C, T_melting):
    Tk, Tm = T + ZERO_CELSIUS, T_melting + ZERO_CELSIUS
    dG = delta_H * (1.0 - Tk / Tm) - delta_C * ((Tm - Tk) - Tk * np.log(Tm / Tk))
    return 1.0 / (1.0 + np.exp(-dG / Tk))

  def rate_numpy(x):
    A, B, C, D = initial_A - x, initial_B - x, x, x
    K_A = vant_hoff_numpy(temperature, lane["log_K0_A"], lane["Q10_A"])
    K_B = vant_hoff_numpy(temperature, lane["log_K0_B"], lane["Q10_B"])
    Ki_C = vant_hoff_numpy(temperature, lane["log_K0i_C"], lane["Q10_C"])
    Ki_D = vant_hoff_numpy(temperature, lane["log_K0i_D"], lane["Q10_D"])
    k_cat = vant_hoff_numpy(temperature, lane_log_k0_cat, lane["Q10_cat"])
    active = gibbs_numpy(temperature, lane["delta_H"], lane["delta_C"], lane["T_melting"])
    return k_cat * active * E0 * A * B / (A + K_A * (1.0 + C / Ki_C)) / (B + K_B * (1.0 + D / Ki_D))

  n_reference = arguments.n_reference_steps
  coefficients = rkc2_coefficients_numpy(detector.n_stages, detector.damping)
  dt_reference = detector.duration / (detector.n_measurements * n_reference)
  print(f"  float64 numpy reference at {n_reference} steps per read-out interval, {n_lanes} lanes ...",
        flush=True)
  x = np.zeros(n_lanes, np.float64)
  mu_tilde_1, rows = coefficients
  reference = []
  for _ in range(detector.n_measurements):
    for _ in range(n_reference):
      slope_0 = rate_numpy(x)
      previous, current = x, x + mu_tilde_1 * dt_reference * slope_0
      for mu, nu, mu_tilde, gamma_tilde in rows:
        following = ((1.0 - mu - nu) * x + mu * current + nu * previous
                     + mu_tilde * dt_reference * rate_numpy(current) + gamma_tilde * dt_reference * slope_0)
        previous, current = current, following
      x = current
    reference.append(x.copy())
  reference = np.stack(reference)  # (n_measurements, n_lanes)

  print(f"  {'n_steps':>8s} {'dt (h)':>12s} {'guard |fine-coarse|':>22s} {'TRUE |shipped-ref64|':>22s}")
  results = {}
  for n_steps in arguments.step_scan:
    probe = build(arguments.config, detector.n_experiments, n_steps_per_measurement=int(n_steps),
                  integration_tolerance=1.0e9)

    # The shipped float32 solver on the SAME lanes, at the SAME calibrated enzymes -- so the only thing
    # that differs from the reference is the integration.
    def shipped(f, t, A0, B0, kin, cat):
      extent, error = probe._integrate(A0, B0, probe.concentration_E * f, t, kin, cat)
      return A0 - extent, error

    lane_kinetic = {name: jnp.asarray(value, jnp.float32) for name, value in lane.items()}
    concentration, errors = jax.jit(jax.vmap(shipped))(
      jnp.asarray(fraction, jnp.float32), jnp.asarray(temperature, jnp.float32),
      jnp.asarray(initial_A, jnp.float32), jnp.asarray(initial_B, jnp.float32),
      lane_kinetic, jnp.asarray(lane_log_k0_cat, jnp.float32)
    )
    extent = initial_A[None, :] - np.asarray(concentration, np.float64).T  # (n_measurements, n_lanes)
    true_error = float(np.max(np.abs(extent - reference)))
    guard = float(np.max(np.asarray(errors)))
    results[int(n_steps)] = (guard, true_error)
    print(f"  {n_steps:8d} {probe.measurement_dt:12.4e} {guard:22.3e} {true_error:22.3e}", flush=True)

  shipped_steps = detector.n_steps_per_measurement
  if shipped_steps in results:
    guard, true_error = results[shipped_steps]
    print(f"\n  SHIPPED n_steps_per_measurement = {shipped_steps}: guard {guard:.3e} mM against "
          f"integration_tolerance {detector.integration_tolerance:.3e} mM ({detector.integration_tolerance / guard:.1f}x "
          f"margin); TRUE error {true_error:.3e} mM, {detector.measurement_noise / true_error:.0f}x below the "
          f"read-out noise floor")
  return {"steps": {k: {"guard": v[0], "true": v[1]} for k, v in results.items()}}


# --------------------------------------------------------------------------------------------- #
# guard -- the dt/dt-half assertion at SCREENING SCALE
# --------------------------------------------------------------------------------------------- #
def guard_error(detector, design, n_events):
  """`max|fine - coarse|` over `n_events` events at one design: the detector's OWN error estimate.

  Literally the lines `EnzymeMMDetector.__call__` runs before it asserts -- `_resolve_design`, then the
  jitted `_generate`, then `float(jnp.max(error))` -- so the number returned here is the number the
  guard tests, not a second implementation of the comparison.
  """
  index = np.arange(n_events, dtype=np.int64)
  fraction, temperature, initial_A, initial_B = detector._resolve_design(design, n_events)
  *_, error = detector._generate(fraction, temperature, initial_A, initial_B, jnp.asarray(index, jnp.int32))
  return float(jnp.max(error))


def _describe(detector, design):
  """One design as `(fraction, temperature, [A]0, [B]0)` per experiment -- so a worst case can be read."""
  blocks = [np.asarray(block, np.float64).reshape(-1) for block in detector._split(jnp.asarray(design))]
  return '  '.join(
    '(' + ', '.join(f'{block[k]:.4g}' for block in blocks) + ')' for k in range(detector.n_experiments)
  )


def mode_guard(detector, arguments):
  print("=" * 100)
  print(f"GUARD  (the dt/dt-half assertion at SCREENING SCALE -- {arguments.n_guard_events} events per "
        f"design)")
  print("=" * 100)
  print(f"  the guard is a MAX over sampled events, so it GROWS with the event count: a margin measured")
  print(f"  at 4096 does not transfer to the {arguments.n_guard_events} a screen runs. Design order per "
        f"experiment: (fraction, T, [A]0, [B]0).")
  tolerances = [detector.integration_tolerance] + [
    value for value in arguments.guard_tolerance if value != detector.integration_tolerance
  ]
  print(f"  measurement_noise {detector.measurement_noise}, integration_tolerance "
        f"{detector.integration_tolerance:.3e}, n_steps_per_measurement {detector.n_steps_per_measurement}")
  print(f"  tolerances reported against: " + "  ".join(f"{value:.3e}" for value in tolerances))

  groups = [
    ("known", known_designs(detector)),
    ("corner", corner_designs(detector)),
    ("random", {f"r{k:03d}": design
                for k, design in enumerate(random_designs(detector, arguments.n_guard_designs,
                                                          arguments.seed + 11))}),
  ]
  results, fired = {}, []
  for group, designs in groups:
    print(f"\n  {group.upper()}  ({len(designs)} designs)")
    errors = {}
    for name, design in designs.items():
      errors[name] = guard_error(detector, design, arguments.n_guard_events)
      flags = "  ".join(
        f"{'FIRES' if errors[name] > value else 'ok':>5s} @{value:.1e}" for value in tolerances
      )
      print(f"    {name:8s} {errors[name]:10.3e} mM   {flags}", flush=True)
      if errors[name] > tolerances[0]:
        fired.append((group, name, errors[name]))
    worst = max(errors, key=errors.get)
    results[group] = {"worst": worst, "error": errors[worst], "errors": errors}
    print(f"    -> worst {worst}: {errors[worst]:.3e} mM   " + "  ".join(
      f"margin {value / errors[worst]:.2f}x @{value:.1e}" for value in tolerances))
    print(f"       {_describe(detector, designs[worst])}")

  overall = max(results, key=lambda group: results[group]["error"])
  worst_design = dict(groups)[overall][results[overall]["worst"]]
  error = results[overall]["error"]
  print(f"\n  OVERALL WORST: {overall}/{results[overall]['worst']}  {error:.3e} mM   " + "  ".join(
    f"margin {value / error:.2f}x @{value:.1e}" for value in tolerances))

  # The guard estimate is computed on the CLEAN trajectory (`_batch` -> `_integrate`), before the
  # read-out noise is added, so it cannot depend on `measurement_noise` -- only the THRESHOLD does.
  # Checked rather than asserted: rebuild at each reported tolerance's own noise and compare. The
  # noise each tolerance belongs to is read off THIS config's own coupling rather than hard-coded, so
  # the label stays right when the tolerance-to-noise fraction changes.
  coupling = detector.integration_tolerance / detector.measurement_noise
  print(f"\n  this config couples integration_tolerance = {coupling:.3g} x measurement_noise")
  for value in tolerances[1:]:
    noise = value / coupling
    other = build(arguments.config, detector.n_experiments, measurement_noise=noise,
                  integration_tolerance=value)
    repeated = guard_error(other, worst_design, arguments.n_guard_events)
    print(f"  same design at measurement_noise {noise:g} / tolerance {value:.3e}: "
          f"{repeated:.3e} mM ({'IDENTICAL' if repeated == error else 'DIFFERS'}), margin "
          f"{value / repeated:.2f}x")

  # The mode's own headline claim, MEASURED rather than asserted: the guard is a max over sampled
  # events, so it climbs with the event count and a margin measured at a smaller sample is not a margin.
  if len(arguments.guard_event_scan) > 0:
    print(f"\n  the same worst design vs EVENT COUNT (the max cannot fall as events are added):")
    for count in arguments.guard_event_scan:
      scaled = guard_error(detector, worst_design, count)
      print(f"    {count:8d} events  {scaled:10.3e} mM   " + "  ".join(
        f"{'FIRES' if scaled > value else 'ok':>5s} @{value:.1e}" for value in tolerances))

  # And finally the REAL host-side assertion, exercised through `__call__` at the worst design.
  try:
    detector(worst_design, np.arange(arguments.n_guard_events, dtype=np.int64))
    print(f"  __call__ at the worst design over {arguments.n_guard_events} events: did NOT raise")
  except RuntimeError as failure:
    print(f"  __call__ at the worst design over {arguments.n_guard_events} events: RAISED -- {failure}")
    fired.append((overall, results[overall]["worst"], error))

  if len(fired) > 0:
    print(f"\n  !! STOP: the guard fires at {len(fired)} design(s) against integration_tolerance "
          f"{tolerances[0]:.3e}. Do not raise the tolerance and do not change the step count -- report "
          f"the numbers.")
  else:
    print(f"\n  the guard holds everywhere tested, against integration_tolerance {tolerances[0]:.3e}.")
  return {group: {"worst": value["worst"], "error": value["error"]} for group, value in results.items()}


# --------------------------------------------------------------------------------------------- #
# fim -- identifiability
# --------------------------------------------------------------------------------------------- #
def _normalised_bounds(detector):
  """`(names, low, high)` of the full latent vector: the drawn parameters plus log(half_time).

  Every latent is scaled by its own prior range onto [-1, 1], so each carries prior variance 1/3 and
  the Fisher matrix is directly comparable across parameters of wildly different units.
  """
  names = list(PARAMETER_NAMES) + ["log_half_time"]
  ranges = dict(detector.parameter_ranges)
  low = [ranges[name][0] for name in PARAMETER_NAMES] + [math.log(detector.half_time_bounds[0])]
  high = [ranges[name][1] for name in PARAMETER_NAMES] + [math.log(detector.half_time_bounds[1])]
  return names, np.asarray(low), np.asarray(high)


def fisher_information(detector, design, n_draws, seed):
  """Expected Fisher information of the measured [A](t) w.r.t. the NORMALISED latent vector.

  The read-out is `A_ij = f_ij(theta) + eps`, `eps ~ N(0, sigma_ij^2)` with the detector's own
  heteroscedastic sigma, so `I_kl = sum_ij (df_ij/dtheta_k)(df_ij/dtheta_l) / sigma_ij^2`. The
  derivative is `jax.jacfwd` straight through the RKC2 solve -- including the turnover calibration, so
  `log_half_time` enters exactly as it does in the simulator.

  Returned per draw (shape `(n_draws, n_latent, n_latent)`): the model is nonlinear, so marginalising
  the nuisances draw by draw and averaging the resulting posterior variances is the honest order of
  operations, and averaging the matrices first is reported only as a summary.
  """
  names, low, high = _normalised_bounds(detector)
  low, high = jnp.asarray(low, jnp.float32), jnp.asarray(high, jnp.float32)
  design = jnp.asarray(design, jnp.float32)

  def readout(normalised):
    physical = 0.5 * (normalised * (high - low) + (low + high))
    parameters = {name: physical[k] for k, name in enumerate(PARAMETER_NAMES)}
    return detector.readout(design, parameters, jnp.exp(physical[-1]))

  def information(normalised):
    jacobian = jax.jacfwd(readout)(normalised)          # (n_experiments, n_measurements, n_latent)
    sigma = detector.measurement_sigma(readout(normalised))
    weighted = jacobian / sigma[..., None]
    flat = weighted.reshape(-1, weighted.shape[-1])
    return flat.T @ flat

  rng = np.random.default_rng(seed)
  draws = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_draws, len(names))), jnp.float32)
  return names, np.asarray(jax.jit(jax.vmap(information))(draws), np.float64)


def marginal_posterior(information, target_index):
  """Posterior variance of the TARGET block after marginalising the nuisances with their own prior.

  `I_eff = I_tt - I_tn (I_nn + 3I)^-1 I_nt` (a uniform prior on [-1, 1] has variance 1/3, i.e.
  precision 3), and the posterior covariance of the target is `(I_eff + 3I)^-1`. Read against the prior
  variance 1/3: a component at 1/3 was not measured at all.
  """
  n = information.shape[-1]
  nuisance_index = [k for k in range(n) if k not in target_index]
  target_index = list(target_index)
  I_tt = information[np.ix_(target_index, target_index)]
  if len(nuisance_index) == 0:
    effective = I_tt
  else:
    I_tn = information[np.ix_(target_index, nuisance_index)]
    I_nn = information[np.ix_(nuisance_index, nuisance_index)] + 3.0 * np.eye(len(nuisance_index))
    effective = I_tt - I_tn @ np.linalg.solve(I_nn, I_tn.T)
  posterior = np.linalg.inv(effective + 3.0 * np.eye(len(target_index)))
  return effective, np.diag(posterior)


def mode_fim(detector, arguments):
  print("=" * 100)
  print("FIM  (identifiability -- the target is CHOSEN from this, section 5 of the plan)")
  print("=" * 100)
  designs = known_designs(detector)

  summary = {}
  for label in ("GOOD", "BAD"):
    names, information = fisher_information(detector, designs[label], arguments.n_fim_draws, arguments.seed)
    mean_information = information.mean(axis=0)

    # (1) SCREEN every latent on its own: posterior variance with everything else marginalised out.
    print(f"\n  {label} design -- per-parameter posterior variance, all others marginalised "
          f"(prior 0.3333; lower = measurable):")
    per_parameter = {}
    for k, name in enumerate(names):
      variances = np.array([marginal_posterior(single, [k])[1][0] for single in information])
      per_parameter[name] = float(variances.mean())
      print(f"    {name:16s} {per_parameter[name]:.4f}")
    summary[label] = per_parameter

    # (2) CANDIDATE TARGET VECTORS, jointly -- the choice is made from this table, not before it.
    candidates = [list(detector.target_parameters)] + [
      list(candidate) for candidate in arguments.candidate_targets
      if list(candidate) != list(detector.target_parameters)
    ]
    for candidate in candidates:
      target_index = [names.index(name) for name in candidate]
      effective, _ = marginal_posterior(mean_information, target_index)
      posterior = np.mean([marginal_posterior(single, target_index)[1] for single in information], axis=0)
      eigenvalues, eigenvectors = np.linalg.eigh(effective)
      condition = float(eigenvalues.max() / max(eigenvalues.min(), 1e-300))
      print(f"\n  {label} design -- candidate target {tuple(candidate)}:")
      print(f"    eigenvalues of I_eff (mean matrix): {np.array2string(eigenvalues, precision=3)}")
      print(f"    condition number: {condition:.3g}")
      print(f"    weakest direction: " + "  ".join(
        f"{name} {eigenvectors[i, 0]:+.2f}" for i, name in enumerate(candidate)))
      print(f"    posterior variance per component (mean over draws): " + "  ".join(
        f"{name} {posterior[i]:.4f}" for i, name in enumerate(candidate)))
      print(f"    PREDICTED LOSS (mean posterior variance): {posterior.mean():.4f}   "
            f"(prior/no-information: 0.3333)")
      summary[label + " " + ",".join(candidate)] = {
        "condition": condition, "posterior": posterior.tolist(), "predicted_loss": float(posterior.mean())
      }
  return summary


# --------------------------------------------------------------------------------------------- #
# calib -- known-good vs known-bad through the SAME proxy the screen uses
# --------------------------------------------------------------------------------------------- #
def mode_calib(detector, arguments):
  from detopt.bo.gbdt import score_design

  print("=" * 100)
  print(f"CALIBRATION  (known-good vs known-bad vs dead, {arguments.n_calib_events} events, the SAME "
        f"proxy the screen uses)")
  print("=" * 100)
  designs = known_designs(detector)
  _, _, _, target = detector(designs["DEAD"], np.arange(arguments.n_calib_events, dtype=np.int64))
  normalised = np.asarray(detector.normalize_target(target))
  ceiling = float(np.mean(np.square(normalised - normalised.mean(axis=0))))
  print(f"  measured ceiling (variance of the normalised target): {ceiling:.4f}")
  out = {"ceiling": ceiling}
  for name, design in designs.items():
    score = score_design(detector, design, n_events=arguments.n_calib_events, event_offset=0, seed=arguments.seed)
    out[name] = {"loss": score.loss, "sem": score.sem, "train": score.train, "val": score.val,
                 "n_learners": score.n_learners}
    print(f"  {name:5s}: loss {score.loss:.4f} +- {score.sem:.4f}   "
          f"(train {score.train:.4f} / val {score.val:.4f}, {score.n_learners} learners)   "
          f"{100.0 * (1.0 - score.loss / ceiling):5.1f}% below the ceiling")
  separation = (out["BAD"]["loss"] - out["GOOD"]["loss"]) / math.sqrt(out["BAD"]["sem"] ** 2 + out["GOOD"]["sem"] ** 2)
  print(f"\n  GOOD vs BAD separation: {separation:.1f} standard errors")
  if separation < 5.0:
    print("  !! STOP: the design the physics says is good is not clearly better -- the target is not "
          "in the data.")
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=CONFIG)
  parser.add_argument("--m", type=int, default=None, help="batch size override (default: the config's)")
  parser.add_argument("--target", nargs="*", default=None, help="target_parameters override")
  parser.add_argument("--candidate-target", dest="candidate_targets", nargs="*", default=[
    "log_K_B,Q10_cat", "log_K_B,Q10_cat,log_Ki_D", "log_K_B,Q10_cat,log_K_A",
    "log_K_B,Q10_cat,log_Ki_D,log_K_A"
  ], help="comma-separated candidate target vectors compared in --mode fim")
  parser.add_argument("--mode", nargs="*", default=["all"],
                      choices=["all", "build", "stability", "steps", "guard", "fim", "calib"])
  parser.add_argument("--n-events", type=int, default=4096)
  parser.add_argument("--n-calib-events", type=int, default=4096)
  parser.add_argument("--n-fim-draws", type=int, default=64)
  parser.add_argument("--n-stability-designs", type=int, default=64)
  parser.add_argument("--n-stability-enzymes", type=int, default=64)
  parser.add_argument("--n-step-designs", type=int, default=8)
  parser.add_argument("--n-step-enzymes", type=int, default=16)
  parser.add_argument("--n-reference-steps", type=int, default=2560)
  parser.add_argument("--step-scan", type=int, nargs="*", default=[20, 40, 80, 160, 320])
  parser.add_argument("--n-guard-events", type=int, default=16384,
                      help="events per design in --mode guard; DEFAULTS TO THE SCREENING COUNT, since "
                           "the guard is a max over events and a smaller sample understates it")
  parser.add_argument("--n-guard-designs", type=int, default=64,
                      help="random designs in --mode guard, on top of the known and corner designs")
  parser.add_argument("--guard-event-scan", type=int, nargs="*", default=[1024, 4096, 65536],
                      help="event counts to re-measure the WORST design at, so the growth of the max "
                           "with the sample is visible rather than assumed")
  parser.add_argument("--guard-tolerance", type=float, nargs="*", default=[],
                      help="extra integration_tolerance values to report the same errors against "
                           "(e.g. the pre-registered noise arms), on top of the config's own")
  parser.add_argument("--seed", type=int, default=0)
  arguments = parser.parse_args()

  arguments.candidate_targets = [entry.split(",") for entry in arguments.candidate_targets]
  modes = arguments.mode
  if "all" in modes:
    modes = ["build", "stability", "steps", "guard", "fim", "calib"]
  overrides = {} if arguments.target is None else {"target_parameters": list(arguments.target)}
  detector = build(arguments.config, arguments.m, **overrides)
  for mode in modes:
    {"build": mode_build, "stability": mode_stability, "steps": mode_steps, "guard": mode_guard,
     "fim": mode_fim, "calib": mode_calib}[mode](detector, arguments)
    print()


if __name__ == "__main__":
  main()
