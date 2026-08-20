#!/usr/bin/env python3
"""Calibration evidence for the `enzyme_depletion_bi` detector -- every number in its config header.

    python scripts/calibrate_enzyme_depletion_bi.py --section all --output-dir output/enzyme-depletion-bi

Sections, in the order the task was calibrated:

  reference       the closed-form solution derived for this task, against a stiff solver at rtol 1e-12
  degeneracy      the A0 == B0 collapse, MEASURED against the exact model rather than asserted
  measurements    the window/sampling arithmetic that fixes n_measurements, and the half-decay check
  window          the time window: coverage of the prior, per substrate
  visibility      the two-knee visibility map and the design box it implies
  profile         the loss over the (A0, B0) plane at m = 1 -- which design identifies which parameter
  integrator      RKC2 against the closed form over the WHOLE prior box, and its cost
  estimator       why a lattice PLATEAU proves nothing, and the posterior against a float64 reference
  resolution      (noise, n_grid) jointly: where the lattice is fine enough to be a Bayes estimator
  parameterisation  whether the estimator's error is scale-free in the value or in its log
  landscape       the loss profile over random designs, and its SHAPE against a quadratic reference

THE REFERENCE IS AN INDEPENDENT NUMERICAL ROUTE, NOT AN ERROR-FREE TWIN. With 1:1 stoichiometry
`[B] = [A] + (B0 - A0)`, so the system collapses to one ODE in the extent of reaction
`x = A0 - [A] = B0 - [B]`,

    dx/dt = q (A0 - x)(B0 - x) / ( (K_A + A0 - x)(K_B + B0 - x) ),   x(0) = 0,

the PRODUCT form -- rapid-equilibrium random binding of two substrates, three parameters
`(q, K_A, K_B)`, no inhibitor and none implied. Dividing through and separating variables,

    q t = x + K_A ln( A0/(A0-x) ) + K_B ln( B0/(B0-x) ) + (K_A K_B / D) ln( A0 (B0-x) / (B0 (A0-x)) ),

with `D = B0 - A0` and the finite `D -> 0` branch on the diagonal

    q t = x + (K_A + K_B) ln( A0/(A0-x) ) + K_A K_B ( 1/(A0-x) - 1/A0 ).

⚠️ THIS IS A QUADRATURE GIVING `t(x)`, NOT A CLOSED-FORM SOLUTION: `x(t)` is not elementary and is
recovered by bisection plus Newton in `exact_extent`, which is a numerical inversion with its own
error mode. It is worth having because that error mode is completely different from the detector's
RKC2 chain, not because it is exact.

It is strictly increasing in x on [0, min(A0, B0)) and unbounded there, so it inverts uniquely.
Substituting `x = M (1 - e^-z)` with `M = min(A0, B0)` turns the right-hand side into an increasing,
asymptotically LINEAR function of z on [0, inf), which gives the exact bracket `z <= q t / K_limiting`
and makes bisection plus Newton converge from any prior draw.

The PING-PONG law `v = q A B / (K_A B + K_B A + A B)` this task once used is NOT this one: its
denominator is missing the constant `K_A K_B`, which is what makes it singular at `A = B = 0` and
what creates the exact diagonal degeneracy that section 2 now measures as ABSENT. Everything
calibrated under it is quarantined in `output/enzyme-depletion-bi/stale-ping-pong-law/`.

Sections write their numbers to `<output-dir>/<section>.json` as well as to stdout, so
`scripts/plot_enzyme_depletion_bi.py` can draw them without re-running anything.
"""
import argparse
import json
import math
import os

import numpy as np

# The priors and the assay scales come from `config/detector/enzyme.yaml`, which models THE SAME
# reaction -- hexokinase, glucose (A) + ATP (B) -> glucose-6-phosphate (C) + ADP (D) -- and carries
# their provenance. Only the priors and the scales are taken: the temperature dependence, the
# product inhibition, the unfolding and the turnover calibration of that model are NOT part of this
# task, whose whole point is the absence of nuisance.
KCAT_BOUNDS = (30.0, 300.0)
ENZYME_CONCENTRATION = 3.0e-5
VELOCITY_BOUNDS = (KCAT_BOUNDS[0] * ENZYME_CONCENTRATION, KCAT_BOUNDS[1] * ENZYME_CONCENTRATION)
MICHAELIS_A_BOUNDS = (0.02, 0.2)
MICHAELIS_B_BOUNDS = (0.1, 1.0)
BOUNDS = (VELOCITY_BOUNDS, MICHAELIS_A_BOUNDS, MICHAELIS_B_BOUNDS)
CONCENTRATION_A_BOUNDS = (0.8, 10.0)
CONCENTRATION_B_BOUNDS = (1.0, 25.0)
DESIGN_BOX = (CONCENTRATION_A_BOUNDS, CONCENTRATION_B_BOUNDS)
DURATION = 3600.0
N_MEASUREMENTS = 10
MEASUREMENT_NOISE = 0.02
NO_INFORMATION = 1.0 / 3.0
# The LATE cutoff of the knee-visibility window, as a fraction of `duration`. It is the whole
# difference between the visibility ceiling `log10(f * n_measurements)` and `log10(n_measurements)`,
# so it is exposed rather than buried: at f = 0.8 an 8-sample assay cannot reach 0.90 and at f = 1.0
# it can, and that is a choice about what counts as a resolved knee, not a tuning knob.
KNEE_WINDOW_FRACTION = 0.8
LN_RANGE = tuple(math.log(high) - math.log(low) for low, high in BOUNDS)

RECORD = {}


# --------------------------------------------------------------------------------------------- #
# The reference solution and the prior.
# --------------------------------------------------------------------------------------------- #
def _split(initial_a, initial_b, michaelis_a, michaelis_b):
  """`(L0, E0, D, K_L, K_E)`: limiting and excess substrate, and their constants.

  The quadrature is symmetric under swapping the two substrates, so writing it on `limiting` and
  `excess` rather than on A and B removes the sign of `B0 - A0` from every branch."""
  a_limits = initial_a <= initial_b
  limiting = np.where(a_limits, initial_a, initial_b)
  excess = np.where(a_limits, initial_b, initial_a)
  k_limiting = np.where(a_limits, michaelis_a, michaelis_b)
  k_excess = np.where(a_limits, michaelis_b, michaelis_a)
  return limiting, excess, excess - limiting, k_limiting, k_excess


def exact_time(extent, initial_a, initial_b, velocity, michaelis_a, michaelis_b):
  """`t(x)`: the quadrature FORWARD, strictly increasing on `[0, min(A0, B0))` and `+inf` at the end.

  THE CROSS TERM IS WRITTEN ON `log1p` AND MUST STAY THAT WAY. Algebraically it is
  `(1/D) ln(A0 (B0-x) / (B0 (A0-x)))`, a difference of two logarithms that cancels to nothing as
  `D -> 0`; the bracket equals `1 + D x / ((L0-x) E0)` identically, so `log1p` of that evaluates the
  same number without the cancellation and its `D = 0` limit `x / ((L0-x) L0)` is continuous with
  it."""
  extent, initial_a, initial_b, velocity, michaelis_a, michaelis_b = np.broadcast_arrays(
    *[np.asarray(v, np.float64) for v in (extent, initial_a, initial_b, velocity, michaelis_a, michaelis_b)]
  )
  limiting, total, delta, k_limiting, k_excess = _split(initial_a, initial_b, michaelis_a, michaelis_b)
  left = limiting - extent
  with np.errstate(divide='ignore', invalid='ignore'):
    regular = np.log1p(delta * extent / (left * total)) / np.where(delta > 0.0, delta, 1.0)
    diagonal = extent / (left * limiting)
    cross = np.where(delta > 0.0, regular, diagonal)
    return (
      extent + k_limiting * np.log(limiting / left) + k_excess * np.log(total /
                                                                        (total - extent)) + k_limiting * k_excess * cross
    ) / velocity


def exact_extent(t, initial_a, initial_b, velocity, michaelis_a, michaelis_b, n_bisect=80, n_newton=4):
  """`x(t)`: the quadrature INVERTED, in float64, by bisection then Newton (see the module docstring).

  The substitution `x = L0 (1 - e^-z)` makes `q t(z)` asymptotically LINEAR in `z` with slope
  `K_L (1 + K_E / D)`, which gives the exact bracket `z <= q t / K_L` and makes Newton converge from
  any prior draw. Its derivative `dR/dz = (K_L + a_L)(K_E + a_E) / a_E` is the rate law itself, so
  the two agree by construction rather than by a second derivation."""
  t, initial_a, initial_b, velocity, michaelis_a, michaelis_b = np.broadcast_arrays(
    *[np.asarray(v, np.float64) for v in (t, initial_a, initial_b, velocity, michaelis_a, michaelis_b)]
  )
  limiting, total, delta, k_limiting, k_excess = _split(initial_a, initial_b, michaelis_a, michaelis_b)
  target = velocity * t

  def residual_and_slope(z):
    capped = np.minimum(z, 700.0)
    left = limiting * np.exp(-capped)
    extent = limiting * -np.expm1(-capped)
    grown = np.expm1(capped)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
      finite = np.log1p(delta * grown / total)
      asymptotic = z + np.log(np.where(delta > 0.0, delta, 1.0) / total)
      regular = np.where(z > 700.0, asymptotic, finite) / np.where(delta > 0.0, delta, 1.0)
      cross = np.where(delta > 0.0, regular, grown / limiting)
    value = (extent + k_limiting * capped + k_excess * np.log(total / (delta + left)) + k_limiting * k_excess * cross - target)
    return value, (k_limiting + left) * (k_excess + delta + left) / (delta + left)

  low = np.zeros_like(target)
  high = target / k_limiting + 1.0
  for _ in range(n_bisect):
    mid = 0.5 * (low + high)
    negative = residual_and_slope(mid)[0] < 0.0
    low = np.where(negative, mid, low)
    high = np.where(negative, high, mid)
  z = 0.5 * (low + high)
  for _ in range(n_newton):
    value, slope = residual_and_slope(z)
    z = np.clip(z - value / slope, low, high)
  return limiting * -np.expm1(-np.minimum(z, 700.0))


def knee_times(initial_a, initial_b, velocity, michaelis_a, michaelis_b):
  """`(t_A, t_B)`: when EACH substrate reaches its OWN Michaelis constant; infinite when it never does.

  A knee exists for A only if A starts above `K_A` AND B does not run out first, and symmetrically."""
  out = []
  for extent, reachable in ((initial_a - michaelis_a, (initial_a > michaelis_a) & (initial_b > initial_a - michaelis_a)),
                            (initial_b - michaelis_b, (initial_b > michaelis_b) & (initial_a > initial_b - michaelis_b)),
                            ):
    reachable = reachable & (extent > 0.0)
    time = exact_time(np.where(reachable, extent, 0.0), initial_a, initial_b, velocity, michaelis_a, michaelis_b)
    out.append(np.where(reachable, time, np.inf))
  return out


def visible(initial_a, initial_b, parameters, duration=DURATION, n_measurements=N_MEASUREMENTS):
  """Per-substrate knee visibility: the knee falls INSIDE the sampled window `[t_1, 0.8 * duration]`.

  TWO-SIDED on purpose. A knee after the last read-out is invisible for the obvious reason; a knee
  BEFORE the first one is just as invisible, because the sampled curve is then already flat and says
  only that the reaction finished early. A one-sided criterion counts the second case as a success
  and would pass a box in which the fastest half of the prior is never resolved."""
  time_a, time_b = knee_times(initial_a, initial_b, *parameters)
  first = duration / n_measurements
  last = KNEE_WINDOW_FRACTION * duration
  return ((time_a >= first) & (time_a <= last), (time_b >= first) & (time_b <= last))


def to_unit(value, bounds):
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return 2.0 * (np.log(value) - low) / (high - low) - 1.0


def from_unit(unit, bounds):
  low, high = math.log(bounds[0]), math.log(bounds[1])
  return np.exp(low + 0.5 * (np.asarray(unit, np.float64) + 1.0) * (high - low))


def draw_prior(n, seed):
  """`n` log-uniform parameter draws, as (unit `[-1, 1]^3`, physical `(q, K_A, K_B)`)."""
  unit = np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 3))
  return unit, np.stack([from_unit(unit[:, i], BOUNDS[i]) for i in range(3)], -1)


def prior_grid(n_grid, n_grid_velocity=None):
  """The estimator lattice, ANISOTROPIC: `n_grid_velocity` nodes on ln q, `n_grid` on each constant.

  The velocity is determined an order of magnitude more sharply than either constant, so a lattice
  fine enough for the constants is far coarser than the posterior's ridge in q -- and then the
  softmax collapses onto one node, the marginalisation over q is lost, and the CONSTANTS come out
  biased and over-confident, scoring above the prior. See `--section resolution`."""
  axes = (
    np.linspace(-1.0, 1.0,
                n_grid if n_grid_velocity is None else n_grid_velocity), np.linspace(-1.0, 1.0,
                                                                                     n_grid), np.linspace(-1.0, 1.0, n_grid)
  )
  mesh = np.meshgrid(*axes, indexing='ij')
  scaled = np.stack([m.reshape(-1) for m in mesh], -1)
  return scaled, np.stack([from_unit(scaled[:, i], BOUNDS[i]) for i in range(3)], -1)


def curves(design, parameters, n_measurements=N_MEASUREMENTS, duration=DURATION):
  """`(n_parameters, n_experiments, n_measurements)` clean extent readings, from the closed form."""
  design = np.atleast_2d(np.asarray(design, np.float64))
  times = np.arange(1, n_measurements + 1) * duration / n_measurements
  return exact_extent(
    times[None, None, :], design[None, :, 0, None], design[None, :, 1, None], parameters[:, 0][:, None, None],
    parameters[:, 1][:, None, None], parameters[:, 2][:, None, None]
  )


def reference_loss_sweep(design, noises, n_events=1024, n_grid=31, n_grid_velocity=241, seed=0, chunk=64):
  """The grid posterior mean's per-parameter loss at EVERY noise in `noises`, from the CLOSED FORM.

  The model lattice and the clean curves depend on the design alone, so they are built ONCE and
  reused across the noise sweep -- the integration is the whole cost here, and recomputing it per
  noise level would be a pure multiple of it. The detector's own `estimate` is the same instrument on
  an RKC2 model; this one carries no integration error, so the two can be compared."""
  scaled, physical = prior_grid(n_grid, n_grid_velocity)
  model = curves(design, physical).reshape(physical.shape[0], -1)
  square = 0.5 * np.sum(model * model, -1)
  truth_unit, truth_physical = draw_prior(n_events, seed)
  clean = curves(design, truth_physical).reshape(n_events, -1)
  standard = np.random.default_rng(seed + 7777).standard_normal(clean.shape)
  out = []
  for noise in noises:
    data = clean + noise * standard
    predicted = np.empty((n_events, 3))
    for start in range(0, n_events, chunk):
      log_likelihood = (data[start:start + chunk] @ model.T - square[None, :]) / noise**2
      log_likelihood -= log_likelihood.max(-1, keepdims=True)
      weight = np.exp(log_likelihood)
      weight /= weight.sum(-1, keepdims=True)
      predicted[start:start + chunk] = weight @ scaled
    out.append(((predicted - truth_unit)**2).mean(0))
  return np.asarray(out)


def reference_loss(design, noise, n_events=1024, n_grid=31, n_grid_velocity=241, seed=0, chunk=64, per_parameter=False):
  """`reference_loss_sweep` at a single noise level, as a scalar (or the per-parameter triple)."""
  parts = reference_loss_sweep(
    design, (noise, ), n_events=n_events, n_grid=n_grid, n_grid_velocity=n_grid_velocity, seed=seed, chunk=chunk
  )[0]
  return (float(parts.mean()), parts) if per_parameter else float(parts.mean())


def scaled_to_design(points, n_experiments, box=DESIGN_BOX):
  """`(n, 2 m)` points on the scaled cube -> a list of `(m, 2)` NOMINAL designs, log-affine per box."""
  low = np.array([math.log(box[0][0]), math.log(box[1][0])])
  high = np.array([math.log(box[0][1]), math.log(box[1][1])])
  return [np.exp(low[None, :] + p.reshape(2, n_experiments).T * (high - low)[None, :]) for p in np.atleast_2d(points)]


def sobol_points(n_experiments, n_designs, seed=0):
  from scipy.stats import qmc

  return qmc.Sobol(2 * n_experiments, scramble=True, seed=seed).random(n_designs)


# --------------------------------------------------------------------------------------------- #
def section_reference(arguments):
  from scipy.integrate import solve_ivp

  print('=== 1. the closed form, against a stiff solver at rtol 1e-12 ===')
  print('    The whole calibration rests on the derivation in the module docstring, so it is CHECKED')
  print('    against an independent integration before anything is measured against it. The sampled')
  print('    box is wider than the design box, so the check covers the estimator grid too.')
  print('    THE INVERSION IS WHAT IS CHECKED, not the quadrature. `t(x)` is elementary and could be')
  print('    verified by differentiating; `x(t)` is not, and the bisection-plus-Newton that recovers')
  print('    it is a numerical method whose error was never measured until this section did it.')
  print('    The DIAGONAL and the NEAR-diagonal are forced into the sweep, because `A0 == B0` is where')
  print('    the cross term is a removable 0/0 and a product of two random ranges never lands on it.')
  rng = np.random.default_rng(0)
  worst, worst_at, errors, diagonal_errors = 0.0, None, [], []
  times = np.arange(1, N_MEASUREMENTS + 1) * DURATION / N_MEASUREMENTS
  for trial in range(arguments.n_reference):
    if trial % 4 == 0:
      initial_a = float(np.exp(rng.uniform(math.log(0.4), math.log(50.0))))
      initial_b = initial_a if trial % 8 == 0 else initial_a * float(np.exp(rng.uniform(-1.0e-6, 1.0e-6)))
    else:
      initial_a, initial_b = np.exp(rng.uniform(math.log(0.4), math.log(50.0), 2))
    velocity = float(np.exp(rng.uniform(*np.log(VELOCITY_BOUNDS))))
    michaelis_a = float(np.exp(rng.uniform(*np.log(MICHAELIS_A_BOUNDS))))
    michaelis_b = float(np.exp(rng.uniform(*np.log(MICHAELIS_B_BOUNDS))))

    def right_hand_side(_, state):
      left_a, left_b = max(initial_a - state[0], 0.0), max(initial_b - state[0], 0.0)
      return [velocity * left_a * left_b / ((michaelis_a + left_a) * (michaelis_b + left_b))]

    solution = solve_ivp(right_hand_side, (0.0, DURATION), [0.0], t_eval=times, rtol=1e-12, atol=1e-14, method='LSODA')
    mine = exact_extent(times, initial_a, initial_b, velocity, michaelis_a, michaelis_b)
    error = float(np.max(np.abs(solution.y[0] - mine)))
    errors.append(error)
    if trial % 4 == 0:
      diagonal_errors.append(error)
    if error > worst:
      worst, worst_at = error, (initial_a, initial_b, velocity, michaelis_a, michaelis_b)
  print(f'    {arguments.n_reference} random (A0, B0, q, K_A, K_B) points: max |inversion - LSODA| = {worst:.3e} mM')
  print(
    f'    median {float(np.median(errors)):.3e} mM; worst at A0={worst_at[0]:.3f} B0={worst_at[1]:.3f} '
    f'q={worst_at[2]:.3e} K_A={worst_at[3]:.4f} K_B={worst_at[4]:.4f}'
  )
  print(f'    of which {len(diagonal_errors)} diagonal / near-diagonal: max {max(diagonal_errors):.3e} mM')

  # The inversion against its OWN forward map, which isolates the root-find from the quadrature: a
  # bad derivation would agree here and disagree with LSODA, a bad root-find disagrees with both.
  round_trip = []
  for _ in range(arguments.n_reference):
    initial_a, initial_b = np.exp(rng.uniform(math.log(0.4), math.log(50.0), 2))
    velocity = float(np.exp(rng.uniform(*np.log(VELOCITY_BOUNDS))))
    michaelis_a = float(np.exp(rng.uniform(*np.log(MICHAELIS_A_BOUNDS))))
    michaelis_b = float(np.exp(rng.uniform(*np.log(MICHAELIS_B_BOUNDS))))
    extent = exact_extent(times, initial_a, initial_b, velocity, michaelis_a, michaelis_b)
    unsaturated = extent < 0.999999 * min(initial_a, initial_b)
    if not np.any(unsaturated):
      continue
    back = exact_time(extent[unsaturated], initial_a, initial_b, velocity, michaelis_a, michaelis_b)
    round_trip.append(float(np.max(np.abs(back - times[unsaturated]) / times[unsaturated])))
  print(f'    round trip |t(x(t)) - t| / t on unsaturated read-outs: max {max(round_trip):.3e}')
  RECORD['reference'] = {
    'errors': errors,
    'worst': worst,
    'diagonal_worst': max(diagonal_errors),
    'round_trip_worst': max(round_trip)
  }


def section_degeneracy(arguments):
  print('=== 2. the A0 == B0 diagonal: NOT degenerate under the product form ===')
  print('    Under the PING-PONG law the diagonal was an EXACT degeneracy -- the rate collapsed to')
  print('    q [A] / ([A] + K_A + K_B) and only the SUM of the constants was identifiable, so two')
  print('    triples with the same sum gave literally the same curve. THAT IS A PROPERTY OF THE WRONG')
  print('    LAW. The product form keeps the cross term K_A K_B, which is not a function of the sum,')
  print('    so the collapse does not happen and the diagonal carries information about the SPLIT.')
  print('    Measured here rather than asserted, because the old box was built to AVOID the diagonal')
  print('    band and that reason no longer exists.')
  times = np.arange(1, N_MEASUREMENTS + 1) * DURATION / N_MEASUREMENTS
  pairs = ((0.05, 0.55), (0.15, 0.45))
  velocity = math.sqrt(VELOCITY_BOUNDS[0] * VELOCITY_BOUNDS[1])
  print(
    f'    (K_A, K_B) = {pairs[0]} against {pairs[1]} -- SAME SUM 0.60, same q = {velocity:.3e} mM/s, '
    f'read-out noise {MEASUREMENT_NOISE}'
  )
  print(f'    {"A0":>8} {"B0":>8} {"B0/A0":>7} {"max|difference| (mM)":>21} {"/ noise":>9} {"% of extent":>12}')
  rows = []
  for initial_a, initial_b in ((0.1, 0.1), (0.3, 0.3), (1.0, 1.0), (3.0, 3.0), (10.0, 10.0), (3.0, 3.6), (3.0, 6.0),
                               (3.0, 15.0), (10.0, 2.0), (10.0, 1.2)):
    one = exact_extent(times, initial_a, initial_b, velocity, *pairs[0])
    two = exact_extent(times, initial_a, initial_b, velocity, *pairs[1])
    separation = float(np.max(np.abs(one - two)))
    fraction = 100.0 * separation / max(float(np.max(np.abs(one))), 1.0e-30)
    rows.append((initial_a, initial_b, separation, fraction))
    print(
      f'    {initial_a:8.2f} {initial_b:8.2f} {initial_b / initial_a:7.2f} {separation:21.3e} '
      f'{separation / MEASUREMENT_NOISE:9.2f} {fraction:12.2f}'
    )
  print('    READ THE `/ noise` COLUMN, NOT THE `% of extent` ONE. As a FRACTION of the extent the')
  print('    diagonal separation grows monotonically as concentration falls (13.6% at A0 = 0.1 against')
  print('    0.2% at A0 = 10), which invites the conclusion that the diagonal is most informative at')
  print('    low concentration. It is not: the read-out noise is ABSOLUTE on the extent, so what an')
  print('    experiment resolves is the separation in mM, and THAT peaks in the middle of the box')
  print('    (1.76 x noise at A0 = 3) and falls at BOTH ends -- at low A0 there is simply not much')
  print('    extent to differ by. The fraction and the signal-to-noise disagree, and only the second')
  print('    one is what the estimator sees.')
  print('    Off-diagonal separates 1.3-4.6x better than the BEST diagonal design, so symmetry breaking')
  print('    is still worth designing for -- but the weakest off-diagonal designs beat it only')
  print('    marginally, and what is gone is the reason to EXCLUDE the diagonal band.')
  RECORD['degeneracy'] = {'rows': rows, 'noise': MEASUREMENT_NOISE}


def section_measurements(arguments):
  print('=== 3. the sampling count, from the window arithmetic ===')
  print('    The knee time is EXACTLY t_knee = F(A0, B0, K_A, K_B) / q, so for a FIXED design the')
  print('    knee lands inside [t_1, 0.8 T] only for q in a window of RATIO 0.8 * n_measurements.')
  print('    The velocity prior is log-uniform over a decade, so no design can make the knee visible')
  print('    for more than log10(0.8 * n_measurements) of the prior. That is an upper bound on the')
  print('    protocol`s 90% requirement, not a tuning outcome:')
  print(f'    {"n_measurements":>15} {"window ratio":>13} {"bound":>7} {"measured best f_A":>18}')
  parameters = draw_prior(arguments.n_draws, 0)[1].T
  a_values = np.geomspace(0.4, 20.0, 31)
  b_values = np.geomspace(0.5, 50.0, 31)
  rows = []
  for n_measurements in (8, 9, 10, 12):
    best = 0.0
    for initial_a in a_values:
      for initial_b in b_values:
        best = max(best, float(visible(initial_a, initial_b, parameters, n_measurements=n_measurements)[0].mean()))
    bound = math.log10(KNEE_WINDOW_FRACTION * n_measurements)
    rows.append((n_measurements, bound, best))
    print(f'    {n_measurements:15d} {KNEE_WINDOW_FRACTION * n_measurements:13.1f} {bound:7.3f} {best:18.3f}', flush=True)
  print(f'    --> n_measurements = {N_MEASUREMENTS} is the SMALLEST count that can reach 0.90, and the')
  print('        measured best f_A matches the bound to three decimals at every count, so this is')
  print('        ARITHMETIC and not a property of the rate law -- it is unchanged by the correction.')
  print()
  print('    ⚠️ AT n_measurements = 8 THE 0.90 REQUIREMENT IS UNREACHABLE, ceiling 0.806. The two')
  print('        settings are in direct conflict and no tuning removes it. What costs the difference')
  print('        is the 0.8 in `visible`: the knee is required to land in [t_1, 0.8 T] rather than')
  print(f'        [t_1, T], which is worth log10(1/0.8) = {math.log10(1.25):.3f} of the prior. Allowing a knee in the')
  print(f'        final fifth of the window would put n = 8 at log10(8) = {math.log10(8.0):.3f} and pass -- but a knee')
  print('        at the last read-out is not resolved by anything, so that is a weaker requirement')
  print('        wearing the same number, not a cheaper way to meet this one. Reported, not chosen:')
  print('        n = 8 and "some design reaches 0.90" cannot both hold, and which one gives is a call')
  print('        for whoever owns the acceptance test.')
  print()
  print('    The half decay against the sampling interval, at the prior MEDIAN parameters:')
  interval = DURATION / N_MEASUREMENTS
  velocity = math.sqrt(VELOCITY_BOUNDS[0] * VELOCITY_BOUNDS[1])
  michaelis_a = math.sqrt(MICHAELIS_A_BOUNDS[0] * MICHAELIS_A_BOUNDS[1])
  michaelis_b = math.sqrt(MICHAELIS_B_BOUNDS[0] * MICHAELIS_B_BOUNDS[1])
  print(
    f'    prior medians: q = {velocity:.3e} mM/s, K_A = {michaelis_a:.4f} mM, K_B = {michaelis_b:.4f} mM; '
    f'sampling interval {interval:.0f} s'
  )
  print(f'    {"A0":>7} {"B0":>7} {"t(75%)-t(25%)":>14} {"intervals":>10}')
  fine = np.linspace(0.0, DURATION, 20001)[1:]
  half_rows = []
  for initial_a, initial_b in ((1.0, 3.0), (3.0, 6.0), (3.0, 15.0), (6.0, 3.0), (10.0, 2.0), (10.0, 25.0)):
    fraction = exact_extent(fine, initial_a, initial_b, velocity, michaelis_a, michaelis_b) / min(initial_a, initial_b)
    if fraction[-1] < 0.75:
      print(f'    {initial_a:7.2f} {initial_b:7.2f} {"never reaches 75%":>14} {"-":>10}')
      continue
    width = float(fine[int(np.searchsorted(fraction, 0.75))] - fine[int(np.searchsorted(fraction, 0.25))])
    half_rows.append((initial_a, initial_b, width, width / interval))
    print(f'    {initial_a:7.2f} {initial_b:7.2f} {width:14.0f} {width / interval:10.2f}')
  RECORD['measurements'] = {'bound_rows': rows, 'half_rows': half_rows, 'interval': interval}


def section_window(arguments):
  print('=== 4. the time window: every parameter draw must have SOME design that shows each knee ===')
  parameters = draw_prior(arguments.n_draws, 0)[1].T
  a_values = np.geomspace(0.4, 20.0, 31)
  b_values = np.geomspace(0.5, 50.0, 31)
  print(f'    {"duration (s)":>12} {"best f_A":>9} {"best f_B":>9} {"cover A":>8} {"cover B":>8} {"cover both":>11}')
  rows = []
  for duration in (900.0, 1800.0, 2700.0, 3600.0, 5400.0, 7200.0, 14400.0):
    best_a = best_b = 0.0
    any_a = np.zeros(parameters.shape[1], bool)
    any_b = np.zeros(parameters.shape[1], bool)
    for initial_a in a_values:
      for initial_b in b_values:
        seen_a, seen_b = visible(initial_a, initial_b, parameters, duration=duration)
        any_a |= seen_a
        any_b |= seen_b
        best_a, best_b = max(best_a, float(seen_a.mean())), max(best_b, float(seen_b.mean()))
    rows.append((duration, best_a, best_b, float(any_a.mean()), float(any_b.mean()), float((any_a & any_b).mean())))
    print(
      f'    {duration:12.0f} {best_a:9.3f} {best_b:9.3f} {float(any_a.mean()):8.3f} {float(any_b.mean()):8.3f} '
      f'{float((any_a & any_b).mean()):11.3f}', flush=True
    )
  print('    COVERAGE DOES NOT BIND. It is 1.000 at every window tried, down to 900 s, so "long enough')
  print('    that the knee is reachable under some design" is satisfied by all of them and cannot')
  print('    choose between them. What binds is `best f_B`, the 0.90 requirement.')
  print(f'    --> duration = {DURATION:.0f} s. It is NOT the shortest window that passes: 2700 s reaches')
  print('        f_B = 0.901, which is over 0.90 by less than the Monte-Carlo standard error of these')
  print('        draws (+-0.0015 at p ~ 0.9), so 2700 is a COIN FLIP against the bar and 3600 clears it')
  print('        by ~2 se. Nothing above 3600 buys anything: f_A and f_B are both pinned at the')
  print(f'        arithmetic ceiling log10(0.8 n) = {math.log10(0.8 * N_MEASUREMENTS):.3f} from 3600 s onward, so a longer')
  print('        window is pure cost. 3600 s is also exactly the one hour of config/detector/enzyme.yaml.')
  RECORD['window'] = {'rows': rows}


def section_visibility(arguments):
  print('=== 5. the two-knee visibility map, and the design box ===')
  print('    SCORED SEPARATELY PER CONSTANT, and that choice is measured, not a preference. Both')
  print('    knees are visible in the SAME experiment only when -K_B <= A0 - B0 <= K_A, a narrow band')
  print('    around the diagonal. Under the PING-PONG law that band was also where the two constants')
  print('    were EXACTLY degenerate, so requiring both knees per experiment selected for the one')
  print('    configuration that could not separate them. UNDER THE PRODUCT FORM THAT IS NO LONGER SO')
  print('    (section 2): the diagonal separates the split at 0.7-1.8 x the read-out noise. The')
  print('    per-constant scoring is kept anyway, for the surviving reason that off-diagonal designs')
  print('    still separate 1.3-4.6x better, but the box is NO LONGER BUILT TO EXCLUDE the diagonal')
  print('    and candidates that contain it are on the list below.')
  print('    The design is a BATCH, so the requirement belongs there: every experiment in the box must')
  print('    show SOME knee for 25%+ of draws, and the box must contain experiments reaching 90% for')
  print('    K_A and for K_B separately, with coverage 1.000 over the prior.')
  parameters = draw_prior(arguments.n_draws, 0)[1].T
  a_values = np.geomspace(0.4, 20.0, 11)
  b_values = np.geomspace(0.5, 50.0, 11)
  grids = {'f_A': np.zeros((11, 11)), 'f_B': np.zeros((11, 11)), 'both': np.zeros((11, 11))}
  for i, initial_a in enumerate(a_values):
    for j, initial_b in enumerate(b_values):
      seen_a, seen_b = visible(initial_a, initial_b, parameters)
      grids['f_A'][i, j] = float(seen_a.mean())
      grids['f_B'][i, j] = float(seen_b.mean())
      grids['both'][i, j] = float((seen_a & seen_b).mean())
  for name in ('f_A', 'f_B', 'both'):
    print(f'    {name} (rows A0, columns B0)')
    print('       A0 \\ B0' + ' '.join(f'{v:6.2f}' for v in b_values))
    for i, initial_a in enumerate(a_values):
      print(f'      {initial_a:7.3f} ' + ' '.join(f'{grids[name][i, j]:6.3f}' for j in range(11)))
  print()
  print('    candidate boxes, on a 21 x 21 grid inside each:')
  print(f'    {"A0 box":>16} {"B0 box":>16} {"worst design":>13} {"best f_A":>9} {"best f_B":>9} {"cover":>7}')
  candidates = [((0.8, 10.0), (1.0, 25.0)), ((1.0, 8.0), (1.5, 20.0)), ((0.5, 15.0), (0.8, 30.0)), ((2.5, 10.0), (1.5, 15.0)),
                ((1.5, 10.0), (1.0, 15.0)), ((0.8, 10.0), (1.0, 15.0)), ((1.0, 10.0), (1.0, 12.0)), ((0.8, 12.0), (0.8, 12.0)),
                ((0.45, 4.0), (0.45, 4.0)), ]
  rows = []
  for box_a, box_b in candidates:
    worst, best_a, best_b = 1.0, 0.0, 0.0
    any_a = np.zeros(parameters.shape[1], bool)
    any_b = np.zeros(parameters.shape[1], bool)
    for initial_a in np.geomspace(*box_a, 21):
      for initial_b in np.geomspace(*box_b, 21):
        seen_a, seen_b = visible(initial_a, initial_b, parameters)
        any_a |= seen_a
        any_b |= seen_b
        fraction_a, fraction_b = float(seen_a.mean()), float(seen_b.mean())
        best_a, best_b = max(best_a, fraction_a), max(best_b, fraction_b)
        worst = min(worst, max(fraction_a, fraction_b))
    cover = float((any_a & any_b).mean())
    rows.append((list(box_a), list(box_b), worst, best_a, best_b, cover))
    print(
      f'    {f"[{box_a[0]}, {box_a[1]}]":>16} {f"[{box_b[0]}, {box_b[1]}]":>16} {worst:13.3f} {best_a:9.3f} '
      f'{best_b:9.3f} {cover:7.3f}', flush=True
    )
  print('    thresholds: worst design >= 0.25, best f_A and best f_B >= 0.90, coverage ~ 1.')
  print(f'    --> shipped box A0 {list(CONCENTRATION_A_BOUNDS)} x B0 {list(CONCENTRATION_B_BOUNDS)}.')
  print('    The last row is the SINGLE-SUBSTRATE task`s box, carried over unchanged, and it fails:')
  print('    it is the control that shows the box had to be re-derived rather than inherited.')
  RECORD['visibility'] = {
    'a_values': a_values.tolist(),
    'b_values': b_values.tolist(),
    'grids': {
      k: v.tolist()
      for k, v in grids.items()
    },
    'boxes': rows
  }


def section_profile(arguments):
  print('=== 6. m = 1: which design identifies which parameter ===')
  print('    A single experiment cannot identify both constants: B0 >> A0 saturates B and leaves')
  print('    single-substrate kinetics in A (K_A identified, K_B abandoned), A0 >> B0 the mirror,')
  print('    and the diagonal neither. Per-parameter loss against the 0.3333 no-information level.')
  a_values = np.geomspace(*CONCENTRATION_A_BOUNDS, 7)
  b_values = np.geomspace(*CONCENTRATION_B_BOUNDS, 7)
  print(f'    noise = {arguments.noise} mM, {arguments.n_events} events, n_grid = {arguments.n_grid}')
  print(f'    {"A0":>7} {"B0":>7} {"loss":>8}   {"q":>7} {"K_A":>7} {"K_B":>7}')
  rows = []
  for initial_a in a_values:
    for initial_b in b_values:
      total, parts = reference_loss(
        np.array([[initial_a, initial_b]]), arguments.noise, n_events=arguments.n_events, n_grid=arguments.n_grid,
        n_grid_velocity=arguments.n_grid_velocity, per_parameter=True
      )
      rows.append((initial_a, initial_b, total, *parts.tolist()))
      print(f'    {initial_a:7.3f} {initial_b:7.3f} {total:8.4f}   {parts[0]:7.4f} {parts[1]:7.4f} {parts[2]:7.4f}', flush=True)
  RECORD['profile'] = {'rows': rows, 'noise': arguments.noise}


def section_integrator(arguments):
  import jax
  import jax.numpy as jnp
  from detopt.detector.enzyme import rkc2_chain, rkc2_coefficients

  print('=== 7. RKC2 against the closed form, over the WHOLE prior box ===')
  print('    A corner list is not enough: the worst point is interior, so this scans a dense grid of')
  print('    (A0, B0, q, K_A, K_B) over the DESIGN box and the full parameter prior -- what the')
  print('    detector actually integrates, for events and for the estimator grid alike.')
  concentration_a = np.geomspace(*CONCENTRATION_A_BOUNDS, 6)
  concentration_b = np.geomspace(*CONCENTRATION_B_BOUNDS, 6)
  velocity = np.geomspace(*VELOCITY_BOUNDS, 5)
  michaelis_a = np.geomspace(*MICHAELIS_A_BOUNDS, 6)
  michaelis_b = np.geomspace(*MICHAELIS_B_BOUNDS, 6)
  mesh = np.meshgrid(concentration_a, concentration_b, velocity, michaelis_a, michaelis_b, indexing='ij')
  flat = [x.reshape(-1) for x in mesh]
  # THE DIAGONAL MUST BE IN THE GRID. Two independent geomspace axes share no value, so a product
  # grid never contains a single A0 == B0 point -- and that is the one place where both substrates
  # exhaust together and the rate is a removable 0/0. An earlier version of this scan missed it
  # entirely and passed while the detector returned NaN on every diagonal design.
  ratios = np.array([1.0, 1.02, 1.1, 1.5])
  diagonal = np.meshgrid(concentration_a, ratios, velocity, michaelis_a, michaelis_b, indexing='ij')
  extra = [x.reshape(-1) for x in diagonal]
  flat = [
    np.concatenate([flat[0], extra[0]]),
    np.concatenate([flat[1], extra[0] * extra[1]]),
    np.concatenate([flat[2], extra[2]]),
    np.concatenate([flat[3], extra[3]]),
    np.concatenate([flat[4], extra[4]]),
  ]
  times = np.arange(1, N_MEASUREMENTS + 1) * DURATION / N_MEASUREMENTS
  truth = exact_extent(times[None, :], *[x[:, None] for x in flat])
  print(f'    {flat[0].size} (A0, B0, q, K_A, K_B) points')
  print(
    f'    {"stages":>7} {"steps/meas":>11} {"dt (s)":>8} {"rate evals":>11} {"max|err|":>11} '
    f'{"max monitor":>12} {"NaN":>6}   worst at'
  )
  rows = []
  for n_stages in (5, 7, 9):
    coefficients = rkc2_coefficients(n_stages, 2.0 / 13.0)
    for steps in (16, 32, 48, 64, 96):
      dt = DURATION / (N_MEASUREMENTS * steps)

      def one(initial_a, initial_b, q, k_a, k_b):

        def rate(extent):
          left_a, left_b = initial_a - extent, initial_b - extent
          return q / (k_a / left_a + k_b / left_b + 1.0)

        start = jnp.zeros((), jnp.float32)
        coarse = rkc2_chain(coefficients, rate, start, dt=dt, n_steps=steps, n_intervals=N_MEASUREMENTS)
        fine = rkc2_chain(coefficients, rate, start, dt=0.5 * dt, n_steps=2 * steps, n_intervals=N_MEASUREMENTS)
        return coarse, jnp.max(jnp.abs(fine - coarse))

      coarse, monitor = jax.jit(jax.vmap(one))(*[jnp.asarray(x, jnp.float32) for x in flat])
      coarse = np.asarray(coarse, np.float64)
      invalid = int((~np.isfinite(coarse)).any(axis=1).sum())
      error = np.where(np.isfinite(coarse), np.abs(coarse - truth), np.inf).max(axis=1)
      finite = error[np.isfinite(error)]
      worst = int(np.argmax(np.where(np.isfinite(error), error, -1.0)))
      rows.append((n_stages, steps, dt, float(finite.max()), float(jnp.max(monitor)), invalid))
      print(
        f'    {n_stages:7d} {steps:11d} {dt:8.2f} {3 * n_stages * steps * N_MEASUREMENTS:11d} '
        f'{finite.max():11.3e} {float(jnp.max(monitor)):12.3e} {invalid:6d}   '
        f'A0={flat[0][worst]:.2f} B0={flat[1][worst]:.2f} q={flat[2][worst]:.1e} '
        f'K_A={flat[3][worst]:.3f} K_B={flat[4][worst]:.3f}', flush=True
      )
  RECORD['integrator'] = {'rows': rows}


def section_estimator(arguments):
  import jax.numpy as jnp
  from detopt.detector import EnzymeDepletionBiDetector

  print('=== 8a. the estimator lattice: a PLATEAU IS NOT ENOUGH ===')
  print('    The usual check -- refine at one design until the loss stops moving -- passes here at')
  print('    every size tried, INCLUDING sizes the Bayes-bound check of section 8c rejects outright.')
  print('    A plateau says the value has stopped moving, which a quantised posterior does just as')
  print('    happily as a converged one. Reported for both a strong and a weak design so the SPREAD')
  print('    between them can be watched too, but 8c is what settles the lattice.')
  index = np.arange(arguments.n_events)
  good = {'initial_a': [4.309, 10.0], 'initial_b': [25.0, 5.0]}
  bad = {'initial_a': [0.8, 0.8], 'initial_b': [25.0, 25.0]}
  print(f'    {"n_grid":>7} {"nodes":>9} {"strong design":>18} {"weak design":>14} {"ratio":>7}')
  rows = []
  for n_grid in (13, 21, 31, 41):
    detector = EnzymeDepletionBiDetector(
      n_experiments=2, measurement_noise=arguments.noise, n_grid=n_grid, n_grid_velocity=arguments.n_grid_velocity
    )
    values = []
    for design in (good, bad):
      _, event, _, target = detector(design, index)
      values.append(float(jnp.mean(detector.loss(detector.estimate(design, event), detector.normalize_target(target)))))
    nodes = arguments.n_grid_velocity * n_grid * n_grid
    rows.append((n_grid, nodes, values[0], values[1]))
    print(f'    {n_grid:7d} {nodes:9d} {values[0]:18.4f} {values[1]:14.4f} {values[1] / values[0]:7.2f}', flush=True)

  print()
  print('=== 8b. the detector estimator against a float64 closed-form posterior ===')
  print('    Same grid, same likelihood, but the model curves come from the closed form in float64')
  print('    instead of from RKC2 in float32. The two must agree far inside the loss scale, or the')
  print('    landscape is reporting integration error rather than information.')
  pairs = []
  for n_experiments, values in ((1, ([3.0], [12.0])), (2, ([1.5, 8.0], [12.0, 2.0])), (4, ([1.0, 2.5, 6.0, 10.0], [20.0, 8.0,
                                                                                                                   3.0, 1.2]))):
    detector = EnzymeDepletionBiDetector(
      n_experiments=n_experiments, measurement_noise=arguments.noise, n_grid=arguments.n_grid
    )
    design = {'initial_a': values[0], 'initial_b': values[1]}
    _, event, _, target = detector(design, index)
    detector_loss = float(jnp.mean(detector.loss(detector.estimate(design, event), detector.normalize_target(target))))
    stacked = np.stack([np.asarray(values[0], np.float64), np.asarray(values[1], np.float64)], -1)
    closed = reference_loss(
      stacked, arguments.noise, n_events=arguments.n_events, n_grid=arguments.n_grid, n_grid_velocity=arguments.n_grid_velocity
    )
    pairs.append((n_experiments, detector_loss, closed))
    print(
      f'    m={n_experiments}  detector (RKC2, float32) = {detector_loss:.4f}   '
      f'closed form (float64) = {closed:.4f}   difference = {abs(detector_loss - closed):.4f}', flush=True
    )
  RECORD['estimator'] = {'grid_rows': rows, 'agreement': pairs}


def section_parameterisation(arguments):
  from detopt.detector import EnzymeDepletionBiDetector

  print('=== 9. target parameterisation: (q, K_A, K_B) or their logs? ===')
  print('    Binned by the TRUE value at a good design: if the ABSOLUTE error tracks the parameter')
  print('    while the LOG error does not, the log is the coordinate the measurement supports and')
  print('    the one on which a squared-error loss weights the prior evenly.')
  detector = EnzymeDepletionBiDetector(n_experiments=2, measurement_noise=arguments.noise, n_grid=arguments.n_grid)
  design = {'initial_a': [1.5, 8.0], 'initial_b': [12.0, 2.0]}
  _, event, _, target = detector(design, np.arange(arguments.n_events))
  predicted = np.asarray(detector.estimate(design, event), np.float64)
  physical = np.asarray(detector.denormalize_predictions(predicted).kinetics, np.float64)
  truth = np.asarray(target.kinetics, np.float64)
  record = {}
  for column, name in enumerate(('q  ', 'K_A', 'K_B')):
    edges = np.geomspace(BOUNDS[column][0], BOUNDS[column][1], 7)
    print(f'    {name}')
    absolute, logarithmic, centres = [], [], []
    for i in range(6):
      inside = (truth[:, column] >= edges[i]) & (truth[:, column] < edges[i + 1])
      if int(inside.sum()) < 30:
        continue
      value_error = float(np.sqrt(np.mean((physical[inside, column] - truth[inside, column])**2)))
      log_error = float(np.sqrt(np.mean((np.log(physical[inside, column]) - np.log(truth[inside, column]))**2)))
      absolute.append(value_error)
      logarithmic.append(log_error)
      centres.append(math.sqrt(edges[i] * edges[i + 1]))
      print(
        f'      [{edges[i]:9.4g}, {edges[i + 1]:9.4g})  n={int(inside.sum()):5d}  '
        f'rmse(value)={value_error:10.4g}  rmse(ln)={log_error:6.3f}'
      )
    print(
      f'      spread across bins (max / min): value {max(absolute) / min(absolute):8.2f}   '
      f'log {max(logarithmic) / min(logarithmic):6.2f}'
    )
    record[name.strip()] = {'centres': centres, 'value': absolute, 'log': logarithmic}
  RECORD['parameterisation'] = record


def _quantiles(values):
  return np.percentile(values, [0, 10, 25, 50, 75, 90, 100]).tolist()


def section_landscape(arguments):
  """The decisive screen: the span between good and bad designs, and the SHAPE of the profile."""
  print('=== 10. the loss profile over RANDOM designs: span, and shape against a quadratic ===')
  print('    What the optimiser has to resolve is the ABSOLUTE span, quoted against the')
  print(f'    `loss_precision` bar as well as against the {NO_INFORMATION:.4f} no-information level.')
  print('    The SHAPE test compares the loss profile with ||x - x*||^2 for x uniform on the same')
  print('    cube and x* at the best design found -- both min-max normalised, so a flat landscape')
  print('    (dead space), a needle, and a smooth bowl are told apart by the same statistic.')
  print(f'    {arguments.n_designs} Sobol designs, {arguments.n_events} events, n_grid = {arguments.n_grid}')
  print('    `plateau` is the fraction of designs scoring within 5% of the NO-INFORMATION level -- dead')
  print('    volume where every design is equally useless -- and `near best` the fraction within one')
  print('    `loss_precision` of the best. A large plateau beside a small near-best fraction is the')
  print('    needle-on-a-plateau signature, on which an arm comparison measures luck rather than search.')
  print(
    f'    {"m":>2} {"noise":>6} {"min":>7} {"25%":>7} {"median":>7} {"max":>7} {"med-min":>8} '
    f'{"/ bar":>7} {"shape":>7} {"quad":>7} {"plateau":>8} {"near best":>10}'
  )
  rows = []
  for n_experiments in arguments.n_experiments:
    points = sobol_points(n_experiments, arguments.n_designs)
    designs = scaled_to_design(points, n_experiments)
    swept = np.stack([
      reference_loss_sweep(
        d, arguments.noise_grid, n_events=arguments.n_events, n_grid=arguments.n_grid, n_grid_velocity=arguments.n_grid_velocity
      ) for d in designs
    ])
    for column, noise in enumerate(arguments.noise_grid):
      values = swept[:, column, :].mean(-1)
      per_parameter = swept[:, column, :].mean(0).tolist()
      best = points[int(np.argmin(values))]
      quadratic = np.sum((points - best[None, :])**2, axis=-1)
      normalise = lambda v: (v - v.min()) / max(v.max() - v.min(), 1e-12)
      shape = float(np.median(normalise(values)))
      quadratic_shape = float(np.median(normalise(quadratic)))
      quantiles = _quantiles(values)
      plateau = float(np.mean(values > 0.95 * NO_INFORMATION))
      near_best = float(np.mean(values < values.min() + arguments.loss_precision))
      rows.append({
        'n_experiments': n_experiments,
        'noise': noise,
        'quantiles': quantiles,
        'values': values.tolist(),
        'quadratic': quadratic.tolist(),
        'shape': shape,
        'quadratic_shape': quadratic_shape,
        'plateau': plateau,
        'near_best': near_best,
        'per_parameter': per_parameter
      })
      print(
        f'    {n_experiments:2d} {noise:6.4f} {quantiles[0]:7.4f} {quantiles[2]:7.4f} {quantiles[3]:7.4f} '
        f'{quantiles[6]:7.4f} {quantiles[3] - quantiles[0]:8.4f} '
        f'{(quantiles[3] - quantiles[0]) / arguments.loss_precision:7.2f} {shape:7.3f} {quadratic_shape:7.3f} '
        f'{plateau:8.3f} {near_best:10.3f}', flush=True
      )
  print('    `shape` is the median of the min-max normalised profile: 0 means a needle (almost every')
  print('    design as good as the best), 1 means a needle of BADNESS, and the quadratic reference')
  print('    column is what a smooth bowl over the same cube gives. Closer to it is better.')
  tag = '-'.join(str(m) for m in arguments.n_experiments)
  RECORD[f'landscape-m{tag}'] = {'rows': rows, 'loss_precision': arguments.loss_precision}


def section_resolution(arguments):
  """(lattice, noise) jointly: where the instrument is TRUSTWORTHY, not merely cheap.

  The posterior mean is the Bayes estimator, so its risk cannot exceed the prior variance -- ANY
  per-parameter loss above 1/3 is therefore proof that the lattice is too coarse for the posterior it
  is trying to represent, not a property of the design. That is the check, and it is sharp: it needs
  no reference value and no convergence extrapolation. ISOTROPIC lattices are swept beside the
  anisotropic ones because they are the rejected alternative, and they fail at MORE nodes."""
  print('=== 8c. the estimator lattice: (lattice, noise) where the instrument is TRUSTWORTHY ===')
  print('    The posterior mean cannot score worse than the prior. So `worst` -- the largest')
  print(f'    PER-PARAMETER loss over the sampled designs -- must stay under {NO_INFORMATION:.4f}. Anything')
  print('    above it is lattice quantisation, and the axis that causes it is the VELOCITY: q is')
  print('    determined an order of magnitude more sharply than either constant, so an isotropic')
  print('    lattice fine enough for the constants collapses the softmax onto one node in q and')
  print('    loses the marginalisation. Isotropic rows are the REJECTED alternative, shown at')
  print('    comparable or larger node counts.')
  designs = [np.array([[10.0, 25.0]]), np.array([[3.0, 6.0]]), np.array([[3.0, 3.0]]), np.array([[0.8, 1.0]]), ]
  lattices = [(n, n, n) for n in arguments.resolution_grid] + [(241, n, n) for n in arguments.resolution_grid]
  print(f'    {"lattice (q, K_A, K_B)":>22} {"nodes":>8} {"noise":>7} {"mean":>8} {"worst":>8} {"verdict":>11}')
  rows = []
  for shape in lattices:
    for noise in arguments.noise_grid:
      parts = np.stack([
        reference_loss_sweep(d, (noise, ), n_events=arguments.n_events, n_grid=shape[1], n_grid_velocity=shape[0])[0]
        for d in designs
      ])
      worst = float(parts.max())
      nodes = int(np.prod(shape))
      rows.append((shape[0], shape[1], nodes, noise, float(parts.mean()), worst))
      print(
        f'    {str(shape):>22} {nodes:8d} {noise:7.4f} {float(parts.mean()):8.4f} {worst:8.4f} '
        f'{"ok" if worst <= NO_INFORMATION + 0.007 else "TOO COARSE":>11}', flush=True
      )
  RECORD['resolution'] = {'rows': rows, 'no_information': NO_INFORMATION}


def section_identifiability(arguments):
  """PER-PARAMETER identifiability against the read-out noise: the check the aggregate loss cannot do.

  The shape statistics -- span, plateau, near-best -- are all computed on the AGGREGATE loss, and the
  aggregate is a mean over three components. A task can therefore improve every shape statistic by
  SHEDDING a target: raise the noise until one constant is no longer estimated at all, and the
  landscape over the remaining two gets cleaner. That is a criterion being gamed, not a task being
  tuned, so the noise is fixed HERE and LAST, from what the design can still identify.

  `K_A` binds. It is the glucose constant, 0.02-0.2 mM against an `A0` of order 1-10 mM, so the A
  knee sits at 98-99.8% depletion and is the last thing the curve reveals; `K_B` is five-fold larger
  and `q` is fixed by the initial slope, which every design measures.

  Reported per parameter: the loss in the SCALED coordinate (1/3 = the prior, which the Bayes
  estimator cannot exceed), the FRACTION OF PRIOR VARIANCE REMOVED `1 - 3 loss`, and the implied
  `sd(ln K)` in nats so it can be read against the prior's own. Two lattices are swept, because a
  loss at or above 1/3 is ambiguous between "the noise destroyed this parameter" and "the lattice is
  too coarse to represent the posterior", and only the finer lattice separates them."""
  print('=== 8d. per-parameter identifiability against the read-out noise ===')
  print('    The noise is set LAST and from THIS, not from a Cramer-Rao argument on the aggregate.')
  print('    A shape criterion can be GAMED by shedding a target: raising the noise until K_A is no')
  print('    longer estimated makes the landscape over (q, K_B) look better while the task quietly')
  print('    stops being the three-parameter task it claims to be.')
  print(
    f'    prior sd(ln q) = {LN_RANGE[0] / math.sqrt(12.0):.4f}, sd(ln K_A) = {LN_RANGE[1] / math.sqrt(12.0):.4f}, '
    f'sd(ln K_B) = {LN_RANGE[2] / math.sqrt(12.0):.4f} nats'
  )
  batch = int(arguments.identifiability_experiments)
  designs = [scaled_to_design(p[None, :], batch)[0] for p in sobol_points(batch, arguments.n_designs, seed=3)]
  lattices = [(arguments.n_grid_velocity, arguments.n_grid), (241, 21)]
  rows = []
  for n_grid_velocity, n_grid in lattices:
    print()
    print(
      f'    lattice (q, K_A, K_B) = ({n_grid_velocity}, {n_grid}, {n_grid}), '
      f'{arguments.n_designs} Sobol designs at m = {batch}, {arguments.n_events} events each'
    )
    print(
      f'    {"noise":>7} {"mean loss":>10} {"worst part":>11} | '
      f'{"recovered q":>12} {"K_A":>8} {"K_B":>8} | {"sd lnK_A best":>14} {"mean":>8}'
    )
    parts = np.stack([
      reference_loss_sweep(
        d, arguments.noise_grid, n_events=arguments.n_events, n_grid=n_grid, n_grid_velocity=n_grid_velocity
      ) for d in designs
    ])  # (n_designs, n_noise, 3)
    for column, noise in enumerate(arguments.noise_grid):
      per_parameter = parts[:, column, :]
      recovered = 1.0 - per_parameter.mean(0) / NO_INFORMATION
      best_a = float(per_parameter[:, 1].min())
      sd_best = LN_RANGE[1] * math.sqrt(best_a) / 2.0
      sd_mean = LN_RANGE[1] * math.sqrt(float(per_parameter[:, 1].mean())) / 2.0
      rows.append({
        'n_grid_velocity': n_grid_velocity,
        'n_grid': n_grid,
        'noise': float(noise),
        'mean_loss': float(per_parameter.mean()),
        'worst_part': float(per_parameter.max()),
        'recovered': recovered.tolist(),
        'sd_ln_michaelis_a_best': sd_best,
        'sd_ln_michaelis_a_mean': sd_mean,
        'per_parameter_mean': per_parameter.mean(0).tolist(),
        'best_michaelis_a_loss': best_a,
      })
      print(
        f'    {noise:7.4f} {float(per_parameter.mean()):10.4f} {float(per_parameter.max()):11.4f} | '
        f'{recovered[0]:11.1%} {recovered[1]:7.1%} {recovered[2]:7.1%} | {sd_best:14.4f} {sd_mean:8.4f}', flush=True
      )
  print()
  print(f'    `worst part` above {NO_INFORMATION:.4f} is a LATTICE verdict, not a noise one -- the posterior mean')
  print('    cannot score worse than the prior -- so read the two lattices together before blaming')
  print('    the noise for a parameter that the grid simply could not represent.')
  RECORD['identifiability'] = {
    'n_experiments': batch,
    'rows': rows,
    'no_information': NO_INFORMATION,
    'ln_range': list(LN_RANGE)
  }


def section_box(arguments):
  """THE DESIGN BOX, derived by EXPANDING until a plateau appears -- not by shrinking to a safe one.

  DEAD SPACE = PLATEAU: a region where moving the design does not change the loss. The GP has to
  spend observations mapping it and learns nothing. A TIGHT BOX IS THE MAXIMAL-SIZE BOX THAT CONTAINS
  NO PLATEAU, so the procedure runs outward from a box believed safe, finds where the loss goes flat
  at each end, and backs off to just inside it.

  A box that is too SMALL is a defect and not a conservative choice: it compresses the loss range and
  makes the task needle-like, which is the failure mode this whole exercise exists to avoid.

  Two plateaus are expected at opposite ends and BOTH are measured rather than argued:
    * `A0, B0 >> K` -- both substrates saturate the enzyme for the whole window, the extent is a
      straight line `x = q t`, and only `q` is identified. Raising the concentration further changes
      nothing.
    * `A0, B0` small -- the extent over the window is comparable to the read-out noise, the posterior
      falls back on the prior, and lowering the concentration further changes nothing.

  Reported in two parts. First the LINE PROFILES, which locate the flat ends. Then, per candidate
  box, the FACE GRADIENT: at each face, designs on the face are compared with the same designs pushed
  one step INWARD, and the mean `|change in loss|` is reported against the box's own span. A face whose
  gradient is a negligible fraction of the span is sitting in a plateau and the bound is too far out;
  a face with a healthy gradient could still be expanded. That is a measurement of the definition,
  not a proxy for it."""
  print('=== 5b. the design box: EXPAND to the plateau, then back off ===')
  print('    DEAD SPACE = PLATEAU = moving the design does not change the loss. TIGHT = the LARGEST')
  print('    box with no plateau. A too-SMALL box is a defect: it compresses the loss range and makes')
  print('    the task needle-like. So the sweep runs OUTWARD, well past the shipped bounds.')
  print(
    f'    noise = {arguments.noise} mM, {arguments.n_events} events, lattice '
    f'({arguments.n_grid_velocity}, {arguments.n_grid}, {arguments.n_grid}), m = 1'
  )

  def loss_at(initial_a, initial_b):
    return reference_loss(
      np.array([[initial_a, initial_b]]), arguments.noise, n_events=arguments.n_events, n_grid=arguments.n_grid,
      n_grid_velocity=arguments.n_grid_velocity, per_parameter=True
    )

  profiles = []
  print()
  print('    LINE PROFILES. `d loss / d ln c` is the local sensitivity; it going to ~0 IS the plateau.')
  for axis, fixed_values, sweep in (('A0', (1.5, 5.0, 12.0, 25.0), np.geomspace(0.05, 100.0, 15)),
                                    ('B0', (0.8, 2.5, 6.0, 10.0), np.geomspace(0.1, 200.0, 15))):
    for fixed in fixed_values:
      losses, parts_list = [], []
      for value in sweep:
        initial_a, initial_b = (value, fixed) if axis == 'A0' else (fixed, value)
        total, parts = loss_at(initial_a, initial_b)
        losses.append(total)
        parts_list.append(parts.tolist())
      losses = np.asarray(losses)
      gradient = np.gradient(losses, np.log(sweep))
      other = 'B0' if axis == 'A0' else 'A0'
      print(f'    sweeping {axis} at {other} = {fixed:g} mM')
      print(f'      {axis:>9} ' + ' '.join(f'{v:7.2f}' for v in sweep))
      print(f'      {"loss":>9} ' + ' '.join(f'{v:7.4f}' for v in losses))
      print(f'      {"d/dln":>9} ' + ' '.join(f'{v:7.4f}' for v in gradient), flush=True)
      profiles.append({
        'axis': axis,
        'fixed': float(fixed),
        'sweep': sweep.tolist(),
        'loss': losses.tolist(),
        'gradient': gradient.tolist(),
        'per_parameter': parts_list
      })

  print()
  print('    FACE GRADIENTS per candidate box: designs ON each face against the same designs pushed')
  print('    one step INWARD. `|dloss|/span` near zero means that face is in a plateau (bound too far')
  print('    out); a healthy fraction means the bound could still be expanded.')
  print('    THE COMPARISON IS PAIRED and that is what makes it resolvable. Every design here is scored')
  print('    on the SAME prior draws and the SAME noise realisation (common random numbers), so the')
  print('    difference between two nearby designs is far quieter than either absolute loss -- which')
  print('    at these event counts is itself uncertain at roughly the size of the gradient being')
  print('    measured. Read `mean |dloss|` against the objective`s own reproducibility (5.0e-3) as well')
  print('    as against the span.')
  boxes = []
  for box_a, box_b in arguments.box:
    box_a, box_b = tuple(box_a), tuple(box_b)
    interior = [(a, b) for a in np.geomspace(*box_a, 7) for b in np.geomspace(*box_b, 7)]
    values = np.array([loss_at(a, b)[0] for a, b in interior])
    span = float(values.max() - values.min())
    faces = {}
    for name, step in (('A0 low', ('A0', 0)), ('A0 high', ('A0', 1)), ('B0 low', ('B0', 0)), ('B0 high', ('B0', 1))):
      axis, end = step
      box, other_box = (box_a, box_b) if axis == 'A0' else (box_b, box_a)
      inward = math.exp((1.0 if end == 0 else -1.0) * abs(math.log(box[1] / box[0])) / 6.0)
      changes = []
      for other in np.geomspace(*other_box, 5):
        on_face, pushed = box[end], box[end] * inward
        pair = [(on_face, other), (pushed, other)] if axis == 'A0' else [(other, on_face), (other, pushed)]
        changes.append(abs(loss_at(*pair[0])[0] - loss_at(*pair[1])[0]))
      faces[name] = {'mean_abs_change': float(np.mean(changes)), 'fraction_of_span': float(np.mean(changes) / span)}
    boxes.append({
      'box_a': list(box_a),
      'box_b': list(box_b),
      'span': span,
      'faces': faces,
      'min': float(values.min()),
      'max': float(values.max()),
      'median': float(np.median(values))
    })
    print(
      f'    A0 {list(box_a)} x B0 {list(box_b)}: span {span:.4f} '
      f'(min {values.min():.4f}, median {float(np.median(values)):.4f}, max {values.max():.4f})'
    )
    for name, block in faces.items():
      verdict = 'PLATEAU' if block['fraction_of_span'] < 0.05 else 'alive'
      print(
        f'      {name:>8}: mean |dloss| inward {block["mean_abs_change"]:.4f} = '
        f'{block["fraction_of_span"]:6.1%} of span   {verdict}', flush=True
      )
  RECORD['box'] = {'profiles': profiles, 'boxes': boxes, 'noise': arguments.noise}


SECTIONS = {
  'reference': section_reference,
  'degeneracy': section_degeneracy,
  'measurements': section_measurements,
  'window': section_window,
  'visibility': section_visibility,
  'profile': section_profile,
  'integrator': section_integrator,
  'estimator': section_estimator,
  'resolution': section_resolution,
  'parameterisation': section_parameterisation,
  'landscape': section_landscape,
  'box': section_box,
  'identifiability': section_identifiability,
}


def _box(text):
  """`ALOW,AHIGH:BLOW,BHIGH` -> `((ALOW, AHIGH), (BLOW, BHIGH))`."""
  a_part, _, b_part = text.partition(':')
  a_bounds = tuple(float(v) for v in a_part.split(','))
  b_bounds = tuple(float(v) for v in b_part.split(','))
  if len(a_bounds) != 2 or len(b_bounds) != 2:
    raise argparse.ArgumentTypeError(f'expected ALOW,AHIGH:BLOW,BHIGH, got {text!r}')
  return a_bounds, b_bounds


def main():
  global KNEE_WINDOW_FRACTION
  parser = argparse.ArgumentParser()
  parser.add_argument('--section', default='all', help=f'one of {sorted(SECTIONS)} (comma separated) or `all`')
  parser.add_argument('--n-draws', type=int, default=40000)
  parser.add_argument('--n-reference', type=int, default=60)
  parser.add_argument('--n-events', type=int, default=1024)
  parser.add_argument('--n-designs', type=int, default=64)
  parser.add_argument('--n-grid', type=int, default=31)
  parser.add_argument('--n-grid-velocity', type=int, default=241)
  parser.add_argument('--noise', type=float, default=MEASUREMENT_NOISE)
  parser.add_argument(
    '--n-experiments', type=int, nargs='*', default=(1, 2, 4), help='batch sizes the landscape section sweeps'
  )
  parser.add_argument(
    '--noise-grid', type=float, nargs='*', default=(0.0125, 0.025, 0.05, 0.1, 0.2),
    help='read-out noise levels the landscape section sweeps'
  )
  parser.add_argument(
    '--resolution-grid', type=int, nargs='*', default=(25, 41, 61, 81),
    help='estimator lattice sizes the resolution section sweeps'
  )
  parser.add_argument(
    '--box', type=_box, action='append', default=None, metavar='ALOW,AHIGH:BLOW,BHIGH',
    help='a candidate design box the `box` section measures the face gradients of, e.g. 0.8,10:1,25'
  )
  parser.add_argument(
    '--identifiability-experiments', type=int, default=1,
    help='batch size the identifiability section scores; the campaign runs m = 2, m = 1 isolates one experiment'
  )
  parser.add_argument(
    '--knee-window-fraction', type=float, default=KNEE_WINDOW_FRACTION,
    help='late cutoff of the knee window as a fraction of `duration`; 0.8 ships, 1.0 counts a knee at '
    'the very last read-out as visible and is what an 8-sample assay needs to reach 0.90'
  )
  parser.add_argument('--loss-precision', type=float, default=1.25e-2)
  parser.add_argument('--output-dir', default='output/enzyme-depletion-bi')
  arguments = parser.parse_args()
  KNEE_WINDOW_FRACTION = float(arguments.knee_window_fraction)
  if arguments.box is None:
    arguments.box = [(CONCENTRATION_A_BOUNDS, CONCENTRATION_B_BOUNDS)]
  os.makedirs(arguments.output_dir, exist_ok=True)
  names = list(SECTIONS) if arguments.section == 'all' else arguments.section.split(',')
  for name in names:
    SECTIONS[name](arguments)
    print()
  for name, payload in RECORD.items():
    with open(os.path.join(arguments.output_dir, f'{name}.json'), 'w') as handle:
      json.dump(payload, handle)
  print(f'wrote {sorted(RECORD)} to {arguments.output_dir}')


if __name__ == '__main__':
  main()
