#!/usr/bin/env python3
"""Validate the `growth` detector before it is screened -- the checks of
output/agent-docs/benchmark-acceptance.md section 4.1-4.3 and section 9 of the candidate plan.

    srun --gres=shard:1 --cpus-per-task=4 -u python scripts/validate_growth.py

EVERYTHING GOES THROUGH SLURM, including "quick" checks: five bare `python -c` validations once drove
this 12-core box to load 44 and made every concurrent screen 10-14x slower.

Stages:

  physics    builds, simulates, read-outs finite, targets inside the prior, and the CONTRACT: the
             event is seeded from `event_index` ALONE, so the same event under two designs is the
             same strain and the same noise draw and the target is a property of the event.
  ctmi       the temperature response is the curve the plan claims, over the WHOLE prior and not for
             one hand-picked strain: mu(T_opt) = mu_opt, single-peaked at T_opt, zero outside
             (T_min, T_max), and 4-6x steeper above the optimum than below.
  integrator the dt/2 assert: the WORST corner of the prior x box (the sharpest substrate-exhaustion
             elbow) and a random sample over both, against `integration_tolerance`; plus the measured
             RKC2 stability boundary of the implemented step, and a convergence chain in dt.
  fisher     identifiability of T_opt, marginalised over the nuisances -- the design-quality ordering
             the physics predicts (bracketing beats one-sided beats dead), and the two concentration
             axes shown to be traps that can waste a culture at a perfectly good temperature.
  nogrowth   the fraction of RANDOM designs whose every culture is outside every strain's growth
             window, i.e. the only genuinely uninformative designs.
  score      the GBDT proxy: the no-information CEILING from a deliberately uninformative design, and
             a KNOWN-GOOD against a KNOWN-BAD design. If those two do not separate, the target is not
             in the data and no campaign will fix it -- this is the calibration that catches an
             encoding mistake (a scaled design passed where a nominal one was wanted makes EVERY
             design score at the ceiling).
"""
import argparse
import math
import os
import time


# BLAS/OpenMP size their thread pools at import time and default to every core on the machine, which
# is wrong under a scheduler: SLURM says WHICH cores this job may use, not how many threads to start.
_allocated = os.environ.get('SLURM_CPUS_PER_TASK', '4')
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
  os.environ.setdefault(_variable, _allocated)

import numpy as np

import jax
import jax.numpy as jnp

import detopt.detector
import detopt.utils.config
from detopt.detector.growth import PARAMETER_NAMES, cardinal_rate

CONFIG = 'config/detector/growth.yaml'


def build(**overrides):
  config = detopt.utils.config.load_config(CONFIG)
  config['growth'] = dict(config['growth'], **overrides)
  return detopt.detector.from_config(config)


def nominal(detector, temperatures, inoculum, substrate):
  """A flat NOMINAL design: one temperature per culture (scalar broadcast), one inoculum, one
  substrate. NOMINAL, not scaled -- the detector's own units."""
  m = detector.n_experiments
  return np.concatenate([
    np.broadcast_to(np.asarray(temperatures, np.float32), (m,)),
    np.full(m, inoculum, np.float32),
    np.full(m, substrate, np.float32)
  ]).astype(np.float32)


# --------------------------------------------------------------------------------------------- #
def physics(detector):
  print('== physics ==')
  print(f'  read-out times (h): {np.array2string(np.asarray(detector.measurement_times), precision=1)}')
  bounds = dict(detector._parameter_ranges)
  n_events = 4096
  index = np.arange(n_events, dtype=np.int64)
  centre = np.asarray(
    detector.flatten_design(detector.to_nominal(np.full(detector.design_dim(), 0.5, np.float32))), np.float32
  )
  ground_truth, event, mask, target = detector(centre, index)
  measurements = np.asarray(event.measurements)
  parameters = np.asarray(ground_truth.parameters)
  print(f'  centre of the box (nominal): T {centre[:detector.n_experiments]} C, '
        f'N0 {centre[detector.n_experiments]:.4g} OD, S0 {centre[2 * detector.n_experiments]:.4g} g/L')
  print(f'  read-outs finite: {bool(np.isfinite(measurements).all())}, '
        f'range [{measurements.min():.4f}, {measurements.max():.4f}] OD, ceiling {detector.od_ceiling:.2f} OD')
  print(f'  read-outs >= detection limit: {bool((measurements >= detector.detection_limit - 1e-6).all())}')
  print(f'  mask all-valid (element == culture): {bool((np.asarray(mask) == 1).all())}, shape {tuple(mask.shape)}')

  optimum = np.asarray(target.optimal_temperature).ravel()
  low, high = detector.optimum_bounds
  print(f'  target inside its prior [{low}, {high}]: {bool(((optimum >= low) & (optimum <= high)).all())}, '
        f'mean {optimum.mean():.2f} C')
  for column, name in enumerate(PARAMETER_NAMES):
    lo, hi = bounds[name]
    inside = bool(((parameters[:, column] >= lo) & (parameters[:, column] <= hi)).all())
    print(f'    {name:<14} drawn in [{lo}, {hi}]: {inside}')
  normalised = np.asarray(detector.normalize_target(target))
  print(f'  normalised target: mean {normalised.mean():+.4f}, variance {normalised.var():.4f} '
        f'(1/3 for a uniform prior -- this is the no-information ceiling)')

  # THE CONTRACT: the event is seeded from `event_index` ALONE. Two very different designs must
  # return the SAME strain and the SAME target for the same index -- otherwise a design could move
  # its own label, and every loss comparison across designs would be meaningless.
  other = nominal(detector, 52.0, 2.0e-3, 0.9)
  ground_other, event_other, _, target_other = detector(other, index)
  same_target = bool(np.array_equal(np.asarray(target_other.optimal_temperature), np.asarray(target.optimal_temperature)))
  same_strain = bool(np.array_equal(np.asarray(ground_other.parameters), parameters))
  print(f'  common random numbers: same strain under a different design {same_strain}, same target {same_target}')
  print(f'  the read-out DOES respond to the design: max |OD - OD\'| = '
        f'{np.abs(np.asarray(event_other.measurements) - measurements).max():.4f} OD')
  # The noise realisation itself must also be common: a design that leaves the biomass unchanged must
  # leave the reading unchanged. Two designs whose temperatures are both far ABOVE every T_max make
  # exactly that pair -- neither grows, so any difference in the reading is a difference in the noise.
  dead_a, dead_b = nominal(detector, 59.9, 5.0e-3, 0.5), nominal(detector, 59.0, 5.0e-3, 0.5)
  _, event_a, _, _ = detector(dead_a, index[:512])
  _, event_b, _, _ = detector(dead_b, index[:512])
  print(f'  noise draw is design-independent: max |OD - OD\'| on two non-growing designs = '
        f'{np.abs(np.asarray(event_a.measurements) - np.asarray(event_b.measurements)).max():.2e} OD')

  # A culture above T_max never grows; a culture at T_opt reaches its plateau inside `duration`.
  probe = np.arange(512, dtype=np.int64)
  drawn = np.asarray(detector(centre, probe)[0].parameters)
  T_opt_drawn, T_max_drawn, yields = drawn[:, 0], drawn[:, 0] + drawn[:, 2], drawn[:, 5]
  m, inoculum, substrate = detector.n_experiments, 1.0e-2, 0.5

  def per_event(temperature):
    """One design PER EVENT: every culture of strain i at strain i's own temperature."""
    return np.concatenate([
      np.repeat(np.asarray(temperature)[:, None], m, axis=1),
      np.full((len(probe), m), inoculum), np.full((len(probe), m), substrate)
    ], axis=1).astype(np.float32)

  _, hot_event, _, _ = detector(per_event(T_max_drawn + 2.0), probe)
  hot = np.asarray(hot_event.measurements)
  print(f'  T = T_max + 2 K: every read-out within noise of the inoculum '
        f'{bool((hot <= inoculum + 5.0 * detector.measurement_noise).all())} '
        f'(max {hot.max():.4f} OD, inoculum {inoculum} OD)')
  _, warm_event, _, _ = detector(per_event(T_opt_drawn), probe)
  warm = np.asarray(warm_event.measurements)[:, 0, :]
  capacity = inoculum + yields * substrate
  print(f'  T = T_opt: reached >95% of the carrying capacity by {detector.duration} h in '
        f'{100.0 * (warm[:, -1] > 0.95 * capacity).mean():.1f}% of strains '
        f'(the plan asks a MID-range culture to finish)')
  monotone = np.diff(warm, axis=-1) > -6.0 * detector.measurement_noise
  print(f'  T = T_opt: read-outs non-decreasing within noise in {100.0 * monotone.all(axis=-1).mean():.1f}% of strains')


# --------------------------------------------------------------------------------------------- #
def ctmi(detector):
  """The temperature response, over the WHOLE prior. Checking one hand-picked strain would miss a
  parameterisation that goes singular in a corner: the CTMI's denominator is a quadratic in T, and a
  strain for which it vanishes INSIDE (T_min, T_max) would produce a pole where the plan claims a
  smooth peak."""
  print('== CTMI over the whole strain prior ==')
  bounds = dict(detector._parameter_ranges)
  rng = np.random.default_rng(3)
  n = 4096
  T_opt = rng.uniform(*bounds['T_opt'], n)
  T_min = T_opt - rng.uniform(*bounds['delta_min'], n)
  T_max = T_opt + rng.uniform(*bounds['delta_max'], n)
  mu_opt = np.exp(rng.uniform(*bounds['log_mu_opt'], n))

  # 0.1 C, not finer: the curve is flat to float32 at its top, so on a 0.01 C grid the DIFFERENCES
  # near the peak are pure roundoff and any monotonicity test reads noise instead of the curve.
  grid = np.linspace(*detector.temperature_bounds, 451, dtype=np.float32)
  rate = np.asarray(
    jax.jit(jax.vmap(cardinal_rate, in_axes=(None, 0, 0, 0, 0)))(
      jnp.asarray(grid), *[jnp.asarray(v, jnp.float32) for v in (T_min, T_opt, T_max, mu_opt)]
    )
  )
  # A float32 COLLAR around each cardinal temperature: `cardinal_rate` decides "inside" in float32
  # while this mask is built in float64, so a grid point within an ulp of T_min or T_max can be
  # inside for one and outside for the other. That is a tie in the last bit, not a physics claim --
  # 1 point in 1.2e7 on this sample -- so the positivity test is stated OUTSIDE the collar.
  collar = 1.0e-4
  inside = (grid[None, :] > T_min[:, None]) & (grid[None, :] < T_max[:, None])
  strictly = (grid[None, :] > T_min[:, None] + collar) & (grid[None, :] < T_max[:, None] - collar)
  print(f'  finite everywhere: {bool(np.isfinite(rate).all())}, max {rate.max():.3f} /h '
        f'against max mu_opt {mu_opt.max():.3f} /h')
  print(f'  zero outside (T_min, T_max): {bool((rate[~inside] == 0.0).all())}, '
        f'strictly positive inside (>{collar} C from either edge): {bool((rate[strictly] > 0.0).all())}')
  peak = grid[np.argmax(rate, axis=1)]
  print(f'  argmax == T_opt: max |argmax - T_opt| = {np.abs(peak - T_opt).max():.3f} C (grid step '
        f'{grid[1] - grid[0]:.3f} C)')
  at_optimum = np.asarray(cardinal_rate(jnp.asarray(T_opt, jnp.float32), *[jnp.asarray(v, jnp.float32)
                                                                          for v in (T_min, T_opt, T_max, mu_opt)]))
  print(f'  mu(T_opt) == mu_opt: max relative error {np.abs(at_optimum / mu_opt - 1.0).max():.2e}')
  # Single-peaked: the curve rises up to its maximum and falls after it, to a float32 tolerance. A
  # sign-change count would be defeated by roundoff on the flat top; this is the property that
  # matters -- no second lobe anywhere in the box.
  argmax = np.argmax(rate, axis=1)
  tolerance = 1.0e-7
  rises = np.array([bool((np.diff(rate[i, :argmax[i] + 1]) >= -tolerance).all()) for i in range(n)])
  falls = np.array([bool((np.diff(rate[i, argmax[i]:]) <= tolerance).all()) for i in range(n)])
  print(f'  single-peaked (rises to the maximum, falls after it, tol {tolerance:g} /h): '
        f'{100.0 * (rises & falls).mean():.1f}% of {n} strains')

  three = 3.0
  below = np.asarray(cardinal_rate(jnp.asarray(T_opt - three, jnp.float32), *[jnp.asarray(v, jnp.float32)
                                                                             for v in (T_min, T_opt, T_max, mu_opt)]))
  above = np.asarray(cardinal_rate(jnp.asarray(T_opt + three, jnp.float32), *[jnp.asarray(v, jnp.float32)
                                                                             for v in (T_min, T_opt, T_max, mu_opt)]))
  slope_below, slope_above = (mu_opt - below) / three, (mu_opt - above) / three
  ratio = slope_above / slope_below
  print(f'  flank asymmetry, LOCAL: |dmu/dT| 3 K above / 3 K below the optimum, median '
        f'{np.median(ratio):.1f}x, range [{ratio.min():.1f}, {ratio.max():.1f}]')
  # The GLOBAL asymmetry -- the plan's "gradual rise over 25-35 K, collapse over 5-8 K" -- is a
  # statement about the flank WIDTHS, not about the slope 3 K from a quadratic top. Measured as the
  # distance from T_opt to half of mu_opt on each side, which is what decides how much of the box a
  # culture on each flank can resolve.
  half = 0.5 * mu_opt[:, None]
  width_below = np.array([T_opt[i] - grid[np.nonzero(rate[i, :argmax[i] + 1] >= half[i])[0][0]] for i in range(n)])
  width_above = np.array([grid[argmax[i] + np.nonzero(rate[i, argmax[i]:] >= half[i])[0][-1]] - T_opt[i]
                          for i in range(n)])
  print(f'  flank asymmetry, GLOBAL: half-maximum width {np.median(width_below):.1f} C below vs '
        f'{np.median(width_above):.1f} C above (median ratio {np.median(width_below / width_above):.1f}x) '
        f'-- the asymmetry the design must exploit')


# --------------------------------------------------------------------------------------------- #
def integrator(detector, n_cultures=262144):
  """The dt/2 chain, on the corner that makes the substrate-exhaustion elbow sharpest.

  The random sample is deliberately LARGE. The assert is a hard failure that aborts whatever is
  running, and a screen touches ~6e6 cultures per batch size (512 designs x 4096 events x m), so a
  margin measured on a few thousand draws says nothing about the tail that will actually be sampled.
  """
  print('== integrator ==')
  bounds = dict(detector._parameter_ranges)

  def error_of(steps, mu, K_S, Y, N0, S0):
    d = build(n_steps_per_measurement=int(steps))
    solve = jax.jit(jax.vmap(lambda mu, K_S, Y, N0, S0: d._integrate(N0, N0 + Y * S0, mu, K_S, Y)))
    biomass, error = solve(*[jnp.asarray(v, jnp.float32) for v in (mu, K_S, Y, N0, S0)])
    return np.asarray(biomass), np.asarray(error)

  # The elbow has width tau = Y*K_S/(mu*C) and the error of a 2nd-order scheme there scales as
  # (Y*K_S)*(dt/tau)^2, i.e. as (dt*mu*C)^2/(Y*K_S) -- so the worst corner is the fastest strain with
  # the weakest substrate affinity, on the design that makes the most biomass.
  mu_max, K_S_min = math.exp(bounds['log_mu_opt'][1]), math.exp(bounds['log_K_S'][0])
  corner = dict(
    mu=np.array([mu_max]), K_S=np.array([K_S_min]), Y=np.array([bounds['biomass_yield'][1]]),
    N0=np.array([detector.inoculum_bounds[1]]), S0=np.array([detector.substrate_bounds[1]])
  )
  capacity = corner['N0'][0] + corner['Y'][0] * corner['S0'][0]
  tau = corner['Y'][0] * K_S_min / (mu_max * capacity)
  dt = detector.duration / (detector.n_measurements * detector.n_steps_per_measurement)
  jacobian_bound = mu_max * capacity / (corner['Y'][0] * K_S_min)
  print(f'  worst corner: mu_opt {mu_max:.2f}/h, K_S {K_S_min:.3f} g/L, Y {corner["Y"][0]:.2f}, '
        f'C {capacity:.2f} OD -> elbow width tau {60 * tau:.2f} min, dt {3600 * dt:.1f} s = {dt / tau:.3f} tau')
  print(f'  |df/dN| <= mu*C/(Y*K_S) = {jacobian_bound:.0f} /h, z = |df/dN| dt = {jacobian_bound * dt:.2f}')

  print('  dt convergence at that corner (the RETURNED chain is the dt one):')
  previous = None
  for steps in (20, 40, 80, 160, 320, 640):
    _, error = error_of(steps, **corner)
    ratio = '' if previous is None or float(error[0]) == 0.0 else f'  ({previous / float(error[0]):.1f}x better)'
    print(f'    n_steps_per_measurement {steps:4d}  dt '
          f'{3600 * detector.duration / (detector.n_measurements * steps):6.1f} s'
          f'  max|fine-coarse| {float(error[0]):.3e} OD{ratio}')
    previous = float(error[0])

  # A random sample over the WHOLE prior x box, not just the corner.
  rng = np.random.default_rng(0)
  n = int(n_cultures)
  sample = dict(
    mu=np.exp(rng.uniform(*bounds['log_mu_opt'], n)), K_S=np.exp(rng.uniform(*bounds['log_K_S'], n)),
    Y=rng.uniform(*bounds['biomass_yield'], n),
    N0=np.exp(rng.uniform(*np.log(detector.inoculum_bounds), n)),
    S0=np.exp(rng.uniform(*np.log(detector.substrate_bounds), n))
  )
  # mu is the rate AT the culture temperature, not mu_opt, so the sample must include cool cultures
  # too: scale the drawn mu_opt by a uniform fraction of the CTMI's range.
  sample['mu'] = sample['mu'] * rng.uniform(0.0, 1.0, n)
  _, error = error_of(detector.n_steps_per_measurement, **sample)
  corner_error = float(error_of(detector.n_steps_per_measurement, **corner)[1][0])
  print(f'  random prior x box, {n} cultures: max {error.max():.3e} OD, '
        f'99.99th pct {np.quantile(error, 0.9999):.3e}, 99.9th pct {np.quantile(error, 0.999):.3e}, '
        f'median {np.median(error):.3e}')
  print(f'  integration_tolerance {detector.integration_tolerance:.3e} OD '
        f'(10% of measurement_noise {detector.measurement_noise}); worst corner {corner_error:.3e} OD, '
        f'i.e. {detector.integration_tolerance / max(error.max(), corner_error, 1e-30):.0f}x inside it')
  print(f'  ASSERT holds on every call: {bool(max(error.max(), corner_error) <= detector.integration_tolerance)}')

  # The stability boundary is MEASURED for the implemented step, in float32, on this device: apply it
  # to y' = -lambda y and find the largest z = lambda dt with |R(z)| <= 1.
  z = np.linspace(0.01, 40.0, 4000)
  step = jax.jit(jax.vmap(lambda z: detector.rkc2_step(lambda y: -z * y, jnp.float32(1.0), jnp.float32(1.0))))
  amplification = np.abs(np.asarray(step(jnp.asarray(z, jnp.float32))))
  stable = z[amplification <= 1.0]
  boundary = float(stable.max()) if len(stable) > 0 else float('nan')
  print(f'  measured RKC2 stability boundary at {detector.n_stages} stages: z <= {boundary:.2f} '
        f'(float32, {jax.default_backend()}) -- margin {boundary / (jacobian_bound * dt):.1f}x')


# --------------------------------------------------------------------------------------------- #
def _readout(detector, theta, temperature, inoculum, substrate):
  """The NOISELESS, censored read-out of a whole batch for strain ``theta`` -- what the Fisher
  information differentiates. ``theta`` is (T_opt, delta_min, delta_max, log_mu_opt, log_K_S, Y)."""
  T_opt, delta_min, delta_max, log_mu_opt, log_K_S, biomass_yield = theta

  def run(T, N0, S0):
    mu = cardinal_rate(T, T_opt - delta_min, T_opt, T_opt + delta_max, jnp.exp(log_mu_opt))
    biomass, _ = detector._integrate(N0, N0 + biomass_yield * S0, mu, jnp.exp(log_K_S), biomass_yield)
    return biomass

  # The detection limit is part of the measurement: a culture below it carries NO information, and
  # `maximum` gives exactly that -- zero derivative on the censored branch.
  return jnp.maximum(jax.vmap(run)(temperature, inoculum, substrate), detector.detection_limit).ravel()


def fisher(detector):
  """Marginal posterior SD of T_opt, I_eff = I_tt - I_tn inv(I_nn) I_nt, averaged over the prior.

  The PRIOR's own precision is added to the information matrix (a uniform range (a, b) has variance
  (b-a)^2/12), so a nuisance a design says nothing about is bounded by its prior instead of making the
  matrix singular -- and the reported number is then a posterior SD in Celsius, directly comparable
  with the prior SD that a no-information design must return.
  """
  print('== identifiability (Fisher, marginalised over the 5 nuisances) ==')
  prior_variance = np.array([(hi - lo) ** 2 / 12.0 for _, (lo, hi) in detector._parameter_ranges])
  print(f'  prior SD of T_opt: {math.sqrt(prior_variance[0]):.2f} C  (what a no-information design returns)')

  n_strains = 256
  rng = np.random.default_rng(1)
  thetas = np.stack([rng.uniform(lo, hi, n_strains) for _, (lo, hi) in detector._parameter_ranges], axis=1)
  m = detector.n_experiments

  def marginal(temperature, inoculum, substrate):
    def one(theta):
      jacobian = jax.jacfwd(_readout, argnums=1)(
        detector, theta, jnp.asarray(temperature, jnp.float32), jnp.asarray(inoculum, jnp.float32),
        jnp.asarray(substrate, jnp.float32)
      )
      information = jacobian.T @ jacobian / (detector.measurement_noise ** 2)
      information = information + jnp.diag(1.0 / jnp.asarray(prior_variance, jnp.float32))
      effective = information[0, 0] - information[0, 1:] @ jnp.linalg.solve(information[1:, 1:], information[1:, 0])
      return 1.0 / jnp.sqrt(effective)

    return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(thetas, jnp.float32)))

  middling = (np.full(m, 1.0e-2, np.float32), np.full(m, 0.5, np.float32))
  designs = {
    'BRACKET (one below, one near, one on the sharp upper flank)': ([30.0, 40.0, 47.0],) + middling,
    'BRACKET, wide': ([25.0, 38.0, 50.0],) + middling,
    'ONE-SIDED, all below the population optima': ([18.0, 22.0, 26.0],) + middling,
    'ONE-SIDED, all above': ([52.0, 56.0, 60.0],) + middling,
    'NO BRACKET, all three at one temperature': ([38.0, 38.0, 38.0],) + middling,
    'GRID, uniform over the box': ([15.0, 37.5, 60.0],) + middling,
    'DEAD, above every T_max': ([58.0, 59.0, 60.0],) + middling,
    # The two concentration axes are traps, not decoration: the same bracketing temperatures are
    # wasted by an inoculum that cannot cross the detection limit in 12 h, or by a substrate that
    # runs out before the first read-out.
    'BRACKET but the inoculum is the smallest in the box': ([30.0, 40.0, 47.0], np.full(m, 2.0e-4, np.float32),
                                                            np.full(m, 0.5, np.float32)),
    'BRACKET but the substrate is the smallest in the box': ([30.0, 40.0, 47.0], np.full(m, 1.0e-2, np.float32),
                                                             np.full(m, 0.03, np.float32)),
  }
  for label, (temperature, inoculum, substrate) in designs.items():
    sd = marginal(np.asarray(temperature, np.float32), inoculum, substrate)
    print(f'    {np.array2string(np.asarray(temperature), precision=1):<22} median sigma(T_opt) '
          f'{np.median(sd):6.2f} C   mean {sd.mean():6.2f} C   <- {label}')


# --------------------------------------------------------------------------------------------- #
def nogrowth(detector):
  """The fraction of RANDOM designs on which EVERY culture is outside EVERY strain's growth window --
  the only genuinely uninformative designs in this system (plan, section 9.3)."""
  print('== no-growth designs ==')
  rng = np.random.default_rng(2)
  n_designs, n_strains = 4096, 512
  temperature = rng.uniform(*detector.temperature_bounds, (n_designs, detector.n_experiments))
  bounds = dict(detector._parameter_ranges)
  T_opt = rng.uniform(*bounds['T_opt'], n_strains)
  T_min = T_opt - rng.uniform(*bounds['delta_min'], n_strains)
  T_max = T_opt + rng.uniform(*bounds['delta_max'], n_strains)
  grows = (temperature[:, :, None] > T_min[None, None, :]) & (temperature[:, :, None] < T_max[None, None, :])
  per_strain = grows.any(axis=1)  # (designs, strains): at least one culture of the batch grows
  print(f'  designs where NO culture grows for ANY strain: {100.0 * (~per_strain.any(axis=1)).mean():.2f}%')
  print(f'  mean fraction of STRAINS with no growing culture, over designs: {100.0 * (~per_strain).mean():.1f}%')
  print(f'  mean fraction of CULTURES that grow, over designs and strains: {100.0 * grows.mean():.1f}%')


# --------------------------------------------------------------------------------------------- #
def score(detector, n_events, seed):
  """The GBDT proxy: the measured ceiling, and a known-good design against a known-bad one."""
  from detopt.bo.gbdt import score_design
  print(f'== proxy score (GBDT, {n_events} events) ==')
  m = detector.n_experiments

  index = np.arange(n_events, dtype=np.int64)
  _, _, _, target = detector(nominal(detector, 37.5, 1.0e-2, 0.5), index)
  normalised = np.asarray(detector.normalize_target(target), np.float32)
  print(f'  target variance (the analytic no-information level): {float(normalised.var()):.4f}')

  designs = {
    'UNINFORMATIVE: every culture above every T_max, inoculum below the detection limit':
      nominal(detector, 60.0, 2.0e-4, 0.03),
    'UNINFORMATIVE: every culture above every T_max, workable inoculum/substrate':
      nominal(detector, 60.0, 1.0e-2, 0.5),
    'KNOWN-BAD: no bracket, all three cultures at one cool temperature':
      nominal(detector, 20.0, 1.0e-2, 0.5),
    'KNOWN-BAD: no bracket, all three at one temperature near the population mean':
      nominal(detector, 38.0, 1.0e-2, 0.5),
    'KNOWN-BAD: the same bracketing temperatures, wasted by the smallest inoculum in the box':
      nominal(detector, np.linspace(30.0, 47.0, m), 2.0e-4, 0.5),
    'KNOWN-GOOD: bracketing, one culture on the sharp upper flank':
      nominal(detector, np.linspace(30.0, 47.0, m), 1.0e-2, 0.5),
    'KNOWN-GOOD: bracketing, wide':
      nominal(detector, np.linspace(25.0, 50.0, m), 1.0e-2, 0.5),
  }
  for label, design in designs.items():
    started = time.time()
    result = score_design(detector, design, n_events=n_events, event_offset=0, seed=seed)
    print(f'    loss {result.loss:.4f} +- {result.sem:.4f}  (train {result.train:.4f} val {result.val:.4f}, '
          f'{result.n_learners} learners, {time.time() - started:.1f} s)  <- {label}')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--stage', nargs='*',
                      default=['physics', 'ctmi', 'integrator', 'fisher', 'nogrowth', 'score'])
  parser.add_argument('--n-events', type=int, default=4096)
  parser.add_argument('--n-experiments', type=int, default=None)
  parser.add_argument('--n-cultures', type=int, default=262144,
                      help='cultures in the integrator stage\'s random prior x box sample')
  parser.add_argument('--seed', type=int, default=0)
  arguments = parser.parse_args()

  overrides = {} if arguments.n_experiments is None else {'n_experiments': arguments.n_experiments}
  detector = build(**overrides)
  print(f'growth detector: {detector.n_experiments} cultures, design_dim {detector.design_dim()}, '
        f'backend {jax.default_backend()}, threads {os.environ["OMP_NUM_THREADS"]}')
  stages = {'physics': physics, 'ctmi': ctmi, 'fisher': fisher, 'nogrowth': nogrowth}
  for stage in arguments.stage:
    if stage in stages:
      stages[stage](detector)
    elif stage == 'integrator':
      integrator(detector, arguments.n_cultures)
    elif stage == 'score':
      score(detector, arguments.n_events, arguments.seed)
    else:
      raise SystemExit(f'unknown stage {stage}')


if __name__ == '__main__':
  main()
