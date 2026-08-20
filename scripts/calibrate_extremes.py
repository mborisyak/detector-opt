#!/usr/bin/env python3
"""Calibration probes for the binary inhibitor-MECHANISM task (`extremes`).

CPU only. Read-only with respect to the shipped configs: every setting under test is an OVERRIDE of
`config/detector/enzyme_extremes.yaml`, never an edit of it.

    python scripts/calibrate_extremes.py <section> [key=value ...]

WHAT EACH SECTION MUST SEPARATE, stated before it is run.

  physics     the claim is that the COMPETITIVE branch is outrun by substrate and the UNCOMPETITIVE
              branch is not. The probe must separate "this branch changes the read-out" from "the
              enzyme is simply slower", so it holds the enzyme fixed and switches ONE branch on at a
              time against the same enzyme's uninhibited curve, and reports the largest change in
              read-out-noise units as a function of the ATP level.

  mirror      the sharper claim, and the one that decides whether ONE experiment can work at all:
              switching the class is EXACTLY a reparameterisation of the enzyme, K_B -> K_B r,
              Ki_D -> Ki_D r, k_cat -> k_cat r with r = (1 + I k2) / (1 + I k1), so at a single
              inhibitor level no read-out can tell the classes apart except through the PRIOR's
              support. The probe must separate "the two curves agree" from "the solver is noisy", so
              it compares the mirrored compound against the shifted enzyme at zero compound and
              reports the disagreement against the integration guard.

  regime      how often a design can discriminate AT ALL. A draw is in the discriminating regime when
              the class-mirrored read-out differs from the drawn one by at least one read-out sigma
              somewhere. The probe must separate the box from the draw, so it reports the fraction at
              the WORST corner of the box, not at a typical design.

  guard       the dt / dt-half integration guard must be satisfied at the event count the instrument
              draws, and it must be able to FIRE: the probe walks the step count down until it
              crosses the tolerance.

  profile     one design coordinate at a time against the instrument's loss, every other coordinate
              held at a stated value. It must separate a coordinate that moves the objective from one
              that does not, so it reports the loss with its standard error.

  resolution  the read-out cadence. It must separate "more read-outs resolve the transition" from
              "more read-outs are more data", so it reports the loss at a fixed total integration
              step count and a fixed noise.

  noise       the read-out noise, chosen so that the loss over RANDOM designs is a bowl rather than a
              flat line. The probe reports the whole distribution over Sobol designs, not its mean.

  convergence the Monte-Carlo marginal. It must separate "the answer" from "the sample size", so it
              reports the loss and the effective sample size against the library size.
"""

import json
import math
import os
import sys

import numpy as np

import jax

jax.config.update('jax_platform_name', 'cpu')

from detopt.analytic import MechanismInstrument, build_detector, sobol_designs
from detopt.utils.config import load_config

REFERENCE = 'config/detector/enzyme_extremes.yaml'
COORDINATES = ('enzyme_fraction', 'substrate_B', 'inhibitor', 'temperature')


def parse(arguments):
  settings = {}
  for item in arguments:
    key, _, value = item.partition('=')
    try:
      settings[key] = json.loads(value)
    except json.JSONDecodeError:
      settings[key] = value
  return settings


def detector(noise, config=REFERENCE, **overrides):
  """The NOISELESS twin of ``config`` with ``overrides`` applied. The integration guard is tied to the
  noise by the repo-wide rule (10% of the read-out noise)."""
  arguments = dict(measurement_noise=0.0, integration_tolerance=0.1 * noise)
  arguments.update(overrides)
  return build_detector(load_config(config), **arguments)


def draw_prior(det, n, generator):
  """``n`` draws of the enzyme prior, the half-conversion time and a compound, as plain arrays."""
  parameters = {name: generator.uniform(low, high, size=n).astype(np.float32) for name, (low, high) in det._parameter_ranges}
  half_time = np.exp(generator.uniform(math.log(det.half_time_bounds[0]), math.log(det.half_time_bounds[1]),
                                       size=n)).astype(np.float32)
  strong = generator.uniform(det.log10_k_bounds[0], det.log10_k_bounds[1], size=n).astype(np.float32)
  preference = generator.uniform(det.preference_bounds[0], det.preference_bounds[1], size=n).astype(np.float32)
  return parameters, half_time, strong, preference


def curves(det):
  """Noise-free read-outs for EXPLICIT kinetics: the detector's own calibration and integration, with
  the enzyme, the turnover and both inhibitor branches supplied rather than drawn."""

  def one(parameters, half_time, fraction, substrate, inhibitor, temperature, k1, k2):
    log_k0_cat = det._calibrate(parameters, half_time)
    measurements, _ = det._run_batch(
      fraction, substrate, inhibitor, temperature, parameters, log_k0_cat, k1, k2, jax.random.PRNGKey(0)
    )
    return measurements

  return jax.jit(jax.vmap(one))


def section_physics(settings):
  noise = settings.get('noise', 0.05)
  n = settings.get('n_draws', 512)
  det = detector(noise, n_experiments=1, n_measurements=settings.get('n_measurements', 32))
  simulate = curves(det)
  generator = np.random.default_rng(settings.get('seed', 0))
  parameters, half_time, _, _ = draw_prior(det, n, generator)
  fraction = np.full((n, 1), 0.5, np.float32)
  temperature = np.full((n, 1), 25.0, np.float32)
  zero = np.zeros((n, 1), np.float32)
  zero_k = np.zeros(n, np.float32)

  print(
    'K_M(ATP) at 0 C over the prior draw, 10/50/90%:', np.round(np.percentile(np.exp(parameters['log_K0_B']), [10, 50, 90]), 4)
  )
  print()
  print('   B0      I     I*k | competitive branch alone   | uncompetitive branch alone')
  print('                     |  median      90%           |  median      90%      (max |dA| / sigma)')
  for substrate_B in settings.get('substrate', [0.1, 0.3, 1.0, 3.0, 10.0]):
    for inhibitor in settings.get('inhibitor', [0.01, 0.1, 1.0]):
      potency = settings.get('potency', 10.0)
      substrate = np.full((n, 1), substrate_B, np.float32)
      dose = np.full((n, 1), inhibitor, np.float32)
      strength = np.full(n, potency, np.float32)
      base = simulate(parameters, half_time, fraction, substrate, zero, temperature, zero_k, zero_k)
      competitive = simulate(parameters, half_time, fraction, substrate, dose, temperature, strength, zero_k)
      uncompetitive = simulate(parameters, half_time, fraction, substrate, dose, temperature, zero_k, strength)
      first = np.abs(np.asarray(competitive - base)).max(axis=(1, 2)) / noise
      second = np.abs(np.asarray(uncompetitive - base)).max(axis=(1, 2)) / noise
      print(
        f'{substrate_B:6.2f} {inhibitor:6.3f} {inhibitor * potency:7.2f} | '
        f'{np.median(first):8.2f} {np.percentile(first, 90):8.2f}   | '
        f'{np.median(second):8.2f} {np.percentile(second, 90):8.2f}'
      )


def mirror_ratio(inhibitor, k1, k2):
  """``r = (1 + I k2) / (1 + I k1)``: the factor the class swap puts on ``K_B``, ``Ki_D`` and the
  turnover at inhibitor concentration ``I``."""
  return (1.0 + inhibitor * k2) / (1.0 + inhibitor * k1)


def section_mirror(settings):
  """The class swap is EXACTLY ``K_B -> K_B r``, ``Ki_D -> Ki_D r``, ``k_cat -> k_cat r``, so at a
  single inhibitor level the two classes are one reparameterisation of the enzyme apart."""
  noise = settings.get('noise', 0.05)
  n = settings.get('n_draws', 256)
  det = detector(noise, n_experiments=1, n_measurements=settings.get('n_measurements', 32))
  simulate = curves(det)
  calibrate = jax.jit(jax.vmap(det._calibrate))
  generator = np.random.default_rng(settings.get('seed', 0))
  parameters, half_time, strong, preference = draw_prior(det, n, generator)
  k1 = np.power(10.0, strong).astype(np.float32)
  k2 = np.power(10.0, strong - preference).astype(np.float32)
  fraction = np.full((n, 1), settings.get('enzyme_fraction', 0.5), np.float32)
  substrate = np.full((n, 1), settings.get('substrate_B', 1.0), np.float32)
  temperature = np.full((n, 1), settings.get('temperature', 25.0), np.float32)

  worst = 0.0
  for inhibitor in settings.get('inhibitor', [0.01, 0.1, 1.0]):
    dose = np.full((n, 1), inhibitor, np.float32)
    ratio = mirror_ratio(inhibitor, k1, k2)
    drawn = simulate(parameters, half_time, fraction, substrate, dose, temperature, k1, k2)
    mirrored = simulate(parameters, half_time, fraction, substrate, dose, temperature, k2, k1)

    shifted = dict(parameters)
    shifted['log_K0_B'] = (parameters['log_K0_B'] + np.log(ratio)).astype(np.float32)
    shifted['log_K0i_D'] = (parameters['log_K0i_D'] + np.log(ratio)).astype(np.float32)
    correction = np.exp(np.asarray(calibrate(shifted, half_time)) - np.asarray(calibrate(parameters, half_time)))
    equivalent = simulate(
      shifted, (half_time * correction / ratio).astype(np.float32), fraction, substrate, dose, temperature, k1, k2
    )
    gap = float(np.max(np.abs(np.asarray(mirrored - equivalent))))
    contrast = float(np.median(np.max(np.abs(np.asarray(mirrored - drawn)), axis=(1, 2))))
    worst = max(worst, gap)
    print(
      f'I={inhibitor:6.3f}  max |mirrored - reparameterised| = {gap:.3e} mM  |  '
      f'median class contrast {contrast:.3e} mM  |  r median {float(np.median(ratio)):.3e}'
    )
  print(
    f'\nworst disagreement {worst:.3e} mM; the read-out noise is {noise:.3e} mM and the integration '
    f'guard is {det.integration_tolerance:.3e} mM'
  )
  support = []
  for inhibitor in settings.get('inhibitor', [0.01, 0.1, 1.0]):
    ratio = mirror_ratio(inhibitor, k1, k2)
    inside = ((parameters['log_K0_B'] + np.log(ratio) >= dict(det._parameter_ranges)['log_K0_B'][0]) &
              (parameters['log_K0_B'] + np.log(ratio) <= dict(det._parameter_ranges)['log_K0_B'][1]) &
              (parameters['log_K0i_D'] + np.log(ratio) >= dict(det._parameter_ranges)['log_K0i_D'][0]) &
              (parameters['log_K0i_D'] + np.log(ratio) <= dict(det._parameter_ranges)['log_K0i_D'][1]))
    support.append((inhibitor, float(np.mean(inside))))
  print(
    '\nfraction of draws whose mirror image stays INSIDE the enzyme prior (the classes are then '
    'indistinguishable from one experiment):'
  )
  for inhibitor, fraction_inside in support:
    print(f'  I={inhibitor:6.3f}: {fraction_inside:5.3f}')


def section_regime(settings):
  """Fraction of prior draws whose CLASS-MIRRORED read-out differs by at least one sigma, at the
  worst corner of the design box."""
  noise = settings.get('noise', 0.05)
  n = settings.get('n_draws', 512)
  threshold = settings.get('threshold', 1.0)
  det = detector(noise, n_experiments=1, n_measurements=settings.get('n_measurements', 32), **box_overrides(settings))
  simulate = curves(det)
  generator = np.random.default_rng(settings.get('seed', 0))
  parameters, half_time, strong, preference = draw_prior(det, n, generator)
  k1 = np.power(10.0, strong).astype(np.float32)
  k2 = np.power(10.0, strong - preference).astype(np.float32)

  print(
    f'box: substrate_B {det.substrate_B_bounds}, inhibitor {det.inhibitor_bounds}, '
    f'enzyme_fraction {det.enzyme_fraction_bounds}, temperature {det.temperature_bounds}'
  )
  print(f'threshold: max |drawn - class-mirrored| >= {threshold} sigma, sigma = {noise}')
  print()
  fractions = []
  for enzyme_fraction in det.enzyme_fraction_bounds:
    for substrate_B in det.substrate_B_bounds:
      for inhibitor in det.inhibitor_bounds:
        for temperature in det.temperature_bounds:
          shape = (n, 1)
          drawn = simulate(
            parameters, half_time, np.full(shape, enzyme_fraction, np.float32), np.full(shape, substrate_B, np.float32),
            np.full(shape, inhibitor, np.float32), np.full(shape, temperature, np.float32), k1, k2
          )
          mirrored = simulate(
            parameters, half_time, np.full(shape, enzyme_fraction, np.float32), np.full(shape, substrate_B, np.float32),
            np.full(shape, inhibitor, np.float32), np.full(shape, temperature, np.float32), k2, k1
          )
          contrast = np.abs(np.asarray(drawn - mirrored)).max(axis=(1, 2)) / noise
          fraction = float(np.mean(contrast >= threshold))
          fractions.append((fraction, enzyme_fraction, substrate_B, inhibitor, temperature))
          print(
            f'f={enzyme_fraction:4.2f} B0={substrate_B:6.3f} I={inhibitor:7.4f} T={temperature:5.1f} '
            f'-> fraction {fraction:5.3f}, median contrast {np.median(contrast):6.2f} sigma'
          )
  worst = min(fractions)
  print(
    f'\nWORST corner: fraction {worst[0]:5.3f} at enzyme_fraction={worst[1]}, substrate_B={worst[2]}, '
    f'inhibitor={worst[3]}, temperature={worst[4]}'
  )


def section_guard(settings):
  noise = settings.get('noise', 0.05)
  n_measurements = settings.get('n_measurements', 32)
  n_events = settings.get('n_events', 4096)
  m = settings.get('n_experiments', 4)
  print(f'noise {noise}, tolerance {0.1 * noise:.3e} mM, {n_events} events, {n_measurements} read-outs, m={m}')
  for total in settings.get('total_steps', [2560, 1280, 640, 320, 160]):
    if total % n_measurements != 0:
      continue
    det = detector(
      noise, n_experiments=m, n_measurements=n_measurements, n_steps_per_measurement=total // n_measurements,
      integration_tolerance=1.0e9, **box_overrides(settings)
    )
    worst = 0.0
    for corner in range(2**4):
      scaled = np.zeros(4 * m)
      for axis in range(4):
        scaled[axis * m:(axis + 1) * m] = float((corner >> axis) & 1)
      nominal = np.asarray(det._to_nominal_flat(scaled.astype(np.float32)))
      fraction, substrate, inhibitor, temperature = det._resolve_design(nominal, n_events)
      *_, error = det._generate(fraction, substrate, inhibitor, temperature, np.arange(n_events, dtype=np.int32))
      worst = max(worst, float(np.max(np.asarray(error))))
    verdict = 'FIRES' if worst > 0.1 * noise else 'passes'
    print(
      f'total steps {total:5d}: worst dt-vs-dt/2 {worst:.3e} mM   {verdict} '
      f'(margin {0.1 * noise / max(worst, 1e-12):8.2f}x)', flush=True
    )


def instrument(settings, noise, **overrides):
  det = detector(noise, **overrides)
  return det, MechanismInstrument(
    det, noise=noise, n_events=settings.get('n_events', 512), n_library=settings.get('n_library', 4096),
    n_noise=settings.get('n_noise', 2), hedge=settings.get('hedge', 1.0e-3), seed=settings.get('seed', 1)
  )


def box_overrides(settings):
  keys = ('substrate_B_bounds', 'inhibitor_bounds', 'enzyme_fraction_bounds', 'temperature_bounds')
  overrides = {key: tuple(settings[key]) for key in keys if key in settings}
  if 'config' in settings:
    overrides['config'] = settings['config']
  return overrides


def section_profile(settings):
  noise = settings.get('noise', 0.05)
  m = settings.get('n_experiments', 4)
  n_measurements = settings.get('n_measurements', 32)
  total = settings.get('total_steps', 1280)
  base = np.asarray(settings.get('base', [0.5] * (4 * m)), np.float64)
  det, inst = instrument(
    settings, noise, n_experiments=m, n_measurements=n_measurements, n_steps_per_measurement=total // n_measurements,
    **box_overrides(settings)
  )
  print(f'm={m}, {n_measurements} read-outs, sigma={noise}, base design (scaled) {base.tolist()}')
  print(f'box: {det.design_bounds()}')
  for axis, name in enumerate(COORDINATES):
    print(f'\n-- {name} (all {m} experiments moved together) --')
    for value in settings.get('grid', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
      design = base.copy()
      design[axis * m:(axis + 1) * m] = value
      nominal = np.asarray(det._to_nominal_flat(design.astype(np.float32)))[axis * m]
      result = inst.evaluate(design)
      print(
        f'  scaled {value:4.2f} (nominal {nominal:10.4g}): loss {result.loss:6.4f} +- {result.standard_error:5.4f}'
        f'  accuracy {result.accuracy:5.3f}  ess {result.effective_sample_size:8.1f}'
        f'  hedged {result.hedge_fraction:5.3f}'
      )


def section_resolution(settings):
  noise = settings.get('noise', 0.05)
  m = settings.get('n_experiments', 4)
  total = settings.get('total_steps', 1280)
  print(f'm={m}, sigma={noise}, total integration steps held at {total}')
  for n_measurements in settings.get('cadence', [4, 8, 16, 32, 64]):
    if total % n_measurements != 0:
      continue
    det, inst = instrument(
      settings, noise, n_experiments=m, n_measurements=n_measurements, n_steps_per_measurement=total // n_measurements,
      **box_overrides(settings)
    )
    rows = []
    for name, design in named_designs(m, settings):
      result = inst.evaluate(design)
      rows.append(f'{name} {result.loss:6.4f}+-{result.standard_error:5.4f} (ess {result.effective_sample_size:7.1f})')
    print(f'  {n_measurements:3d} read-outs: ' + '  |  '.join(rows), flush=True)


def named_designs(m, settings):
  """A spread design (the inhibitor dose split across the batch) and a degenerate one (every
  experiment identical), both in the unit cube."""
  spread = np.full(4 * m, 0.5)
  degenerate = np.full(4 * m, 0.5)
  if m >= 2:
    spread[m:2 * m] = np.resize([0.25, 0.85], m)
    spread[2 * m:3 * m] = np.repeat(np.resize([0.2, 1.0], 2), m // 2)[:m]
  else:
    spread[2 * m:3 * m] = 0.9
  designs = [('spread', spread), ('degenerate', degenerate)]
  for index, point in enumerate(sobol_designs(4 * m, settings.get('n_sobol', 2), seed=3)):
    designs.append((f'sobol{index}', point))
  return designs


def section_noise(settings):
  m = settings.get('n_experiments', 4)
  n_measurements = settings.get('n_measurements', 32)
  total = settings.get('total_steps', 1280)
  n_designs = settings.get('n_designs', 64)
  points = sobol_designs(4 * m, n_designs, seed=settings.get('design_seed', 11))
  records = {}
  for noise in settings.get('noises', [0.05, 0.1, 0.2]):
    det, inst = instrument(
      settings, noise, n_experiments=m, n_measurements=n_measurements, n_steps_per_measurement=total // n_measurements,
      **box_overrides(settings)
    )
    losses, sizes = [], []
    for point in points:
      result = inst.evaluate(point)
      losses.append(result.loss)
      sizes.append(result.effective_sample_size)
    losses = np.asarray(losses)
    records[noise] = losses.tolist()
    quantiles = np.percentile(losses, [0, 10, 50, 90, 100])
    print(
      f'sigma={noise:6.4f}: over {n_designs} Sobol designs  min {quantiles[0]:6.4f}  10% {quantiles[1]:6.4f}  '
      f'median {quantiles[2]:6.4f}  90% {quantiles[3]:6.4f}  max {quantiles[4]:6.4f}  '
      f'span {quantiles[4] - quantiles[0]:6.4f}  sd {losses.std():6.4f}  median ess {np.median(sizes):8.1f}', flush=True
    )
  output = settings.get('output')
  if output is not None:
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as handle:
      json.dump({'n_experiments': m, 'n_measurements': n_measurements, 'designs': points.tolist(), 'losses': records}, handle)
    print(f'wrote {output}')


def section_convergence(settings):
  noise = settings.get('noise', 0.05)
  m = settings.get('n_experiments', 4)
  n_measurements = settings.get('n_measurements', 32)
  total = settings.get('total_steps', 1280)
  print(f'm={m}, {n_measurements} read-outs, sigma={noise}')
  for n_library in settings.get('libraries', [512, 1024, 2048, 4096, 8192, 16384]):
    rows = []
    for name, design in named_designs(m, settings):
      det, inst = instrument(
        dict(settings, n_library=n_library), noise, n_experiments=m, n_measurements=n_measurements,
        n_steps_per_measurement=total // n_measurements, **box_overrides(settings)
      )
      result = inst.evaluate(design)
      rows.append(
        f'{name} {result.loss:6.4f}+-{result.standard_error:5.4f} '
        f'(ess {result.effective_sample_size:8.1f}, hedged {result.hedge_fraction:5.3f})'
      )
    print(f'  library {n_library:6d}: ' + '  |  '.join(rows), flush=True)


SECTIONS = {
  'physics': section_physics,
  'mirror': section_mirror,
  'regime': section_regime,
  'guard': section_guard,
  'profile': section_profile,
  'resolution': section_resolution,
  'noise': section_noise,
  'convergence': section_convergence
}


def main():
  if len(sys.argv) < 2 or sys.argv[1] not in SECTIONS:
    raise SystemExit(f'usage: {sys.argv[0]} <{"|".join(SECTIONS)}> [key=value ...]')
  SECTIONS[sys.argv[1]](parse(sys.argv[2:]))


if __name__ == '__main__':
  main()
