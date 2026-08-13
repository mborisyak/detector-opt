#!/usr/bin/env python3
"""Pre-screen validation of the INHIBITOR-MECHANISM candidate task (CPU only).

Step 1-3 of the screening protocol (`output/agent-docs/benchmark-acceptance.md` section 4), plus the
checks the plan (`output/agent-docs/task-inhibitor-mechanism.md` section 9) asks for BEFORE any
landscape or campaign:

  1  builds / simulates / targets inside their priors / read-outs finite / the label is a property of
     the EVENT and not of the design;
  1b thermal unfolding: MEASURED justification for not simulating it inside this temperature box;
  2  the integrator: the dt vs dt/2 disagreement at the read-out times against the CONFIGURABLE
     `integration_tolerance`, that the guard actually raises, and the stability margin;
  3  turnover realism: the calibrated k_cat against the literature hexokinase band;
  4  the CEILING, measured by scoring a deliberately uninformative design;
  5  KNOWN-GOOD vs KNOWN-BAD: designs the physics says are good and bad, scored. If they are not
     clearly different, the target is not in the data and no campaign can fix it;
  6  class contrast: the noiseless read-out separation between the three mechanisms, in units of the
     read-out noise -- the SNR of the target itself, with no estimator in it;
  7  identifiability: the marginalised Fisher information for (log10 k1, log10 k2), nuisances = the
     kinetic parameters + the turnover, and the plan's converse test (a design the physics says is
     bad must measure WORSE).

Run it through SLURM, e.g.

    sbatch --cpus-per-task=4 -J validate-inhibitor -o output/screen/validate-inhibitor.log \
        --wrap="python -u scripts/validate_inhibitor.py"
"""
import os
import pathlib
import sys

# BLAS/OpenMP default to every core on the machine, which is wrong under a scheduler: SLURM says
# which cores this job may use, not how many threads it should start. Set before numpy is imported.
_allocated = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, _allocated)

# Use the tree this script lives in (`detopt` is not installed anywhere).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
import time

import numpy as np

import jax
import jax.numpy as jnp

import detopt.detector
import detopt.utils.config
from detopt.bo.gbdt import score_design
from detopt.detector.enzyme import gibbs_fraction
from detopt.detector.enzyme_inhibitor import CLASS_NAMES, N_CLASSES, PARAMETER_NAMES

CONFIG = "config/detector/enzyme_inhib.yaml"
# The measured float32 RKC2 stability boundary at 5 stages (scripts/check_stability.py, quoted by the
# baseline enzyme config): z = dt * |df/dx| must stay inside it.
RKC2_BOUNDARY_5_STAGES = 16.60


def build(n_experiments=None):
  config = detopt.utils.config.load_config(str(pathlib.Path(__file__).resolve().parent.parent / CONFIG))
  (name,) = [k for k in config if isinstance(config[k], dict)]
  if n_experiments is not None:
    config[name] = dict(config[name], n_experiments=int(n_experiments))
  return detopt.detector.from_config(config)


def draw_event(detector, index):
  """The enzyme and compound at ``index`` -- the same draw ``_event`` makes, exposed so the checks
  below can hold parts of it fixed. Returns ``(parameters, half_time, log_k0_cat, mechanism,
  log10_k1, log10_k2)``."""
  key_parameters, key_half_time, key_compound, _ = jax.random.split(jax.random.PRNGKey(index), 4)
  parameters = detector._draw_parameters(key_parameters)
  low, high = detector.half_time_bounds
  half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=math.log(low), maxval=math.log(high)))
  mechanism, log10_k1, log10_k2 = detector._draw_compound(key_compound)
  return parameters, half_time, detector._calibrate(parameters, half_time), mechanism, log10_k1, log10_k2


def noiseless_readout(detector, design, parameters, log_k0_cat, k1, k2):
  """[A] at every read-out time of every experiment, WITHOUT the measurement noise -- the physical
  signal a design produces for one (enzyme, compound) pair. ``design`` is a flat NOMINAL vector."""
  n = detector.n_experiments
  design = jnp.asarray(design, jnp.float32)
  fraction, substrate, inhibitor, temperature = design[:n], design[n:2 * n], design[2 * n:3 * n], design[3 * n:]

  def run(enzyme_fraction, substrate_B, experiment_inhibitor, experiment_temperature):
    A0, B0, E0 = detector._initial_state(enzyme_fraction, substrate_B)
    extent, _ = detector._integrate(
      A0, B0, E0, experiment_inhibitor, experiment_temperature, parameters, log_k0_cat, k1, k2,
      n_steps=detector.n_steps_per_measurement, n_intervals=detector.n_measurements
    )
    return A0 - extent

  return jax.vmap(run)(fraction, substrate, inhibitor, temperature)


def named_designs(detector):
  """The calibration designs: what the physics says is good, and what it says is bad.

  Stated from the PRIOR alone, never from a landscape: the low ATP arm is the floor of the ATP box
  (0.1 mM, at the bottom of the K_M(ATP) prior), the saturating arm its ceiling (10 mM), the weak
  inhibitor dose the floor of the inhibitor box (1e-3 mM) and the strong dose the geometric centre of
  the Ki prior (10^-1.5 = 0.0316 mM), where the population's median compound is half-inhibited.
  """
  n = detector.n_experiments
  low_B, high_B = detector.substrate_B_bounds
  weak_I = detector.inhibitor_bounds[0]
  strong_I = float(np.sqrt(detector.inhibitor_potency_bounds[0] * detector.inhibitor_potency_bounds[1]))
  middle_B = float(np.sqrt(low_B * high_B))
  fraction, warm = 0.5, 0.5 * sum(detector.temperature_bounds)

  def design(cells):
    cells = [cells[i % len(cells)] for i in range(n)]
    return np.asarray(
      [fraction] * n + [c[0] for c in cells] + [c[1] for c in cells] + [warm] * n, np.float32
    )

  return {
    # The 2x2 the physics implies: (low / saturating ATP) x (weak / potent inhibitor dose).
    "KNOWN-GOOD 2x2, two inhibitor doses":
      design([(low_B, weak_I), (low_B, strong_I), (high_B, weak_I), (high_B, strong_I)]),
    # The same crossing with a STRONGER dose -- the top of the inhibitor box, which brackets even the
    # weakest compounds in the library.
    "KNOWN-GOOD 2x2, strong dose":
      design([(low_B, weak_I), (low_B, detector.inhibitor_bounds[1]), (high_B, weak_I),
              (high_B, detector.inhibitor_bounds[1])]),
    # BAD: the inhibitor contrast without the substrate contrast. k1 and k2 enter every experiment in
    # the same combination, so the mechanism cannot be separated from the potency.
    "KNOWN-BAD no substrate contrast":
      design([(middle_B, weak_I), (middle_B, strong_I), (middle_B, weak_I), (middle_B, strong_I)]),
    # BAD: the substrate contrast without the inhibitor contrast -- every experiment at the weakest
    # dose, where the compound barely acts.
    "KNOWN-BAD no inhibitor contrast":
      design([(low_B, weak_I), (high_B, weak_I), (low_B, weak_I), (high_B, weak_I)]),
    # The deliberately UNINFORMATIVE design: weakest inhibitor, no substrate contrast, smallest
    # enzyme dose, coldest temperature. This is the ceiling probe.
    "uninformative (ceiling probe)": np.asarray(
      [detector.enzyme_fraction_bounds[0]] * n + [low_B] * n + [weak_I] * n
      + [detector.temperature_bounds[0]] * n, np.float32
    ),
  }


def section_1(detector, n_events=4096):
  print("1. BUILDS / SIMULATES / TARGETS IN PRIOR", flush=True)
  designs = named_designs(detector)
  design = designs["KNOWN-GOOD 2x2, two inhibitor doses"]
  index = np.arange(n_events, dtype=np.int64)
  ground_truth, event, mask, target = detector(design, index)

  measurements = np.asarray(event.measurements)
  print(f"   read-outs finite: {bool(np.isfinite(measurements).all())}   "
        f"[A] in [{measurements.min():.4f}, {measurements.max():.4f}] mM (A0 = {detector.concentration_A})")

  one_hot = np.asarray(target.mechanism)
  counts = one_hot.sum(axis=0)
  print("   class counts over %d: %s   (uniform expectation %d)" % (
    n_events, ", ".join(f"{name}={int(c)}" for name, c in zip(CLASS_NAMES, counts)), n_events // N_CLASSES))
  print(f"   one-hot rows: {bool(np.all(one_hot.sum(axis=1) == 1))}")

  log10_k1, log10_k2 = np.asarray(ground_truth.log10_k1)[:, 0], np.asarray(ground_truth.log10_k2)[:, 0]
  strong, preference = np.maximum(log10_k1, log10_k2), np.abs(log10_k1 - log10_k2)
  low, high = detector.log10_k_bounds
  print(f"   log10 k_strong in [{strong.min():.3f}, {strong.max():.3f}]  prior [{low:.3f}, {high:.3f}]  "
        f"inside: {bool(strong.min() >= low - 1e-4 and strong.max() <= high + 1e-4)}")
  klass = np.argmax(one_hot, axis=1)
  for i, name in enumerate(CLASS_NAMES):
    band = detector.preference_bounds_noncompetitive if name == "mostly_noncompetitive" else detector.preference_bounds
    own = preference[klass == i]
    print(f"   |d| | {name:22s} in [{own.min():.3f}, {own.max():.3f}]  band {band}  "
          f"inside: {bool(own.min() >= band[0] - 1e-4 and own.max() <= band[1] + 1e-4)}")
  gap = detector.preference_bounds_noncompetitive[1], detector.preference_bounds[0]
  print(f"   compounds in the UNSAMPLED gap {gap}: {int(((preference > gap[0]) & (preference < gap[1])).sum())} "
        f"(must be 0)")

  half_time = np.asarray(ground_truth.half_time)[:, 0]
  print(f"   half_time in [{half_time.min():.3f}, {half_time.max():.3f}] h  prior {detector.half_time_bounds}")
  parameters = np.asarray(ground_truth.parameters)
  inside = all(
    parameters[:, i].min() >= r[0] - 1e-4 and parameters[:, i].max() <= r[1] + 1e-4
    for i, (_, r) in enumerate(detector._parameter_ranges)
  )
  print(f"   all {len(PARAMETER_NAMES)} kinetic parameters inside their prior ranges: {inside}")
  print(f"   mask all-valid: {bool(np.all(np.asarray(mask) == 1))}")

  # THE LABEL IS A PROPERTY OF THE EVENT. Same indices under a different design must give the same
  # compound, the same enzyme and the same label -- only the read-out may move.
  other = designs["KNOWN-BAD no substrate contrast"]
  ground_truth_2, event_2, _, target_2 = detector(other, index)
  same_label = bool(np.array_equal(one_hot, np.asarray(target_2.mechanism)))
  same_truth = bool(np.array_equal(parameters, np.asarray(ground_truth_2.parameters))
                    and np.array_equal(log10_k1, np.asarray(ground_truth_2.log10_k1)[:, 0]))
  moved = float(np.abs(measurements - np.asarray(event_2.measurements)).max())
  print(f"   target/ground truth identical under a DIFFERENT design: label {same_label}, truth {same_truth}; "
        f"read-out moved by up to {moved:.4f} mM")
  return designs


def section_1b(detector, n_points=51):
  """Thermal unfolding is not simulated. Both halves of that decision, MEASURED."""
  print("\n1b. THERMAL UNFOLDING -- why it is NOT simulated (plan section 9.1 asks gibbs_fraction > 0.98)", flush=True)
  temperature = jnp.linspace(*detector.temperature_bounds, n_points)
  # The BASELINE enzyme.yaml prior, over the CORNERS of (delta_H, delta_C) x the ends of T_melting --
  # a prior is a set, so the check has to be the worst case over it, not the median enzyme.
  print("   baseline enzyme.yaml prior (delta_H 4.5e4-1.0e5, delta_C 2.5e3-4.5e3, T_m 45-58 C):")
  worst = 1.0
  for enthalpy in (4.5e4, 1.0e5):
    for capacity in (2.5e3, 4.5e3):
      for melting in (45.0, 58.0):
        folded = np.asarray(gibbs_fraction(temperature, enthalpy, capacity, melting))
        worst = min(worst, float(folded.min()))
        print(f"      delta_H {enthalpy:.1e}  delta_C {capacity:.1e}  T_m {melting:.0f} C   folded over the box "
              f"[{folded.min():.3f}, {folded.max():.3f}]  {'passes' if folded.min() > 0.98 else 'FAILS'} > 0.98")
  print(f"   -> worst corner of the baseline prior: {worst:.3f} folded inside the box, so inheriting those")
  print("      three parameters would re-import a threshold (COLD denaturation) at the cold end.")
  # A REAL hexokinase: yeast HK melts at ~41.9 C; with the heat-capacity term dropped the sigmoid is
  # the plain van't Hoff one.
  for enthalpy in (4.5e4, 1.0e5):
    folded = np.asarray(gibbs_fraction(temperature, enthalpy, 0.0, 41.9))
    print(f"   delta_C = 0, T_m = 41.9 C (yeast HK), delta_H {enthalpy:.1e}   folded over the box "
          f"[{folded.min():.3f}, {folded.max():.3f}]  {'passes' if folded.min() > 0.98 else 'FAILS'} > 0.98"
          f"   (moves {100 * (folded.max() - folded.min()):.1f}% across the box)")
  print("   -> for a REAL enzyme in this box the folded fraction is a near-constant, monotone in T,")
  print("      which the turnover calibration absorbs; simulating it would add three ESTIMATE")
  print("      thermodynamic parameters that change nothing measurable.")


def section_2(detector, designs, n_events=512):
  print("\n2. INTEGRATOR (dt vs dt/2 at the read-out times) + STABILITY", flush=True)
  worst, worst_name = 0.0, None
  rng = np.random.default_rng(0)
  probes = dict(designs)
  for i in range(8):  # random corners of the box, where the stiffest combinations live
    corner = np.asarray(detector.flatten_design(detector.to_nominal(
      rng.integers(0, 2, detector.design_dim()).astype(np.float32))), np.float32)
    probes[f"box corner {i}"] = corner
  for name, design in probes.items():
    _, event, _, _ = detector(design, np.arange(n_events, dtype=np.int64))
    # __call__ raises above the tolerance, so reaching here means it passed; re-measure the error by
    # running the two chains again through the detector's own estimate.
    error = float(jnp.max(_integration_error(detector, design, n_events)))
    if error > worst:
      worst, worst_name = error, name
  print(f"   observed max|fine - coarse| over {len(probes)} designs x {n_events} events: {worst:.3e} mM "
        f"(worst at '{worst_name}')")
  print(f"   integration_tolerance: {detector.integration_tolerance:.3e} mM  -> margin "
        f"{detector.integration_tolerance / worst:.0f}x")
  print(f"   read-out noise: {detector.measurement_noise} mM  -> the integration error is "
        f"{detector.measurement_noise / worst:.0f}x below what the assay resolves")

  # THE GUARD IS LIVE: tighten the tolerance below the observed error and confirm __call__ raises.
  tolerance = detector.integration_tolerance
  detector.integration_tolerance = 0.5 * worst
  try:
    detector(probes[worst_name], np.arange(n_events, dtype=np.int64))
    print("   !! assert did NOT fire at half the observed error -- the guard is not live")
  except RuntimeError as e:
    print(f"   assert is live: raises at tolerance {0.5 * worst:.3e} -- \"{str(e)[:96]}...\"")
  finally:
    detector.integration_tolerance = tolerance

  # STIFFNESS: the largest |df/dx| the prior x box can produce, against the measured RKC2 boundary.
  slope = _worst_slope(detector, n_draws=512)
  dt = detector.duration / (detector.n_measurements * detector.n_steps_per_measurement)
  print(f"   worst |df/dx| over 512 (prior x box-corner) draws: {slope:.1f} /h   dt = {dt:.6f} h")
  print(f"   z = dt |df/dx| = {dt * slope:.3f}  against the measured RKC2 boundary "
        f"{RKC2_BOUNDARY_5_STAGES} at {detector.n_stages} stages -> margin {RKC2_BOUNDARY_5_STAGES / (dt * slope):.1f}x")


def _integration_error(detector, design, n_events):
  """The detector's own dt-vs-dt/2 estimate for every event of ``design`` (the quantity ``__call__``
  asserts on)."""
  n = detector.n_experiments
  design = jnp.asarray(design, jnp.float32)
  fraction, substrate = design[:n], design[n:2 * n]
  inhibitor, temperature = design[2 * n:3 * n], design[3 * n:]

  def one(index):
    parameters, half_time, log_k0_cat, _, log10_k1, log10_k2 = draw_event(detector, index)

    def run(enzyme_fraction, substrate_B, experiment_inhibitor, experiment_temperature):
      A0, B0, E0 = detector._initial_state(enzyme_fraction, substrate_B)
      _, error = detector._integrate(
        A0, B0, E0, experiment_inhibitor, experiment_temperature, parameters, log_k0_cat,
        jnp.power(10.0, log10_k1), jnp.power(10.0, log10_k2),
        n_steps=detector.n_steps_per_measurement, n_intervals=detector.n_measurements
      )
      return error

    return jnp.max(jax.vmap(run)(fraction, substrate, inhibitor, temperature))

  return jax.jit(jax.vmap(one))(jnp.arange(n_events, dtype=jnp.int32))


def _worst_slope(detector, n_draws):
  """The largest local decay rate ``|df/dx|`` the prior crossed with the box corners produces, over a
  grid of extents. This is what the RKC2 step must stay stable against."""
  corners = np.array([[a, b, c, d] for a in (0.0, 1.0) for b in (0.0, 1.0) for c in (0.0, 1.0) for d in (0.0, 1.0)])

  def slope(index, corner):
    parameters, half_time, log_k0_cat, _, log10_k1, log10_k2 = draw_event(detector, index)
    fraction, substrate_B = (corner[0] * (detector.enzyme_fraction_bounds[1] - detector.enzyme_fraction_bounds[0])
                             + detector.enzyme_fraction_bounds[0],
                             jnp.power(10.0, corner[1] * (math.log10(detector.substrate_B_bounds[1])
                                                          - math.log10(detector.substrate_B_bounds[0]))
                                       + math.log10(detector.substrate_B_bounds[0])))
    inhibitor = jnp.power(10.0, corner[2] * (math.log10(detector.inhibitor_bounds[1])
                                             - math.log10(detector.inhibitor_bounds[0]))
                          + math.log10(detector.inhibitor_bounds[0]))
    temperature = corner[3] * (detector.temperature_bounds[1] - detector.temperature_bounds[0]) \
        + detector.temperature_bounds[0]
    A0, B0, E0 = detector._initial_state(fraction, substrate_B)
    limit = jnp.minimum(A0, B0)
    extents = jnp.linspace(0.0, 0.999, 64) * limit
    rate = lambda x: detector._rate(x, A0, B0, E0, inhibitor, temperature, parameters, log_k0_cat,
                                    jnp.power(10.0, log10_k1), jnp.power(10.0, log10_k2))
    return jnp.max(jnp.abs(jax.vmap(jax.grad(rate))(extents)))

  index = jnp.arange(n_draws, dtype=jnp.int32)
  worst = jax.jit(jax.vmap(lambda c: jax.vmap(lambda i: slope(i, c))(index)))(jnp.asarray(corners, jnp.float32))
  return float(jnp.max(worst))


def section_3(detector, n_events=4096):
  """Turnover realism: the CALIBRATED k_cat against the literature hexokinase band (30-300 /s)."""
  print("\n3. TURNOVER REALISM (a consequence of the stock + the half-time prior, not a target)", flush=True)

  def k_cat(index):
    parameters, half_time, log_k0_cat, _, _, _ = draw_event(detector, index)
    from detopt.detector.enzyme import vant_hoff
    return vant_hoff(detector.temperature_bounds[1], log_k0_cat, parameters['Q10_cat'])

  values = np.asarray(jax.jit(jax.vmap(k_cat))(jnp.arange(n_events, dtype=jnp.int32))) / 3600.0
  print(f"   calibrated k_cat at {detector.temperature_bounds[1]:.0f} C: median {np.median(values):.1f} /s, "
        f"10-90% [{np.quantile(values, 0.1):.1f}, {np.quantile(values, 0.9):.1f}] /s "
        f"(literature hexokinase 30-300 /s)")


def section_4_5(detector, designs, n_events=16384):
  """The CEILING (an uninformative design) and KNOWN-GOOD vs KNOWN-BAD, on the proxy."""
  print(f"\n4./5. CEILING, KNOWN-GOOD vs KNOWN-BAD  ({n_events} events each, the screen's proxy)", flush=True)
  scores = {}
  for name, design in designs.items():
    started = time.time()
    score = score_design(detector, design, n_events=n_events, seed=0)
    scores[name] = score
    print(f"   {name:38s} loss {score.loss:.4f} +- {score.sem:.4f}   "
          f"(train {score.train:.4f} / val {score.val:.4f}, {score.n_learners} trees, "
          f"{time.time() - started:.2f} s)")
  print(f"   analytic no-information level (cross-entropy / ln {N_CLASSES}): 1.0")
  best = min(scores.items(), key=lambda kv: kv[1].loss)
  worst_named = scores["uninformative (ceiling probe)"]
  separation = (worst_named.loss - best[1].loss) / math.sqrt(worst_named.sem ** 2 + best[1].sem ** 2)
  print(f"   best '{best[0]}' beats the ceiling probe by {separation:.1f} sigma")
  return scores, best[0]


def confusion(detector, design, n_events=16384):
  """Held-out accuracy and the 3x3 confusion matrix of the proxy classifier at one design."""
  import xgboost as xgb
  from detopt.bo.gbdt import sample_design
  features, target = sample_design(detector, design, np.arange(n_events, dtype=np.int64))
  label = np.argmax(target, axis=1)
  n_train = int(round(0.75 * len(label)))
  train = xgb.DMatrix(features[:n_train], label=label[:n_train])
  validation = xgb.DMatrix(features[n_train:], label=label[n_train:])
  model = xgb.train(
    {'objective': 'multi:softprob', 'num_class': N_CLASSES, 'eval_metric': 'mlogloss', 'eta': 0.08,
     'tree_method': 'hist', 'grow_policy': 'lossguide', 'max_leaves': 31, 'max_depth': 0,
     'min_child_weight': 40, 'seed': 0, 'nthread': int(os.environ.get('SLURM_CPUS_PER_TASK', 4))},
    train, num_boost_round=400, evals=[(validation, 'val')], early_stopping_rounds=50, verbose_eval=False
  )
  guess = np.argmax(model.predict(validation, iteration_range=(0, model.best_iteration + 1)), axis=1)
  truth = label[n_train:]
  matrix = np.zeros((N_CLASSES, N_CLASSES))
  for i in range(N_CLASSES):
    for j in range(N_CLASSES):
      matrix[i, j] = np.mean(guess[truth == i] == j)
  return float(np.mean(guess == truth)), matrix


def section_6(detector, designs, n_compounds=256):
  """CLASS CONTRAST -- the physical separability of the target at a design, with no estimator in it.

  Hold the enzyme, its turnover and the dominant branch's potency FIXED; swap ONLY the compound's
  class (each at the midpoint of its own band); measure how far the NOISELESS read-out moves, in
  units of the read-out noise."""
  print(f"\n6. CLASS CONTRAST -- noiseless separation between the three mechanisms, in read-out sigmas "
        f"(median over {n_compounds} compounds)", flush=True)
  typed = 0.5 * sum(detector.preference_bounds)
  noncompetitive = 0.5 * sum(detector.preference_bounds_noncompetitive)
  # (competitive, noncompetitive, uncompetitive) as (log10 k1, log10 k2) offsets from k_strong.
  variants = ((0.0, -typed), (0.0, -noncompetitive), (-typed, 0.0))

  def separations(design):
    def one(index):
      parameters, half_time, log_k0_cat, _, log10_k1, log10_k2 = draw_event(detector, index)
      strong = jnp.maximum(log10_k1, log10_k2)
      readouts = jnp.stack([
        noiseless_readout(detector, design, parameters, log_k0_cat,
                          jnp.power(10.0, strong + a), jnp.power(10.0, strong + b))
        for a, b in variants
      ])
      return readouts

    readouts = jax.jit(jax.vmap(one))(jnp.arange(n_compounds, dtype=jnp.int32))
    return np.asarray(readouts)

  for name, design in designs.items():
    readouts = separations(design)
    print(f"   {name}")
    for i, j in ((0, 1), (0, 2), (1, 2)):
      difference = readouts[:, i] - readouts[:, j]
      rms = np.sqrt(np.mean(np.square(difference), axis=(1, 2))) / detector.measurement_noise
      print(f"      {CLASS_NAMES[i]:22s} vs {CLASS_NAMES[j]:22s} {np.median(rms):7.3f}   "
            f"(90th pct {np.quantile(rms, 0.9):.3f}; the batch carries "
            f"{detector.n_experiments * detector.n_measurements} read-outs)")


def section_7(detector, designs, n_compounds=128):
  """IDENTIFIABILITY -- the marginalised Fisher information for (log10 k1, log10 k2).

  ``I_eff = I_tt - I_tn inv(I_nn) I_nt`` (the Schur complement over the nuisances): what knowing the
  two branch strengths costs once you admit you do not know the Michaelis constants or the turnover.
  The nuisances are the drawn kinetic parameters, each scaled by its own prior half-range, and the
  calibrated turnover, scaled by the half-time prior's log half-range -- so a unit of nuisance is a
  prior width and the numbers are comparable across designs. The targets stay in DEX.

  ``I_eff`` alone is nearly rank-1 here (only the ``k2`` direction is measured pointwise), so the
  frequentist error on ``d = log10(k1/k2)`` is infinite at EVERY design and says nothing. The
  reported ``sigma(d)`` therefore includes the POPULATION PRIOR -- the empirical covariance of
  ``(log10 k1, log10 k2)`` over the library -- as ``inv(I_eff + inv(Cov_prior))``, which is the
  information a Bayesian classifier actually has. ``sigma(d) / sigma_prior(d)`` is then bounded by 1:
  it is the fraction of the mechanism axis's prior width the design fails to resolve, and 1.000 means
  the design taught nothing about the mechanism."""
  print(f"\n7. IDENTIFIABILITY -- marginalised Fisher information, nuisances = the {len(PARAMETER_NAMES)} "
        f"kinetic parameters + the turnover (median over {n_compounds} compounds)", flush=True)
  half_range = jnp.asarray([0.5 * (r[1] - r[0]) for _, r in detector._parameter_ranges], jnp.float32)
  turnover_scale = 0.5 * (math.log(detector.half_time_bounds[1]) - math.log(detector.half_time_bounds[0]))

  def information(design, index):
    parameters, half_time, log_k0_cat, _, log10_k1, log10_k2 = draw_event(detector, index)
    base = jnp.stack([parameters[name] for name in PARAMETER_NAMES])

    def model(theta):
      # theta = (log10 k1, log10 k2, 9 scaled kinetic parameters, scaled turnover)
      kinetic = base + theta[2:2 + len(PARAMETER_NAMES)] * half_range
      drawn = {name: kinetic[i] for i, name in enumerate(PARAMETER_NAMES)}
      return noiseless_readout(
        detector, design, drawn, log_k0_cat + theta[-1] * turnover_scale,
        jnp.power(10.0, theta[0]), jnp.power(10.0, theta[1])
      ).ravel()

    theta = jnp.concatenate([jnp.stack([log10_k1, log10_k2]), jnp.zeros(len(PARAMETER_NAMES) + 1)])
    jacobian = jax.jacfwd(model)(theta)
    return jnp.matmul(jacobian.T, jacobian) / (detector.measurement_noise ** 2)

  for name, design in designs.items():
    matrices = np.asarray(jax.jit(jax.vmap(lambda i: information(design, i)))(
      jnp.arange(n_compounds, dtype=jnp.int32)))
    effective, sigma_d = [], []
    for matrix in matrices:
      target_block, cross = matrix[:2, :2], matrix[:2, 2:]
      nuisance = matrix[2:, 2:]
      schur = target_block - cross @ np.linalg.pinv(nuisance, rcond=1e-10) @ cross.T
      effective.append(np.diag(schur))
      # sigma(d) = sqrt(v^T I_eff^-1 v) with v = (1, -1). A SINGULAR I_eff means infinite variance,
      # so the inverse is taken eigenvalue-wise with the eigenvalues clipped from BELOW at a
      # numerical floor -- a pseudo-inverse would do the opposite (drop the singular directions and
      # report sigma = 0, i.e. "perfectly measured" exactly where nothing is measured at all).
      eigenvalues, vectors = np.linalg.eigh(0.5 * (schur + schur.T))
      projection = np.square(vectors.T @ np.array([1.0, -1.0]))
      sigma_d.append(math.sqrt(float(np.sum(projection / np.maximum(eigenvalues, 1e-24)))))
    effective = np.asarray(effective)
    print(f"   {name:38s} I_eff(log10 k1) {np.median(effective[:, 0]):10.3g}  "
          f"I_eff(log10 k2) {np.median(effective[:, 1]):10.3g}  median sigma(d) {np.median(sigma_d):9.3g} dex")


def main():
  detector = build()
  print(f"design dimension {detector.design_dim()} = 4 x {detector.n_experiments} experiments; "
        f"target {N_CLASSES} classes; loss = cross-entropy / ln {N_CLASSES} (mechanism class)\n", flush=True)

  designs = section_1(detector)
  section_1b(detector)
  section_2(detector, designs)
  section_3(detector)
  scores, best = section_4_5(detector, designs)
  accuracy, matrix = confusion(detector, designs[best])
  print(f"\n   held-out accuracy at \"{best}\": {100 * accuracy:.1f}% (chance {100 / N_CLASSES:.1f}%)")
  print("   confusion (rows = truth, columns = guess), as a fraction of each true class:")
  for i, name in enumerate(CLASS_NAMES):
    print(f"     {name:24s} " + "  ".join(f"{v:.3f}" for v in matrix[i]))
  section_6(detector, designs)
  section_7(detector, designs)

  # Cost, measured: what the screen will spend per design.
  design = designs[best]
  started = time.time()
  for offset in range(4):
    score_design(detector, design, n_events=4096, event_offset=offset * 4096, seed=0)
  print(f"\n8. COST: {(time.time() - started) / 4:.2f} s per design at 4096 events "
        f"(the screen's setting), steady state", flush=True)


if __name__ == "__main__":
  main()
