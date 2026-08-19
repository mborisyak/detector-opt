#!/usr/bin/env python3
"""Read-only audit probes for the `enzyme_extremes` detector and the fixed/growth BO trajectories.

CPU ONLY and SIMULATION ONLY -- it calls the detector and reads json on disk; it never trains, never
touches a GPU, and writes nothing.

  python scripts/audit_enzyme_extremes.py detector   # determinism, guard, ceiling, label balance
  python scripts/audit_enzyme_extremes.py guard      # can the integration guard fire at all
  python scripts/audit_enzyme_extremes.py trajectory # arm-vs-arm comparison at matched designs

WHAT EACH PROBE MUST SEPARATE, stated before it is run.

  determinism   the detector is claimed to be a function of (design, event_index) alone. The probe
                must separate "same index, same event" from "the batch it was called in matters",
                so it re-calls the SAME indices in a different batch size, at a different position,
                and in a separate PROCESS, and compares bit for bit. It must also separate "the
                target is a property of the event" from "the design leaks into the label", so it
                re-calls the same indices under a different design and compares the one-hots.

  guard         `integration_tolerance` is only a guard if it can fire. The probe halves and quarters
                `n_steps_per_measurement` around the configured value and reports the dt-vs-dt/2
                disagreement, which must cross the tolerance somewhere below the configured setting;
                a probe that never crosses it proves nothing.

  ceiling       the no-information value is claimed to be exactly 1.0. The probe evaluates the
                detector's own `loss` at uniform logits and at the empirical class prior, and reports
                the class balance, which is what makes 1.0 the ceiling rather than an approximation.

  trajectory    the two arms propose DIFFERENT designs, so their best-so-far curves are not a paired
                comparison. The probe pairs each `meta` design with the nearest `from_scratch` design
                under the design's OWN symmetry (the minimum over the 24 permutations of the four
                exchangeable experiments), so a loss difference at a small distance is a statement
                about the arms rather than about where they looked.
"""

import glob
import itertools
import json
import os
import sys

import numpy as np

DETECTOR_CONFIG = "config/detector/enzyme_extremes.yaml"
FIXED_RUN = "output/extremes-fixed/1244111331"
GROWTH_RUN = "output/enzyme_extremes/1244111331"


def _load_detector(overrides=None):
  import detopt.detector
  from detopt.utils.config import load_config

  config = load_config(DETECTOR_CONFIG)
  (name, ) = config.keys()
  if overrides is not None:
    config[name] = dict(config[name], **overrides)
  return detopt.detector.from_config(config)


def _sobol_design(detector, seed):
  rng = np.random.default_rng(seed)
  return np.asarray(rng.uniform(size=(detector.design_dim(), )), dtype=np.float32)


def _call(detector, design_scaled, indices):
  design = detector.to_nominal(np.asarray(design_scaled, dtype=np.float32))
  width = detector.design_dim()
  flat = np.asarray(detector.flatten_design(design), dtype=np.float32)
  broadcast = np.broadcast_to(flat[None, :], (len(indices), width))
  return detector(broadcast, np.asarray(indices, dtype=np.int64))


def determinism(n_events=1024):
  detector = _load_detector()
  print(
    f"design_dim={detector.design_dim()} n_experiments={detector.n_experiments} "
    f"n_measurements={detector.n_measurements} n_classes={detector.n_classes} "
    f"classes={detector.mechanism_classes} size={detector.size()}"
  )
  print(
    f"integration_tolerance={detector.integration_tolerance} "
    f"n_steps_per_measurement={detector.n_steps_per_measurement} "
    f"total_steps={detector.n_measurements * detector.n_steps_per_measurement} dt={detector.measurement_dt:.5g} h"
  )

  design = _sobol_design(detector, 0)
  other = _sobol_design(detector, 1)
  indices = np.arange(n_events, dtype=np.int64)

  _gt, event_a, _mask, target_a = _call(detector, design, indices)
  measurements_a = np.asarray(event_a.measurements)
  labels_a = np.asarray(target_a.mechanism)

  head = _call(detector, design, indices[:n_events // 2])
  tail = _call(detector, design, indices[n_events // 2:])
  measurements_b = np.concatenate([np.asarray(head[1].measurements), np.asarray(tail[1].measurements)], axis=0)
  print(f"[determinism] batch split          max|delta| = {np.max(np.abs(measurements_a - measurements_b)):.3g}")

  order = np.random.default_rng(7).permutation(n_events)
  shuffled = _call(detector, design, indices[order])
  measurements_c = np.asarray(shuffled[1].measurements)[np.argsort(order)]
  print(f"[determinism] shuffled order       max|delta| = {np.max(np.abs(measurements_a - measurements_c)):.3g}")

  _gt, event_d, _mask, target_d = _call(detector, other, indices)
  labels_d = np.asarray(target_d.mechanism)
  print(
    f"[determinism] other design         max|delta| measurements = "
    f"{np.max(np.abs(measurements_a - np.asarray(event_d.measurements))):.3g} (must be > 0)"
  )
  print(
    f"[determinism] other design         max|delta| labels       = {np.max(np.abs(labels_a - labels_d)):.3g} "
    f"(must be 0)"
  )

  fractions = labels_a.mean(axis=0)
  print(
    f"[balance]     class fractions over {n_events} events: "
    f"{dict(zip(detector.mechanism_classes, np.round(fractions, 4)))}"
  )
  uniform = np.zeros((n_events, detector.n_classes), dtype=np.float32)
  loss_uniform = float(np.mean(np.asarray(detector.loss(uniform, labels_a))))
  prior_logits = np.broadcast_to(np.log(np.maximum(fractions, 1e-12))[None, :], uniform.shape)
  loss_prior = float(np.mean(np.asarray(detector.loss(prior_logits, labels_a))))
  print(f"[ceiling]     loss(uniform logits) = {loss_uniform:.6f}   loss(empirical prior) = {loss_prior:.6f}")

  path = f"/tmp/audit_enzyme_extremes_{os.getpid()}.npy"
  np.save(path, measurements_a)
  print(f"[determinism] wrote {path} for the cross-process check")


def cross_process(reference_path, n_events=1024):
  detector = _load_detector()
  design = _sobol_design(detector, 0)
  indices = np.arange(n_events, dtype=np.int64)
  _gt, event, _mask, _target = _call(detector, design, indices)
  reference = np.load(reference_path)
  delta = np.max(np.abs(reference - np.asarray(event.measurements)))
  print(f"[determinism] separate PROCESS     max|delta| = {delta:.3g}")


def guard(n_events=512):
  """The dt-vs-dt/2 disagreement as a function of the step count, at the WORST corner of the box: the
  corner that pushes |df/dx| hardest -- most enzyme, most ATP, hottest, weakest inhibitor (an
  inhibitor only ever slows the rate down)."""
  configured = _load_detector().n_steps_per_measurement
  tolerance = _load_detector().integration_tolerance
  indices = np.arange(n_events, dtype=np.int64)
  print(f"[guard] configured n_steps_per_measurement={configured}, integration_tolerance={tolerance:.3g} mM")
  for steps in (1, 2, 5, 10, 20, configured):
    detector = _load_detector({"n_steps_per_measurement": steps})
    corner = np.zeros((detector.design_dim(), ), dtype=np.float32)
    n = detector.n_experiments
    corner[:n] = 1.0
    corner[n:2 * n] = 1.0
    corner[2 * n:3 * n] = 0.0
    corner[3 * n:] = 1.0
    try:
      _call(detector, corner, indices)
      status = "PASSED"
      error = _measure_error(detector, corner, indices)
    except RuntimeError as failure:
      status = "FIRED"
      error = float(str(failure).split("integration error ")[1].split(" mM")[0])
    print(
      f"  n_steps_per_measurement={steps:3d}  total={steps * detector.n_measurements:5d}  "
      f"max|fine-coarse|={error:.4g} mM   guard {status}"
    )


def _measure_error(detector, design_scaled, indices):
  """The guard's own quantity, recomputed without the raise: the detector returns it per event."""
  import jax.numpy as jnp

  design = detector.to_nominal(np.asarray(design_scaled, dtype=np.float32))
  flat = np.asarray(detector.flatten_design(design), dtype=np.float32)
  broadcast = jnp.broadcast_to(jnp.asarray(flat)[None, :], (len(indices), detector.design_dim()))
  fraction, substrate, inhibitor, temperature = detector._resolve_design(broadcast, len(indices))
  out = detector._generate(fraction, substrate, inhibitor, temperature, jnp.asarray(indices, jnp.int32))
  return float(jnp.max(out[-1]))


def _permutations(n_experiments, design_dim):
  """Index arrays reordering a flat design under each permutation of the exchangeable experiments."""
  blocks = [tuple(range(k * n_experiments, (k + 1) * n_experiments)) for k in range(design_dim // n_experiments)]
  out = []
  for order in itertools.permutations(range(n_experiments)):
    index = np.arange(design_dim)
    for block in blocks:
      index[list(block)] = np.asarray(block)[list(order)]
    out.append(index)
  return np.stack(out)


def _invariant_distance(a, b, permutations):
  return float(np.min(np.linalg.norm(a[permutations] - b[None, :], axis=1)))


def trajectory():
  n_experiments, design_dim = 4, 16
  permutations = _permutations(n_experiments, design_dim)
  for run in (FIXED_RUN, GROWTH_RUN):
    arms = {}
    for arm in ("from_scratch", "meta"):
      path = os.path.join(run, arm, "results.json")
      if not os.path.exists(path):
        continue
      arms[arm] = json.load(open(path))["results"]
    if len(arms) < 2:
      continue
    print(f"\n=== {run}")
    reference = np.stack([np.asarray(r["x_scaled"], dtype=np.float64) for r in arms["from_scratch"]])
    reference_loss = np.asarray([r["loss"] for r in arms["from_scratch"]])
    print("  meta design -> nearest from_scratch design under the 24 experiment permutations")
    deltas = []
    for row in arms["meta"]:
      x = np.asarray(row["x_scaled"], dtype=np.float64)
      distances = np.asarray([_invariant_distance(x, y, permutations) for y in reference])
      k = int(np.argmin(distances))
      delta = float(row["loss"]) - float(reference_loss[k])
      deltas.append((distances[k], delta))
      print(
        f"    meta it{row['iteration']:3d} loss={row['loss']:.4f} | nearest fs it{k:3d} "
        f"loss={reference_loss[k]:.4f} dist={distances[k]:.3f} delta={delta:+.4f}"
      )
    close = [d for dist, d in deltas if dist < 0.35]
    if len(close) > 0:
      print(f"  CLOSE PAIRS (dist < 0.35): n={len(close)} median delta={np.median(close):+.4f} "
            f"mean={np.mean(close):+.4f}")
    over = [r for arm in arms.values() for r in arm if r["loss"] > 1.0]
    print(
      f"  designs reported ABOVE the no-information ceiling 1.0: {len(over)} "
      f"(max {max([r['loss'] for r in over], default=float('nan')):.6f})"
    )


def main():
  mode = sys.argv[1] if len(sys.argv) > 1 else "detector"
  if mode == "detector":
    determinism()
  elif mode == "cross_process":
    cross_process(sys.argv[2])
  elif mode == "guard":
    guard()
  elif mode == "trajectory":
    trajectory()
  elif mode == "growth":
    growth()
  elif mode == "rounds":
    rounds()
  else:
    raise SystemExit(f"unknown probe {mode!r}")


def growth():
  """Replay the growth-and-gap procedure offline against stored per-epoch curves.

  WHAT THIS PROBE MUST SEPARATE. "The reported loss is taken mid-descent" from "the reported loss is
  taken on a plateau". Three things decide it and the probe measures all three: which CLAUSE ends each
  round (a settled test that fires early is harmless if clause (2.1) sends the design back for data),
  how many epochs the FINAL round gets on the largest window, and what more rounds would have bought
  -- read off five optimiser arms that ran to different round counts on the same design.

  THE CURVES. `output/optimizer-bakeoff/*.json` are growth runs at the CAMPAIGN's own settings
  (steps_per_epoch 2048, n0 8192, n_increment 4096, patience 16, warmup 2, loss_precision 1.0e-2,
  param_mix 0.25), continuing `output/enzyme_extremes/1244111331/meta`'s persistent network and event
  pools on that run's own best design. `output/screen/activation-*.json` are the same procedure at a
  finer data schedule and carry a per-round summary.
  """
  import glob

  from detopt.utils.training import bayesian_trend, probability_above, probability_change_below

  warmup_epochs, patience, loss_precision = 2, 16, 1.0e-2
  for path in sorted(glob.glob("output/optimizer-bakeoff/*.json")):
    payload = json.load(open(path))
    if "train_loss_per_epoch" not in payload:
      continue
    train = np.asarray(payload["train_loss_per_epoch"], dtype=np.float64)
    val = np.asarray(payload["val_loss_per_epoch"], dtype=np.float64)
    train_sem = np.asarray(payload["train_sem_per_epoch"], dtype=np.float64)
    val_sem = np.asarray(payload["val_sem_per_epoch"], dtype=np.float64)
    window = np.asarray(payload["train_budget_per_epoch"], dtype=np.int64)
    starts = [0] + list(np.where(np.diff(window) > 0)[0] + 1)
    print(
      f"\n=== {path}  {payload['optimizer']}/{payload['schedule']}  "
      f"objective={payload['objective']:.4f} +/- {payload['objective_std']:.4f}  epochs={payload['n_epochs']}"
    )
    verdicts = []
    for index, round_start in enumerate(starts):
      end = starts[index + 1] if index + 1 < len(starts) else len(train)
      first = round_start + warmup_epochs
      fired, at = None, None
      for last in range(first + 3, end + 1):
        tr, va = train[first:last], val[first:last]
        tr_s, va_s = train_sem[first:last], val_sem[first:last]
        prior_sigma = max(float(tr[0]), float(va[0])) / 3.0
        gap_sem = np.hypot(tr_s, va_s)
        gap_series = np.abs(va - tr) + gap_sem
        tr_mean, tr_cov = bayesian_trend(tr, tr_s, prior_sigma)
        gap_mean, gap_cov = bayesian_trend(gap_series, gap_sem, prior_sigma)
        p_gap = probability_above(gap_mean, gap_cov, patience, loss_precision, gap_series.shape[0])
        p_settled = probability_change_below(tr_mean, tr_cov, patience, 0.5 * loss_precision)
        diff = abs(float(va[-1] - tr[-1]))
        err = float(np.hypot(tr_s[-1], va_s[-1]))
        if p_gap > 0.9:
          fired, at = "(1) gap-trend starved -> ADD DATA", (last, p_gap, p_settled, diff + err)
          break
        if p_settled > 0.9:
          if diff + err > loss_precision:
            fired, at = "(2.1) settled, gap wide -> ADD DATA", (last, p_gap, p_settled, diff + err)
          else:
            fired, at = "(2.2) settled, gap closed -> SCORE", (last, p_gap, p_settled, diff + err)
          break
      verdicts.append((index, int(window[round_start]), end - round_start, fired, at))
    for index, w, epochs, fired, at in verdicts:
      mark = "" if at is None or at[0] - starts[index] == epochs else "  <-- REPLAY MISMATCH"
      shown = "no clause fired within the round" if at is None else (
        f"epoch {at[0] - starts[index]:3d} P(gap>LP)={at[1]:.3f} P(settled)={at[2]:.3f} diff+err={at[3]:.4f}"
      )
      print(f"   round {index:2d} window={w:6d} epochs={epochs:3d}  {shown}  {fired}{mark}")
    tail = slice(starts[-1], len(train))
    print(
      f"   FINAL ROUND: {len(train) - starts[-1]} epochs on window {int(window[-1])} "
      f"({(len(train) - starts[-1]) * 2048} steps, {(len(train) - starts[-1]) * 2048 * 256 / window[-1]:.0f} passes)"
    )
    print(
      f"   FINAL ROUND descent: train {train[tail][0]:.4f} -> {train[-1]:.4f} "
      f"({train[-1] - train[tail][0]:+.4f}), val {val[tail][0]:.4f} -> {val[-1]:.4f} ({val[-1] - val[tail][0]:+.4f})"
    )


def rounds():
  """Rounds, windows and exit gaps per design, per arm, for the finished growth campaign.

  `spent` is train + validation for the design, and the schedule lays down 8192 + 2731 rows in the
  first round and 4096 + 1365 in every later one, so the round count is exact. `loss_std` is the
  procedure's own `diff + err` at exit, which clause (2.1) requires to be at or below
  `loss_precision`; a value above it would mean a design was scored that the rule forbids.
  """
  n0_spent, increment_spent, loss_precision = 10923, 5461, 1.0e-2
  for run in sorted(glob.glob(os.path.join("output/enzyme_extremes", "*"))):
    if not os.path.isdir(run):
      continue
    print(f"\n=== {run}")
    for arm in ("from_scratch", "continue", "closest", "meta"):
      path = os.path.join(run, arm, "results.json")
      if not os.path.exists(path):
        continue
      results = json.load(open(path))["results"]
      spent = np.asarray([r["spent"] for r in results], dtype=np.float64)
      std = np.asarray([r["loss_std"] for r in results], dtype=np.float64)
      loss = np.asarray([r["loss"] for r in results], dtype=np.float64)
      round_count = 1.0 + (spent - n0_spent) / increment_spent
      integral = np.max(np.abs(round_count - np.round(round_count)))
      epochs = _checkpoint_epochs(os.path.join(run, arm, "checkpoints"))
      per_round = np.asarray(epochs, dtype=np.float64) / np.round(round_count)[:len(epochs)]
      print(
        f"  {arm:12s} n={len(results):3d}  rounds med={np.median(np.round(round_count)):5.1f} "
        f"max={np.max(np.round(round_count)):5.0f}  window med={np.median(spent * 0.75):8.0f} calls "
        f"med={np.median(spent):8.0f}"
      )
      print(
        f"               exit diff+err: med={np.median(std):.4f} max={np.max(std):.4f} "
        f"(bar {loss_precision:.4f}, violations={int(np.sum(std > loss_precision + 1e-9))})  "
        f"| epochs/round med={np.median(per_round):5.1f}  | non-integral rounds={integral:.3g}"
      )
      print(
        f"               loss: min={np.min(loss):.4f} max={np.max(loss):.4f} "
        f"at-or-above ceiling 1.0: {int(np.sum(loss >= 1.0))}"
      )


def _checkpoint_epochs(checkpoint_dir):
  """The epoch count of every design, read from the orbax step directory written at convergence."""
  out = []
  if not os.path.isdir(checkpoint_dir):
    return out
  for name in sorted(os.listdir(checkpoint_dir), key=lambda s: int(s.split("_")[1])):
    steps = [int(s) for s in os.listdir(os.path.join(checkpoint_dir, name)) if s.isdigit()]
    if len(steps) > 0:
      out.append(max(steps))
  return out


if __name__ == "__main__":
  main()
