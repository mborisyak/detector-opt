#!/usr/bin/env python3
"""Tables for the convergence-rule bias benchmark: the three step quantities kept APART.

    python scripts/report_bias_cross.py --task linear output/costbias/cb_a.json output/costbias/cb_b.json

THE THREE QUANTITIES, which are routinely conflated and on whose difference the whole question turns:

  CALLS / WINDOW      how much data the design bought -- the budget quantity;
  EPOCHS             how long it optimised on that data;
  REMAINING DESCENT  `final - c` from `A * exp(alpha * t) + c` fitted to the TRAIN curve after the
                     LAST data addition (`scripts/fit_tail.py`'s model, applied to the per-epoch
                     history every measurement already stores) -- the direct "did it stop early".

A step-count difference with EQUAL remaining descent is efficiency and costs nothing. A step-count
difference WITH different remaining descent is the criterion biasing the arms against each other.

THE EXIT SPLIT. The scoring gate is `gap + err <= loss_precision` with `gap = |val - train|` and
`err = hypot(train_sem, val_sem)`, and `err = c / sqrt(window)`. An arm that exits at a smaller window
carries a larger `err` and therefore meets the same bar with a larger true gap left over, so the two
terms are reported SEPARATELY at every exit.

THE NOISE-TO-TREND RATIO, per arm: the rms residual about the fitted tail against the fitted slope at
the final epoch. A "no new best in patience epochs" rule stops on noise once the trend falls below the
scatter; the Bayesian rule tests a posterior instead, so noise widens it rather than firing it. If
replay changes the batch composition it plausibly changes this ratio, so it is measured.

WHAT IS AND IS NOT A BIAS. `excess = loss - bayes_risk` exists on `linear` alone. On the enzyme tasks
there is no closed form and none is invented: `test` on a fixed held-out pool at the exact design is
reported instead, and only DIFFERENCES between cells at the SAME design are interpretable.

THIS SCRIPT MEASURES NOTHING. It reads the JSONs the probes wrote.
"""

from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np
import scipy.optimize
import scipy.stats

from detopt.utils.training import bayesian_trend, probability_above, probability_change_below


def tail_fit(values):
  """`(a, alpha, c, rms_residual)` of `a * exp(alpha * t) + c` over `values`, or None if the fit is not
  usable.

  REJECTED, and counted rather than silently dropped: fewer than four points; a `curve_fit` that does
  not converge; a non-finite parameter; and a RUNAWAY -- an extrapolated asymptote further from the
  final value than five times the whole range the series covered in that round. An exponential fitted
  to an almost-flat noisy tail can put `c` arbitrarily far away with an arbitrarily small `alpha`, and
  such a fit says nothing about remaining descent. Its `remaining` is reported as None.
  """
  loss = np.asarray(values, np.float64)
  if loss.size < 4:
    return None
  t = np.arange(loss.size, dtype=np.float64)
  guess = (loss[0] - loss[-1], -1.0 / max(loss.size / 3.0, 1.0), loss[-1])
  try:
    (a, alpha,
     c), _ = scipy.optimize.curve_fit(lambda t, a, alpha, c: a * np.exp(alpha * t) + c, t, loss, p0=guess, maxfev=20000)
  except (RuntimeError, TypeError, ValueError):
    return None
  if not all(np.isfinite(v) for v in (a, alpha, c)):
    return None
  span = float(np.max(loss) - np.min(loss))
  if abs(float(loss[-1]) - float(c)) > 5.0 * span + 1e-12:
    return None
  residual = float(np.sqrt(np.mean((a * np.exp(alpha * t) + c - loss)**2)))
  return float(a), float(alpha), float(c), residual


def enrich(row):
  """Add the per-row tail quantities, in place. `remaining_train` is the descent the TRAIN curve still
  had left at the stop; `slope_at_stop` is the fitted derivative at the final epoch and
  `noise_to_trend` its ratio against the rms residual."""
  history = row["per_epoch"]
  window = np.asarray(history["window"], np.int64)
  changed = np.flatnonzero(np.diff(window) != 0)
  start = int(changed[-1]) + 1 if changed.size > 0 else 0
  row["final_round_epochs"] = int(window.size - start)
  for name, key in (("train", "remaining_train"), ("val", "remaining_val")):
    series = np.asarray(history[name], np.float64)[start:]
    fit = tail_fit(series)
    if fit is None:
      row[key] = None
      row[f"{name}_tail_alpha"] = None
      row[f"{name}_tail_residual"] = None
      row[f"{name}_slope_at_stop"] = None
      continue
    a, alpha, c, residual = fit
    row[key] = float(series[-1] - c)
    row[f"{name}_tail_alpha"] = alpha
    row[f"{name}_tail_residual"] = residual
    row[f"{name}_slope_at_stop"] = float(a * alpha * math.exp(alpha * (series.size - 1)))
  if row["train_tail_residual"] is not None and row["train_slope_at_stop"] is not None:
    denominator = abs(row["train_slope_at_stop"])
    row["noise_to_trend"] = float(row["train_tail_residual"] / denominator) if denominator > 0.0 else float("inf")
  else:
    row["noise_to_trend"] = None
  return row


def clause_trace(row, warmup_epochs, loss_precision):
  """Replay the SACRED procedure epoch by epoch over the stored history and record WHICH CLAUSE fired.

  `design.py` is deterministic given the per-epoch train/validation means and SEMs, which every
  measurement stores, so this reconstructs the decisions without touching the loop. It answers the
  question the growth schedule raises: a settled test can be overruled by clause (2.1), which ADDS
  DATA, so the score may be decided by the gap gate rather than by the convergence test. Returns the
  counts of each firing and the two probabilities at the exit epoch.
  """
  history = row["per_epoch"]
  train = np.asarray(history["train"], np.float64)
  validation = np.asarray(history["val"], np.float64)
  train_sem = np.asarray(history["train_sem"], np.float64)
  validation_sem = np.asarray(history["val_sem"], np.float64)
  window = np.asarray(history["window"], np.int64)
  patience = int(row["patience"])
  counts = {"clause_1": 0, "clause_2_1": 0, "clause_2_2": 0, "clause_3": 0}
  exit_probabilities = (None, None)
  round_start = 0
  for epoch in range(train.size):
    if epoch > 0 and window[epoch] != window[epoch - 1]:
      round_start = epoch
    if epoch - round_start + 1 <= warmup_epochs:
      continue
    first = round_start + warmup_epochs
    tr = train[first:epoch + 1]
    va = validation[first:epoch + 1]
    tr_s = train_sem[first:epoch + 1]
    va_s = validation_sem[first:epoch + 1]
    if tr.shape[0] < 3:
      continue
    prior_sigma = max(float(tr[0]), float(va[0])) / 3.0
    gap_sem = np.hypot(tr_s, va_s)
    gap_series = np.abs(va - tr) + gap_sem
    tr_mean, tr_cov = bayesian_trend(tr, tr_s, prior_sigma)
    gap_mean, gap_cov = bayesian_trend(gap_series, gap_sem, prior_sigma)
    p_gap = float(probability_above(gap_mean, gap_cov, patience, loss_precision, gap_series.shape[0]))
    if p_gap > 0.9:
      counts["clause_1"] += 1
      exit_probabilities = (p_gap, None)
      continue
    p_settled = float(probability_change_below(tr_mean, tr_cov, patience, 0.5 * loss_precision))
    exit_probabilities = (p_gap, p_settled)
    if p_settled <= 0.9:
      counts["clause_3"] += 1
      continue
    slack = abs(float(va[-1]) - float(tr[-1])) + float(gap_sem[-1])
    if slack > loss_precision:
      counts["clause_2_1"] += 1
    else:
      counts["clause_2_2"] += 1
  row["n_clause_1"] = counts["clause_1"]
  row["n_clause_2_1"] = counts["clause_2_1"]
  row["n_clause_2_2"] = counts["clause_2_2"]
  row["n_clause_3"] = counts["clause_3"]
  row["p_gap_at_exit"] = exit_probabilities[0]
  row["p_settled_at_exit"] = exit_probabilities[1]
  row["additions_from_gap_gate"] = counts["clause_2_1"]
  row["additions_from_trend"] = counts["clause_1"]
  return row


def clause_table(rows, title):
  """Which clause bought the data, per cell -- the gap-gate route against the trend route."""
  groups = collections.defaultdict(list)
  for row in rows:
    groups[(row["arm"], row.get("cell", row["dropconnect"]), row["patience"])].append(row)
  print(f"\n### {title}   (medians; additions attributed to the clause that fired)")
  print(
    f"{'arm':<13}{'cell':>16}{'pat':>5}{'n':>4}{'add(1)':>9}{'add(2.1)':>10}{'train-on':>10}"
    f"{'P(gap>LP)':>11}{'P(settled)':>12}"
  )
  for key in sorted(groups, key=lambda k: (k[0], str(k[1]), k[2])):
    arm, label, patience = key
    cell = groups[key]
    print(
      f"{arm:<13}{str(label):>16}{patience:>5}{len(cell):>4}{median(cell, 'n_clause_1'):>9.1f}"
      f"{median(cell, 'n_clause_2_1'):>10.1f}{median(cell, 'n_clause_3'):>10.1f}"
      f"{median(cell, 'p_gap_at_exit'):>11.3f}{median(cell, 'p_settled_at_exit'):>12.3f}"
    )


def median(rows, key):
  values = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
  return float(np.median(values)) if len(values) > 0 else float("nan")


def cell_table(rows, has_floor, title):
  """One line per `arm x dropconnect x patience` cell."""
  groups = collections.defaultdict(list)
  for row in rows:
    groups[(row["arm"], row["dropconnect"], row["patience"])].append(row)
  excess_key = "excess_paired" if has_floor else "test"
  excess_name = "excess" if has_floor else "test"
  print(f"\n### {title}   (medians; n = measurements in the cell)")
  print(
    f"{'arm':<13}{'dc':>5}{'pat':>5}{'n':>4}{'window':>9}{'calls':>9}{'epochs':>8}{'lastrnd':>8}"
    f"{'reported':>10}{excess_name:>10}{'remain_tr':>11}{'remain_va':>11}{'gap':>9}{'err':>9}{'err/slack':>10}"
    f"{'tail_rms':>10}{'tail_slope':>12}{'n2t':>10}{'wall_s':>8}"
  )
  for key in sorted(groups):
    arm, dropconnect, patience = key
    cell = groups[key]
    gap = median(cell, "diff")
    err = median(cell, "err")
    share = err / (gap + err) if (gap + err) > 0.0 else float("nan")
    print(
      f"{arm:<13}{dropconnect:>5.2f}{patience:>5}{len(cell):>4}{median(cell, 'window'):>9.0f}"
      f"{median(cell, 'calls_train_val'):>9.0f}{median(cell, 'n_epochs'):>8.0f}{median(cell, 'final_round_epochs'):>8.0f}"
      f"{median(cell, 'val'):>10.4f}{median(cell, excess_key):>10.5f}{median(cell, 'remaining_train'):>11.5f}"
      f"{median(cell, 'remaining_val'):>11.5f}{gap:>9.5f}{err:>9.5f}{share:>10.3f}"
      f"{median(cell, 'train_tail_residual'):>10.6f}{median(cell, 'train_slope_at_stop'):>12.2e}"
      f"{median(cell, 'noise_to_trend'):>10.1f}{median(cell, 'wall_s'):>8.0f}"
    )


def sign_test(differences):
  """`(n, n_positive, two_sided_p)` of a sign test on the non-zero differences."""
  values = [d for d in differences if d is not None and np.isfinite(d) and d != 0.0]
  if len(values) == 0:
    return 0, 0, float("nan")
  positive = sum(1 for d in values if d > 0.0)
  p = float(scipy.stats.binomtest(positive, len(values), 0.5).pvalue)
  return len(values), positive, p


def paired_arms(rows, has_floor, title):
  """`meta` against `from_scratch`, PAIRED on (seed, rank, dropconnect, patience)."""
  index = {}
  for row in rows:
    index[(row["arm"], row["seed"], row["rank"], row["dropconnect"], row["patience"])] = row
  keys = (
    "window", "calls_train_val", "n_epochs", "remaining_train", "remaining_val", "diff", "err", "val",
    "excess_paired" if has_floor else "test", "noise_to_trend"
  )
  groups = collections.defaultdict(lambda: collections.defaultdict(list))
  for (arm, seed, rank, dropconnect, patience), row in index.items():
    if arm != "meta":
      continue
    other = index.get(("from_scratch", seed, rank, dropconnect, patience))
    if other is None:
      continue
    for key in keys:
      if row.get(key) is None or other.get(key) is None:
        continue
      groups[(dropconnect, patience)][key].append(float(row[key]) - float(other[key]))
  if len(groups) == 0:
    print(f"\n### {title}: no matched pairs")
    return
  print(f"\n### {title}   (meta - from_scratch, paired; sign test on the pairs)")
  print(f"{'dc':>5}{'pat':>5}{'pairs':>7}  " + "".join(f"{k[:12]:>22}" for k in keys))
  for key in sorted(groups):
    dropconnect, patience = key
    cells = groups[key]
    n_pairs = max(len(v) for v in cells.values())
    line = f"{dropconnect:>5.2f}{patience:>5}{n_pairs:>7}  "
    for name in keys:
      differences = cells[name]
      if len(differences) == 0:
        line += f"{'-':>22}"
        continue
      n, positive, p = sign_test(differences)
      line += f"{np.median(differences):>13.5f} {positive:>2}/{n:<2}{'*' if p < 0.05 else ' '}"
    print(line)


def floor_pair_table(rows, title):
  """SLOT 4: the within-pair difference between two designs of the SAME floor, paired on the seed.

  `scripts/floor_matched_pairs.py` writes the pair members at ranks `2k` and `2k + 1`, so the pairing
  is positional and needs no matching on the floor here. The question is whether the reported loss
  differs between two designs that are exactly equally good; the residual floor difference is printed
  alongside so it can be seen to be negligible against whatever is found.
  """
  buckets = collections.defaultdict(dict)
  for row in rows:
    buckets[(row["rank"] // 2, row["seed"], row["arm"], row["patience"], row["loss_precision"])][row["rank"] % 2] = row
  complete = {k: v for k, v in buckets.items() if len(v) == 2}
  if len(complete) == 0:
    print(f"\n### {title}: no complete pairs yet")
    return
  print(f"\n### {title}   (member b minus member a; both members share a floor to within the tolerance)")
  print(
    f"{'pair':>5}{'floor':>10}{'d floor':>10}{'cells':>7}{'d reported':>12}{'d held-out':>12}"
    f"{'d excess':>11}{'d window':>10}{'d epochs':>10}{'signs':>8}"
  )
  per_pair = collections.defaultdict(list)
  for key, members in complete.items():
    per_pair[key[0]].append(members)
  for pair in sorted(per_pair):
    members = per_pair[pair]
    floors = [m[0]["bayes_risk"] for m in members] + [m[1]["bayes_risk"] for m in members]
    reported = [m[1]["val"] - m[0]["val"] for m in members]
    held_out = [m[1]["test"] - m[0]["test"] for m in members]
    excess = [
      m[1]["excess_paired"] - m[0]["excess_paired"] for m in members
      if m[0].get("excess_paired") is not None and m[1].get("excess_paired") is not None
    ]
    window = [m[1]["window"] - m[0]["window"] for m in members]
    epochs = [m[1]["n_epochs"] - m[0]["n_epochs"] for m in members]
    n, positive, _p = sign_test(reported)
    print(
      f"{pair:>5}{float(np.mean(floors)):>10.5f}"
      f"{abs(members[0][1]['bayes_risk'] - members[0][0]['bayes_risk']):>10.2e}{len(members):>7}"
      f"{float(np.median(reported)):>12.5f}{float(np.median(held_out)):>12.5f}"
      f"{(float(np.median(excess)) if len(excess) > 0 else float('nan')):>11.5f}"
      f"{float(np.median(window)):>10.0f}{float(np.median(epochs)):>10.0f}{f'{positive}/{n}':>8}"
    )
  everything = [m[1]["val"] - m[0]["val"] for members in per_pair.values() for m in members]
  n, positive, p = sign_test(everything)
  print(
    f"  POOLED over pairs: median |d reported| {float(np.median(np.abs(everything))):.5f}, "
    f"signed median {float(np.median(everything)):+.5f}, {positive}/{n} positive, sign test p = {p:.4g}"
  )
  print("  ⚠️ the SIGN is arbitrary (member order is the pairs file's); the MAGNITUDE is the result.")


def matched_offset(rows, title, axis):
  """THE MATCHED-DESIGN OFFSET, first class: `meta - from_scratch` on the SAME design, same seed, same
  cell, for the number BO consumes.

  The probes score a fixed set of designs under both arms, so every difference here is paired by
  construction and none of it is a difference in which designs the arms chose to visit. The offset is
  quoted in ABSOLUTE loss units and in units of the cell's own `loss_precision`, because "two
  different rulers" is a statement about the offset relative to the bar the criterion claims to meet.
  """
  index = {}
  for row in rows:
    index[(row["arm"], row["seed"], row["rank"], row["loss_precision"], row["dropconnect"], row["patience"])] = row
  buckets = collections.defaultdict(lambda: collections.defaultdict(list))
  for key, row in index.items():
    arm, seed, rank, precision, dropconnect, patience = key
    if arm != "meta":
      continue
    other = index.get(("from_scratch", seed, rank, precision, dropconnect, patience))
    if other is None:
      continue
    cell = {"loss_precision": precision, "dropconnect": dropconnect, "patience": patience}[axis]
    for name in ("val", "test", "excess_paired", "window", "calls_train_val", "n_epochs"):
      if row.get(name) is None or other.get(name) is None:
        continue
      buckets[(cell, precision)][name].append(float(row[name]) - float(other[name]))
  if len(buckets) == 0:
    print(f"\n### {title}: no matched arm pairs")
    return
  print(f"\n### {title}   (meta - from_scratch on IDENTICAL designs; sign test over the pairs)")
  print(
    f"{axis:>12}{'pairs':>7}{'d reported':>12}{'/bar':>8}{'signs':>8}{'p':>8}"
    f"{'d held-out':>12}{'/bar':>8}{'signs':>8}{'p':>8}{'d window':>10}{'d epochs':>10}"
  )
  for key in sorted(buckets):
    cell, precision = key
    values = buckets[key]
    line = f"{cell:>12g}{max(len(v) for v in values.values()):>7}"
    for name in ("val", "test"):
      differences = values.get(name, [])
      if len(differences) == 0:
        line += f"{'-':>12}{'-':>8}{'-':>8}{'-':>8}"
        continue
      n, positive, p = sign_test(differences)
      median_difference = float(np.median(differences))
      line += f"{median_difference:>12.5f}{median_difference / precision:>8.2f}{f'{positive}/{n}':>8}{p:>8.3f}"
    for name in ("window", "n_epochs"):
      differences = values.get(name, [])
      line += f"{float(np.median(differences)):>10.0f}" if len(differences) > 0 else f"{'-':>10}"
    print(line)


def patience_profile(rows, has_floor, title):
  """Calls and epochs against patience, per arm -- the "data or only epochs" question."""
  groups = collections.defaultdict(list)
  for row in rows:
    groups[(row["arm"], row["dropconnect"])].append(row)
  print(f"\n### {title}   (median calls / epochs / remaining descent, by patience)")
  for key in sorted(groups):
    arm, dropconnect = key
    cell = groups[key]
    patiences = sorted({r["patience"] for r in cell})
    calls = [median([r for r in cell if r["patience"] == p], "calls_train_val") for p in patiences]
    epochs = [median([r for r in cell if r["patience"] == p], "n_epochs") for p in patiences]
    remaining = [median([r for r in cell if r["patience"] == p], "remaining_train") for p in patiences]
    excess_key = "excess_paired" if has_floor else "test"
    excess = [median([r for r in cell if r["patience"] == p], excess_key) for p in patiences]
    print(f"  {arm} dropconnect {dropconnect:g}")
    print("    patience  " + "".join(f"{p:>12}" for p in patiences))
    print("    calls     " + "".join(f"{c:>12.0f}" for c in calls))
    print("    epochs    " + "".join(f"{e:>12.0f}" for e in epochs))
    print("    remain_tr " + "".join(f"{r:>12.5f}" for r in remaining))
    print("    bias      " + "".join(f"{x:>12.5f}" for x in excess))


def difficulty_correlation(rows, has_floor, title):
  """Is the bias DESIGN-DEPENDENT? Spearman of the bias against three difficulty proxies, over the
  designs of each cell, then pooled over cells."""
  if not has_floor:
    print(f"\n### {title}: no closed-form floor -- not computed")
    return
  proxies = ("bayes_risk", "n_epochs", "window", "train_tail_alpha")
  groups = collections.defaultdict(list)
  for row in rows:
    groups[(row["arm"], row["dropconnect"], row["patience"], row["seed"])].append(row)
  print(f"\n### {title}   (Spearman of excess_paired against each proxy, WITHIN a cell; pooled over cells)")
  print(f"{'proxy':<18}{'cells':>7}{'median rho':>12}{'rho>0':>8}{'sign p':>10}")
  for proxy in proxies:
    correlations = []
    for cell in groups.values():
      pairs = [(r[proxy], r["excess_paired"]) for r in cell if r.get(proxy) is not None and r.get("excess_paired") is not None]
      if len(pairs) < 4:
        continue
      x = [p[0] for p in pairs]
      y = [p[1] for p in pairs]
      if len(set(x)) < 3:
        continue
      correlations.append(float(scipy.stats.spearmanr(x, y).statistic))
    correlations = [c for c in correlations if np.isfinite(c)]
    if len(correlations) == 0:
      print(f"{proxy:<18}{0:>7}{'-':>12}{'-':>8}{'-':>10}")
      continue
    n, positive, p = sign_test(correlations)
    print(f"{proxy:<18}{len(correlations):>7}{np.median(correlations):>12.3f}{positive:>4}/{n:<3}{p:>10.4f}")


def per_design_bias(rows, has_floor, title):
  """The bias, per DESIGN, at the config's own patience -- a constant offset is harmless to BO, one
  that varies with the design's difficulty distorts the ranking."""
  if not has_floor:
    print(f"\n### {title}: no closed-form floor -- reporting held-out test instead")
  key = "excess_paired" if has_floor else "test"
  groups = collections.defaultdict(list)
  for row in rows:
    groups[(row["rank"], row["arm"], row["patience"])].append(row)
  ranks = sorted({r["rank"] for r in rows})
  patiences = sorted({r["patience"] for r in rows})
  arms = sorted({r["arm"] for r in rows})
  print(f"\n### {title}   ({key}, median over seeds and dropconnect)")
  header = f"{'rank':>5}{'floor':>9}"
  for arm in arms:
    for patience in patiences:
      header += f"{arm[:4] + '/' + str(patience):>12}"
  print(header)
  for rank in ranks:
    sample = next(r for r in rows if r["rank"] == rank)
    floor = sample.get("bayes_risk")
    line = f"{rank:>5}" + (f"{floor:>9.4f}" if floor is not None else f"{'n/a':>9}")
    for arm in arms:
      for patience in patiences:
        cell = groups.get((rank, arm, patience), [])
        line += f"{median(cell, key):>12.5f}" if len(cell) > 0 else f"{'-':>12}"
    print(line)


def rewind_table(rows, title):
  """The restart cells (H7/H7a): one line per `arm x cell`, plus the SPREAD across cells within an arm.

  The spread IS the measure of how much the restart machinery does for that arm; it is quoted as a
  range over the cell medians rather than as four unrelated numbers.
  """
  groups = collections.defaultdict(list)
  for row in rows:
    groups[(row["arm"], row["cell"])].append(row)
  print(f"\n### {title}   (medians over designs and seeds)")
  print(
    f"{'arm':<13}{'cell':<16}{'n':>4}{'window':>9}{'calls':>9}{'epochs':>8}{'rounds':>8}{'gap':>9}{'err':>9}"
    f"{'reported':>10}{'held-out':>10}{'excess':>10}{'displ':>9}{'remain_tr':>11}"
  )
  for key in sorted(groups):
    arm, cell = key
    subset = groups[key]
    print(
      f"{arm:<13}{cell:<16}{len(subset):>4}{median(subset, 'window'):>9.0f}{median(subset, 'calls_train_val'):>9.0f}"
      f"{median(subset, 'n_epochs'):>8.0f}{median(subset, 'n_rounds'):>8.0f}{median(subset, 'diff'):>9.5f}"
      f"{median(subset, 'err'):>9.5f}{median(subset, 'val'):>10.4f}{median(subset, 'test'):>10.4f}"
      f"{median(subset, 'excess_reported'):>10.5f}{median(subset, 'displacement'):>9.4f}"
      f"{median(subset, 'remaining_train'):>11.5f}"
    )
  print(f"\n### {title}: SPREAD across restart cells, within an arm")
  print(f"{'arm':<13}{'quantity':<20}{'min':>12}{'max':>12}{'max-min':>12}{'relative':>12}")
  for arm in sorted({r["arm"] for r in rows}):
    for quantity in ("window", "n_epochs", "excess_reported", "test", "diff"):
      values = [
        median([r for r in rows if r["arm"] == arm and r["cell"] == cell], quantity)
        for cell in sorted({r["cell"]
                            for r in rows})
      ]
      values = [v for v in values if np.isfinite(v)]
      if len(values) == 0:
        continue
      low, high = min(values), max(values)
      relative = (high - low) / abs(np.median(values)) if np.median(values) != 0.0 else float("nan")
      print(f"{arm:<13}{quantity:<20}{low:>12.5f}{high:>12.5f}{high - low:>12.5f}{relative:>12.3f}")
  print(f"\n### {title}: within-design displacement ||p-q||/||p|| by POSITION in the sequence")
  positions = sorted({r["position"] for r in rows})
  print(f"{'arm':<13}{'cell':<16}" + "".join(f"{'pos ' + str(p):>12}" for p in positions))
  for key in sorted({(r["arm"], r["cell"]) for r in rows}):
    arm, cell = key
    line = f"{arm:<13}{cell:<16}"
    for position in positions:
      subset = [r for r in rows if r["arm"] == arm and r["cell"] == cell and r["position"] == position]
      line += f"{median(subset, 'displacement'):>12.4f}" if len(subset) > 0 else f"{'-':>12}"
    print(line)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+", help="JSONs written by probe_cost_bias / probe_bias_cross")
  parser.add_argument("--task", required=True, help="label for the tables")
  parser.add_argument(
    "--knob", default=None,
    help="keep only rows whose `knob` field equals this (probe_cost_bias wrote one-knob-at-a-time cells)"
  )
  parser.add_argument(
    "--loss-precision", type=float, default=None,
    help="the `loss_precision` the measurements ran under; the clause replay needs it and the older "
    "JSONs do not carry it"
  )
  parser.add_argument("--warmup-epochs", type=int, default=2)
  parser.add_argument(
    "--pairs", action="store_true",
    help="rows come from `scripts/floor_matched_pairs.py`: report the within-pair difference between "
    "designs of the SAME floor (ranks 2k and 2k+1 are one pair)"
  )
  arguments = parser.parse_args()

  rows = []
  for path in arguments.inputs:
    with open(path) as f:
      rows.extend(json.load(f)["rows"])
  if arguments.knob is not None:
    rows = [r for r in rows if r.get("knob") == arguments.knob]
  rows = [r for r in rows if "per_epoch" in r]
  for row in rows:
    enrich(row)
    precision = arguments.loss_precision if arguments.loss_precision is not None else row.get("loss_precision")
    if precision is None:
      raise SystemExit("report_bias_cross: rows carry no loss_precision -- pass --loss-precision")
    clause_trace(row, arguments.warmup_epochs, float(precision))

  unfitted = sum(1 for r in rows if r.get("remaining_train") is None)
  statuses = collections.Counter(r["status"] for r in rows)
  has_floor = any(r.get("bayes_risk") is not None for r in rows)
  print(f"== {arguments.task}: {len(rows)} measurements from {len(arguments.inputs)} file(s)")
  print(f"   status {dict(statuses)} | closed-form floor {has_floor} | tails not fitted {unfitted}/{len(rows)}")
  print(f"   seeds {sorted({r['seed'] for r in rows})} | ranks {sorted({r['rank'] for r in rows})}")
  print(f"   patience {sorted({r['patience'] for r in rows})} | dropconnect {sorted({r['dropconnect'] for r in rows})}")
  failed = [r for r in rows if r["status"] != "converged"]
  if len(failed) > 0:
    print(f"   ⚠️ {len(failed)} cell(s) did NOT converge and are LEFT IN the tables, marked by status:")
    for row in failed:
      print(
        f"      {row['arm']} s{row['seed']} rank {row['rank']} dc {row['dropconnect']:g} "
        f"patience {row['patience']}: {row['status']}"
      )

  if all("cell" in r for r in rows):
    rewind_table(rows, f"{arguments.task}: restart cells")
    clause_table(rows, f"{arguments.task}: which clause bought the data")
    return
  cell_table(rows, has_floor, f"{arguments.task}: cells")
  if arguments.pairs:
    floor_pair_table(rows, f"{arguments.task}: floor-matched pairs")
  clause_table(rows, f"{arguments.task}: which clause bought the data")
  patience_profile(rows, has_floor, f"{arguments.task}: patience profile")
  for axis in ("loss_precision", "patience", "dropconnect"):
    if len({r.get(axis) for r in rows}) >= 1 and all(r.get(axis) is not None for r in rows):
      matched_offset(rows, f"{arguments.task}: matched-design offset by {axis}", axis)
  paired_arms(rows, has_floor, f"{arguments.task}: arm contrast")
  per_design_bias(rows, has_floor, f"{arguments.task}: per design")
  difficulty_correlation(rows, has_floor, f"{arguments.task}: design dependence")


if __name__ == "__main__":
  main()
