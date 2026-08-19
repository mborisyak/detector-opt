#!/usr/bin/env python3
"""The doubling criterion in its BETWEEN-RUN form, pooled over `scripts/screen_task.py` reports.

    python scripts/criterion_between_runs.py output/mm-tune/proxy/m4-n05*.json

WHY THIS EXISTS RATHER THAN `screen_task.py`'s OWN VERDICT. That verdict reads `best@n` and `best@2n`
off ONE trajectory, so its condition (b) -- "the doubling improved the curve" -- is true whenever the
run improved at all and can never be false in a way that carries information about the task:
best-so-far is monotone non-increasing, so within a run P(best@2n <= best@n) = 1 by construction.
The quantity asked for here is

    P(best-so-far at 2n designs  <  best-so-far at n designs), THE TWO CURVES FROM DIFFERENT RUNS,

which is the probability that a run given twice the search beats an INDEPENDENT run given half of it.
It is 0.5 exactly when doubling buys nothing, so it measures what a saturated task fails to do. It is
the two-sample AUC (Mann-Whitney) of `best@2n` against `best@n`.

HOW IT IS ESTIMATED. Over all R(R-1) ORDERED pairs of distinct runs, with ties at 0.5. Excluding the
diagonal is what makes it between-run; a U-statistic over all off-diagonal pairs uses the sample far
better than R/2 disjoint pairs would. Its standard error is therefore NOT sqrt(P(1-P)/R): the pairs
share runs. The delete-one JACKKNIFE over RUNS is used instead, which is the honest unit -- one run
is one draw -- and is reported next to the naive binomial figure so the difference is visible.

THE TIE-BREAKER RATIOS, both readings, since the phrase "spread (loss@n - loss@2n) / spread of
loss@n" admits two:

    (a) sd(best@n - best@2n) / sd(best@n)   -- the spread of the difference, over the spread at n
    (b) mean(best@n - best@2n) / sd(best@n) -- the mean improvement in units of the spread at n

Both are computed for INDEPENDENT draws, as specified. Note what that does to (a): for independent
draws var(a - b) = var(a) + var(b), so reading (a) is sqrt(1 + var(b)/var(a)) and is >= 1 MECHANICALLY,
whatever the task does. It rewards a task whose spread at 2n is LARGE relative to n, which is the
opposite of what a tie-breaker on signal-to-noise wants. The PAIRED reading of the same expression --
the two values from the same run -- is reported beside it, because that is the only reading under
which (a) is informative, and the reader should see both rather than one chosen here.
"""
import argparse
import glob
import json

import numpy as np


def load_curves(paths, minimum_length):
  """Every stored best-so-far curve at least `minimum_length` long, pooled across reports.

  Runs are keyed by (label, seed) so re-reading a report twice, or overlapping `--seed-offset`
  blocks, cannot inflate the sample with duplicates -- the whole statistic is a count over
  independent runs, and a silently doubled run would shrink the error bar on nothing.
  """
  curves, seen, short, labels = [], set(), 0, set()
  for path in sorted({p for pattern in paths for p in glob.glob(pattern)}):
    with open(path) as handle:
      report = json.load(handle)
    test = report.get("iteration_test")
    if test is None:
      continue
    settings = report.get("settings", {})
    offset = int(settings.get("seed_offset", 0))
    labels.add(report.get("label", "?"))
    for index, curve in enumerate(test["curves"]):
      key = (report.get("label", "?"), offset + index)
      if key in seen:
        continue
      seen.add(key)
      if len(curve) < minimum_length:
        short += 1
        continue
      curves.append([float(v) for v in curve])
  return np.array(curves, dtype=float), short, sorted(labels)


def between_run_probability(at_n, at_2n):
  """AUC over ordered pairs of DISTINCT runs, ties at 0.5, with its delete-one jackknife error."""
  comparison = (at_2n[:, None] < at_n[None, :]).astype(float) + 0.5 * (at_2n[:, None] == at_n[None, :])
  np.fill_diagonal(comparison, np.nan)
  estimate = float(np.nanmean(comparison))
  runs = len(at_n)
  partials = np.empty(runs)
  for k in range(runs):
    keep = np.setdiff1d(np.arange(runs), [k])
    partials[k] = float(np.nanmean(comparison[np.ix_(keep, keep)]))
  jackknife = float(np.sqrt((runs - 1) / runs * np.sum(np.square(partials - partials.mean()))))
  return estimate, jackknife


def report_pair(curves, n, strict, soft):
  """One doubling pair, printed. Returns the row so a caller can rank settings on it."""
  at_n, at_2n = curves[:, n - 1], curves[:, 2 * n - 1]
  runs = len(curves)
  probability, jackknife = between_run_probability(at_n, at_2n)
  binomial = float(np.sqrt(probability * (1.0 - probability) / runs))
  spread_n, spread_2n = float(at_n.std(ddof=1)), float(at_2n.std(ddof=1))
  mean_gain = float(at_n.mean() - at_2n.mean())
  independent_difference_spread = float(np.sqrt(at_n.var(ddof=1) + at_2n.var(ddof=1)))
  ratio_a = independent_difference_spread / spread_n if spread_n > 0 else float("nan")
  ratio_b = mean_gain / spread_n if spread_n > 0 else float("nan")
  paired_a = float((at_n - at_2n).std(ddof=1)) / spread_n if spread_n > 0 else float("nan")
  low, high = probability - 1.96 * jackknife, probability + 1.96 * jackknife
  if low > strict:
    verdict = "PASS-STRICT"
  elif probability > strict:
    verdict = "above 0.75 but CI straddles it"
  elif low > soft:
    verdict = "PASS-SOFT"
  elif probability > soft:
    verdict = "above 0.50 but CI straddles it"
  else:
    verdict = "FAIL"
  print(
    f"  {n:2d} -> {2 * n:2d}  P = {probability:.3f} +- {jackknife:.3f} (jackknife over {runs} runs; "
    f"binomial {binomial:.3f})  95% CI [{low:.3f}, {high:.3f}]  {verdict}"
  )
  print(
    f"          best@{n}  mean {at_n.mean():.4f} sd {spread_n:.4f}   "
    f"best@{2 * n} mean {at_2n.mean():.4f} sd {spread_2n:.4f}   mean gain {mean_gain:.4f}"
  )
  print(
    f"          ratio (a) independent sd(diff)/sd@n = {ratio_a:.3f}  [paired reading {paired_a:.3f}]   "
    f"ratio (b) mean gain / sd@n = {ratio_b:.3f}"
  )
  return {
    "n": n,
    "runs": runs,
    "probability": probability,
    "jackknife_se": jackknife,
    "binomial_se": binomial,
    "ci_low": low,
    "ci_high": high,
    "verdict": verdict,
    "mean_at_n": float(at_n.mean()),
    "sd_at_n": spread_n,
    "mean_at_2n": float(at_2n.mean()),
    "sd_at_2n": spread_2n,
    "mean_gain": mean_gain,
    "ratio_a_independent": ratio_a,
    "ratio_a_paired": paired_a,
    "ratio_b": ratio_b
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("reports", nargs="+", help="screen_task.py JSONs (globs allowed) to POOL")
  parser.add_argument(
    "--pairs", type=int, nargs="*", default=[5, 10, 20], help="the n of each doubling; 5 <= n <= 20, so 5->10, 10->20, 20->40"
  )
  parser.add_argument("--strict", type=float, default=0.75)
  parser.add_argument("--soft", type=float, default=0.50)
  parser.add_argument("--output", default=None)
  arguments = parser.parse_args()

  longest = 2 * max(arguments.pairs)
  curves, short, labels = load_curves(arguments.reports, longest)
  print(
    f"pooled {len(curves)} runs of >= {longest} designs from labels {labels}"
    f"{f' ({short} shorter runs dropped)' if short > 0 else ''}"
  )
  if len(curves) < 4:
    raise SystemExit("criterion_between_runs: fewer than 4 usable runs -- nothing to estimate")
  rows = [report_pair(curves, n, arguments.strict, arguments.soft) for n in arguments.pairs if 2 * n <= curves.shape[1]]
  if arguments.output is not None:
    with open(arguments.output, "w") as handle:
      json.dump({"labels": labels, "runs": len(curves), "pairs": rows}, handle, indent=2)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
  main()
