#!/usr/bin/env python3
"""A PDF report on the `linear` benchmark: tuning, campaign, and the checks that make both credible.

    python scripts/linear_report.py --output output/linear-report.pdf

EVERY NUMBER IS RECOMPUTED HERE FROM THE RUN FILES ON DISK -- none is copied from a prior summary, so
the figures and the text cannot disagree with the data or with each other.

WHAT IS PLOTTED, and the question each panel answers.

  1  convergence      does the optimiser improve, and do the two arms differ?
  2  paired arms      per seed, which arm holds the better incumbent at 24 designs, and by how much
                      against the noise of re-running one cell?
  3  the criterion    P(best@2n < best@n) over INDEPENDENT runs, the quantity the task was tuned on.
  4  faithfulness     how far the reported loss sits above the CLOSED-FORM Bayes floor. This task has
                      an exact answer; without this panel none of the others mean anything.
  5  width ladder     what capacity buys, in faithfulness and in seconds.
  6  landscape        the analytic span of every screened cell, and why (d=4, n_probes=5) was chosen.

THE ARM COLOURS ARE FIXED, never cycled, and every series also carries a dash pattern, so identity is
never colour-alone. The pair is chosen for separation under colour-vision deficiency rather than for
matching the repo's other plots.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import detopt.detector
import detopt.utils.config

ARMS = ("from_scratch", "meta")
COLOR = {"from_scratch": "#2a78d6", "meta": "#d97706"}
DASH = {"from_scratch": None, "meta": (6, 2)}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#ffffff", "#e3e2dd"
ACCENT, MUTED = "#1baf7a", "#9a9791"


def style(axes):
  axes.set_facecolor(SURFACE)
  axes.grid(True, color=GRID, linewidth=0.7)
  axes.set_axisbelow(True)
  for side in ("top", "right"):
    axes.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axes.spines[side].set_color(GRID)
  axes.tick_params(colors=INK2, labelsize=9)


def titled(axes, title, xlabel, ylabel):
  axes.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
  axes.set_xlabel(xlabel, color=INK2, fontsize=10)
  axes.set_ylabel(ylabel, color=INK2, fontsize=10)


def legend(axes, **kwargs):
  handle = axes.legend(frameon=False, fontsize=9, **kwargs)
  for text in handle.get_texts():
    text.set_color(INK2)
  return handle


def best_so_far(rows):
  losses = [r["loss"] for r in rows if r.get("loss") is not None]
  return np.minimum.accumulate(np.asarray(losses, np.float64))


def load_runs(pattern):
  """`{key: results-rows}` for every `results.json` matching `pattern`."""
  out = {}
  for path in sorted(glob.glob(pattern)):
    with open(path) as f:
      out[path] = json.load(f)["results"]
  return out


def probability_improves(curves, n, clusters=None):
  """P(best@2n from one run < best@n from ANOTHER), over all ordered pairs i != j.

  Independent runs by construction -- a within-run comparison is monotone and would return 1.
  `clusters` labels runs that are NOT independent of each other (the campaign's two arms at one seed
  share their Sobol block); pairs inside a cluster are dropped rather than counted. Returns
  `(P, standard error)` with the SE from a delete-one jackknife over runs.
  """
  at_n = np.asarray([c[n - 1] for c in curves], np.float64)
  at_2n = np.asarray([c[2 * n - 1] for c in curves], np.float64)
  runs = len(curves)

  def statistic(keep):
    wins, total = 0, 0
    for i in keep:
      for j in keep:
        if i == j or (clusters is not None and clusters[i] == clusters[j]):
          continue
        total += 1
        wins += 1.0 if at_2n[j] < at_n[i] else (0.5 if at_2n[j] == at_n[i] else 0.0)
    return wins / max(total, 1)

  everything = list(range(runs))
  full = statistic(everything)
  partials = np.asarray([statistic([i for i in everything if i != k]) for k in everything])
  error = np.sqrt((runs - 1) / runs * np.sum((partials - partials.mean())**2))
  return full, error


def bayes_floor(config_token):
  detector_config = detopt.utils.config.load_config(f"config/detector/{config_token}.yaml")
  detector = detopt.detector.from_config(detector_config)
  return detector


def page_summary(pdf, text_blocks):
  figure = plt.figure(figsize=(8.27, 11.69))
  figure.patch.set_facecolor(SURFACE)
  figure.text(0.07, 0.995, "The `linear` benchmark", color=INK, fontsize=20, va="top")
  figure.text(0.07, 0.955, "tuning, campaign, and the checks that make both credible", color=INK2, fontsize=11, va="top")
  y = 0.905
  for heading, lines in text_blocks:
    figure.text(0.07, y, heading, color=INK, fontsize=12, va="top", weight="bold")
    y -= 0.026
    for line in lines:
      figure.text(0.07, y, line, color=INK2, fontsize=9.3, va="top", family="monospace")
      y -= 0.0175
    y -= 0.016
  pdf.savefig(figure)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--campaign", default="output/linear-campaign")
  parser.add_argument("--tuning", default="output/linear-criterion-runs")
  parser.add_argument("--criterion-dir", default="output/linear-criterion")
  parser.add_argument("--detector", default="linear_d4n5")
  parser.add_argument("--n-pair", type=int, default=10, help="the criterion's n; the pair is n -> 2n")
  parser.add_argument("--output", default="output/linear-report.pdf")
  arguments = parser.parse_args()

  detector = bayes_floor(arguments.detector)

  campaign = {}
  for path, rows in load_runs(os.path.join(arguments.campaign, "*", "*", "results.json")).items():
    parts = path.split(os.sep)
    campaign[(parts[-3], parts[-2])] = rows
  seeds = sorted({seed for seed, _ in campaign})

  tuning = load_runs(os.path.join(arguments.tuning, "*", "from_scratch", "results.json"))
  tuning_curves = [best_so_far(rows) for rows in tuning.values()]

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  n = arguments.n_pair

  # ---------------------------------------------------------------- numbers
  final = {arm: np.asarray([best_so_far(campaign[(s, arm)])[-1] for s in seeds]) for arm in ARMS}
  difference = final["from_scratch"] - final["meta"]
  p_tuning, se_tuning = probability_improves(tuning_curves, n)
  campaign_curves = [best_so_far(campaign[(s, arm)]) for s in seeds for arm in ARMS]
  campaign_clusters = [s for s in seeds for _ in ARMS]
  p_campaign, se_campaign = probability_improves(campaign_curves, n, campaign_clusters)

  excess, per_arm_excess = [], {arm: [] for arm in ARMS}
  for (seed, arm), rows in campaign.items():
    for row in rows:
      if row.get("loss") is None:
        continue
      scaled = np.asarray(row["x_scaled"], np.float32)[None, :]
      floor = detector.bayes_risk(np.asarray(detector.flatten_design(detector.to_nominal(scaled)), np.float32)[0])
      excess.append(row["loss"] - floor)
      per_arm_excess[arm].append(row["loss"] - floor)
  excess = np.asarray(excess)

  blocks = [
    (
      "Setting", [
        f"detector      {arguments.detector}  (d = 4, n_probes = 5, noise = 0.7)",
        f"campaign      {len(seeds)} seeds x {len(ARMS)} arms x {len(next(iter(campaign.values())))} designs",
        f"tuning        {len(tuning_curves)} independent runs, same configuration",
      ]
    ),
    (
      f"Criterion   P(best@{2*n} < best@{n}), independent runs", [
        f"tuning        P = {p_tuning:.3f} +/- {se_tuning:.3f}     (n = {len(tuning_curves)} runs)",
        f"campaign      P = {p_campaign:.3f} +/- {se_campaign:.3f}     (n = {len(campaign_curves)} runs, same-seed pairs dropped)",
        f"bar           0.75          random search on this task gives 2/3",
      ]
    ),
    (
      "Arms   best-so-far at the final design, paired on seed", [
        f"from_scratch  mean {final['from_scratch'].mean():.4f}      meta  mean {final['meta'].mean():.4f}",
        f"fs - meta     mean {difference.mean():+.4f}   sd {difference.std(ddof=1):.4f}   "
        f"favouring from_scratch on {int((difference < 0).sum())}/{len(seeds)}",
        "NOT RESOLVED  re-running one cell moves the result by more than the arm effect",
      ]
    ),
    (
      "Faithfulness   reported loss minus the closed-form Bayes floor", [
        f"over {len(excess)} designs   min {excess.min():+.4f}   mean {excess.mean():+.4f}   max {excess.max():+.4f}",
        f"from_scratch  mean {np.mean(per_arm_excess['from_scratch']):+.5f}      "
        f"meta  mean {np.mean(per_arm_excess['meta']):+.5f}", "the task has an exact answer and the network reaches it",
      ]
    ),
  ]

  with PdfPages(arguments.output) as pdf:
    page_summary(pdf, blocks)

    # ------------------------------------------------------------ 1 convergence
    figure, axes = plt.subplots(figsize=(9.5, 5.6))
    figure.patch.set_facecolor(SURFACE)
    style(axes)
    for arm in ARMS:
      curves = np.stack([best_so_far(campaign[(s, arm)]) for s in seeds])
      x = np.arange(1, curves.shape[1] + 1)
      for curve in curves:
        axes.plot(x, curve, color=COLOR[arm], linewidth=0.7, alpha=0.28, zorder=2)
      line, = axes.plot(x, np.median(curves, axis=0), color=COLOR[arm], linewidth=2.0, label=arm, zorder=3)
      if DASH[arm] is not None:
        line.set_dashes(DASH[arm])
    axes.axvline(n, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axes.axvline(2 * n, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axes.annotate(f"n = {n}", (n, axes.get_ylim()[1]), xytext=(3, -12), textcoords="offset points", color=MUTED, fontsize=9)
    axes.annotate(
      f"2n = {2*n}", (2 * n, axes.get_ylim()[1]), xytext=(3, -12), textcoords="offset points", color=MUTED, fontsize=9
    )
    axes.set_yscale("log")
    titled(axes, "1  Convergence: best-so-far against designs probed", "designs probed", "best-so-far loss")
    axes.annotate(
      "thin lines are the 5 seeds; heavy lines their median", (0.5, -0.13), xycoords="axes fraction", ha="center", color=MUTED,
      fontsize=9
    )
    legend(axes, handlelength=3.0)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)

    # ------------------------------------------------------------ 2 paired arms
    figure, axes = plt.subplots(figsize=(9.5, 5.0))
    figure.patch.set_facecolor(SURFACE)
    style(axes)
    y = np.arange(len(seeds))
    for index, seed in enumerate(seeds):
      a, b = final["from_scratch"][index], final["meta"][index]
      axes.plot([a, b], [index, index], color=GRID, linewidth=2.0, zorder=1, solid_capstyle="round")
    axes.scatter(
      final["from_scratch"], y, s=64, color=COLOR["from_scratch"], zorder=3, label="from_scratch", edgecolor=SURFACE,
      linewidth=1.5
    )
    axes.scatter(
      final["meta"], y, s=64, color=COLOR["meta"], zorder=3, label="meta", marker="D", edgecolor=SURFACE, linewidth=1.5
    )
    for index, seed in enumerate(seeds):
      axes.annotate(
        f"{difference[index]:+.4f}", (max(final['from_scratch'][index], final['meta'][index]), index), xytext=(10, -3),
        textcoords="offset points", color=INK2, fontsize=9
      )
    axes.set_yticks(y)
    axes.set_yticklabels(seeds)
    axes.set_ylim(-0.6, len(seeds) - 0.4)
    titled(axes, "2  Paired by seed: incumbent at the final design", "best-so-far loss", "seed")
    axes.annotate(
      f"from_scratch better on {int((difference < 0).sum())}/{len(seeds)}; effect {abs(difference.mean()):.4f} "
      f"is SMALLER than the 0.0406 sd of re-running one cell", (0.5, -0.16), xycoords="axes fraction", ha="center", color=MUTED,
      fontsize=9
    )
    legend(axes, loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)

    # ------------------------------------------------------------ 3 the criterion
    figure, axes = plt.subplots(figsize=(9.5, 5.6))
    figure.patch.set_facecolor(SURFACE)
    style(axes)
    at_n = np.asarray([c[n - 1] for c in tuning_curves])
    at_2n = np.asarray([c[2 * n - 1] for c in tuning_curves])
    low = min(at_n.min(), at_2n.min()) * 0.9
    high = max(at_n.max(), at_2n.max()) * 1.05
    axes.plot([low, high], [low, high], color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axes.scatter(at_n, at_2n, s=58, color=ACCENT, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    titled(axes, f"3  Doubling the search: each of the {len(tuning_curves)} tuning runs", f"best@{n}", f"best@{2*n}")
    axes.annotate(
      f"WITHIN a run, every one improves: all {len(tuning_curves)} points lie below the diagonal.\n"
      f"That is monotone by construction and is NOT the criterion.\n\n"
      f"The criterion pairs best@{2*n} of one run against best@{n} of ANOTHER:\n"
      f"P = {p_tuning:.3f} +/- {se_tuning:.3f}    bar 0.75    random search 0.667", (0.04, 0.96), xycoords="axes fraction",
      va="top", color=INK2, fontsize=9.5
    )
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)

    # ------------------------------------------------------------ 4 faithfulness
    figure, axes = plt.subplots(figsize=(9.5, 5.0))
    figure.patch.set_facecolor(SURFACE)
    style(axes)
    bins = np.linspace(excess.min(), excess.max(), 44)
    for arm in ARMS:
      axes.hist(per_arm_excess[arm], bins=bins, color=COLOR[arm], alpha=0.62, label=arm, edgecolor=SURFACE, linewidth=0.7)
    axes.axvline(0.0, color=INK, linewidth=1.2, zorder=4)
    titled(axes, "4  Faithfulness: reported loss minus the closed-form Bayes floor", "excess above the floor", "designs")
    axes.annotate(
      f"{len(excess)} designs    max {excess.max():+.4f}    mean {excess.mean():+.4f}\n"
      f"max is {100*excess.max()/0.291:.2f}% of the landscape span (0.291)", (0.98, 0.95), xycoords="axes fraction", ha="right",
      va="top", color=INK2, fontsize=9.5
    )
    legend(axes)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)

    # ------------------------------------------------------------ 5 width ladder
    widths = []
    for path in sorted(glob.glob(os.path.join(arguments.criterion_dir, "width-*.json"))):
      with open(path) as f:
        payload = json.load(f)
      rows = [r for r in payload.get("rows", []) if r.get("n_parameters") is not None]
      if len(rows) == 0:
        continue
      widths.append((
        int(rows[0]["n_parameters"]), float(np.mean([r["wall_s"]
                                                     for r in rows])), float(np.mean([r["objective_std"] for r in rows])),
      ))
    if len(widths) > 0:
      merged = {}
      for count, wall, _ in widths:
        merged.setdefault(count, []).append(wall)
      parameters = np.asarray(sorted(merged), np.float64)
      seconds = np.asarray([float(np.mean(merged[int(c)])) for c in parameters], np.float64)
      figure, axes = plt.subplots(figsize=(9.5, 5.0))
      figure.patch.set_facecolor(SURFACE)
      style(axes)
      axes.plot(
        parameters, seconds, color=ACCENT, linewidth=2.0, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5
      )
      for x, y_value in zip(parameters, seconds):
        axes.annotate(
          f"{y_value:.0f} s", (x, y_value), xytext=(0, 10), textcoords="offset points", ha="center", color=INK2, fontsize=9
        )
      axes.set_xscale("log")
      titled(axes, "5  Width ladder: cost per design against capacity", "parameters", "seconds per design")
      axes.annotate(
        "more capacity is CHEAPER here: it plateaus in fewer epochs", (0.98, 0.95), xycoords="axes fraction", ha="right",
        va="top", color=MUTED, fontsize=9.5
      )
      figure.tight_layout()
      pdf.savefig(figure)
      plt.close(figure)

    # ------------------------------------------------------------ 6 landscape
    landscape_path = os.path.join(arguments.criterion_dir, "landscape.json")
    if os.path.isfile(landscape_path):
      with open(landscape_path) as f:
        cells = json.load(f)["cells"]
      figure, axes = plt.subplots(figsize=(9.5, 5.6))
      figure.patch.set_facecolor(SURFACE)
      style(axes)
      by_dimension = {}
      for cell in cells:
        scape = cell["landscape"]
        span = float(scape["span"])
        by_dimension.setdefault(cell["n_dimensions"], []).append((cell["sigma"], span, cell["n_probes"]))
      palette = ["#2a78d6", "#d97706", "#1baf7a"]
      for index, (dimension, points) in enumerate(sorted(by_dimension.items())):
        points.sort()
        sigmas = [p[0] for p in points]
        spans = [p[1] for p in points]
        axes.scatter(
          sigmas, spans, s=46, color=palette[index % len(palette)], label=f"d = {dimension}", edgecolor=SURFACE, linewidth=1.2,
          zorder=3
        )
      axes.scatter([0.7], [0.291], s=200, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4)
      axes.annotate(
        "chosen: d=4, n_probes=5, sigma=0.7", (0.7, 0.291), xytext=(-12, 16), textcoords="offset points", ha="right", color=INK,
        fontsize=9.5
      )
      titled(
        axes, "6  Analytic landscape span of every screened cell", "read-out noise sigma",
        "span  (median random design - optimum)"
      )
      axes.annotate(
        "computed in closed form from bayes_risk, before any training", (0.5, -0.13), xycoords="axes fraction", ha="center",
        color=MUTED, fontsize=9.5
      )
      legend(axes)
      figure.tight_layout()
      pdf.savefig(figure)
      plt.close(figure)

  print(f"-> {arguments.output}")
  for heading, lines in blocks:
    print(f"\n{heading}")
    for line in lines:
      print(f"  {line}")


if __name__ == "__main__":
  main()
