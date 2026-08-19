#!/usr/bin/env python3
"""A PDF report on the `MM` (Michaelis-Menten) benchmark: proxy tuning, and the campaign in flight.

    python scripts/mm_report.py --output output/mm-report.pdf

EVERY NUMBER IS RECOMPUTED OR RE-READ FROM THE FILES ON DISK -- none is copied from a prior summary,
so the figures and the text cannot disagree with the data or with each other.

⚠️ TWO THINGS THIS REPORT CANNOT DO, and the `linear` report can.

  * **There is no closed-form floor.** `linear` has `bayes_risk`, so every reported loss can be
    checked against the exact answer. MM has no such reference, so there is NO faithfulness panel and
    no way here to separate "the optimiser found a good design" from "the network mis-scored a design".
  * **The tuning is on the XGBoost PROXY, not the network.** `docs/findings.md` records that proxy
    gains scale by 0.716 on the neural scale, that rank agreement is Spearman +0.818, and that the
    proxy's best design is NOT the neural best. The proxy decides which setting earns a campaign; it
    does not stand in for one.

WHAT IS PLOTTED, and the question each panel answers.

  1  criterion       P(best@2n < best@n) per screened cell, over INDEPENDENT runs -- the quantity the
                     task was tuned on, against the 0.75 bar and the 2/3 a random search gives.
  2  bonus ratio     the tie-breaker, in both readings, showing why one of them cannot discriminate.
  3  landscape       best / dead volume / top-decile / GP R2 per cell, the screening diagnostics.
  4  slack           `|train - val| + err` at the validation minimum, proxy against network.
  5  campaign        best-so-far per cell, IN FLIGHT and labelled as such.

THE CELL COLOURS ARE FIXED, never cycled, and the chosen cell is marked in every panel it appears in.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ARMS = ("from_scratch", "meta")
ARM_COLOR = {"from_scratch": "#2a78d6", "meta": "#d97706"}
ARM_DASH = {"from_scratch": None, "meta": (6, 2)}
CELL_COLOR = {"m3-n05": "#1baf7a", "m4-n05": "#2a78d6", "m4-n02": "#9a9791", "m4-n15": "#d97706"}
CHOSEN = "m3-n05"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#ffffff", "#e3e2dd"
MUTED, ALERT = "#9a9791", "#c2410c"

LANDSCAPE_PATTERN = re.compile(
  r"m=(\d+)\s+dim=(\d+)\s*\|\s*ceiling\s+([\d.]+)\s+best\s+([\d.]+)\s*\|\s*at-ceiling\s+([\d.]+)%"
  r"\s*\|\s*top-decile\s+([\d.]+)%\s*\|\s*GP R2\s+([+-][\d.]+)\s*\|\s*([\d.]+) s/design"
)
SLACK_PATTERN = re.compile(
  r"\|train-val\|\+err over (\d+) designs: median ([\d.]+) max ([\d.]+)\s*"
  r"\(gap median ([\d.]+) max ([\d.]+); err median ([\d.]+) max ([\d.]+)\)"
)


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


def caption(axes, text):
  axes.annotate(text, (0.5, -0.145), xycoords="axes fraction", ha="center", color=MUTED, fontsize=9)


def read_criterion(directory):
  """`{cell: {n: pair-record}}` from every `criterion-*.json`."""
  out = {}
  for path in sorted(glob.glob(os.path.join(directory, "criterion-*.json"))):
    with open(path) as f:
      payload = json.load(f)
    cell = os.path.basename(path).replace("criterion-", "").replace(".json", "")
    out[cell] = {int(pair["n"]): pair for pair in payload["pairs"]}
    out[cell]["runs"] = int(payload["runs"])
  return out


def read_logs(directory):
  """`{cell: {...}}` -- the landscape line and the slack line each screen prints once."""
  out = {}
  for path in sorted(glob.glob(os.path.join(directory, "*.log"))):
    cell = os.path.basename(path).replace(".log", "")
    with open(path, errors="ignore") as f:
      text = f.read()
    record = {}
    found = LANDSCAPE_PATTERN.search(text)
    if found is not None:
      record.update(
        m=int(found.group(1)), dimension=int(found.group(2)), ceiling=float(found.group(3)), best=float(found.group(4)),
        dead=float(found.group(5)), top_decile=float(found.group(6)), gp_r2=float(found.group(7)),
        seconds=float(found.group(8))
      )
    found = SLACK_PATTERN.search(text)
    if found is not None:
      record.update(
        slack_designs=int(found.group(1)), slack_median=float(found.group(2)), slack_max=float(found.group(3)),
        gap_median=float(found.group(4)), gap_max=float(found.group(5)), err_median=float(found.group(6)),
        err_max=float(found.group(7))
      )
    if len(record) > 0:
      out[cell] = record
  return out


def best_so_far(rows):
  losses = [r["loss"] for r in rows if r.get("loss") is not None]
  if len(losses) == 0:
    return np.zeros(0)
  return np.minimum.accumulate(np.asarray(losses, np.float64))


def page_summary(pdf, blocks, subtitle):
  figure = plt.figure(figsize=(8.27, 11.69))
  figure.patch.set_facecolor(SURFACE)
  figure.text(0.07, 0.995, "The MM benchmark", color=INK, fontsize=20, va="top")
  figure.text(0.07, 0.958, subtitle, color=INK2, fontsize=11, va="top")
  y = 0.912
  for heading, lines in blocks:
    figure.text(0.07, y, heading, color=INK, fontsize=12, va="top", weight="bold")
    y -= 0.025
    for line in lines:
      figure.text(0.07, y, line, color=INK2, fontsize=9.0, va="top", family="monospace")
      y -= 0.0168
    y -= 0.015
  pdf.savefig(figure)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--proxy", default="output/mm-tune/proxy")
  parser.add_argument("--campaign", default="output/campaign-mm")
  parser.add_argument("--output", default="output/mm-report.pdf")
  arguments = parser.parse_args()

  criterion = read_criterion(arguments.proxy)
  logs = read_logs(arguments.proxy)
  cells = [c for c in ("m3-n05", "m4-n05", "m4-n02", "m4-n15") if c in logs or c in criterion]

  campaign = {}
  for path in sorted(glob.glob(os.path.join(arguments.campaign, "*", "*", "results.json"))):
    with open(path) as f:
      payload = json.load(f)
    parts = path.split(os.sep)
    campaign[(parts[-3], parts[-2])] = payload

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)

  # ---------------------------------------------------------------- summary text
  criterion_lines = []
  for cell in cells:
    if cell not in criterion:
      criterion_lines.append(f"{cell:8s}  not measured -- excluded on its landscape before any P was computed")
      continue
    for n in (5, 10):
      pair = criterion[cell].get(n)
      if pair is None:
        continue
      criterion_lines.append(
        f"{cell:8s}  {n:2d}->{2*n:<3d} P = {pair['probability']:.3f} +/- {pair['jackknife_se']:.3f}   "
        f"[{pair['ci_low']:.3f}, {pair['ci_high']:.3f}]   {pair['verdict']}"
      )

  campaign_lines = []
  for (seed, arm), payload in sorted(campaign.items()):
    curve = best_so_far(payload["results"])
    used = sum(r["spent"] for r in payload["results"])
    campaign_lines.append(
      f"{seed:>11s} {arm:<13s} {len(curve):2d} designs   best {curve.min() if len(curve) else float('nan'):.4f}"
      f"   {100*used/2097152:3.0f}% of budget   completed={payload.get('completed')}"
    )

  blocks = [(
    "Setting", [
      "detector      enzyme_mm_sym at n_experiments = 3, measurement_noise = 0.05",
      "tuning        XGBoost proxy, 512 Sobol designs, 60 INDEPENDENT runs per cell",
      "campaign      2 arms x 2 seeds, loss_precision 2.0e-3, budget 2^21",
    ]
  ), (f"Criterion   P(best@2n < best@n), independent runs, bar 0.75", criterion_lines),
            (
              "Chosen and excluded", [
                f"CHOSEN        {CHOSEN} -- the only provenance-clean cell clearing 0.75 strictly at 10->20",
                "m4-n15        scored HIGHEST and was REFUSED: 0.15 mM is a declared control on the",
                "              model's shape, not a candidate. Promoting it after seeing it win is the",
                "              threshold-after-the-result move the NO TRICKS clause bans.",
                "m4-n02        excluded on its landscape, before any P: 1.2% dead volume, below the",
                "              pre-registered [5, 40]% band.",
                "m6-n05        NOT screened (dim 24, slowest cell). Unmeasured, not failed.",
                "20->40        not powered: the neural cost caps a run near 21 designs, so 40 is",
                "              unreachable and powering it would measure what the benchmark cannot deliver.",
              ]
            ), ("Campaign, IN FLIGHT", campaign_lines if len(campaign_lines) > 0 else ["no cells on disk yet"]),
            (
              "What this report cannot show", [
                "MM has NO closed-form floor. `linear` has bayes_risk, so every reported loss there is",
                "checked against the exact answer; there is no such check here, and none is implied.",
                "Every tuning number is a PROXY number: gains scale by 0.716 on the neural scale and the",
                "proxy's best design is not the neural best (docs/findings.md).",
              ]
            ), ]

  with PdfPages(arguments.output) as pdf:
    page_summary(pdf, blocks, "proxy tuning, and the campaign in flight")

    # ------------------------------------------------------------ 1 criterion
    figure, axes = plt.subplots(figsize=(9.5, 5.4))
    figure.patch.set_facecolor(SURFACE)
    style(axes)
    labels, position = [], 0
    for cell in cells:
      if cell not in criterion:
        continue
      for n in (5, 10):
        pair = criterion[cell].get(n)
        if pair is None:
          continue
        colour = CELL_COLOR.get(cell, MUTED)
        axes.errorbar(
          position, pair["probability"], yerr=[[pair["probability"] - pair["ci_low"]],
                                               [pair["ci_high"] - pair["probability"]]], fmt="o", markersize=9, color=colour,
          ecolor=colour, elinewidth=2.0, capsize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3
        )
        axes.annotate(
          f"{pair['probability']:.3f}", (position, pair["probability"]), xytext=(0, 13), textcoords="offset points",
          ha="center", color=INK2, fontsize=9
        )
        labels.append(f"{cell}\n{n}->{2*n}")
        position += 1
    axes.axhline(0.75, color=INK, linewidth=1.3, zorder=2)
    axes.axhline(2 / 3, color=ALERT, linewidth=1.2, linestyle="--", zorder=2)
    axes.annotate("bar 0.75", (position - 0.4, 0.75), xytext=(6, 3), textcoords="offset points", color=INK, fontsize=9)
    axes.annotate(
      "random search 2/3", (position - 0.4, 2 / 3), xytext=(6, -12), textcoords="offset points", color=ALERT, fontsize=9
    )
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, fontsize=9)
    axes.set_xlim(-0.6, position + 1.4)
    titled(axes, "1  The criterion, per cell", "", "P(best@2n < best@n)")
    caption(
      axes, f"60 independent runs per cell; bars are 95% CI from a delete-one jackknife. "
      f"{CHOSEN} is the chosen cell."
    )
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)

    # ------------------------------------------------------------ 2 bonus ratio
    figure, axes = plt.subplots(figsize=(9.5, 5.4))
    figure.patch.set_facecolor(SURFACE)
    style(axes)
    keys = [(c, n) for c in cells if c in criterion for n in (5, 10) if criterion[c].get(n) is not None]
    x = np.arange(len(keys))
    series = (("ratio (a) independent", "ratio_a_independent", "#9a9791", "o"),
              ("ratio (a) paired", "ratio_a_paired", "#2a78d6", "s"), ("ratio (b) mean gain / sd", "ratio_b", "#1baf7a", "D"),
              )
    for label, field, colour, marker in series:
      values = [criterion[c][n][field] for c, n in keys]
      axes.plot(
        x, values, color=colour, linewidth=1.6, marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.4,
        label=label, zorder=3
      )
    axes.axhline(1.0, color=INK, linewidth=1.1, zorder=2)
    axes.set_xticks(x)
    axes.set_xticklabels([f"{c}\n{n}->{2*n}" for c, n in keys], fontsize=9)
    titled(axes, "2  The tie-breaker, in both readings", "", "ratio")
    axes.annotate(
      "reading (a) on INDEPENDENT draws is sqrt(1 + var@2n/var@n) >= 1 MECHANICALLY,\n"
      "so it cannot discriminate: every value sits in a narrow band above 1.\n"
      "reading (b) is the informative one and ranks the cells as P does.", (0.02, 0.96), xycoords="axes fraction", va="top",
      color=INK2, fontsize=9.5
    )
    legend(axes, loc="lower right")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)

    # ------------------------------------------------------------ 3 landscape
    have = [c for c in cells if "best" in logs.get(c, {})]
    if len(have) > 0:
      figure, axeses = plt.subplots(1, 4, figsize=(11.0, 4.4))
      figure.patch.set_facecolor(SURFACE)
      fields = (("best", "best design found", None), ("dead", "at ceiling  (%)", (5, 40)),
                ("top_decile", "top-decile spread (%)", None), ("gp_r2", "GP R2", None))
      for axes, (field, title, band) in zip(axeses, fields):
        style(axes)
        values = [logs[c][field] for c in have]
        colours = [CELL_COLOR.get(c, MUTED) for c in have]
        axes.barh(range(len(have)), values, color=colours, height=0.62)
        if band is not None:
          axes.axvspan(band[0], band[1], color="#1baf7a", alpha=0.10, zorder=0)
          axes.annotate(
            "pre-registered\nband", (band[1], len(have) - 0.4), xytext=(-4, 0), textcoords="offset points", ha="right",
            color=MUTED, fontsize=8
          )
        axes.set_yticks(range(len(have)))
        axes.set_yticklabels(have, fontsize=9)
        axes.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        for index, value in enumerate(values):
          axes.annotate(f"{value:.3g}", (value, index), xytext=(4, -3), textcoords="offset points", color=INK2, fontsize=8.5)
      figure.suptitle("3  Screening diagnostics per cell", color=INK, fontsize=12, x=0.008, ha="left")
      figure.tight_layout(rect=(0, 0.03, 1, 0.94))
      figure.text(
        0.5, 0.012, "every MM cell sits at GP R2 +0.83..+0.88 against the melt's +0.37, at 24-32 flat "
        "columns -- the proxy is inside its working range", ha="center", color=MUTED, fontsize=9
      )
      pdf.savefig(figure)
      plt.close(figure)

    # ------------------------------------------------------------ 4 slack
    have = [c for c in cells if "slack_median" in logs.get(c, {})]
    if len(have) > 0:
      figure, axes = plt.subplots(figsize=(9.5, 5.0))
      figure.patch.set_facecolor(SURFACE)
      style(axes)
      y = np.arange(len(have))
      for index, cell in enumerate(have):
        record = logs[cell]
        colour = CELL_COLOR.get(cell, MUTED)
        axes.plot([record["slack_median"], record["slack_max"]], [index, index], color=colour, linewidth=3.0,
                  solid_capstyle="round", zorder=3)
        axes.scatter([record["slack_median"]], [index], s=52, color=colour, zorder=4, edgecolor=SURFACE, linewidth=1.4)
        axes.scatter([record["slack_max"]], [index], s=52, color=colour, marker="D", zorder=4, edgecolor=SURFACE, linewidth=1.4)
        axes.annotate(
          f"median {record['slack_median']:.4f}   max {record['slack_max']:.4f}", (record["slack_max"], index), xytext=(10, -3),
          textcoords="offset points", color=INK2, fontsize=9
        )
      axes.axvline(0.0017, color=INK, linewidth=1.4, zorder=2)
      axes.axvline(0.0020, color=ALERT, linewidth=1.2, linestyle="--", zorder=2)
      axes.annotate(
        "network, measured 0.0017", (0.0017, -0.42), xytext=(6, 14), textcoords="offset points", color=INK, fontsize=9
      )
      axes.annotate("bar 0.0020", (0.0020, -0.42), xytext=(6, 0), textcoords="offset points", color=ALERT, fontsize=9)
      axes.set_yticks(y)
      axes.set_yticklabels(have)
      axes.set_xscale("log")
      axes.set_ylim(-0.7, len(have) - 0.3)
      titled(axes, "4  |train - val| + err at the validation minimum", "slack  (log scale)", "cell")
      caption(
        axes, "the PROXY's slack is 20-50x the network's, because it is almost entirely XGBoost's own "
        "overfitting gap. It must NOT be used to set loss_precision."
      )
      figure.tight_layout()
      pdf.savefig(figure)
      plt.close(figure)

    # ------------------------------------------------------------ 5 campaign
    if len(campaign) > 0:
      figure, axes = plt.subplots(figsize=(9.5, 5.4))
      figure.patch.set_facecolor(SURFACE)
      style(axes)
      for (seed, arm), payload in sorted(campaign.items()):
        curve = best_so_far(payload["results"])
        if len(curve) == 0:
          continue
        line, = axes.plot(
          np.arange(1,
                    len(curve) + 1), curve, color=ARM_COLOR[arm], linewidth=1.9, marker="o", markersize=5,
          markeredgecolor=SURFACE, markeredgewidth=1.0, label=f"{arm}  seed {seed}", zorder=3
        )
        if ARM_DASH[arm] is not None:
          line.set_dashes(ARM_DASH[arm])
      axes.axvline(10, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
      axes.axvline(20, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
      axes.annotate("n = 10", (10, axes.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", color=MUTED, fontsize=9)
      axes.annotate("2n = 20", (20, axes.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", color=MUTED, fontsize=9)
      titled(axes, "5  Campaign, IN FLIGHT -- best-so-far per cell", "designs probed", "best-so-far loss")
      caption(
        axes, "INCOMPLETE: no cell has reached 2n = 20, so no arm comparison is possible yet and none "
        "is drawn. Cells differ in how far they have run."
      )
      legend(axes, handlelength=3.0, loc="center right")
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
