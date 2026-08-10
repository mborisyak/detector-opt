#!/usr/bin/env python3
"""The figure behind the box decision: does an OPTIMISER's iterations pay off more than a shotgun's?

Three panels, all in DEGREES CELSIUS of target resolution (sqrt of the normalised loss times the
T_melting prior half-range) so the vertical axis is a physical quantity and comparable across panels:

1-2. Best-so-far against iteration, median over seeds, BO against its own random-search null, for the
     two admissible boxes at each read-out noise. Colour carries the BOX and line style the METHOD,
     so neither is identified by colour alone.
3.   Every seed's own 60 -> 120 improvement, as a slope. This is the panel the box decision actually
     turns on: the wide box doubles the median gain but scatters the endpoints, and a campaign has to
     resolve the former against the latter.

    python scripts/plot_iteration_payoff.py --output output/iteration_payoff.png
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("AGG")

import numpy as np
from matplotlib.figure import Figure

# The repo's chart palette (detopt/utils/viz/bo.py). Blue and orange are the two most reliably
# separable hues under every common colour-vision deficiency; validated for lightness band, chroma,
# CVD separation (worst adjacent dE 24.7 protan) and contrast against the surface.
NARROW, WIDE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2933", "#52606d", "#c9d1d9"

ARMS = {
  "narrow_5": ("output/iters120/pi-bo-*.json", "output/iters120/null-random-*.json"),
  "wide_5": ("output/wide/w5-bo-*.json", "output/wide/w5-random-*.json"),
  "narrow_2": ("output/noise_lo/lo-bo-*.json", "output/noise_lo/lo-random-*.json"),
  "wide_2": ("output/wide/w2-bo-*.json", "output/wide/w2-random-*.json")
}


def _settings(path):
  """The settings block a run recorded, so every label on this figure comes from the data."""
  return json.load(open(path))["settings"]


def curves(pattern):
  """Per-seed best-so-far in C, as a (seeds, iterations) array."""
  runs, celsius = [], None
  for path in sorted(glob.glob(pattern)):
    payload = json.load(open(path))
    celsius = 0.5 * (payload["settings"]["melting_bounds"][1] - payload["settings"]["melting_bounds"][0])
    runs.append(np.minimum.accumulate([r["loss"] for r in payload["results"]]))
  if len(runs) == 0:
    raise SystemExit(f"no runs matching {pattern}")
  length = min(len(r) for r in runs)
  return np.sqrt(np.stack([r[:length] for r in runs])) * celsius


def style(axis):
  axis.grid(True, alpha=0.35, lw=0.6, color=GRID)
  axis.set_axisbelow(True)
  for side in ("top", "right"):
    axis.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axis.spines[side].set_color(GRID)
  axis.tick_params(colors=MUTED, labelsize=9, length=3)


def convergence(axis, arms, title):
  finals = []
  for (name, colour, label) in arms:
    bo, null = (curves(p) for p in ARMS[name])
    for data, dash, method in ((bo, "-", "BO"), (null, (0, (5, 3)), "random")):
      median = np.median(data, axis=0)
      iterations = np.arange(1, median.size + 1)
      axis.plot(
        iterations, median, lw=2.0, color=colour, linestyle=dash,
        marker="o" if dash == "-" else "s", ms=5, markevery=25, markeredgecolor="white",
        markeredgewidth=1.2, label=f"{label} — {method}", zorder=3 if dash == "-" else 2
      )
      axis.fill_between(iterations, data.min(axis=0), data.max(axis=0), color=colour, alpha=0.10, lw=0)
    finals.append((float(np.median(bo, axis=0)[-1]), bo.shape[1], colour))
  # Direct labels on the BO lines, so identity never rests on colour alone. The two ends sit within
  # ~0.04 C of each other, so they are pushed apart vertically rather than overprinted.
  for rank, (value, length, colour) in enumerate(sorted(finals, key=lambda f: f[0])):
    axis.annotate(
      f"{value:.2f}", (length, value), color=colour, fontsize=9, fontweight="bold",
      xytext=(7, -9 if rank == 0 else 9), textcoords="offset points", va="center"
    )
  axis.set_xlabel("BO iteration", color=MUTED, fontsize=9)
  axis.set_title(title, color=INK, fontsize=10.5, pad=8, loc="left")
  axis.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="upper right")
  style(axis)


def slopes(axis, arms):
  """Every seed's own 60 -> 120 improvement. Effect against spread, which is what a campaign must
  resolve: the wide box doubles the median gain and sextuples the endpoint scatter."""
  for offset, (name, colour, label) in zip((-0.09, 0.09), arms):
    data = curves(ARMS[name][0])
    if data.shape[1] < 120:
      raise SystemExit(f"{name} has only {data.shape[1]} iterations; this panel compares 60 with 120")
    for row in data:
      axis.plot([offset, 1 + offset], [row[59], row[119]], color=colour, lw=1.2, alpha=0.45,
                marker="o", ms=4, markeredgecolor="white", markeredgewidth=0.8, zorder=2)
    median = [float(np.median(data[:, 59])), float(np.median(data[:, 119]))]
    axis.plot([offset, 1 + offset], median, color=colour, lw=3.0, marker="o", ms=9,
              markeredgecolor="white", markeredgewidth=1.5, zorder=4, label=label)
    axis.annotate(
      f"median −{median[0] - median[1]:.3f} °C\nseed sd {np.std(data[:, 119], ddof=1):.3f} °C",
      (1 + offset, median[1]), color=colour, fontsize=8.5, fontweight="bold",
      xytext=(12, 0), textcoords="offset points", va="center"
    )
  axis.set_xticks([0, 1])
  axis.set_xticklabels(["60 iterations", "120 iterations"])
  axis.set_xlim(-0.4, 2.25)
  axis.set_title("What one seed gains by doubling the budget (σ = 0.05 mM)", color=INK, fontsize=10.5,
                 pad=8, loc="left")
  axis.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="upper right")
  style(axis)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--output", default="output/iteration_payoff.png")
  parser.add_argument(
    "--stacked", action="store_true",
    help="one panel per row instead of side by side -- for print, where a 3:1 figure squeezed into a "
    "text column leaves each panel too small to read"
  )
  arguments = parser.parse_args()

  figure = Figure(figsize=(8.0, 15.0) if arguments.stacked else (15.5, 5.0), facecolor="white")
  axes = figure.subplots(3, 1) if arguments.stacked else figure.subplots(1, 3)
  left, middle, right = axes

  def label(name):
    """Box label read from the runs themselves, never hardcoded -- a figure must not be able to
    claim settings its data does not have."""
    lo, hi = _settings(sorted(glob.glob(ARMS[name][0]))[0])["temperature_bounds"]
    return f"T ∈ [{lo:g}, {hi:g}]"

  def noise_label(name):
    return f"read-out noise σ = {_settings(sorted(glob.glob(ARMS[name][0]))[0])['measurement_noise']:g} mM"

  five = [("narrow_5", NARROW, label("narrow_5")), ("wide_5", WIDE, label("wide_5"))]
  two = [("narrow_2", NARROW, label("narrow_2")), ("wide_2", WIDE, label("wide_2"))]
  convergence(left, five, noise_label("narrow_5") + " (as configured)")
  convergence(middle, two, noise_label("narrow_2"))
  left.set_ylabel("best design so far  (°C RMSE on T*)", color=INK, fontsize=9.5)
  slopes(right, five)
  right.set_ylabel("°C RMSE on T*", color=INK, fontsize=9.5)

  # The stacked layout is half as wide, so the same headline would run off the canvas.
  headline = ("Widening the box makes the optimiser's\nextra iterations pay — and scatters the endpoints"
              if arguments.stacked else
              "Widening the box makes the optimiser's extra iterations pay — and scatters the endpoints")
  subtitle = ("8 BO seeds (permutation-invariant kernel) against 10 random-search\n"
              "instances per setting; median line, band = seed min–max. Lower is better."
              if arguments.stacked else
              "8 BO seeds (permutation-invariant kernel) against 10 random-search instances per setting; "
              "median line, band = seed min–max. Lower is better.")
  figure.suptitle(headline, fontsize=12 if arguments.stacked else 13, color=INK,
                  x=0.012, ha="left", y=0.995, va="top", linespacing=1.35)
  figure.text(0.012, 0.968 if arguments.stacked else 0.925, subtitle, fontsize=9, color=MUTED,
              ha="left", va="top", linespacing=1.4)
  figure.tight_layout(rect=(0, 0, 1, 0.935 if arguments.stacked else 0.90))
  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  figure.savefig(arguments.output, dpi=140, facecolor="white")
  print(f"wrote {arguments.output}")


if __name__ == "__main__":
  main()
