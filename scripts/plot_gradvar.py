"""Per-epoch gradient statistics of the structured against the uniform minibatch (`scripts/probe_gradient_variance.py`).

    python scripts/plot_gradvar.py output/ablation-gradvar-cern --out output/plots/ablations-2026-09-08 \
        --measure mean_variance            (or mean_moment_over_std, mean_moment_over_rms, ...)

The probe directory holds ``<task>/<seed>/design-<k>.json``. One panel per (task, seed, design), rows = (task, seed),
columns = design index. In each panel the chosen ``--measure`` is drawn per epoch for the STRUCTURED arm measured
along its own training (solid, meta colour) and for the UNIFORM arm along its own (solid, the next palette colour),
plus, dashed, the OTHER composition measured at the same parameters -- so a dashed line next to a solid one of the
other colour is the same batch composition at the other arm's parameters. The legend carries each arm's exit loss.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_median import colour_of, INK_2, SERIES

ARM_COLOUR = {"structured": colour_of("meta", 0), "uniform": SERIES[0]}


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("root")
  parser.add_argument("--out", default=None)
  parser.add_argument("--measure", default="mean_variance")
  parser.add_argument("--log", action="store_true", help="log scale on the y axis")
  args = parser.parse_args()
  cells = {}
  for path in sorted(glob.glob(os.path.join(args.root, "*", "*", "design-*.json"))):
    task, seed, name = path.split(os.sep)[-3:]
    with open(path) as handle:
      cells[(task, seed, int(name[len("design-"):-len(".json")]))] = json.load(handle)
  if len(cells) == 0:
    raise SystemExit(f"no <task>/<seed>/design-<k>.json under {args.root}")
  rows = sorted({(t, s) for t, s, _ in cells})
  designs = sorted({k for _, _, k in cells})
  figure, axes = plt.subplots(len(rows), len(designs), figsize=(4.0 * len(designs), 2.6 * len(rows)), squeeze=False)
  for i, (task, seed) in enumerate(rows):
    for j, k in enumerate(designs):
      axis = axes[i, j]
      payload = cells.get((task, seed, k))
      if payload is None:
        axis.set_visible(False)
        continue
      for arm in ("structured", "uniform"):
        series = payload["arms"][arm]["series"]
        own = np.asarray(series[f"{arm}_{args.measure}"])
        other = "uniform" if arm == "structured" else "structured"
        cross = np.asarray(series[f"{other}_{args.measure}"])
        epochs = np.arange(1, len(own) + 1)
        axis.plot(epochs, own, color=ARM_COLOUR[arm], linewidth=1.5, label=f"{arm} (loss {payload['arms'][arm]['loss']:.3f})")
        axis.plot(epochs, cross, color=ARM_COLOUR[other], linewidth=0.9, linestyle="--", alpha=0.7)
      if args.log:
        axis.set_yscale("log")
      axis.set_title(f"{task}  {seed}  design {k}", fontsize=8.5, loc="left")
      axis.legend(fontsize=6.5, frameon=False)
      axis.grid(True, linewidth=0.4)
      axis.tick_params(labelsize=7)
  for axis in axes[-1]:
    axis.set_xlabel("epoch", fontsize=8, color=INK_2)
  for axis in axes[:, 0]:
    axis.set_ylabel(args.measure, fontsize=8, color=INK_2)
  figure.suptitle(
    f"{args.measure} per epoch: solid = each arm's own batch composition along its own training; "
    "dashed = the other composition at the same parameters", fontsize=10, x=0.01, ha="left"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.97))
  out_dir = args.out if args.out is not None else args.root
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f"gradvar-{args.measure.replace('_', '-')}.png")
  figure.savefig(path, dpi=150)
  print(f"wrote {path} ({len(cells)} panels)")


if __name__ == "__main__":
  main()
