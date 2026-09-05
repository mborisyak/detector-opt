#!/usr/bin/env python3
"""Cosine-probe figures: per seed, every (optimiser, arm) on shared axes -- loss per epoch with data additions
marked, the schedule multiplier, the kernel norm -- plus a summary of epochs and converged loss per seed.

    python scripts/plot_probe_cosine.py output/probe-cosine/cern output/plots/probe-cosine
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt

STYLE = {
  ("adamaxw", "off"): ("#1f4e79", "-"),
  ("adamaxw", "on"): ("#d1495b", "-"),
  ("adamaxw", "on2x"): ("#e0a400", "-"),
  ("nadamw", "off"): ("#2a9d8f", "-"),
  ("nadamw", "on"): ("#8e44ad", "-"),
  ("nadamw", "on2x"): ("#7f5539", "-"),
  ("adamw", "off"): ("#6c757d", "-"),
  ("adamw", "on"): ("#f28e2b", "-"),
  ("adamaxw099", "off"): ("#1f4e79", "--"),
  ("adamaxw099", "on"): ("#d1495b", "--"),
}
ORDER = [("adamaxw", "off"), ("adamaxw", "on"), ("adamaxw", "on2x"), ("nadamw", "off"), ("nadamw", "on"), ("nadamw", "on2x"),
         ("adamw", "off"), ("adamw", "on"), ("adamaxw099", "off"), ("adamaxw099", "on")]


def load(root):
  cells = {}
  for path in sorted(glob.glob(f"{root}/**/cosine-*.json", recursive=True)):
    record = json.load(open(path))
    optimizer = path.split("/")[-3]
    arm = os.path.basename(path).split("-")[1].split(".")[0]
    cells.setdefault(str(record["seed"]), {})[(optimizer, arm)] = record
  return cells


def draw_seed(seed, series, out_dir):
  fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.5]})
  for key in ORDER:
    if key not in series:
      continue
    record = series[key]
    color, ls = STYLE[key]
    epochs = record["epoch_records"]
    x = range(len(epochs))
    label = (f"{key[0]} cosine {key[1]}: {record['epochs']} epochs, {len(record['additions'])} additions, "
             f"loss {record['loss']:.4f}±{record['loss_std']:.4f}")
    axes[0].plot(x, record["train_per_epoch"], color=color, ls=ls, lw=1.0, label=label)
    axes[0].plot(x, record["val_per_epoch"], color=color, ls=":", lw=0.8, alpha=0.7)
    for i in record["additions"]:
      axes[0].axvline(i, color=color, lw=0.5, alpha=0.35)
    axes[1].plot(x, [e["multiplier"] for e in epochs], color=color, ls=ls, lw=1.0)
    axes[2].plot(x, [e["norm_out"]["kernel"] for e in epochs], color=color, ls=ls, lw=1.0)
  axes[0].set_ylabel("loss (solid train, dotted val)")
  axes[0].set_ylim(top=min(1.0, axes[0].get_ylim()[1]))
  axes[0].legend(loc="upper right", fontsize=8)
  any_record = next(iter(series.values()))
  axes[0].set_title(f"intersect, {any_record['arm']}, design index {any_record['design_index']}, shrink 0.3 / noise 0.01, seed {seed}")
  axes[1].set_ylabel("lr multiplier")
  axes[2].set_ylabel("kernel L2 norm")
  axes[2].set_xlabel("epoch (vertical lines: data additions)")
  for ax in axes:
    ax.grid(alpha=0.3)
  fig.tight_layout()
  path = os.path.join(out_dir, f"seed-{seed}.png")
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return path


def draw_summary(cells, out_dir):
  seeds = sorted(cells)
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
  width = 0.13
  for j, key in enumerate(ORDER):
    color, _ = STYLE[key]
    xs, epochs, losses, errors, calls = [], [], [], [], []
    for i, seed in enumerate(seeds):
      record = cells[seed].get(key)
      if record is None:
        continue
      xs.append(i + (j - len(ORDER) / 2 + 0.5) * width)
      epochs.append(record["epochs"])
      losses.append(record["loss"])
      errors.append(record["loss_std"])
      calls.append(record["spent"])
    if len(xs) == 0:
      continue
    label = f"{key[0]} cosine {key[1]}"
    axes[0].bar(xs, epochs, width=width, color=color, label=label)
    axes[1].errorbar(xs, losses, yerr=errors, fmt="o", color=color, capsize=3, label=label)
    axes[2].bar(xs, calls, width=width, color=color, label=label)
  for ax, title in zip(axes, ("epochs to convergence", "converged loss (± reported std)", "detector calls spent")):
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds, fontsize=8)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
  axes[0].legend(fontsize=7)
  any_record = next(iter(next(iter(cells.values())).values()))
  fig.suptitle(f"intersect, {any_record['arm']}, design index {any_record['design_index']}, shrink 0.3 / noise 0.01 -- per seed")
  fig.tight_layout()
  path = os.path.join(out_dir, "summary.png")
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return path


if __name__ == "__main__":
  root, out_dir = sys.argv[1], sys.argv[2]
  os.makedirs(out_dir, exist_ok=True)
  cells = load(root)
  for seed, series in cells.items():
    print(draw_seed(seed, series, out_dir), "series:", [f"{o}/{a}" for o, a in ORDER if (o, a) in series])
  print(draw_summary(cells, out_dir))
