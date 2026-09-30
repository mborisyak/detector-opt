"""Read-out of a one-design probe (`scripts/probe_retention.py`): per-epoch curves per cell, arms overlaid, plus a summary.

    python scripts/plot_probe.py output/ablation-uniform-cern --out output/plots/ablations-2026-09-08 --name uniform

The probe directory holds ``<task>/<seed>/<arm>.json``; every json is one arm's training of the SAME design
from the SAME checkpoint. Two figures:

``<name>-curves.png``  one panel per (task, seed): the objective (solid) and the training loss (dashed) per
                       epoch for every arm, both shifted by the design's penalty so they sit on the
                       reported-loss scale; a tick at each epoch where the training window grew; the
                       trajectory's own recorded loss for that design as a dotted reference line. The
                       legend carries each arm's exit loss and spend.
``<name>-summary.png`` exit loss against spend, one marker per (cell, arm), arms of one cell joined by a
                       line, so the shift of the whole probe reads at a glance.

Colours follow `plot_median.py` for the campaign arms; other arms take the palette in order.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_median import colour_of, INK_2


def cells_of(root):
  """``{(task, seed): {arm: payload}}``."""
  out = {}
  for path in sorted(glob.glob(os.path.join(root, '*', '*', '*.json'))):
    task, seed, name = path.split(os.sep)[-3:]
    with open(path) as handle:
      out.setdefault((task, seed), {})[name[:-len('.json')]] = json.load(handle)
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('root')
  parser.add_argument('--out', default=None)
  parser.add_argument('--name', default='probe')
  parser.add_argument('--cols', type=int, default=3)
  args = parser.parse_args()
  cells = cells_of(args.root)
  if len(cells) == 0:
    raise SystemExit(f'no <task>/<seed>/<arm>.json under {args.root}')
  arms = sorted({arm for payloads in cells.values() for arm in payloads})
  out_dir = args.out if args.out is not None else args.root
  os.makedirs(out_dir, exist_ok=True)

  cols = min(args.cols, len(cells))
  rows = int(np.ceil(len(cells) / cols))
  figure, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows), squeeze=False)
  for axis, ((task, seed), payloads) in zip(axes.flat, sorted(cells.items())):
    reference = next(iter(payloads.values()))['reference_loss']
    axis.axhline(reference, color=INK_2, linestyle=':', linewidth=1.0, label=f'recorded ({reference:.3f})')
    for slot, arm in enumerate(arms):
      if arm not in payloads:
        continue
      p = payloads[arm]
      colour = colour_of(arm, slot)
      offset = p['design_penalty'] if p.get('design_penalty') is not None else 0.0
      objective = np.asarray(p['objective_per_epoch']) + offset
      epochs = np.arange(1, len(objective) + 1)
      axis.plot(epochs, objective, color=colour, linewidth=1.5, label=f"{arm}  {p['loss']:.3f} @ {p['spent'] / 1000:.0f}k")
      axis.plot(epochs, np.asarray(p['train_per_epoch']) + offset, color=colour, linewidth=1.0, linestyle='--')
      window = np.asarray(p['window_per_epoch'])
      grew = np.flatnonzero(np.diff(window) > 0) + 2
      axis.plot(grew, objective[grew - 1], linestyle='none', marker='|', markersize=9, color=colour)
    k = next(iter(payloads.values()))['design_index']
    axis.set_title(f'{task}  {seed}  design {k}', fontsize=9, loc='left')
    axis.legend(fontsize=6.5, frameon=False)
    axis.grid(True, linewidth=0.4)
    axis.tick_params(labelsize=7)
  for axis in axes.flat[len(cells):]:
    axis.set_visible(False)
  for axis in axes[-1]:
    axis.set_xlabel('epoch', fontsize=8, color=INK_2)
  for axis in axes[:, 0]:
    axis.set_ylabel('loss (solid: objective, dashed: training)', fontsize=8, color=INK_2)
  figure.suptitle(
    'One design, one checkpoint, every arm: per-epoch losses; ticks mark window growth', fontsize=10, x=0.01, ha='left'
  )
  figure.tight_layout(rect=(0, 0, 1, 0.96))
  path = os.path.join(out_dir, f'{args.name}-curves.png')
  figure.savefig(path, dpi=160)
  print(f'wrote {path} ({len(cells)} panels)')

  figure, axis = plt.subplots(figsize=(6.4, 4.2))
  for (task, seed), payloads in sorted(cells.items()):
    present = [arm for arm in arms if arm in payloads]
    xs = [payloads[a]['spent'] for a in present]
    ys = [payloads[a]['loss'] for a in present]
    axis.plot(xs, ys, color=INK_2, linewidth=0.8, alpha=0.6)
    for slot, arm in enumerate(arms):
      if arm in payloads:
        axis.plot(
          payloads[arm]['spent'], payloads[arm]['loss'], linestyle='none', marker='o' if task == 'angle' else 's', markersize=6,
          color=colour_of(arm, slot), label=arm if (task, seed) == min(cells) else None
        )
  axis.set_xlabel('detector calls spent on the design', fontsize=9, color=INK_2)
  axis.set_ylabel('loss at exit', fontsize=9, color=INK_2)
  axis.set_title(
    'Exit loss against spend; a line joins the arms of one cell (circles: angle, squares: intersect)', fontsize=9, loc='left'
  )
  axis.legend(fontsize=8, frameon=False)
  axis.grid(True, linewidth=0.4)
  figure.tight_layout()
  path = os.path.join(out_dir, f'{args.name}-summary.png')
  figure.savefig(path, dpi=160)
  print(f'wrote {path}')


if __name__ == '__main__':
  main()
