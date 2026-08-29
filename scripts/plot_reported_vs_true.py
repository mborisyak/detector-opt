#!/usr/bin/env python3
"""Reported loss against TRUE design quality, per arm, on a task with a closed form.

    python scripts/plot_reported_vs_true.py output/linear-d2n3-10seed --detector linear_d2n3

`linear` gives `LinearDetector.bayes_risk(design)` -- the loss a perfect estimator reaches at that
design -- so each arm's incumbent can be scored with NO trainer in the loop. The difference between
that and what the run REPORTED is the estimator's own contribution, and it is not the same for every
arm: on d2n3 over 10 seeds `meta` reported 0.0070 below its own floor against 0.0026-0.0036 for the
other three, which is enough to invert a ranking.

LEFT: reported against true, one point per cell, with the identity line. A point BELOW the line is a
cell whose run reported a loss better than the design can actually deliver.
RIGHT: the paired arm difference on both instruments, which is where the inversion shows: a bar that
crosses zero on TRUE while clearing it on REPORTED is an arm ranking produced by the estimator.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
import yaml
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, INK_3 = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880'
ARM_COLOUR = {'from_scratch': '#2a78d6', 'continue': '#eb6834', 'closest': '#1baf7a', 'meta': '#eda100'}
ORDER = ('meta', 'from_scratch', 'closest', 'continue')

plt.rcParams.update({
  'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
  'axes.edgecolor': INK_3, 'axes.linewidth': 0.8, 'axes.labelcolor': INK_2, 'text.color': INK,
  'xtick.color': INK_2, 'ytick.color': INK_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
  'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8, 'legend.frameon': False,
  'grid.color': '#e6e5e0', 'grid.linewidth': 0.7, 'lines.linewidth': 2.0,
})


def load(tree, detector_config):
  import detopt.detector
  det = detopt.detector.from_config(yaml.safe_load(open(detector_config)))
  out = collections.defaultdict(dict)
  for path in sorted(glob.glob(os.path.join(tree, '*', '*', 'results.json'))):
    payload = json.load(open(path))
    if not payload.get('completed'):
      continue
    seed, arm = path.split(os.sep)[-3:-1]
    out[arm][seed] = (payload['best_loss'], float(det.bayes_risk(payload['best_design'])))
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('tree')
  parser.add_argument('--detector', required=True, help='name under config/detector/')
  parser.add_argument('--baseline', default='from_scratch', help='arm the paired panel compares against')
  parser.add_argument('--name', default='reported-vs-true.png')
  arguments = parser.parse_args()

  data = load(arguments.tree, f'config/detector/{arguments.detector}.yaml')
  arms = [a for a in ORDER if a in data]
  figure, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.8))

  lo = min(min(v) for arm in arms for v in data[arm].values())
  hi = max(max(v) for arm in arms for v in data[arm].values())
  pad = 0.05 * (hi - lo)
  left.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_3, linewidth=0.9, linestyle=(0, (3, 3)))
  for arm in arms:
    true = [v[1] for v in data[arm].values()]
    reported = [v[0] for v in data[arm].values()]
    bias = np.mean(np.array(reported) - np.array(true))
    left.scatter(true, reported, s=34, color=ARM_COLOUR[arm], alpha=0.85, edgecolor=SURFACE, linewidth=0.8,
                 label=f'{arm}  ({bias:+.4f})')
  left.set_xlabel('TRUE quality of the chosen design   bayes_risk(best_design)')
  left.set_ylabel('loss the run REPORTED')
  left.set_title('a  below the line = reported better than achievable\n     (legend: mean optimism)',
                 loc='left', pad=8)
  left.grid(True, alpha=0.9)
  left.set_axisbelow(True)
  left.legend(loc='upper left')
  for side in ('top', 'right'):
    left.spines[side].set_visible(False)

  others = [a for a in arms if a != arguments.baseline]
  ypos = np.arange(len(others))
  for i, arm in enumerate(others):
    seeds = sorted(set(data[arm]) & set(data[arguments.baseline]))
    for j, (key, off) in enumerate((('TRUE', -0.16), ('REPORTED', +0.16))):
      idx = 1 if key == 'TRUE' else 0
      diff = np.array([data[arm][s][idx] - data[arguments.baseline][s][idx] for s in seeds])
      err = diff.std(ddof=1) / np.sqrt(len(diff))
      right.errorbar(diff.mean(), i + off, xerr=err, fmt='o', markersize=7, capsize=3,
                     color=ARM_COLOUR[arm], alpha=1.0 if key == 'TRUE' else 0.45,
                     markerfacecolor=ARM_COLOUR[arm] if key == 'TRUE' else SURFACE)
      right.annotate(key, xy=(diff.mean(), i + off), xytext=(0, 9 if key == 'TRUE' else -13),
                     textcoords='offset points', fontsize=6.5, color=INK_2, ha='center')
  right.axvline(0.0, color=INK_3, linewidth=0.9, linestyle=(0, (3, 3)))
  right.set_yticks(ypos)
  right.set_yticklabels(others)
  right.set_xlabel(f'paired difference against {arguments.baseline}   (negative = better)')
  right.set_title('b  filled = TRUE quality, hollow = REPORTED\n     bars are 1 SEM over seeds', loc='left', pad=8)
  right.grid(True, alpha=0.9, axis='x')
  right.set_axisbelow(True)
  for side in ('top', 'right', 'left'):
    right.spines[side].set_visible(False)

  figure.suptitle(f'{os.path.basename(arguments.tree.rstrip("/"))}: what the estimator contributes to the arm ranking',
                  x=0.01, ha='left', fontsize=10, color=INK)
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  out = os.path.join(arguments.tree, 'plots')
  os.makedirs(out, exist_ok=True)
  target = os.path.join(out, arguments.name)
  figure.savefig(target, dpi=160)
  print(f'wrote {target}')


if __name__ == '__main__':
  main()
