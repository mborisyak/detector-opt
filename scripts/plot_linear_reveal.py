#!/usr/bin/env python3
"""What withholding the design costs on `linear_d3n4`, and how far a single seed can be trusted.

    python scripts/plot_linear_reveal.py

`linear` is the validation task because `LinearDetector.bayes_risk(design)` is the closed-form best
loss at any design, so "worse" is measurable rather than relative.

⛔️ NO BEST-SO-FAR COMPARISON ACROSS CELLS. BO proposes design `n_init` from a GP fitted to the
observed losses, so a float-level perturbation in those observations moves the proposal and the
trajectories diverge from there. Two cells with identical configs banked 20 and 15 designs with bests
0.0787 and 0.1205. Only the shared random prefix is comparable, and that is all the left two panels
use. The right panel shows the divergence itself, which is why.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, INK_3 = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
ARMS = ['from_scratch', 'continue', 'closest']
N_INIT = 5
ROOT = 'output'

plt.rcParams.update({
  'figure.facecolor': SURFACE,
  'axes.facecolor': SURFACE,
  'savefig.facecolor': SURFACE,
  'axes.edgecolor': INK_3,
  'axes.linewidth': 0.8,
  'axes.labelcolor': INK_2,
  'text.color': INK,
  'xtick.color': INK_2,
  'ytick.color': INK_2,
  'xtick.labelsize': 8,
  'ytick.labelsize': 8,
  'axes.labelsize': 9,
  'axes.titlesize': 10,
  'legend.fontsize': 8,
  'legend.frameon': False,
  'grid.color': '#e6e5e0',
  'grid.linewidth': 0.7,
  'lines.linewidth': 2.0,
})


def cell(tree, arm):
  with open(os.path.join(ROOT, tree, '1244111331', arm, 'results.json')) as handle:
    return json.load(handle)


def prefix(tree, arm, field):
  return [row[field] for row in cell(tree, arm)['results'][:N_INIT]]


def floor():
  import detopt.detector
  payload = cell('linear-d3n4-revealed', 'meta')
  detector = detopt.detector.from_config(payload['config']['detector'])
  return [float(detector.bayes_risk(row['design'])) for row in payload['results'][:N_INIT]]


def spread(axis, tree, field, colour, label, marker):
  """One band per condition: the three per-design arms are ONE trainer with three warm starts, so
  their spread is reproducibility, not an arm effect."""
  values = np.array([prefix(tree, arm, field) for arm in ARMS], float)
  axis.fill_between(range(N_INIT), values.min(axis=0), values.max(axis=0), color=colour, alpha=0.18, linewidth=0)
  axis.plot(
    range(N_INIT), np.median(values, axis=0), marker, color=colour, markersize=5, label=label, markeredgecolor=SURFACE,
    markeredgewidth=0.8
  )


figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

spread(axes[0], 'linear-d3n4-withheld', 'loss', ORANGE, 'design withheld', 'o-')
spread(axes[0], 'linear-d3n4-revealed', 'loss', BLUE, 'design revealed', 'o-')
axes[0].plot(range(N_INIT), floor(), color=INK_3, linewidth=1.6, linestyle=(0, (4, 3)), label='closed-form floor (bayes_risk)')
axes[0].set_ylabel('loss at that design')
axes[0].set_title('Per-design loss, same designs', loc='left', pad=8)

spread(axes[1], 'linear-d3n4-withheld', 'spent', ORANGE, 'design withheld', 'o-')
spread(axes[1], 'linear-d3n4-revealed', 'spent', BLUE, 'design revealed', 'o-')
axes[1].set_yscale('log')
axes[1].set_ylabel('detector calls at that design')
axes[1].set_title('Per-design spend, same designs', loc='left', pad=8)

for axis in axes[:2]:
  axis.set_xticks(range(N_INIT))
  axis.set_xlabel(f'design index (the {N_INIT} shared random proposals)')
  axis.legend(loc='best')

a = cell('linear-d3n4-withheld', 'meta')['results']
b = cell('linear-d3n4-revealed', 'meta')['results']
shared = [i for i in range(min(len(a), len(b))) if np.allclose(a[i]['x_scaled'], b[i]['x_scaled'])]
delta = [abs(a[i]['loss'] - b[i]['loss']) for i in range(min(len(a), len(b)))]
axes[2].semilogy(
  range(len(delta)), np.maximum(delta, 1e-9), 'o-', color=AQUA, markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
  label='|loss difference|'
)
axes[2].axvline(len(shared) - 0.5, color=INK_3, linewidth=1.2, linestyle=(0, (4, 3)))
axes[2].annotate(
  ' designs stop matching here\n (first BO-guided proposal)', (len(shared) - 0.5, 1e-4), fontsize=7.5, color=INK_2, va='center',
  ha='left'
)
axes[2].set_xlabel('design index')
axes[2].set_ylabel('|loss difference| between two IDENTICAL meta runs')
axes[2].set_title('Reproducibility of one seed', loc='left', pad=8)
axes[2].legend(loc='lower right')

for axis in axes:
  axis.grid(True, alpha=0.9)
  axis.set_axisbelow(True)
  for side in ('top', 'right'):
    axis.spines[side].set_visible(False)

figure.suptitle(
  'linear d=3, 4 probes, seed 1244111331 -- withholding the design, and what one seed can support', x=0.01, ha='left',
  fontsize=10, color=INK
)
figure.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs('output/linear-d3n4-revealed/plots', exist_ok=True)
target = 'output/linear-d3n4-revealed/plots/linear-reveal-vs-withhold.png'
figure.savefig(target, dpi=160)
print(f'wrote {target}')
