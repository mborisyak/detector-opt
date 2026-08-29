#!/usr/bin/env python3
"""Figures for every experiment run AFTER the growth-procedure fix.

    python scripts/plot_growth_experiments.py --output output/figures/growth

WHAT IS AND IS NOT PLOTTED. Only runs whose training went through ``trainer.train`` -- i.e. with the
growth decision, the exit test and the ``param_mix`` rewind live. Everything under
``output/VOID-probe-semihyper-NO-GROWTH-PROCEDURE`` is excluded by construction: it was produced by a
hand-rolled fixed schedule and is not evidence about anything.

FIVE FIGURES, one per question:

1. ``emnist-meta-probe``    alpha-conv vs alpha-hyper on arm `meta`, validation loss per epoch at the
                            three probed designs. Window growth marked, because a curve that crosses a
                            growth boundary is not comparable across it.
2. ``emnist-arch-probe``    plain CNN (celu / leaky-tanh) vs resnet on arm `from_scratch`.
3. ``emnist-campaign``      best-so-far against detector calls, conditioned vs blinded generator, four
                            arms each. The x axis is CALLS, not design index: the budget is what binds.
4. ``extremes-paired``      the conditioned/blinded cost ratio at the PAIRED design indices only --
                            beyond `n_init` the two runs explore different designs, so index-matching
                            would compare unlike things.
5. ``ship-arm-contrasts``   paired arm contrasts on SHiP with 95% CIs, per design.

CONVENTIONS. One axis per panel, never two. Categorical hues in fixed slot order, never cycled, so a
series keeps its colour across figures. Legend whenever there are two or more series, plus direct
labels where the palette check warns on contrast. Grid and spines recessive; text in ink, never in a
series colour.
"""

import argparse
import glob
import json
import math
import os
import statistics as st

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_3 = '#8a8880'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']


def style():
  plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK_3,
    'axes.linewidth': 0.8,
    'axes.labelcolor': INK_2,
    'axes.titlecolor': INK,
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
    'font.size': 9,
  })


def finish(ax, xlabel, ylabel, title=None):
  ax.grid(True, alpha=0.9, zorder=0)
  ax.set_axisbelow(True)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if title is not None:
    ax.set_title(title, loc='left', pad=8)


def load_probe(path):
  with open(path) as f:
    return json.load(f)


def figure_emnist_meta(root, out):
  """alpha-conv vs alpha-hyper on `meta`: validation loss per epoch, per design."""
  cells = {}
  for name in ('alpha', 'alphahyper'):
    p = f'{root}/probe-growth-emnist/{name}/meta/1244111331/probe.json'
    if not os.path.exists(p):
      return None
    cells[name] = {r['iteration']: r for r in load_probe(p)['designs']}
  labels = {'alpha': 'alpha-conv (gate learned free)', 'alphahyper': 'alpha-hyper (gate generated)'}
  designs = sorted(cells['alpha'])
  fig, axes = plt.subplots(1, len(designs), figsize=(4.1 * len(designs), 3.5), sharey=False)
  for ax, des in zip(np.atleast_1d(axes), designs):
    for slot, name in enumerate(('alpha', 'alphahyper')):
      r = cells[name][des]
      v = r['validation_per_epoch']
      ax.plot(range(1, len(v) + 1), v, color=SERIES[slot], label=labels[name], zorder=3)
      w = r['window_per_epoch']
      for i in range(1, len(w)):
        if w[i] != w[i - 1]:
          ax.axvline(i + 1, color=SERIES[slot], alpha=0.25, linewidth=1.0, zorder=1)
      ax.plot([len(v)], [v[-1]], marker='o', markersize=5, color=SERIES[slot], zorder=4)
    a, h = cells['alpha'][des], cells['alphahyper'][des]
    finish(ax, 'epoch', 'validation loss' if des == designs[0] else '', f'design {des}   spent {a["spent"]:,} / {h["spent"]:,}')
  np.atleast_1d(axes)[0].legend(loc='upper right')
  fig.suptitle(
    'EMNIST, arm meta: generated gate vs free gate  (growth procedure live; vertical ticks = window growth)', x=0.01, ha='left',
    fontsize=10, color=INK
  )
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  fig.savefig(f'{out}/emnist-meta-probe.png', dpi=160)
  plt.close(fig)
  return 'emnist-meta-probe.png'


def figure_emnist_arch(root, out):
  """plain CNN (celu / leaky-tanh) vs resnet on `from_scratch`."""
  spec = [('plain-celu', 'plain CNN, celu'), ('plain-leakytanh', 'plain CNN, leaky-tanh'),
          ('resnet-celu', 'resnet (alpha-conv), celu')]
  cells = {}
  for key, _ in spec:
    p = f'{root}/probe-growth-arch/{key}/from_scratch/1244111331/probe.json'
    if not os.path.exists(p):
      return None
    cells[key] = {r['iteration']: r for r in load_probe(p)['designs']}
  designs = sorted(cells[spec[0][0]])
  fig, axes = plt.subplots(1, len(designs), figsize=(4.1 * len(designs), 3.5))
  for ax, des in zip(np.atleast_1d(axes), designs):
    for slot, (key, label) in enumerate(spec):
      r = cells[key][des]
      v = r['validation_per_epoch']
      ax.plot(range(1, len(v) + 1), v, color=SERIES[slot], label=label, zorder=3)
      ax.plot([len(v)], [v[-1]], marker='o', markersize=5, color=SERIES[slot], zorder=4)
    finish(ax, 'epoch', 'validation loss' if des == designs[0] else '', f'design {des}')
  np.atleast_1d(axes)[0].legend(loc='upper right')
  fig.suptitle('EMNIST, arm from_scratch: architecture and activation', x=0.01, ha='left', fontsize=10, color=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  fig.savefig(f'{out}/emnist-arch-probe.png', dpi=160)
  plt.close(fig)
  return 'emnist-arch-probe.png'


def figure_emnist_campaign(root, out):
  """Best-so-far against detector calls: conditioned vs blinded generator, four arms."""
  arms = ['from_scratch', 'continue', 'closest', 'meta']
  panels = [('campaign-emnist-hyper', 'generator SEES the design'),
            ('campaign-emnist-hyperzero', 'generator BLINDED (zero_design)')]
  fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
  drew = False
  for ax, (tag, title) in zip(axes, panels):
    for slot, arm in enumerate(arms):
      p = f'{root}/{tag}/330924253/{arm}/results.json'
      if not os.path.exists(p):
        continue
      with open(p) as f:
        rows = json.load(f)['results']
      calls = np.cumsum([r['spent'] for r in rows])
      best = np.minimum.accumulate([r['loss'] for r in rows])
      ax.step(calls, best, where='post', color=SERIES[slot], label=arm, zorder=3)
      ax.plot([calls[-1]], [best[-1]], marker='o', markersize=6, color=SERIES[slot], zorder=4)
      ax.annotate(
        f' {arm} ({len(rows)})', (calls[-1], best[-1]), fontsize=7.5, color=INK_2, va='center', ha='left', xytext=(4, 0),
        textcoords='offset points'
      )
      drew = True
    ax.axvline(240000, color=INK_3, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
      'budget = dataset size', (240000, ax.get_ylim()[1]), fontsize=7.5, color=INK_3, va='top', ha='right', xytext=(-4, -4),
      textcoords='offset points'
    )
    finish(ax, 'cumulative detector calls', 'best loss so far', title)
    ax.set_xlim(0, 300000)
  if not drew:
    plt.close(fig)
    return None
  axes[0].legend(loc='upper right', title='arm', title_fontsize=8)
  fig.suptitle(
    'EMNIST hypernetwork campaign, seed 330924253  (label = arm and designs scored)', x=0.01, ha='left', fontsize=10, color=INK
  )
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  fig.savefig(f'{out}/emnist-campaign.png', dpi=160)
  plt.close(fig)
  return 'emnist-campaign.png'


def figure_extremes(root, out):
  """Conditioned/blinded cost ratio at the PAIRED design indices only."""
  sp, des = {}, {}
  for tag in ('xhyper', 'xhyperzero'):
    p = f'{root}/campaign-extremes-{tag}/1244111331/from_scratch/results.json'
    if not os.path.exists(p):
      return None
    with open(p) as f:
      rows = json.load(f)['results']
    sp[tag] = [r['spent'] for r in rows]
    des[tag] = [tuple(r['x_scaled']) for r in rows]
  k = min(len(sp['xhyper']), len(sp['xhyperzero']))
  paired = [i for i in range(k) if des['xhyper'][i] == des['xhyperzero'][i]]
  unpaired = [i for i in range(k) if i not in paired]
  fig, ax = plt.subplots(figsize=(7.6, 4.2))
  ax.axhline(1.0, color=INK_3, linewidth=1.2, zorder=2)
  for i in paired:
    ax.plot([i], [sp['xhyper'][i] / sp['xhyperzero'][i]], marker='o', markersize=9, color=SERIES[0], zorder=4)
  for i in unpaired:
    ax.plot([i], [sp['xhyper'][i] / sp['xhyperzero'][i]], marker='o', markersize=9, markerfacecolor=SURFACE,
            markeredgecolor=INK_3, markeredgewidth=1.4, zorder=3)
  lr = [math.log(sp['xhyper'][i] / sp['xhyperzero'][i]) for i in paired]
  if len(lr) > 1:
    sem = st.stdev(lr) / math.sqrt(len(lr))
    lo, hi, mid = math.exp(st.mean(lr) - 1.96 * sem), math.exp(st.mean(lr) + 1.96 * sem), math.exp(st.mean(lr))
    ax.axhspan(lo, hi, color=SERIES[0], alpha=0.12, zorder=1)
    ax.axhline(mid, color=SERIES[0], linewidth=1.6, linestyle=(0, (5, 3)), zorder=3)
    ax.annotate(
      f'paired geometric mean {mid:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]  n={len(lr)}\nspans 1.0 -> NOT established', (0.98, 0.04),
      xycoords='axes fraction', ha='right', va='bottom', fontsize=8, color=INK_2
    )
  ax.set_yscale('log')
  ax.set_yticks([0.5, 0.75, 1.0, 1.5, 2.0])
  ax.set_yticklabels(['0.5', '0.75', '1.0', '1.5', '2.0'])
  ax.set_xticks(range(k))
  finish(
    ax, 'design index', 'cost ratio  conditioned / blinded',
    'extremes: filled = paired (same physical design)   hollow = unpaired (BO diverged)'
  )
  fig.tight_layout()
  fig.savefig(f'{out}/extremes-paired.png', dpi=160)
  plt.close(fig)
  return 'extremes-paired.png'


def figure_ship_contrasts(out, mirror='/tmp/mix1e2-results'):
  """Paired SHiP arm contrasts with 95% CIs, per design."""
  cells = {}
  for p in sorted(glob.glob(f'{mirror}/d*/probe.json')):
    tag = os.path.basename(os.path.dirname(p))
    if tag.endswith('-v100'):
      continue
    with open(p) as f:
      d = json.load(f)
    cells.setdefault(int(d['design']), []).extend([r for r in d['rows'] if r.get('repeat', 0) >= 0])
  if len(cells) == 0:
    return None
  contrasts = [('meta', 'continue', 0), ('meta_ratio', 'continue', 1), ('meta_ratio', 'meta', 2)]
  fig, ax = plt.subplots(figsize=(7.6, 4.2))
  ax.axvline(0.0, color=INK_3, linewidth=1.2, zorder=2)
  ticks, labels = [], []
  y = 0
  for des in sorted(cells):
    rows = cells[des]
    for slot, (a, b, _) in enumerate(contrasts):
      pa = {(r['seed'], r['repeat']): r['objective'] for r in rows if r['arm'] == a}
      pb = {(r['seed'], r['repeat']): r['objective'] for r in rows if r['arm'] == b}
      keys = sorted(set(pa) & set(pb))
      if len(keys) < 2:
        continue
      d = [pa[k] - pb[k] for k in keys]
      m = st.mean(d)
      sem = st.stdev(d) / math.sqrt(len(d))
      ax.plot([m - 1.96 * sem, m + 1.96 * sem], [y, y], color=SERIES[slot], linewidth=2.4, zorder=3, solid_capstyle='round')
      ax.plot([m], [y], marker='o', markersize=8, color=SERIES[slot], zorder=4)
      ax.annotate(
        f'  {m:+.4f}  n={len(d)}', (m + 1.96 * sem, y), fontsize=7.5, color=INK_2, va='center', ha='left', xytext=(4, 0),
        textcoords='offset points'
      )
      ticks.append(y)
      labels.append(f'design {des}   {a} - {b}')
      y -= 1
    y -= 0.5
  ax.set_yticks(ticks)
  ax.set_yticklabels(labels, fontsize=8)
  finish(
    ax, 'paired difference in objective   (negative = first arm better)', '',
    'SHiP probe at 1.0e-2: arm contrasts paired within (design, seed, repeat)'
  )
  ax.set_xlim(-0.010, 0.016)
  fig.tight_layout()
  fig.savefig(f'{out}/ship-arm-contrasts.png', dpi=160)
  plt.close(fig)
  return 'ship-arm-contrasts.png'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', default='output')
  parser.add_argument('--output', default='output/figures/growth')
  arguments = parser.parse_args()
  os.makedirs(arguments.output, exist_ok=True)
  style()
  made = [
    figure_emnist_meta(arguments.root, arguments.output),
    figure_emnist_arch(arguments.root, arguments.output),
    figure_emnist_campaign(arguments.root, arguments.output),
    figure_extremes(arguments.root, arguments.output),
    figure_ship_contrasts(arguments.output),
  ]
  for name in made:
    print(f'  {"wrote" if name else "SKIPPED (data absent)"} {name or ""}', flush=True)


if __name__ == '__main__':
  main()
