#!/usr/bin/env python3
"""The rewind probe: does `rewind` help, and does it act differently in each arm?

    python scripts/plot_rewind_probe.py output/ship-rewind-probe --out output/plots

LEFT/CENTRE, one panel per arm: the SAME (seed, design) trained with the rewind and without, plotted
against each other on held-out test loss. The diagonal is "no effect"; a point ABOVE it is a design
the rewind made WORSE. Paired, so each point is one design measured twice.

RIGHT: the relative within-design displacement ||p - q|| / ||p||, which is the quantity the rewind
acts on -- a fresh draw for `from_scratch`, the carried network for `meta`, so it is far larger in
one than the other. That asymmetry is why the same rule does different amounts of work per arm.
"""
import argparse, glob, json, os
import numpy as np, matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, INK_3 = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({
  'figure.facecolor': SURFACE,
  'axes.facecolor': SURFACE,
  'savefig.facecolor': SURFACE,
  'axes.edgecolor': INK_3,
  'axes.linewidth': 0.8,
  'text.color': INK,
  'axes.labelcolor': INK_2,
  'xtick.color': INK_2,
  'ytick.color': INK_2,
  'xtick.labelsize': 8,
  'ytick.labelsize': 8,
  'axes.labelsize': 9,
  'axes.titlesize': 10,
  'legend.fontsize': 8,
  'legend.frameon': False,
  'grid.color': '#e6e5e0',
  'grid.linewidth': 0.7
})

p = argparse.ArgumentParser()
p.add_argument('root')
p.add_argument('--out', default='output/plots')
p.add_argument('--name', default='rewind_probe.png')
a = p.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.root, '*.json'))):
  rows += json.load(open(f))['rows']
rows = [r for r in rows if r.get('status') == 'converged']
arms = ['from_scratch', 'meta']
figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

for axis, arm in zip(axes[:2], arms):
  cells = {
    pm: {(r['seed'], r['design_index']): r
         for r in rows if r['arm'] == arm and r['rewind'] == pm}
    for pm in (0.25, 0.0)
  }
  common = sorted(set(cells[0.25]) & set(cells[0.0]))
  x = np.array([cells[0.0][k]['test'] for k in common])
  y = np.array([cells[0.25][k]['test'] for k in common])
  lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
  pad = 0.05 * (hi - lo)
  axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_3, lw=1.2, ls='--', label='no effect')
  axis.scatter(x, y, s=46, color=BLUE, alpha=0.85, edgecolor='none')
  d = y - x
  sem = d.std(ddof=1) / np.sqrt(len(d))
  axis.set(
    title=f"{arm}   rewind vs none   (n={len(d)})", xlabel='test loss, NO rewind (rewind 0)',
    ylabel='test loss, rewind (rewind 0.25)', xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad)
  )
  axis.text(
    0.04, 0.94, f"rewind - none = {d.mean():+.4f} $\\pm$ {sem:.4f}\n({d.mean()/sem:+.2f}$\\sigma$)   "
    f"rewind better {int((d < 0).sum())}/{len(d)}", transform=axis.transAxes, va='top', fontsize=9, color=INK
  )
  axis.grid(True, alpha=0.25)
  axis.legend(loc='lower right')

axis = axes[2]
for arm, colour in zip(arms, (BLUE, ORANGE)):
  vals = [r['displacement'] for r in rows if r['arm'] == arm and r['displacement'] is not None]
  axis.scatter(np.full(len(vals), arm), vals, s=40, color=colour, alpha=0.65, edgecolor='none')
  axis.plot([arm], [np.median(vals)], marker='_', ms=42, color=colour, mew=2.4)
axis.set(title='what the rewind acts on', ylabel=r'$\|p-q\|\,/\,\|p\|$ per design')
axis.grid(True, alpha=0.25, axis='y')
figure.suptitle('Rewind probe on ship-intersect-w2x-2m: paired per (seed, design), held-out test', y=1.0)
figure.tight_layout()
os.makedirs(a.out, exist_ok=True)
figure.savefig(os.path.join(a.out, a.name), dpi=140, bbox_inches='tight')
print('wrote', os.path.join(a.out, a.name))
