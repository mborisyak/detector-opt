"""Paired arm comparison for the `meta` regime screen on `linear`.

PRE-REGISTERED before any screen result was read (the wave-1 jobs were submitted first):

  * unit of analysis  -- a SEED, paired between `meta` and `from_scratch`; both arms share the same
    `n_init` Sobol prefix, so the pairing removes the dominant nuisance, which is the prefix draw.
  * score             -- `LinearDetector.bayes_risk` of the proposed design, the exact floor, so no
    held-out sample and no reported-loss bias enters. `best(k)` is the minimum over the first k.
  * primary statistic -- the AREA under the best-so-far curve, `mean_k best(k)` over
    `k = n_init + 1 .. n_designs`. One number per run, uses the whole trajectory rather than one
    arbitrary cut, and is monotone in "found a good design sooner".
  * secondary         -- `best(k)` at k = 8, 12, 16, 20, 24, 32, reported for every k without
    selection.
  * tests             -- exact binomial sign test on the paired differences (ties dropped) and the
    Wilcoxon signed-rank statistic; both one-sided in the direction `meta` better, both reported
    whatever they say.

⚠️ HISTORICAL. The runs this reads were produced by a data procedure that has since been removed
from the tree, so the numbers stay readable but cannot be regenerated. Every arm spent the same
calls on every design there, so design counts matched by construction and the selection effect that
makes `best_loss` favour whichever arm scored more designs could not arise -- which is NOT true of
the growth procedure this repository now runs.
"""

import argparse
import collections
import itertools
import json
import math
import os

import numpy as np


def bayes_risk(flat, n_dimensions, noise):
  flat = np.asarray(flat, np.float64).reshape(-1)
  n_probes = flat.size // n_dimensions
  probe = flat.reshape(n_dimensions, n_probes).T
  rows = np.concatenate([probe, np.ones((n_probes, 1), np.float64)], axis=-1)
  precision = rows.T @ rows / noise**2 + np.eye(n_dimensions + 1)
  return float(np.trace(np.linalg.inv(precision)) / (n_dimensions + 1))


def sign_test(deltas):
  """One-sided exact binomial P(at least this many wins | fair coin), ties dropped."""
  kept = deltas[np.abs(deltas) > 0]
  n = len(kept)
  wins = int(np.sum(kept < 0))
  if n == 0:
    return wins, 0, float('nan')
  tail = sum(math.comb(n, i) for i in range(wins, n + 1)) / 2**n
  return wins, n, tail


def wilcoxon(deltas):
  """Signed-rank statistic and its normal-approximation one-sided p, ties dropped."""
  kept = deltas[np.abs(deltas) > 0]
  n = len(kept)
  if n < 3:
    return float('nan'), float('nan')
  order = np.argsort(np.abs(kept))
  ranks = np.empty(n, np.float64)
  ranks[order] = np.arange(1, n + 1)
  w_negative = float(np.sum(ranks[kept < 0]))
  mean = n * (n + 1) / 4.0
  sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
  z = (w_negative - mean) / sd
  return w_negative, 0.5 * math.erfc(z / math.sqrt(2.0))


def collect(root, n_dimensions, noise):
  runs = collections.defaultdict(dict)
  for dirpath, _, filenames in os.walk(root):
    if 'results.json' not in filenames:
      continue
    arm = os.path.basename(dirpath)
    seed = os.path.basename(os.path.dirname(dirpath))
    with open(os.path.join(dirpath, 'results.json')) as handle:
      payload = json.load(handle)
    results = payload['results']
    true = np.array([bayes_risk(r['design'], n_dimensions, noise) for r in results])
    if payload.get('completed', False) is not True:
      print(f'  SKIPPING {dirpath}: results.json records completed={payload.get("completed")!r} '
            f'({len(results)} designs) -- an unfinished run is not a measurement')
      continue
    runs[seed][arm] = dict(true=true, best=np.minimum.accumulate(true),
                           completed=True)
  return runs



def root_of(root, seed, arm):
  return os.path.join(root, seed, arm, 'results.json')


def json_loss(path):
  with open(path) as handle:
    return np.array([r['loss'] for r in json.load(handle)['results']])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True)
  parser.add_argument('--dimensions', type=int, required=True)
  parser.add_argument('--noise', type=float, required=True)
  parser.add_argument('--reference', default='from_scratch')
  parser.add_argument('--arm', default='meta')
  parser.add_argument('--n-init', type=int, default=5)
  parser.add_argument('--cuts', type=int, nargs='+', default=[8, 12, 16, 20, 24, 32])
  parser.add_argument('--optimum', type=float, default=None)
  args = parser.parse_args()

  runs = collect(args.root, args.dimensions, args.noise)
  paired = sorted(s for s in runs if args.arm in runs[s] and args.reference in runs[s])
  if len(paired) == 0:
    print(f'{args.root}: no paired seeds yet')
    return
  length = min(len(runs[s][a]['best']) for s in paired for a in (args.arm, args.reference))
  print(f'root={args.root}  d={args.dimensions}  noise={args.noise}  '
        f'paired seeds={len(paired)}  common designs={length}')
  if args.optimum is not None:
    print(f'  scores below are bayes_risk; the exact optimum for this cell is {args.optimum:.6f}')

  area = {}
  for arm in (args.arm, args.reference):
    area[arm] = np.array([runs[s][arm]['best'][args.n_init:length].mean() for s in paired])
  delta = area[args.arm] - area[args.reference]
  wins, n_effective, p_sign = sign_test(delta)
  w_stat, p_wilcoxon = wilcoxon(delta)
  print()
  print(f'PRIMARY  area under best-so-far, k = {args.n_init + 1}..{length}')
  print(f'  {args.reference:14s} median {np.median(area[args.reference]):.6f}  '
        f'mean {area[args.reference].mean():.6f}')
  print(f'  {args.arm:14s} median {np.median(area[args.arm]):.6f}  mean {area[args.arm].mean():.6f}')
  print(f'  paired delta ({args.arm} - {args.reference}): median {np.median(delta):+.6f}  '
        f'mean {delta.mean():+.6f}  sd {delta.std(ddof=1):.6f}')
  print(f'  sign test {wins}/{n_effective} wins for {args.arm}  one-sided p={p_sign:.4f}   '
        f'wilcoxon W-={w_stat:.1f} one-sided p={p_wilcoxon:.4f}')

  print()
  print(f'SECONDARY  best bayes_risk among first k')
  print('   k   ' + args.reference.ljust(12) + args.arm.ljust(12) + 'delta(median)   wins   p(sign)')
  for k in args.cuts:
    if k > length:
      continue
    a = np.array([runs[s][args.arm]['best'][k - 1] for s in paired])
    b = np.array([runs[s][args.reference]['best'][k - 1] for s in paired])
    d = a - b
    wins_k, n_k, p_k = sign_test(d)
    print(f'  {k:3d}   {np.median(b):<12.6f}{np.median(a):<12.6f}{np.median(d):+.6f}     '
          f'{wins_k}/{n_k}   {p_k:.4f}')

  print()
  print('REGRESSOR EXCESS  mean over designs of (reported - bayes_risk) / bayes_risk')
  print('  this is the WELL-POWERED readout: one value per design rather than one per run, so it')
  print('  separates "meta learns a better regressor" from "meta happened to search luckier".')
  excess = {}
  for arm in (args.arm, args.reference):
    excess[arm] = np.array([
      float(np.mean((json_loss(root_of(args.root, s, arm)) - runs[s][arm]['true']) / runs[s][arm]['true']))
      for s in paired
    ])
  gap = excess[args.arm] - excess[args.reference]
  wins_e, n_e, p_e = sign_test(gap)
  print(f'  {args.reference:14s} mean {excess[args.reference].mean():+.5f}  sd {excess[args.reference].std(ddof=1):.5f}')
  print(f'  {args.arm:14s} mean {excess[args.arm].mean():+.5f}  sd {excess[args.arm].std(ddof=1):.5f}')
  print(f'  paired delta: mean {gap.mean():+.5f}  sd {gap.std(ddof=1):.5f}  '
        f'{wins_e}/{n_e} seeds where {args.arm} is the better regressor  one-sided p={p_e:.4f}')

  print()
  print('PER-SEED area under best-so-far')
  for i, seed in enumerate(paired):
    print(f'  {seed:>10}  {args.reference}={area[args.reference][i]:.6f}  '
          f'{args.arm}={area[args.arm][i]:.6f}  delta={delta[i]:+.6f}')


if __name__ == '__main__':
  main()
