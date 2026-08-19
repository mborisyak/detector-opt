"""Anatomy of the best-so-far descent in a finished BO campaign, arm by arm.

READ-ONLY. Consumes `results.json` / `convergence.json` / `median.json` already on disk and computes
statistics. It trains nothing, launches nothing and writes nothing except its own stdout.

The question it answers: `from_scratch` probes fewer designs than `meta` but each of its best-so-far
improvements is a larger step. Is that because it finds genuinely better designs, or because its
reported losses are noisier so its running minimum is more strongly selected on noise?

WHAT IS MEASURED, and why each piece is here.

* The first `n_init` designs of every run are a SHARED Sobol block: the same designs at the same seed
  in both arms, proposed with no GP in the loop. They are therefore PAIRED evaluations of identical
  designs under the two trainers, and they are the only replicate structure the campaign has. Their
  loss difference estimates the combined reported-loss noise of the two arms plus any level offset;
  the first Sobol design is a further control, because `ContinualTrainer` has an empty replay pool at
  `w0 == 0` and both arms then train the same way. Every step statistic below is reported with the
  Sobol block held out, since it carries no arm signal in the proposals.

* Step anatomy: for each cell, the improving steps of the running minimum, their count, their sizes,
  and the total descent. Total descent is the sum of the steps, so "fewer, larger steps" is a claim
  about granularity only if the totals match.

* Noise probes that do not need a training run: (a) the paired Sobol differences; (b) the pairwise
  |loss difference| of two designs in the same run as a function of their permutation-invariant
  distance in the scaled cube, whose small-distance end is noise-dominated; (c) the dispersion of
  each arm's losses about its own seed's Sobol-block level, upper tail and lower tail separately,
  since noise widens both and a genuinely better search only lowers one.

* Regime probes: `loss_std` (the design's own `diff + err` at its stopping epoch, censored at
  `loss_precision`), `spent` (the design's window), and whether step size moves with either.

* Exploration probes: permutation-invariant distance between consecutive proposals and from each new
  proposal to the incumbent. The enzyme batch is a SET of `n_experiments` experiments, so a design is
  only defined up to permuting them, and any distance that ignores that overstates the movement.

The design layout for `enzyme_mm_sym_m3` is four fields of `n_experiments` coordinates each, in flat
order `enzyme_fraction, temperature, concentration_A, concentration_B`, so experiment `k` is the
tuple `(x[k], x[n + k], x[2n + k], x[3n + k])` and the four blocks permute TOGETHER.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

CAMPAIGN = 'output/campaign-mm-p5e3'
SEEDS = ['1244111331', '126382657', '330924253', '750143450', '9577242']
ARMS = ['from_scratch', 'meta']
N_INIT = 5
N_EXPERIMENTS = 3
N_TARGETS = 3
LOSS_PRECISION = 5.0e-3
CEILING = 1.0 / 3.0
CEILING_CUT = 0.325
BASE_SPENT = 2731
SPENT_INCREMENT = 1365


def load_cell(root, seed, arm):
  """One cell's `results.json` rows, or None when the cell is absent or unfinished."""
  path = os.path.join(root, seed, arm, 'results.json')
  if not os.path.exists(path):
    return None
  with open(path) as handle:
    payload = json.load(handle)
  return payload


def permutations_of_blocks(n_experiments, n_fields):
  """Index arrays that reorder a flat design under each permutation of the exchangeable elements."""
  base = np.arange(n_experiments * n_fields).reshape(n_fields, n_experiments)
  out = []
  for order in itertools.permutations(range(n_experiments)):
    out.append(base[:, list(order)].reshape(-1))
  return np.stack(out)


def invariant_distance(a, b, perms):
  """Euclidean distance in the scaled cube, minimised over permutations of the exchangeable set."""
  return float(np.min(np.linalg.norm(a[perms] - b[None, :], axis=1)))


def improving_steps(losses):
  """Indices (0-based) and sizes of the running minimum's decrements, the first design excluded."""
  indices, sizes = [], []
  best = losses[0]
  for i in range(1, len(losses)):
    if losses[i] < best:
      indices.append(i)
      sizes.append(best - losses[i])
      best = losses[i]
  return np.asarray(indices, dtype=int), np.asarray(sizes, dtype=float)


def describe(values, label, unit=''):
  """A one-line five-number summary that never pretends a short sample is a distribution."""
  v = np.asarray(values, dtype=float)
  if v.size == 0:
    return f'{label:<34} n=0'
  return (
    f'{label:<34} n={v.size:<4d} min={v.min():.4f} p25={np.percentile(v, 25):.4f} '
    f'med={np.median(v):.4f} p75={np.percentile(v, 75):.4f} max={v.max():.4f} '
    f'mean={v.mean():.4f}{unit}'
  )


def mannwhitney(a, b):
  """Two-sided Mann-Whitney U with the sample sizes, so the reader can judge the power."""
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  if a.size < 2 or b.size < 2:
    return float('nan'), 0, 0
  return float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue), a.size, b.size


def main():
  root = sys.argv[1] if len(sys.argv) > 1 else CAMPAIGN
  perms = permutations_of_blocks(N_EXPERIMENTS, 4)
  cells = {}
  for seed in SEEDS:
    for arm in ARMS:
      payload = load_cell(root, seed, arm)
      if payload is None:
        continue
      rows = payload['results']
      cells[(seed, arm)] = {
        'rows': rows,
        'loss': np.array([r['loss'] for r in rows], dtype=float),
        'loss_std': np.array([r['loss_std'] for r in rows], dtype=float),
        'spent': np.array([r['spent'] for r in rows], dtype=float),
        'time_s': np.array([r['time_s'] for r in rows], dtype=float),
        'x': np.array([r['x_scaled'] for r in rows], dtype=float),
        'completed': bool(payload['completed']),
        'calls': int(payload['detector_calls_used']),
      }

  print('=' * 110)
  print('1. INVENTORY')
  print('=' * 110)
  print(
    f'{"seed":>12} {"arm":<13} {"n":>3} {"done":>5} {"calls":>8} {"final best":>11} '
    f'{"best@sobol":>11} {"mean spent":>11} {"med spent":>10}'
  )
  for (seed, arm), c in sorted(cells.items()):
    losses = c['loss']
    sobol_best = np.min(losses[:N_INIT]) if losses.size >= N_INIT else float('nan')
    print(
      f'{seed:>12} {arm:<13} {losses.size:>3} {str(c["completed"]):>5} {c["calls"]:>8} '
      f'{np.min(losses):>11.4f} {sobol_best:>11.4f} {c["spent"].mean():>11.1f} '
      f'{np.median(c["spent"]):>10.1f}'
    )

  usable = {k: v for k, v in cells.items() if v['completed'] and v['loss'].size > N_INIT}
  print(f'\nusable complete cells: {len(usable)} of {len(cells)}')

  print()
  print('=' * 110)
  print('2. THE SHARED SOBOL BLOCK -- identity check and paired noise')
  print('=' * 110)
  identical = True
  for seed in SEEDS:
    a = usable.get((seed, 'from_scratch'))
    b = usable.get((seed, 'meta'))
    if a is None or b is None:
      continue
    delta = np.abs(a['x'][:N_INIT] - b['x'][:N_INIT]).max()
    if delta > 1e-6:
      identical = False
    print(f'  seed {seed:>12}  max |x_fs - x_meta| over the first {N_INIT} designs = {delta:.3e}')
  print(f'  Sobol block identical across arms: {identical}')

  paired_first, paired_rest, paired_all = [], [], []
  print(f'\n  paired loss on IDENTICAL designs (from_scratch - meta):')
  for seed in SEEDS:
    a = usable.get((seed, 'from_scratch'))
    b = usable.get((seed, 'meta'))
    if a is None or b is None:
      continue
    d = a['loss'][:N_INIT] - b['loss'][:N_INIT]
    paired_first.append(d[0])
    paired_rest.extend(d[1:].tolist())
    paired_all.extend(d.tolist())
    formatted = '  '.join(f'{x:+.4f}' for x in d)
    print(f'  seed {seed:>12}  {formatted}')
  paired_first = np.asarray(paired_first)
  paired_rest = np.asarray(paired_rest)
  paired_all = np.asarray(paired_all)
  for name, arr in (('design 1 only (no replay in either arm)', paired_first),
                    ('designs 2..5 (replay active in meta)', paired_rest), ('all 5 Sobol designs', paired_all)):
    if arr.size == 0:
      continue
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan')
    sem = sd / np.sqrt(arr.size) if arr.size > 1 else float('nan')
    t_p = float(stats.ttest_1samp(arr, 0.0).pvalue) if arr.size > 1 else float('nan')
    print(f'  {name:<42} n={arr.size:<3} mean={arr.mean():+.4f} sd={sd:.4f} '
          f'sem={sem:.4f} t-test vs 0: p={t_p:.3f}')
  if paired_all.size > 1:
    sd_all = float(np.std(paired_all, ddof=1))
    print(f'\n  combined per-arm reported-loss noise, if the two arms are equally noisy:')
    print(f'    sd(difference) = {sd_all:.4f}  ->  sigma_per_arm ~ {sd_all / np.sqrt(2.0):.4f}')
    print(f'    for reference, loss_precision = {LOSS_PRECISION:.4f}')

  print()
  print('=' * 110)
  print('3. BEST-SO-FAR STEP ANATOMY (Sobol block held out)')
  print('=' * 110)
  print(
    f'{"seed":>12} {"arm":<13} {"n":>3} {"n_post":>6} {"steps_all":>9} {"steps_post":>10} '
    f'{"descent_post":>12} {"med_step_post":>13} {"max_step_post":>13}'
  )
  per_arm = {arm: {'steps_post': [], 'n_steps_post': [], 'descent_post': [], 'n_post': [], 'n_designs': []} for arm in ARMS}
  for seed in SEEDS:
    for arm in ARMS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      losses = c['loss']
      indices, sizes = improving_steps(losses)
      post = indices >= N_INIT
      running = np.minimum.accumulate(losses)
      descent_post = running[N_INIT - 1] - running[-1]
      sizes_post = sizes[post]
      per_arm[arm]['steps_post'].extend(sizes_post.tolist())
      per_arm[arm]['n_steps_post'].append(int(sizes_post.size))
      per_arm[arm]['descent_post'].append(float(descent_post))
      per_arm[arm]['n_post'].append(int(losses.size - N_INIT))
      per_arm[arm]['n_designs'].append(int(losses.size))
      med = np.median(sizes_post) if sizes_post.size > 0 else float('nan')
      mx = sizes_post.max() if sizes_post.size > 0 else float('nan')
      print(
        f'{seed:>12} {arm:<13} {losses.size:>3} {losses.size - N_INIT:>6} {sizes.size:>9} '
        f'{sizes_post.size:>10} {descent_post:>12.4f} {med:>13.4f} {mx:>13.4f}'
      )

  print()
  for arm in ARMS:
    d = per_arm[arm]
    print(f'  {arm}:')
    print('    ' + describe(d['steps_post'], 'post-Sobol step sizes'))
    print(f'    designs probed per seed         {d["n_designs"]}  median {np.median(d["n_designs"]):.1f}')
    print(f'    post-Sobol proposals per seed   {d["n_post"]}  median {np.median(d["n_post"]):.1f}')
    print(f'    improving steps per seed        {d["n_steps_post"]}  median {np.median(d["n_steps_post"]):.1f}')
    print(
      f'    post-Sobol descent per seed     {[round(x, 4) for x in d["descent_post"]]}  '
      f'median {np.median(d["descent_post"]):.4f}'
    )
    hit = np.asarray(d['n_steps_post'], dtype=float)
    tries = np.asarray(d['n_post'], dtype=float)
    print(f'    hit rate (improving/proposal)   {hit.sum() / tries.sum():.3f}  '
          f'({int(hit.sum())} of {int(tries.sum())})')

  print('\n  paired-by-seed comparisons (n=5 pairs, sign test and Wilcoxon):')
  for key, label in (('n_designs', 'designs probed'), ('n_post', 'post-Sobol proposals'), ('n_steps_post', 'improving steps'),
                     ('descent_post', 'post-Sobol descent')):
    a = np.asarray(per_arm['from_scratch'][key], dtype=float)
    b = np.asarray(per_arm['meta'][key], dtype=float)
    if a.size != b.size or a.size < 2:
      continue
    diff = a - b
    wins = int(np.sum(diff > 0))
    losses_count = int(np.sum(diff < 0))
    sign_p = float(stats.binomtest(wins, wins + losses_count, 0.5).pvalue) if wins + losses_count > 0 else float('nan')
    wil_p = float('nan')
    if np.any(diff != 0):
      wil_p = float(stats.wilcoxon(a, b, zero_method='zsplit').pvalue)
    print(
      f'    {label:<24} fs {a.tolist()}  meta {b.tolist()}  '
      f'fs>meta in {wins}/{wins + losses_count}  sign p={sign_p:.3f}  wilcoxon p={wil_p:.3f}'
    )

  p, na, nb = mannwhitney(per_arm['from_scratch']['steps_post'], per_arm['meta']['steps_post'])
  print(f'\n  pooled post-Sobol step sizes, Mann-Whitney: p={p:.4f}  (n_fs={na}, n_meta={nb})')
  print('  ARITHMETIC IDENTITY: mean step = total descent / number of steps.')
  for arm in ARMS:
    d = per_arm[arm]
    total = float(np.sum(d['descent_post']))
    n_steps = int(np.sum(d['n_steps_post']))
    print(f'    {arm:<13} total descent {total:.4f} over {n_steps} steps -> mean step '
          f'{total / max(n_steps, 1):.4f}')

  print()
  print('=' * 110)
  print('4. loss_std AND spent -- the design-level regime')
  print('=' * 110)
  for arm in ARMS:
    all_std, all_spent, post_std, post_spent = [], [], [], []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      all_std.extend(c['loss_std'].tolist())
      all_spent.extend(c['spent'].tolist())
      post_std.extend(c['loss_std'][N_INIT:].tolist())
      post_spent.extend(c['spent'][N_INIT:].tolist())
    per_arm[arm]['post_std'] = post_std
    per_arm[arm]['post_spent'] = post_spent
    at_bar = float(np.mean(np.asarray(all_std) > LOSS_PRECISION - 1e-5))
    print(f'  {arm}:')
    print('    ' + describe(all_std, 'loss_std, all designs'))
    print('    ' + describe(post_std, 'loss_std, post-Sobol'))
    print(f'    fraction with loss_std within 1e-5 of the {LOSS_PRECISION} bar: {at_bar:.3f}')
    print('    ' + describe(all_spent, 'spent, all designs'))
    print('    ' + describe(post_spent, 'spent, post-Sobol'))
  p, na, nb = mannwhitney(per_arm['from_scratch']['post_std'], per_arm['meta']['post_std'])
  print(f'\n  post-Sobol loss_std, Mann-Whitney: p={p:.4f}  (n_fs={na}, n_meta={nb})')
  p, na, nb = mannwhitney(per_arm['from_scratch']['post_spent'], per_arm['meta']['post_spent'])
  print(f'  post-Sobol spent,    Mann-Whitney: p={p:.4f}  (n_fs={na}, n_meta={nb})')

  print()
  print('=' * 110)
  print('5. DOES A BIG STEP COME FROM A NOISY OR AN EXPENSIVE DESIGN?')
  print('=' * 110)
  for arm in ARMS:
    sizes, stds, spents = [], [], []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      indices, step_sizes = improving_steps(c['loss'])
      for idx, size in zip(indices, step_sizes):
        if idx < N_INIT:
          continue
        sizes.append(size)
        stds.append(c['loss_std'][idx])
        spents.append(c['spent'][idx])
    sizes = np.asarray(sizes)
    stds = np.asarray(stds)
    spents = np.asarray(spents)
    print(f'  {arm}: n_steps={sizes.size}')
    if sizes.size >= 3:
      r_std = stats.spearmanr(sizes, stds)
      r_spent = stats.spearmanr(sizes, spents)
      print(f'    step size vs loss_std of the causing design: rho={r_std.statistic:+.3f} p={r_std.pvalue:.3f}')
      print(f'    step size vs spent    of the causing design: rho={r_spent.statistic:+.3f} p={r_spent.pvalue:.3f}')

  print()
  print('=' * 110)
  print('6. WINNER\'S CURSE SIGNATURE -- dispersion about the seed\'s own Sobol level')
  print('=' * 110)
  print('  Centering each cell by its own Sobol-block MEDIAN removes the seed effect; the Sobol block')
  print('  is identical across arms, so what remains is the arm. Noise widens BOTH tails; a genuinely')
  print('  better search lowers the bottom without inflating the top.')
  for arm in ARMS:
    centred = []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      level = float(np.median(c['loss'][:N_INIT]))
      centred.extend((c['loss'][N_INIT:] - level).tolist())
    centred = np.asarray(centred)
    per_arm[arm]['centred'] = centred
    above = centred[centred > 0.0]
    below = centred[centred < 0.0]
    print(f'  {arm}:')
    print('    ' + describe(centred, 'centred post-Sobol loss'))
    print(
      f'    upper spread p90-p50 = {np.percentile(centred, 90) - np.percentile(centred, 50):.4f}   '
      f'lower spread p50-p10 = {np.percentile(centred, 50) - np.percentile(centred, 10):.4f}'
    )
    print(
      f'    n above level = {above.size} (mean {above.mean() if above.size > 0 else float("nan"):+.4f}), '
      f'n below = {below.size} (mean {below.mean() if below.size > 0 else float("nan"):+.4f})'
    )
    print(f'    sd = {np.std(centred, ddof=1):.4f}   max = {centred.max():+.4f}   min = {centred.min():+.4f}')
  a = per_arm['from_scratch']['centred']
  b = per_arm['meta']['centred']
  print(f'\n  Levene equal-variance test on the centred losses: p={stats.levene(a, b).pvalue:.4f}')
  print(f'  Ansari-Bradley scale test:                        p={stats.ansari(a, b).pvalue:.4f}')
  p, na, nb = mannwhitney(a, b)
  print(f'  Mann-Whitney on the centred losses (location):    p={p:.4f}  (n_fs={na}, n_meta={nb})')

  print()
  print('=' * 110)
  print('7. NOISE FROM NEAR-DUPLICATE DESIGNS -- |dloss| against permutation-invariant distance')
  print('=' * 110)
  print('  Every within-run pair of designs, pooled over seeds. At small distance the loss difference')
  print('  is noise-dominated, so the small-distance end of each arm is a noise probe that needs no')
  print('  training run.')
  bins = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 10.0]
  for arm in ARMS:
    dist, dloss = [], []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      x = c['x']
      losses = c['loss']
      for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
          dist.append(invariant_distance(x[i], x[j], perms))
          dloss.append(abs(losses[i] - losses[j]))
    dist = np.asarray(dist)
    dloss = np.asarray(dloss)
    print(f'  {arm}: n_pairs={dist.size}, min distance {dist.min():.4f}')
    for lo, hi in zip(bins[:-1], bins[1:]):
      mask = (dist >= lo) & (dist < hi)
      if np.sum(mask) == 0:
        continue
      print(
        f'    d in [{lo:.2f}, {hi:.2f})  n={int(np.sum(mask)):>4}  '
        f'median |dloss| = {np.median(dloss[mask]):.4f}  mean = {dloss[mask].mean():.4f}'
      )

  print()
  print('=' * 110)
  print('8. EXPLORATION -- how far each proposal moves (permutation-invariant)')
  print('=' * 110)
  for arm in ARMS:
    consec, to_incumbent = [], []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      x = c['x']
      losses = c['loss']
      running_argmin = np.minimum.accumulate(losses)
      for i in range(N_INIT, x.shape[0]):
        consec.append(invariant_distance(x[i], x[i - 1], perms))
        incumbent = int(np.argmin(losses[:i]))
        to_incumbent.append(invariant_distance(x[i], x[incumbent], perms))
      del running_argmin
    per_arm[arm]['consec'] = consec
    per_arm[arm]['to_incumbent'] = to_incumbent
    print(f'  {arm}:')
    print('    ' + describe(consec, 'distance to previous proposal'))
    print('    ' + describe(to_incumbent, 'distance to the incumbent'))
  for key, label in (('consec', 'consecutive'), ('to_incumbent', 'to incumbent')):
    p, na, nb = mannwhitney(per_arm['from_scratch'][key], per_arm['meta'][key])
    print(f'  {label:<14} Mann-Whitney: p={p:.4f}  (n_fs={na}, n_meta={nb})')

  print()
  print('=' * 110)
  print('9. THE MEDIAN CURVE ITSELF (what the plot shows)')
  print('=' * 110)
  median_path = os.path.join(root, 'median.json')
  if os.path.exists(median_path):
    with open(median_path) as handle:
      payload = json.load(handle)
    for arm, entry in payload['runs'].items():
      block = entry.get('self_evaluated')
      if block is None:
        continue
      curve = np.asarray(block['median_best_so_far'], dtype=float)
      calls = np.asarray(block['calls'], dtype=float)
      drops = -np.diff(curve)
      drops = drops[drops > 1e-9]
      print(
        f'  {arm}: grid points {curve.size}, distinct decrements {drops.size}, '
        f'total descent {curve[0] - curve[-1]:.4f}'
      )
      print('    ' + describe(drops, 'median-curve decrement sizes'))
      print(f'    start {curve[0]:.4f} -> end {curve[-1]:.4f}, calls {calls[0]:.0f} -> {calls[-1]:.0f}')
  else:
    print(f'  {median_path} not found')

  print()
  print('=' * 110)
  print('10. THE NO-INFORMATION CEILING')
  print('=' * 110)
  print(f'  The loss is MSE in prior-half-range units averaged over {N_TARGETS} targets, so a network that')
  print(f'  learns nothing scores the prior variance of a uniform on [-1, 1] = 1/3 = {CEILING:.4f}. The')
  print('  reported loss is therefore CENSORED FROM ABOVE, which invalidates any "noise widens both')
  print('  tails" reading of section 6: no amount of noise can push a design above the ceiling.')
  pooled = []
  for arm in ARMS:
    losses, post_losses = [], []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      losses.extend(c['loss'].tolist())
      post_losses.extend(c['loss'][N_INIT:].tolist())
    losses = np.asarray(losses)
    post_losses = np.asarray(post_losses)
    pooled.append((int(np.sum(losses >= CEILING_CUT)), losses.size))
    print(f'  {arm}: max reported loss = {losses.max():.4f} (ceiling {CEILING:.4f})')
    print(
      f'    at the ceiling (loss >= {CEILING_CUT}): {int(np.sum(losses >= CEILING_CUT))} of {losses.size} all, '
      f'{int(np.sum(post_losses >= CEILING_CUT))} of {post_losses.size} post-Sobol'
    )
  table = [[pooled[0][0], pooled[0][1] - pooled[0][0]], [pooled[1][0], pooled[1][1] - pooled[1][0]]]
  print(f'  Fisher exact on "at the ceiling" x arm (all designs): p={stats.fisher_exact(table)[1]:.4f}')

  print()
  print('=' * 110)
  print('11. PAIRED SOBOL DIFFERENCES SPLIT BY WHETHER THE SCHEDULE MATCHED')
  print('=' * 110)
  print('  `spent` fixes the window and the number of growth rounds. When the two arms spend the same')
  print('  on the same design they ran the SAME schedule on the SAME data, so their loss difference is')
  print('  the arm alone; when they differ, the convergence rule fired at a different round and the')
  print('  difference confounds arm with schedule.')
  matched, mismatched = [], []
  for seed in SEEDS:
    a = usable.get((seed, 'from_scratch'))
    b = usable.get((seed, 'meta'))
    if a is None or b is None:
      continue
    for i in range(N_INIT):
      delta = float(a['loss'][i] - b['loss'][i])
      if abs(a['spent'][i] - b['spent'][i]) < 0.5:
        matched.append(delta)
      else:
        mismatched.append(delta)
  for name, arr in (('same spent (same schedule)', np.asarray(matched)), ('different spent', np.asarray(mismatched))):
    if arr.size < 2:
      continue
    print(
      f'  {name:<28} n={arr.size:<3} mean={arr.mean():+.4f} sd={np.std(arr, ddof=1):.4f} '
      f'median={np.median(arr):+.4f} max|d|={np.abs(arr).max():.4f}'
    )
    print(f'    {"":<26} |d| < 0.003 in {int(np.sum(np.abs(arr) < 0.003))} of {arr.size}')

  print()
  print('=' * 110)
  print('12. WHERE THE BUDGET GOES -- growth rounds and the cost tail')
  print('=' * 110)
  print(f'  spent = {BASE_SPENT} + rounds * {SPENT_INCREMENT}, so the round count is exactly recoverable.')
  for arm in ARMS:
    rounds, shares = [], []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      rounds.extend(((c['spent'] - BASE_SPENT) / SPENT_INCREMENT).tolist())
      order = np.sort(c['spent'])[::-1]
      shares.append(float(order[:3].sum() / c['spent'].sum()))
    print(f'  {arm}:')
    print('    ' + describe(rounds, 'growth rounds per design'))
    print(
      f'    share of the run budget in its 3 most expensive designs, per seed: '
      f'{[round(s, 3) for s in shares]}  median {np.median(shares):.3f}'
    )
  a_rounds, b_rounds = [], []
  for seed in SEEDS:
    a_rounds.extend((usable[(seed, 'from_scratch')]['spent'] / 1.0).tolist())
    b_rounds.extend((usable[(seed, 'meta')]['spent'] / 1.0).tolist())
  p, na, nb = mannwhitney(a_rounds, b_rounds)
  print(f'  spent over all designs, Mann-Whitney: p={p:.4f}  (n_fs={na}, n_meta={nb})')
  print(
    f'  p90 spent: from_scratch {np.percentile(a_rounds, 90):.0f}, meta {np.percentile(b_rounds, 90):.0f}; '
    f'max: {np.max(a_rounds):.0f} vs {np.max(b_rounds):.0f}'
  )

  print()
  print('=' * 110)
  print('13. EXACT PERMUTATION TEST ON THE MEDIAN-CURVE STATISTICS')
  print('=' * 110)
  print('  The observation was made on the MEDIAN-ACROSS-SEEDS curve, which is a pointwise median of')
  print('  five step functions and is NOT any seed\'s own curve. The arm label is exchangeable within a')
  print('  seed under the null, so swapping the two arms of a seed gives an exact reference set of')
  print(
    f'  2^{len(SEEDS)} = {2 ** len(SEEDS)} relabellings. Two-sided p can therefore never fall below '
    f'{2.0 / 2 ** len(SEEDS):.4f}.'
  )
  cells_for_perm = {}
  for seed in SEEDS:
    a = usable.get((seed, 'from_scratch'))
    b = usable.get((seed, 'meta'))
    if a is None or b is None:
      continue
    cells_for_perm[seed] = (a, b)
  seed_list = list(cells_for_perm)

  def curve_of(cell):
    return np.cumsum(cell['spent']), np.minimum.accumulate(cell['loss'])

  def median_step(curves):
    start = max(float(x[0]) for x, _ in curves)
    grid = np.unique(np.concatenate([x for x, _ in curves]))
    grid = grid[grid >= start]
    stack = np.stack([y[np.searchsorted(x, grid, side='right') - 1] for x, y in curves])
    return grid, np.median(stack, axis=0)

  def statistics_of(assignment):
    left = [curve_of(cells_for_perm[s][assignment[i]]) for i, s in enumerate(seed_list)]
    right = [curve_of(cells_for_perm[s][1 - assignment[i]]) for i, s in enumerate(seed_list)]
    out = []
    for curves in (left, right):
      _, med = median_step(curves)
      drops = -np.diff(med)
      drops = drops[drops > 1e-9]
      out.append((
        drops.size, float(drops.max()) if drops.size > 0 else 0.0, float(med[0] - med[-1]),
        float(np.mean(drops)) if drops.size > 0 else 0.0
      ))
    return out

  observed = statistics_of([0] * len(seed_list))
  names = ('n decrements', 'largest decrement', 'total descent', 'mean decrement')
  print(f'\n  observed median curves ({len(seed_list)} seeds):')
  for k, name in enumerate(names):
    print(
      f'    {name:<20} from_scratch {observed[0][k]:>10.4f}   meta {observed[1][k]:>10.4f}   '
      f'difference {observed[0][k] - observed[1][k]:+10.4f}'
    )
  reference = []
  for bits in itertools.product([0, 1], repeat=len(seed_list)):
    left, right = statistics_of(list(bits))
    reference.append([left[k] - right[k] for k in range(len(names))])
  reference = np.asarray(reference)
  print(f'\n  exact two-sided p over all {reference.shape[0]} relabellings:')
  for k, name in enumerate(names):
    obs = observed[0][k] - observed[1][k]
    p_exact = float(np.mean(np.abs(reference[:, k]) >= abs(obs) - 1e-12))
    print(
      f'    {name:<20} observed difference {obs:+8.4f}   p={p_exact:.4f}   '
      f'reference range [{reference[:, k].min():+.4f}, {reference[:, k].max():+.4f}]'
    )

  print()
  print('  Same test on statistics that ARE per-seed averages, for comparison:')

  def per_seed_statistics(assignment):
    out = []
    for which in (0, 1):
      counts, steps, descents = [], [], []
      for i, s in enumerate(seed_list):
        cell = cells_for_perm[s][assignment[i] if which == 0 else 1 - assignment[i]]
        indices, sizes = improving_steps(cell['loss'])
        sizes_post = sizes[indices >= N_INIT]
        running = np.minimum.accumulate(cell['loss'])
        counts.append(sizes_post.size)
        steps.extend(sizes_post.tolist())
        descents.append(running[N_INIT - 1] - running[-1])
      out.append((
        float(np.median(counts)), float(np.mean(steps)) if len(steps) > 0 else 0.0, float(np.median(descents)),
        float(np.median(counts))
      ))
    return out

  observed_ps = per_seed_statistics([0] * len(seed_list))
  names_ps = ('median n steps', 'mean step size', 'median descent', 'median n steps (dup)')
  reference_ps = []
  for bits in itertools.product([0, 1], repeat=len(seed_list)):
    left, right = per_seed_statistics(list(bits))
    reference_ps.append([left[k] - right[k] for k in range(len(names_ps))])
  reference_ps = np.asarray(reference_ps)
  for k, name in enumerate(names_ps[:3]):
    obs = observed_ps[0][k] - observed_ps[1][k]
    p_exact = float(np.mean(np.abs(reference_ps[:, k]) >= abs(obs) - 1e-12))
    print(
      f'    {name:<20} observed difference {obs:+8.4f}   p={p_exact:.4f}   '
      f'reference range [{reference_ps[:, k].min():+.4f}, {reference_ps[:, k].max():+.4f}]'
    )
  print('\n  distinct values each reference statistic takes (a coarse statistic has no power at all):')
  for k, name in enumerate(names):
    print(f'    {name:<20} {np.unique(np.round(reference[:, k], 6)).size} distinct over 32 relabellings')

  print()
  print('=' * 110)
  print('14. BUDGET CONCENTRATION AND INCUMBENT ISOLATION')
  print('=' * 110)
  shares = {}
  for arm in ARMS:
    shares[arm] = []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      order = np.sort(c['spent'])[::-1]
      shares[arm].append(float(order[:3].sum() / c['spent'].sum()))
  a = np.asarray(shares['from_scratch'])
  b = np.asarray(shares['meta'])
  wins = int(np.sum(a > b))
  sign_p = float(stats.binomtest(wins, a.size, 0.5).pvalue)
  print(f'  share of budget in the 3 most expensive designs, paired by seed:')
  print(f'    from_scratch {[round(float(x), 3) for x in a]}')
  print(f'    meta         {[round(float(x), 3) for x in b]}')
  print(
    f'    from_scratch higher in {wins}/{a.size} seeds, sign test p={sign_p:.4f}, '
    f'wilcoxon p={float(stats.wilcoxon(a, b).pvalue):.4f}'
  )
  for arm in ARMS:
    rounds = []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      rounds.extend(((c['spent'] - BASE_SPENT) / SPENT_INCREMENT).tolist())
    rounds = np.asarray(rounds)
    print(
      f'    {arm:<13} designs needing >= 20 growth rounds: {int(np.sum(rounds >= 20))} of {rounds.size} '
      f'({np.mean(rounds >= 20):.3f})'
    )

  print('\n  incumbent isolation: how far the winning design sits below its own nearest neighbours.')
  print('  A winner selected on noise is an isolated low outlier; a genuine optimum has company.')
  for arm in ARMS:
    gaps, distances, margins = [], [], []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      x = c['x']
      losses = c['loss']
      best = int(np.argmin(losses))
      others = [i for i in range(losses.size) if i != best]
      d = np.array([invariant_distance(x[best], x[i], perms) for i in others])
      order = np.argsort(d)
      nearest = [others[i] for i in order[:3]]
      gaps.append(float(np.min(losses[nearest]) - losses[best]))
      distances.append(float(d[order[0]]))
      margins.append(float(np.sort(losses)[1] - losses[best]))
    print(f'  {arm}:')
    print(
      f'    loss of the best 3 neighbours minus the winner: {[round(g, 4) for g in gaps]}  '
      f'median {np.median(gaps):.4f}'
    )
    print(
      f'    distance to the nearest other design:          {[round(d, 3) for d in distances]}  '
      f'median {np.median(distances):.3f}'
    )
    print(
      f'    margin over the run\'s second-best design:      {[round(m, 4) for m in margins]}  '
      f'median {np.median(margins):.4f}'
    )

  print('\n  share of each seed\'s post-Sobol descent contributed by its single largest step:')
  for arm in ARMS:
    fractions = []
    for seed in SEEDS:
      c = usable.get((seed, arm))
      if c is None:
        continue
      indices, sizes = improving_steps(c['loss'])
      sizes_post = sizes[indices >= N_INIT]
      if sizes_post.size == 0 or sizes_post.sum() <= 0.0:
        fractions.append(float('nan'))
        continue
      fractions.append(float(sizes_post.max() / sizes_post.sum()))
    finite = [f for f in fractions if np.isfinite(f)]
    print(
      f'    {arm:<13} {[None if not np.isfinite(f) else round(f, 3) for f in fractions]}  '
      f'median {np.median(finite):.3f}'
    )

  print()
  print('=' * 110)
  print('15. IS THE MEDIAN CURVE\'S GRANULARITY JUST THE PER-SEED STEP COUNT?')
  print('=' * 110)
  print('  Over the same 32 relabellings, correlate the median curve\'s number of decrements with the')
  print('  plain SUM of the five seeds\' own decrements. A tight relation means the plot\'s granularity')
  print('  is an aggregation consequence of how many steps the seeds took, not a separate phenomenon.')
  pairs = []
  for bits in itertools.product([0, 1], repeat=len(seed_list)):
    for which in (0, 1):
      assignment = [b if which == 0 else 1 - b for b in bits]
      curves, total_steps = [], 0
      for i, s in enumerate(seed_list):
        cell = cells_for_perm[s][assignment[i]]
        curves.append(curve_of(cell))
        total_steps += improving_steps(cell['loss'])[1].size
      _, med = median_step(curves)
      drops = -np.diff(med)
      pairs.append((total_steps, int(np.sum(drops > 1e-9))))
  pairs = np.asarray(pairs, dtype=float)
  rho = stats.spearmanr(pairs[:, 0], pairs[:, 1])
  print(
    f'  n={pairs.shape[0]} relabelled arms;  sum of per-seed decrements vs median-curve decrements: '
    f'rho={rho.statistic:+.3f} p={rho.pvalue:.2e}'
  )
  print(
    f'  observed: from_scratch sum {int(pairs[0, 0])} -> {int(pairs[0, 1])} median decrements;  '
    f'meta sum {int(pairs[1, 0])} -> {int(pairs[1, 1])}'
  )

  print()
  print('=' * 110)
  print('16. DISTANCE-MATCHED NOISE COMPARISON')
  print('=' * 110)
  print('  Section 7 again, restricted to close pairs and checked for a matched distance distribution,')
  print('  so a difference in |dloss| cannot be blamed on one arm simply sampling further apart.')
  for cut in (0.6, 1.0):
    holder = {}
    for arm in ARMS:
      dist, dloss = [], []
      for seed in SEEDS:
        c = usable.get((seed, arm))
        if c is None:
          continue
        x, losses = c['x'], c['loss']
        for i in range(x.shape[0]):
          for j in range(i + 1, x.shape[0]):
            d = invariant_distance(x[i], x[j], perms)
            if d < cut:
              dist.append(d)
              dloss.append(abs(losses[i] - losses[j]))
      holder[arm] = (np.asarray(dist), np.asarray(dloss))
    a_d, a_l = holder['from_scratch']
    b_d, b_l = holder['meta']
    p_dist, _, _ = mannwhitney(a_d, b_d)
    p_loss, na, nb = mannwhitney(a_l, b_l)
    print(f'  pairs closer than {cut}:  n_fs={a_d.size}, n_meta={b_d.size}')
    print(
      f'    distance distributions match?  median {np.median(a_d):.3f} vs {np.median(b_d):.3f}, '
      f'Mann-Whitney p={p_dist:.4f}'
    )
    print(
      f'    |dloss|                        median {np.median(a_l):.4f} vs {np.median(b_l):.4f}, '
      f'Mann-Whitney p={p_loss:.4f}'
    )


if __name__ == '__main__':
  main()
