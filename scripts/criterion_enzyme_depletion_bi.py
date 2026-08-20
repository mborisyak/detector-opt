#!/usr/bin/env python3
"""The doubling criterion for `enzyme_depletion_bi`, pooled over the per-seed-block reports.

    python scripts/criterion_enzyme_depletion_bi.py output/enzyme-depletion-bi/Pfix-m2-0.02-s*.json

WHAT IS COMPUTED, and why it is not the within-run version. The campaign objective is

    P( best-so-far at 2n designs  <  best-so-far at n designs ),  THE TWO FROM DIFFERENT RUNS.

Read off ONE trajectory the comparison is monotone by construction -- best-so-far never rises -- so
the within-run probability is 1 whatever the task does, and carries no information. Taken between
INDEPENDENT runs it is 0.5 exactly when doubling the search buys nothing, which is what a saturated
task fails to beat. It is the two-sample AUC (Mann-Whitney) of `best@2n` against `best@n`.

ESTIMATED over all R(R-1) ORDERED pairs of distinct runs, ties at 0.5. Excluding the diagonal is what
makes it between-run. The standard error is NOT `sqrt(P(1-P)/R)`: that formula is for a proportion of
R independent Bernoulli trials, and this is a U-statistic averaging over R(R-1) dependent pairs. A
delete-one JACKKNIFE over RUNS is used instead -- one run is one draw -- and the naive figure is
printed beside it so the difference stays visible.

MEASURED, the naive figure is roughly TWICE the jackknife (0.035-0.043 against 0.020-0.024 at R = 128),
i.e. CONSERVATIVE rather than optimistic. That is the expected direction and not a reassurance to skip
the jackknife with: averaging over all pairs uses the sample better than R independent comparisons
would, so the U-statistic is more efficient than a proportion of the same R. Reporting the naive
number as the error bar would therefore understate the evidence, and it is the WRONG bar either way
because it does not know the pairs share runs.

AGAINST THE NULL, ALWAYS. A candidate task is interesting only if BO EXCEEDS the random-search arm at
the SAME setting; a high P with an equally high null says the doubling helped a search that is not
searching. The two arms share seeds, so the difference is jackknifed PAIRED, over the same delete-one
blocks, rather than by differencing two independent error bars.

The reports are pooled by `(noise, seed)`; a seed appearing in two blocks is counted ONCE, because
the statistic is a count over independent runs and a silently duplicated run shrinks the error bar on
nothing.
"""
import argparse
import glob
import json
import os

import numpy as np

PALETTE = {'bo': '#2a78d6', 'random': '#eb6834'}


def load(paths):
  """`{noise: {arm: (n_seeds, n_checkpoints) re-evaluated incumbent loss}}` plus the checkpoints.

  THE QUANTITY IS `reevaluated`, NOT `observed` AND NOT `incumbents`. `observed` is the objective on
  the block the search itself minimised, so its best-so-far carries the winner's curse -- the argmin
  of a noisy curve is biased low, and the bias grows with the number of designs, which is exactly the
  axis the criterion varies. `reevaluated` re-scores the incumbent DESIGN on an independent event
  block and is free of it. `incumbents` is not a loss at all: it is the incumbent design in scaled
  coordinates, `2 m` numbers wide, which for `m = 2` is the same width as a four-checkpoint list and
  will silently read as one."""
  by_noise, checkpoints, settings = {}, None, {}
  for path in sorted({p for pattern in paths for p in glob.glob(pattern)}):
    with open(path) as handle:
      report = json.load(handle)
    noise = float(report['measurement_noise'])
    if checkpoints is None:
      checkpoints = [int(c) for c in report['checkpoints']]
    elif [int(c) for c in report['checkpoints']] != checkpoints:
      raise ValueError(f'{path} has checkpoints {report["checkpoints"]}, expected {checkpoints}')
    start = int(report['seed_start'])
    seeds = start + np.arange(int(report['n_seeds']))
    bucket = by_noise.setdefault(noise, {})
    settings.setdefault(
      noise, {
        'n_experiments': report['n_experiments'],
        'n_grid': report['n_grid'],
        'duration': report['duration'],
        'n_measurements': report['n_measurements'],
        'kernel': report['kernel'],
        'concentration_a_bounds': report['concentration_a_bounds'],
        'concentration_b_bounds': report['concentration_b_bounds'],
        'no_information_loss': report['no_information_loss']
      }
    )
    for arm, block in report['arms'].items():
      reevaluated = np.stack([np.asarray(block['reevaluated'][str(c)], float) for c in checkpoints], axis=-1)
      store = bucket.setdefault(arm, {})
      for seed, row in zip(seeds, reevaluated):
        store[int(seed)] = row
  out = {}
  for noise, arms in by_noise.items():
    seeds = sorted(set.intersection(*[set(v) for v in arms.values()]))
    out[noise] = {arm: np.stack([arms[arm][s] for s in seeds]) for arm in arms}
    out[noise]['_seeds'] = np.asarray(seeds)
  return out, checkpoints, settings


def auc(at_2n, at_n):
  """`P(best@2n < best@n)` over ORDERED pairs of DISTINCT runs, ties at 0.5."""
  n = len(at_2n)
  comparison = (at_2n[:, None] < at_n[None, :]).astype(float) + 0.5 * (at_2n[:, None] == at_n[None, :])
  np.fill_diagonal(comparison, 0.0)
  return float(comparison.sum() / (n * (n - 1)))


def jackknife(statistic, n):
  """`(value, standard error)` of `statistic(keep_mask)` by delete-one over runs."""
  everything = np.ones(n, bool)
  full = statistic(everything)
  partial = np.empty(n)
  for k in range(n):
    keep = everything.copy()
    keep[k] = False
    partial[k] = statistic(keep)
  return full, float(np.sqrt((n - 1) / n * np.sum((partial - partial.mean())**2)))


def analyse(curves, checkpoints):
  """Every (n -> 2n) checkpoint pair, per arm, plus the paired BO-minus-null difference."""
  rows = []
  for lower, upper in [(i, j) for i in range(len(checkpoints)) for j in range(len(checkpoints))
                       if checkpoints[j] == 2 * checkpoints[i]]:
    arms = {}
    for arm in ('bo', 'random'):
      values = curves[arm]
      n_seeds = values.shape[0]
      point, error = jackknife(lambda keep, v=values: auc(v[keep, upper], v[keep, lower]), n_seeds)
      arms[arm] = {
        'probability': point,
        'jackknife_se': error,
        'naive_se': float(np.sqrt(point * (1.0 - point) / n_seeds)),
        'n_seeds': int(n_seeds),
        'median_at_n': float(np.median(values[:, lower])),
        'median_at_2n': float(np.median(values[:, upper]))
      }

    def difference(keep):
      return auc(curves['bo'][keep, upper],
                 curves['bo'][keep, lower]) - auc(curves['random'][keep, upper], curves['random'][keep, lower])

    delta, delta_error = jackknife(difference, curves['bo'].shape[0])
    rows.append({
      'n': checkpoints[lower],
      'two_n': checkpoints[upper],
      'arms': arms,
      'excess_over_null': delta,
      'excess_jackknife_se': delta_error
    })
  return rows


def plot(results, checkpoints, settings, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  noises = sorted(results)
  figure, axes = plt.subplots(2, len(noises), figsize=(6.4 * len(noises), 9.6), squeeze=False)
  for column, noise in enumerate(noises):
    rows = results[noise]['rows']
    curves = results[noise]['curves']
    labels = [f'{r["n"]}->{r["two_n"]}' for r in rows]
    positions = np.arange(len(rows))
    upper = axes[0][column]
    for offset, arm in ((-0.11, 'bo'), (0.11, 'random')):
      point = np.array([r['arms'][arm]['probability'] for r in rows])
      error = np.array([r['arms'][arm]['jackknife_se'] for r in rows])
      upper.errorbar(
        positions + offset, point, yerr=error, fmt='o', color=PALETTE[arm], markersize=8, capsize=4, linewidth=2, label={
          'bo': 'BO',
          'random': 'random search (null)'
        }[arm], zorder=3
      )
      for x, y in zip(positions + offset, point):
        upper.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 11), ha='center', fontsize=8, color='#3d3d3a')
    upper.axhline(0.75, color='#4a3aa7', linewidth=2, linestyle='--', zorder=2)
    upper.annotate(
      'acceptance 0.75', (len(rows) - 0.5, 0.75), textcoords='offset points', xytext=(-4, 5), ha='right', fontsize=9,
      color='#4a3aa7'
    )
    upper.axhline(0.5, color='#8a8a85', linewidth=1.5, linestyle=':', zorder=2)
    upper.annotate(
      '0.5 = doubling buys nothing', (len(rows) - 0.5, 0.5), textcoords='offset points', xytext=(-4, 5), ha='right', fontsize=9,
      color='#6a6a65'
    )
    upper.set_xticks(positions)
    upper.set_xticklabels(labels)
    upper.set_ylim(0.30, 1.0)
    upper.set_xlim(-0.5, len(rows) - 0.5)
    upper.set_xlabel('designs, n -> 2n')
    upper.set_ylabel('P(best@2n < best@n), independent runs')
    box = settings[noise]
    upper.set_title(
      f'noise {noise:g} mM, m = {box["n_experiments"]}, {rows[0]["arms"]["bo"]["n_seeds"]} seeds/arm\n'
      f'error bars: delete-one jackknife over seeds', fontsize=10
    )
    upper.legend(loc='upper right', fontsize=9, framealpha=0.95)
    upper.grid(axis='y', color='#e6e6e2', linewidth=0.8, zorder=0)
    upper.set_axisbelow(True)

    lower = axes[1][column]
    pair = next((i, j) for i in range(len(checkpoints)) for j in range(len(checkpoints))
                if checkpoints[j] == 2 * checkpoints[i] and checkpoints[i] == max(r['n'] for r in rows))
    for arm in ('bo', 'random'):
      values = curves[arm]
      lower.scatter(
        values[:, pair[0]], values[:, pair[1]], s=26, color=PALETTE[arm], alpha=0.75, edgecolors='#fcfcfb', linewidths=0.6,
        label={
          'bo': 'BO',
          'random': 'random search (null)'
        }[arm], zorder=3
      )
    shown = np.concatenate([curves[a][:, i] for a in ('bo', 'random') for i in pair])
    limits = [float(shown.min()) * 0.95, float(shown.max()) * 1.05]
    lower.plot(limits, limits, color='#8a8a85', linewidth=1.5, linestyle=':', zorder=2)
    lower.annotate(
      'y = x (no gain)', (limits[1], limits[1]), textcoords='offset points', xytext=(-6, -14), ha='right', fontsize=9,
      color='#6a6a65'
    )
    lower.set_xlim(*limits)
    lower.set_ylim(*limits)
    lower.set_xlabel(f'best-so-far at n = {checkpoints[pair[0]]} designs')
    lower.set_ylabel(f'best-so-far at 2n = {checkpoints[pair[1]]} designs')
    lower.set_title(
      f'per-seed spread at the widest pair, noise {noise:g}\n'
      f'the criterion is an ALL-PAIRS statistic over these, not the paired diagonal', fontsize=10
    )
    lower.legend(loc='upper left', fontsize=9, framealpha=0.95)
    lower.grid(color='#e6e6e2', linewidth=0.8, zorder=0)
    lower.set_axisbelow(True)

  figure.suptitle(
    'Doubling criterion on the CORRECTED product-form two-substrate task: BO against its own random-search null', fontsize=13
  )
  figure.text(
    0.5, 0.005,
    'SETTLES: whether doubling the design budget still pays on the corrected rate law, and whether BO beats random at the '
    'same setting.\nFALSIFIED BY: BO within one jackknife SE of the null (search is not searching), or P at 0.5 (saturated). '
    'Within-run reading is 1.0 by construction and is not shown.', ha='center', fontsize=9, color='#3d3d3a'
  )
  figure.tight_layout(rect=(0, 0.035, 1, 0.96))
  figure.savefig(path, dpi=150)
  print(f'wrote {path}')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('reports', nargs='+', help='the per-seed-block json reports to pool')
  parser.add_argument('--output', default='output/enzyme-depletion-bi/criterion.json')
  parser.add_argument('--plot', default='output/enzyme-depletion-bi/figures/criterion.png')
  arguments = parser.parse_args()

  curves, checkpoints, settings = load(arguments.reports)
  results, record = {}, {}
  for noise in sorted(curves):
    arms = {k: v for k, v in curves[noise].items() if not k.startswith('_')}
    rows = analyse(arms, checkpoints)
    results[noise] = {'rows': rows, 'curves': arms}
    record[str(noise)] = {'rows': rows, 'n_seeds': int(arms['bo'].shape[0]), 'settings': settings[noise]}
    print(f'=== noise {noise:g} mM, {arms["bo"].shape[0]} pooled seeds, checkpoints {checkpoints} ===')
    print(f'    {"pair":>9} {"arm":>8} {"P(2n<n)":>9} {"jack SE":>9} {"naive SE":>9} {"median@n":>10} {"median@2n":>10}')
    for row in rows:
      pair = '{}->{}'.format(row['n'], row['two_n'])
      for arm in ('bo', 'random'):
        block = row['arms'][arm]
        print(
          f'    {pair:>9} {arm:>8} {block["probability"]:9.3f} {block["jackknife_se"]:9.3f} '
          f'{block["naive_se"]:9.3f} {block["median_at_n"]:10.4f} {block["median_at_2n"]:10.4f}'
        )
      significant = abs(row['excess_over_null']) > 2.0 * row['excess_jackknife_se']
      print(
        f'    {"":>9} {"BO-null":>8} {row["excess_over_null"]:+9.3f} {row["excess_jackknife_se"]:9.3f}  '
        f'{"EXCEEDS the null" if significant and row["excess_over_null"] > 0 else "not separated from the null"}'
      )
    print()

  os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
  with open(arguments.output, 'w') as handle:
    json.dump(record, handle, indent=2)
  print(f'wrote {arguments.output}')
  os.makedirs(os.path.dirname(arguments.plot), exist_ok=True)
  plot(results, checkpoints, settings, arguments.plot)


if __name__ == '__main__':
  main()
