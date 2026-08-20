#!/usr/bin/env python3
"""Figures for the semi-analytic `extremes` study: one per experiment count, plus a summary.

    python scripts/plot_extremes_analytic.py --run output/extremes-analytic/m1-bo.json \
        --landscape output/extremes-analytic/m1-landscape.json --output output/extremes-analytic/m1.png

Each figure carries three panels, left to right:

  best-so-far    the quantity the criterion is built from -- median over seeds with the
                 interquartile band, both arms, the no-information level 1.0 marked, and the two
                 cutoffs the criterion compares.
  landscape      the loss over a Sobol sample of designs, sorted. This is the calibration evidence:
                 a bowl with structure, not a flat line at the ceiling.
  criterion      (E[loss@first] - E[loss@second]) / 1.0 per arm, observed and re-scored on an
                 independent event stream, with the standard error over seeds.
"""

import argparse
import json
import math
import os

import numpy as np

import matplotlib

matplotlib.use('AGG')

import matplotlib.pyplot as plt

ARM_COLOUR = {'bo': '#1F6FEB', 'random': '#E8710A'}
ARM_LABEL = {'bo': 'Bayesian optimisation', 'random': 'random search'}
GUESS_COLOUR = '#6B6B6B'
INK = '#1A1A1A'
MUTED = '#5C5C5C'


def style(axes):
  axes.spines['top'].set_visible(False)
  axes.spines['right'].set_visible(False)
  axes.spines['left'].set_color('#C9C9C9')
  axes.spines['bottom'].set_color('#C9C9C9')
  axes.tick_params(colors=MUTED, labelsize=9)
  axes.grid(True, axis='y', color='#E6E6E6', linewidth=0.8)
  axes.set_axisbelow(True)
  for label in (axes.xaxis.label, axes.yaxis.label):
    label.set_color(INK)
    label.set_fontsize(10)
  axes.title.set_color(INK)
  axes.title.set_fontsize(11)


def best_so_far(payload, arm):
  return np.minimum.accumulate(np.asarray([run['losses'] for run in payload['runs'][arm]]), axis=1)


def panel_curves(axes, payload):
  first, second = payload['cutoffs'][0], payload['cutoffs'][-1]
  iterations = np.arange(1, payload['n_iterations'] + 1)
  for arm in payload['runs']:
    curves = best_so_far(payload, arm)
    low, mid, high = np.percentile(curves, [25, 50, 75], axis=0)
    axes.fill_between(iterations, low, high, color=ARM_COLOUR[arm], alpha=0.16, linewidth=0)
    axes.plot(iterations, mid, color=ARM_COLOUR[arm], linewidth=2.0, label=ARM_LABEL[arm])
  axes.axhline(1.0, color=GUESS_COLOUR, linewidth=1.2, linestyle=(0, (4, 3)))
  axes.annotate(
    'a guess (1.0)', xy=(iterations[-1], 1.0), xytext=(-4, 4), textcoords='offset points', ha='right', va='bottom', color=MUTED,
    fontsize=8
  )
  for cutoff in (first, second):
    axes.axvline(cutoff, color='#C9C9C9', linewidth=1.0, linestyle=':')
    axes.annotate(
      f'{cutoff}', xy=(cutoff, axes.get_ylim()[0]), xytext=(2, 4), textcoords='offset points', color=MUTED, fontsize=8
    )
  axes.set_xlabel('designs evaluated')
  axes.set_ylabel('best loss so far  (cross-entropy / ln 2)')
  axes.set_title(f'best-so-far, {len(payload["runs"]["bo"])} seeds')
  axes.legend(frameon=False, fontsize=9, labelcolor=INK)


def panel_landscape(axes, landscape, payload):
  losses = np.asarray(
    landscape['losses'][str(landscape['noise'])] if 'noise' in landscape else list(landscape['losses'].values())[0]
  )
  order = np.sort(losses)
  rank = np.arange(1, len(order) + 1) / len(order)
  axes.plot(rank, order, color='#7A5AF8', linewidth=2.0)
  axes.axhline(1.0, color=GUESS_COLOUR, linewidth=1.2, linestyle=(0, (4, 3)))
  axes.set_xlabel('fraction of random designs at or below')
  axes.set_ylabel('loss')
  axes.set_title(f'design landscape, {len(order)} Sobol designs')
  axes.annotate(f'span {order[-1] - order[0]:.3f}', xy=(0.04, 0.06), xycoords='axes fraction', color=MUTED, fontsize=9)


def panel_criterion(axes, payload):
  first, second = payload['cutoffs'][0], payload['cutoffs'][-1]
  rows, offset = [], 0.0
  ticks, labels = [], []
  for arm in payload['runs']:
    for kind, marker in (('observed', 'o'), ('rescored', 's')):
      early = np.asarray([run['cutoffs'][str(first)][kind] for run in payload['runs'][arm]])
      late = np.asarray([run['cutoffs'][str(second)][kind] for run in payload['runs'][arm]])
      gain = early - late
      error = gain.std(ddof=1) / math.sqrt(len(gain))
      axes.errorbar(
        offset, gain.mean(), yerr=error, fmt=marker, color=ARM_COLOUR[arm], markersize=8, capsize=4, elinewidth=1.5,
        markeredgecolor='white', markeredgewidth=1.0
      )
      axes.annotate(
        f'{gain.mean():.3f}', xy=(offset, gain.mean()), xytext=(9, -3), textcoords='offset points', color=INK, fontsize=9
      )
      ticks.append(offset)
      labels.append(f'{ARM_LABEL[arm].split()[0]}\n{kind}')
      rows.append((arm, kind, gain.mean(), error))
      offset += 1.0
  axes.axhline(0.0, color=GUESS_COLOUR, linewidth=1.0)
  axes.set_xticks(ticks)
  axes.set_xticklabels(labels, fontsize=8, color=MUTED)
  axes.set_xlim(-0.6, offset - 0.4)
  axes.set_ylabel(f'(E[loss@{first}] - E[loss@{second}]) / 1.0')
  axes.set_title('criterion, mean +- s.e. over seeds')
  return rows


def figure(payload, landscape, path, title):
  fig, panels = plt.subplots(1, 3, figsize=(14.0, 4.4))
  fig.patch.set_facecolor('#FCFCFB')
  for axes in panels:
    axes.set_facecolor('#FCFCFB')
    style(axes)
  panel_curves(panels[0], payload)
  if landscape is not None:
    panel_landscape(panels[1], landscape, payload)
  else:
    panels[1].set_visible(False)
  panel_criterion(panels[2], payload)
  fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha='left')
  fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
  plt.close(fig)
  return path


def summary(payloads, path):
  fig, axes = plt.subplots(figsize=(6.4, 4.4))
  fig.patch.set_facecolor('#FCFCFB')
  axes.set_facecolor('#FCFCFB')
  style(axes)
  counts = sorted(payloads)
  for arm in ('bo', 'random'):
    values, errors = [], []
    for count in counts:
      payload = payloads[count]
      first, second = payload['cutoffs'][0], payload['cutoffs'][-1]
      early = np.asarray([run['cutoffs'][str(first)]['observed'] for run in payload['runs'][arm]])
      late = np.asarray([run['cutoffs'][str(second)]['observed'] for run in payload['runs'][arm]])
      gain = early - late
      values.append(gain.mean())
      errors.append(gain.std(ddof=1) / math.sqrt(len(gain)))
    axes.errorbar(
      counts, values, yerr=errors, fmt='o-', color=ARM_COLOUR[arm], markersize=8, capsize=4, linewidth=2.0,
      markeredgecolor='white', markeredgewidth=1.0, label=ARM_LABEL[arm]
    )
    for count, value in zip(counts, values):
      axes.annotate(f'{value:.3f}', xy=(count, value), xytext=(6, 5), textcoords='offset points', color=INK, fontsize=9)
  axes.axhline(0.0, color=GUESS_COLOUR, linewidth=1.0)
  axes.set_xticks(counts)
  axes.set_xlabel('experiments per design')
  axes.set_ylabel('(E[loss@10] - E[loss@20]) / 1.0')
  axes.set_title('criterion against batch size')
  axes.legend(frameon=False, fontsize=9, labelcolor=INK)
  fig.tight_layout()
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
  plt.close(fig)
  return path


def summarise_payload(payload):
  first, second = payload['cutoffs'][0], payload['cutoffs'][-1]
  for arm, runs in payload['runs'].items():
    for kind in ('observed', 'rescored'):
      early = np.asarray([run['cutoffs'][str(first)][kind] for run in runs])
      late = np.asarray([run['cutoffs'][str(second)][kind] for run in runs])
      gain = early - late
      print(
        f'  m={payload["n_experiments"]} {arm:8s} {kind:9s} n={len(gain):4d}: '
        f'E[loss@{first}]={early.mean():.4f} E[loss@{second}]={late.mean():.4f} '
        f'criterion={gain.mean():.4f}+-{gain.std(ddof=1) / math.sqrt(len(gain)):.4f} '
        f'P(improve)={float(np.mean(late < early)):.3f}'
      )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run', action='append', required=True, help='a BO json from analytic_extremes_bo.py')
  parser.add_argument('--landscape', action='append', default=[], help='a landscape json, one per --run')
  parser.add_argument('--output-root', required=True)
  parser.add_argument('--summary', default=None)
  arguments = parser.parse_args()

  payloads, landscapes = {}, {}
  for index, path in enumerate(arguments.run):
    with open(path) as handle:
      payload = json.load(handle)
    count = payload['n_experiments']
    if count in payloads:
      for arm, runs in payload['runs'].items():
        payloads[count]['runs'].setdefault(arm, []).extend(runs)
    else:
      payloads[count] = payload
  for path in arguments.landscape:
    with open(path) as handle:
      landscape = json.load(handle)
    landscapes[landscape['n_experiments']] = landscape
  for count in sorted(payloads):
    payload, landscape = payloads[count], landscapes.get(count)
    written = figure(
      payload, landscape, os.path.join(arguments.output_root, f'extremes-m{count}.png'),
      f'`extremes`, {count} experiment(s) per design, {payload["n_measurements"]} read-outs, '
      f'read-out noise {payload["noise"]} mM'
    )
    print(f'wrote {os.path.abspath(written)}')
    summarise_payload(payload)
  if arguments.summary is not None and len(payloads) > 1:
    written = summary(payloads, arguments.summary)
    print(f'wrote {os.path.abspath(written)}')


if __name__ == '__main__':
  main()
