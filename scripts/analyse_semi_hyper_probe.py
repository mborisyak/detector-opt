#!/usr/bin/env python3
"""Read the fixed-design probe cells and answer the two questions a curve can answer.

    python scripts/analyse_semi_hyper_probe.py output/probe-semihyper

Every ``probe.json`` under the root is loaded and keyed by ``(architecture, arm)`` from its path,
with the seed as the repeat. Two read-outs, both per DESIGN, because the designs differ in how much
there is to learn and averaging over them hides that:

* SAMPLES TO A FIXED LOSS LEVEL -- the first epoch whose validation loss is at or below the level,
  converted to optimisation samples and to the detector calls the window held at that point. Levels
  are chosen from the data so that every condition reaches them, since a level one condition never
  reaches gives a censored number rather than a large one.
* LOSS AT A FIXED SAMPLE COUNT -- stated as a MULTIPLE OF ``loss_precision``, the only scale on which
  a loss difference on this task means anything.

The three repeats are reported individually, then as median and full range. Nothing here averages a
difference over repeats before showing the spread: the run-to-run floor on this task is a sizeable
fraction of the effects at stake, so a centre without its band is not a result.
"""

import json
import os
import sys

import numpy as np

LOSS_PRECISION = 0.1


def load(root):
  """``{(architecture, arm): {seed: payload}}`` over every ``probe.json`` under ``root``."""
  cells = {}
  for directory, _sub, files in os.walk(root):
    if 'probe.json' not in files:
      continue
    with open(os.path.join(directory, 'probe.json')) as f:
      payload = json.load(f)
    parts = os.path.relpath(directory, root).split(os.sep)
    if len(parts) < 3:
      continue
    architecture, arm, seed = parts[-3], parts[-2], parts[-1]
    if payload.get('completed') is not True:
      print(f'[skip] {directory}: incomplete', file=sys.stderr)
      continue
    cells.setdefault((architecture, arm), {})[seed] = payload
  return cells


def curve(payload, iteration):
  for design in payload['designs']:
    if int(design['iteration']) == int(iteration):
      return design['curve']
  return None


def crossing(rows, level):
  """The first row whose validation loss is at or below ``level``, or ``None``."""
  for row in rows:
    if row['validation_loss'] <= level:
      return row
  return None


def summarise(values):
  finite = [v for v in values if v is not None]
  if len(finite) < len(values):
    return 'not reached'
  return f'{np.median(finite):.0f} [{min(finite):.0f}, {max(finite):.0f}]'


def main(root):
  cells = load(root)
  if len(cells) == 0:
    raise SystemExit(f'no completed probe.json under {root}')
  conditions = sorted(cells)
  iterations = sorted({int(d['iteration']) for payload in cells[conditions[0]].values() for d in payload['designs']})
  print(f'conditions: {conditions}')
  print(f'repeats: ' + ', '.join(f'{c}={sorted(cells[c])}' for c in conditions))

  for iteration in iterations:
    print(f'\n================ design {iteration} ================')
    reachable, longest = [], 0
    for condition in conditions:
      for payload in cells[condition].values():
        rows = curve(payload, iteration)
        reachable.append(min(row['validation_loss'] for row in rows))
        longest = max(longest, len(rows))
    floor = max(reachable)
    levels = [round(floor + step, 3) for step in (0.02, 0.05, 0.1, 0.2, 0.4)]
    print(f'  every condition reaches {floor:.4f}; levels: {levels}')

    print(f'\n  samples to reach a level (median [min, max] over repeats)')
    header = f'  {"level":>7} ' + ' '.join(f'{"/".join(c):>34}' for c in conditions)
    print(header)
    for level in levels:
      row = f'  {level:7.3f} '
      for condition in conditions:
        values = []
        for payload in cells[condition].values():
          hit = crossing(curve(payload, iteration), level)
          values.append(None if hit is None else hit['samples'])
        row += f'{summarise(values):>35}'
      print(row)

    print(f'\n  detector calls in the window at that level')
    print(header)
    for level in levels:
      row = f'  {level:7.3f} '
      for condition in conditions:
        values = []
        for payload in cells[condition].values():
          hit = crossing(curve(payload, iteration), level)
          values.append(None if hit is None else hit['window_train'] + hit['window_validation'])
        row += f'{summarise(values):>35}'
      print(row)

    print(f'\n  validation loss at a fixed epoch, in units of loss_precision = {LOSS_PRECISION}')
    print(header)
    for epoch in (longest // 4, longest // 2, longest):
      row = f'  ep{epoch:5d} '
      for condition in conditions:
        values = [curve(payload, iteration)[epoch - 1]['validation_loss'] for payload in cells[condition].values()]
        centre, low, high = np.median(values), min(values), max(values)
        row += f'{centre / LOSS_PRECISION:12.2f} [{low / LOSS_PRECISION:.2f}, {high / LOSS_PRECISION:.2f}]'.rjust(35)
      print(row)

    print(f'\n  best validation loss over the whole curve, in units of loss_precision')
    row = '  ' + ' ' * 8
    for condition in conditions:
      values = [min(r['validation_loss'] for r in curve(payload, iteration)) for payload in cells[condition].values()]
      centre, low, high = np.median(values), min(values), max(values)
      row += f'{centre / LOSS_PRECISION:12.2f} [{low / LOSS_PRECISION:.2f}, {high / LOSS_PRECISION:.2f}]'.rjust(35)
    print(row)


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else 'output/probe-semihyper')
