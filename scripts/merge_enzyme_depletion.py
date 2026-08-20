#!/usr/bin/env python3
"""Merge the per-seed-block outputs of `scripts/enzyme_depletion_bo.py` into one campaign file.

    python scripts/merge_enzyme_depletion.py --parts output/enzyme-depletion/m4_s*.json \
        --output output/enzyme-depletion/m4.json

The campaign is split into short SLURM chunks so it backfills on a busy node; the seeds are
disjoint blocks of one sequence, so merging is concatenation. Everything outside the per-seed lists
must agree between the parts, and the script fails rather than silently merging two different tunes.
"""
import argparse
import glob
import json

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--parts', nargs='+', required=True)
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()
  paths = sorted(p for pattern in arguments.parts for p in glob.glob(pattern))
  if len(paths) == 0:
    raise SystemExit(f'no parts matched {arguments.parts}')

  merged, seeds = None, []
  for path in paths:
    with open(path) as handle:
      part = json.load(handle)
    if merged is None:
      # A COPY, never the first part itself. Emptying the skeleton in place clears the very arrays
      # about to be read back out of it, so that block vanishes from the merge without a word -- it
      # cost three campaigns a seed block before it was caught by counting the rows.
      merged = {key: value for key, value in part.items() if key != 'arms'}
      merged['n_seeds'] = 0
      merged['arms'] = {
        arm: {
          'observed': [],
          'incumbents': [],
          'reevaluated': {
            k: []
            for k in data['reevaluated']
          }
        }
        for arm, data in part['arms'].items()
      }
    for key in ('n_experiments', 'measurement_noise', 'n_iterations', 'n_events', 'kernel', 'checkpoints'):
      if part[key] != merged[key]:
        raise SystemExit(f'{path}: {key} = {part[key]!r} but the first part has {merged[key]!r}')
    if part['seed_start'] in seeds:
      raise SystemExit(f'{path}: seed block {part["seed_start"]} appears twice')
    seeds.append(part['seed_start'])
    merged['n_seeds'] += part['n_seeds']
    for arm, data in part['arms'].items():
      merged['arms'][arm]['observed'].extend(data['observed'])
      merged['arms'][arm]['incumbents'].extend(data['incumbents'])
      for key, values in data['reevaluated'].items():
        merged['arms'][arm]['reevaluated'][key].extend(values)
  merged['seed_start'] = min(seeds)
  with open(arguments.output, 'w') as handle:
    json.dump(merged, handle)

  guess = merged['no_information_loss']
  first, last = merged['checkpoints'][0], merged['checkpoints'][-1]
  print(
    f'{arguments.output}: {merged["n_seeds"]} seeds from {len(paths)} parts, '
    f'{merged["n_experiments"]} experiment(s), noise {merged["measurement_noise"]}'
  )
  for arm, data in merged['arms'].items():
    early = np.asarray(data['reevaluated'][str(first)], float)
    late = np.asarray(data['reevaluated'][str(last)], float)
    print(
      f'  {arm}: E[loss @ {first}] = {early.mean():.4f}, E[loss @ {last}] = {late.mean():.4f}, '
      f'criterion = {(early.mean() - late.mean()) / guess:.4f} '
      f'+- {np.std(early - late) / np.sqrt(early.size) / guess:.4f}'
    )


if __name__ == '__main__':
  main()
