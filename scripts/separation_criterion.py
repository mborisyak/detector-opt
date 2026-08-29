"""Judge a campaign against the SEPARATION criterion of `docs/CHECKLIST.md`.

    python scripts/separation_criterion.py output/extremes-p1e2-5seed \
        --budget 1310720 --exclude 122012491/continue 499913522/meta

The checklist states it as: meta should separate by CONVERGENCE, not by the final point -- given
enough budget every strategy arrives at the same place, and the question is how fast. Formally:

    there is a contiguous interval, of area > 10% of the budget, of the mean best-so-far curve that
    is below every other curve taking into account errors of the mean estimates
    (curve1 - curve2 > sigma1 + sigma2)

so an arm separates at a point when, against EVERY other arm, the gap between the two means exceeds
the sum of their standard errors, and it passes when the LONGEST CONTIGUOUS run of such points spans
more than 10% of the budget. A run that is merely long in total but broken into fragments does not
pass; the criterion says contiguous and that is enforced.

DATA PATH. Cells are read through `plot_median.load`, the same function the figures use, so a number
here and a picture there cannot disagree. Only PAIRED seeds are used -- seeds complete in every arm --
because a mean over a different seed set per arm compares arms and seed luck at once.

WHAT IT WILL NOT DO. It does not pick the interval, the grid, or the arm after seeing the curves, and
it reports the longest contiguous run for EVERY arm, not just the winner, so a marginal pass cannot be
presented as a clean one. It reports the run in absolute calls as well as a percentage, because "10%
of the budget" is meaningless without saying what the budget was.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from plot_median import load

ARMS = ('from_scratch', 'continue', 'closest', 'meta')


def curves(data, seeds, grid):
  """``{arm: (mean, sem)}`` over ``seeds``, each best-so-far carried forward onto ``grid``."""
  out = {}
  for arm, cells in data.items():
    stack = []
    for seed in seeds:
      calls, best = cells[seed]
      stack.append(np.interp(grid, calls, best, left=best[0], right=best[-1]))
    stack = np.asarray(stack)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else np.zeros(len(grid))
    out[arm] = (stack.mean(axis=0), sem)
  return out


def longest_run(flags):
  """``(length, start, stop)`` of the longest contiguous True run; ``stop`` exclusive."""
  best = (0, 0, 0)
  start = None
  for i, flag in enumerate(list(flags) + [False]):
    if flag and start is None:
      start = i
    elif not flag and start is not None:
      if i - start > best[0]:
        best = (i - start, start, i)
      start = None
  return best


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('tree')
  parser.add_argument('--budget', type=int, required=True)
  parser.add_argument('--exclude', nargs='*', default=(), help='cells to drop, as seed/arm (CAPPED cells)')
  parser.add_argument('--points', type=int, default=2048, help='grid points across the budget')
  parser.add_argument('--fraction', type=float, default=0.10, help='required contiguous fraction of the budget')
  arguments = parser.parse_args()

  data = load(arguments.tree, partial=False, exclude=set(arguments.exclude))
  present = [arm for arm in ARMS if arm in data]
  if len(present) < 2:
    print(f'{arguments.tree}: only {len(present)} arm(s) present; nothing to separate')
    return
  seeds = sorted(set.intersection(*[set(data[arm]) for arm in present]))
  if len(seeds) < 2:
    print(f'{arguments.tree}: {len(seeds)} paired seed(s); a standard error needs at least 2')
    return

  grid = np.linspace(0.0, float(arguments.budget), arguments.points)
  fitted = curves(data, seeds, grid)
  step = grid[1] - grid[0]

  print(f'{arguments.tree}')
  print(f'  paired seeds : {len(seeds)}  {seeds}')
  print(f'  budget       : {arguments.budget:,}   grid {arguments.points} points, {step:,.0f} calls apart')
  print(
    f'  requirement  : one contiguous interval > {100 * arguments.fraction:.0f}% of budget '
    f'= {arguments.fraction * arguments.budget:,.0f} calls\n'
  )

  verdict = False
  for arm in present:
    mean, sem = fitted[arm]
    separated = np.ones(len(grid), bool)
    for other in present:
      if other == arm:
        continue
      other_mean, other_sem = fitted[other]
      separated &= (other_mean - mean) > (sem + other_sem)
    length, start, stop = longest_run(separated)
    span = length * step
    passed = span > arguments.fraction * arguments.budget
    verdict |= passed
    where = f'{grid[start]:,.0f}-{grid[stop - 1]:,.0f} calls' if length > 0 else 'nowhere'
    print(
      f'  {arm:<13} separated at {100 * separated.mean():5.1f}% of grid points | '
      f'longest contiguous {span:>10,.0f} calls = {100 * span / arguments.budget:5.2f}% | {where}'
      f'{"  <-- PASSES" if passed else ""}'
    )

  print(f'\n  VERDICT: {"SEPARATES" if verdict else "DOES NOT SEPARATE"}')


if __name__ == '__main__':
  main()
