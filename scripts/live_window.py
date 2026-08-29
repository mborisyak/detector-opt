"""Growth window of the design a cell is training RIGHT NOW, against `iteration_limit`.

The window of the LIVE design is the only early warning for a capped cell that can work: statistics
over converged designs cannot warn, because a design that caps never emits a `[converged/bayes]` line
and is absent from them by construction. This reads the design currently training instead.

Two ways to get it wrong, both MEASURED on real trees rather than imagined:

* `grep '[grow]' | tail -1` is WRONG. After a design converges its last growth line stays the last one
  in the file until the next design grows, so the reading is that design's EXIT window -- a healthy,
  completed design. 6 of 36 extremes logs were in that state at one instant, one of them at 80.5% of
  the cap, above the alert threshold.
* Requiring only "grow lines after the last `[iter N] training...`" is ALSO wrong, and fails on
  FINISHED cells. A run ends `[iter N] training...` -> `[budget] pool exhausted; finishing BO`, so its
  last design was abandoned mid-growth and has no `[converged/bayes]`: the naive live test calls it
  live. 29 of 36 completed extremes cells reported a phantom live window this way.

So the state is reported explicitly rather than inferred from a bare number:

  live      -- a design is growing now; this is the only state the cap warning applies to
  between   -- last design converged, next has not started growing
  finished  -- the run ended (`pool exhausted` / `Best loss:`); the window is the ABANDONED design's
  capped    -- the run died on the `iteration_limit` cap; the window is where it died
"""
import argparse
import os
import re

ITER = re.compile(r'^\[iter \d+\] training\.\.\.', re.M)
GROW = re.compile(r'\[grow\] window -> (\d+)')
EXIT = re.compile(r'\[converged/bayes\]')
CAP = re.compile(r'did not reach precision within iteration_limit')
DONE = re.compile(r'\[budget\] pool exhausted|^Best loss:', re.M)


def live_window(path):
  """``(window, grow_steps, state)``; ``window`` is ``None`` when the trailing design never grew.

  The STATE is decided first and the window second. A run can end before its abandoned final design
  emits any `[grow]` line -- 7 of 36 completed extremes cells did -- and reporting that as "nothing has
  grown yet" would put a FINISHED cell in the same bucket as one that has not started.
  """
  with open(path, errors='ignore') as handle:
    text = handle.read()
  starts = list(ITER.finditer(text))
  if len(starts) == 0:
    return None
  tail = text[starts[-1].end():]
  if CAP.search(tail) is not None:
    state = 'capped'
  elif DONE.search(tail) is not None:
    state = 'finished'
  elif EXIT.search(tail) is not None:
    state = 'between'
  else:
    state = 'live'
  grows = GROW.findall(tail)
  window = int(grows[-1]) if len(grows) > 0 else None
  return window, len(grows), state


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('logs', nargs='+')
  parser.add_argument('--iteration-limit', type=int, required=True)
  parser.add_argument('--n-increment', type=int, required=True)
  parser.add_argument('--max-age', type=int, default=600, help='skip logs older than this many seconds')
  parser.add_argument('--threshold', type=float, default=80.0, help='percent of the cap worth reporting')
  parser.add_argument('--all-states', action='store_true', help='print finished and capped cells too')
  arguments = parser.parse_args()

  now = int(os.path.getmtime(max(arguments.logs, key=os.path.getmtime)))
  for path in sorted(arguments.logs, key=os.path.getmtime, reverse=True):
    if now - int(os.path.getmtime(path)) > arguments.max_age:
      continue
    result = live_window(path)
    name = os.path.basename(path).replace('.log', '')
    if result is None:
      print(f'  {name:<34} no design has started')
      continue
    window, steps, state = result
    if state != 'live' and not arguments.all_states:
      print(f'  {name:<34} {state} (no live design)')
      continue
    if window is None:
      print(f'  {name:<34} {state:<8} trailing design never grew')
      continue
    percent = 100.0 * window / arguments.iteration_limit
    left = (arguments.iteration_limit - window) // arguments.n_increment
    flag = '  <-- ABOVE THRESHOLD' if state == 'live' and percent >= arguments.threshold else ''
    print(f'  {name:<34} {state:<8} window {window:>7,} = {percent:5.1f}% of cap, {steps} steps, {left} left{flag}')


if __name__ == '__main__':
  main()
