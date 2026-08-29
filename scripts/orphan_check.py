"""Cells that LOOK alive but are not scheduled anywhere -- a silently dead cell.

A fault grep over the logs only finds failures that WRITE something. Every failure in this campaign so
far has been the `iteration_limit` cap, which raises a self-documenting RuntimeError, so a grep looks
reliable because of what has happened to fail rather than because of what it measures. A cell killed
externally -- OOM-kill, wall-clock limit, node event -- can write nothing at all, and a staleness filter
then ages it out of the live listing, so it is invisible in both places at once.

This checks an invariant instead: a cell whose log says a design is `live` must have a RUNNING job.
It does not depend on the dead process having spoken.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_window import live_window


def running_cells(pattern):
  """``{name}`` of SLURM jobs currently RUNNING whose name matches ``pattern``."""
  out = subprocess.run(['squeue', '-h', '-t', 'RUNNING', '-o', '%j'], capture_output=True, text=True).stdout
  return {line.strip() for line in out.splitlines() if pattern in line}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('logs', nargs='+')
  parser.add_argument('--job-prefix', required=True, help='SLURM job-name prefix, e.g. ep1e2-')
  arguments = parser.parse_args()

  live = []
  for path in arguments.logs:
    result = live_window(path)
    if result is not None and result[2] == 'live':
      live.append(os.path.basename(path).replace('.log', ''))
  jobs = running_cells(arguments.job_prefix)
  orphans = [cell for cell in live if arguments.job_prefix + cell not in jobs]

  print(f'  live cells {len(live)}   RUNNING jobs {len(jobs)}   orphans {len(orphans) if orphans else "none"}')
  for cell in orphans:
    print(f'    ORPHAN {cell} -- log says a design is live, no RUNNING job. Died silently.')
  return 1 if len(orphans) > 0 else 0


if __name__ == '__main__':
  raise SystemExit(main())
