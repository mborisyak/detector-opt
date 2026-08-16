#!/usr/bin/env python3
"""Aggregate a scripts/probe_mps.sh sweep into the table that picks the concurrency K.

    python scripts/probe_mps_report.py output/mps-probe/on [output/mps-probe/off ...]

Per level N it reports the wall clock of the level, the spread of the workers' own totals, the
aggregate throughput N * t_1 / t_N, the detector calls per second summed over workers, the worst
worker's peak device memory, and the mean GPU utilisation and board power sampled during the level.

READ IT AS FOLLOWS, and these readings are the whole point of the probe:

  * throughput FLAT in N            -> the processes are time-slicing. On an `off` sweep that is
                                       expected; on an `on` sweep it means the MPS server was not
                                       actually used and the numbers are void.
  * throughput saturating at N = 6  -> 24 cores / 4 per job. The host CPU is the ceiling and MPS
                                       settings are the wrong knob; give jobs fewer threads instead.
  * throughput saturating above 6 with GPU utilisation near its ceiling
                                    -> the device is the ceiling, which is the intended state.
  * peak device memory * K near the card's capacity
                                    -> memory binds first; K must come down regardless of throughput.

The per-worker totals INCLUDE process start-up and JIT compilation, which is honest for this
question: a campaign job pays them too, once per design boundary crossing at worst. `build_s` is
reported separately so a level whose slowdown is entirely compilation contention is visible as such.
"""

import glob
import json
import os
import sys


def level_number(path):
  return int(os.path.basename(path)[1:])


def read_workers(level_dir):
  records = []
  for path in sorted(glob.glob(os.path.join(level_dir, "w*", "probe.json"))):
    with open(path) as handle:
      records.append(json.load(handle))
  return records


def read_gpu(level_dir):
  path = os.path.join(level_dir, "gpu.csv")
  if not os.path.exists(path):
    return None, None
  utilisation, power = [], []
  with open(path) as handle:
    for line in handle:
      fields = [field.strip() for field in line.split(",")]
      if len(fields) < 5:
        continue
      try:
        utilisation.append(float(fields[1].rstrip(" %")))
        power.append(float(fields[3].rstrip(" W")))
      except ValueError:
        continue
  if len(utilisation) == 0:
    return None, None
  return sum(utilisation) / len(utilisation), sum(power) / len(power)


def read_wall(sweep_dir):
  path = os.path.join(sweep_dir, "wall.txt")
  wall = {}
  if not os.path.exists(path):
    return wall
  with open(path) as handle:
    for line in handle:
      fields = line.split()
      if len(fields) >= 2:
        wall[int(fields[0])] = {"seconds": int(fields[1]), "failed": int(fields[2]) if len(fields) > 2 else 0}
  return wall


def report(sweep_dir):
  wall = read_wall(sweep_dir)
  levels = sorted(glob.glob(os.path.join(sweep_dir, "n*")), key=level_number)
  print(f"\n=== {sweep_dir} ===")
  header = (f"{'N':>3} {'wall_s':>7} {'worker_s med':>12} {'min':>7} {'max':>7} {'build_s':>8} "
            f"{'thru':>6} {'calls/s':>9} {'peak GiB':>9} {'GPU %':>6} {'W':>6} {'fail':>5}")
  print(header)
  print("-" * len(header))
  baseline = None
  rows = []
  for level_dir in levels:
    number = level_number(level_dir)
    workers = read_workers(level_dir)
    if len(workers) == 0:
      print(f"{number:>3} {'-':>7} {'NO WORKER OUTPUT -- every worker in this level failed':>12}")
      continue
    totals = sorted(worker["total_s"] for worker in workers)
    median = totals[len(totals) // 2]
    builds = sorted(worker["build_s"] for worker in workers)
    calls = sum(worker["spent"] for worker in workers)
    peak = max(worker["memory"].get("peak_bytes_in_use", 0) for worker in workers) / 2**30
    utilisation, power = read_gpu(level_dir)
    if number == 1:
      baseline = median
    throughput = (number * baseline / median) if baseline is not None else float("nan")
    level_wall = wall.get(number, {})
    seconds = level_wall.get("seconds", int(max(totals)))
    rows.append({
      "n": number, "wall_s": seconds, "median_s": median, "throughput": throughput,
      "peak_gib": peak, "workers": len(workers), "failed": level_wall.get("failed", 0),
      "gpu_percent": utilisation, "power_w": power, "calls_per_s": calls / seconds if seconds > 0 else 0.0,
    })
    print(f"{number:>3} {seconds:>7} {median:>12.1f} {totals[0]:>7.1f} {totals[-1]:>7.1f} "
          f"{builds[len(builds) // 2]:>8.1f} {throughput:>6.2f} {calls / max(seconds, 1):>9.0f} "
          f"{peak:>9.2f} {utilisation if utilisation is not None else float('nan'):>6.1f} "
          f"{power if power is not None else float('nan'):>6.0f} {level_wall.get('failed', 0):>5}")
  if len(rows) > 0:
    best = max(rows, key=lambda row: row["throughput"])
    print(f"\nbest aggregate throughput: N = {best['n']} at {best['throughput']:.2f}x "
          f"({best['peak_gib']:.2f} GiB peak per worker, {best['peak_gib'] * best['n']:.1f} GiB at that N)")
  return rows


if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  for directory in sys.argv[1:]:
    report(directory)
