"""Section-5 margin measurement, per run and pooled BY ARM across seeds.

Reads the authoritative archive (output/slurm-log-archive), parses every [converged/bayes]
line, and reports mean/max of diff+err against the bar, plus how many designs sit exactly at
the bar. Sums are computed in integer units of 1e-4 because diff/err are printed at 4 dp, so
"exactly at the bar" is an exact integer test, not a float comparison.

126382657/continue is carried as a MAX-ONLY row: its job log was deleted before the repo
archiver started, so it has no per-design table and cannot contribute a mean.
"""
import glob
import json
import os
import re

ARCHIVE = "/home/max/dev/detector-opt/output/slurm-log-archive"
OUTPUT = "/home/max/dev/detector-opt/output/enzyme_extremes"
PATTERN = re.compile(
    r"\[converged/bayes\].*?diff=([0-9.]+)\s+err=([0-9.]+)\s+prec=([0-9.]+).*?window=(\d+)")
ARMS = ["from_scratch", "continue", "closest", "meta"]
MAX_ONLY = {("126382657", "continue"): {"designs": 15, "max_units": 100}}


def load_runs():
  runs = []
  for path in sorted(glob.glob(os.path.join(ARCHIVE, "rule_bo_output_enzyme_extremes_*.log"))):
    stem = os.path.basename(path)[len("rule_bo_output_enzyme_extremes_"):-len(".log")]
    seed, rest = stem.split("_", 1)
    arm = rest.rsplit("_", 1)[0]
    sums, bar = [], None
    with open(path) as handle:
      for line in handle:
        found = PATTERN.search(line)
        if found is None:
          continue
        diff, err, prec, _window = found.groups()
        sums.append(round(float(diff) * 1e4) + round(float(err) * 1e4))
        bar = round(float(prec) * 1e4)
    runs.append({"seed": seed, "arm": arm, "sums": sums, "bar": bar, "path": path})
  return runs


def design_count(seed, arm):
  for name in ("results.json", "partial.json"):
    path = os.path.join(OUTPUT, seed, arm, name)
    if os.path.exists(path):
      with open(path) as handle:
        data = json.load(handle)
      return len(data.get("results", [])), name == "results.json"
  return 0, False


runs = load_runs()
bar = next((r["bar"] for r in runs if r["bar"] is not None), 100)

print("PER RUN (sums in units of 1e-4; bar=%d)" % bar)
for run in runs:
  designs, finished = design_count(run["seed"], run["arm"])
  sums = run["sums"]
  if len(sums) == 0:
    print("  %s/%-12s converged=0 designs=%d" % (run["seed"], run["arm"], designs))
    continue
  complete = "complete" if finished and len(sums) >= designs else "in-flight"
  print("  %s/%-12s n=%2d designs=%2d mean=%.4f (%3.0f%%) max=%.4f (%3.0f%%) at-bar=%d [%s]" %
        (run["seed"], run["arm"], len(sums), designs,
         sum(sums) / len(sums) / 1e4, 100 * (sum(sums) / len(sums)) / bar,
         max(sums) / 1e4, 100 * max(sums) / bar,
         sum(1 for s in sums if s == bar), complete))

for (seed, arm), info in MAX_ONLY.items():
  if not any(r["seed"] == seed and r["arm"] == arm for r in runs):
    print("  %s/%-12s n=MAX-ONLY designs=%2d mean=n/a max=%.4f (%3.0f%%) at-bar>=1 [log lost]" %
          (seed, arm, info["designs"], info["max_units"] / 1e4, 100 * info["max_units"] / bar))

print("\nBY ARM (pooled across seeds; means over per-design tables only)")
for arm in ARMS:
  pooled, designs, complete_runs, total_runs = [], 0, 0, 0
  for run in runs:
    if run["arm"] != arm:
      continue
    total_runs += 1
    pooled.extend(run["sums"])
    n_designs, finished = design_count(run["seed"], run["arm"])
    designs += n_designs
    if finished and len(run["sums"]) >= n_designs:
      complete_runs += 1
  extra = ""
  for (seed, a), info in MAX_ONLY.items():
    if a == arm and not any(r["seed"] == seed and r["arm"] == a for r in runs):
      designs += info["designs"]
      total_runs += 1
      extra = " (+1 max-only run, %d designs, max=%.4f, excluded from mean)" % (
          info["designs"], info["max_units"] / 1e4)
  if len(pooled) == 0:
    print("  %-12s no data%s" % (arm, extra))
    continue
  print("  %-12s runs=%d (%d complete) designs=%d n=%d mean=%.4f (%3.0f%%) max=%.4f (%3.0f%%) at-bar=%d%s" %
        (arm, total_runs, complete_runs, designs, len(pooled),
         sum(pooled) / len(pooled) / 1e4, 100 * (sum(pooled) / len(pooled)) / bar,
         max(pooled) / 1e4, 100 * max(pooled) / bar,
         sum(1 for s in pooled if s == bar), extra))
