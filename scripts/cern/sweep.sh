#!/bin/bash
# Health sweep for the CERN campaign. Run on lxplus, from the repo root.
#
# THE STALL THRESHOLD IS 90 MINUTES, AND IT IS REGIME-DRIVEN. Measured over completed selection runs,
# median seconds per design is 280 (norewind) to 1180 (sp-03), and sp-03's worst single design took
# 3383 s = 56 min. A 45-minute bound -- which is what this used -- flags healthy shrink-and-perturb
# cells as stalled, because they legitimately go the better part of an hour between designs. 90 min
# clears the worst observed design by 60%.
#
# `condor_q` FAILING IS NOT AN EMPTY QUEUE. It is reported as UNREACHABLE, never as zero jobs.
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90
echo -n "ORCHESTRATOR: "
P=$(ps -u "$USER" -o cmd --no-headers | grep -c "[r]un_campaign.sh")
L=$([ -f .snakemake/locks/0.input.lock ] && echo held || echo NONE)
echo "$P run_campaign proc(s), AFS lock $L"
echo -n "CONDOR: "; condor_q -totals 2>&1 | grep -o "Total for maborisy.*" | head -1 || echo "UNREACHABLE (not an empty queue)"
echo -n "HELD: "; condor_q -constraint "JobStatus==5" -af ClusterId 2>/dev/null | wc -l
echo "PROGRESS: $(find output/*/select -name done.txt 2>/dev/null | wc -l)/90 select  (angle $(find output/angle/select -name done.txt 2>/dev/null|wc -l)/45, intersect $(find output/intersect/select -name done.txt 2>/dev/null|wc -l)/45)"
echo "ERRORS: $(grep -cE '^Error|Exception' logs/campaign.log)   RENEWALS: $(grep -c '\[renew\]' logs/campaign.log)"
# The cells condor is ACTUALLY running, from the +SnakeCell tag the submit script attaches. This is
# what makes the stall test definitive: a partial cell that is not in this list is QUEUED, not stalled.
# Jobs submitted before the tag existed report `unknown` and simply fall back to the mtime heuristic.
# `condor_q -af` prints `undefined` for a classad a job does not carry, so jobs submitted before
# the tag existed must be filtered out too -- leaving them in makes `live` non-empty, matches no
# cell, and silently reports every running cell as queued, disabling the stall test outright.
condor_q -constraint 'JobStatus==2' -af SnakeCell 2>/dev/null \
  | grep -vxE 'unknown|undefined|' > /tmp/sweep_running_cells.txt || true
RUNNING=$(condor_q -constraint 'JobStatus==2' -af ClusterId 2>/dev/null | wc -l)
export RUNNING
python3 - <<'PYEOF'
import json, os, glob, time

# ⛔️ STALL = RUNNING **AND** SILENT. Both halves are needed, and each was learned the hard way:
#   * A partial cell may simply be QUEUED -- its `results.json` is frozen because nothing is executing
#     it, not because anything is wrong. `+SnakeCell` says which cells condor is really running.
#   * The silence bound cannot be a constant. A 5-design SELECTION run finishes a design in ~300 s; a
#     full-budget TEST cell takes 2200-2900 s. So the bound is 3x the CELL'S OWN median `time_s`,
#     floored at 45 min.
# Guessing either half from file mtimes produced three false alarms on 2026-09-02.
try:
    live = {l.strip() for l in open("/tmp/sweep_running_cells.txt") if l.strip()}
except OSError:
    live = set()

now, rows = time.time(), []
for pat in ("output/*/select/*/*/*/results.json", "output/*/test/*/*/results.json"):
    for f in glob.glob(pat):
        d = os.path.dirname(f)
        if os.path.exists(os.path.join(d, "done.txt")):
            continue
        try:
            payload = json.load(open(f))
        except Exception:
            continue
        times = sorted(float(r.get("time_s", 0)) for r in payload["results"] if r.get("time_s"))
        limit = max(2700.0, 3.0 * times[len(times) // 2] if times else 0.0) / 60.0
        rows.append((int((now - os.path.getmtime(f)) / 60), int(limit), len(payload["results"]), d))
rows.sort(reverse=True)

running = int(os.environ.get("RUNNING", "0"))
if live:
    candidates = [r for r in rows if r[3] in live]
    basis = "+SnakeCell tag"
else:
    candidates = rows[-running:] if 0 < running < len(rows) else rows
    basis = "mtime heuristic (jobs predate the tag)"
stale = [r for r in candidates if r[0] > r[1]]
print("PARTIAL %d, running %d (%s), STALE %d" % (len(rows), len(candidates), basis, len(stale)))
for age, limit, n, cell in (stale or candidates)[:3]:
    print("   %-4dmin (limit %4dmin) %2d designs  %s" % (age, limit, n, "/".join(cell.split("/")[1:])))
if len(rows) > len(candidates):
    print("   (%d partial cells are QUEUED, not stalled)" % (len(rows) - len(candidates)))
PYEOF
