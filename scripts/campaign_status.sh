#!/usr/bin/env bash
# ONE heartbeat across every campaign now running -- local SLURM and CERN HTCondor -- in one screen.
#
#   scripts/campaign_status.sh [--cern]
#
# `--cern` adds the HTCondor half, which costs an ssh round trip and an `aklog`; without it the script
# is local-only and instant. `scripts/monitor_bo.sh` is the equivalent for the preemptible cloud box
# and does NOT apply here: it probes a remote host for a snakemake driver and an MPS socket.
#
# EACH CHECK IS HERE BECAUSE IT FAILED SILENTLY AT LEAST ONCE TONIGHT:
#
#   AllocMem, not CPUAlloc. Two 12000 MB GPU cells take 24000 of this node's 26000, so ANY concurrent
#   CPU job at 1800 MB is the difference between two shards busy and one idle. Cores are never the
#   binding constraint here; memory is. A campaign can look healthy with a shard sitting idle.
#
#   HELD JOBS ARE REPORTED LOUDLY. `scontrol hold` is how a cheap campaign is let past an expensive
#   one, and a held job stays held forever if whoever held it goes away.
#
#   AFS TOKEN. `condor_q` keeps working after the token lapses while every read under /afs fails with
#   `Permission denied`, so results look frozen while the jobs are fine. Every remote command is
#   prefixed with `aklog`, which needs no password while the Kerberos ticket is valid.
#
#   THE ERROR SCAN IS SCOPED TO THE LIVE TREES. A bare `output/*/logs/*.log` glob matches every old
#   campaign on disk, so a real failure arrives buried in stale hits from trees nobody is watching.
#
#   QUERY FAILURE IS NOT AN EMPTY RESULT. A `condor_q` that errors prints nothing, which is
#   indistinguishable from "the cluster finished" unless the exit status is checked. Acting on that
#   confusion nearly put two processes on one output directory.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== $(date '+%F %H:%M:%S')  local SLURM ==="
squeue -h -o '%j %t %r' 2>/dev/null | awk '
  {p2=substr($1,1,2); p1=substr($1,1,1)
   k = (p2=="D2") ? "linear-d2n3" : (p2=="D4") ? "linear-d4n5" : (p2=="X4") ? "analytic" : (p1=="E") ? "enzyme" : "other"
   c[k" "$2]++; if ($3=="JobHeldUser") held[k]++}
  END {for (x in c) printf "  %-22s %s\n", x, c[x]
       for (h in held) printf "  ⚠️  %s: %d HELD\n", h, held[h]}' | sort
HELDIDS=$(squeue -h -o '%i %t %r' 2>/dev/null | awk '$3=="JobHeldUser"{print $1}' | tr '\n' ' ')
[ -n "$HELDIDS" ] && echo "  ⚠️  release them with:  scontrol release $HELDIDS"

scontrol show node machine 2>/dev/null | tr ' ' '\n' | grep -E '^(CPUAlloc|CPUTot|AllocMem|RealMemory)=' | sed 's/^/  /' | tr '\n' ' '; echo
FREE=$(scontrol show node machine 2>/dev/null | tr ' ' '\n' | grep -oP 'RealMemory=\K\d+')
USED=$(scontrol show node machine 2>/dev/null | tr ' ' '\n' | grep -oP 'AllocMem=\K\d+')
[ -n "${FREE:-}" ] && [ -n "${USED:-}" ] && echo "  memory headroom: $((FREE-USED)) MB  (a 12000 MB GPU cell needs 12000)"

echo
echo "=== campaign trees ==="
for T in linear-d2n3-10seed linear-d4n5-5seed enzyme-extremes-5seed ship-addr-prec1e2 ship-addr-prec2e2; do
  [ -d "output/$T" ] || continue
  python3 - "$T" <<'PY'
import json, glob, sys, os
t = sys.argv[1]
ps = sorted(glob.glob(f'output/{t}/*/*/results.json'))
done = 0; rows = []
for p in ps:
    try: d = json.load(open(p))
    except Exception: continue
    b = d['config']['training']['budget']
    done += 1 if d.get('completed') else 0
    rows.append((p.split('/')[2][:6] + '/' + p.split('/')[3][:4], len(d['results']), 100 * d['detector_calls_used'] // b, d['best_loss']))
print(f'  {t}: {done} completed / {len(ps)} with results')
for name, n, pct, best in rows[:6]:
    print(f'      {name:14s} {n:3d} designs  {pct:3d}%  best {best:.5f}')
if len(rows) > 6: print(f'      ... and {len(rows)-6} more')
PY
done

echo
echo "=== error signatures (logs) ==="
LIVE="linear-d2n3-10seed linear-d4n5-5seed enzyme-extremes-5seed ship-addr-prec1e2 ship-addr-prec2e2"
LOGS=$(for T in $LIVE; do ls output/$T/logs/*.log 2>/dev/null; ls output/$T/*/*/run.log 2>/dev/null; done)
HITS=$([ -n "$LOGS" ] && grep -ilE 'traceback|cuda error|out of memory|oom-kill|killed|xid [0-9]' $LOGS 2>/dev/null | head -5)
[ -z "$HITS" ] && echo "  none" || echo "$HITS" | sed 's/^/  ⚠️  /'

if [ "${1:-}" = "--cern" ]; then
  echo
  echo "=== CERN HTCondor ==="
  OUT=$(timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=25 lxplus '
    aklog 2>/dev/null
    condor_q -af ClusterId JobStatus 2>/dev/null | sort | uniq -c
    echo "RC=$?"
    for D in ship-addr-prec1e2 ship-addr-prec2e2; do
      echo "TREE=$D banked=$(ls /afs/cern.ch/work/m/maborisy/$D/*/*/results.json 2>/dev/null | wc -l) done=$(grep -l "^status=COMPLETED" /afs/cern.ch/work/m/maborisy/$D/*/*/status.txt 2>/dev/null | wc -l) capped=$(ls /afs/cern.ch/work/m/maborisy/$D/*/*/CAPPED.txt 2>/dev/null | wc -l)"
    done')
  if [ $? -ne 0 ] || ! echo "$OUT" | grep -q 'RC=0'; then
    echo "  ⚠️  QUERY FAILED -- state UNKNOWN, not empty. Do not act on this reading."
  else
    echo "$OUT" | grep -v 'RC=' | sed 's/^/  /'
  fi
fi
