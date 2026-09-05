#!/bin/bash
# snakemake `cluster-generic` status command: running / success / failed for a condor job id.
#
# ⛔️ THE SCHEDD IS NOT THE ONLY SOURCE OF TRUTH, AND MUST NOT BE THE ONLY ONE CONSULTED. A failed
# `condor_q` means "I could not ask", never "the job is gone" -- reading it as an empty queue once
# produced a fabricated loss report here. But answering `running` on every failed query is just as
# wrong in the other direction: on 2026-08-31 the CERN schedd `bigbird11` refused connections for
# over an hour (SECMAN:2007) while a cell ran to completion, and snakemake sat waiting on a job that
# had finished 24 minutes earlier. A campaign would hang there indefinitely.
#
# So the ORDER is: ask the schedd; if it cannot be reached, read the job's own USER LOG, which condor
# writes to shared storage and which records termination independently of the schedd (this is the
# same file `condor_wait` blocks on). Only if BOTH are unavailable do we report `running`, which is
# the safe answer when nothing is known -- it stalls rather than declaring a false completion.
set -eu
ID=$1
CLUSTER=${ID%%.*}
LOG=/afs/cern.ch/work/m/maborisy/detector-linear/logs/condor/${CLUSTER}.log

schedd_state() {
  local q
  q=$(condor_q -json "$ID" 2>/dev/null) || return 1
  python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
print(d[0].get('JobStatus','') if d else 'gone')
" <<< "$q" 2>/dev/null || return 1
}

if ST=$(schedd_state); then
  case "$ST" in
    1|2|5|6|7) echo running; exit 0 ;;                 # idle/running/held/transferring/suspended
    3|4)       ;;                                       # removed / completed -- fall through
  esac
  if H=$(condor_history -limit 1 -json "$ID" 2>/dev/null); then
    CODE=$(python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
print(d[0].get('ExitCode','none') if d else 'none')
" <<< "$H" 2>/dev/null || echo none)
    case "$CODE" in 0) echo success; exit 0 ;; none) ;; *) echo failed; exit 0 ;; esac
  fi
fi

# Schedd unreachable, or it knew nothing useful: the user log is authoritative about termination.
if [ -s "$LOG" ]; then
  if grep -q "^005 " "$LOG"; then                       # 005 = Job terminated
    if grep -q "Normal termination (return value 0)" "$LOG"; then echo success; else echo failed; fi
    exit 0
  fi
  if grep -qE "^009 |^012 " "$LOG"; then echo failed; exit 0; fi   # 009 aborted, 012 held
fi
echo running
