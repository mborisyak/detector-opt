#!/usr/bin/env bash
# ONE heartbeat of the cloud campaign: probe `bo`, mirror its results here, print a one-line verdict.
#
#   scripts/monitor_bo.sh [task] [prefix]
#
# The campaign lives at `<prefix>/<task>` on BOTH machines -- `output/cloud/enzyme_extremes` by
# default -- so the mirror lands at the same path here as it occupies there and there is nothing to
# translate. The prefix is what keeps it clear of the workstation's own campaigns, which run the same
# task with the same seeds at the bare `output/` prefix.
#
# The box is PREEMPTIBLE, so "is it still there and still working" is a question that has to be asked
# on a schedule rather than assumed. This script is the sensor; deciding what to do about a bad
# reading is the caller's job (see scripts/launch_bo_campaign.sh for the recovery command).
#
# WHAT IT CHECKS, and why each one can fail independently:
#
#   REACHABLE   ssh answers at all. A preempted instance fails here.
#   DRIVER      a snakemake process is alive. The disk is persistent and bo.py resumes from
#               partial.json, so a preemption costs the design in flight -- but a dead DRIVER stops
#               submitting new work while leaving every finished artefact in place, which looks
#               exactly like a healthy idle campaign. This is the failure to watch for.
#   MPS         the MPS control socket exists. The daemon does not survive a reboot, and without it
#               concurrent jobs TIME-SLICE instead of co-running: no error, no warning, roughly the
#               single-job throughput spread over K jobs. Silent, and expensive.
#   CLIENTS     how many processes nvidia-smi shows with Type `M+C`, i.e. are actually ATTACHED to
#               the MPS server. This is stronger than the socket test and catches the other half of
#               the same failure: a daemon that is up while a job bypassed it (almost always a
#               missing CUDA_MPS_PIPE_DIRECTORY in that job's environment) shows plain `C` and
#               time-slices against the rest. Fewer clients than running GPU jobs is MPS_BYPASSED.
#   PROGRESS    the newest mtime across EVERY PHASE's artefact -- partial.json and results.json for
#               the BO runs, verification.json for the re-scoring, comparison.txt and median.json for
#               the roll-ups. Compared against the previous beat's, which is kept in the state file,
#               so "driver alive but wedged" is distinguishable from "driver alive and working".
#
#               ⚠️ IT MUST SPAN EVERY PHASE OR IT FIRES FALSELY AT THE END. Watching only the BO
#               files, as it first did, reports STALLED from the moment the last BO run finishes:
#               those mtimes can never advance again, while verifications and comparisons are still
#               landing. That is a false alarm on a healthy campaign, and it fires precisely when
#               attention is least warranted.
#   DONE        how many of the expected results.json / verification.png exist.
#
# It then rsyncs bo's output tree here (no --delete, so nothing local is ever removed by a partial
# transfer) and appends the reading to logs/monitor-bo.log.
#
# ⚠️ THE PULL IS RESULTS ONLY -- `*.json` and `*.png`, never `checkpoints/`, never the `.npz` state.
# bo's disk is PERSISTENT, so resume state has no reason to cross the Atlantic: it lives in the only
# place a resume would ever read it. What the pull would otherwise carry is ruinous. `trainer.npz` is
# the event POOL, 1.2 GB PER RUN, rewritten at EVERY design boundary -- across 20 runs that is ~24 GB
# re-transferred every time it changes, and on this 5.7 MB/s link one full pull takes over an hour,
# so beats would overlap and the link would stay saturated forever with data that is not a result.
# MEASURED: dropping it took the mirror from 2.8 GB to 7.7 MB and a beat from minutes to 5 s.
#
# Every campaign artefact has a JSON form, so `*.json` loses nothing: results.json, partial.json,
# convergence.json, convergence_all.json (the data behind comparison.txt) and median.json. The PNGs
# come too because they are small and are what a human actually looks at.
#
# ⚠️ A STALE `partial.json` IS DELETED ONCE `results.json` LANDS BESIDE IT. `bo.py` removes the
# partial when a run completes, but the pull deliberately has no `--delete` (a truncated transfer
# must never remove local results), so the removal does not propagate and the mirror ends up holding
# BOTH files for a finished run. Anything that sums over `*.json` then counts that run twice -- which
# it did here, inflating spent-budget from 44% to 74% and turning a 2-3 h ETA into 0.8 h. Only the
# partial is removed, and only when the finished file it duplicates is present.
#
# THE WHOLE BEAT IS UNDER `flock -n`. A beat that runs long (a big first pull) must not have the next
# scheduled beat start beside it -- two rsyncs over one tree fight for the link and for the state
# file. A beat that cannot take the lock exits quietly rather than queueing.
#
# EVERY PROCESS PATTERN IS BRACKET-TRICKED (`[s]cripts/bo.py`). The probe runs over ssh, so the
# pattern is inside the remote command's own command line and an unbracketed `pgrep -f` matches
# ITSELF -- which read as a live campaign on a box where nothing was running at all.
#
# Exit status is 0 when everything reads healthy and 1 when any check fails, so a caller can branch
# on it without parsing the line.
set -uo pipefail

if [ "${MONITOR_BO_LOCKED:-0}" != "1" ]; then
  export MONITOR_BO_LOCKED=1
  exec flock -n /home/max/.monitor-bo.lock "$0" "$@"
fi

TASK=${1:-enzyme_extremes}
PREFIX=${2:-output/cloud}
HOST=${HOST:-bo}
LOCAL_ROOT=${LOCAL_ROOT:-/home/max/dev/detector-opt}
MIRROR=${MIRROR:-$LOCAL_ROOT/$PREFIX}
CAMPAIGN=$PREFIX/$TASK
STATE=$MIRROR/.monitor-state
LOG=$LOCAL_ROOT/logs/monitor-bo.log
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$MIRROR" "$(dirname "$LOG")"

emit() {
  echo "$STAMP $*" | tee -a "$LOG"
}

REMOTE=$(timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "
  set -u
  campaign=$CAMPAIGN
  driver=\$(pgrep -f '[s]nakemake.*Snakefile.cloud' 2>/dev/null | wc -l)
  boruns=\$(pgrep -f '[s]cripts/bo.py' 2>/dev/null | wc -l)
  verifies=\$(pgrep -f '[s]cripts/verify_trajectory.py' 2>/dev/null | wc -l)
  mps=0; [ -S \"\$HOME/.mps/control\" ] && mps=1
  results=\$(find \$HOME/detector-opt/\$campaign -name results.json 2>/dev/null | wc -l)
  verifications=\$(find \$HOME/detector-opt/\$campaign -name verification.png 2>/dev/null | wc -l)
  comparisons=\$(find \$HOME/detector-opt/\$campaign -name comparison.txt 2>/dev/null | wc -l)
  newest=\$(find \$HOME/detector-opt/\$campaign \\( -name 'partial.json' -o -name 'results.json' \
            -o -name 'verification.json' -o -name 'comparison.txt' -o -name 'median.json' \\) \
            -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  newest=\${newest:-0}
  gpu=\$(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
  clients=\$(nvidia-smi 2>/dev/null | grep -c 'M+C')
  up=\$(cut -d. -f1 /proc/uptime)
  echo \"\$driver \$boruns \$verifies \$mps \$results \$verifications \$comparisons \${newest%.*} \$up \$gpu \$clients\"
" 2>/dev/null)

if [ -z "$REMOTE" ]; then
  emit "UNREACHABLE host=$HOST -- preempted, rebooting, or the network is down"
  exit 1
fi

read -r DRIVER BORUNS VERIFIES MPS RESULTS VERIFICATIONS COMPARISONS NEWEST UPTIME GPU CLIENTS <<< "$REMOTE"

PREVIOUS_NEWEST=0
PREVIOUS_UPTIME=0
if [ -f "$STATE" ]; then
  read -r PREVIOUS_NEWEST PREVIOUS_UPTIME < "$STATE" 2>/dev/null || true
fi
printf '%s %s\n' "$NEWEST" "$UPTIME" > "$STATE"

timeout 900 rsync -a --no-i-r --prune-empty-dirs \
  --exclude 'checkpoints/' \
  --include '*/' --include '*.json' --include '*.png' --exclude '*' \
  "$HOST:detector-opt/$PREFIX/" "$MIRROR/" >/dev/null 2>&1
PULL=$?

for finished in "$MIRROR/$TASK"/*/*/results.json; do
  [ -e "$finished" ] || continue
  rm -f "$(dirname "$finished")/partial.json"
done

STATUS="OK"
NOTES=""
if [ "${UPTIME:-0}" -lt "${PREVIOUS_UPTIME:-0}" ]; then
  STATUS="REBOOTED"
  NOTES="$NOTES uptime went backwards (${PREVIOUS_UPTIME}s -> ${UPTIME}s): the box was preempted and came back."
fi
if [ "${DRIVER:-0}" -eq 0 ]; then
  STATUS="NO_DRIVER"
  NOTES="$NOTES no snakemake driver: nothing is being submitted."
fi
if [ "${MPS:-0}" -eq 0 ]; then
  [ "$STATUS" = "OK" ] && STATUS="NO_MPS"
  NOTES="$NOTES MPS control socket absent: concurrent jobs are TIME-SLICING."
fi
RUNNING=$(( ${BORUNS:-0} + ${VERIFIES:-0} ))
if [ "${MPS:-0}" -eq 1 ] && [ "$RUNNING" -gt 0 ] && [ "${CLIENTS:-0}" -lt "$RUNNING" ]; then
  [ "$STATUS" = "OK" ] && STATUS="MPS_BYPASSED"
  NOTES="$NOTES only $CLIENTS of $RUNNING GPU jobs show as MPS clients (Type M+C): the rest bypassed the server and are TIME-SLICING."
fi
if [ "${NEWEST:-0}" -le "${PREVIOUS_NEWEST:-0}" ] && [ "${PREVIOUS_NEWEST:-0}" -gt 0 ]; then
  [ "$STATUS" = "OK" ] && STATUS="STALLED"
  NOTES="$NOTES no output file advanced since the previous beat."
fi
if [ "$PULL" -ne 0 ]; then
  NOTES="$NOTES (rsync pull exited $PULL)"
fi

emit "$STATUS driver=$DRIVER bo=$BORUNS verify=$VERIFIES mps=$MPS clients=$CLIENTS results=$RESULTS verifications=$VERIFICATIONS comparisons=$COMPARISONS uptime=${UPTIME}s gpu=$GPU$NOTES"

[ "$STATUS" = "OK" ]
