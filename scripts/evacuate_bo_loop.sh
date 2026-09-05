#!/bin/bash
# CONTINUOUS evacuation of `bo` to this workstation.
#
#   setsid nohup scripts/evacuate_bo_loop.sh > logs/evacuate-bo.log 2>&1 < /dev/null &
#
# WHY A LOOP AND NOT A COMMAND. `bo` is a spot instance with a known finite lifetime; four of the five
# hosts to answer to that name have already been destroyed, twice mid-provision. A campaign result that
# exists only on `bo` is not a result. Syncing on request means the mirror is as stale as the last time
# somebody remembered.
#
# FULL TREE, not results-only: the Orbax checkpoints are what a resumed cell restores from, so a
# results-only mirror would force every in-flight cell to restart from scratch on a rebuilt box.
# The whole tree was 89 MB / 2.6 s at 44 cells, so the cost is negligible.
#
# rsync is incremental, so each pass after the first moves only what changed. A pass that fails
# (host gone) is logged and retried -- the loop does not exit, because `bo` may come back.
set -u
INTERVAL="${1:-900}"
DEST=output/final-bo
cd "$(dirname "$0")/.." || exit 90
mkdir -p "$DEST" logs

# ⛔️ ONE COPY ONLY, ENFORCED HERE RATHER THAN BY THE CALLER. Two loops rsyncing into the same tree
# race each other, and the duplicates are invisible: a `ps` filter that tests the wrong field reports
# "not running" for a loop that IS, so every retry silently adds another copy. Four accumulated once
# that way. The guard belongs in the script because that is the only place a mistaken launcher cannot
# skip it.
exec 200>/tmp/evacuate-bo.lock
if ! flock -n 200; then
  echo "[$(date +%H:%M:%S)] another copy holds /tmp/evacuate-bo.lock -- exiting"
  exit 0
fi

while true; do
  START=$(date +%s)
  if rsync -a --timeout=120 bo:/root/detector-opt/output/ "$DEST/" 2>/dev/null; then
    CELLS=$(find "$DEST" -name results.json 2>/dev/null | wc -l)
    DONE=$(find "$DEST" -name done.txt 2>/dev/null | wc -l)
    VER=$(find "$DEST" -name verified.txt 2>/dev/null | wc -l)
    SIZE=$(du -sh "$DEST" 2>/dev/null | cut -f1)
    echo "[$(date '+%H:%M:%S')] ok  cells=$CELLS done=$DONE verified=$VER size=$SIZE ($(( $(date +%s) - START ))s)"
  else
    echo "[$(date '+%H:%M:%S')] FAILED -- bo unreachable or rsync error; mirror left at its last good state"
  fi
  sleep "$INTERVAL"
done
