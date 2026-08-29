#!/bin/bash
# Release the two HELD CERN clusters once the meta_reinit test cells (cluster 20611373) have all left
# the queue.
#
# SAFETY RULES THIS ENCODES:
#  * An ssh/condor_q FAILURE IS NOT AN EMPTY QUEUE. Every poll must carry a sentinel proving the
#    remote command actually ran; without it the round is skipped, never treated as "done".
#  * Two CONSECUTIVE clean confirmations are required before releasing, so one transient blip cannot
#    trigger it.
#  * Releases exactly once, then exits. `flock` keeps a second copy from ever starting.
set -u
WATCH=20611373
RELEASE="20610527 20610528"
LOG=output/release-watch/watch.log
INTERVAL=600
MAX_ROUNDS=420

exec 9>output/release-watch/.lock
flock -n 9 || { echo "another watcher holds the lock; exiting" >> "$LOG"; exit 0; }

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }
say "watcher started: waiting for cluster $WATCH to drain, then releasing $RELEASE"

clean=0
for round in $(seq 1 "$MAX_ROUNDS"); do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=60 lxplus \
        "condor_q -af ClusterId 2>/dev/null | sort -u | tr '\n' ' '; echo SENTINEL_OK" 2>/dev/null)
  if [[ "$out" != *SENTINEL_OK* ]]; then
    say "round $round: poll FAILED (no sentinel) -- skipping, NOT treating as drained"
    clean=0
    sleep "$INTERVAL"; continue
  fi
  clusters=${out%SENTINEL_OK*}
  if [[ " $clusters " == *" $WATCH "* ]]; then
    say "round $round: cluster $WATCH still in queue [$clusters]"
    clean=0
  else
    clean=$((clean + 1))
    say "round $round: cluster $WATCH absent (confirmation $clean/2) [$clusters]"
    if [ "$clean" -ge 2 ]; then
      say "RELEASING $RELEASE"
      ssh -o BatchMode=yes lxplus "condor_release $RELEASE 2>&1; echo ---; condor_q -totals 2>&1 | grep maborisy" >> "$LOG" 2>&1
      say "release issued; watcher exiting"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
say "MAX_ROUNDS reached without draining -- exiting WITHOUT releasing; jobs remain HELD"
