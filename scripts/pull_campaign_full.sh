#!/usr/bin/env bash
# Pull a FINISHED campaign in full from `bo` -- everything, not just the results.
#
#   scripts/pull_campaign_full.sh [task] [prefix]
#   scripts/pull_campaign_full.sh enzyme_extremes output/cloud
#
# WHY THIS IS SEPARATE FROM THE HEARTBEAT'S PULL. `scripts/monitor_bo.sh` mirrors only `*.json` and
# `*.png` because it runs every 15 minutes and the tree is dominated by `trainer.npz` -- the event
# pool, 1.17 GB PER RUN, rewritten at every design boundary. Mirroring that continuously would keep a
# 5.7 MB/s transatlantic link saturated for the whole campaign moving state that is not a result.
#
# Once a campaign is FINISHED the calculus inverts: nothing is being rewritten, the pools are stable,
# and a complete local copy is what survives the box being deleted. MEASURED on the first campaign:
# 24 GB total, 23.4 GB of it the 20 pools, ~5 MB of checkpoints per run. At the measured link speed
# that is of order an hour, which is why it runs in the background and under its own lock rather than
# inside a heartbeat beat.
#
# It is safe to re-run: rsync transfers only what changed, so an interrupted pull resumes. `--delete`
# is NOT used -- a truncated transfer must never remove local results. `--partial` keeps the progress
# of a file cut off mid-transfer.
#
# The local path MIRRORS the remote one exactly (`<prefix>/<task>`), which is why the campaign runs
# under a prefix in the first place: the workstation runs its own campaigns over the same task and
# the same seeds at the bare `output/` prefix, and one tree must never land on the other.
set -euo pipefail

TASK=${1:-enzyme_extremes}
PREFIX=${2:-output/cloud}
HOST=${HOST:-bo}
# The checkout on bo that owns this campaign -- one per campaign, since snakemake locks its working
# directory and two campaigns cannot share a tree.
REMOTE_ROOT=${REMOTE_ROOT:-detector-opt}
LOCAL_ROOT=${LOCAL_ROOT:-/home/max/dev/detector-opt}
CAMPAIGN=$PREFIX/$TASK
LOG=$LOCAL_ROOT/logs/pull-full-$TASK.log
LOCK=/home/max/.pull-campaign-full.lock

mkdir -p "$LOCAL_ROOT/$CAMPAIGN" "$(dirname "$LOG")"

if [ "${PULL_FULL_LOCKED:-0}" != "1" ]; then
  export PULL_FULL_LOCKED=1
  exec flock -n "$LOCK" "$0" "$@"
fi

{
  echo "=== $(date -u +%FT%TZ) full pull start: $HOST:$REMOTE_ROOT/$CAMPAIGN -> $LOCAL_ROOT/$CAMPAIGN"
  status=0
  timeout 21600 rsync -a --partial --info=stats2 \
    "$HOST:$REMOTE_ROOT/$CAMPAIGN/" "$LOCAL_ROOT/$CAMPAIGN/" || status=$?
  echo "=== $(date -u +%FT%TZ) full pull finished, rsync exit $status"
  du -sh "$LOCAL_ROOT/$CAMPAIGN"
  exit $status
} >> "$LOG" 2>&1
