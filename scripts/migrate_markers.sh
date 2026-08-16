#!/usr/bin/env bash
# Give a campaign tree built under the OLD scheme the completion markers the new rules expect.
#
#   scripts/migrate_markers.sh <campaign tree> [--dry-run]
#   scripts/migrate_markers.sh output/cloud/enzyme_extremes
#
# WHY IT IS NEEDED, and what happens without it. The `bo` and `verify` rules used to declare
# `results.json` and `verification.png` as their outputs; they now declare `done.txt` and
# `verified.txt`, because snakemake DELETES a rule's declared outputs before running it and the
# trajectory is the one file a resume cannot afford to lose. A tree produced before that change has
# no markers at all, so the new rules see every run as unbuilt and RE-RUN ALL OF THEM -- discarding
# finished trajectories and the event pools they were paid for, silently, at the cost of a full
# campaign.
#
# WHAT COUNTS AS COMPLETE, and the distinction matters:
#
#   done.txt      results.json exists AND its `completed` field is true. A trajectory that stopped
#                 early keeps its rows and gets NO marker, so the new scheme reopens it and carries
#                 on -- which is the behaviour the change exists to provide.
#   verified.txt  verification.png AND verification_comparison.png both exist. Those were the old
#                 rule's declared outputs, so their presence is exactly what "this verification
#                 finished" used to mean. A verification.json with only some points re-scored is
#                 resumable state, not completion, and is deliberately not treated as such.
#
# Markers are stamped with the mtime of the artefact that justifies them, not with now, so snakemake
# still sees a marker as newer than the trajectory it depends on and does not schedule work on the
# strength of a timestamp this script invented.
set -euo pipefail

TREE=${1:?usage: migrate_markers.sh <campaign tree> [--dry-run]}
DRY=${2:-}

[ -d "$TREE" ] || { echo "no such tree: $TREE" >&2; exit 1; }

made_done=0 made_verified=0 skipped_incomplete=0 skipped_unverified=0

for run in "$TREE"/*/*/; do
  [ -d "$run" ] || continue
  results="$run/results.json"
  [ -f "$results" ] || continue

  if python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('completed') is True else 1)" \
       "$results" 2>/dev/null; then
    if [ ! -f "$run/done.txt" ]; then
      if [ "$DRY" = "--dry-run" ]; then echo "would touch $run/done.txt"; else
        touch -r "$results" "$run/done.txt"
      fi
      made_done=$((made_done + 1))
    fi
  else
    skipped_incomplete=$((skipped_incomplete + 1))
  fi

  if [ -f "$run/verification.png" ] && [ -f "$run/verification_comparison.png" ]; then
    if [ ! -f "$run/verified.txt" ]; then
      if [ "$DRY" = "--dry-run" ]; then echo "would touch $run/verified.txt"; else
        touch -r "$run/verification.png" "$run/verified.txt"
      fi
      made_verified=$((made_verified + 1))
    fi
  else
    skipped_unverified=$((skipped_unverified + 1))
  fi
done

echo "$TREE: done.txt +$made_done, verified.txt +$made_verified, " \
     "left unmarked: $skipped_incomplete incomplete run(s), $skipped_unverified unfinished verification(s)"
