#!/bin/bash
# Mirror the `bo` campaign's results to this workstation.
#
#   scripts/sync_bo_results.sh [dest]        # default output/final-bo
#
# ⚠️ RESULTS ONLY -- THIS MIRROR CANNOT BE RESTORED FROM. Checkpoints are not pulled, and
# `verify_trajectory.py` restores the network each design was REPORTED with from
# `checkpoints/design_NNNN`. A cell rebuilt from this mirror therefore has results.json and done.txt
# but no checkpoints, and its verification dies with
#
#   FileNotFoundError: no checkpoint at .../checkpoints/design_0000
#
# THIS HAS ALREADY HAPPENED. Cells relayed between sites with these filters looked complete -- done.txt
# present, designs banked -- and only failed hours later when verification reached them; they had to be
# recomputed. Use this script for PLOTTING ONLY. To move work between machines, or to protect a spot
# instance, copy the whole tree (`scripts/evacuate_bo_loop.sh`).
#
# WHY THIS EXISTS. `bo` is a name that has pointed at four hosts; three were preemptible and were
# destroyed, twice in one afternoon. The current instance is persistent, but a site whose results live
# only on itself is one provider action away from losing them. Run this on a schedule, not once.
#
# The layout is preserved so `plot_median.py` can be pointed straight at a mirrored tree:
#
#   python scripts/plot_median.py <dest>/intersect/test --mean
set -eu
DEST="${1:-output/final-bo}"
SRC=bo:/root/detector-opt/output

mkdir -p "$DEST"
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='results.json' --include='done.txt' --include='verified.txt' \
  --include='verification.json' --include='selection.txt' --include='selected.txt' \
  --include='status.txt' --include='CAPPED.txt' \
  --exclude='*' \
  "$SRC/" "$DEST/"

echo "mirrored into $DEST:"
for task in angle intersect linear; do
  [ -d "$DEST/$task" ] || continue
  printf "  %-12s select %3s done / %3s verified   test %3s done / %3s verified\n" "$task" \
    "$(find "$DEST/$task" -path '*/select/*' -name done.txt 2>/dev/null | wc -l)" \
    "$(find "$DEST/$task" -path '*/select/*' -name verified.txt 2>/dev/null | wc -l)" \
    "$(find "$DEST/$task" -path '*/test/*' -name done.txt 2>/dev/null | wc -l)" \
    "$(find "$DEST/$task" -path '*/test/*' -name verified.txt 2>/dev/null | wc -l)"
done
du -sh "$DEST"
