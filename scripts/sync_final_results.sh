#!/bin/bash
# Mirror the FINAL campaigns' results from AFS for local plotting.
#
#   scripts/sync_final_results.sh [dest]        # default output/final-cern
#
# WHY A SECOND SCRIPT. `sync_ship_results.sh` pulls the older `ship-*` trees, which sit BESIDE the
# repo on AFS. The final campaigns write INSIDE their repos instead -- `detector-opt/output/{angle,
# intersect}` and `detector-linear/output/linear` -- because snakemake resolves its paths relative to
# the working directory. Different source roots, so a different mirror.
#
# ⚠️ RESULTS ONLY. Checkpoints, optimiser state and event pools are NOT pulled: a single SHiP cell's
# checkpoint directory dwarfs every `results.json` in the campaign put together, and nothing local
# reads them. Never `find` over AFS to enumerate a tree -- it times out; the include/exclude filters
# below let rsync do the walking.
#
# The layout is preserved so `plot_median.py` can be pointed straight at a mirrored tree:
#
#   python scripts/plot_median.py <dest>/angle/test --mean
#   python scripts/plot_median.py <dest>/linear/d3n4/test --mean
#
# because that script globs `<tree>/*/*/results.json`, which is exactly `<seed>/<strategy>/`.
set -eu
DEST="${1:-output/final-cern}"
SHIP=lxplus:/afs/cern.ch/work/m/maborisy/detector-opt/output
LIN=lxplus:/afs/cern.ch/work/m/maborisy/detector-linear/output

mkdir -p "$DEST"
FILTERS=(
  --include='*/'
  --include='results.json' --include='done.txt' --include='verified.txt'
  --include='selection.txt' --include='selected.txt'
  --include='verification.json' --include='CAPPED.txt' --include='status.txt'
  --exclude='*'
)

for task in angle intersect; do
  echo "== ${task}"
  rsync -a --prune-empty-dirs "${FILTERS[@]}" "$SHIP/$task/" "$DEST/$task/" || echo "   (absent)"
done

echo "== linear"
rsync -a --prune-empty-dirs "${FILTERS[@]}" "$LIN/linear/" "$DEST/linear/" || echo "   (absent)"

echo
echo "mirrored into $DEST:"
for t in "$DEST"/angle "$DEST"/intersect; do
  [ -d "$t" ] || continue
  printf "  %-28s select %3d  test %3d  verified %3d\n" "$(basename "$t")" \
    "$(find "$t/select" -name results.json 2>/dev/null | wc -l)" \
    "$(find "$t/test" -name results.json 2>/dev/null | wc -l)" \
    "$(find "$t/test" -name verified.txt 2>/dev/null | wc -l)"
done
if [ -d "$DEST/linear" ]; then
  printf "  %-28s select %3d  test %3d  verified %3d\n" linear \
    "$(find "$DEST/linear/select" -name results.json 2>/dev/null | wc -l)" \
    "$(find "$DEST/linear" -path '*/test/*' -name results.json 2>/dev/null | wc -l)" \
    "$(find "$DEST/linear" -path '*/test/*' -name verified.txt 2>/dev/null | wc -l)"
fi
du -sh "$DEST" 2>/dev/null | sed 's/^/  size: /'
