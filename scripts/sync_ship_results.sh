#!/bin/bash
# Mirror the plot-relevant slice of the CERN SHiP trees to this workstation.
#
# Pulls ONLY results.json / status.txt / CAPPED.txt -- never checkpoints (optimizer.npz,
# trainer.npz) or run.log, which are orders of magnitude larger and no plot reads them. Plotting
# then runs locally with no AFS round-trip.
#
# Usage: scripts/sync_ship_results.sh [dest]   (default output/ship-cern)
set -u
DEST="${1:-output/ship-cern}"
SRC=lxplus:/afs/cern.ch/work/m/maborisy
TREES="ship-addr-prec1e2 ship-addr-prec1e2-2M ship-addr-prec1e2-nl ship-addr-w2x ship-addr-w15x-deep
       ship-addr-b3taper ship-addr-norewind ship-addr-w2x-2m ship-addr-w2x-2m-norewind ship-addr-b3taper-2m
       ship-intersect-w2x ship-angle-w2x
       ship-intersect-w2x-2m ship-angle-w2x-2m ship-angle-w2x-3m"
# RETIRED, not pulled. Each is fully mirrored under output/ship-cern/ and carries a RETIRED.txt with
# its result, so a pull only costs time; re-add a name here if that stops being true.
#
#   ship-intersect-w2x-2m-reveal   COMPLETE 5/5   the reveal control, concluded
#   ship-angle-w2x-3m-reveal       COMPLETE 5/5   the reveal control, concluded
#   ship-intersect-w2x-2m-random   PARTIAL        stopped by request; frozen 5/5, online 2/5
mkdir -p "$DEST"
for t in $TREES; do
  rsync -a --prune-empty-dirs \
        --include='*/' --include='results.json' --include='status.txt' --include='CAPPED.txt' --exclude='*' \
        "$SRC/$t/" "$DEST/$t/" 2>/dev/null
  n=$(find "$DEST/$t" -name results.json 2>/dev/null | wc -l)
  printf "  %-30s %3d cells\n" "$t" "$n"
done
echo "  total $(du -sh "$DEST" | cut -f1) in $DEST"
