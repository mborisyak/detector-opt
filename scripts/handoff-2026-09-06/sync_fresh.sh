#!/bin/bash
# Mirror the fresh CERN SHiP campaign's small files (results, verification, markers, configs) to the workstation.
set -u
DEST=/home/max/dev/detector-opt/output/ship-cern-fresh
SRC=lxplus:/afs/cern.ch/work/m/maborisy/detector-opt/output
mkdir -p "$DEST"
for t in intersect angle; do
  rsync -a --prune-empty-dirs \
        --include='*/' --include='results.json' --include='verification.json' --include='done.txt' --include='verified.txt' \
        --include='config.yaml' --include='CAPPED.txt' --include='status.txt' --exclude='*' \
        "$SRC/$t/" "$DEST/$t/" 2>&1 | tail -1
done
echo "mirrored: done=$(ls $DEST/*/select/*/*/*/done.txt | wc -l) verified=$(ls $DEST/*/select/*/*/*/verified.txt | wc -l) verification.json=$(ls $DEST/*/select/*/*/*/verification.json | wc -l)"
