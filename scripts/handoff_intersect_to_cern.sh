#!/bin/bash
# Move the intersect campaign from the mirror onto CERN and start it there.
#
#   scripts/handoff_intersect_to_cern.sh            # check only, change nothing
#   scripts/handoff_intersect_to_cern.sh --go       # relay and launch
#
# WHY THIS EXISTS. `bo` is a spot box that has already died twice mid-campaign. When it goes, the
# work survives only in the workstation mirror that `evacuate_bo_loop.sh` refreshes, and CERN is the
# only other place SHiP compute may run. This turns the mirror back into a running campaign there.
#
# ⛔️ CHECKPOINTS ARE NOT OPTIONAL. `verify_trajectory.py` restores the network each design was
# REPORTED with from `checkpoints/design_NNNN/<epoch>/`. A results-only relay looks complete -- done.txt
# present, designs banked -- and then fails hours later with "holds no saved epoch". That has happened
# twice: once from a results-only filter, once from a cell whose checkpoint directory existed but was
# EMPTY, which an `[ -d ]` pre-flight check happily passed. This script copies the WHOLE tree and
# audits by CONTENT, not by directory existence.
#
# ⚠️ --nolock IS DELIBERATE. The angle orchestrator already holds the working directory's lock. The
# two target disjoint output trees, so a second orchestrator is safe here and a lock would only block.
set -u
GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$(dirname "$0")/.." || exit 90
MIRROR=output/final-bo/intersect
REMOTE=/afs/cern.ch/work/m/maborisy/detector-opt

say() { printf '  %s\n' "$*"; }

say "=== 1. what the mirror holds ==="
say "cells=$(find $MIRROR -name results.json | wc -l) done=$(find $MIRROR -name done.txt | wc -l) verified=$(find $MIRROR -name verified.txt | wc -l) size=$(du -sh $MIRROR | cut -f1)"
say "last mirror pass: $(tail -1 logs/evacuate-bo.log)"

say "=== 2. audit checkpoints BY CONTENT for every done-but-unverified cell ==="
BAD=$(python3 - <<'PY'
import glob, os
bad = []
for d in glob.glob("output/final-bo/intersect/select/*/*/*/"):
    if not os.path.exists(os.path.join(d, "done.txt")):    continue
    if os.path.exists(os.path.join(d, "verified.txt")):    continue
    designs = glob.glob(os.path.join(d, "checkpoints", "design_*"))
    usable = [p for p in designs if len(os.listdir(p)) > 0]
    if len(usable) == 0:
        bad.append(os.path.relpath(d, "output/final-bo/intersect/select"))
print("\n".join(bad))
PY
)
if [ -n "$BAD" ]; then
  say "⚠️ cells that CANNOT verify (no usable checkpoint) -- look for a copy at CERN before re-running:"
  printf '     %s\n' $BAD
else
  say "all done-but-unverified cells hold a usable checkpoint"
fi

if [ "$GO" -eq 0 ]; then
  say ""
  say "CHECK ONLY -- nothing changed. Re-run with --go to relay and launch."
  exit 0
fi

say "=== 3. relay the WHOLE tree (checkpoints included) ==="
rsync -a --update "$MIRROR/" "lxplus:$REMOTE/output/intersect/" || exit 91
say "relayed"

say "=== 4. launch the intersect orchestrator at CERN ==="
ssh -o BatchMode=yes lxplus "cd $REMOTE || exit 9
  export PATH=/afs/cern.ch/work/m/maborisy/pyenv/versions/3.14.3/bin:\$PATH
  setsid nohup flock -n /tmp/intersect-cern.lock snakemake -s ship.snake -j40 output/intersect/campaign.txt \
    --executor cluster-generic \
    --cluster-generic-submit-cmd 'scripts/cern/snakemake_submit.sh {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}' \
    --cluster-generic-status-cmd scripts/cern/snakemake_status.sh \
    --cluster-generic-cancel-cmd scripts/cern/snakemake_cancel.sh \
    --config 'preamble=source scripts/cern/lcg_env.sh && ' \
    --latency-wait 60 --keep-going --rerun-incomplete --nolock \
    > logs/intersect-cern.log 2>&1 < /dev/null &
  sleep 45
  echo \"  intersect at CERN: \$(ls output/intersect/select/*/*/*/done.txt 2>/dev/null|wc -l)/45 done, \$(ls output/intersect/select/*/*/*/verified.txt 2>/dev/null|wc -l)/45 verified\"
  echo \"  condor: \$(condor_q -af JobStatus 2>/dev/null | sort | uniq -c | tr '\n' ' ')\"
  echo \"  errors: \$(grep -cE '^Error' logs/intersect-cern.log 2>/dev/null)\""

say "=== 5. STOP THE EVACUATION LOOP -- it is syncing from a dead host ==="
say "run:  ps -eo pid,cmd | awk '\$2==\"bash\" && \$3 ~ /evacuate_bo_loop/ {print \$1}'   then kill that pid"
