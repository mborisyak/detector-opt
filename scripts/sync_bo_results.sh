#!/usr/bin/env bash
# Pull campaign results back from the preemptible box `bo` to this workstation.
#
# `bo` is PREEMPTIBLE and has been reclaimed mid-provision at least once, so results have to land
# here while the run is alive, not at the end. Run this on a loop from the workstation.
#
# THE HEAVY FILES ARE EXCLUDED DELIBERATELY. `trainer.npz` and `optimizer.npz` are the resume state
# and the event pools; on this workstation they are 181.3 GB of a 184.4 GB output tree while the
# whole scientific record is 1.64 GB. They are also useless here: a resume happens ON bo, against
# bo's own copy. What is pulled is the record -- results.json, convergence.json, the logs and the
# plots -- plus `checkpoints/`, which is what `verify_trajectory.py` needs to re-score a run.
#
# `flock -n` so a second copy cannot start: a duplicated sweep here once truncated its own log.
# Never `--delete`: bo's tree is the live one, and a cell that has not been written yet must not
# remove what an earlier pass already brought back.
#
#   bash scripts/sync_bo_results.sh output/campaign-emnist
set -euo pipefail

REMOTE=${REMOTE:-bo}
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/work/repo}
SSH_OPTS=${SSH_OPTS:-}
LOCAL_ROOT=${LOCAL_ROOT:-/home/max/dev/detector-opt}
LOCK=${LOCK:-/tmp/sync-bo-results.lock}

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <output-subdir> [<output-subdir> ...]" >&2
  exit 2
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another sync holds $LOCK -- skipping this pass"
  exit 0
fi

for sub in "$@"; do
  mkdir -p "${LOCAL_ROOT}/${sub}"
  echo "=== ${sub}"
  rsync -az --partial ${SSH_OPTS:+-e "ssh ${SSH_OPTS}"} \
    --exclude '*.npz' --exclude 'ship-4gpu/' \
    "${REMOTE}:${REMOTE_ROOT}/${sub}/" "${LOCAL_ROOT}/${sub}/"
  find "${LOCAL_ROOT}/${sub}" -name results.json | while read -r f; do
    python3 -c "
import json,sys,os
d=json.load(open(sys.argv[1])); r=d.get('results',[])
l=[x['loss'] for x in r if x.get('loss') is not None]
print(f\"  {os.path.relpath(sys.argv[1], sys.argv[2]):<44} {len(r):3d} designs  best {min(l) if len(l)>0 else float('nan'):.4f}  completed={d.get('completed')}\")
" "$f" "${LOCAL_ROOT}/${sub}"
  done
done
