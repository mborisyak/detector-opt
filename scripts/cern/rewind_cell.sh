#!/bin/bash
# ONE SEED of the REWIND PROBE on a CERN HTCondor GPU node.
#
# THREE CONDITIONS on the SAME mid-BO designs (user, 2026-08-30):
#   1. trained WITH the rewind      (`rewind` 0.25, the settled default)
#   2. trained WITHOUT it           (`rewind` 0.0)
#   3. trained on the FULL WINDOW with NO GROWTH (`FixedWindowTrainer`), at each growth cell's own
#      final window and its own slice of the event index -- so the only difference is the PATH to the
#      dataset, not the dataset.
#
# Both arms are run: the rewind pulls the network back toward whatever `_init_design_network`
# returned, which is a FRESH DRAW for `from_scratch` and the CARRIED network for `meta`, so `p - q` --
# the thing it acts on -- is far larger in one than the other. That asymmetry is the point.
#
# ENV IS `geom_cell.sh`'s, verbatim; see docs/lxplus-htcondor-gpu.md.
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
source $V/setup.sh 2>/dev/null
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=$D/lcgvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $D/detector-opt || exit 90

CONFIG=$1
SEED=$2
TRAJ=$3
AFSOUT=$4
PY=$D/lcgvenv/bin/python
SCRATCH=${_CONDOR_SCRATCH_DIR:-/tmp}
mkdir -p "$AFSOUT" || exit 91
OUTJSON=$SCRATCH/rewind_s${SEED}.json
# Resume from whatever a previous attempt banked, so a cell cut off by +MaxRuntime continues.
[ -f "$AFSOUT/rewind_s${SEED}.json" ] && cp "$AFSOUT/rewind_s${SEED}.json" "$OUTJSON"

echo "host:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "cell:    config=$CONFIG seed=$SEED traj=$TRAJ"
echo "started: $(date -u)"

sync_out() { cp "$OUTJSON" "$AFSOUT/" 2>/dev/null; }
( while true; do sleep 300; sync_out; done ) & SYNCPID=$!
on_term() { sync_out; echo "status=TRUNCATED" > "$AFSOUT/status_s${SEED}.txt"; exit 143; }
trap on_term TERM INT

set -o pipefail
$PY -u scripts/probe_rewind.py "=$CONFIG" \
    --trajectory "$TRAJ" \
    --seeds "$SEED" \
    --arms from_scratch meta \
    --param-mix 0.25 0.0 \
    --n-designs 3 \
    --fixed-window-control \
    --resume \
    --output "$OUTJSON" 2>&1 | tee -a "$SCRATCH/rewind_s${SEED}.log"
RC=$?
set +o pipefail
kill "$SYNCPID" 2>/dev/null; wait "$SYNCPID" 2>/dev/null; sync_out
cp "$SCRATCH/rewind_s${SEED}.log" "$AFSOUT/" 2>/dev/null
{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "config=$CONFIG seed=$SEED"; echo "finished=$(date -u)"; } > "$AFSOUT/status_s${SEED}.txt"
exit $RC
