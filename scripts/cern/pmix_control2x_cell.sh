#!/bin/bash
# One cell of the rewind probe: ONE mid-BO design trained from scratch at one lambda, one trial.
# and i+1 with NO training, warm-start it onto i+1, and cold train i+1 as the control.
#
# Same environment contract as campaign_cell.sh; every export is load-bearing (docs/lxplus-htcondor-gpu.md).
# NO `set -u`: the LCG view's setup.sh would exit 1.
#
# READ-ONLY on the source campaign: it reads `results.json` for the design trajectory and writes only
# into its own output directory, so it cannot disturb a cell that is still running. It does NOT read
# the campaign's checkpoints -- the warm start is re-derived here, which is deliberate, because 64 of
# the 105 per-design checkpoints in those arms were never committed.
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
source $V/setup.sh 2>/dev/null
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=$D/lcgvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $D/detector-opt || exit 90

SEED=$1
DESIGN=$2
PMIX=$3
TRIAL=$4
AFSOUT=$5
PY=$D/lcgvenv/bin/python
RUN=$D/ship-addr-prec1e2/$SEED/from_scratch
mkdir -p "$AFSOUT" || exit 91
rm -f "$AFSOUT/status_control2x.txt"

echo "host:   $(hostname)"
echo "gpu:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
echo "probe:  seed=$SEED design=$DESIGN rewind=$PMIX trial=$TRIAL"
echo "start:  $(date -u)"

set -o pipefail
# A SECOND CONTROL: twice the window, drawn INDEPENDENTLY of the growth run. The 1x control asked
# whether the path matters at the same data; this one asks whether the growth run's margin survives
# giving the control twice as much data, on events the growth run never saw. Written to a separate
# `control_x2_independent` key, so the 1x control in the same file is untouched.
$PY -u scripts/probe_rewind.py --run "$RUN" --design "$DESIGN" --param-mix "$PMIX" \
    --seed "$TRIAL" --device cuda --output "$AFSOUT" --control-only --control-scale 2 --control-independent 2>&1 | tee "$AFSOUT/run_control2x.log"
RC=$?
set +o pipefail

{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "seed=$SEED design=$DESIGN rewind=$PMIX trial=$TRIAL"; echo "host=$(hostname)"; echo "finished=$(date -u)"; } > "$AFSOUT/status_control2x.txt"
exit $RC
