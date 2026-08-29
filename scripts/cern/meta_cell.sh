#!/bin/bash
# One (i, i+1) pair of the representation-transfer probe: cold train at i, evaluate that network at i
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
PAIR=$2
REPR=$3
REVEAL=$4
AFSOUT=$5
PY=$D/lcgvenv/bin/python
RUN=$D/ship-addr-prec2e2/$SEED/continue
mkdir -p "$AFSOUT" || exit 91
rm -f "$AFSOUT/status.txt"

echo "host:   $(hostname)"
echo "gpu:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
echo "probe:  seed=$SEED pair=($PAIR,$((PAIR+1))) repr=$REPR reveal=$REVEAL"
echo "start:  $(date -u)"

set -o pipefail
$PY -u scripts/probe_strip_transfer.py --run "$RUN" --pair "$PAIR" --representation "$REPR" \
    --reveal "$REVEAL" --n-events 131072 --device cuda --output "$AFSOUT" \
    --n0 8192 --n-increment 8192 --only-meta 2>&1 | tee "$AFSOUT/run.log"
RC=$?
set +o pipefail

{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "seed=$SEED pair=$PAIR repr=$REPR reveal=$REVEAL"; echo "host=$(hostname)"; echo "finished=$(date -u)"; } > "$AFSOUT/status.txt"
exit $RC
