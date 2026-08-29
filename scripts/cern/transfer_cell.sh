#!/bin/bash
# One block of the `continue` transfer probe: restore the network at designs FIRST..LAST of a completed
# cell and evaluate each, without training, on its own design and the next HORIZON designs.
#
# Same environment contract as campaign_cell.sh -- every export is load-bearing, see
# docs/lxplus-htcondor-gpu.md. NO `set -u`: the LCG view's setup.sh would exit 1.
#
# READ-ONLY on the source run. It restores checkpoints and writes nothing into the campaign tree; its
# own output goes to a separate directory, so it cannot disturb a cell that is still running.
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
source $V/setup.sh 2>/dev/null
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=$D/lcgvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $D/detector-opt || exit 90

SEED=$1
FIRST=$2
LAST=$3
AFSOUT=$4
PY=$D/lcgvenv/bin/python
RUN=$D/ship-addr-prec2e2/$SEED/continue
mkdir -p "$AFSOUT" || exit 91
rm -f "$AFSOUT/status.txt"

echo "host:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
echo "probe:   seed=$SEED i=$FIRST..$LAST run=$RUN"
echo "started: $(date -u)"

[ -d "$RUN/checkpoints" ] || { echo "status=NO_CHECKPOINTS" > "$AFSOUT/status.txt"; exit 92; }

set -o pipefail
$PY -u scripts/probe_continue_transfer.py --run "$RUN" --first "$FIRST" --last "$LAST" \
    --horizon 3 --n-events 131072 --device cuda --output "$AFSOUT" 2>&1 | tee "$AFSOUT/run.log"
RC=$?
set +o pipefail

{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "seed=$SEED first=$FIRST last=$LAST"; echo "host=$(hostname)"; echo "finished=$(date -u)"; } > "$AFSOUT/status.txt"
echo "status rc=$RC"
exit $RC
