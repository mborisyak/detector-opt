#!/bin/bash
# INDEPENDENT VERIFICATION of ONE hyper-parameter-selection cell's FINAL result.
#
# WHY. The selection ranks regimes by the loss each run REPORTED for its own designs. That number is
# faithful but optimistic, and the ranking between two regimes separated by ~0.005 is inside the range
# where optimism and noise can decide it. This re-scores each cell's ANSWER -- its last incumbent -- on
# a freshly sampled, disjoint train/validation/test split the run never saw, so the regimes can be
# compared on a held-out number instead of a self-reported one.
#
# `verify.n_points=1` selects the LAST incumbent only: the run's answer, not its whole trajectory. The
# test stage verifies 10 points per cell because there the trajectory is the object of study; here only
# the endpoint enters the selection.
#
# IT WRITES OUTSIDE `output/<task>/select/`. That tree is the input of a COMPLETED selection whose
# verdict the running test stage already consumes, and the snakemake orchestrator is still live against
# it. Verification lands in `output/<task>/verify-select/` so the selection tree stays byte-identical.
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

TASK=$1
SEED=$2
ARM=$3
REGIME=$4
# The source tree is named explicitly rather than assumed, so a probe run beside the selection can be
# verified by the same script. Defaults to the selection tree.
TREE=${5:-select}
TRAJ=output/$TASK/$TREE/$SEED/$ARM/$REGIME/results.json
AFSOUT=$D/detector-opt/output/$TASK/verify-select/$SEED/$ARM/$REGIME
PY=$D/lcgvenv/bin/python
SCRATCH=${_CONDOR_SCRATCH_DIR:-/tmp}

[ -f "$TRAJ" ] || { echo "missing trajectory $TRAJ"; exit 92; }
# A HALF-FINISHED RUN IS NOT AN ANSWER. `verify.n_points=1` re-scores the LAST incumbent, which for an
# unfinished trajectory is wherever it happened to be interrupted -- a different quantity from the
# regime's result, and one that would silently enter the comparison as if it were the same. Refuse it.
$PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('completed') else 93)" "$TRAJ" \
  || { echo "trajectory not completed: $TRAJ"; exit 93; }
mkdir -p "$AFSOUT" || exit 91

echo "host:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "cell:    task=$TASK seed=$SEED arm=$ARM regime=$REGIME tree=$TREE"
echo "started: $(date -u)"

set -o pipefail
$PY -u scripts/verify_trajectory.py "=$TASK" \
    trajectory="$TRAJ" \
    output="$AFSOUT" \
    seed=$SEED \
    verify.n_points=1 \
    progress=plain 2>&1 | tee -a "$SCRATCH/vs_${TASK}_${SEED}_${ARM}_${REGIME}.log"
RC=$?
set +o pipefail
cp "$SCRATCH/vs_${TASK}_${SEED}_${ARM}_${REGIME}.log" "$AFSOUT/" 2>/dev/null
{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "task=$TASK seed=$SEED arm=$ARM regime=$REGIME"; echo "finished=$(date -u)"; } > "$AFSOUT/status.txt"
exit $RC
