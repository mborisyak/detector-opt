#!/bin/bash
# INDEPENDENT VERIFICATION of one finished campaign cell, on a CERN HTCondor GPU node.
#
# `scripts/verify_trajectory.py` re-scores <= `verify.n_points` INCUMBENTS of a cell's trajectory on
# data the run never saw: fresh events at the fixed design, split 6:2:2, the design's own reported
# checkpoint continued for `verify.epochs`, best-validation parameters scored ONCE on held-out test.
# What it answers is whether the number a cell REPORTED survives independent data.
#
# ENV IS `geom_cell.sh`'s, verbatim and load-bearing; see docs/lxplus-htcondor-gpu.md.
#
# THE CELL IS COPIED TO SCRATCH AND THE RESULT COPIED BACK. Verification reads `results.json` plus the
# per-design checkpoints (which is why it can only run where those live) and writes `verification.json`
# + `verification.png` + `plots/`. Only those come back, so a verification can never damage the
# trajectory it is checking -- the run's own outputs are never rewritten.
#
# RESUMABLE BY CONSTRUCTION: points already in `verification.json` under identical settings are reused,
# so a cell killed by +MaxRuntime resumes where it stopped when resubmitted.
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
AFSCELL=$3
# Optional: how many INCUMBENTS to re-score. Absent -> the config's `verify.n_points`.
# `_select_points` always includes the LAST incumbent, so `1` is exactly "the final design only".
NPOINTS=${4:-}
PY=$D/lcgvenv/bin/python
SCRATCH=${_CONDOR_SCRATCH_DIR:-/tmp}
WORK=$SCRATCH/cell
mkdir -p "$WORK" || exit 91

[ -f "$AFSCELL/results.json" ] || { echo "no results.json in $AFSCELL"; exit 93; }
rsync -a "$AFSCELL/results.json" "$AFSCELL/verification.json" "$WORK/" 2>/dev/null
rsync -a "$AFSCELL/checkpoints" "$WORK/" 2>/dev/null

echo "host:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "cell:    config=$CONFIG seed=$SEED cell=$AFSCELL"
echo "resume:  $([ -f "$WORK/verification.json" ] && echo yes || echo no)"
echo "started: $(date -u)"

sync_out() { rsync -a --include='verification.json' --include='verification.png' --include='plots/***' \
                    --exclude='*' "$WORK/" "$AFSCELL/" 2>/dev/null; }
( while true; do sleep 300; sync_out; done ) & SYNCPID=$!
on_term() { sync_out; { echo "status=TRUNCATED"; echo "exit=143"
  echo "reason=SIGTERM (+MaxRuntime); verification.json is banked, resubmit to RESUME"; } > "$AFSCELL/verify_status.txt"
  sync_out; exit 143; }
trap on_term TERM INT

set -o pipefail
$PY -u scripts/verify_trajectory.py "=$CONFIG" trajectory="$WORK" seed="$SEED" \
    ${NPOINTS:+verify.n_points=$NPOINTS} progress=plain 2>&1 | tee -a "$WORK/verify.log"
RC=$?
set +o pipefail
kill "$SYNCPID" 2>/dev/null; wait "$SYNCPID" 2>/dev/null; sync_out

{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "config=$CONFIG seed=$SEED cell=$AFSCELL"; echo "host=$(hostname)"
  echo "finished=$(date -u)"; } > "$AFSCELL/verify_status.txt"
echo "status=$RC"
exit $RC
