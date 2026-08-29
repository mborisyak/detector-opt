#!/bin/bash
# ONE WIDTH of the strip regressor, trained cold at a single design.
#
# `--only-control` is one from-scratch training round and nothing else, so the only thing that differs
# between the cells of this sweep is `--channels`. The 9-dimensional target (vertex + p1 + p2) is why
# no block may be narrower than 9: a narrower one is a bottleneck the target cannot pass through.
#
# Same environment contract as the other cells; every export is load-bearing.
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
source $V/setup.sh 2>/dev/null
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=$D/lcgvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $D/detector-opt || exit 90

# The tag is the width with dashes (16-16-16-16-16); HTCondor's `queue ... from` splits on COMMAS, so
# the channel list cannot be passed as one field. Convert here, in the one place that knows both forms.
TAG=$1
CHANNELS=$(echo "$TAG" | tr '-' ',')
AFSOUT=$2
PY=$D/lcgvenv/bin/python
RUN=$D/ship-addr-prec2e2/1244111331/continue
mkdir -p "$AFSOUT" || exit 91
rm -f "$AFSOUT/status.txt"

echo "host:     $(hostname)"
echo "gpu:      $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1 | head -1)"
echo "channels: $CHANNELS   (tag $TAG)"
echo "start:    $(date -u)"

set -o pipefail
$PY -u scripts/probe_strip_transfer.py --run "$RUN" --pair 9 --representation strip --reveal none \
    --only-control --n-events 8192 --n0 131072 --n-increment 65536 --iteration-limit 524288 \
    --loss-precision 1.0e-2 --learning-rate 2.0e-4 --channels "$CHANNELS" \
    --device cuda --output "$AFSOUT" 2>&1 | tee "$AFSOUT/run.log"
RC=$?
set +o pipefail

{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "channels=$CHANNELS"; echo "host=$(hostname)"; echo "finished=$(date -u)"; } > "$AFSOUT/status.txt"
exit $RC
