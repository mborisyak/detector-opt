#!/bin/bash
# Does the 1D res-net (`StripRegressor`) train on a VOLTA node?
#
# The capability >= 8.0 floor on every strip job rests on `probe_conv_backend.py`, which exercises a
# convolution at this network's shapes but never builds the network. This runs the REAL thing: the
# strip detector, the real regressor and the real growth loop, through `probe_strip_transfer.py
# --only-control`, which is one cold training round and nothing else.
#
# It is capped on purpose (`n0 = n_increment = iteration_limit`), so the round ends at its first data
# request and the job is minutes rather than hours. A capped round is a NORMAL exit here and is NOT
# the result under test: what is under test is whether the backward pass runs at all. Read the log --
# a Volta failure shows as an exception out of the first training step, not as a capped round.
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
source $V/setup.sh 2>/dev/null
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=$D/lcgvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_FLAGS
cd $D/detector-opt || exit 90

AFSOUT=$D/ship-volta-test/strip
PY=$D/lcgvenv/bin/python
RUN=$D/ship-addr-prec2e2/1244111331/continue
mkdir -p "$AFSOUT" || exit 91
rm -f "$AFSOUT/status.txt"

echo "host:   $(hostname)"
echo "gpu:    $(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "start:  $(date -u)"

set -o pipefail
$PY -u scripts/probe_strip_transfer.py --run "$RUN" --pair 9 --representation strip \
    --reveal none --n-events 8192 --device cuda --output "$AFSOUT" \
    --n0 2048 --n-increment 2048 --iteration-limit 2048 --only-control 2>&1 | tee "$AFSOUT/run.log"
RC=$?
set +o pipefail

{ echo "status=$([ $RC -eq 0 ] && echo COMPLETED || echo EXIT_$RC)"; echo "exit=$RC"
  echo "gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1 | head -1)"
  echo "host=$(hostname)"; echo "finished=$(date -u)"; } > "$AFSOUT/status.txt"
exit $RC
