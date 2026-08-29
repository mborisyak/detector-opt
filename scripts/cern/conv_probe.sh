#!/bin/bash
# Environment + convolution diagnostic on a CERN GPU node, run TWICE on the SAME node so the two
# library configurations are compared without node-to-node variation.
#
# WHY: the first run of this probe showed EVERY convolution backward failing (dense as well as
# grouped), with `libcublasLt` loaded from the venv wheel while `libcublas` and `libcudart` came from
# the CVMFS CUDA 12.5 view -- a mismatched set.  Pass B puts all three venv wheel dirs ahead of CVMFS
# so cuBLAS, cuBLASLt, nvrtc and cuDNN come from one place.  That did NOT help.
# Pass C drops the wheel override entirely: the venv ships cuDNN 9.24.0.43 while the view ships
# 9.3.0, and `<unknown cudnn status: 5003>` is XLA failing to recognise a status enum from a cuDNN
# NEWER than the one jaxlib 0.9.0 was built against.
#
# NO XLA_FLAGS: the flags that mask these failures cost autotuning and CUDA graphs.
# NO `set -u`: the LCG view's setup.sh dereferences unset variables.
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
W=$D/lcgvenv/lib/python3.13/site-packages/nvidia
source $V/setup.sh 2>/dev/null
VIEWPP=${PYTHONPATH}
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_FLAGS
cd $D/detector-opt || exit 90
BASE=$LD_LIBRARY_PATH
echo "host: $(hostname)"
echo "############ PASS A: cudnn wheel only (the current strip_cell.sh configuration) ############"
LD_LIBRARY_PATH=$W/cudnn/lib:$BASE $D/lcgvenv/bin/python -u scripts/probe_conv_backend.py
echo "############ PASS F: CVMFS cudnn 9.20.0.48 (inside the >=9.8,<10.0 the plugin declares) ############"
C920=/cvmfs/sft.cern.ch/lcg/releases/cudnn/9.20.0.48-c2e81/x86_64-el9-gcc14-opt/lib
PYTHONPATH=$D/detector-opt:$D/lcgshim:$VIEWPP LD_LIBRARY_PATH=$C920:$BASE python -u scripts/probe_conv_backend.py
echo "############ PASS E: the VIEW's python + shim, so no venv site-packages is injected ############"
# A venv interpreter puts ITS OWN site-packages on sys.path whatever PYTHONPATH says, so the wheel's
# cuDNN 9.24 wins over the view's matched 9.3.0.  The view's python has no such injection.
PYTHONPATH=$D/detector-opt:$D/lcgshim:$VIEWPP LD_LIBRARY_PATH=$BASE python -u scripts/probe_conv_backend.py
echo "done"
