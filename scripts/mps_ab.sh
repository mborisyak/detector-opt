#!/bin/bash
# A/B: does connecting to the MPS daemon change concurrent throughput at fixed K?
# Same K, same config, same machine, back to back. ONLY the pipe-directory env differs.
set -u
cd ~/detector-opt
K=${K:-4}; EPOCHS=${EPOCHS:-30}; PY=/home/max/venv/bin/python
mkdir -p output/mpsab
for mode in nompsA mps nompsB; do
  echo "=== $mode : K=$K ==="
  pids=()
  for ((r=0;r<K;r++)); do
    if [ "$mode" = "mps" ]; then
      env CUDA_MPS_PIPE_DIRECTORY=/home/max/.mps CUDA_MPS_LOG_DIRECTORY=/home/max/.mps-log \
        XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -u scripts/probe_epoch_cost.py =bo \
        --n-models 1 --epochs $EPOCHS --output output/mpsab/${mode}_r${r}.json > output/mpsab/${mode}_r${r}.log 2>&1 &
    else
      env XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -u scripts/probe_epoch_cost.py =bo \
        --n-models 1 --epochs $EPOCHS --output output/mpsab/${mode}_r${r}.json > output/mpsab/${mode}_r${r}.log 2>&1 &
    fi
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== $mode : done ==="
done
echo "AB COMPLETE"
