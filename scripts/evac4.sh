#!/bin/bash
# Periodic pull from the 4-GPU spot box, then a final complete sweep once its queue drains.
# Excludes trainer.npz ONLY: that file is the 3.1 GB persisted EVENT POOL, regenerable from the
# detector data plus the seed, and it is 12 x 3.1 GB across the campaign. Everything that is a
# RESULT -- results.json, convergence.json/png, plots, run.log, checkpoints, optimizer.npz -- is taken.
set -u
KH=/home/max/.claude/jobs/299c4b92/tmp/known_hosts2
SSH="ssh -i /home/max/.ssh/id_github -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
H=root@86.38.182.83.compute.verda.run
DEST=/home/max/dev/detector-opt/output/ship-4gpu/
pull() {
  RSYNC_RSH="$SSH" rsync -a --exclude='trainer.npz' \
    "$H:/mnt/results/output/ship-4gpu/" "$DEST" 2>&1 | tail -2
}
idle=0
for i in $(seq 1 400); do
  pull
  n=$($SSH $H 'squeue -h | wc -l' 2>/dev/null)
  echo "$(date -Is) synced; remote queue=${n:-unreachable}"
  if [ "${n:-1}" = "0" ]; then idle=$((idle+1)); else idle=0; fi
  if [ "$idle" -ge 2 ]; then
    echo "$(date -Is) queue empty twice -- FINAL SWEEP"
    pull
    echo "$(date -Is) EVACUATION COMPLETE"
    break
  fi
  sleep 300
done
