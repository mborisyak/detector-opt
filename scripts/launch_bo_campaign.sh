#!/usr/bin/env bash
# Start (or RESTART) the cloud campaign on `bo`. Run ON bo.
#
#   scripts/launch_bo_campaign.sh <local_gpu K> [cores] [n_seeds] [task] [prefix]
#   scripts/launch_bo_campaign.sh 8 24 5 enzyme_extremes output/cloud
#
# THE PREFIX GIVES THIS CAMPAIGN ITS OWN NAMESPACE, and costs nothing to use because the Snakefile
# already reads a target path as `<prefix>/<config>/...` with the prefix being any path, however many
# directories deep. So `output/cloud/enzyme_extremes` is the same task as `output/enzyme_extremes` in
# a different tree, with no rule, wildcard or config change. It matters because the workstation runs
# its own campaigns at the bare `output/` prefix over the SAME task and the SAME seeds: mirroring one
# tree onto the other would silently overwrite results. With the prefix, the remote path and the
# local path are identical and nothing collides.
#
# This is the ONE command that brings the campaign back after a preemption, which is why it is a
# script rather than a command line remembered by whoever launched it. It is safe to run when the
# campaign is already up: `flock -n` makes a second driver impossible, and every finished artefact is
# skipped by snakemake, so a re-run resumes rather than redoes.
#
# WHAT IT DOES IN ORDER, and each step is here because skipping it fails silently:
#
#   1. START MPS IF IT IS DOWN. The daemon does not survive a reboot, and this box is preemptible, so
#      after every preemption it is down. Without it, K concurrent jobs TIME-SLICE the device: no
#      error, no warning, and roughly single-job throughput spread over K jobs.
#   2. TAKE THE LOCK. Two drivers over one output tree is a duplicated campaign, and the local box
#      has already lost work to exactly that.
#   3. RUN UNDER tmux, NOT nohup/setsid. A detached process that outlives its supervisor is an orphan
#      nobody can stop; a tmux session is addressable, attachable and killable.
#
# `--rerun-triggers mtime` IS LOAD-BEARING ON A PREEMPTIBLE BOX, and it must sit BEFORE the `--`
# separator: the flag takes multiple values, so placing it after the targets makes argparse swallow
# them and the launch dies with "invalid choice: output/.../comparison.txt". Its job is to stop a
# RESTART from destroying a finished campaign. Snakemake's default triggers include `code` and
# `software-env`, so an edit to a rule's shell command -- or environment drift after a re-provision --
# marks completed outputs stale, and a scheduled `bo` job DELETES results.json before re-running it.
# That would throw away paid-for event pools, not merely loop position. With mtime alone, a resume
# resumes. Config files are deliberately not declared inputs for the same reason, so this loses
# nothing that was ever relied on; force explicitly when settings really change.
#
# K COMES FROM THE MEASUREMENT, not from a guess: scripts/probe_mps.sh sweeps concurrency and
# scripts/probe_mps_report.py prints the table it is read off. `cores` bounds snakemake's own core
# accounting, and the heavy rules declare cpus_per_task=4, so cores/4 is a SECOND cap on concurrency
# -- set cores >= 4*K or the local_gpu setting will not be what actually binds.
set -euo pipefail

K=${1:?usage: launch_bo_campaign.sh <local_gpu K> [cores] [n_seeds] [task] [prefix]}
CORES=${2:-24}
N_SEEDS=${3:-5}
TASK=${4:-enzyme_extremes}
PREFIX=${5:-output/cloud}

REPO=$HOME/detector-opt
VENV=$HOME/venv
LOCK=$HOME/.detector-opt-campaign.lock
SESSION=campaign
LOG=$REPO/logs/snakemake-$TASK.log
CAMPAIGN=$PREFIX/$TASK

export CUDA_MPS_PIPE_DIRECTORY="$HOME/.mps"
export CUDA_MPS_LOG_DIRECTORY="$HOME/.mps/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$REPO/logs"

if [ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then
  echo "MPS: starting the control daemon"
  nvidia-cuda-mps-control -d
  sleep 2
else
  echo "MPS: already up"
fi
[ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] || { echo "MPS FAILED TO START -- refusing to launch" >&2; exit 1; }

SEEDS=$("$VENV/bin/python" - "$N_SEEDS" <<'PY'
import random
import sys
rng = random.Random(123456)
print(" ".join(str(rng.randint(0, 2 ** 31 - 1)) for _ in range(int(sys.argv[1]))))
PY
)
echo "seeds: $SEEDS"

TARGETS=""
for seed in $SEEDS; do
  TARGETS="$TARGETS $CAMPAIGN/$seed/comparison.txt"
done
TARGETS="$TARGETS $CAMPAIGN/median.json"
echo "campaign tree: $CAMPAIGN"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
fi

tmux new-session -d -s "$SESSION" -c "$REPO" "
  source $VENV/bin/activate
  export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY
  flock -n $LOCK snakemake -s Snakefile.cloud \
    -c$CORES --resources local_gpu=$K --config n_seeds=$N_SEEDS \
    --rerun-triggers mtime \
    -- $TARGETS >> $LOG 2>&1
  echo \"driver exited \$? at \$(date -u +%FT%TZ)\" >> $LOG
  sleep 86400
" 2>/dev/null || echo "tmux session '$SESSION' is already running a driver; not starting a second"

sleep 3
echo "--- tail $LOG ---"
tail -5 "$LOG" 2>/dev/null || echo "(no log yet)"
echo
echo "attach:  tmux attach -t $SESSION"
echo "watch:   tail -f $LOG"
