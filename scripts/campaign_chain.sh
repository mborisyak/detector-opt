#!/usr/bin/env bash
# One step of the campaign chain: if the running campaign has FINISHED, pull it in full and start the
# next one. Run from the workstation, once per heartbeat. Does nothing and exits 0 when the campaign
# is still running, so it is safe to call on every beat.
#
#   scripts/campaign_chain.sh
#
# THE CHAIN, in order:
#
#   1. enzyme_extremes       at output/cloud        -- narrow network, 12648 parameters, ran at K=8
#   2. enzyme_extremes_wide  at output/cloud-wide   -- 22888 parameters, growth 1.5x, runs at K=12
#
# ⚠️ K AND CORES MUST AGREE OR THE ONE YOU SET IS NOT THE ONE THAT BINDS. The heavy rules declare
# `cpus_per_task=3`, so snakemake admits floor(CORES/3) GPU jobs regardless of `--resources
# local_gpu=K`: at K=12 with the old CORES=24 the campaign would silently run at 8. CORES is
# therefore 36 = 12 x 3. That overcommits the box's 24 physical cores on paper, which is fine --
# a job sits at ~0.9 of a core in steady state and bursts to ~2.5 only while XLA compiles -- but the
# BLAS caps in Snakefile.cloud's ENV stay at 3 per job, so the acquisition phase can put 36 threads
# on 24 cores. That is mild oversubscription during a small fraction of each design, accepted rather
# than fixed, because editing Snakefile.cloud would change every rule's shell command and risk
# snakemake treating the FINISHED campaign's outputs as stale (see launch_bo_campaign.sh).
#
# K=12 against K=8 is worth +14% aggregate throughput on the measured sweep (6.46x vs 5.66x) at the
# cost of per-worker efficiency falling from 71% to 54%. Memory is not a factor either way: 12 x
# 2.65 GiB = 32 GiB of the card's 96 GiB, against a memory ceiling of ~36 processes.
#
# DONE is `median.json` existing ON BO, which is the last artefact snakemake produces: it depends on
# every results.json and every verification, so its presence means the whole DAG completed. Checking
# the LOCAL mirror instead would be wrong -- the light heartbeat pull could lag, and a missing file
# would read as "not finished" and silently stall the chain.
#
# ORDER MATTERS: the next campaign is LAUNCHED FIRST, then the finished one is pulled. The launch is
# one fast ssh; the pull is ~24 GB and of order an hour on this link. Doing the pull first would
# leave the GPU idle for that hour for no reason -- the two campaigns write different trees, and bo
# has 1.2 TB free, so the finished tree is in no danger while it is being copied.
#
# THIS SCRIPT IS SYNCHRONOUS AND THE CALLER BACKGROUNDS IT. Nothing here detaches: no `setsid`, no
# `nohup`, no `&`. A monitor that outlives its supervisor is an orphan polling forever with nobody to
# stop it, so the long pull runs in the foreground of whatever backgrounded this script and can be
# killed with it.
#
# Marker files under logs/ record what has already been triggered, so a re-run does not launch a
# second driver or a second pull. `flock -n` inside launch_bo_campaign.sh and pull_campaign_full.sh
# are the real guards; these are belt and braces on this side.
#
# ⚠️ THE PULL MARKER IS WRITTEN ONLY ON SUCCESS, and that is the whole difference between a pull that
# happens and one that quietly does not. Writing it before the transfer -- as this first did -- means
# an interrupted or failed rsync is recorded as done and never retried, and the failure is invisible
# because the next beat skips it. Written afterwards, a failure simply leaves the marker absent and
# the next beat tries again; rsync resumes rather than restarting, so retrying is cheap. A beat that
# finds the lock already held also fails here, which is correct: that pull is still running, and this
# one should do nothing and come back later.
set -euo pipefail

HOST=${HOST:-bo}
LOCAL_ROOT=${LOCAL_ROOT:-/home/max/dev/detector-opt}
MARKERS=$LOCAL_ROOT/logs
K=${K:-12}
CORES=${CORES:-36}
N_SEEDS=${N_SEEDS:-5}

mkdir -p "$MARKERS"

finished_on_bo() {
  timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "test -f \$HOME/detector-opt/$1/$2/median.json" 2>/dev/null
}

pull_full() {
  local task=$1 prefix=$2 status=0
  local marker=$MARKERS/.pulled-$task
  if [ -f "$marker" ]; then return 0; fi
  echo "chain: $task finished -- full pull (~24 GB, of order an hour on this link)"
  bash "$LOCAL_ROOT/scripts/pull_campaign_full.sh" "$task" "$prefix" || status=$?
  if [ "$status" -eq 0 ]; then
    date -u +%FT%TZ > "$marker"
    echo "chain: full pull of $task COMPLETE -- $(du -sh "$LOCAL_ROOT/$prefix/$task" | cut -f1)"
  else
    echo "chain: full pull of $task did NOT complete (exit $status) -- no marker written, " \
         "it will be retried on the next beat. See logs/pull-full-$task.log"
  fi
}

launch() {
  local task=$1 prefix=$2
  local marker=$MARKERS/.launched-$task
  if [ -f "$marker" ]; then return 0; fi
  echo "chain: launching $task at $prefix (K=$K, cores=$CORES, seeds=$N_SEEDS)"
  timeout 300 ssh -o BatchMode=yes "$HOST" \
    "bash ~/detector-opt/scripts/launch_bo_campaign.sh $K $CORES $N_SEEDS $task $prefix" 2>&1 | tail -5
  date -u +%FT%TZ > "$marker"
}

if ! finished_on_bo output/cloud enzyme_extremes; then
  echo "chain: enzyme_extremes still running -- nothing to do"
  exit 0
fi

pull_full enzyme_extremes output/cloud

# NOTHING IS LAUNCHED FROM HERE. The next campaign is the 1.5x-budget CONTINUATION of this one, and
# it runs on a COPY of this tree under its own prefix -- which needs the new resume semantics
# (results.json + done.txt) and the pool/index fixes pushed to bo first, a 24 GB tree copied there,
# its markers migrated, and its verification markers removed. Those are consequential, ordered steps
# on a finished campaign's only copy, so they are done deliberately rather than fired by a heartbeat.
# The wide-network campaign follows the continuation.
echo "chain: enzyme_extremes is COMPLETE and its full tree has been pulled."
echo "chain: NEXT is the 1.5x-budget continuation on a copy -- set up deliberately, not from here."

if finished_on_bo output/cloud-cont enzyme_extremes_cont; then
  pull_full enzyme_extremes_cont output/cloud-cont
  echo "chain: continuation COMPLETE and pulled."
fi

if finished_on_bo output/cloud-wide enzyme_extremes_wide; then
  pull_full enzyme_extremes_wide output/cloud-wide
  echo "chain: ALL CAMPAIGNS COMPLETE"
fi
