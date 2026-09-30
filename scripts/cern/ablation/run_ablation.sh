#!/bin/bash
# Launch the replay-weight ablation (ablation_lambda.snake) on CERN HTCondor, detached, with a renewing ticket.
#
#   setsid nohup scripts/cern/ablation/run_ablation.sh > logs/ablation.log 2>&1 < /dev/null &
#
# The twin of scripts/cern/extremes/run_extremes.sh: the orphan reap is SCOPED to JobBatchName `ablation-`
# (a bare `condor_rm maborisy` would kill whatever other campaign runs out of this tree), the lock is its
# own, and snakemake runs with `--nolock` because another orchestrator may hold this working directory's
# lock -- the DAGs share no output path. The process lives in the login session it was started from.
set -u
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90

ORPHANS=$(condor_q -constraint 'regexp("^ablation-", JobBatchName)' -af ClusterId 2>/dev/null | sort -u)
if [ -n "${ORPHANS:-}" ]; then
  echo "[reap] removing $(echo "$ORPHANS" | wc -l) untracked ablation job(s) from a previous orchestrator"
  for c in $ORPHANS; do condor_rm "$c" >/dev/null 2>&1; done
  sleep 10
fi

renew_loop() {
  while sleep 1800; do
    kinit -R >/dev/null 2>&1 && aklog >/dev/null 2>&1 \
      && echo "[renew] $(date '+%H:%M') ticket renewed" \
      || echo "[renew] $(date '+%H:%M') RENEWAL FAILED"
  done
}
renew_loop &
RENEW=$!
trap 'kill $RENEW 2>/dev/null' EXIT INT TERM

flock -n /tmp/ablation-cern.lock snakemake -s ablation_lambda.snake -j40 --nolock "$@" \
  --executor cluster-generic \
  --cluster-generic-submit-cmd \
    "scripts/cern/ablation/snakemake_submit.sh {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}" \
  --cluster-generic-status-cmd scripts/cern/snakemake_status.sh \
  --cluster-generic-cancel-cmd scripts/cern/snakemake_cancel.sh \
  --config preamble="source scripts/cern/lcg_env.sh && " \
  --latency-wait 60 --keep-going --rerun-incomplete
STATUS=$?
kill $RENEW 2>/dev/null
exit $STATUS
