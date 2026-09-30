#!/bin/bash
# Launch the batch-composition ablation (ablation_uniform.snake) on CERN HTCondor, detached, with a renewing ticket.
#
#   setsid nohup scripts/cern/ablation/run_uniform.sh > logs/ablation-uniform.log 2>&1 < /dev/null &
#
# Same shape as run_ablation.sh: the orphan reap is SCOPED to JobBatchName `uniform-`, the lock is its own, and
# snakemake runs with `--nolock` because other orchestrators hold this working directory's lock (disjoint
# outputs). The process lives in the login session it was started from.
set -u
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90

ORPHANS=$(condor_q -constraint 'regexp("^uniform-", JobBatchName)' -af ClusterId 2>/dev/null | sort -u)
if [ -n "${ORPHANS:-}" ]; then
  echo "[reap] removing $(echo "$ORPHANS" | wc -l) untracked uniform-ablation job(s) from a previous orchestrator"
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

SUBMIT=scripts/cern/ablation/snakemake_submit_uniform.sh
flock -n /tmp/uniform-cern.lock snakemake -s ablation_uniform.snake -j40 --nolock "$@" \
  --executor cluster-generic \
  --cluster-generic-submit-cmd \
    "$SUBMIT {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}" \
  --cluster-generic-status-cmd scripts/cern/snakemake_status.sh \
  --cluster-generic-cancel-cmd scripts/cern/snakemake_cancel.sh \
  --config preamble="source scripts/cern/lcg_env.sh && " \
  --latency-wait 60 --keep-going --rerun-incomplete
STATUS=$?
kill $RENEW 2>/dev/null
exit $STATUS
