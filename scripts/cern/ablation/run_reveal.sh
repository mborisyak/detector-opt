#!/bin/bash
# Launch the reveal ablation (ablation_reveal.snake) on CERN HTCondor, detached, with a renewing ticket.
#
#   setsid nohup scripts/cern/ablation/run_reveal.sh > logs/ablation-reveal.log 2>&1 < /dev/null &
#   STAGE=test setsid nohup scripts/cern/ablation/run_reveal.sh > logs/ablation-reveal-test.log 2>&1 < /dev/null &
#
# STAGE (select, the default, or test) picks the seed list and output prefix in ablation_reveal.snake; the two
# stages share the batch-name scope and the lock, so run them one after the other, never side by side.
#
# Same shape as run_ablation.sh: the orphan reap is SCOPED to JobBatchName `reveal-`, the lock is its own, and
# snakemake runs with `--nolock` because other orchestrators hold this working directory's lock (disjoint
# outputs). The process lives in the login session it was started from.
set -u
STAGE=${STAGE:-select}
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90

ORPHANS=$(condor_q -constraint 'regexp("^reveal-", JobBatchName)' -af ClusterId 2>/dev/null | sort -u)
if [ -n "${ORPHANS:-}" ]; then
  echo "[reap] removing $(echo "$ORPHANS" | wc -l) untracked reveal-ablation job(s) from a previous orchestrator"
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

SUBMIT=scripts/cern/ablation/snakemake_submit_reveal.sh
flock -n /tmp/reveal-cern.lock snakemake -s ablation_reveal.snake -j40 --nolock "$@" \
  --executor cluster-generic \
  --cluster-generic-submit-cmd \
    "$SUBMIT {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}" \
  --cluster-generic-status-cmd scripts/cern/snakemake_status.sh \
  --cluster-generic-cancel-cmd scripts/cern/snakemake_cancel.sh \
  --config preamble="source scripts/cern/lcg_env.sh && " stage="$STAGE" \
  --latency-wait 60 --keep-going --rerun-incomplete
STATUS=$?
kill $RENEW 2>/dev/null
exit $STATUS
