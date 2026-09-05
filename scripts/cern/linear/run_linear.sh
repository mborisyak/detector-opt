#!/bin/bash
# Launch the linear-ladder campaign on CERN HTCondor, from its OWN working directory.
#
# WHY A SEPARATE TREE: snakemake locks a working directory and the SHiP orchestrator holds that lock
# on `detector-opt`. A second campaign therefore needs its own. Only code and config are copied --
# the linear detector is fully synthetic (`n_dimensions`, `n_probes`, `noise`, `probe_bounds`, no
# `data_dir`), so nothing else has to travel with it.
#
# ⛔️ THE REAP IS SCOPED BY JobBatchName. A bare `condor_rm maborisy` would kill the SHiP campaign
# running out of the other tree, so this removes only jobs whose batch name begins `linear-`.
set -u
cd /afs/cern.ch/work/m/maborisy/detector-linear || exit 90

ORPHANS=$(condor_q -constraint 'regexp("^linear-", JobBatchName)' -af ClusterId 2>/dev/null | sort -u)
if [ -n "${ORPHANS:-}" ]; then
  echo "[reap] removing $(echo "$ORPHANS" | wc -l) untracked linear job(s) from a previous orchestrator"
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

flock -n /tmp/linear-cern.lock snakemake -s linear.snake -j40 \
  --executor cluster-generic \
  --cluster-generic-submit-cmd "scripts/cern/snakemake_submit.sh {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}" \
  --cluster-generic-status-cmd scripts/cern/snakemake_status.sh \
  --cluster-generic-cancel-cmd scripts/cern/snakemake_cancel.sh \
  --config preamble="source scripts/cern/lcg_env.sh && " \
  --latency-wait 60 --keep-going --rerun-incomplete
STATUS=$?
kill $RENEW 2>/dev/null
exit $STATUS
