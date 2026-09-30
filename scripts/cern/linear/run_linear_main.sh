#!/bin/bash
# Launch (or resume) the linear-ladder campaign (linear.snake) on CERN HTCondor from the MAIN tree, detached, with a
# renewing ticket.
#
#   setsid nohup scripts/cern/linear/run_linear_main.sh > logs/linear-cern.log 2>&1 < /dev/null &
#   scripts/cern/linear/run_linear_main.sh -n          # dry run: read the reason lines before any launch
#
# The workstation ran the selection stage and the first test cells; on 2026-09-08 the user moved the remaining test
# cells here ("stop local queue and push remaining jobs to CERN"). The state travelled with them: the selection
# markers (so the DAG starts at `selected.txt`), every test cell's results.json, checkpoints and optimiser state (so
# bo.py resumes each interrupted cell at its design boundary), and the finished cells' markers. run_linear.sh is the
# older twin that used its own `detector-linear` tree; this one runs in `detector-opt` beside the other
# orchestrators, hence `--nolock` (disjoint outputs) and its own flock. `--rerun-triggers mtime`: a snakefile edit
# must never rebuild finished cells (memory: snakemake_code_change_rerun_trigger). The reap is SCOPED to
# JobBatchName `linear-`; `NOREAP=1` skips it for a relaunch that must leave running jobs alone (rebalancing) -- then
# the explicit targets MUST exclude the running jobs' cells, or the new DAG resubmits them.
set -u
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90

ORPHANS=$(condor_q -constraint 'regexp("^linear-", JobBatchName)' -af ClusterId 2>/dev/null | sort -u)
if [ "${NOREAP:-0}" = "1" ]; then
  echo "[reap] skipped (NOREAP=1): $(echo "$ORPHANS" | grep -c .) linear job(s) left running; the targets must exclude their cells"
  ORPHANS=""
fi
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

SUBMIT=scripts/cern/linear/snakemake_submit_main.sh
flock -n /tmp/linear-cern.lock snakemake -s linear.snake -j60 --nolock --rerun-triggers=mtime "$@" \
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
