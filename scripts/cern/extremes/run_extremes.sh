#!/bin/bash
# Launch the extremes campaign (extremes.snake) on CERN HTCondor, from the `detector-opt` tree.
#
#   setsid nohup scripts/cern/extremes/run_extremes.sh > logs/extremes.log 2>&1 < /dev/null &
#
# The twin of scripts/cern/run_campaign.sh (SHiP) and scripts/cern/linear/run_linear.sh. It runs out of `detector-opt`
# because the SHiP campaign there is COMPLETE (2026-09-06 10:02) and snakemake's working-directory lock is free; a
# SHiP relaunch in the same tree while this runs would need its own tree (see run_linear.sh for why).
#
# ⛔️ THE REAP IS SCOPED BY JobBatchName. A bare `condor_rm maborisy` would kill every campaign of this user, so this
# removes only jobs whose batch name begins `extremes-` -- the name scripts/cern/extremes/snakemake_submit.sh gives
# them. Condor jobs outlive the orchestrator (an lxplus reboot, an expired token), and a fresh orchestrator has no
# record of them: any extremes job still queued at startup is untracked and is removed; finished cells left a
# `done.txt` and are skipped by snakemake.
#
# ⛔️ THE TICKET MUST BE RENEWED FOR THE WHOLE RUN (AFS write access for snakemake's state and the results): `kinit -R`
# renews from the existing TGT with no credentials, `aklog` turns it into a fresh AFS token. The 2026-09-04 SHiP
# orchestrator died 24-25 h after launch, at the token lifetime, most likely because its credential cache went away
# with the login session that launched it: launch from a session that stays open, or accept that a campaign longer
# than a token needs a relaunch (`condor_q` first: the cells keep running, only new submissions stop).
#
# ⚠️ THE `flock` IS NODE-LOCAL (/tmp is per lxplus node); snakemake's own lock under `.snakemake/` on AFS is what
# stops a second orchestrator on another node.
set -u
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90

ORPHANS=$(condor_q -constraint 'regexp("^extremes-", JobBatchName)' -af ClusterId 2>/dev/null | sort -u)
if [ -n "${ORPHANS:-}" ]; then
  echo "[reap] removing $(echo "$ORPHANS" | wc -l) untracked extremes job(s) from a previous orchestrator"
  for c in $ORPHANS; do condor_rm "$c" >/dev/null 2>&1; done
  sleep 10
fi

renew_loop() {
  while sleep 1800; do
    kinit -R >/dev/null 2>&1 && aklog >/dev/null 2>&1 \
      && echo "[renew] $(date '+%H:%M') ticket renewed" \
      || echo "[renew] $(date '+%H:%M') RENEWAL FAILED -- token will expire"
  done
}
renew_loop &
RENEW=$!
trap 'kill $RENEW 2>/dev/null' EXIT INT TERM

# Targets are passed through: `scripts/cern/extremes/run_extremes.sh output/extremes/selected.txt` stops after the
# selection. Rerun-incomplete and keep-going match the other orchestrators.
flock -n /tmp/extremes-cern.lock snakemake -s extremes.snake -j40 "$@" \
  --executor cluster-generic \
  --cluster-generic-submit-cmd "scripts/cern/extremes/snakemake_submit.sh {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}" \
  --cluster-generic-status-cmd scripts/cern/snakemake_status.sh \
  --cluster-generic-cancel-cmd scripts/cern/snakemake_cancel.sh \
  --config preamble="source scripts/cern/lcg_env.sh && " \
  --latency-wait 60 \
  --keep-going \
  --rerun-incomplete
STATUS=$?
kill $RENEW 2>/dev/null
exit $STATUS
