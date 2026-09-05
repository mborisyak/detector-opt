#!/bin/bash
# Launch the full docs/final.md campaign on CERN HTCondor, with a renewing Kerberos/AFS ticket.
#
#   setsid nohup scripts/cern/run_campaign.sh > logs/campaign.log 2>&1 < /dev/null &
#
# ⛔️ THE TICKET MUST BE RENEWED FOR THE WHOLE RUN. The orchestrator needs AFS write access for its
# entire life (snakemake state, logs, reading results), and an AFS token lasts ~25 h. `kinit -R`
# renews from the existing TGT with NO credentials -- valid while inside the renewable window, which
# is days -- so this needs no keytab and never prompts. `aklog` converts the renewed TGT into a fresh
# AFS token. Without this the orchestrator dies mid-campaign with AFS write errors that look like
# something else entirely.
#
# ⛔️ `flock -n` SO A SECOND COPY CANNOT START. Two orchestrators would submit every cell twice.
set -u
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90

# ⛔️ REAP ORPHANS BEFORE STARTING. Condor jobs OUTLIVE this orchestrator: if the lxplus node it runs
# on reboots -- and these nodes cycle often, hours not days -- every job it submitted keeps running
# and keeps writing to its output directory. A fresh orchestrator has no record of them, sees no
# `done.txt`, and submits a SECOND copy of each: two processes writing the same files.
#
# So any of our jobs still queued at startup are by definition untracked, and are removed. A job that
# had already finished left a `done.txt` and is skipped by snakemake anyway, so nothing complete is
# lost -- only in-flight work is repeated, which is the cheaper mistake.
#
# ⚠️ THE `flock` BELOW IS NODE-LOCAL (/tmp is not shared across the lxplus pool), so it does NOT stop
# a second orchestrator on another node. Snakemake's own lock under `.snakemake/` is on AFS and is
# what actually provides that guarantee.
ORPHANS=$(condor_q -af ClusterId 2>/dev/null | sort -u)
if [ -n "${ORPHANS:-}" ]; then
  echo "[reap] removing $(echo "$ORPHANS" | wc -l) untracked job(s) from a previous orchestrator"
  condor_rm maborisy >/dev/null 2>&1 || true
  sleep 10
fi

renew_loop() {
  while sleep 1800; do
    kinit -R >/dev/null 2>&1 && aklog >/dev/null 2>&1 \
      && echo "[renew] $(date '+%H:%M') ticket renewed" \
      || echo "[renew] $(date '+%H:%M') RENEWAL FAILED -- token will expire"
  done
}
# ⚠️ NOT `exec`. `exec` replaces this shell, which DISCARDS the trap and leaves the renewal loop
# running forever after snakemake exits -- one such orphan was left behind on 2026-08-31. snakemake
# is run as a child so the trap fires and the loop is cleaned up on every exit path.
renew_loop &
RENEW=$!
trap 'kill $RENEW 2>/dev/null' EXIT INT TERM

# TARGETS ARE PASSED THROUGH, so this launcher can drive one task instead of the whole DAG:
#
#   scripts/cern/run_campaign.sh                          # both tasks (the default `all`)
#   scripts/cern/run_campaign.sh output/angle/campaign.txt # angle only
#
# Splitting by task is how the two sites share the work: CERN takes `angle`, `bo` takes `intersect`.
# The halves are DISJOINT, so neither needs the other's filesystem and each orchestrator manages its
# own queue -- see docs/provisioning.md 3b for why a single shared cluster is not an option here.
flock -n /tmp/ship-snakemake.lock snakemake -s ship.snake -j40 "$@" \
  --executor cluster-generic \
  --cluster-generic-submit-cmd "scripts/cern/snakemake_submit.sh {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}" \
  --cluster-generic-status-cmd scripts/cern/snakemake_status.sh \
  --cluster-generic-cancel-cmd scripts/cern/snakemake_cancel.sh \
  --config preamble="source scripts/cern/lcg_env.sh && " \
  --latency-wait 60 \
  --keep-going \
  --rerun-incomplete
STATUS=$?
kill $RENEW 2>/dev/null
exit $STATUS
