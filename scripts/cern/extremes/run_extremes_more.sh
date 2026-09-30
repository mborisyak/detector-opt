#!/bin/bash
# Build the SECOND batch of extremes test seeds (extremes.snake SEEDS_TEST[10:], draws 11-20) on CERN HTCondor while
# the first orchestrator is still finishing its own DAG.
#
#   setsid nohup scripts/cern/extremes/run_extremes_more.sh > logs/extremes-more.log 2>&1 < /dev/null &
#
# Differs from run_extremes.sh in three ways, all forced by running BESIDE it: NO orphan reap (the first orchestrator's
# jobs share the `extremes-` batch names and are not orphans), its OWN lock, and EXPLICIT targets -- the 30 new cells'
# `verified.txt` -- so the DAG never contains the first batch's unfinished verifications. `campaign.txt` is left to the
# first orchestrator (10 seeds); once both are done, `snakemake -s extremes.snake --nolock output/extremes/campaign.txt`
# re-touches it over all 20. ⚠️ A later run_extremes.sh launch WOULD reap these jobs (same batch names): never start
# one while this runs.
set -u
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 90

TARGETS=""
for seed in 2093442589 799314451 104246563 553867472 1620698205 1588951686 183442464 774926522 261261008 1890636464; do
  for strategy in from_scratch continue meta; do
    TARGETS="$TARGETS output/extremes/test/$seed/$strategy/verified.txt"
  done
done

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

SUBMIT=scripts/cern/extremes/snakemake_submit.sh
flock -n /tmp/extremes-more-cern.lock snakemake -s extremes.snake -j40 --nolock "$@" $TARGETS \
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
