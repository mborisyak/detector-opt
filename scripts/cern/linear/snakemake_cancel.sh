#!/bin/bash
# snakemake `cluster-generic` cancel command. WITHOUT THIS, a snakemake that is interrupted or dies
# leaves every job it submitted running in the condor queue, orphaned -- nothing else will ever
# reap them, and the next run submits duplicates alongside.
#
# Job ids arrive as arguments, one per job. `condor_rm` on an id that has already left the queue is
# harmless, so failures are not fatal here.
for id in "$@"; do
  condor_rm "$id" >/dev/null 2>&1 || true
done
