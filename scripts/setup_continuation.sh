#!/usr/bin/env bash
# Prepare the 1.5x-budget CONTINUATION of a finished campaign, on a COPY of it. Run ON bo.
#
#   scripts/setup_continuation.sh [--go]
#
# Without --go it reports what it would do and changes nothing. It NEVER touches the source tree.
#
# THE SOURCE CAMPAIGN IS NOT MODIFIED, and that is the whole design: the continuation runs against
# `output/cloud-cont/enzyme_extremes_cont`, a copy, so the finished campaign at
# `output/cloud/enzyme_extremes` stays exactly as it was reported. If the continuation goes wrong,
# nothing that has already been measured is at risk.
#
# STEPS, in order, each of which fails loudly rather than half-succeeding:
#
#   1. REFUSE unless the source campaign is finished (`median.json` present). Continuing a campaign
#      that is still adding designs would race its own driver.
#   2. REFUSE unless the destination is absent or explicitly cleared, so a re-run cannot silently
#      merge into a half-built copy.
#   3. Copy the tree. `results.json`, the state pair (`optimizer.npz` / `trainer.npz`) and the
#      per-design `checkpoints/` are ALL required: the pools carry the events already paid for, and
#      `closest`/`meta` read historical networks from the checkpoints by design number.
#   4. REMOVE `done.txt` -- this is the invalidation that reopens each BO run at the design it
#      stopped on, so the raised budget buys more designs instead of being ignored. NOTE the copy
#      came from a tree built under the OLD scheme and so has no markers at all; there is deliberately
#      no call to scripts/migrate_markers.sh here, because that script exists to STOP finished runs
#      being re-run and a continuation wants exactly the opposite. Migrating and then deleting would
#      be the same end state reached twice.
#   5. REMOVE `verified.txt` and the verification artefacts -- the old verification re-scored a
#      SHORTER trajectory, so it does not describe the continued run and must not be carried over as
#      though it did. `verification.json` goes too: it is resumable state keyed to points that are
#      about to change.
#
# What it deliberately does NOT do is launch anything. Check the report, then launch with
#   scripts/launch_bo_campaign.sh 12 36 5 enzyme_extremes_cont output/cloud-cont
set -euo pipefail

GO=${1:-}
# THREE CHECKOUTS, ONE PER CAMPAIGN, and this is not tidiness -- snakemake locks its WORKING
# DIRECTORY, not merely the files it writes. Two campaigns launched from one checkout fail outright
# with "LockException: Directory cannot be locked", verified on this box. `--nolock` would force it,
# but they would then share `.snakemake/` metadata and incomplete-file tracking, which is exactly
# what the lock protects. So the source campaign, the wide campaign and this continuation each get
# their own tree and can run at the same time.
SRC=${SRC:-$HOME/detector-opt/output/cloud/enzyme_extremes}
DST_REPO=${DST_REPO:-$HOME/detector-opt-cont}
DST=${DST:-$DST_REPO/output/cloud-cont/enzyme_extremes_cont}
REPO=$DST_REPO

say() { printf '%s\n' "$*"; }
act() { if [ "$GO" = "--go" ]; then eval "$@"; else say "  would: $*"; fi; }

[ -f "$SRC/median.json" ] || { say "REFUSING: $SRC/median.json absent -- the source campaign is not finished"; exit 1; }
say "source:      $SRC  (finished)"
say "destination: $DST"

if [ -d "$DST" ] && [ "$GO" = "--go" ]; then
  say "REFUSING: $DST already exists. Remove it by hand if you mean to rebuild the copy."
  exit 1
fi

say
say "1. copy the tree (results.json + optimizer.npz + trainer.npz + checkpoints/)"
act "mkdir -p '$(dirname "$DST")'"
act "cp -a '$SRC' '$DST'"

say
say "2. invalidate the BO runs (reopens each at the design it stopped on)"
act "find '$DST' -name done.txt -delete"

say
say "3. invalidate verification (the old one re-scored a shorter trajectory)"
act "find '$DST' \\( -name verified.txt -o -name 'verification*.png' -o -name verification.json \\) -delete"

say
say "4. clear the cross-run roll-ups, which describe the parent campaign"
act "find '$DST' -maxdepth 2 \\( -name 'comparison.txt' -o -name 'convergence_all.*' -o -name 'median.*' \\) -delete"

say
if [ "$GO" != "--go" ]; then
  say "DRY RUN -- nothing changed. Re-run with --go to apply."
else
  say "READY. Launch with:"
  say "  bash $DST_REPO/scripts/launch_bo_campaign.sh <K> <3*K> 5 enzyme_extremes_cont output/cloud-cont"
  say
  say "PICK K AGAINST WHAT ELSE IS RUNNING. Aggregate throughput is close to a box-level constant"
  say "past ~8 workers (5.66x at 8, 6.46x at 12), so splitting workers between campaigns buys"
  say "parallel progress rather than extra throughput -- while the HOST does degrade: a job holds"
  say "~0.94 of a core, so 24 jobs saturate all 24 before the GP/EI phase's BLAS threads count at"
  say "all. With the wide campaign at 12, K=6 here keeps the total at 18."
  du -sh "$DST"
fi
