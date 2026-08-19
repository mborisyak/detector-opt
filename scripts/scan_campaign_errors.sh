#!/usr/bin/env bash
# Scan a campaign log for GENUINE failures. Run from the workstation or on bo.
#
#   scripts/scan_campaign_errors.sh <logfile>
#
# ⚠️ `pool exhausted` IS NOT AN ERROR. `bo.py` prints "[budget] pool exhausted; finishing BO after N
# completed iterations." when the budget pool FILLS, which is the normal and intended end of a run --
# there is one per completed run. A naive grep for "pool exhausted" therefore reports a healthy
# 20-run campaign as 20 failures, which it did here. The real pool failure is `Pool overflow`, raised
# from `Pool.append` when a chunk would exceed capacity.
#
# Likewise `nan` matches inside ordinary words, so it is anchored to a word boundary and to the
# spellings jax/numpy actually emit.
#
# COUNTS USE `| wc -l`, NEVER `grep -c ... || echo 0`. `grep -c` PRINTS its zero and ALSO exits 1, so
# the fallback appends a second zero and the variable becomes the two-line string "0\n0", which then
# fails every numeric test with "integer expression expected". Same trap as in the heartbeat probe.
set -uo pipefail
LOG=${1:?usage: scan_campaign_errors.sh <logfile>}
PATTERN='Traceback|Pool overflow|did not reach precision within iteration_limit|CUDA_ERROR|RESOURCE_EXHAUSTED|out of memory|Killed|\bnan\b|\bNaN\b'
n=$(grep -iE "$PATTERN" "$LOG" 2>/dev/null | wc -l)
echo "$LOG: $n genuine error line(s)"
[ "$n" -gt 0 ] && grep -inE "$PATTERN" "$LOG" | head -10
completed=$(grep "pool exhausted; finishing BO" "$LOG" 2>/dev/null | wc -l)
echo "  (runs that finished normally: $completed)"
exit 0
