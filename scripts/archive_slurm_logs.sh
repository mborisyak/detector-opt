#!/bin/bash
# Copy every SLURM job log under .snakemake/slurm_logs into output/slurm-log-archive before snakemake
# deletes it.
#
# WHY THIS EXISTS. The snakemake SLURM executor removes a job's log as soon as it reports that job
# successful (`keep_successful_logs` defaults to false), and those logs are the ONLY place the
# trainer's per-design convergence margins are written: the `[converged/bayes] ... diff=... err=...`
# lines. `results.json` stores the losses and `convergence.json` stores the loss curves, but neither
# stores `diff` or `err`, and their maximum over designs IS the pre-registration section 5
# measurement of `loss_precision`. The first `enzyme_extremes` run finished at 09:03 on 2026-08-15 and
# its log was gone before it could be read, which is what prompted this.
#
# WHY A PERIODIC COPY RATHER THAN `tail -F`. A tail per log survives deletion by holding the
# descriptor, but it also never exits, so a long campaign accumulates one idle process per job. A
# single copier loop has no such tail. Losing the last few seconds of a log is not a risk here: a run
# writes its final line, exits, and is only deleted at snakemake's NEXT status poll, which is up to
# 180 s later.
#
# The preferable fix for a campaign not yet launched is `--slurm-keep-successful-logs` on the
# snakemake command line; this script is for one already running.
#
# Usage, from the repository root, under flock so a second copy cannot start:
#   flock -n /home/max/.detector-opt-logarchive.lock ./scripts/archive_slurm_logs.sh &
set -u

interval=${INTERVAL:-15}
root=${ROOT:-.snakemake/slurm_logs}
archive=${ARCHIVE:-output/slurm-log-archive}

mkdir -p "$archive"

while true; do
  while IFS= read -r log; do
    [ -f "$log" ] || continue
    destination="$archive/$(printf '%s' "${log#"$root"/}" | tr '/' '_')"
    cp -f "$log" "$destination" 2>/dev/null || true
  done < <(find "$root" -name '*.log' -type f 2>/dev/null)
  sleep "$interval"
done
