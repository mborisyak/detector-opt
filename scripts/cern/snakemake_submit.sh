#!/bin/bash
# snakemake `cluster-generic` submit command for CERN HTCondor.
#
#   snakemake -s ship.snake -j40 --executor cluster-generic \
#     --cluster-generic-submit-cmd "scripts/cern/snakemake_submit.sh {rule} {threads} {resources.mem_mb} {resources.runtime} {resources.local_gpu}" \
#     --cluster-generic-status-cmd scripts/cern/snakemake_status.sh
#
# cluster-generic appends the jobscript path as the last argument and reads the job id from stdout,
# so this prints the ClusterId and NOTHING else.
set -eu
RULE=$1; THREADS=$2; MEM=$3; RUNTIME=$4; GPU=$5; JOBSCRIPT=$6
D=/afs/cern.ch/work/m/maborisy/detector-opt
mkdir -p "$D/logs/condor"
SUB=$(mktemp /tmp/snakemake-condor.XXXXXX.sub)
{
  echo "universe       = vanilla"
  echo "executable     = $D/scripts/cern/snakemake_exec.sh"
  echo "arguments      = $JOBSCRIPT"
  echo "output         = $D/logs/condor/\$(ClusterId).out"
  echo "error          = $D/logs/condor/\$(ClusterId).err"
  echo "log            = $D/logs/condor/\$(ClusterId).log"
  echo "request_cpus   = $THREADS"
  echo "request_memory = $MEM"
  # `runtime` is snakemake's MINUTES; +MaxRuntime is condor's SECONDS. Passing one as the other is
  # how a job silently gets 40 minutes instead of 48 hours.
  echo "+MaxRuntime    = $((RUNTIME * 60))"
  echo "+JobBatchName  = \"ship-$RULE\""
  # ⛔️ TAG THE JOB WITH THE CELL IT RUNS. Without this a condor job cannot be mapped back to an output
  # directory -- cluster-generic passes only an opaque jobscript path -- so a monitor cannot tell a
  # cell that is RUNNING SLOWLY from one that is merely QUEUED with a stale `results.json`. Guessing
  # from file mtimes produced three false stall alarms on 2026-09-02. The jobscript names its target,
  # so the cell is recoverable from it and is recorded as a classad the queue can be filtered on.
  CELL=$(grep -oE "output/[A-Za-z0-9_./-]+/(select|test)/[A-Za-z0-9_/-]+" "$JOBSCRIPT" 2>/dev/null | head -1)
  echo "+SnakeCell     = \"${CELL:-unknown}\""
  # GPU MEMORY FLOOR 10 GB: a search cell at the full 2^21 budget holds 6.1 GiB of pools and ran on the
  # 12 GB (10564 MB usable) H100 MIG slices without OOM on 2026-09-04, so those slices are admitted.
  if [ "$GPU" -gt 0 ]; then
    echo "request_gpus   = 1"
    echo "requirements   = TARGET.GPUs_GlobalMemoryMb >= 10000"
  fi
  echo "queue"
} > "$SUB"
OUT=$(condor_submit -terse "$SUB")
rm -f "$SUB"
# `-terse` prints "<cluster>.<proc> - <cluster>.<proc>"; the status command wants the full id.
echo "$OUT" | head -1 | cut -d' ' -f1
