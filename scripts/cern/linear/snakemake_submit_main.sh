#!/bin/bash
# snakemake `cluster-generic` submit command for the linear ladder run from the MAIN tree on CERN HTCondor: batch
# name `linear-<rule>` (what run_linear_main.sh scopes its orphan reap by), SnakeCell from the cell path, GPU memory
# floor 4000 MB (a linear cell was measured at 310-500 MiB of VRAM; the floor only keeps it off the smallest cards).
#
# cluster-generic appends the jobscript path as the last argument and reads the job id from stdout, so this prints
# the ClusterId and NOTHING else. `runtime` is snakemake's MINUTES; +MaxRuntime is condor's SECONDS.
# ONE CPU per job (user, 2026-09-09: \"1 CPU for us is enough\"): the snakefile's `threads` is not forwarded; CERN's
# schedd added one more on top of it, and a 3-CPU request could not use GPU slots whose cores other jobs hold.
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
  echo "request_cpus   = 1"
  echo "request_memory = $MEM"
  echo "+MaxRuntime    = $((RUNTIME * 60))"
  echo "+JobBatchName  = \"linear-$RULE\""
  CELL=$(grep -oE "output/linear/[A-Za-z0-9_./-]+/(select|test)/[A-Za-z0-9_/-]+" "$JOBSCRIPT" 2>/dev/null | head -1)
  echo "+SnakeCell     = \"${CELL:-unknown}\""
  if [ "$GPU" -gt 0 ]; then
    echo "request_gpus   = 1"
    echo "requirements   = TARGET.GPUs_GlobalMemoryMb >= 4000"
  fi
  echo "queue"
} > "$SUB"
OUT=$(condor_submit -terse "$SUB")
rm -f "$SUB"
echo "$OUT" | head -1 | cut -d' ' -f1
