#!/bin/bash
# snakemake `cluster-generic` submit command for the EXTREMES campaign on CERN HTCondor -- the twin of
# scripts/cern/snakemake_submit.sh (SHiP) with two differences: the batch name `extremes-<rule>`, which is what
# scripts/cern/extremes/run_extremes.sh scopes its orphan reap by, and a GPU memory floor of 8000 MB instead of
# 10000: an extremes cell holds 1310720 events of 4 x 32 floats in its pools (~0.7 GB) plus a small set regressor,
# nowhere near a SHiP search cell's 6.1 GiB, so the smaller slices are admitted.
#
# cluster-generic appends the jobscript path as the last argument and reads the job id from stdout, so this prints
# the ClusterId and NOTHING else. `runtime` is snakemake's MINUTES; +MaxRuntime is condor's SECONDS.
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
  echo "+MaxRuntime    = $((RUNTIME * 60))"
  echo "+JobBatchName  = \"extremes-$RULE\""
  CELL=$(grep -oE "output/[A-Za-z0-9_./-]+/(select|test)/[A-Za-z0-9_/-]+" "$JOBSCRIPT" 2>/dev/null | head -1)
  echo "+SnakeCell     = \"${CELL:-unknown}\""
  if [ "$GPU" -gt 0 ]; then
    echo "request_gpus   = 1"
    echo "requirements   = TARGET.GPUs_GlobalMemoryMb >= 8000"
  fi
  echo "queue"
} > "$SUB"
OUT=$(condor_submit -terse "$SUB")
rm -f "$SUB"
echo "$OUT" | head -1 | cut -d' ' -f1
