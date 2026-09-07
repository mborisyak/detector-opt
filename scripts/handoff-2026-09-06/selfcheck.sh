#!/bin/bash
# Half-hourly self-check of both campaigns. Read-only.
cd /home/max/dev/detector-opt || exit 1
echo "=== LOCAL linear ==="
echo "orchestrator pids: $(pgrep -f 'snakemake -s linear.snake' | tr '\n' ' ')| flock: $(pgrep -f 'flock -n /tmp/linear-snakemake.lock' | tr '\n' ' ')"
echo "squeue: $(squeue -h -o '%i %t %M' | tr '\n' ';')"
echo "done=$(ls output/linear/select/*/*/*/*/done.txt 2>/dev/null | wc -l)/180 verified=$(ls output/linear/select/*/*/*/*/verified.txt 2>/dev/null | wc -l)/180 log_errors=$(grep -cE '^Error|Traceback|RESOURCE_EXHAUSTED|Error in rule' logs/linear-local.log 2>/dev/null)"
echo "gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
for d in output/linear/select/*/*/*/*/; do [ -e "$d/results.json" ] && echo "  $(echo $d | cut -d/ -f4-7): designs=$(grep -o '"x_scaled"' "$d/results.json" | wc -l) mtime=$(date -r "$d/results.json" +%H:%M) done=$([ -e "$d/done.txt" ] && echo y || echo n)"; done | tail -8
echo "=== CERN ship ==="
ssh -o BatchMode=yes -o ConnectTimeout=30 lxplus 'cd /afs/cern.ch/work/m/maborisy/detector-opt && echo "condor: $(condor_q maborisy -af JobStatus | sort | uniq -c | tr "\n" ";")  held: $(condor_q maborisy -constraint "JobStatus==5" -af ClusterId | wc -l)" && echo "done=$(ls output/*/select/*/*/*/done.txt 2>/dev/null | wc -l)/90 verified=$(ls output/*/select/*/*/*/verified.txt 2>/dev/null | wc -l)/90 log_errors=$(grep -cE "^Error|Traceback|RESOURCE_EXHAUSTED" logs/campaign.log)" && echo "orchestrator: $(pgrep -f run_campaign.sh | wc -l) run_campaign, $(pgrep -f "snakemake -s ship.snake" | wc -l) snakemake on $(hostname)" && now=$(date +%s) && condor_q maborisy -constraint "JobStatus==2" -af JobCurrentStartDate SnakeCell | awk -v now=$now "{printf \"  running %3d min  %s\n\", (now-\$1)/60, \$2}" | sort -rn | head -5 && for f in output/*/select/*/*/*/results.json; do echo "  $(echo $f | cut -d/ -f2-6): designs=$(grep -o "\"x_scaled\"" $f | wc -l) mtime=$(date -r $f +%H:%M)"; done 2>/dev/null | sort -t= -k2 -n | head -4' 2>&1 | tail -16
